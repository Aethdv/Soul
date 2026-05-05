//! Gradient computation against dataset entries for evaluation tuning.

use super::SoulEntry;
use crate::{
    core::{
        board::{Position, bitboard::atk_king, spatial::SpatialTensor},
        defs::{Color, PieceType, Square},
        phase::compute_phase_weights_f64,
        psqt,
    },
    engine::mobility::{Mobility, OPEN_UNITY, SafetyMetrics, compute_openness_raw},
};

/// Pre-computed instance-level features, populated at tuner startup.
///
/// Features are packed into raw byte arrays for SoA (Structure of Arrays) layout:
/// the i-th entry in each vec corresponds to the i-th SoulEntry in the training set.
/// This avoids pointer-chasing through AoS structs during the hot gradient loop.
pub struct FeatureSlots {
    pub mobility: Vec<[i8; 8]>,
    pub safety_us: Vec<[u8; 4]>,
    pub safety_them: Vec<[u8; 4]>,
    pub xray_ortho: Vec<i8>,
}

impl FeatureSlots {
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            mobility: Vec::with_capacity(cap),
            safety_us: Vec::with_capacity(cap),
            safety_them: Vec::with_capacity(cap),
            xray_ortho: Vec::with_capacity(cap),
        }
    }

    /// Reconstruct a Position from the entry, and append the packed results to the slot arrays.
    ///
    /// The FEN round-trip (`to_fen` → `Position::from_fen`) is one-time startup
    /// cost per entry — it never executes inside the training loop.
    /// A direct nibble→Position decoder would eliminate the intermediate string allocation,
    /// but the current cost is negligible next to the feature computation itself
    pub fn push_entry(&mut self, entry: &SoulEntry) {
        let pos = Position::from_fen(&entry.to_fen());

        let pinned_w = pos.pinned_pieces(Color::White);
        let pinned_b = pos.pinned_pieces(Color::Black);
        let tensor = SpatialTensor::compute(&pos, pinned_w.0, pinned_b.0);
        let mob = Mobility::compute_all(&pos, pos.stm, &tensor, pinned_w, pinned_b);

        let mut mob_arr = [0i8; 8];
        mob_arr[0] = mob.metrics_us.mobility.clamp(-127, 127) as i8;
        mob_arr[1] = mob.metrics_us.shadow_mobility.clamp(-127, 127) as i8;
        mob_arr[2] = mob.metrics_us.threats.clamp(-127, 127) as i8;
        mob_arr[3] = mob.metrics_us.shadow_threats.clamp(-127, 127) as i8;
        mob_arr[4] = mob.metrics_them.mobility.clamp(-127, 127) as i8;
        mob_arr[5] = mob.metrics_them.shadow_mobility.clamp(-127, 127) as i8;
        mob_arr[6] = mob.metrics_them.threats.clamp(-127, 127) as i8;
        mob_arr[7] = mob.metrics_them.shadow_threats.clamp(-127, 127) as i8;
        self.mobility.push(mob_arr);

        self.safety_us.push(pack_safety(&mob.safety_us));
        self.safety_them.push(pack_safety(&mob.safety_them));

        let w_ksq = pos.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = pos.pieces(PieceType::King, Color::Black).lsb();
        let w_ring = atk_king(w_ksq).0;
        let b_ring = atk_king(b_ksq).0;
        let xray_val = (tensor.w_ortho_xray() & b_ring).count_ones() as i32 - (tensor.b_ortho_xray() & w_ring).count_ones() as i32;
        let stm_xray = if pos.stm == Color::White { xray_val } else { -xray_val };
        self.xray_ortho.push(stm_xray as i8);
    }
}

/// Byte layout: [attackers, weak (i8→u8), shield (i8→u8), ortho<<4|diag (4‑bit each)].
#[inline]
fn pack_safety(m: &SafetyMetrics) -> [u8; 4] {
    [
        m.attackers as u8,
        m.weak.clamp(-128, 127) as i8 as u8,
        m.shield.clamp(-128, 127) as i8 as u8,
        ((m.ortho_exposure.clamp(0, 15) as u8) << 4) | (m.diag_exposure.clamp(0, 15) as u8),
    ]
}

/// Accumulate parameter gradients for `entry` using pre-computed features
/// from `slots[idx]`. `idx` must be the parallel index in both the training
/// entry array and the `FeatureSlots` arrays (populated together at startup).
pub fn accumulate_gradient_cached(
    entry: &SoulEntry,
    slots: &FeatureSlots,
    idx: usize,
    values: &[f64],
    gradient: f64,
    grads: &mut [f64],
) {
    let f = extract_board(entry);
    let (mg_w, eg_w) = compute_phase_weights_f64(&f.phase_counts, values);

    // PSQT gradients
    for i in 0..f.piece_count {
        let (pt, sq_idx, sign) = f.piece_data[i];
        let mg_idx = (pt * 64) + psqt::mirror_sq(sq_idx);
        let eg_idx = (pt * 64) + 32 + psqt::mirror_sq(sq_idx);
        // SAFETY: pt (0..5), mirror_sq (0..31). Max idx = 5*64+32+31 = 383 < 384.
        unsafe {
            *grads.get_unchecked_mut(mg_idx) += gradient * sign * mg_w;
            *grads.get_unchecked_mut(eg_idx) += gradient * sign * eg_w;
        }
    }

    let material_offset = psqt::LAYOUT.material_offset;
    for pt in 0..6 {
        let diff = f.mat_diffs[pt];
        if diff.abs() > 0.001 {
            let mg_idx = material_offset + pt;
            let eg_idx = material_offset + 6 + pt;
            grads[mg_idx] += gradient * diff * mg_w;
            grads[eg_idx] += gradient * diff * eg_w;
        }
    }

    let king_safety_offset = psqt::LAYOUT.king_safety_offset;
    let attacker_offset = psqt::LAYOUT.attacker_offset;

    let us = unpack_safety(slots.safety_us[idx]);
    let them = unpack_safety(slots.safety_them[idx]);

    if us.attackers > 0 {
        let attacker_idx = attacker_offset + us.attackers.min(5);
        grads[attacker_idx] += gradient * (-f64::from(us.weak) / 10.0) * mg_w;
    }
    if them.attackers > 0 {
        let attacker_idx = attacker_offset + them.attackers.min(5);
        grads[attacker_idx] += gradient * (f64::from(them.weak) / 10.0) * mg_w;
    }

    let shield_diff = f64::from(us.shield) - f64::from(them.shield);
    let ortho_diff = f64::from(us.ortho_exposure) - f64::from(them.ortho_exposure);
    let diag_diff = f64::from(us.diag_exposure) - f64::from(them.diag_exposure);

    grads[king_safety_offset] += gradient * shield_diff * mg_w;
    grads[king_safety_offset + 1] -= gradient * ortho_diff * mg_w;
    grads[king_safety_offset + 2] -= gradient * diag_diff * mg_w;

    let xray_offset = psqt::LAYOUT.xray_offset;
    grads[xray_offset] += gradient * f64::from(slots.xray_ortho[idx]) * mg_w;

    let (openness, closedness) = openness_factors(f.white_pawns, f.black_pawns);
    let mobility_open_offset = psqt::LAYOUT.mobility_open_offset;
    let mobility_closed_offset = psqt::LAYOUT.mobility_closed_offset;

    let mobility = slots.mobility[idx];
    for i in 0..4 {
        let diff = f64::from(mobility[i]) - f64::from(mobility[i + 4]);
        let g_diff = gradient * diff;

        grads[mobility_open_offset + i] += g_diff * openness * mg_w;
        grads[mobility_open_offset + 4 + i] += g_diff * openness * eg_w;

        grads[mobility_closed_offset + i] += g_diff * closedness * mg_w;
        grads[mobility_closed_offset + 4 + i] += g_diff * closedness * eg_w;
    }
}

/// Compute the STM-relative eval for `entry` using pre-computed features
/// from `slots[idx]`. See `accumulate_gradient_cached` for the `idx` invariant.
#[inline]
pub fn eval_soul_cached(entry: &SoulEntry, slots: &FeatureSlots, idx: usize, values: &[f64]) -> f64 {
    let f = extract_board(entry);
    let (mg_w, eg_w) = compute_phase_weights_f64(&f.phase_counts, values);
    let mut score = 0.0;

    for i in 0..f.piece_count {
        let (pt, sq_idx, sign) = f.piece_data[i];
        let mg_idx = (pt * 64) + psqt::mirror_sq(sq_idx);
        let eg_idx = (pt * 64) + 32 + psqt::mirror_sq(sq_idx);
        // SAFETY: pt (0..5), mirror_sq (0..31). Max idx = 5*64+32+31 = 383 < 384.
        unsafe {
            score += sign * (*values.get_unchecked(mg_idx) * mg_w + *values.get_unchecked(eg_idx) * eg_w);
        }
    }

    let material_offset = psqt::LAYOUT.material_offset;

    for pt in 0..6 {
        let diff = f.mat_diffs[pt];
        // SAFETY: pt bounded 0..5, indices map strictly into material parameter slots.
        unsafe {
            let mg_idx = material_offset + pt;
            let eg_idx = material_offset + 6 + pt;
            score += diff * (*values.get_unchecked(mg_idx) * mg_w + *values.get_unchecked(eg_idx) * eg_w);
        }
    }

    let king_safety_offset = psqt::LAYOUT.king_safety_offset;
    let attacker_offset = psqt::LAYOUT.attacker_offset;

    let us = unpack_safety(slots.safety_us[idx]);
    let them = unpack_safety(slots.safety_them[idx]);

    let shield_w = values[king_safety_offset];
    let ortho_w = values[king_safety_offset + 1];
    let diag_w = values[king_safety_offset + 2];
    let us_attacker_w = values[attacker_offset + us.attackers.min(5)];
    let them_attacker_w = values[attacker_offset + them.attackers.min(5)];

    score += (us.score(shield_w, ortho_w, diag_w, us_attacker_w) - them.score(shield_w, ortho_w, diag_w, them_attacker_w)) * mg_w;

    let (openness_f, closedness_f) = openness_factors(f.white_pawns, f.black_pawns);
    let mobility_open_offset = psqt::LAYOUT.mobility_open_offset;
    let mobility_closed_offset = psqt::LAYOUT.mobility_closed_offset;

    let mobility = slots.mobility[idx];
    for i in 0..4 {
        let diff = f64::from(mobility[i]) - f64::from(mobility[i + 4]);
        let mobility_w =
            interpolate_weight(values, mobility_open_offset + i, mobility_closed_offset + i, mg_w, eg_w, openness_f, closedness_f);
        score += diff * mobility_w;
    }

    let xray_offset = psqt::LAYOUT.xray_offset;
    score += f64::from(slots.xray_ortho[idx]) * values[xray_offset];

    score
}

struct BoardFeatures {
    white_pawns: u64,
    black_pawns: u64,
    mat_diffs: [f64; 6],
    phase_counts: [f64; 6],
    piece_data: [(usize, usize, f64); 32],
    piece_count: usize,
}

/// Extract perspective-normalised board features from a nibble-encoded entry.
///
/// The entry stores raw colors (bit 3 = 0=White, 1=Black).
/// We normalize to the side-to-move's perspective:
/// STM pieces map to "Us",
/// STM's opponent maps to "Them".
fn extract_board(entry: &SoulEntry) -> BoardFeatures {
    let mut f = BoardFeatures {
        white_pawns: 0,
        black_pawns: 0,
        mat_diffs: [0.0; 6],
        phase_counts: [0.0; 6],
        piece_data: [(0, 0, 0.0); 32],
        piece_count: 0,
    };

    let stm_black = (entry.stm_and_ep & 0x80) != 0;
    let mut occ = entry.occupancy;
    let mut idx = 0usize;

    while occ != 0 {
        let sq = Square(occ.trailing_zeros() as u8);
        occ &= occ - 1;

        let nibble = super::quant::next_nibble(&entry.pieces, &mut idx);
        let pt_raw = (nibble & 0x07) as usize;
        let is_black = (nibble & 0x08) != 0;
        let pt = if pt_raw == 6 { 3 } else { pt_raw }; // unmoved rook → rook
        debug_assert!(pt <= 5, "malformed nibble: pt={pt}");
        if pt > 5 {
            continue;
        }

        let us_piece = is_black == stm_black;
        let sq_idx = if is_black { usize::from(sq.0) } else { usize::from(sq.0 ^ 0x38) };
        let sign = if us_piece { 1.0 } else { -1.0 };

        f.piece_data[f.piece_count] = (pt, sq_idx, sign);
        f.piece_count += 1;

        f.mat_diffs[pt] += sign;
        f.phase_counts[pt] += 1.0;

        if pt == 0 {
            let bit = 1u64 << sq.0;
            if is_black {
                f.black_pawns |= bit;
            } else {
                f.white_pawns |= bit;
            }
        }
    }
    f
}

#[inline]
fn unpack_safety(raw: [u8; 4]) -> SafetyMetrics {
    SafetyMetrics {
        attackers: raw[0] as usize,
        weak: raw[1] as i8 as i32,
        shield: raw[2] as i8 as i32,
        ortho_exposure: (raw[3] >> 4) as i32,
        diag_exposure: (raw[3] & 0x0F) as i32,
    }
}

#[inline(always)]
fn openness_factors(white_pawns: u64, black_pawns: u64) -> (f64, f64) {
    let open_i32 = compute_openness_raw(white_pawns, black_pawns);
    let openness = f64::from(open_i32) / f64::from(OPEN_UNITY);
    (openness, 1.0 - openness)
}

/// Openness-weighted blend of open/closed mobility weights.
#[inline(always)]
fn interpolate_weight(
    values: &[f64],
    open_offset: usize,
    closed_offset: usize,
    mg_w: f64,
    eg_w: f64,
    openness: f64,
    closedness: f64,
) -> f64 {
    let w_mg_val =
        ((values[open_offset] * openness * 1024.0 + values[closed_offset] * closedness * 1024.0 + 512.0) / 1024.0).floor();
    let w_eg_val =
        ((values[open_offset + 4] * openness * 1024.0 + values[closed_offset + 4] * closedness * 1024.0 + 512.0) / 1024.0).floor();
    w_mg_val * mg_w + w_eg_val * eg_w
}
