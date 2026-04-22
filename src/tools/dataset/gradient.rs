//! Gradient computation against dataset entries for evaluation tuning.

use crate::{
    core::{
        defs::{Color, Square},
        psqt,
    },
    tools::dataset::{PackedPiece, SoulEntry},
};

pub fn accumulate_gradient(entry: &SoulEntry, values: &[f64], gradient: f64, grads: &mut [f64]) {
    // Single pass tally + feature extraction
    let f = extract_features(entry);

    let (mg_w, eg_w) = crate::core::phase::compute_phase_weights_f64(&f.phase_counts, values);

    // 1. PSQT gradients
    for i in 0..usize::from(entry.piece_count) {
        let (pt, sq_idx, sign) = f.piece_data[i];
        let mg_idx = (pt * 64) + psqt::mirror_sq(sq_idx);
        let eg_idx = (pt * 64) + 32 + psqt::mirror_sq(sq_idx);
        // SAFETY: pt (0..5) and mirror_sq (0..63) ensure mg_idx and eg_idx stay within the 384-element PSQT bounds.
        unsafe {
            *grads.get_unchecked_mut(mg_idx) += gradient * sign * mg_w;
            *grads.get_unchecked_mut(eg_idx) += gradient * sign * eg_w;
        }
    }

    // 2. Material gradients
    let mat_base = psqt::LAYOUT.material_offset;
    for pt in 0..6 {
        let diff = f.mat_diffs[pt];
        if diff.abs() > 0.001 {
            let mg_idx = mat_base + pt;
            let eg_idx = mat_base + 6 + pt;
            grads[mg_idx] += gradient * diff * mg_w;
            grads[eg_idx] += gradient * diff * eg_w;
        }
    }

    // 3. King Safety gradients
    let atk_offset = psqt::LAYOUT.attacker_offset;
    let safety_offset = psqt::LAYOUT.king_safety_offset;

    let us = entry.safety_us.to_metrics();
    let them = entry.safety_them.to_metrics();

    let us_atk = us.attackers.min(5);
    let them_atk = them.attackers.min(5);

    // Safety gradients (Tapered)
    if us_atk > 0 {
        let idx = atk_offset + us_atk;
        grads[idx] += gradient * (-f64::from(us.weak) / 10.0) * mg_w;
    }

    if them_atk > 0 {
        let idx = atk_offset + them_atk;
        grads[idx] += gradient * (f64::from(them.weak) / 10.0) * mg_w;
    }

    let shield_diff = f64::from(us.shield) - f64::from(them.shield);
    let ortho_diff = f64::from(us.ortho_exposure) - f64::from(them.ortho_exposure);
    let diag_diff = f64::from(us.diag_exposure) - f64::from(them.diag_exposure);

    grads[safety_offset] += gradient * shield_diff * mg_w;
    grads[safety_offset + 1] -= gradient * ortho_diff * mg_w;
    grads[safety_offset + 2] -= gradient * diag_diff * mg_w;

    // 4. X-Ray gradients
    let xray_offset = psqt::LAYOUT.xray_offset;
    grads[xray_offset] += gradient * f64::from(entry.xray_ortho) * mg_w;

    // 5. Mobility gradients
    let (openness, closedness) = compute_openness(f.us_pawns, f.them_pawns);
    let mob_open_offset = psqt::LAYOUT.mobility_open_offset;
    let mob_closed_offset = psqt::LAYOUT.mobility_closed_offset;

    for i in 0..4 {
        let diff = f64::from(entry.mobility[i]) - f64::from(entry.mobility[i + 4]);
        let g_diff = gradient * diff;

        grads[mob_open_offset + i] += g_diff * openness * mg_w;
        grads[mob_open_offset + 4 + i] += g_diff * openness * eg_w;

        grads[mob_closed_offset + i] += g_diff * closedness * mg_w;
        grads[mob_closed_offset + 4 + i] += g_diff * closedness * eg_w;
    }
}

#[inline]
pub fn eval_soul(entry: &SoulEntry, values: &[f64]) -> f64 {
    let f = extract_features(entry);

    let (mg_w, eg_w) = crate::core::phase::compute_phase_weights_f64(&f.phase_counts, values);
    let mut score = 0.0;

    // PSQT contrib
    for i in 0..usize::from(entry.piece_count) {
        let (pt, sq_idx, sign) = f.piece_data[i];
        let mg_idx = (pt * 64) + psqt::mirror_sq(sq_idx);
        let eg_idx = (pt * 64) + 32 + psqt::mirror_sq(sq_idx);
        // SAFETY: pt (0..5) and mirror_sq (0..63) ensure mg_idx and eg_idx stay within the 384-element PSQT bounds.
        unsafe {
            score += sign * (*values.get_unchecked(mg_idx) * mg_w + *values.get_unchecked(eg_idx) * eg_w);
        }
    }

    // Material contrib
    let mat_base = psqt::LAYOUT.material_offset;
    for pt in 0..6 {
        let diff = f.mat_diffs[pt];
        let mg_idx = mat_base + pt;
        let eg_idx = mat_base + 6 + pt;
        // SAFETY: pt is bounded 0..5, guaranteeing mg_idx and eg_idx map strictly into the material parameter slots.
        unsafe {
            score += diff * (*values.get_unchecked(mg_idx) * mg_w + *values.get_unchecked(eg_idx) * eg_w);
        }
    }

    // King Safety contrib
    let safety_offset = psqt::LAYOUT.king_safety_offset;
    let atk_offset = psqt::LAYOUT.attacker_offset;

    let shield_w = values[safety_offset];
    let ortho_w = values[safety_offset + 1];
    let diag_w = values[safety_offset + 2];

    let us = entry.safety_us.to_metrics();
    let them = entry.safety_them.to_metrics();
    let us_atk_w = values[atk_offset + us.attackers.min(5)];
    let them_atk_w = values[atk_offset + them.attackers.min(5)];

    let us_safety = us.score(shield_w, ortho_w, diag_w, us_atk_w);
    let them_safety = them.score(shield_w, ortho_w, diag_w, them_atk_w);

    score += (us_safety - them_safety) * mg_w;

    // Mobility contrib
    let (openness_f, closedness_f) = compute_openness(f.us_pawns, f.them_pawns);
    let open_off = psqt::LAYOUT.mobility_open_offset;
    let closed_off = psqt::LAYOUT.mobility_closed_offset;

    for i in 0..4 {
        let diff = f64::from(entry.mobility[i]) - f64::from(entry.mobility[i + 4]);
        let mob_w = interpolate_weight(values, open_off + i, closed_off + i, mg_w, eg_w, openness_f, closedness_f);
        score += diff * mob_w;
    }

    // X-Ray contrib
    let xray_offset = psqt::LAYOUT.xray_offset;
    score += f64::from(entry.xray_ortho) * values[xray_offset];

    score
}

// ──────── Private Helpers ────────

#[inline(always)]
unsafe fn read_piece(pieces: *const PackedPiece, i: usize) -> PackedPiece {
    // SAFETY: pieces points to a 2-aligned array inside a repr(C) SoulEntry.
    unsafe { *pieces.add(i) }
}

#[inline(always)]
fn pieces_ptr(entry: &SoulEntry) -> *const PackedPiece {
    core::ptr::addr_of!(entry.pieces) as *const PackedPiece
}

struct SpatialFeatures {
    us_pawns: u64,
    them_pawns: u64,
    ksq_us: Square,
    ksq_them: Square,
    mat_diffs: [f64; 6],
    phase_counts: [f64; 6],
    piece_data: [(usize, usize, f64); 32],
}

#[inline(always)]
fn extract_features(entry: &SoulEntry) -> SpatialFeatures {
    let mut f = SpatialFeatures {
        us_pawns: 0,
        them_pawns: 0,
        ksq_us: Square(0),
        ksq_them: Square(0),
        mat_diffs: [0.0; 6],
        phase_counts: [0.0; 6],
        piece_data: [(0, 0, 0.0); 32],
    };

    let ptr = pieces_ptr(entry);
    let count = usize::from(entry.piece_count);

    for i in 0..count {
        // SAFETY: bounded by entry.piece_count.
        let (pt, color, sq) = unsafe { read_piece(ptr, i) }.unpack();
        let color_u8 = color as u8;

        if pt == 5 {
            if color == Color::White {
                f.ksq_us = sq;
            } else {
                f.ksq_them = sq;
            }
        }

        // White (0) -> sq ^ 0x38, Black (1) -> sq ^ 0
        let sq_idx = usize::from(sq.0 ^ ((1 - color_u8) * 0x38));
        // White (0) -> 1.0, Black (1) -> -1.0
        let sign = 1.0 - 2.0 * f64::from(color_u8);

        f.piece_data[i] = (pt, sq_idx, sign);

        // Material tally
        f.mat_diffs[pt] += sign;
        f.phase_counts[pt] += 1.0;

        if pt == 0 {
            let bit = 1u64 << sq.0;
            if color == Color::White {
                f.us_pawns |= bit;
            } else {
                f.them_pawns |= bit;
            }
        }
    }
    f
}

#[inline(always)]
fn compute_openness(us_pawns: u64, them_pawns: u64) -> (f64, f64) {
    let open_i32 = crate::engine::mobility::compute_openness_raw(us_pawns, them_pawns);
    let openness = f64::from(open_i32) / f64::from(crate::engine::mobility::OPEN_UNITY);
    (openness, 1.0 - openness)
}

#[inline(always)]
fn interpolate_weight(
    values: &[f64],
    open_off: usize,
    closed_off: usize,
    mg_w: f64,
    eg_w: f64,
    openness: f64,
    closedness: f64,
) -> f64 {
    let w_mg_val = ((values[open_off] * openness * 1024.0 + values[closed_off] * closedness * 1024.0 + 512.0) / 1024.0).floor();
    let w_eg_val =
        ((values[open_off + 4] * openness * 1024.0 + values[closed_off + 4] * closedness * 1024.0 + 512.0) / 1024.0).floor();
    w_mg_val * mg_w + w_eg_val * eg_w
}
