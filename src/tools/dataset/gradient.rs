//! Packed tuning record [`FeatureRecord`] and its forward/backward passes.
//!
//! [`eval_record`] computes the STM-relative eval from the packed features;
//! [`accumulate_record_grad`] scatters the gradient back through the same
//! terms. [`TermSource`] impls bridge each term's generic scatter to the
//! record's fields.

use std::array;

use super::SoulEntry;
use crate::{
    core::{
        board::Position,
        defs::{Color, Square},
        phase::{compute_phase_f64, compute_phase_weights_f64},
        psqt,
    },
    engine::{
        combiner::{Accumulators, Combiner, LinearCombiner, taper},
        eval::{
            BackwardPawnTerm, BishopPairTerm, DefendedPawnTerm, DoubledPawnTerm, EnemyKingDistTerm, IsolatedPawnTerm,
            MinorBehindPawnTerm, PassedPawnTerm, PhalanxTerm, RookOpenTerm, SharedFeatures, TempoTerm, XrayTerm, apply_all_inputs,
            evaluate_fast, extract_phase, scatter_all_terms,
        },
        mobility::{
            KingSafetyInput, KingSafetyTerm, MobilityInput, MobilityTerm, SafetyMetrics, SideMetrics, compute_openness_raw,
        },
        term::{self, TermSource},
    },
    weave::Vf64x4,
};

/// All tuner-side features for one position, packed into a 132-byte record
/// (three cache lines) so the hot loop reads one record instead of streaming
/// a dozen arrays. Computed once at startup ([`FeatureRecord::from_entry`]);
/// only `values` changes across epochs. The PSQT gather index is pre-resolved
/// and the board decode folded in, so the loop never re-walks the nibble array.
///
/// Fields are STM-relative (us − them); the perspective flip happens once,
/// at pack time.
#[repr(C)]
pub struct FeatureRecord {
    /// Bits 0..14 = MG PSQT index (≤ 351); EG = MG + 32 (≤ 383).
    /// Bit 15 = piece sign (set = "them", subtracted).
    pub piece_idx: [u16; 32],
    pub passed_pawn: [i8; 6],
    pub enemy_king_dist: [i8; 6],
    pub phalanx: [i8; 6],
    pub defended_pawn: [i8; 6],
    /// `[us·4, them·4]`: mobility, shadow_mobility, threats, shadow_threats.
    pub mobility: [i8; 8],
    /// `[attackers, weak, shield, ortho<<4 | diag]`, king-safety metrics.
    pub safety_us: [u8; 4],
    pub safety_them: [u8; 4],
    /// Material count differential per piece type (us − them).
    pub mat_diffs: [i8; 6],
    /// Piece counts per type, for the tapered phase weight.
    pub phase_counts: [u8; 6],
    /// Raw `compute_openness_raw` result; openness = `open_raw / OPEN_UNITY`.
    /// Stored raw (not as a float) to keep the openness math bit-exact.
    pub open_raw: i32,
    /// For volatility filtering at training time.
    pub static_eval: i16,
    pub xray_ortho: i8,
    pub bishop_pair: i8,
    pub rook_open: i8,
    pub doubled_pawn: i8,
    pub isolated_pawn: i8,
    pub backward_pawn: i8,
    pub tempo: i8,
    pub minor_behind_pawn: i8,
    pub piece_count: u8,
}

const _: () = assert!(size_of::<FeatureRecord>() == 132);

impl FeatureRecord {
    /// Startup cost per entry: FEN round-trip + `SharedFeatures::compute`.
    /// A nibble→`Position` decoder would skip the string, but the parse is
    /// negligible next to the feature work. Neither runs in the hot loop.
    pub fn from_entry(entry: &SoulEntry) -> Self {
        let pos = Position::from_fen(&entry.to_fen());

        // SharedFeatures is White-relative; the record is STM-relative, so flip
        // perspective here. Side-symmetric metrics (mobility, safety) swap halves
        // for Black; white-minus-black differentials (xray, pairs, passers) negate.
        let sf = SharedFeatures::compute(&pos);
        let black = pos.stm == Color::Black;

        let (mob_us, mob_them, saf_us, saf_them) = if black {
            (&sf.data.metrics_them, &sf.data.metrics_us, &sf.data.safety_them, &sf.data.safety_us)
        } else {
            (&sf.data.metrics_us, &sf.data.metrics_them, &sf.data.safety_us, &sf.data.safety_them)
        };

        let pack_side = |m: &SideMetrics| {
            [
                m.mobility.clamp(-127, 127) as i8,
                m.shadow_mobility.clamp(-127, 127) as i8,
                m.threats.clamp(-127, 127) as i8,
                m.shadow_threats.clamp(-127, 127) as i8,
            ]
        };

        let mut mobility = [0i8; 8];
        mobility[..4].copy_from_slice(&pack_side(mob_us));
        mobility[4..].copy_from_slice(&pack_side(mob_them));

        let sign = if black { -1 } else { 1 };

        let (piece_idx, piece_count, mat_diffs, phase_counts, open_raw) = pack_board(entry);

        let acc = pos.get_initial_accumulator();
        let phase = extract_phase(&acc);

        Self {
            piece_idx,
            passed_pawn: array::from_fn(|i| (sf.passed_pawn[i] * sign) as i8),
            enemy_king_dist: array::from_fn(|i| (sf.enemy_king_dist[i] * sign) as i8),
            phalanx: array::from_fn(|i| (sf.phalanx[i] * sign) as i8),
            defended_pawn: array::from_fn(|i| (sf.defended_pawn[i] * sign) as i8),
            mobility,
            safety_us: pack_safety(saf_us),
            safety_them: pack_safety(saf_them),
            mat_diffs,
            phase_counts,
            open_raw,
            static_eval: evaluate_fast(&pos, &acc, phase) as i16,
            xray_ortho: (sf.xray_ortho * sign) as i8,
            bishop_pair: (sf.bishop_pair_diff * sign) as i8,
            rook_open: (sf.rook_open_diff * sign) as i8,
            doubled_pawn: (sf.doubled_pawn_diff * sign) as i8,
            isolated_pawn: (sf.isolated_pawn_diff * sign) as i8,
            backward_pawn: (sf.backward_pawn_diff * sign) as i8,
            tempo: (sf.tempo * sign) as i8,
            minor_behind_pawn: (sf.minor_behind_pawn_diff * sign) as i8,
            piece_count,
        }
    }
}

/// Compute the STM-relative eval for `record` under the parameter vector `values`.
///
/// Fills the same buckets `fill_accumulators` fills from a board, through the
/// same registered terms, so [`LinearCombiner`] owns every rounding site.
#[inline]
pub fn eval_record(record: &FeatureRecord, values: &[f64]) -> f64 {
    let l = &psqt::LAYOUT;
    let phase_counts: [f64; 6] = array::from_fn(|i| f64::from(record.phase_counts[i]));
    let phase = compute_phase_f64(&phase_counts, values);

    let mut lane_mg = 0.0;
    let mut lane_eg = 0.0;

    // PSQT: a data-dependent gather over the 384-entry table, the one loop whose
    // index can't be proven in bounds and whose body runs up to 32× per position.
    for i in 0..record.piece_count as usize {
        let (mg_idx, eg_idx, sign) = decode_piece(record.piece_idx[i]);

        // SAFETY: mg_idx ≤ 351 (5·64+31), eg_idx = mg_idx+32 ≤ 383 < 384. See `decode_piece`.
        unsafe {
            lane_mg += sign * *values.get_unchecked(mg_idx);
            lane_eg += sign * *values.get_unchecked(eg_idx);
        }
    }

    // Zero diff adds nothing, so eval omits the zero-diff guard the gradient scatter keeps.
    let mat = l.material_offset;

    for pt in 0..6 {
        lane_mg += f64::from(record.mat_diffs[pt]) * values[mat + pt];
        lane_eg += f64::from(record.mat_diffs[pt]) * values[mat + 6 + pt];
    }

    let mut buckets = Accumulators::<f64> {
        mg_eg: taper(lane_mg, lane_eg, phase),
        mobility: 0.0,
        bonus_mg: 0.0,
        bonus_eg: 0.0,
        safety_us: 0.0,
        safety_them: 0.0,
        xray: 0.0,
    };

    apply_all_inputs(record, values, phase, &mut buckets);
    LinearCombiner::forward(&buckets, phase)
}

/// Accumulate parameter gradients for `record` into `grads`,
/// scaled by the upstream `gradient` (∂loss/∂score).
pub fn accumulate_record_grad(record: &FeatureRecord, values: &[f64], gradient: f64, grads: &mut [f64]) {
    let phase_counts: [f64; 6] = array::from_fn(|i| f64::from(record.phase_counts[i]));
    let (mg_w, eg_w) = compute_phase_weights_f64(&phase_counts, values);

    let l = &psqt::LAYOUT;

    for i in 0..record.piece_count as usize {
        let (mg_idx, eg_idx, sign) = decode_piece(record.piece_idx[i]);

        // SAFETY: mg_idx ≤ 351, eg_idx = mg_idx+32 ≤ 383 < 384. See `decode_piece`.
        unsafe {
            *grads.get_unchecked_mut(mg_idx) += gradient * sign * mg_w;
            *grads.get_unchecked_mut(eg_idx) += gradient * sign * eg_w;
        }
    }

    let mat = l.material_offset;

    for pt in 0..6 {
        let diff = f64::from(record.mat_diffs[pt]);

        if diff.abs() > 0.001 {
            grads[mat + pt] += gradient * diff * mg_w;
            grads[mat + 6 + pt] += gradient * diff * eg_w;
        }
    }

    let upstreams = term::BucketUpstreams {
        mg_eg: term::TaperPair { d_mg: gradient * mg_w, d_eg: gradient * eg_w },
        mobility: term::TaperPair { d_mg: gradient * mg_w, d_eg: gradient * eg_w },
        bonus: term::TaperPair { d_mg: gradient * mg_w, d_eg: gradient * eg_w },
        king_safety: gradient * mg_w,
        xray: gradient * mg_w,
    };

    scatter_all_terms(record, &upstreams, grads);
}

/// Sign bit of a packed [`FeatureRecord::piece_idx`] slot.
const PIECE_SIGN: u16 = 0x8000;

/// Decode a packed piece slot into `(mg_idx, eg_idx, sign)` for the PSQT gather.
///
/// `mg_idx` is the MG PSQT index (≤ 351); `eg_idx = mg_idx + 32` (≤ 383); `sign`
/// is +1.0 for our pieces, −1.0 for theirs.
#[inline(always)]
fn decode_piece(packed: u16) -> (usize, usize, f64) {
    let mg_idx = (packed & !PIECE_SIGN) as usize;
    let sign = if packed & PIECE_SIGN != 0 { -1.0 } else { 1.0 };

    (mg_idx, mg_idx + 32, sign)
}

/// Walk the entry's pieces once, producing the PSQT gather indices, material
/// differentials, phase counts, and raw openness, STM-normalized.
///
/// Mirrors the encoder's nibble layout: bits 0-2 = type, bit 3 = color. An
/// unmoved-rook code (6) folds back to a rook (3).
fn pack_board(entry: &SoulEntry) -> ([u16; 32], u8, [i8; 6], [u8; 6], i32) {
    let mut piece_idx = [0u16; 32];
    let mut count = 0usize;
    let mut mat_diffs = [0i32; 6];
    let mut phase_counts = [0u8; 6];
    let mut white_pawns = 0u64;
    let mut black_pawns = 0u64;

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

        let mg_idx = (pt * 64 + psqt::mirror_sq(sq_idx)) as u16;
        piece_idx[count] = if us_piece { mg_idx } else { mg_idx | PIECE_SIGN };
        count += 1;

        mat_diffs[pt] += if us_piece { 1 } else { -1 };
        phase_counts[pt] += 1;

        if pt == 0 {
            let bit = 1u64 << sq.0;

            if is_black {
                black_pawns |= bit;
            } else {
                white_pawns |= bit;
            }
        }
    }

    let mat_diffs = array::from_fn(|i| mat_diffs[i] as i8);
    (piece_idx, count as u8, mat_diffs, phase_counts, compute_openness_raw(white_pawns, black_pawns))
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

// TermSource impls bridging FeatureRecord → each generic term's scatter.
macro_rules! impl_fr_sources {
    ( $( ($term:ty, scalar, $field:ident) ),* ; $( ($arr_term:ty, array, $arr_field:ident, $n:literal) ),* ) => {
        $(
            impl term::TermSource<$term> for FeatureRecord {
                type Input = f64;
                #[inline(always)]
                fn extract(&self) -> f64 { f64::from(self.$field) }
            }
        )*
        $(
            impl term::TermSource<$arr_term> for FeatureRecord {
                type Input = [f64; $n];
                #[inline(always)]
                fn extract(&self) -> [f64; $n] {
                    std::array::from_fn(|i| f64::from(self.$arr_field[i]))
                }
            }
        )*
    };
}

impl_fr_sources! {
    (BishopPairTerm, scalar, bishop_pair),
    (RookOpenTerm, scalar, rook_open),
    (DoubledPawnTerm, scalar, doubled_pawn),
    (IsolatedPawnTerm, scalar, isolated_pawn),
    (BackwardPawnTerm, scalar, backward_pawn),
    (TempoTerm, scalar, tempo),
    (MinorBehindPawnTerm, scalar, minor_behind_pawn),
    (XrayTerm, scalar, xray_ortho);

    (PassedPawnTerm, array, passed_pawn, 6),
    (EnemyKingDistTerm, array, enemy_king_dist, 6),
    (PhalanxTerm, array, phalanx, 6),
    (DefendedPawnTerm, array, defended_pawn, 6)
}

impl TermSource<MobilityTerm> for FeatureRecord {
    type Input = MobilityInput;

    #[inline(always)]
    fn extract(&self) -> MobilityInput {
        MobilityInput {
            diff: Vf64x4::from([
                f64::from(self.mobility[0]) - f64::from(self.mobility[4]),
                f64::from(self.mobility[1]) - f64::from(self.mobility[5]),
                f64::from(self.mobility[2]) - f64::from(self.mobility[6]),
                f64::from(self.mobility[3]) - f64::from(self.mobility[7]),
            ]),
            openness: self.open_raw,
        }
    }
}

impl TermSource<KingSafetyTerm> for FeatureRecord {
    type Input = KingSafetyInput;

    #[inline(always)]
    fn extract(&self) -> KingSafetyInput {
        KingSafetyInput { us: unpack_safety(self.safety_us), them: unpack_safety(self.safety_them) }
    }
}
