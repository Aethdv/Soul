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
        defs::{Color, Square, TOTAL_PHASE},
        phase::compute_phase_f64,
        psqt,
    },
    engine::{
        combiner::{Accumulators, Combiner, CombinerParams, LinearCombiner, taper},
        eval::{
            BackwardPawnTerm, BishopPairTerm, DefendedPawnTerm, DoubledPawnTerm, EnemyKingDistTerm, IsolatedPawnTerm,
            MinorBehindPawnTerm, PassedPawnTerm, PhalanxTerm, RookOpenTerm, SharedFeatures, TempoTerm, XrayTerm, apply_all_inputs,
            evaluate_fast, extract_phase, scatter_all_terms,
        },
        eval_params::LAYOUT,
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
#[derive(Default)]
pub struct FeatureRecord {
    pub piece_slot: [PieceSlot; 32],
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
    /// Bit `i` set means slot `i` is theirs and subtracts. Sits here so its
    /// alignment costs no padding.
    pub piece_signs: u32,
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

const _: () = assert!(size_of::<FeatureRecord>() == 104);

impl FeatureRecord {
    /// `(mg_index, eg_index, sign)` for the `i`th piece of the gather.
    #[inline(always)]
    fn piece(&self, i: usize) -> (usize, usize, f64) {
        let mg = self.piece_slot[i].mg_index();
        let sign = if self.piece_signs & (1 << i) == 0 { 1.0 } else { -1.0 };

        (mg, mg + 32, sign)
    }

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

        let acc = pos.get_initial_accumulator();
        let phase = extract_phase(&acc);

        let mut record = Self {
            mobility,
            safety_us: pack_safety(saf_us),
            safety_them: pack_safety(saf_them),
            static_eval: evaluate_fast(&pos, &acc, phase) as i16,
            ..Default::default()
        };

        record.pack_board(entry);
        record.pack_terms(&sf, sign);
        record
    }
}

/// A forward pass, kept whole: [`accumulate_record_grad`] reads the buckets a
/// non-linear combiner differentiates, and the phase both halves taper by.
pub struct RecordEval {
    pub score: f64,
    pub phase: f64,
    pub buckets: Accumulators<f64>,
    /// Carried, since `accumulate_record_grad` is handed this and not `values`.
    pub combiner: CombinerParams<f64>,
}

#[inline]
pub fn eval_record(record: &FeatureRecord, values: &[f64]) -> f64 {
    eval_record_full(record, values).score
}

/// Compute the STM-relative eval for `record` under the parameter vector `values`.
///
/// Fills the same buckets `fill_accumulators` fills from a board, through the
/// same registered terms, so [`LinearCombiner`] owns every rounding site.
#[inline]
pub fn eval_record_full(record: &FeatureRecord, values: &[f64]) -> RecordEval {
    let l = &LAYOUT;
    let phase_counts: [f64; 6] = array::from_fn(|i| f64::from(record.phase_counts[i]));
    let phase = compute_phase_f64(&phase_counts, values);

    let mut lane_mg = 0.0;
    let mut lane_eg = 0.0;

    // PSQT: a data-dependent gather over the 384-entry table, the one loop whose
    // index can't be proven in bounds and whose body runs up to 32× per position.
    for i in 0..record.piece_count as usize {
        let (mg_idx, eg_idx, sign) = record.piece(i);

        // SAFETY: `PieceSlot::new` is the only way to build a slot and takes a piece
        // type ≤ 5 with a mirrored square ≤ 31, so `mg_index() ≤ 5·64+31 = 351` and
        // `eg_idx = mg_idx+32 ≤ 383`, both inside the 384-entry PSQT block.
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
        danger_us: 0.0,
        danger_them: 0.0,
        xray: 0.0,
    };

    apply_all_inputs(record, values, phase, &mut buckets);

    let combiner = CombinerParams::from_values(values);

    RecordEval { score: LinearCombiner::forward(&buckets, phase, &combiner), phase, buckets, combiner }
}

/// Accumulate parameter gradients for `record` into `grads`,
/// scaled by the upstream `gradient` (∂loss/∂score).
pub fn accumulate_record_grad(record: &FeatureRecord, eval: &RecordEval, gradient: f64, grads: &mut [f64]) {
    let mg_w = eval.phase / f64::from(TOTAL_PHASE);
    let eg_w = 1.0 - mg_w;

    let l = &LAYOUT;

    for i in 0..record.piece_count as usize {
        let (mg_idx, eg_idx, sign) = record.piece(i);

        // SAFETY: as in `eval_record_full`, the bound is `PieceSlot`'s to keep.
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

    let upstreams = LinearCombiner::backward(&eval.buckets, eval.phase, &eval.combiner, gradient, grads);

    scatter_all_terms(record, &upstreams, grads);
}

/// One piece's PSQT address, as `piece_type · 32 + mirror_sq`.
///
/// The table is addressed `piece_type · 64 + mirror_sq`, but `mirror_sq` only
/// ever reaches 31, so half of every 64-block is unreachable and the index fits
/// a byte with the blocks packed tight. [`PieceSlot::mg_index`] spreads them
/// back out.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct PieceSlot(u8);

impl PieceSlot {
    /// The only constructor, which is what makes the bound in `mg_index` a
    /// property of the type rather than of its call sites.
    #[inline(always)]
    fn new(piece_type: usize, mirror_sq: usize) -> Self {
        debug_assert!(piece_type <= 5 && mirror_sq <= 31, "slot out of range: {piece_type}, {mirror_sq}");

        Self((piece_type * 32 + mirror_sq) as u8)
    }

    /// The MG index, at most 351.
    ///
    /// `pt · 64 + sq` is `(pt · 32 + sq) + pt · 32`, and `pt · 32` is the slot
    /// with its square masked away, so restoring the wider stride is one AND and
    /// one add.
    #[inline(always)]
    const fn mg_index(self) -> usize {
        let slot = self.0 as usize;

        slot + (slot & !31)
    }
}

impl FeatureRecord {
    /// Walks the entry's pieces once, filling the gather slots, their signs, the
    /// material differentials, the phase counts and the raw openness.
    ///
    /// Mirrors the encoder's nibble layout: bits 0-2 = type, bit 3 = color. An
    /// unmoved-rook code (6) folds back to a rook (3).
    fn pack_board(&mut self, entry: &SoulEntry) {
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

            self.piece_slot[count] = PieceSlot::new(pt, psqt::mirror_sq(sq_idx));

            if !us_piece {
                self.piece_signs |= 1 << count;
            }
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

        self.piece_count = count as u8;
        self.mat_diffs = array::from_fn(|i| mat_diffs[i] as i8);
        self.phase_counts = phase_counts;
        self.open_raw = compute_openness_raw(white_pawns, black_pawns);
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

macro_rules! record_terms {
    (
        scalar { $( ($term:ty, $field:ident, $src:ident) ),* $(,)? }
        array  { $( ($arr_term:ty, $arr_field:ident, $arr_src:ident, $n:literal) ),* $(,)? }
    ) => {
        impl FeatureRecord {
            fn pack_terms(&mut self, sf: &SharedFeatures, sign: i32) {
                $( self.$field = (sf.$src * sign) as i8; )*
                $(
                    for i in 0..$n {
                        self.$arr_field[i] = (sf.$arr_src[i] * sign) as i8;
                    }
                )*
            }
        }

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

record_terms! {
    scalar {
        (BishopPairTerm, bishop_pair, bishop_pair_diff),
        (RookOpenTerm, rook_open, rook_open_diff),
        (DoubledPawnTerm, doubled_pawn, doubled_pawn_diff),
        (IsolatedPawnTerm, isolated_pawn, isolated_pawn_diff),
        (BackwardPawnTerm, backward_pawn, backward_pawn_diff),
        (TempoTerm, tempo, tempo),
        (MinorBehindPawnTerm, minor_behind_pawn, minor_behind_pawn_diff),
        (XrayTerm, xray_ortho, xray_ortho),
    }
    array {
        (PassedPawnTerm, passed_pawn, passed_pawn, 6),
        (EnemyKingDistTerm, enemy_king_dist, enemy_king_dist, 6),
        (PhalanxTerm, phalanx, phalanx, 6),
        (DefendedPawnTerm, defended_pawn, defended_pawn, 6),
    }
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The bound the PSQT gather's `get_unchecked` rests on, over every slot that
    /// can exist.
    #[test]
    fn every_slot_addresses_its_own_psqt_entry() {
        for pt in 0..6 {
            for sq in 0..32 {
                let mg = PieceSlot::new(pt, sq).mg_index();

                assert_eq!(mg, pt * 64 + sq, "slot ({pt}, {sq}) lands on the wrong entry");
                assert!(mg + 32 < 384, "slot ({pt}, {sq}) reaches past the table");
            }
        }
    }
}
