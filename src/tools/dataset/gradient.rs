//! Packed tuning record [`FeatureRecord`] and evaluation gradient pipelines.
//!
//! [`eval_record`] evaluates a packed position from the side-to-move perspective.
//! [`accumulate_record_grad`] propagates loss gradients back to evaluation weights.
//! [`TermSource`] implementations connect each term's generic accumulator logic to record fields.

use super::SoulEntry;
use crate::{
    core::{
        defs::{Color, TOTAL_PHASE},
        phase::compute_phase_f64,
        psqt,
    },
    engine::{
        combiner::{Accumulators, Combiner, CombinerParams, LinearCombiner, taper},
        eval::{SharedFeatures, XrayTerm, apply_all_inputs, evaluate_fast, extract_phase, scatter_all_terms},
        eval_params::LAYOUT,
        mobility::{KingSafetyInput, KingSafetyTerm, MobilityInput, MobilityTerm, SafetyMetrics, compute_openness_raw},
        term::{self, TermSource},
    },
    weave::Vf64x4,
};

/// Packed 104-byte training position representation.
///
/// Stores pre-resolved evaluation features and PSQT indices in side-to-move
/// perspective (us - them). Feature extraction runs once at startup; only weight
/// values change across training epochs.
#[repr(C)]
#[derive(Default)]
pub struct FeatureRecord {
    pub piece_slot: [PieceSlot; 32],
    pub passed_pawn: [i8; 6],
    pub enemy_king_dist: [i8; 6],
    pub phalanx: [i8; 6],
    pub defended_pawn: [i8; 6],
    /// Metric differentials (us - them): mobility, shadow mobility, threats, shadow
    /// threats. i16 because these pass ±127.
    pub mobility_diff: [i16; 4],
    /// Packed king safety metrics: `[attackers, weak, shield, (ortho << 4) | diag]`.
    pub safety_us: [u8; 4],
    pub safety_them: [u8; 4],
    /// Piece count differential per type (us - them).
    pub mat_diffs: [i8; 6],
    /// Total piece counts per type across both sides for game phase calculation.
    pub phase_counts: [u8; 6],
    /// Fixed-point openness from `compute_openness_raw`, kept as the integer so the
    /// openness math matches the engine's.
    pub open_raw: i32,
    /// Fast static evaluation for volatility filtering.
    pub static_eval: i16,
    /// Search evaluation label (side-to-move relative), or [`SoulEntry::NO_SCORE`].
    pub score: i16,
    /// Game outcome label (side-to-move relative: 0 = loss, 1 = draw, 2 = win).
    pub result: u8,
    pub xray_ortho: i8,
    pub bishop_pair_diff: i8,
    pub rook_open_diff: i8,
    pub doubled_pawn_diff: i8,
    pub isolated_pawn_diff: i8,
    pub backward_pawn_diff: i8,
    pub tempo: i8,
    pub minor_behind_pawn_diff: i8,
    pub piece_count: u8,
    /// Number of friendly pieces. Indices `0..us_count` add to score; `us_count..piece_count` subtract.
    pub us_count: u8,
}

const _: () = assert!(size_of::<FeatureRecord>() == 104);

impl FeatureRecord {
    /// Partitions active piece slots into friendly (additive) and opponent (subtractive) slices.
    #[inline(always)]
    fn gather(&self) -> (&[PieceSlot], &[PieceSlot]) {
        self.piece_slot[..self.piece_count as usize].split_at(self.us_count as usize)
    }

    /// Extracts and packs evaluation features from a dataset entry.
    pub fn from_entry(entry: &SoulEntry) -> Self {
        let pos = entry.to_board();

        // SharedFeatures are White-relative; flip metrics when Black is to move.
        let shared = SharedFeatures::compute(&pos);
        let is_black = pos.stm == Color::Black;

        let (mob_us, mob_them, saf_us, saf_them) = if is_black {
            (&shared.data.metrics_them, &shared.data.metrics_us, &shared.data.safety_them, &shared.data.safety_us)
        } else {
            (&shared.data.metrics_us, &shared.data.metrics_them, &shared.data.safety_us, &shared.data.safety_them)
        };

        let mobility_diff = [
            (mob_us.mobility - mob_them.mobility) as i16,
            (mob_us.shadow_mobility - mob_them.shadow_mobility) as i16,
            (mob_us.threats - mob_them.threats) as i16,
            (mob_us.shadow_threats - mob_them.shadow_threats) as i16,
        ];

        let sign = if is_black { -1 } else { 1 };

        let acc = pos.get_initial_accumulator();
        let phase = extract_phase(&acc);

        let mut record = Self {
            mobility_diff,
            safety_us: pack_safety(saf_us),
            safety_them: pack_safety(saf_them),
            static_eval: evaluate_fast(&pos, &acc, phase) as i16,
            score: entry.score,
            result: entry.result,
            ..Default::default()
        };

        record.pack_board(entry);
        record.pack_terms(&shared, sign);
        record
    }
}

/// Intermediate state from a forward evaluation pass, consumed by backward gradient accumulation.
pub struct RecordEval {
    pub score: f64,
    pub phase: f64,
    pub buckets: Accumulators<f64>,
    pub combiner: CombinerParams<f64>,
}

#[inline]
pub fn eval_record(record: &FeatureRecord, values: &[f64]) -> f64 { eval_record_full(record, values).score }

/// Evaluates a packed record using the provided weight vector.
///
/// Fills the same buckets `fill_accumulators` fills from a board, through the same
/// registered terms, so `LinearCombiner` owns every rounding site.
#[inline]
pub fn eval_record_full(record: &FeatureRecord, values: &[f64]) -> RecordEval {
    let layout = &LAYOUT;
    let phase_counts = record.phase_counts.map(f64::from);
    let phase = compute_phase_f64(&phase_counts, values);

    // Sum PSQT contributions separately for friendly and opponent pieces.
    let (ours, theirs) = record.gather();

    let sum_psqt = |slots: &[PieceSlot]| {
        slots.iter().fold((0.0, 0.0), |(mg, eg), slot| {
            let (mg_idx, eg_idx) = slot.indices();
            // SAFETY: PieceSlot::new takes a piece type <= 5 with a mirrored square <= 31,
            // so mg_idx <= 5·64+31 = 351 and eg_idx = mg_idx+32 <= 383, both inside the
            // 384-entry PSQT block. Default gives slot zero, which is also in range.
            unsafe { (mg + *values.get_unchecked(mg_idx), eg + *values.get_unchecked(eg_idx)) }
        })
    };

    let (us_mg, us_eg) = sum_psqt(ours);
    let (them_mg, them_eg) = sum_psqt(theirs);

    let mut mg_total = us_mg - them_mg;
    let mut eg_total = us_eg - them_eg;

    // Most types are level in most positions, so the zero products are worth skipping.
    // The gradient scatter guards the same way.
    let mat_offset = layout.material_offset;
    for pt in 0..6 {
        let diff = f64::from(record.mat_diffs[pt]);
        if diff != 0.0 {
            mg_total += diff * values[mat_offset + pt];
            eg_total += diff * values[mat_offset + 6 + pt];
        }
    }

    let mut buckets = Accumulators::<f64> {
        mg_eg: taper(mg_total, eg_total, phase),
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

/// Propagates upstream loss gradients (dLoss/dScore) back into the parameter gradient buffer.
pub fn accumulate_record_grad(record: &FeatureRecord, eval: &RecordEval, gradient: f64, grads: &mut [f64]) {
    let mg_weight = eval.phase / f64::from(TOTAL_PHASE);
    let eg_weight = 1.0 - mg_weight;

    let layout = &LAYOUT;
    let (ours, theirs) = record.gather();

    let mut scatter = |slots: &[PieceSlot], mg_step: f64, eg_step: f64| {
        for slot in slots {
            let (mg_idx, eg_idx) = slot.indices();
            // SAFETY: Invariants guaranteed by PieceSlot construction.
            unsafe {
                *grads.get_unchecked_mut(mg_idx) += mg_step;
                *grads.get_unchecked_mut(eg_idx) += eg_step;
            }
        }
    };

    let mg_step = gradient * mg_weight;
    let eg_step = gradient * eg_weight;

    scatter(ours, mg_step, eg_step);
    scatter(theirs, -mg_step, -eg_step);

    let mat_offset = layout.material_offset;
    for pt in 0..6 {
        let diff = f64::from(record.mat_diffs[pt]);
        if diff.abs() > 0.001 {
            grads[mat_offset + pt] += gradient * diff * mg_weight;
            grads[mat_offset + 6 + pt] += gradient * diff * eg_weight;
        }
    }

    let upstreams = LinearCombiner::backward(&eval.buckets, eval.phase, &eval.combiner, gradient, grads);
    scatter_all_terms(record, &upstreams, grads);
}

/// Compact 8-bit PSQT entry address (`piece_type * 32 + mirror_sq`).
///
/// Encoded with a tight 32-entry stride since `mirror_sq <= 31`.
/// [`PieceSlot::mg_index`] expands this to the standard 64-entry table stride.
#[repr(transparent)]
#[derive(Clone, Copy, Default)]
pub struct PieceSlot(u8);

impl PieceSlot {
    #[inline(always)]
    fn new(piece_type: usize, mirror_sq: usize) -> Self {
        debug_assert!(piece_type <= 5 && mirror_sq <= 31, "slot out of range: {piece_type}, {mirror_sq}");
        Self((piece_type * 32 + mirror_sq) as u8)
    }

    /// Computes the MG index (range 0..=351).
    ///
    /// Expands `piece_type * 32 + sq` into `piece_type * 64 + sq` via `slot + (slot & !31)`.
    #[inline(always)]
    const fn mg_index(self) -> usize {
        let slot = self.0 as usize;
        slot + (slot & !31)
    }

    /// Returns `(mg_index, eg_index)`. The endgame table sits 32 entries after middlegame.
    #[inline(always)]
    const fn indices(self) -> (usize, usize) {
        let mg = self.mg_index();
        (mg, mg + 32)
    }
}

impl FeatureRecord {
    /// Unpacks piece nibbles into partitioned gather slots, material differentials,
    /// phase counts, and pawn openness.
    fn pack_board(&mut self, entry: &SoulEntry) {
        let mut us_count = 0usize;
        let mut them_slots = [PieceSlot::default(); 32];
        let mut them_count = 0usize;
        let mut mat_diffs = [0i32; 6];
        let mut phase_counts = [0u8; 6];
        let mut white_pawns = 0u64;
        let mut black_pawns = 0u64;

        let stm_black = (entry.stm_and_ep & 0x80) != 0;

        for (sq, nibble) in super::quant::packed_pieces(entry.occupancy, &entry.pieces) {
            let raw_type = (nibble & 0x07) as usize;
            let is_black = (nibble & 0x08) != 0;
            let piece_type = if raw_type == 6 { 3 } else { raw_type }; // Map castling rook back to normal rook
            debug_assert!(piece_type <= 5, "malformed nibble: pt={piece_type}");

            if piece_type > 5 {
                continue;
            }

            let is_us = is_black == stm_black;
            let sq_idx = if is_black { sq.as_usize() } else { sq.flip_rank().as_usize() };
            let slot = PieceSlot::new(piece_type, psqt::mirror_sq(sq_idx));

            if is_us {
                self.piece_slot[us_count] = slot;
                us_count += 1;
            } else {
                them_slots[them_count] = slot;
                them_count += 1;
            }

            mat_diffs[piece_type] += if is_us { 1 } else { -1 };
            phase_counts[piece_type] += 1;

            if piece_type == 0 {
                let bit = 1u64 << sq.0;
                if is_black {
                    black_pawns |= bit;
                } else {
                    white_pawns |= bit;
                }
            }
        }

        self.piece_slot[us_count..us_count + them_count].copy_from_slice(&them_slots[..them_count]);
        self.us_count = us_count as u8;
        self.piece_count = (us_count + them_count) as u8;
        self.mat_diffs = mat_diffs.map(|diff| diff as i8);
        self.phase_counts = phase_counts;
        self.open_raw = compute_openness_raw(white_pawns, black_pawns);
    }
}

/// Packs safety metrics into 4 bytes:
/// - Byte 0: Attacker count
/// - Byte 1: Weak square count (signed i8 as u8)
/// - Byte 2: Pawn shield score (signed i8 as u8)
/// - Byte 3: `(ortho_exposure << 4) | diag_exposure` (each 4 bits, 0..=15)
#[inline]
fn pack_safety(m: &SafetyMetrics) -> [u8; 4] {
    [m.attackers as u8, m.weak as u8, m.shield as u8, ((m.ortho_exposure as u8) << 4) | m.diag_exposure as u8]
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

macro_rules! record_bonus {
    ( [] $( $block:ident = $kind:ident ( $term:ident, $($spec:tt)* ) ; )* ) => {
        impl FeatureRecord {
            fn pack_terms(&mut self, shared: &SharedFeatures, sign: i32) {
                self.xray_ortho = (shared.xray_ortho * sign) as i8;
                $( record_bonus!(@pack $kind self, shared, sign, $($spec)*); )*
            }
        }

        $( record_bonus!(@source $kind $term, $($spec)*); )*
    };

    (@pack scalar $rec:ident, $shared:ident, $sign:ident, $field:ident, $($rest:tt)*) => {
        $rec.$field = ($shared.$field * $sign) as i8
    };

    (@pack array $rec:ident, $shared:ident, $sign:ident, $field:ident, $($rest:tt)*) => {
        for (out, &raw) in $rec.$field.iter_mut().zip($shared.$field.iter()) {
            *out = (raw * $sign) as i8;
        }
    };

    (@source scalar $term:ident, $field:ident, $($rest:tt)*) => {
        impl term::TermSource<$crate::engine::eval::$term> for FeatureRecord {
            type Input = f64;

            #[inline(always)]
            fn extract(&self) -> f64 { f64::from(self.$field) }
        }
    };

    (@source array $term:ident, $field:ident, $mg:ident, $eg:ident, $n:literal) => {
        impl term::TermSource<$crate::engine::eval::$term> for FeatureRecord {
            type Input = [f64; $n];

            #[inline(always)]
            fn extract(&self) -> [f64; $n] {
                self.$field.map(f64::from)
            }
        }
    };
}

crate::bonus_terms!(record_bonus);

impl TermSource<MobilityTerm> for FeatureRecord {
    type Input = MobilityInput;

    #[inline(always)]
    fn extract(&self) -> MobilityInput {
        MobilityInput {
            diff: Vf64x4::from([
                f64::from(self.mobility_diff[0]),
                f64::from(self.mobility_diff[1]),
                f64::from(self.mobility_diff[2]),
                f64::from(self.mobility_diff[3]),
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

impl TermSource<XrayTerm> for FeatureRecord {
    type Input = f64;

    #[inline(always)]
    fn extract(&self) -> f64 { f64::from(self.xray_ortho) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{board::Position, defs::PieceType};

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

    #[test]
    fn packed_differentials_recount_from_a_fresh_board() {
        use crate::tools::dataset::quant;

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
        ];

        for fen in fens {
            let pos = Position::from_fen(fen);
            let entry = quant::from_board(&pos, 0.0, None);
            let record = FeatureRecord::from_entry(&entry);

            for pt in PieceType::ALL {
                let us = pos.pieces(pt, pos.stm).popcount() as i32;
                let them = pos.pieces(pt, pos.stm.opposite()).popcount() as i32;
                assert_eq!(i32::from(record.mat_diffs[pt.as_usize()]), us - them, "{fen}");
            }
            assert_eq!(i32::from(record.piece_count), pos.occupancy().popcount() as i32, "{fen}");
        }
    }
}
