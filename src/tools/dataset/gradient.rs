//! Cached evaluation and gradients for the tuner.

use super::{SoulEntry, compute_phase_f64};
use crate::{
    core::{
        defs::{Color, TOTAL_PHASE},
        psqt,
    },
    engine::{
        combiner::{self, Accumulators, CombinerParams, taper},
        eval::{SharedFeatures, XrayTerm, apply_all_inputs, evaluate_psqt, extract_phase, scatter_all_terms},
        eval_params::LAYOUT,
        mobility::{KingSafetyInput, KingSafetyTerm, MobilityInput, MobilityTerm, SafetyMetrics},
        term::{self, TermSource},
    },
    weave::F64x4,
};

#[repr(C)]
#[derive(Default)]
pub struct FeatureRecord {
    psqt_slot: [PsqtSlot; 32],
    pub passed_pawn: [i8; 6],
    pub enemy_king_dist: [i8; 6],
    pub phalanx: [i8; 6],
    pub defended_pawn: [i8; 6],
    pub mobility_diff: [i16; 4],
    pub safety_us: [u8; 4],
    pub safety_them: [u8; 4],
    pub material_diff: [i8; 6],
    pub phase_counts: [u8; 6],
    pub openness: i32,
    pub static_eval: i16,
    pub score: i16,
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
    pub us_count: u8,
}

pub struct RecordEval {
    pub score: f64,
    phase: f64,
    buckets: Accumulators<f64>,
    combiner_params: CombinerParams<f64>,
}

const _: () = assert!(size_of::<FeatureRecord>() == 104);

impl FeatureRecord {
    #[inline(always)]
    fn split_sides(&self) -> (&[PsqtSlot], &[PsqtSlot]) {
        self.psqt_slot[..self.piece_count as usize].split_at(self.us_count as usize)
    }

    pub fn from_entry(entry: &SoulEntry) -> Self {
        let pos = entry.to_board();

        // SharedFeatures are White-relative; flip metrics when Black is to move.
        let shared = SharedFeatures::compute(&pos);
        let is_black = pos.stm == Color::Black;

        let (mob_us, mob_them, saf_us, saf_them) = if is_black {
            (
                &shared.spatial.metrics_them, &shared.spatial.metrics_us, &shared.spatial.safety_them, &shared.spatial.safety_us,
            )
        } else {
            (
                &shared.spatial.metrics_us, &shared.spatial.metrics_them, &shared.spatial.safety_us, &shared.spatial.safety_them,
            )
        };

        let mobility_diff = mob_us.diff(mob_them).map(|d| d as i16);

        let sign = if is_black { -1 } else { 1 };

        let acc = pos.psqt_accumulator();
        let phase = extract_phase(&acc);

        let mut record = Self {
            mobility_diff,
            safety_us: pack_safety(saf_us),
            safety_them: pack_safety(saf_them),
            static_eval: evaluate_psqt(&pos, &acc, phase) as i16,
            score: entry.score,
            result: entry.result,
            openness: shared.openness,
            ..Default::default()
        };

        record.pack_board(entry);
        record.pack_terms(&shared, sign);
        record
    }
}

#[inline]
pub fn eval_record(record: &FeatureRecord, values: &[f64]) -> f64 { eval_record_full(record, values).score }

/// Evaluates a packed record with the same terms and combiner as board evaluation.
#[inline]
pub fn eval_record_full(record: &FeatureRecord, values: &[f64]) -> RecordEval {
    assert!(values.len() >= LAYOUT.total, "values must hold the full parameter vector");
    let layout = &LAYOUT;
    let phase_counts = record.phase_counts.map(f64::from);
    let phase = compute_phase_f64(&phase_counts, values);

    // Sum PSQT contributions separately for friendly and opponent pieces.
    let (ours, theirs) = record.split_sides();

    let sum_psqt = |slots: &[PsqtSlot]| {
        slots.iter().fold((0.0, 0.0), |(mg, eg), slot| {
            let (mg_idx, eg_idx) = slot.indices();
            // SAFETY: values is full length; PsqtSlot holds type <= 5 and square <= 31,
            // so mg_idx <= 351 and eg_idx <= 383.
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
        let diff = f64::from(record.material_diff[pt]);
        if diff != 0.0 {
            mg_total += diff * values[mat_offset + pt];
            eg_total += diff * values[mat_offset + 6 + pt];
        }
    }

    let mut buckets = Accumulators::<f64> {
        material_psqt: taper(mg_total, eg_total, phase),
        mobility: 0.0,
        bonus_mg: 0.0,
        bonus_eg: 0.0,
        safety_us: 0.0,
        safety_them: 0.0,
        pressure_us: 0.0,
        pressure_them: 0.0,
        xray: 0.0,
    };

    apply_all_inputs(record, values, phase, &mut buckets);
    let combiner_params = CombinerParams::from_values(values);
    RecordEval { score: combiner::combine(&buckets, phase, &combiner_params), phase, buckets, combiner_params }
}

/// Adds `d_score` times each parameter's partial derivative into `grads`.
pub fn accumulate_record_grad(record: &FeatureRecord, eval: &RecordEval, d_score: f64, grads: &mut [f64]) {
    assert!(grads.len() >= LAYOUT.total, "grads must hold the full parameter vector");
    let mg_weight = eval.phase / f64::from(TOTAL_PHASE);
    let eg_weight = 1.0 - mg_weight;

    let layout = &LAYOUT;
    let (ours, theirs) = record.split_sides();

    let mut scatter = |slots: &[PsqtSlot], mg_step: f64, eg_step: f64| {
        for slot in slots {
            let (mg_idx, eg_idx) = slot.indices();
            // SAFETY: grads is full length and PsqtSlot indices are in bounds.
            unsafe {
                *grads.get_unchecked_mut(mg_idx) += mg_step;
                *grads.get_unchecked_mut(eg_idx) += eg_step;
            }
        }
    };

    let mg_step = d_score * mg_weight;
    let eg_step = d_score * eg_weight;

    scatter(ours, mg_step, eg_step);
    scatter(theirs, -mg_step, -eg_step);

    let mat_offset = layout.material_offset;

    for pt in 0..6 {
        let diff = f64::from(record.material_diff[pt]);
        if diff != 0.0 {
            grads[mat_offset + pt] += d_score * diff * mg_weight;
            grads[mat_offset + 6 + pt] += d_score * diff * eg_weight;
        }
    }

    let upstreams = combiner::backward(&eval.buckets, eval.phase, &eval.combiner_params, d_score, grads);
    scatter_all_terms(record, &upstreams, grads);
}

#[repr(transparent)]
#[derive(Clone, Copy, Default)]
struct PsqtSlot(u8);

impl PsqtSlot {
    #[inline(always)]
    fn new(piece_type: usize, mirror_sq: usize) -> Self {
        debug_assert!(piece_type <= 5 && mirror_sq <= 31, "slot out of range: {piece_type}, {mirror_sq}");
        Self((piece_type * 32 + mirror_sq) as u8)
    }

    #[inline(always)]
    const fn mg_index(self) -> usize {
        let slot = self.0 as usize;
        slot + (slot & !31)
    }

    #[inline(always)]
    const fn indices(self) -> (usize, usize) {
        let mg = self.mg_index();
        (mg, mg + 32)
    }
}

impl FeatureRecord {
    fn pack_board(&mut self, entry: &SoulEntry) {
        let mut us_count = 0usize;
        let mut them_slots = [PsqtSlot::default(); 32];
        let mut them_count = 0usize;
        let mut material_diff = [0i8; 6];
        let mut phase_counts = [0u8; 6];

        let stm_black = entry.stm() == Color::Black;

        for (sq, nibble) in super::packing::unpack_pieces(entry.occupancy, &entry.pieces) {
            let raw_type = (nibble & 0x07) as usize;
            let is_black = (nibble & 0x08) != 0;
            let piece_type = if raw_type == 6 { 3 } else { raw_type };
            let is_us = is_black == stm_black;
            let sq_idx = if is_black { sq.as_usize() } else { sq.flip_rank().as_usize() };
            let slot = PsqtSlot::new(piece_type, psqt::mirror_sq(sq_idx));

            if is_us {
                self.psqt_slot[us_count] = slot;
                us_count += 1;
            } else {
                them_slots[them_count] = slot;
                them_count += 1;
            }

            material_diff[piece_type] += if is_us { 1 } else { -1 };
            phase_counts[piece_type] += 1;
        }

        self.psqt_slot[us_count..us_count + them_count].copy_from_slice(&them_slots[..them_count]);
        self.us_count = us_count as u8;
        self.piece_count = (us_count + them_count) as u8;
        self.material_diff = material_diff;
        self.phase_counts = phase_counts;
    }
}

// Bytes: attacked_zone, weak, shield, then ortho/diag exposure nibbles (maxima 14/13).
#[inline]
fn pack_safety(m: &SafetyMetrics) -> [u8; 4] {
    [m.attacked_zone as u8, m.weak as u8, m.shield as u8, ((m.ortho_exposure as u8) << 4) | m.diag_exposure as u8]
}

#[inline]
fn unpack_safety(raw: [u8; 4]) -> SafetyMetrics {
    SafetyMetrics {
        attacked_zone: raw[0] as usize,
        weak: raw[1] as i32,
        shield: raw[2] as i32,
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
        MobilityInput { diff: F64x4::from(self.mobility_diff.map(f64::from)), openness: self.openness }
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
                let mg = PsqtSlot::new(pt, sq).mg_index();
                assert_eq!(mg, pt * 64 + sq, "slot ({pt}, {sq}) lands on the wrong entry");
                assert!(mg + 32 < 384, "slot ({pt}, {sq}) reaches past the table");
            }
        }
    }

    #[test]
    fn packed_differentials_recount_from_a_fresh_board() {
        use crate::tools::dataset::packing;

        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1",
        ];

        for fen in fens {
            let pos = Position::from_fen(fen);
            let entry = packing::from_board(&pos, 0.0, None);
            let record = FeatureRecord::from_entry(&entry);

            for pt in PieceType::ALL {
                let us = pos.pieces(pt, pos.stm).popcount() as i32;
                let them = pos.pieces(pt, pos.stm.opposite()).popcount() as i32;
                assert_eq!(i32::from(record.material_diff[pt.as_usize()]), us - them, "{fen}");
            }
            assert_eq!(i32::from(record.piece_count), pos.occ.popcount() as i32, "{fen}");
        }
    }
}
