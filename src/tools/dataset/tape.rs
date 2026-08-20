//! Forward-mode automatic differentiation (AD) using dual numbers.
//!
//! Tracks partial derivatives alongside evaluation values using dual numbers.
//! Serves as an exact ground-truth oracle for verifying analytic linear gradient routines.

use crate::{
    core::{
        board::Position as Board,
        defs::{Color, PieceType, TOTAL_PHASE},
        phase::compute_phase_f64,
        psqt,
    },
    engine::{
        autograd::{
            EnvVec8, EvalMath,
            dual::{DUAL_N, DualNode, DualVec8},
        },
        combiner::{Combiner, CombinerParams, LinearCombiner},
        eval::{EvalParams, SharedFeatures, evaluate_generic, fill_accumulators, scatter_all_terms},
        eval_params::LAYOUT,
        wdl::sigmoid,
    },
};

mod scatter {
    #[inline(always)]
    pub(super) fn scatter_scalar(grad: &[f32], slot: &mut usize, outer: f64, out: &mut [f64], offset: usize) {
        out[offset] += outer * f64::from(grad[*slot]);
        *slot += 1;
    }

    #[inline(always)]
    pub(super) fn scatter_vec4(grad: &[f32], slot: &mut usize, outer: f64, out: &mut [f64], offset: usize) {
        for i in 0..4 {
            out[offset + i] += outer * f64::from(grad[*slot + i]);
        }
        *slot += 4;
    }

    #[inline(always)]
    pub(super) fn scatter_array6(grad: &[f32], slot: &mut usize, outer: f64, out: &mut [f64], offset: usize) {
        for i in 0..6 {
            out[offset + i] += outer * f64::from(grad[*slot + i]);
        }
        *slot += 6;
    }
}

macro_rules! impl_scatter {
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr, $konst:expr) ),* $(,)?) => {
        paste::paste! {
            impl DualEvalResult {
                pub fn scatter_dynamic(&self, outer_deriv: f64, param_grads: &mut [f64]) {
                    let mut slot = 2; // Slots 0 and 1 are reserved for MG and EG accumulator lanes
                    $(
                        scatter::[<scatter_ $ty:lower>](
                            &self.grad,
                            &mut slot,
                            outer_deriv,
                            param_grads,
                            LAYOUT.$offset_field + $extra,
                        );
                    )*
                }
            }
        }
    };
}

crate::define_tunables!(impl_scatter);

/// Output derivatives from a forward-mode dual number evaluation pass.
pub struct DualEvalResult {
    /// Partial derivatives per tracked parameter, zero-padded to `DUAL_N`.
    pub grad: [f32; DUAL_N],
}

/// Evaluates a position and accumulates parameter gradients using forward-mode AD.
///
/// Serves as the ground-truth oracle for `eval_linear_grad`. Returns squared error loss.
#[inline]
pub fn eval_dual_fused(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut lane_vals, &mut piece_counts);

    let mut phase_dual = DualNode::zero();
    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = LAYOUT.phase_offset + pt;
        if phase_idx < values.len() {
            phase_dual += DualNode::constant(*count) * DualNode::constant(values[phase_idx]);
        }
    }

    let phase = phase_dual
        .math_clamp(DualNode::zero(), DualNode::constant(f64::from(TOTAL_PHASE)))
        .trunc();

    let mut dual_acc = DualVec8::zero();
    dual_acc.0[0] = DualNode::seed(lane_vals[0], 0);
    dual_acc.0[1] = DualNode::seed(lane_vals[1], 1);

    for (dual, &val) in dual_acc.0[2..8].iter_mut().zip(&lane_vals[2..8]) {
        *dual = DualNode::constant(val);
    }

    let params = EvalParams::<DualNode>::load_tunable(values);
    let features = SharedFeatures::compute(board);

    let result = evaluate_generic::<DualNode>(board, &dual_acc, phase, &params, Some(&features));
    let score = result.val;

    let sig = sigmoid(score, k);
    let err = sig - target;
    let outer_deriv = 2.0 * err * sig * (1.0 - sig) * k;

    let eval_result = DualEvalResult { grad: result.grad };
    eval_result.scatter_dynamic(outer_deriv, param_grads);

    let d_mg = outer_deriv * f64::from(result.grad[0]);
    let d_eg = outer_deriv * f64::from(result.grad[1]);

    scatter_psqt(board, d_mg, d_eg, param_grads);
    err * err
}

/// Evaluates a position and accumulates parameter gradients using analytic linear derivatives.
///
/// Every parameter enters the eval as `param · feature`, so its gradient is that feature's
/// coefficient and can be written straight out: one f64 per slot, against `DUAL_N` f32 ops
/// per arithmetic op on the dual path. The dual path stays as the oracle for this one.
///
/// Returns squared error loss.
#[inline]
pub fn eval_linear_grad(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut lane_vals, &mut piece_counts);

    // Phase weights are kept fixed; game phase definition must not drift during tuning.
    let phase = compute_phase_f64(&piece_counts, values);

    let params = EvalParams::<f64>::load_tunable(values);
    let features = SharedFeatures::compute(board);
    let mut acc = <f64 as EvalMath>::Vec8::zero();
    acc.0 = lane_vals;

    let buckets = fill_accumulators::<f64>(&acc, phase, &features, &params);
    let combiner = CombinerParams::from_eval(&params);
    let white_score = LinearCombiner::forward(&buckets, phase, &combiner);
    let stm_sign: f64 = if board.stm == Color::White { 1.0 } else { -1.0 };
    let score = white_score * stm_sign;

    let sig = sigmoid(score, k);
    let err = sig - target;
    let outer_deriv = 2.0 * err * sig * (1.0 - sig) * k;
    let stm_outer_deriv = outer_deriv * stm_sign;

    let upstreams = LinearCombiner::backward(&buckets, phase, &combiner, stm_outer_deriv, param_grads);

    let d_mg = upstreams.mg_eg.d_mg;
    let d_eg = upstreams.mg_eg.d_eg;

    scatter_psqt(board, d_mg, d_eg, param_grads);
    scatter_all_terms(&features, &upstreams, param_grads);
    err * err
}

/// Evaluates a board using 64-bit floating point weights.
#[inline(always)]
pub fn eval_f64(board: &Board, values: &[f64]) -> f64 { eval_f64_with_acc(board, values).0 }

/// Evaluates a board and returns the final score along with accumulator lane sums and piece counts.
pub fn eval_f64_with_acc(board: &Board, values: &[f64]) -> (f64, [f64; 8], [f64; 6]) {
    let mut accum = <f64 as EvalMath>::Vec8::zero();
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut accum.0, &mut piece_counts);

    let phase = compute_phase_f64(&piece_counts, values);
    let params = EvalParams::<f64>::load_tunable(values);
    let features = SharedFeatures::compute(board);

    (evaluate_generic::<f64>(board, &accum, phase, &params, Some(&features)), accum.0, piece_counts)
}

/// Scatters material and PSQT parameter gradients across all active pieces on the board.
///
/// Both gradient paths end here. A copy of this loop in each would let one bug sit in
/// both, and the oracle would agree with itself.
#[inline(always)]
fn scatter_psqt(board: &Board, d_mg: f64, d_eg: f64, param_grads: &mut [f64]) {
    debug_assert!(
        param_grads.len() >= LAYOUT.mobility_open_offset,
        "param_grads buffer too small: {} < {} (material + PSQT layout requirement)",
        param_grads.len(),
        LAYOUT.mobility_open_offset,
    );

    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        let mat_mg_idx = LAYOUT.material_offset + pt;
        let mat_eg_idx = LAYOUT.material_offset + 6 + pt;

        let mut bb_w = board.pieces(piece, Color::White);
        let count_w = bb_w.popcount() as f64;

        param_grads[mat_mg_idx] += d_mg * count_w;
        param_grads[mat_eg_idx] += d_eg * count_w;

        while bb_w.is_not_empty() {
            let sq = bb_w.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq.flip_rank()));
            let mg_idx = pt * 64 + mirror_idx;
            let eg_idx = pt * 64 + 32 + mirror_idx;
            param_grads[mg_idx] += d_mg;
            param_grads[eg_idx] += d_eg;
        }

        let mut bb_b = board.pieces(piece, Color::Black);
        let count_b = bb_b.popcount() as f64;
        param_grads[mat_mg_idx] -= d_mg * count_b;
        param_grads[mat_eg_idx] -= d_eg * count_b;

        while bb_b.is_not_empty() {
            let sq = bb_b.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq));
            let mg_idx = pt * 64 + mirror_idx;
            let eg_idx = pt * 64 + 32 + mirror_idx;
            param_grads[mg_idx] -= d_mg;
            param_grads[eg_idx] -= d_eg;
        }
    }
}

/// Accumulates material and PSQT contributions into middlegame (lane 0) and endgame (lane 1) totals.
#[inline(always)]
fn accumulate_lane_vals(board: &Board, values: &[f64], lane_vals: &mut [f64], piece_counts: &mut [f64; 6]) {
    debug_assert!(
        values.len() >= LAYOUT.mobility_open_offset,
        "values buffer too short: {} < {} (needs PSQT + material footprint)",
        values.len(),
        LAYOUT.mobility_open_offset
    );

    // Lanes 2-7 exist to mirror the SIMD accumulator's layout; the f64 path never
    // reads them, so only MG and EG get written here.
    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        piece_counts[pt] = f64::from(board.role_bb[pt].popcount());

        let mat_mg = values[LAYOUT.material_offset + pt];
        let mat_eg = values[LAYOUT.material_offset + 6 + pt];

        let mut bb_w = board.pieces(piece, Color::White);
        let count_w = bb_w.popcount() as f64;
        lane_vals[0] += count_w * mat_mg;
        lane_vals[1] += count_w * mat_eg;

        while bb_w.is_not_empty() {
            let sq = bb_w.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq.flip_rank()));
            lane_vals[0] += values[pt * 64 + mirror_idx];
            lane_vals[1] += values[pt * 64 + 32 + mirror_idx];
        }

        let mut bb_b = board.pieces(piece, Color::Black);
        let count_b = bb_b.popcount() as f64;
        lane_vals[0] -= count_b * mat_mg;
        lane_vals[1] -= count_b * mat_eg;

        while bb_b.is_not_empty() {
            let sq = bb_b.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq));
            lane_vals[0] -= values[pt * 64 + mirror_idx];
            lane_vals[1] -= values[pt * 64 + 32 + mirror_idx];
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{env, hint::black_box, ops::Range};

    use super::*;
    use crate::{
        core::{board::Position, defs::TOTAL_PHASE},
        engine::{
            combiner::Accumulators,
            eval::evaluate,
            eval_params::{BLOCKS, LAYOUT, PHASE, collect_parameters, default_values},
        },
        tools::dataset::{FeatureRecord, SoulEntry, accumulate_record_grad, eval_record, eval_record_full},
    };

    const FENS: &[&str] = &[
        // White-to-move positions
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        // Black-to-move positions
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
        // Bishop pair imbalance
        "r1bqkbnr/1pp2ppp/p1p5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        // Rook open-file imbalance
        "2r3k1/pp3ppp/4p3/8/8/8/PPP2PPP/3R2K1 w - - 0 1",
        // Passed pawn
        "8/2k5/8/3K4/2P5/8/8/8 w - - 0 1",
        "8/4P3/6k1/4K3/8/8/8/8 w - - 0 1",
        // Passer far from the enemy king
        "8/8/P7/8/2K5/8/8/7k w - - 0 1",
        // Doubled pawn
        "4k3/pp4pp/8/8/8/2P5/2P2PPP/4K3 w - - 0 1",
        // Isolated pawn
        "4k3/p1p1p1p1/7p/8/3P4/8/PP4PP/4K3 w - - 0 1",
        // Phalanx
        "4k3/p5p1/2p3pp/8/3PP3/8/P5PP/4K3 w - - 0 1",
        // Defended pawn
        "4k3/2p3pp/1p3p2/8/2P2P2/1P4P1/7P/4K3 w - - 0 1",
        // Backward pawn
        "8/3p2k1/2p5/4p3/2P1P3/3P4/K7/8 w - - 0 1",
        // Minor behind pawn
        "4k3/5p2/5n2/8/4P3/4N3/8/4K3 w - - 0 1",
        // King under attack, where the danger curve is nonzero
        "r4rk1/pp3ppp/3p4/2pb4/3n2nq/N1NP4/PPPB1PPP/R2Q1RK1 w - - 0 1",
    ];

    const TARGET: f64 = 0.5;
    const K: f64 = 0.005;

    fn term_for(slot: usize) -> &'static str { BLOCKS.iter().rev().find(|b| slot >= b.offset).map_or("out of range", |b| b.name) }

    fn assert_oracle_matches(context: &str, values: &[f64]) {
        for fen in FENS {
            let pos = Position::from_fen(fen);

            let mut linear_grad = vec![0.0f64; values.len()];
            let linear_eval = eval_linear_grad(&pos, values, TARGET, K, &mut linear_grad);

            let mut dual_grad = vec![0.0f64; values.len()];
            let dual_eval = eval_dual_fused(&pos, values, TARGET, K, &mut dual_grad);

            assert!(
                (linear_eval - dual_eval).abs() < 1e-4,
                "[{context}] eval mismatch on '{fen}': linear={linear_eval} dual={dual_eval}",
            );

            for (i, (&lg, &dg)) in linear_grad.iter().zip(dual_grad.iter()).enumerate() {
                assert!(
                    (lg - dg).abs() < 1e-3,
                    "[{context}] {} scatter drift at slot {i} on '{fen}': linear={lg} dual={dg}",
                    term_for(i),
                );
            }
        }
    }

    fn with_phase(mut values: Vec<f64>) -> Vec<f64> {
        for (pt, &w) in PHASE.iter().enumerate() {
            values[LAYOUT.phase_offset + pt] = f64::from(w);
        }
        values
    }

    fn full_values() -> Vec<f64> {
        let mut values = vec![0.0f64; LAYOUT.total];
        for (n, v) in values.iter_mut().enumerate() {
            *v = (n % 17) as f64 - 8.0;
        }
        with_phase(values)
    }

    fn values_in_range(range: Range<usize>) -> Vec<f64> {
        let mut values = vec![0.0f64; LAYOUT.total];
        for i in range {
            values[i] = (i % 17) as f64 - 8.0;
        }
        with_phase(values)
    }

    fn danger_buckets(danger_us: f64, danger_them: f64) -> Accumulators<f64> {
        Accumulators::<f64> {
            mg_eg: 0.0,
            mobility: 0.0,
            bonus_mg: 0.0,
            bonus_eg: 0.0,
            safety_us: 0.0,
            safety_them: 0.0,
            danger_us,
            danger_them,
            xray: 0.0,
        }
    }

    const TEST_CURVE: f64 = 32.0;

    fn curvature(c: f64) -> CombinerParams<f64> { CombinerParams { king_danger: c } }

    #[test]
    fn test_king_danger_slope_oracle() {
        let phase = f64::from(TOTAL_PHASE);
        let shipped = curvature(TEST_CURVE);
        let h = 32.0;

        for p in [0.0, 64.0, 150.0, 300.0, 465.0] {
            let mut grads = vec![0.0f64; LAYOUT.total];
            let analytic = LinearCombiner::backward(&danger_buckets(p, 0.0), phase, &shipped, 1.0, &mut grads)
                .king_safety
                .danger_us;

            let (hi, lo) = (danger_buckets(p + h, 0.0), danger_buckets((p - h).max(0.0), 0.0));
            let rise = LinearCombiner::forward(&hi, phase, &shipped) - LinearCombiner::forward(&lo, phase, &shipped);
            let measured = rise / (hi.danger_us - lo.danger_us);
            assert!((analytic - measured).abs() < 0.05, "danger slope at {p}: analytic {analytic}, finite difference {measured}");
        }
    }

    #[test]
    fn test_king_danger_curvature_oracle() {
        let phase = f64::from(TOTAL_PHASE);
        let span = 64.0;

        for (us, them) in [(0.0, 0.0), (150.0, 0.0), (0.0, 300.0), (465.0, 150.0)] {
            let buckets = danger_buckets(us, them);
            let mut grads = vec![0.0f64; LAYOUT.total];

            LinearCombiner::backward(&buckets, phase, &curvature(TEST_CURVE), 1.0, &mut grads);
            let analytic = grads[LAYOUT.king_danger_offset];

            let hi = LinearCombiner::forward(&buckets, phase, &curvature(span));
            let lo = LinearCombiner::forward(&buckets, phase, &curvature(0.0));
            let measured = (hi - lo) / span;
            assert!(
                (analytic - measured).abs() < 0.05,
                "curvature grad at ({us}, {them}): analytic {analytic}, finite difference {measured}",
            );
        }
    }

    const ROUND_SITES: f64 = 9.0;

    #[test]
    fn test_score_is_homogeneous_in_its_weights_oracle() {
        let base = default_values(&collect_parameters());
        let (lo, hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);
        let mut asserted = 0;

        for curve in [0.0f64, 32.0, 96.0] {
            for f in [2.0f64, 4.0] {
                let mut one = base.clone();
                one[LAYOUT.king_danger_offset] = curve;

                let mut scaled = one.clone();
                for i in (0..scaled.len()).filter(|i| !(lo..hi).contains(i)) {
                    scaled[i] *= if i == LAYOUT.king_danger_offset { f.recip() } else { f };
                }

                for fen in FENS {
                    let pos = Position::from_fen(fen);
                    let plain = eval_f64(&pos, &one);

                    if plain.abs() < 40.0 {
                        continue;
                    }

                    let (want, got) = (f * plain, eval_f64(&pos, &scaled));
                    asserted += 1;

                    let bound = ROUND_SITES * (1.0 + f);
                    assert!((got - want).abs() <= bound, "curve {curve}, scale {f} on '{fen}': got {got}, want {want}");
                }
            }
        }
        assert!(asserted >= 12, "insufficient positions with large enough eval to test: {asserted}");
    }

    #[test]
    fn test_i32_matches_f64_oracle() {
        let values = default_values(&collect_parameters());
        for fen in FENS {
            let pos = Position::from_fen(fen);
            let engine = f64::from(evaluate(&pos, &pos.get_initial_accumulator()));
            let tuner = eval_f64(&pos, &values);
            assert!((engine - tuner).abs() < 1e-9, "engine {engine} vs tuner {tuner} on '{fen}'");
        }
    }

    #[test]
    fn test_shipped_values_oracle() {
        let values = default_values(&collect_parameters());
        assert_oracle_matches("shipped defaults", &values);
    }

    #[test]
    fn test_linear_oracle_verification() { assert_oracle_matches("pipeline", &full_values()); }

    #[test]
    fn test_mobility_term_oracle() {
        assert_oracle_matches("MobilityTerm alone", &values_in_range(LAYOUT.mobility_open_offset..LAYOUT.king_safety_offset));
    }

    #[test]
    fn test_king_safety_term_oracle() {
        assert_oracle_matches("KingSafetyTerm alone", &values_in_range(LAYOUT.king_safety_offset..LAYOUT.xray_offset));
    }

    #[test]
    fn test_xray_term_oracle() {
        assert_oracle_matches("XrayTerm alone", &values_in_range(LAYOUT.xray_offset..LAYOUT.xray_offset + LAYOUT.xray_len));
    }

    macro_rules! bonus_term_oracles {
        ( [] $( $block:ident = $kind:ident ( $($spec:tt)* ) ; )* ) => {
            #[test]
            fn test_bonus_terms_oracle() {
                $( bonus_term_oracles!(@one $kind $block, $($spec)*); )*
            }
        };

        (@one scalar $block:ident, $term:ident, $field:ident, $mg:ident, $eg:ident) => {
            paste::paste! {
                assert_oracle_matches(
                    concat!(stringify!($term), " alone"),
                    &values_in_range(LAYOUT.[<$block _offset>]..LAYOUT.[<$block _offset>] + 2),
                )
            }
        };

        (@one array $block:ident, $term:ident, $field:ident, $mg:ident, $eg:ident, $n:literal) => {
            paste::paste! {
                assert_oracle_matches(
                    concat!(stringify!($term), " alone"),
                    &values_in_range(LAYOUT.[<$block _mg_offset>]..LAYOUT.[<$block _eg_offset>] + $n),
                )
            }
        };
    }

    crate::bonus_terms!(bonus_term_oracles);

    #[test]
    fn test_encoded_path_oracle() {
        let values = full_values();
        let n_params = values.len();

        for fen in FENS {
            let pos = Position::from_fen(fen);
            let entry = SoulEntry::from_board(&pos, TARGET, Some(20));

            let rt_pos = Position::from_fen(&entry.to_fen());
            let orig_score = eval_f64(&pos, &values);
            let rt_score = eval_f64(&rt_pos, &values);
            assert!(
                (orig_score - rt_score).abs() < 1e-4,
                "Round-trip score mismatch on '{fen}': orig={orig_score} reconstructed={rt_score}",
            );

            let record = FeatureRecord::from_entry(&entry);
            let record_eval = eval_record_full(&record, &values);

            let board_score = eval_f64(&pos, &values);
            let entry_score = record_eval.score;
            assert!(
                (board_score - entry_score).abs() < 1e-4,
                "Score mismatch on '{fen}': board={board_score} encoded={entry_score}",
            );

            let mut dual_grads = vec![0.0f64; n_params];
            let dual_loss = eval_dual_fused(&pos, &values, TARGET, K, &mut dual_grads);

            let sig = sigmoid(entry_score, K);
            let err = sig - TARGET;
            let outer = 2.0 * err * sig * (1.0 - sig) * K;

            let mut encoded_grads = vec![0.0f64; n_params];
            accumulate_record_grad(&record, &record_eval, outer, &mut encoded_grads);

            let enc_loss = err * err;
            assert!((dual_loss - enc_loss).abs() < 1e-4, "Loss mismatch on '{fen}': dual={dual_loss} encoded={enc_loss}");

            for (i, (&dual, &encoded)) in dual_grads.iter().zip(encoded_grads.iter()).enumerate() {
                let diff = (dual - encoded).abs();
                assert!(
                    diff < 1e-3,
                    "[encoded] {} scatter drift at slot {i} on '{fen}': dual={dual} encoded={encoded} diff={diff}",
                    term_for(i),
                );
            }
        }
    }

    #[test]
    fn test_encoded_block_coverage_oracle() {
        let base = default_values(&collect_parameters());
        let records: Vec<FeatureRecord> = FENS
            .iter()
            .map(|fen| FeatureRecord::from_entry(&SoulEntry::from_board(&Position::from_fen(fen), TARGET, Some(20))))
            .collect();

        for block in BLOCKS {
            let mut bumped = base.clone();
            for slot in &mut bumped[block.offset..block.offset + block.len] {
                *slot += 100.0;
            }

            let moved = records.iter().any(|r| (eval_record(r, &base) - eval_record(r, &bumped)).abs() > 1e-9);
            assert!(
                moved,
                "block `{}` never changes the cached evaluation: unpacked in from_entry or uncovered by test FENs",
                block.name
            );
        }
    }

    #[test]
    fn test_encoded_matches_board_at_defaults_oracle() {
        let values = default_values(&collect_parameters());
        for fen in FENS {
            let pos = Position::from_fen(fen);
            let record = FeatureRecord::from_entry(&SoulEntry::from_board(&pos, TARGET, Some(20)));
            let board = eval_f64(&pos, &values);
            let cached = eval_record(&record, &values);
            assert!((board - cached).abs() < 1e-9, "board {board} vs cached {cached} on '{fen}'");
        }
    }

    #[test]
    #[ignore]
    fn measure_gradient_ops() {
        const BENCH: &str = include_str!("../../data/bench.fens");

        let iters: usize = env::var("SOUL_OPS_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(2_000);
        let mode = env::var("SOUL_OPS_MODE").unwrap_or_default();
        let values = default_values(&collect_parameters());
        let boards: Vec<Position> = BENCH.lines().filter(|line| !line.trim().is_empty()).map(Position::from_fen).collect();
        let records: Vec<FeatureRecord> = boards
            .iter()
            .map(|board| FeatureRecord::from_entry(&SoulEntry::from_board(board, TARGET, Some(20))))
            .collect();

        let mut grads = vec![0.0f64; values.len()];

        fn drive(iters: usize, n: usize, mut body: impl FnMut(usize) -> f64) -> f64 {
            let mut sink = 0.0;
            for _ in 0..iters {
                for i in 0..n {
                    sink += body(i);
                }
            }
            sink
        }

        let n = boards.len();

        let sink = match mode.as_str() {
            "grad" => {
                drive(iters, n, |i| eval_linear_grad(black_box(&boards[i]), black_box(&values), TARGET, K, black_box(&mut grads)))
            },
            "record" => drive(iters, n, |i| eval_record(black_box(&records[i]), black_box(&values))),
            "recordgrad" => drive(iters, n, |i| {
                let record = black_box(&records[i]);
                let eval = eval_record_full(record, black_box(&values));
                accumulate_record_grad(record, &eval, 1.0, black_box(&mut grads));
                eval.score
            }),
            "loss" => drive(iters, n, |i| {
                let err = sigmoid(black_box(eval_f64(black_box(&boards[i]), black_box(&values))), K) - TARGET;
                err * err
            }),
            _ => drive(iters, n, |i| eval_f64(black_box(&boards[i]), black_box(&values))),
        };

        black_box(sink);
        println!("positions {}", iters * boards.len());
    }
}
