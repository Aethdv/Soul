//! Forward-mode AD evaluation using dual numbers.
//!
//! Instead of building a tape and running `backward()`, we carry `DUAL_N`
//! partial derivatives alongside each value. PSQT gradients are recovered
//! by multiplying the 8 accumulator-lane gradients by each piece's ±1
//! contribution.

use soul::{
    core::{
        board::Position as Board,
        defs::{Color, PieceType},
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
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr) ),* $(,)?) => {
        paste::paste! {
            impl DualEvalResult {
                pub fn scatter_dynamic(&self, outer_deriv: f64, param_grads: &mut [f64]) {
                    let mut slot = 2; // MG=0, EG=1
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

soul::define_tunables!(impl_scatter);

/// Consumed by `scatter_dynamic` once the outer loss derivative is known.
pub struct DualEvalResult {
    /// One slot per dual-tracked input (`DUAL_SLOTS`), zero-padded to `DUAL_N`.
    pub grad: [f32; DUAL_N],
}

/// Eval + gradient scatter via forward-mode AD.
///
/// Correctness oracle over `eval_linear_grad`: comparing loss and gradients
/// across the same inputs verifies the hand-derived formulas.
///
/// Returns the squared error for loss tracking.
#[inline]
pub fn eval_dual_fused(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    // PSQT + material in plain f64 (no dual, just lane sums and piece counts)
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

    let phase = phase_dual.math_clamp(DualNode::constant(0.0), DualNode::constant(24.0)).trunc();

    let mut dual_acc = DualVec8::zero();
    dual_acc.0[0] = DualNode::seed(lane_vals[0], 0);
    dual_acc.0[1] = DualNode::seed(lane_vals[1], 1);

    for (dual, &val) in dual_acc.0[2..8].iter_mut().zip(&lane_vals[2..8]) {
        *dual = DualNode::constant(val);
    }

    let params = EvalParams::<DualNode>::load_tunable(values);
    let features = SharedFeatures::compute(board);

    // Forward pass
    let result = evaluate_generic::<DualNode>(board, &dual_acc, phase, &params, Some(&features));
    let score = result.val;

    // Sigmoid + loss derivative
    let sig = 1.0 / (1.0 + (-k * score).clamp(-700.0, 700.0).exp());
    let err = sig - target;
    let outer_deriv = 2.0 * err * sig * (1.0 - sig) * k;

    // Non-PSQT gradients
    let dummy = DualEvalResult { grad: result.grad };
    dummy.scatter_dynamic(outer_deriv, param_grads);

    // Re-iterate piece bitboards for PSQT gradients
    let d_mg = outer_deriv * f64::from(result.grad[0]);
    let d_eg = outer_deriv * f64::from(result.grad[1]);

    debug_assert!(
        param_grads.len() >= LAYOUT.mobility_open_offset,
        "param_grads too small: {} < {} (material+PSQT footprint)",
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
    err * err
}

/// Because every parameter is linear (`param · feature`), the gradient w.r.t.
/// each one is just its feature coefficient on the board. Computing these
/// directly is cheaper than the dual path:
///
/// - Linear: one f64 write per slot.
/// - Dual: `DUAL_N` f32 ops per arithmetic op, where `DUAL_N = (DUAL_SLOTS + 7) & !7`
///   and `DUAL_SLOTS = 2 + ∑slot_width(tunables)`.
///
/// Returns the squared error for loss tracking.
#[inline]
pub fn eval_linear_grad(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    // ── PSQT + Material accumulator
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut lane_vals, &mut piece_counts);

    let phase_raw = compute_phase(&piece_counts, values);

    // dJ/dPhaseWeight is deliberately omitted: phase weights must stay at their
    // engineered values for the MG/EG interpolation to be meaningful.
    let phase = phase_raw.clamp(0.0, 24.0).trunc();

    let params = EvalParams::<f64>::load_tunable(values);
    let features = SharedFeatures::compute(board);
    let mut acc = <f64 as EvalMath>::Vec8::zero();
    acc.0 = lane_vals;

    let buckets = fill_accumulators::<f64>(&acc, phase, &features, &params);
    let combiner = CombinerParams::from_eval(&params);
    let white_score = LinearCombiner::forward(&buckets, phase, &combiner);
    let stm_sign: f64 = if board.stm == Color::White { 1.0 } else { -1.0 };
    let score = white_score * stm_sign;

    // ── Sigmoid + loss derivative
    // `d` folds the STM sign into the outer derivative once,
    // so every downstream scatter can stay STM-agnostic.
    let sig = 1.0 / (1.0 + (-k * score).clamp(-700.0, 700.0).exp());
    let err = sig - target;
    let outer = 2.0 * err * sig * (1.0 - sig) * k;
    let d = outer * stm_sign;

    debug_assert!(
        param_grads.len() >= LAYOUT.mobility_open_offset,
        "param_grads too small: {} < {} (material+PSQT footprint)",
        param_grads.len(),
        LAYOUT.mobility_open_offset,
    );

    // Combiner owns every upstream derivative. PSQT / material scatter
    // (out-of-band, accumulator-level) pulls `mg_eg`; term scatter reads
    // the rest via `scatter_all_terms`.
    let upstreams = LinearCombiner::backward(&buckets, phase, &combiner, d, param_grads);

    // ── PSQT + material (accumulator-level, not a LinearTerm)
    // One board sweep writes both gradients for every active piece.
    let d_mg = upstreams.mg_eg.d_mg;
    let d_eg = upstreams.mg_eg.d_eg;

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
    // Adding a new term = one line in `register_terms!`
    scatter_all_terms(&features, &upstreams, param_grads);
    err * err
}

/// Substitutes f64 for i32 in the engine eval path.
#[inline(always)]
pub fn eval_f64(board: &Board, values: &[f64]) -> f64 {
    eval_f64_with_acc(board, values).0
}

/// Returns the eval score and the lane-sum accumulator + piece counts,
/// so `eval_linear_grad` can reuse them instead of re-iterating the board.
pub fn eval_f64_with_acc(board: &Board, values: &[f64]) -> (f64, [f64; 8], [f64; 6]) {
    let mut trace_acc = <f64 as EvalMath>::Vec8::zero();
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut trace_acc.0, &mut piece_counts);

    let phase_raw = compute_phase(&piece_counts, values);
    let phase = phase_raw.math_clamp(0.0, 24.0).trunc();

    // Same generator the DualNode path uses: no per-term hand literal to drift.
    let params = EvalParams::<f64>::load_tunable(values);
    let features = SharedFeatures::compute(board);

    (evaluate_generic::<f64>(board, &trace_acc, phase, &params, Some(&features)), trace_acc.0, piece_counts)
}

#[inline(always)]
fn compute_phase(piece_counts: &[f64; 6], values: &[f64]) -> f64 {
    let mut phase_raw = 0.0;

    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = LAYOUT.phase_offset + pt;

        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }
    phase_raw
}

#[inline(always)]
fn accumulate_lane_vals(board: &Board, values: &[f64], lane_vals: &mut [f64], piece_counts: &mut [f64; 6]) {
    debug_assert!(
        values.len() >= LAYOUT.mobility_open_offset,
        "values too short: {} < {} (needs PSQT + material footprint)",
        values.len(),
        LAYOUT.mobility_open_offset
    );

    // Only lanes [0] (MG) and [1] (EG) carry signal. Lanes 2–7 mirror the
    // SIMD accumulator layout, but the f64 evaluator path never reads them.
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
    use std::ops::Range;

    use soul::{
        core::{board::Position, defs::TOTAL_PHASE},
        engine::{
            combiner::Accumulators,
            eval::evaluate,
            eval_params::{BLOCKS, LAYOUT, PHASE, collect_parameters},
        },
        tools::dataset::{FeatureRecord, SoulEntry, accumulate_record_grad, eval_record, eval_record_full},
    };

    use super::*;
    use crate::evaltune::training::sigmoid;

    // Each must round-trip cleanly through `SoulEntry`.
    const FENS: &[&str] = &[
        // White-to-move
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        // Black-to-move
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4",
        // Bishop pair imbalance
        "r1bqkbnr/1pp2ppp/p1p5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 5",
        // Rook open-file imbalance
        "2r3k1/pp3ppp/4p3/8/8/8/PPP2PPP/3R2K1 w - - 0 1",
        // Passed pawns: mid-board and near-promotion
        "8/2k5/8/3K4/2P5/8/8/8 w - - 0 1",
        "8/4P3/6k1/4K3/8/8/8/8 w - - 0 1",
        // Passer far from the enemy king
        "8/8/P7/8/2K5/8/8/7k w - - 0 1",
        // Doubled pawns
        "4k3/pp4pp/8/8/8/2P5/2P2PPP/4K3 w - - 0 1",
        // Isolated pawns
        "4k3/p1p1p1p1/7p/8/3P4/8/PP4PP/4K3 w - - 0 1",
        // Phalanx
        "4k3/p5p1/2p3pp/8/3PP3/8/P5PP/4K3 w - - 0 1",
        // Defended pawns
        "4k3/2p3pp/1p3p2/8/2P2P2/1P4P1/7P/4K3 w - - 0 1",
        // Backward pawn
        "8/3p2k1/2p5/4p3/2P1P3/3P4/K7/8 w - - 0 1",
        // Minor behind pawn
        "4k3/5p2/5n2/8/4P3/4N3/8/4K3 w - - 0 1",
    ];

    const TARGET: f64 = 0.5;
    const K: f64 = 0.005;

    // Names the owning block for drift-localized oracle failures.
    fn term_for(slot: usize) -> &'static str {
        BLOCKS.iter().rev().find(|b| slot >= b.offset).map_or("out of range", |b| b.name)
    }

    // Compares eval vs gradients across both paths, identifies drift by term name.
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

    /// The phase block keeps its shipped weights whatever else a test vector holds.
    /// Junk or zeros there make `phase_raw` negative on any real position, and a
    /// phase clamped to 0 zeroes `d_mg` and tapers the king-safety block away, so
    /// half of every gradient gets compared as 0 against 0.
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

    // Isolates one LinearTerm: nonzero values in `range`, the phase block, zero elsewhere.
    fn values_in_range(range: Range<usize>) -> Vec<f64> {
        let mut values = vec![0.0f64; LAYOUT.total];

        for i in range {
            values[i] = (i % 17) as f64 - 8.0;
        }
        with_phase(values)
    }

    /// Pressure on each king, everything else zeroed, so the two tests below read
    /// the curve alone.
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

    /// Fixed rather than read from `KING_DANGER`: the derivatives have to hold at
    /// any curvature, and a shipped zero leaves the slope test verifying no curve.
    const TEST_CURVE: f64 = 32.0;

    fn curvature(c: f64) -> CombinerParams<f64> {
        CombinerParams { king_danger: c }
    }

    /// No fen in the set has a king pressured enough to check this: `weak` peaks
    /// at 2, which moves the gradient by a few percent of a value already under
    /// the comparison tolerance. So difference the slope directly.
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

            assert!((analytic - measured).abs() < 0.05, "danger slope at {p}: analytic {analytic}, finite difference {measured}",);
        }
    }

    /// The curvature is the combiner's own weight, so no term's `scatter` touches
    /// it and `register_terms!` cannot miss it. Differenced against the forward for
    /// the same reason as the slope: the fen set has no besieged king in it.
    #[test]
    fn test_king_danger_curvature_oracle() {
        let phase = f64::from(TOTAL_PHASE);
        // The curve is linear in its curvature, so the gradient at any point equals
        // the secant over any interval. Wide, so the trunc inside it averages out.
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

    /// Truncation sites one evaluation passes through, each losing under a unit:
    /// the PSQT lanes, the two mobility tapers, `(w_atk·weak)/10` per side, the
    /// bonus and safety tapers, and the curvature per side when it is live.
    const ROUND_SITES: f64 = 9.0;

    /// The property the tuner's gauge trades on: scale the weights and K absorbs it.
    ///
    /// Asserted only where the eval has some size to it, since the truncation sites
    /// round a near-zero score to noise. A site's cost does not grow with the scale,
    /// but `f·eval(θ)` carries `f` times whatever the base evaluation lost, so the
    /// tolerance has to. A curvature scaled with the rest instead costs several
    /// times that, and only a nonzero one can be scaled wrongly, so the sweep
    /// carries some.
    #[test]
    fn test_score_is_homogeneous_in_its_weights_oracle() {
        let base: Vec<f64> = collect_parameters().iter().map(|t| t.value).collect();
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

                    assert!((got - want).abs() <= bound, "curve {curve}, ×{f} on '{fen}': {got} against {want}");
                }
            }
        }

        assert!(asserted >= 12, "no position carried enough eval to test: {asserted}");
    }

    /// The engine plays the `i32` monomorphization and the tuner fits the `f64`
    /// one, which is only sound if they are the same function. They diverge
    /// wherever one truncates and the other does not, and nothing else here
    /// compares across the two.
    #[test]
    fn test_i32_matches_f64_oracle() {
        let values: Vec<f64> = collect_parameters().iter().map(|t| t.value).collect();

        for fen in FENS {
            let pos = Position::from_fen(fen);
            let engine = f64::from(evaluate(&pos, &pos.get_initial_accumulator()));
            let tuner = eval_f64(&pos, &values);

            assert!((engine - tuner).abs() < 1e-9, "engine {engine} vs tuner {tuner} on '{fen}'");
        }
    }

    /// Every other test here runs at junk magnitudes, one to two orders of
    /// magnitude under the shipped weights. Harmless for a linear term, and no
    /// basis at all for anything whose shape depends on scale.
    #[test]
    fn test_shipped_values_oracle() {
        let values: Vec<f64> = collect_parameters().iter().map(|t| t.value).collect();

        assert_oracle_matches("shipped defaults", &values);
    }

    #[test]
    fn test_linear_oracle_verification() {
        assert_oracle_matches("pipeline", &full_values());
    }

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

    #[test]
    fn test_bishop_pair_term_oracle() {
        assert_oracle_matches(
            "BishopPairTerm alone",
            &values_in_range(LAYOUT.bishop_pair_offset..LAYOUT.bishop_pair_offset + LAYOUT.bishop_pair_len),
        );
    }

    #[test]
    fn test_rook_open_term_oracle() {
        assert_oracle_matches(
            "RookOpenTerm alone",
            &values_in_range(LAYOUT.rook_open_offset..LAYOUT.rook_open_offset + LAYOUT.rook_open_len),
        );
    }

    #[test]
    fn test_passed_pawn_term_oracle() {
        assert_oracle_matches(
            "PassedPawnTerm alone",
            &values_in_range(LAYOUT.passed_pawn_mg_offset..LAYOUT.passed_pawn_eg_offset + LAYOUT.passed_pawn_eg_len),
        );
    }

    #[test]
    fn test_enemy_king_dist_term_oracle() {
        assert_oracle_matches(
            "EnemyKingDistTerm alone",
            &values_in_range(LAYOUT.enemy_king_dist_mg_offset..LAYOUT.enemy_king_dist_eg_offset + LAYOUT.enemy_king_dist_eg_len),
        );
    }

    #[test]
    fn test_doubled_pawn_term_oracle() {
        assert_oracle_matches(
            "DoubledPawnTerm alone",
            &values_in_range(LAYOUT.doubled_pawn_offset..LAYOUT.doubled_pawn_offset + LAYOUT.doubled_pawn_len),
        );
    }

    #[test]
    fn test_isolated_pawn_term_oracle() {
        assert_oracle_matches(
            "IsolatedPawnTerm alone",
            &values_in_range(LAYOUT.isolated_pawn_offset..LAYOUT.isolated_pawn_offset + LAYOUT.isolated_pawn_len),
        );
    }

    #[test]
    fn test_phalanx_term_oracle() {
        assert_oracle_matches(
            "PhalanxTerm alone",
            &values_in_range(LAYOUT.phalanx_mg_offset..LAYOUT.phalanx_eg_offset + LAYOUT.phalanx_eg_len),
        );
    }

    #[test]
    fn test_defended_pawn_term_oracle() {
        assert_oracle_matches(
            "DefendedPawnTerm alone",
            &values_in_range(LAYOUT.defended_pawn_mg_offset..LAYOUT.defended_pawn_eg_offset + LAYOUT.defended_pawn_eg_len),
        );
    }

    #[test]
    fn test_backward_pawn_term_oracle() {
        assert_oracle_matches(
            "BackwardPawnTerm alone",
            &values_in_range(LAYOUT.backward_pawn_offset..LAYOUT.backward_pawn_offset + LAYOUT.backward_pawn_len),
        );
    }

    #[test]
    fn test_tempo_term_oracle() {
        assert_oracle_matches("TempoTerm alone", &values_in_range(LAYOUT.tempo_offset..LAYOUT.tempo_offset + LAYOUT.tempo_len));
    }

    #[test]
    fn test_minor_behind_pawn_term_oracle() {
        assert_oracle_matches(
            "MinorBehindPawnTerm alone",
            &values_in_range(LAYOUT.minor_behind_pawn_offset..LAYOUT.minor_behind_pawn_offset + LAYOUT.minor_behind_pawn_len),
        );
    }

    /// `accumulate_record_grad` shares math with the board-based gradient paths.
    /// A drift here corrupts every `run_encoded` session.
    #[test]
    fn test_encoded_path_oracle() {
        let values = full_values();
        let n_params = values.len();

        for fen in FENS {
            let pos = Position::from_fen(fen);
            let entry = SoulEntry::from_board(&pos, TARGET, None, Some(20));

            // Position → SoulEntry → to_fen → Position.
            // If this fails, to_fen() is corrupting the board.
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

    /// A term reaches the cached eval through `from_entry`'s packing, and nothing
    /// checks that: a field left unpacked stays zero, the score never moves, and
    /// the drift asserts above stay quiet because both paths agree on it.
    #[test]
    fn test_encoded_block_coverage_oracle() {
        let base: Vec<f64> = collect_parameters().iter().map(|t| t.value).collect();

        let records: Vec<FeatureRecord> = FENS
            .iter()
            .map(|fen| FeatureRecord::from_entry(&SoulEntry::from_board(&Position::from_fen(fen), TARGET, None, Some(20))))
            .collect();

        for block in BLOCKS {
            let mut bumped = base.clone();

            for slot in &mut bumped[block.offset..block.offset + block.len] {
                *slot += 100.0;
            }

            let moved = records.iter().any(|r| (eval_record(r, &base) - eval_record(r, &bumped)).abs() > 1e-9);

            assert!(moved, "block `{}` never moves the cached eval: unpacked in from_entry, or no FEN reaches it", block.name);
        }
    }

    /// Runs at the shipped defaults, not `full_values`: those junk parameters put
    /// the phase on a boundary where every truncation is a no-op, so every test
    /// built on them agrees with the board path wherever it rounds.
    #[test]
    fn test_encoded_matches_board_at_defaults_oracle() {
        let values: Vec<f64> = collect_parameters().iter().map(|t| t.value).collect();

        for fen in FENS {
            let pos = Position::from_fen(fen);
            let record = FeatureRecord::from_entry(&SoulEntry::from_board(&pos, TARGET, None, Some(20)));

            let board = eval_f64(&pos, &values);
            let cached = eval_record(&record, &values);

            assert!((board - cached).abs() < 1e-9, "board {board} vs cached {cached} on '{fen}'");
        }
    }
}
