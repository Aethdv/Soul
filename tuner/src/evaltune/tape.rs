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
        combiner::{Combiner, LinearCombiner},
        eval::{EvalParams, SharedFeatures, evaluate_generic, scatter_all_terms},
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
        let phase_idx = psqt::LAYOUT.phase_offset + pt;

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
        param_grads.len() >= psqt::LAYOUT.mobility_open_offset,
        "param_grads too small: {} < {} (material+PSQT footprint)",
        param_grads.len(),
        psqt::LAYOUT.mobility_open_offset,
    );

    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        let mat_mg_idx = psqt::LAYOUT.material_offset + pt;
        let mat_eg_idx = psqt::LAYOUT.material_offset + 6 + pt;

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
    let score = eval_f64(board, values);

    // ── Sigmoid + loss derivative
    // `d` folds the STM sign into the outer derivative once,
    // so every downstream scatter can stay STM-agnostic.
    let sig = 1.0 / (1.0 + (-k * score).clamp(-700.0, 700.0).exp());
    let err = sig - target;
    let outer = 2.0 * err * sig * (1.0 - sig) * k;
    let stm_sign: f64 = if board.stm == Color::White { 1.0 } else { -1.0 };
    let d = outer * stm_sign;

    let features = SharedFeatures::compute(board);

    debug_assert!(
        param_grads.len() >= psqt::LAYOUT.mobility_open_offset,
        "param_grads too small: {} < {} (material+PSQT footprint)",
        param_grads.len(),
        psqt::LAYOUT.mobility_open_offset,
    );

    // Combiner owns every upstream derivative. PSQT / material scatter
    // (out-of-band, accumulator-level) pulls `mg_eg`; term scatter reads
    // the rest via `scatter_all_terms`.
    let upstreams = LinearCombiner::backward(phase, d, param_grads);

    // ── PSQT + material (accumulator-level, not a LinearTerm)
    // One board sweep writes both gradients for every active piece.
    let d_mg = upstreams.mg_eg.d_mg;
    let d_eg = upstreams.mg_eg.d_eg;

    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        let mat_mg_idx = psqt::LAYOUT.material_offset + pt;
        let mat_eg_idx = psqt::LAYOUT.material_offset + 6 + pt;

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
        let phase_idx = psqt::LAYOUT.phase_offset + pt;

        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }
    phase_raw
}

#[inline(always)]
fn accumulate_lane_vals(board: &Board, values: &[f64], lane_vals: &mut [f64], piece_counts: &mut [f64; 6]) {
    debug_assert!(
        values.len() >= psqt::LAYOUT.mobility_open_offset,
        "values too short: {} < {} (needs PSQT + material footprint)",
        values.len(),
        psqt::LAYOUT.mobility_open_offset
    );

    // Only lanes [0] (MG) and [1] (EG) carry signal. Lanes 2–7 mirror the
    // SIMD accumulator layout, but the f64 evaluator path never reads them.
    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        piece_counts[pt] = f64::from(board.role_bb[pt].popcount());

        let mat_mg = values[psqt::LAYOUT.material_offset + pt];
        let mat_eg = values[psqt::LAYOUT.material_offset + 6 + pt];

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
        core::{board::Position, psqt::LAYOUT},
        engine::eval_params::{BLOCKS, collect_parameters},
        tools::dataset::{FeatureRecord, SoulEntry, accumulate_record_grad, eval_record},
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

    // Names the owning eval layer for drift-localized oracle failures.
    fn term_for(slot: usize) -> &'static str {
        if slot < LAYOUT.mobility_open_offset {
            "PSQT/material (accumulator-level)"
        } else if slot < LAYOUT.king_safety_offset {
            "MobilityTerm"
        } else if slot < LAYOUT.attacker_offset {
            "KingSafetyTerm (shield/ortho/diag)"
        } else if slot < LAYOUT.xray_offset {
            "KingSafetyTerm (attackers)"
        } else if slot < LAYOUT.bishop_pair_offset {
            "XrayTerm"
        } else if slot < LAYOUT.rook_open_offset {
            "BishopPairTerm"
        } else if slot < LAYOUT.passed_pawn_mg_offset {
            "RookOpenTerm"
        } else if slot < LAYOUT.enemy_king_dist_mg_offset {
            "PassedPawnTerm"
        } else if slot < LAYOUT.doubled_pawn_offset {
            "EnemyKingDistTerm"
        } else if slot < LAYOUT.isolated_pawn_offset {
            "DoubledPawnTerm"
        } else if slot < LAYOUT.phalanx_mg_offset {
            "IsolatedPawnTerm"
        } else if slot < LAYOUT.defended_pawn_mg_offset {
            "PhalanxTerm"
        } else if slot < LAYOUT.backward_pawn_offset {
            "DefendedPawnTerm"
        } else if slot < LAYOUT.tempo_offset {
            "BackwardPawnTerm"
        } else if slot < LAYOUT.minor_behind_pawn_offset {
            "TempoTerm"
        } else {
            "MinorBehindPawnTerm"
        }
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

    fn full_values() -> Vec<f64> {
        let mut values = vec![0.0f64; LAYOUT.total];

        for (n, v) in values.iter_mut().enumerate() {
            *v = (n % 17) as f64 - 8.0;
        }
        values
    }

    // Isolates one LinearTerm: nonzero values in `range`, zero elsewhere.
    fn values_in_range(range: Range<usize>) -> Vec<f64> {
        let mut values = vec![0.0f64; LAYOUT.total];

        for i in range {
            values[i] = (i % 17) as f64 - 8.0;
        }
        values
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

            let board_score = eval_f64(&pos, &values);
            let entry_score = eval_record(&record, &values);
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
            accumulate_record_grad(&record, &values, outer, &mut encoded_grads);

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
