//! Forward-mode AD evaluation using dual numbers.
//!
//! Instead of building a tape and running backward(), we carry partial
//! derivatives for all 35 non-linear inputs alongside each value.
//! PSQT gradients are recovered by multiplying the 8 accumulator-lane
//! gradients by each piece's ±1 contribution.

use soul::{
    core::{
        board::Position as Board,
        defs::{Color, PieceType},
        psqt,
    },
    engine::autograd::{EnvVec8, EvalMath},
    weave::Vf64x4,
};

/// Active piece entry for PSQT gradient scatter.
#[derive(Copy, Clone, Default)]
pub struct ActivePiece {
    pub mg_idx:     u32,
    pub eg_idx:     u32,
    pub mat_mg_idx: u32,
    pub mat_eg_idx: u32,
    pub sign:       f32,
}

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
    ($( ($name:ident, $ty:ident, $offset:expr) ),* $(,)?) => {
        paste::paste! {
            impl DualEvalResult {
                pub fn scatter_dynamic(&self, outer_deriv: f64, param_grads: &mut [f64]) {
                    let mut slot = 2; // Mg=0, Eg=1
                    $( scatter::[<scatter_ $ty:lower>](&self.grad, &mut slot, outer_deriv, param_grads, $offset); )*
                }
            }
        }
    }
}

soul::define_tunables!(impl_scatter);

#[inline(always)]
fn accumulate_lane_vals(board: &Board, values: &[f64], lane_vals: &mut [f64], piece_counts: &mut [f64; 6]) {
    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        piece_counts[pt] = f64::from(board.role_bb[pt].popcount());

        let mat_mg = values
            .get(psqt::LAYOUT.material_offset + pt)
            .copied()
            .unwrap_or(0.0);
        let mat_eg = values
            .get(psqt::LAYOUT.material_offset + 6 + pt)
            .copied()
            .unwrap_or(0.0);

        // White pieces
        let mut bb_w = board.pieces(piece, Color::White);
        let count_w = bb_w.popcount() as f64;
        lane_vals[0] += count_w * mat_mg;
        lane_vals[1] += count_w * mat_eg;
        while bb_w.is_not_empty() {
            let sq = bb_w.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq.flip_rank()));

            lane_vals[0] += values.get(pt * 64 + mirror_idx).copied().unwrap_or(0.0);
            lane_vals[1] += values
                .get(pt * 64 + 32 + mirror_idx)
                .copied()
                .unwrap_or(0.0);
        }

        // Black pieces
        let mut bb_b = board.pieces(piece, Color::Black);
        let count_b = bb_b.popcount() as f64;
        lane_vals[0] -= count_b * mat_mg;
        lane_vals[1] -= count_b * mat_eg;
        while bb_b.is_not_empty() {
            let sq = bb_b.pop_lsb();
            let mirror_idx = psqt::mirror_sq(usize::from(sq));

            lane_vals[0] -= values.get(pt * 64 + mirror_idx).copied().unwrap_or(0.0);
            lane_vals[1] -= values
                .get(pt * 64 + 32 + mirror_idx)
                .copied()
                .unwrap_or(0.0);
        }
    }
}

/// Result of a forward-mode dual evaluation.
///
/// Captures the score and raw gradient array from the dual forward pass,
/// plus the piece locations needed to scatter PSQT gradients. Designed
/// as a snapshot that can be consumed later once the outer loss derivative
/// is known.
pub struct DualEvalResult {
    pub score:        f64,
    /// Raw partial derivatives from the dual pass (29 active slots in 32).
    pub grad:         [f32; soul::engine::autograd::dual::DUAL_N],
    /// Board pieces recorded during accumulation for PSQT scatter.
    pub active:       [ActivePiece; 32],
    pub active_count: usize,
}

impl DualEvalResult {
    /// Scatter gradients into `param_grads`, scaled by `outer_deriv`.
    pub fn scatter_grads(&self, outer_deriv: f64, param_grads: &mut [f64]) {
        self.scatter_dynamic(outer_deriv, param_grads);

        // PSQT gradients: d(output)/d(psqt_i) = d(output)/d(lane_j) * d(lane_j)/d(psqt_i)
        let d_mg = outer_deriv * f64::from(self.grad[0]); // Mg slot = 0
        let d_eg = outer_deriv * f64::from(self.grad[1]); // Eg slot = 1

        debug_assert!(
            param_grads.len() >= psqt::LAYOUT.mobility_open_offset,
            "param_grads too small: {} < {} (material+PSQT footprint)",
            param_grads.len(),
            psqt::LAYOUT.mobility_open_offset,
        );
        for i in 0..self.active_count {
            let a = &self.active[i];
            let s = f64::from(a.sign);
            param_grads[a.mg_idx as usize] += d_mg * s;
            param_grads[a.eg_idx as usize] += d_eg * s;
            param_grads[a.mat_mg_idx as usize] += d_mg * s;
            param_grads[a.mat_eg_idx as usize] += d_eg * s;
        }
    }
}

/// Run forward-mode AD on a position. Returns a `DualEvalResult` containing
/// the score and all information needed to scatter gradients later.
pub fn eval_dual_forward(board: &Board, values: &[f64]) -> DualEvalResult {
    use soul::engine::autograd::{
        EnvVec8,
        dual::{DualNode, DualVec8},
    };

    // PSQT + Material accumulator (plain f64, no tape)
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    let mut active = [ActivePiece::default(); 32];
    let mut active_count = 0;

    for piece in PieceType::ALL {
        let pt = piece.as_usize();
        piece_counts[pt] = f64::from(board.role_bb[pt].popcount());

        let mat_mg_idx = psqt::LAYOUT.material_offset + pt;
        let mat_eg_idx = psqt::LAYOUT.material_offset + 6 + pt;
        let mat_mg_val = values.get(mat_mg_idx).copied().unwrap_or(0.0);
        let mat_eg_val = values.get(mat_eg_idx).copied().unwrap_or(0.0);

        let mut bb_w = board.pieces(piece, Color::White);
        while bb_w.is_not_empty() {
            let sq = bb_w.pop_lsb();
            let sq_idx = usize::from(sq.flip_rank());
            let mirror_idx = psqt::mirror_sq(sq_idx);
            let mg_idx = pt * 64 + mirror_idx;
            let eg_idx = pt * 64 + 32 + mirror_idx;

            lane_vals[0] += values.get(mg_idx).copied().unwrap_or(0.0) + mat_mg_val;
            lane_vals[1] += values.get(eg_idx).copied().unwrap_or(0.0) + mat_eg_val;

            active[active_count] = ActivePiece {
                mg_idx:     mg_idx as u32,
                eg_idx:     eg_idx as u32,
                mat_mg_idx: mat_mg_idx as u32,
                mat_eg_idx: mat_eg_idx as u32,
                sign:       1.0,
            };
            active_count += 1;
        }

        let mut bb_b = board.pieces(piece, Color::Black);
        while bb_b.is_not_empty() {
            let sq = bb_b.pop_lsb();
            let sq_idx = usize::from(sq);
            let mirror_idx = psqt::mirror_sq(sq_idx);
            let mg_idx = pt * 64 + mirror_idx;
            let eg_idx = pt * 64 + 32 + mirror_idx;

            lane_vals[0] -= values.get(mg_idx).copied().unwrap_or(0.0) + mat_mg_val;
            lane_vals[1] -= values.get(eg_idx).copied().unwrap_or(0.0) + mat_eg_val;

            active[active_count] = ActivePiece {
                mg_idx:     mg_idx as u32,
                eg_idx:     eg_idx as u32,
                mat_mg_idx: mat_mg_idx as u32,
                mat_eg_idx: mat_eg_idx as u32,
                sign:       -1.0,
            };
            active_count += 1;
        }
    }

    let mut phase_dual = DualNode::zero();
    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;
        if phase_idx < values.len() {
            phase_dual += DualNode::constant(*count) * DualNode::constant(values[phase_idx]);
        }
    }
    let phase = phase_dual
        .math_clamp(DualNode::constant(0.0), DualNode::constant(24.0))
        .trunc();

    // Seed DualNode values
    // Only lanes 0 (MG) and 1 (EG) need gradient tracking.
    // Lane 2 is the Phase counter. Lanes 3-7 are unused padding.
    let mut dual_acc = DualVec8::zero();
    dual_acc.0[0] = DualNode::seed(lane_vals[0], 0);
    dual_acc.0[1] = DualNode::seed(lane_vals[1], 1);
    for (dual, &val) in dual_acc.0[2..8].iter_mut().zip(&lane_vals[2..8]) {
        *dual = DualNode::constant(val);
    }

    let params = soul::engine::eval::EvalParams::<DualNode>::load_tunable(values);

    // Build the macroscopic features boundary
    let openness = soul::engine::mobility::Mobility::compute_openness(board);
    let pinned_w = board.pinned_pieces(Color::White);
    let pinned_b = board.pinned_pieces(Color::Black);
    let tensor = soul::core::board::spatial::SpatialTensor::compute(board, pinned_w.0, pinned_b.0);
    let data =
        soul::engine::mobility::Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);

    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_king_ring = soul::core::board::bitboard::atk_king(w_ksq).0;
    let b_king_ring = soul::core::board::bitboard::atk_king(b_ksq).0;

    let xray_ortho = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

    let features = soul::engine::eval::MacroFeatures {
        openness,
        data,
        xray_ortho,
    };

    // Forward pass with dual numbers
    let result =
        soul::engine::eval::evaluate_generic::<DualNode>(board, &dual_acc, phase, &params, Some(&features));

    DualEvalResult {
        score: result.val,
        grad: result.grad,
        active,
        active_count,
    }
}

/// Fully fused eval + gradient scatter via forward-mode AD.
///
/// Single-pass: computes eval with dual numbers, sigmoid, loss derivative,
/// and scatters all gradients — no intermediate storage. PSQT gradients
/// are recovered by re-iterating piece bitboards (still hot in L1).
///
/// In the production training loop, `eval_linear_grad` handles gradients directly.
/// This function serves as the correctness oracle, run both and compare to verify
/// hand-derived gradient formulas.
///
/// Returns the squared error for loss tracking.
#[inline]
pub fn eval_dual_fused(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    use soul::engine::autograd::{
        EnvVec8,
        dual::{DualNode, DualVec8},
    };

    // PSQT + Material accumulator (plain f64 sums, no piece tracking)
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut lane_vals, &mut piece_counts);

    let mut phase_dual = DualNode::zero();
    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;
        if phase_idx < values.len() {
            phase_dual += DualNode::constant(*count) * DualNode::constant(values[phase_idx]);
        }
    }
    let phase = phase_dual
        .math_clamp(DualNode::constant(0.0), DualNode::constant(24.0))
        .trunc();

    // Seed DualNode values
    let mut dual_acc = DualVec8::zero();
    dual_acc.0[0] = DualNode::seed(lane_vals[0], 0);
    dual_acc.0[1] = DualNode::seed(lane_vals[1], 1);
    for (dual, &val) in dual_acc.0[2..8].iter_mut().zip(&lane_vals[2..8]) {
        *dual = DualNode::constant(val);
    }

    let params = soul::engine::eval::EvalParams::<DualNode>::load_tunable(values);

    // Build the macroscopic features boundary
    let openness = soul::engine::mobility::Mobility::compute_openness(board);
    let pinned_w = board.pinned_pieces(Color::White);
    let pinned_b = board.pinned_pieces(Color::Black);
    let tensor = soul::core::board::spatial::SpatialTensor::compute(board, pinned_w.0, pinned_b.0);
    let data =
        soul::engine::mobility::Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);

    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_king_ring = soul::core::board::bitboard::atk_king(w_ksq).0;
    let b_king_ring = soul::core::board::bitboard::atk_king(b_ksq).0;

    let xray_ortho = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

    let features = soul::engine::eval::MacroFeatures {
        openness,
        data,
        xray_ortho,
    };

    // Forward pass
    let result =
        soul::engine::eval::evaluate_generic::<DualNode>(board, &dual_acc, phase, &params, Some(&features));
    let score = result.val;

    // Sigmoid + loss derivative
    let sig = 1.0 / (1.0 + (-k * score).clamp(-700.0, 700.0).exp());
    let err = sig - target;
    let outer_deriv = 2.0 * err * sig * (1.0 - sig) * k;

    // Scatter gradients immediately (no intermediate storage)

    // EvalParams gradients (slots 2..29)
    let dummy = DualEvalResult {
        score:        0.0,
        grad:         result.grad,
        active:       [ActivePiece::default(); 32],
        active_count: 0,
    };
    dummy.scatter_dynamic(outer_deriv, param_grads);

    // PSQT gradients — re-iterate board pieces (still hot in L1)
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

/// Direct gradient extraction for the linear HCE.
///
/// Since the eval is fully linear in its parameters (every param appears as
/// `param · board_feature`), gradients are just the feature coefficients
/// computable from board state + phase + openness in ~90 f64 ops, vs ~2500
/// f32 ops for the dual number path.
///
/// Returns squared error for loss tracking.
#[inline]
pub fn eval_linear_grad(board: &Board, values: &[f64], target: f64, k: f64, param_grads: &mut [f64]) -> f64 {
    use soul::engine::mobility::Mobility;

    // PSQT + Material accumulator (for lane values + PSQT scatter)
    let mut lane_vals = [0.0f64; 8];
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut lane_vals, &mut piece_counts);

    let mut phase_raw = 0.0;
    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;
        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }

    // NOTE: dJ/dPhaseWeight is intentionally omitted from the
    // gradient scattering process below because the tuner architecture requires
    // the game phase thresholds to be strictly fixed constants.
    let phase = phase_raw.clamp(0.0, 24.0) as i32;
    let t_mg = phase as f64 / 24.0;
    let t_eg = (24.0 - phase as f64) / 24.0;
    let score = eval_f64(board, values);

    // Sigmoid + loss derivative
    let sig = 1.0 / (1.0 + (-k * score).clamp(-700.0, 700.0).exp());
    let err = sig - target;
    let outer = 2.0 * err * sig * (1.0 - sig) * k;

    let stm_sign: f64 = if board.stm == Color::White { 1.0 } else { -1.0 };
    let d = outer * stm_sign;

    // Mobility + safety features
    let pinned_w = board.pinned_pieces(Color::White);
    let pinned_b = board.pinned_pieces(Color::Black);
    let tensor = soul::core::board::spatial::SpatialTensor::compute(board, pinned_w.0, pinned_b.0);
    let mob_data = Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);
    let openness = Mobility::compute_openness(board);
    let o = openness as f64;
    let c = (1024 - openness) as f64;

    let lo = psqt::LAYOUT.mobility_open_offset;
    let lc = psqt::LAYOUT.mobility_closed_offset;
    let ks = psqt::LAYOUT.king_safety_offset;
    let ao = psqt::LAYOUT.attacker_offset;
    let xr = psqt::LAYOUT.xray_offset;

    // Feature difference vector (same as evaluate_score_diff builds)
    let idx_us = mob_data.safety_us.attackers.min(5);
    let idx_them = mob_data.safety_them.attackers.min(5);

    let diff = [
        (mob_data.metrics_us.mobility - mob_data.metrics_them.mobility) as f64,
        (mob_data.metrics_us.shadow_mobility - mob_data.metrics_them.shadow_mobility) as f64,
        (mob_data.metrics_us.threats - mob_data.metrics_them.threats) as f64,
        (mob_data.metrics_us.shadow_threats - mob_data.metrics_them.shadow_threats) as f64,
    ];

    // Scatter gradients

    // PSQT gradients
    let d_mg = d * t_mg;
    let d_eg = d * t_eg;

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

    // Mobility weight gradients
    // d(score)/d(mg_mob_open[j]) = t_mg · diff[j] · (openness/1024)
    // d(score)/d(mg_mob_closed[j]) = t_mg · diff[j] · (closedness/1024)
    // Same pattern for EG weights with t_eg
    let o_frac = o / 1024.0;
    let c_frac = c / 1024.0;

    let v_diff = Vf64x4::from(diff);
    let v_d_om = Vf64x4::splat(d * t_mg * o_frac);
    let v_d_oe = Vf64x4::splat(d * t_eg * o_frac);
    let v_d_cm = Vf64x4::splat(d * t_mg * c_frac);
    let v_d_ce = Vf64x4::splat(d * t_eg * c_frac);

    // SAFETY: The params array layout strictly guarantees that mobility parameter blocks
    // (lo, lo+4, lc, lc+4) are contiguous 4-float arrays fully within the param_grads buffer.
    unsafe {
        let p_lo_m = param_grads.as_mut_ptr().add(lo);
        let p_lo_e = param_grads.as_mut_ptr().add(lo + 4);
        let p_lc_m = param_grads.as_mut_ptr().add(lc);
        let p_lc_e = param_grads.as_mut_ptr().add(lc + 4);

        (Vf64x4::loadu(p_lo_m) + v_diff * v_d_om).storeu(p_lo_m);
        (Vf64x4::loadu(p_lo_e) + v_diff * v_d_oe).storeu(p_lo_e);
        (Vf64x4::loadu(p_lc_m) + v_diff * v_d_cm).storeu(p_lc_m);
        (Vf64x4::loadu(p_lc_e) + v_diff * v_d_ce).storeu(p_lc_e);
    }

    // King safety gradients
    // Safety params are tapered (mg only), so coeff is d · t_mg.
    let safety_coeff = d * t_mg;

    let shield_diff = (mob_data.safety_us.shield - mob_data.safety_them.shield) as f64;
    let ortho_diff = (mob_data.safety_us.ortho_exposure - mob_data.safety_them.ortho_exposure) as f64;
    let diag_diff = (mob_data.safety_us.diag_exposure - mob_data.safety_them.diag_exposure) as f64;

    param_grads[ks] += safety_coeff * shield_diff;
    param_grads[ks + 1] -= safety_coeff * ortho_diff;
    param_grads[ks + 2] -= safety_coeff * diag_diff;

    // Attacker weight gradients
    for atk_k in 0..6 {
        let mut atk_deriv = 0.0;
        if atk_k == idx_us {
            atk_deriv -= mob_data.safety_us.weak as f64 / 10.0;
        }
        if atk_k == idx_them {
            atk_deriv += mob_data.safety_them.weak as f64 / 10.0;
        }
        if atk_deriv != 0.0 {
            param_grads[ao + atk_k] += safety_coeff * atk_deriv;
        }
    }

    // X-Ray features
    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_king_ring = soul::core::board::bitboard::atk_king(w_ksq).0;
    let b_king_ring = soul::core::board::bitboard::atk_king(b_ksq).0;

    // NOTE: The symmetrical subtraction here correctly yields the difference.
    // When multiplying by d (which incorporates stm_sign and loss derivative),
    // the sign flips correctly cancel out exactly. This ensures that the gradient applied to
    // param_grads[xr] naturally aligns with the STM-relative score evaluation!
    let w_xray_ortho = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

    param_grads[xr] += d * t_mg * w_xray_ortho as f64;

    err * err
}

/// Score-only evaluation using `evaluate_generic::<f64>`.
///
/// No gradient tracking — used for K-factor line search, validation loss,
/// and as the score source in `eval_linear_grad`.
/// Mirrors the engine's integer eval exactly, substituting f64 for i32.
#[inline(always)]
pub fn eval_f64(board: &Board, values: &[f64]) -> f64 {
    eval_f64_with_acc(board, values).0
}

/// Score-only evaluation that also returns the trace accumulator and piece counts.
///
/// Used by `eval_linear_grad` to avoid redundant board iterations.
#[allow(clippy::too_many_lines)]
pub fn eval_f64_with_acc(board: &Board, values: &[f64]) -> (f64, [f64; 8], [f64; 6]) {
    use soul::engine::eval::evaluate_generic;

    let mut trace_acc = <f64 as soul::engine::autograd::EvalMath>::Vec8::zero();
    let mut piece_counts = [0.0f64; 6];

    accumulate_lane_vals(board, values, &mut trace_acc.0, &mut piece_counts);

    let mut phase_raw = 0.0;
    for (pt, count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;
        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }
    let phase = phase_raw.math_clamp(0.0, 24.0).trunc();

    use soul::engine::autograd::traits::F64Vec4;
    let lo = psqt::LAYOUT.mobility_open_offset;
    let lc = psqt::LAYOUT.mobility_closed_offset;
    let ks = psqt::LAYOUT.king_safety_offset;
    let ao = psqt::LAYOUT.attacker_offset;
    let xr = psqt::LAYOUT.xray_offset;

    let params = soul::engine::eval::EvalParams {
        mg_mob_open:   F64Vec4([values[lo], values[lo + 1], values[lo + 2], values[lo + 3]]),
        eg_mob_open:   F64Vec4([values[lo + 4], values[lo + 5], values[lo + 6], values[lo + 7]]),
        mg_mob_closed: F64Vec4([values[lc], values[lc + 1], values[lc + 2], values[lc + 3]]),
        eg_mob_closed: F64Vec4([values[lc + 4], values[lc + 5], values[lc + 6], values[lc + 7]]),
        w_shield:      values[ks],
        w_ortho:       values[ks + 1],
        w_diag:        values[ks + 2],
        atk_weights:   [
            values[ao],
            values[ao + 1],
            values[ao + 2],
            values[ao + 3],
            values[ao + 4],
            values[ao + 5],
        ],
        w_xray_ortho:  values[xr],
    };

    // Build the macroscopic features boundary
    let openness = soul::engine::mobility::Mobility::compute_openness(board);
    let pinned_w = board.pinned_pieces(Color::White);
    let pinned_b = board.pinned_pieces(Color::Black);
    let tensor = soul::core::board::spatial::SpatialTensor::compute(board, pinned_w.0, pinned_b.0);
    let data =
        soul::engine::mobility::Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);

    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_king_ring = soul::core::board::bitboard::atk_king(w_ksq).0;
    let b_king_ring = soul::core::board::bitboard::atk_king(b_ksq).0;

    let xray_ortho = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

    let features = soul::engine::eval::MacroFeatures {
        openness,
        data,
        xray_ortho,
    };

    (
        evaluate_generic::<f64>(board, &trace_acc, phase, &params, Some(&features)),
        trace_acc.0,
        piece_counts,
    )
}

#[cfg(test)]
mod tests {
    use soul::core::board::Position;

    use super::*;

    #[test]
    fn test_linear_oracle_verification() {
        let fens = [
            "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        ];

        let mut values = vec![0.0f64; soul::core::psqt::LAYOUT.xray_offset + 1];
        for (n, v) in values.iter_mut().enumerate() {
            *v = (n % 17) as f64 - 8.0;
        }

        let target = 0.5;
        let k = 0.005;

        for fen in &fens {
            let pos = Position::from_fen(fen);

            let mut linear_grad = vec![0.0f64; values.len()];
            let linear_eval = eval_linear_grad(&pos, &values, target, k, &mut linear_grad);

            let mut dual_grad = vec![0.0f64; values.len()];
            let dual_eval = eval_dual_fused(&pos, &values, target, k, &mut dual_grad);

            assert!((linear_eval - dual_eval).abs() < 1e-4, "Eval mismatch on fn: {}", fen);

            for (i, (&lg, &dg)) in linear_grad.iter().zip(dual_grad.iter()).enumerate() {
                assert!((lg - dg).abs() < 1e-3, "Gradient mismatch at index {} on {}", i, fen);
            }
        }
    }
}
