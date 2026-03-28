//! Hand-Crafted Evaluation.
//!
//! # Architecture
//!
//! Computes the heuristic value of a leaf node. The evaluation relies on an
//! incrementally updated SIMD accumulator for piece-square material (PSQT),
//! combined with dynamically computed spatial features (mobility, king safety, threats).
//!
//! The function is generic over `EvalMath` (either `i32` or `AutogradNode`).
//! During search, `i32` is monomorphized into direct compiler-optimized arithmetic.
//! During tuning, `AutogradNode` constructs a computational graph
//! to track parameter gradients via forward-mode auto-differentiation.

use crate::{
    core::{
        board::Position,
        defs::{Color, LANE_PHASE, PieceType, TOTAL_PHASE},
    },
    engine::{
        autograd::EvalMath,
        mobility::{Mobility, MobilityData},
        search_params::SearchParams,
    },
    weave::Vi16x8,
};

// ──────── Engine Entry Points ────────

/// The standard integer evaluation used in the alpha-beta search.
#[inline]
pub fn evaluate(board: &Position, acc: &Vi16x8) -> i32 {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    evaluate_generic::<i32>(board, acc, phase, &params, None)
}

/// A highly stripped-down evaluation used by `eval_uncertainty`.
/// Skips all positional bonuses (mobility, king safety, etc.)
/// and relies entirely on the accumulator (which carries both material
/// and static positional PSQT values).
#[inline(always)]
pub fn evaluate_fast(board: &Position, acc: &Vi16x8, phase: i32) -> i32 {
    use crate::engine::autograd::EvalMath;
    let score = i32::tapered(acc, phase);

    if board.stm == Color::White {
        score
    } else {
        -score
    }
}

/// Uncertainty bounds — dynamically estimating the error of lazy evaluation.
///
/// A quick material tally isn't always accurate. To know if we can safely skip the
/// expensive full positional evaluation (Lazy Eval), we calculate a volatility margin.
/// This margin scales with the game phase and the specific 'volatility' of the pieces present.
/// The early game allows for huge positional swings, whereas endgame material count is far more rigid.
#[inline]
pub fn lazy_eval_margin(board: &Position, phase: i32, params: &SearchParams) -> i32 {
    let volatility = board.piece_count(PieceType::Pawn) * params.vol_pawn
        + board.piece_count(PieceType::Knight) * params.vol_knight
        + board.piece_count(PieceType::Bishop) * params.vol_bishop
        + board.piece_count(PieceType::Rook) * params.vol_rook
        + board.piece_count(PieceType::Queen) * params.vol_queen
        + board.piece_count(PieceType::King) * params.vol_king;

    let div = std::cmp::max(params.lazy_eval_divisor, 1);
    let scaled = (volatility * phase) / div;

    params.lazy_eval_margin + scaled
}

// ──────── Core Evaluation Logic ────────

/// The core evaluation function.
///
/// Generic over `EvalMath` — monomorphized to `i32` for the search hot path,
/// `AutogradNode` for backpropagation during tuning.
///
/// WARNING: Autograd Linearity Booby Trap
/// If you introduce any non-linear math (e.g. `feature * feature * weight` or `max(feature, 0)`)
/// to the evaluation parameters, `eval_linear_grad` in `tuner/src/evaltune/tape.rs` will silently compute
/// mathematically invalid gradients because it assumes perfect parameter linearity (`y = w * x`).
/// Any non-linear eval updates require a manual calculus derivation update in `tape.rs`.
/// If you aren't sure, use `DualNode` oracle mode to verify structural gradients.
///
/// Non-PSQT weights come from `params` rather than const arrays,
/// so the autograd tape can track them.
#[inline(always)]
pub fn evaluate_generic<T: EvalMath<Scalar = T>>(
    board: &Position,
    acc: &T::Vec8,
    phase: T,
    params: &EvalParams<T>,
    features: Option<&MacroFeatures>,
) -> T {
    let features_store;
    let features = if let Some(f) = features {
        f
    } else {
        use crate::core::board::spatial::SpatialTensor;

        let pinned_w = board.pinned_pieces(Color::White);
        let pinned_b = board.pinned_pieces(Color::Black);
        let tensor = SpatialTensor::compute(board, pinned_w.0, pinned_b.0);

        let data = Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);
        let openness = Mobility::compute_openness(board);

        let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
        let w_king_ring = crate::core::board::bitboard::atk_king(w_ksq).0;
        let b_king_ring = crate::core::board::bitboard::atk_king(b_ksq).0;

        let xray_ortho = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
            - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

        features_store = MacroFeatures {
            openness,
            data,
            xray_ortho,
        };
        &features_store
    };

    let score = compute_macro_eval(acc, phase, features, params);

    if board.stm == Color::White {
        score
    } else {
        -score
    }
}

pub struct DetailedEval {
    pub psqt:     i32,
    pub mobility: i32,
    pub safety:   i32,
    pub total:    i32,
}

/// A version of evaluate that returns individual components for debugging and visualization.
pub fn detailed_eval(board: &Position, acc: &Vi16x8) -> DetailedEval {
    use crate::core::board::spatial::SpatialTensor;

    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();

    let psqt = i32::tapered(acc, phase);

    let pinned_w = board.pinned_pieces(Color::White);
    let pinned_b = board.pinned_pieces(Color::Black);
    let tensor = SpatialTensor::compute(board, pinned_w.0, pinned_b.0);

    let data = Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);
    let openness = Mobility::compute_openness(board);

    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_king_ring = crate::core::board::bitboard::atk_king(w_ksq).0;
    let b_king_ring = crate::core::board::bitboard::atk_king(b_ksq).0;

    let xray_ortho_raw = (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

    let features = MacroFeatures {
        openness,
        data,
        xray_ortho: xray_ortho_raw,
    };

    let w_atk_us = params.atk_weights[features.data.safety_us.attackers.min(5)];
    let w_atk_them = params.atk_weights[features.data.safety_them.attackers.min(5)];

    let s_us_score = features
        .data
        .safety_us
        .score(params.w_shield, params.w_ortho, params.w_diag, w_atk_us);
    let s_them_score =
        features
            .data
            .safety_them
            .score(params.w_shield, params.w_ortho, params.w_diag, w_atk_them);

    let xray_diff = params.w_xray_ortho * features.xray_ortho;
    let safety = (s_us_score - s_them_score + xray_diff) * phase / crate::core::defs::TOTAL_PHASE;

    let mobility = Mobility::evaluate_score_diff::<i32>(
        &features.data.metrics_us,
        &features.data.metrics_them,
        features.openness,
        phase,
        params.mg_mob_open,
        params.mg_mob_closed,
        params.eg_mob_open,
        params.eg_mob_closed,
    );

    let total = psqt + safety + mobility;
    let (p, m, s, t) = if board.stm == Color::White {
        (psqt, mobility, safety, total)
    } else {
        (-psqt, -mobility, -safety, -total)
    };

    DetailedEval {
        psqt:     p,
        mobility: m,
        safety:   s,
        total:    t,
    }
}

/// Macroscopic features extracted from the position.
/// Used to bridge the engine's search eval and the tuner's gradient extraction.
pub struct MacroFeatures {
    pub openness:   i32,
    pub data:       MobilityData,
    pub xray_ortho: i32,
}

/// The single source of truth for evaluation math.
/// Generic over `EvalMath` to support both `i32` search and `DualNode` tuning.
#[inline(always)]
pub fn compute_macro_eval<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &MacroFeatures,
    params: &EvalParams<T>,
) -> T {
    let mut score = T::tapered(acc, phase);

    let w_atk_us = params.atk_weights[features.data.safety_us.attackers.min(5)];
    let w_atk_them = params.atk_weights[features.data.safety_them.attackers.min(5)];

    let s_us_score = features
        .data
        .safety_us
        .score(params.w_shield, params.w_ortho, params.w_diag, w_atk_us);
    let s_them_score =
        features
            .data
            .safety_them
            .score(params.w_shield, params.w_ortho, params.w_diag, w_atk_them);

    let xray_diff = params.w_xray_ortho * T::from_i32(features.xray_ortho);

    // Tapered king safety: only applies heavily in the middlegame
    // Combines direct king safety and x-ray attacks into the ring.
    let safety_diff = s_us_score - s_them_score + xray_diff;
    score += ((safety_diff * phase) / T::from_i32(crate::core::defs::TOTAL_PHASE)).trunc();

    score += Mobility::evaluate_score_diff::<T>(
        &features.data.metrics_us,
        &features.data.metrics_them,
        features.openness,
        phase,
        params.mg_mob_open,
        params.mg_mob_closed,
        params.eg_mob_open,
        params.eg_mob_closed,
    );

    score
}

// ──────── Parameters & Utilities ────────

/// Extracts the game phase directly from the accumulator's dedicated lane.
#[inline(always)]
pub fn extract_phase(acc: &Vi16x8) -> i32 {
    i32::from(acc.extract::<{ LANE_PHASE as i32 }>()).clamp(0, TOTAL_PHASE)
}

/// Non-PSQT evaluation weights, generic over the math type.
///
/// - For the search hot path: `EvalParams::<i32>::from_const()`
///   inlines to direct constant loads.
/// - For the tuner: constructed with `AutogradNode::parameter()`
///   so gradient flows to the values array.
macro_rules! impl_eval_params {
    ($( ($name:ident, $ty:ident, $offset:expr) ),* $(,)?) => {
        pub struct EvalParams<T: EvalMath> {
            $( pub $name: <T as EvalMath>::$ty, )*
        }

        impl<T: EvalMath<Scalar = T>> EvalParams<T> {
            #[allow(dead_code)]
            pub fn load_tunable(values: &[f64]) -> Self {
                // Slots 0 and 1 are reserved for the PSQT accumulator gradients:
                // slot 0 tracks the MiddleGame (MG) material/positional score,
                // slot 1 tracks the EndGame (EG) material/positional score.
                // The dynamically tuned EvalParams begin at slot 2.
                let mut slot = 2;
                paste::paste! {
                    Self {
                        $( $name: T::[<load_ $ty:lower>](values, $offset, &mut slot), )*
                    }
                }
            }
        }
    }
}

crate::define_tunables!(impl_eval_params);

impl EvalParams<i32> {
    /// Load from compile-time const arrays. The compiler inlines this entirely.
    #[inline(always)]
    pub fn from_const() -> Self {
        use crate::engine::eval_params::{
            ATTACKER_WEIGHTS, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_SAFETY_WEIGHTS, MG_MOBILITY_CLOSED,
            MG_MOBILITY_OPEN, XRAY_WEIGHTS,
        };

        Self {
            mg_mob_open:   MG_MOBILITY_OPEN,
            mg_mob_closed: MG_MOBILITY_CLOSED,
            eg_mob_open:   EG_MOBILITY_OPEN,
            eg_mob_closed: EG_MOBILITY_CLOSED,
            w_shield:      KING_SAFETY_WEIGHTS[0],
            w_ortho:       KING_SAFETY_WEIGHTS[1],
            w_diag:        KING_SAFETY_WEIGHTS[2],
            atk_weights:   ATTACKER_WEIGHTS,
            w_xray_ortho:  XRAY_WEIGHTS[0],
        }
    }
}
