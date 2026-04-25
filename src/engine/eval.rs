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
        combiner::{Accumulators, Combiner, LinearCombiner},
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

    if board.stm == Color::White { score } else { -score }
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
/// to the evaluation parameters, the affected [`crate::engine::term::LinearTerm::scatter`]
/// impl will silently compute mathematically invalid gradients because
/// [`LinearTerm`] assumes perfect parameter linearity (`y = w · x`).
/// Non-linear shapes belong in the [`crate::engine::combiner::Combiner`] layer
/// or soon a future `NonlinearTerm`.
///
/// If you aren't sure, just run the `test_linear_oracle_verification` test in
/// `tuner/src/evaltune/tape.rs` — it compares against the dual-number forward
/// pass and fails on any drift.
///
/// Non-PSQT weights come from `params` rather than const arrays,
/// so the autograd tape can track them.
#[inline(always)]
pub fn evaluate_generic<T: EvalMath<Scalar = T>>(
    board: &Position,
    acc: &T::Vec8,
    phase: T,
    params: &EvalParams<T>,
    features: Option<&SharedFeatures>,
) -> T {
    let features_store;
    let features = if let Some(f) = features {
        f
    } else {
        features_store = SharedFeatures::compute(board);
        &features_store
    };

    let score = compute_macro_eval(acc, phase, features, params);

    if board.stm == Color::White { score } else { -score }
}

pub struct DetailedEval {
    pub psqt: i32,
    pub mobility: i32,
    pub bonus: i32,
    pub safety: i32,
    pub total: i32,
}

/// A version of evaluate that returns individual components for debugging and visualization.
pub fn detailed_eval(board: &Position, acc: &Vi16x8) -> DetailedEval {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    let features = SharedFeatures::compute(board);

    let buckets = fill_accumulators::<i32>(acc, phase, &features, &params);

    let psqt = buckets.mg_eg;
    let mobility = buckets.mobility;
    let bonus = buckets.bonus;
    let total = LinearCombiner::forward(&buckets, phase);
    let safety = total - psqt - mobility - bonus;

    let (p, m, b, s, t) = if board.stm == Color::White {
        (psqt, mobility, bonus, safety, total)
    } else {
        (-psqt, -mobility, -bonus, -safety, -total)
    };

    DetailedEval { psqt: p, mobility: m, bonus: b, safety: s, total: t }
}

/// Macroscopic features extracted once per position.
///
/// Single feature-extraction boundary: computed once, consumed by both
/// the engine's score pass (`compute_macro_eval`) and every registered
/// [`crate::engine::term::LinearTerm::scatter`] in the tuner's backward
/// pass. Adding a new eval term that depends on fresh board state should
/// extend this struct, not duplicate extraction.
pub struct SharedFeatures {
    pub openness: i32,
    pub data: MobilityData,
    pub xray_ortho: i32,
    /// `+1` if white has the bishop pair and black doesn't, `-1` for the
    /// reverse, `0` otherwise.
    pub bishop_pair_diff: i32,
}

impl SharedFeatures {
    /// Single-pass extraction: spatial tensor, mobility/safety metrics, openness,
    /// king-ring x-ray differential. Mirrors what both the engine and the tuner
    /// need, so neither side recomputes.
    #[inline]
    pub fn compute(board: &Position) -> Self {
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

        let xray_ortho =
            (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32 - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        Self { openness, data, xray_ortho, bishop_pair_diff }
    }
}

/// X-ray king-ring differential: `w_xray_ortho · xray_ortho`. Writes
/// `acc.xray`; shares the king-safety block's scalar upstream.
pub struct XrayTerm;

impl crate::engine::term::LinearTerm for XrayTerm {
    /// Scalar upstream — x-ray is tapered MG-only inside the combiner's
    /// king-safety block, same cadence as king safety.
    type Upstream = f64;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
        acc.xray = params.w_xray_ortho * T::from_i32(features.xray_ortho);
    }

    #[inline]
    fn scatter(features: &SharedFeatures, upstream: f64, grads: &mut [f64]) {
        grads[crate::core::psqt::LAYOUT.xray_offset] += upstream * features.xray_ortho as f64;
    }
}

/// ── Bishop Pair (~9 Elo) ──
/// Tapered bonus for holding both bishops.
/// Routes to the shared `bonus` bucket via `acc.bonus += mg · phase + eg · eg_phase`
/// (× the +1/-1/0 `bishop_pair_diff` feature).
pub struct BishopPairTerm;

impl crate::engine::term::LinearTerm for BishopPairTerm {
    type Upstream = crate::engine::term::TaperPair;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>) {
        let feature = T::from_i32(features.bishop_pair_diff);
        let eg_phase = T::from_i32(TOTAL_PHASE) - phase;
        let tapered = params.w_bp_mg * phase + params.w_bp_eg * eg_phase;
        acc.bonus += (tapered * feature / T::from_i32(TOTAL_PHASE)).trunc();
    }

    #[inline]
    fn scatter(features: &SharedFeatures, upstream: crate::engine::term::TaperPair, grads: &mut [f64]) {
        let feature = features.bishop_pair_diff as f64;
        grads[crate::engine::eval_params::LAYOUT.bishop_pair_offset] += upstream.d_mg * feature;
        grads[crate::engine::eval_params::LAYOUT.bishop_pair_offset + 1] += upstream.d_eg * feature;
    }
}

crate::register_terms! {
    crate::engine::mobility::MobilityTerm => mobility,
    crate::engine::mobility::KingSafetyTerm => king_safety,
    BishopPairTerm => bonus,
    XrayTerm => xray,
}

/// The single source of truth for evaluation math.
/// Generic over `EvalMath` to support both `i32` search and `DualNode` tuning.
#[inline(always)]
pub fn compute_macro_eval<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> T {
    let buckets = fill_accumulators::<T>(acc, phase, features, params);
    LinearCombiner::forward(&buckets, phase)
}

/// Build the per-bucket accumulator by initializing the PSQT-level
/// `mg_eg` bucket from the SIMD accumulator, zeroing the rest, and letting
/// every registered [`LinearTerm`] populate its owned bucket(s) via
/// [`apply_all_terms`]. Isolated so both `compute_macro_eval` and
/// `detailed_eval` produce identical bucket values from one code path.
#[inline]
fn fill_accumulators<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> Accumulators<T> {
    let mut buckets = Accumulators::<T> {
        mg_eg: T::tapered(acc, phase),
        mobility: T::zero(),
        bonus: T::zero(),
        safety_us: T::zero(),
        safety_them: T::zero(),
        xray: T::zero(),
    };
    apply_all_terms::<T>(features, params, phase, &mut buckets);
    buckets
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
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr) ),* $(,)?) => {
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
                        $(
                            $name: T::[<load_ $ty:lower>](
                                values,
                                $crate::engine::eval_params::LAYOUT.$offset_field + $extra,
                                &mut slot,
                            ),
                        )*
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
            ATTACKER_WEIGHTS, BISHOP_PAIR_WEIGHTS, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_SAFETY_WEIGHTS, MG_MOBILITY_CLOSED,
            MG_MOBILITY_OPEN, XRAY_WEIGHTS,
        };

        Self {
            mg_mob_open: MG_MOBILITY_OPEN,
            mg_mob_closed: MG_MOBILITY_CLOSED,
            eg_mob_open: EG_MOBILITY_OPEN,
            eg_mob_closed: EG_MOBILITY_CLOSED,
            w_shield: KING_SAFETY_WEIGHTS[0],
            w_ortho: KING_SAFETY_WEIGHTS[1],
            w_diag: KING_SAFETY_WEIGHTS[2],
            atk_weights: ATTACKER_WEIGHTS,
            w_xray_ortho: XRAY_WEIGHTS[0],
            w_bp_mg: BISHOP_PAIR_WEIGHTS[0],
            w_bp_eg: BISHOP_PAIR_WEIGHTS[1],
        }
    }
}
