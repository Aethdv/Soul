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
        board::{Position, bitboard, spatial::SpatialTensor},
        defs::{Color, LANE_PHASE, PieceType, TOTAL_PHASE},
        psqt,
    },
    engine::{
        autograd::EvalMath,
        combiner::{Accumulators, Combiner, LinearCombiner},
        eval_params::{
            self, ATTACKER_WEIGHTS, BISHOP_PAIR_WEIGHTS, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_SAFETY_WEIGHTS,
            MG_MOBILITY_CLOSED, MG_MOBILITY_OPEN, XRAY_WEIGHTS,
        },
        mobility::{self, Mobility, MobilityData},
        search_params::SearchParams,
        term::{self},
    },
    weave::Vi16x8,
};

/// The standard integer evaluation used in the alpha-beta search.
#[inline]
pub fn evaluate(board: &Position, acc: &Vi16x8) -> i32 {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    evaluate_generic::<i32>(board, acc, phase, &params, None)
}

/// Stripped-down eval for volatility filtering — accumulator-only, no spatial features.
#[inline(always)]
pub fn evaluate_fast(board: &Position, acc: &Vi16x8, phase: i32) -> i32 {
    let score = i32::tapered(acc, phase);
    if board.stm == Color::White { score } else { -score }
}

/// Volatility-scaled margin for lazy eval pruning; scales with game phase.
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

/// Generic evaluation — monomorphized to `i32` for search, `AutogradNode` for tuning.
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

/// Macroscopic features extracted once per position — consumed by both the
/// engine's score pass and every registered `LinearTerm::scatter` in the tuner.
/// New eval terms that depend on fresh board state should extend this struct,
/// not duplicate extraction.
pub struct SharedFeatures {
    pub openness: i32,
    pub data: MobilityData,
    pub xray_ortho: i32,
    /// `+1` if white has the bishop pair and black doesn't,
    /// `-1` for the reverse, `0` otherwise.
    pub bishop_pair_diff: i32,
}

impl SharedFeatures {
    #[inline]
    pub fn compute(board: &Position) -> Self {
        let pinned_w = board.pinned_pieces(Color::White);
        let pinned_b = board.pinned_pieces(Color::Black);
        let tensor = SpatialTensor::compute(board, pinned_w.0, pinned_b.0);

        let data = Mobility::compute_all(board, Color::White, &tensor, pinned_w, pinned_b);
        let openness = Mobility::compute_openness(board);

        let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
        let w_king_ring = bitboard::atk_king(w_ksq).0;
        let b_king_ring = bitboard::atk_king(b_ksq).0;

        let xray_ortho =
            (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32 - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        Self { openness, data, xray_ortho, bishop_pair_diff }
    }
}

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

/// Extracts the game phase directly from the accumulator's dedicated lane.
#[inline(always)]
pub fn extract_phase(acc: &Vi16x8) -> i32 {
    i32::from(acc.extract::<{ LANE_PHASE as i32 }>()).clamp(0, TOTAL_PHASE)
}

// ──────── Eval terms ────────

/// X-ray king-ring differential; shares the king-safety block's scalar upstream.
pub struct XrayTerm;

impl term::LinearTerm for XrayTerm {
    /// Scalar upstream — x-ray is tapered MG-only inside the combiner's
    /// king-safety block, same cadence as king safety.
    type Upstream = f64;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
        acc.xray = params.w_xray_ortho * T::from_i32(features.xray_ortho);
    }

    #[inline]
    fn scatter(features: &SharedFeatures, upstream: f64, grads: &mut [f64]) {
        grads[psqt::LAYOUT.xray_offset] += upstream * features.xray_ortho as f64;
    }
}

/// Tapered bonus for holding both bishops (~9 Elo).
pub struct BishopPairTerm;

impl term::LinearTerm for BishopPairTerm {
    type Upstream = term::TaperPair;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>) {
        let feature = T::from_i32(features.bishop_pair_diff);
        let eg_phase = T::from_i32(TOTAL_PHASE) - phase;
        let tapered = params.w_bp_mg * phase + params.w_bp_eg * eg_phase;
        acc.bonus += (tapered * feature / T::from_i32(TOTAL_PHASE)).trunc();
    }

    #[inline]
    fn scatter(features: &SharedFeatures, upstream: term::TaperPair, grads: &mut [f64]) {
        let feature = features.bishop_pair_diff as f64;
        grads[eval_params::LAYOUT.bishop_pair_offset] += upstream.d_mg * feature;
        grads[eval_params::LAYOUT.bishop_pair_offset + 1] += upstream.d_eg * feature;
    }
}

crate::register_terms! {
    mobility::MobilityTerm => mobility,
    mobility::KingSafetyTerm => king_safety,
    BishopPairTerm => bonus,
    XrayTerm => xray,
}

// ──────── Combiner glue ────────

/// Build the per-bucket accumulator by initializing the PSQT-level `mg_eg` bucket
/// from the SIMD accumulator, zeroing the rest, and applying every registered term.
/// Isolated so both `compute_macro_eval` and `detailed_eval` produce identical
/// bucket values from one code path.
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

// ──────── Parameter generation ────────

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
