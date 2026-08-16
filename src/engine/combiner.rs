//! Combines intermediate evaluation buckets into the final scalar score.
//!
//! Evaluation proceeds in two stages:
//! 1. Evaluation terms populate [`Accumulators`] buckets.
//! 2. A [`Combiner`] aggregates these buckets into a final score.
//!
//! Separating term evaluation from score combination allows the autograd / tuner
//! to backpropagate a single upstream gradient per bucket into term features,
//! keeping combiner non-linearities decoupled from feature extraction.
//!
//! [`LinearCombiner`] adds the pre-tapered buckets straight through, tapers the raw
//! bonus pair, and tapers the king-safety block middlegame-only. Future combiners
//! (sigmoid danger, winnable eg-scale) implement [`Combiner`] without touching a term.

use crate::{
    core::defs::TOTAL_PHASE,
    engine::{
        autograd::EvalMath,
        eval::EvalParams,
        eval_params::LAYOUT,
        term::{BucketUpstreams, KingSafetyUpstream, TaperPair},
    },
};

/// Fixed-point divisor for the quadratic king danger, `2^15`.
///
/// Sized so one integer step of the curvature moves the shipped curve by about 3%
/// rather than 25%. Pressure tops out near 460, so `pressure²` reaches about 216k and
/// leaves `i32` room for curvatures into the thousands.
pub const DANGER_SCALE: i32 = 32768;

/// Intermediate evaluation buckets populated by evaluation terms prior to combination.
///
/// Buckets are either:
/// - Pre-tapered (`mg_eg`, `mobility`): Blended directly inside SIMD lanes
///   to maintain vectorization efficiency.
/// - Raw (`bonus_mg`/`bonus_eg`, `safety_us`, `safety_them`, `xray`):
///   Tapered once in the combiner to avoid accumulating per-term rounding errors.
pub struct Accumulators<T: EvalMath> {
    /// Material and PSQT scores from the SIMD accumulator.
    pub mg_eg: T,
    /// Mobility score differential, pre-tapered in SIMD lanes.
    pub mobility: T,
    /// Raw middlegame bonus sum across all active positional terms.
    pub bonus_mg: T,
    /// Raw endgame bonus sum across all active positional terms.
    pub bonus_eg: T,
    /// King shelter minus king exposure for the side to move.
    pub safety_us: T,
    /// King shelter minus king exposure for the opponent.
    pub safety_them: T,
    /// Non-negative attacker pressure on our king.
    pub danger_us: T,
    /// Non-negative attacker pressure on the opponent's king.
    pub danger_them: T,
    /// Raw x-ray / king-ring attack differential.
    pub xray: T,
}

/// Parameters owned and tuned directly by the combiner.
pub struct CombinerParams<T: EvalMath> {
    /// Curvature `c` on king attacker pressure.
    pub king_danger: T,
}

impl<T: EvalMath<Scalar = T>> CombinerParams<T> {
    #[inline(always)]
    pub fn from_eval(params: &EvalParams<T>) -> Self {
        Self { king_danger: params.w_king_danger }
    }
}

impl CombinerParams<f64> {
    #[inline(always)]
    pub fn from_values(values: &[f64]) -> Self {
        Self { king_danger: values[LAYOUT.king_danger_offset] }
    }
}

/// The one non-linearity in the eval: `p + trunc(p²·c / DANGER_SCALE)`, so pressure
/// accelerates instead of scaling with the weight table alone. Curvature is a numerator
/// so that `c = 0` is exactly the linear block; as a divisor it could only approach it.
#[inline(always)]
fn king_danger<T: EvalMath<Scalar = T>>(pressure: T, curvature: T) -> T {
    pressure + ((pressure * pressure * curvature) / T::from_i32(DANGER_SCALE)).trunc()
}

/// `trunc((mg·phase + eg·(TOTAL_PHASE - phase)) / TOTAL_PHASE)`, the one rounding site in
/// the combiner. Truncation is toward zero, so a negative score rounds up, and `eg = 0`
/// degenerates to the middlegame-only taper the king-safety block wants.
#[inline(always)]
pub fn taper<T: EvalMath<Scalar = T>>(mg: T, eg: T, phase: T) -> T {
    let total = T::from_i32(TOTAL_PHASE);
    ((mg * phase + eg * (total - phase)) / total).trunc()
}

/// Combines [`Accumulators`] into a final evaluation scalar.
///
/// Combiners operate strictly on aggregate bucket values and combiner parameters;
/// they do not access individual board features.
pub trait Combiner {
    /// Collapses bucket values into a final evaluation score.
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T, params: &CombinerParams<T>) -> T;

    /// Per-bucket upstream gradients ∂L/∂bucket, with any combiner-owned ∂L/∂θ accumulated
    /// into `grads`. A non-linear combiner reads `buckets` too, since its partials depend on
    /// the values it collapses.
    fn backward(
        buckets: &Accumulators<f64>,
        phase: f64,
        params: &CombinerParams<f64>,
        d_loss: f64,
        grads: &mut [f64],
    ) -> BucketUpstreams;
}

pub struct LinearCombiner;

impl Combiner for LinearCombiner {
    #[inline(always)]
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T, params: &CombinerParams<T>) -> T {
        let c = params.king_danger;
        let danger = king_danger(buckets.danger_us, c) - king_danger(buckets.danger_them, c);
        let safety_diff = buckets.safety_us - buckets.safety_them - danger + buckets.xray;
        let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
        let safety = taper(safety_diff, T::zero(), phase);
        buckets.mg_eg + buckets.mobility + bonus + safety
    }

    #[inline]
    fn backward(
        buckets: &Accumulators<f64>,
        phase: f64,
        params: &CombinerParams<f64>,
        d_loss: f64,
        grads: &mut [f64],
    ) -> BucketUpstreams {
        let t_mg = phase / f64::from(TOTAL_PHASE);
        let t_eg = 1.0 - t_mg;
        let taper_pair = TaperPair { d_mg: d_loss * t_mg, d_eg: d_loss * t_eg };
        let safety_block = d_loss * t_mg;
        let scale = f64::from(DANGER_SCALE);

        // ∂/∂p (p + c·p² / scale) = 1 + 2·c·p / scale, and the block subtracts our danger
        // and adds theirs.
        let slope = |p: f64| 1.0 + 2.0 * p * params.king_danger / scale;
        let king_safety = KingSafetyUpstream {
            shelter: safety_block,
            danger_us: -safety_block * slope(buckets.danger_us),
            danger_them: safety_block * slope(buckets.danger_them),
        };

        // ∂L/∂c = (∂L/∂safety) · (∂safety/∂c) = safety_block · (p_them² - p_us²) / scale
        let (us, them) = (buckets.danger_us, buckets.danger_them);
        grads[LAYOUT.king_danger_offset] += safety_block * (them * them - us * us) / scale;

        BucketUpstreams { mg_eg: taper_pair, mobility: taper_pair, bonus: taper_pair, king_safety, xray: safety_block }
    }
}
