//! Evaluation score combination, collapsing per-bucket term outputs
//! into the final scalar eval.
//!
//! Two stages:
//! 1) Terms fill [`Accumulators`] buckets.
//! 2) A [`Combiner`] collapses them. Separating these lets the tuner propagate one
//!    upstream derivative per bucket through each term's scatter, rather than
//!    baking the combiner shape into every term.
//!
//! The current combiner ([`LinearCombiner`]) sums pre-tapered buckets and applies
//! a phase taper only to the king-safety block. Future combiners (sigmoid danger,
//! winnable eg-scale) implement [`Combiner`] without touching any term.

use crate::{
    core::defs::TOTAL_PHASE,
    engine::{
        autograd::EvalMath,
        eval::EvalParams,
        eval_params::LAYOUT,
        term::{BucketUpstreams, KingSafetyUpstream, TaperPair},
    },
};

/// Per-bucket score accumulators filled by the term layer before combination.
///
/// Fields are either pre-tapered (`mg_eg`, `mobility`), where the mg/eg blend
/// happened inside the SIMD lanes to keep them vectorized, or raw
/// (`bonus_mg`/`bonus_eg`, `safety_us`, `safety_them`, `xray`), which the combiner
/// tapers. Tapering raw keeps the divide-and-truncate at one site, so per-term
/// rounding never accumulates.
pub struct Accumulators<T: EvalMath> {
    /// Material + PSQT, read from the SIMD accumulator.
    pub mg_eg: T,
    /// Mobility differential, blended inside the SIMD lanes to preserve vectorization.
    pub mobility: T,
    /// Raw bonus coefficients; each term adds its pure `mg · feature` and
    /// `eg · feature`, and the combiner tapers the summed pair once.
    pub bonus_mg: T,
    pub bonus_eg: T,
    /// Shelter minus exposure for the side to move.
    pub safety_us: T,
    /// Shelter minus exposure for the opponent.
    pub safety_them: T,
    /// Attacker pressure on our king, kept apart from shelter so the combiner
    /// can curve it. Non-negative.
    pub danger_us: T,
    /// Attacker pressure on theirs.
    pub danger_them: T,
    /// Raw x-ray king-ring differential; separate bucket so it can be rerouted independently.
    pub xray: T,
}

/// The combiner's own weights, the ones no term owns.
pub struct CombinerParams<T: EvalMath> {
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

/// Denominator of the curvature weight, sized so one integer step of
/// `KING_DANGER` is ~3% of the shipped curve rather than 25%. `pressure²` tops
/// out near 216k, leaving `i32` room for curvatures into the thousands.
pub const DANGER_SCALE: i32 = 32768;

/// The one non-linearity in the eval: pressure accelerates instead of scaling
/// with the weight table alone. Curvature is a numerator so that zero is exactly
/// the linear block; as a divisor it could only approach one.
#[inline(always)]
fn king_danger<T: EvalMath<Scalar = T>>(pressure: T, curvature: T) -> T {
    pressure + ((pressure * pressure * curvature) / T::from_i32(DANGER_SCALE)).trunc()
}

/// The tapered mg/eg blend: weight by phase, divide back to centipawns, truncate.
/// The one rounding site in the combiner; `eg = zero` degenerates to the mg-only
/// taper the king-safety block wants.
#[inline(always)]
pub fn taper<T: EvalMath<Scalar = T>>(mg: T, eg: T, phase: T) -> T {
    let total = T::from_i32(TOTAL_PHASE);
    ((mg * phase + eg * (total - phase)) / total).trunc()
}

/// Strategy for collapsing [`Accumulators`] into a final evaluation scalar.
///
/// A combiner consumes only bucket values and combiner-owned params; it
/// never reads raw features. Non-linearities over bucket values (sigmoid
/// king danger, quadratic pressure, eg scale factors) belong here.
pub trait Combiner {
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T, params: &CombinerParams<T>) -> T;

    /// Returns per-bucket upstream gradients given the loss derivative and game phase.
    /// Non-linear combiners also write their own param gradients into `grads`, and
    /// read `buckets`, since their partials depend on the values they collapse.
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

        // d/dp of p + c·p²/scale, and the block subtracts our danger and adds theirs.
        let slope = |p: f64| 1.0 + 2.0 * p * params.king_danger / scale;
        let king_safety = KingSafetyUpstream {
            shelter: safety_block,
            danger_us: -safety_block * slope(buckets.danger_us),
            danger_them: safety_block * slope(buckets.danger_them),
        };

        // d/dc: the parameter the combiner owns rather than routes.
        let (us, them) = (buckets.danger_us, buckets.danger_them);
        grads[LAYOUT.king_danger_offset] += safety_block * (them * them - us * us) / scale;

        BucketUpstreams { mg_eg: taper_pair, mobility: taper_pair, bonus: taper_pair, king_safety, xray: safety_block }
    }
}
