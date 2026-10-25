//! Combines evaluation buckets and propagates their upstream gradients.
//! Nonlinearities that couple tunable weights live here rather than in individual terms.

#[cfg(feature = "dataset")]
use crate::engine::{
    eval_params::LAYOUT,
    term::{BucketUpstreams, KingSafetyUpstream, TaperPair},
};
use crate::{
    core::defs::TOTAL_PHASE,
    engine::{autograd::EvalMath, eval::EvalParams},
};

/// Fixed-point divisor for quadratic king danger,
/// balancing curvature resolution against `i32` headroom.
const DANGER_SCALE: i32 = 32_768;

/// Side labels follow the input features' perspective.
pub(crate) struct Accumulators<T: EvalMath> {
    pub(crate) material_psqt: T,
    pub(crate) mobility: T,
    pub(crate) bonus_mg: T,
    pub(crate) bonus_eg: T,
    pub(crate) safety_us: T,
    pub(crate) safety_them: T,
    pub(crate) pressure_us: T,
    pub(crate) pressure_them: T,
    pub(crate) xray: T,
}

pub(crate) struct CombinerParams<T: EvalMath> {
    pub(crate) danger_curvature: T,
}

impl<T: EvalMath<Scalar = T>> CombinerParams<T> {
    #[inline(always)]
    pub(crate) fn from_eval(params: &EvalParams<T>) -> Self { Self { danger_curvature: params.w_king_danger } }
}

#[cfg(feature = "dataset")]
impl CombinerParams<f64> {
    #[inline(always)]
    pub(crate) fn from_values(values: &[f64]) -> Self { Self { danger_curvature: values[LAYOUT.king_danger_offset] } }
}

/// MG/EG interpolation, truncated toward zero after summing both contributions.
#[inline(always)]
pub(crate) fn taper<T: EvalMath<Scalar = T>>(mg: T, eg: T, phase: T) -> T {
    let total = T::from_i32(TOTAL_PHASE);
    ((mg * phase + eg * (total - phase)) / total).trunc()
}

/// Tapering separately would round each contribution before summing.
#[inline(always)]
pub(super) fn safety_block<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T, params: &CombinerParams<T>) -> T {
    let curvature = params.danger_curvature;
    // Curving the pressure difference would break color symmetry.
    let danger = curve_danger(buckets.pressure_us, curvature) - curve_danger(buckets.pressure_them, curvature);
    taper(buckets.safety_us - buckets.safety_them - danger + buckets.xray, T::zero(), phase)
}

#[inline(always)]
pub(crate) fn combine<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T, params: &CombinerParams<T>) -> T {
    let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
    buckets.material_psqt + buckets.mobility + bonus + safety_block(buckets, phase, params)
}

/// `d_score` is ∂L/∂score. Adds combiner-parameter gradients to `grads`.
/// Truncation uses the same straight-through derivative as the autograd path.
#[cfg(feature = "dataset")]
#[inline]
pub(crate) fn backward(
    buckets: &Accumulators<f64>,
    phase: f64,
    params: &CombinerParams<f64>,
    d_score: f64,
    grads: &mut [f64],
) -> BucketUpstreams {
    let t_mg = phase / f64::from(TOTAL_PHASE);
    let t_eg = 1.0 - t_mg;
    let taper_pair = TaperPair { d_mg: d_score * t_mg, d_eg: d_score * t_eg };
    let d_mg = taper_pair.d_mg;
    let scale = f64::from(DANGER_SCALE);

    let slope = |p: f64| 1.0 + 2.0 * p * params.danger_curvature / scale;
    let king_safety = KingSafetyUpstream {
        safety: d_mg,
        pressure_us: -d_mg * slope(buckets.pressure_us),
        pressure_them: d_mg * slope(buckets.pressure_them),
    };

    let (us, them) = (buckets.pressure_us, buckets.pressure_them);
    grads[LAYOUT.king_danger_offset] += d_mg * (them * them - us * us) / scale;

    BucketUpstreams { material_psqt: taper_pair, mobility: taper_pair, bonus: taper_pair, king_safety, xray: d_mg }
}

#[inline(always)]
fn curve_danger<T: EvalMath<Scalar = T>>(pressure: T, curvature: T) -> T {
    pressure + ((pressure * pressure * curvature) / T::from_i32(DANGER_SCALE)).trunc()
}
