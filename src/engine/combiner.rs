//! Evaluation score combination — collapses per-bucket term outputs
//! into the final scalar eval.
//!
//! # Architecture
//!
//! Two stages:
//! 1) Terms fill [`Accumulators`] buckets.
//!
//! 2) A [`Combiner`] collapses them. Separating these lets the tuner propagate one
//!    upstream derivative per bucket through each term's scatter,
//!    rather than baking the combiner shape into every term.
//!
//! # Notes
//!
//! The current combiner ([`LinearCombiner`]) sums pre-tapered buckets
//! and applies a phase taper only to the king-safety block. Future
//! combiners (sigmoid danger, winnable eg-scale) implement [`Combiner`]
//! without touching any term.

use crate::{
    core::defs::TOTAL_PHASE,
    engine::{
        autograd::EvalMath,
        term::{BucketUpstreams, TaperPair},
    },
};

/// Per-bucket score accumulators filled by the term layer before combination.
///
/// Fields are either **pre-tapered** (`mg_eg`, `mobility`, `bonus`) — the mg/eg blend
/// happened inside the term — or **raw** (`safety_us`, `safety_them`, `xray`), tapered
/// together by the combiner as a single `(us - them + xray) * phase / TOTAL_PHASE` block.
pub struct Accumulators<T: EvalMath> {
    /// Material + PSQT, read from the SIMD accumulator.
    pub mg_eg: T,
    /// Mobility differential, blended inside the SIMD lanes to preserve vectorization.
    pub mobility: T,
    /// Simple linear bonuses; each term adds its own `mg * phase + eg * eg_phase` share.
    pub bonus: T,
    /// Raw king-safety score for the side to move.
    pub safety_us: T,
    /// Raw king-safety score for the opponent.
    pub safety_them: T,
    /// Raw x-ray king-ring differential; separate bucket so it can be rerouted independently.
    pub xray: T,
}

/// Strategy for collapsing [`Accumulators`] into a final evaluation scalar.
///
/// A combiner consumes only bucket values and combiner-owned params; it
/// never reads raw features. Non-linearities over bucket values (sigmoid
/// king danger, quadratic pressure, eg scale factors) belong here.
pub trait Combiner {
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T) -> T;

    /// Returns per-bucket upstream gradients given the loss derivative and game phase.
    /// Non-linear combiners also write their own param gradients into `grads`.
    fn backward(phase: f64, d_loss: f64, grads: &mut [f64]) -> BucketUpstreams;
}

pub struct LinearCombiner;

impl Combiner for LinearCombiner {
    #[inline(always)]
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T) -> T {
        let safety_diff = buckets.safety_us - buckets.safety_them + buckets.xray;
        let safety_tapered = (safety_diff * phase / T::from_i32(TOTAL_PHASE)).trunc();
        buckets.mg_eg + buckets.mobility + buckets.bonus + safety_tapered
    }

    #[inline]
    fn backward(phase: f64, d_loss: f64, _grads: &mut [f64]) -> BucketUpstreams {
        let t_mg = phase / f64::from(TOTAL_PHASE);
        let t_eg = 1.0 - t_mg;
        let taper = TaperPair { d_mg: d_loss * t_mg, d_eg: d_loss * t_eg };
        let safety_block = d_loss * t_mg;
        BucketUpstreams { mg_eg: taper, mobility: taper, bonus: taper, king_safety: safety_block, xray: safety_block }
    }
}
