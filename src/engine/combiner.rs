//! Score combination layer.
//!
//! # Architecture
//!
//! Evaluation runs in two stages:
//!
//! 1. Terms produce per-bucket scores, written into [`Accumulators`].
//!    Each term reads `SharedFeatures` and writes one or more bucket fields.
//! 2. Combiner reads the filled accumulators and produces the final
//!    single-scalar evaluation. All non-linearities beyond the per-term
//!    internal shape (mobility's openness/phase blend, king safety's
//!    attacker-pressure curve) live here.
//!
//! Splitting these stages is what lets the tuner propagate a single
//! upstream derivative per bucket through each term's scatter, rather
//! than baking the combiner shape into every term.
//!
//! Soul's current combiner ([`LinearCombiner`]) is linear in the tunable
//! parameters; the only operation beyond summation is a multiplicative
//! taper that folds the raw king-safety block into the phase blend.
//! Future combiners (sigmoid over king danger, winnable eg-scale) slot
//! in by implementing [`Combiner`] without touching any term.

use crate::{
    core::defs::TOTAL_PHASE,
    engine::{
        autograd::EvalMath,
        term::{BucketUpstreams, TaperPair},
    },
};

/// Per-bucket evaluation scores, filled by the term layer prior to combination.
/// Every field is a score contribution in the same units;
/// either pre-tapered (mg_eg, mobility)
/// or raw awaiting combiner taper (safety_us, safety_them, xray).
pub struct Accumulators<T: EvalMath> {
    /// Tapered material + PSQT score, read straight from the SIMD accumulator.
    /// Pre-tapered because the accumulator owns the mg/eg blend via `EvalMath::tapered`.
    pub mg_eg: T,
    /// Tapered mobility differential.
    /// Pre-tapered because the openness interpolation and phase blend both happen inside
    /// `madd`-packed i16 SIMD lanes — extracting them into the combiner would lose the vectorization.
    pub mobility: T,
    /// Shared pre-tapered bucket for simple linear bonuses.
    /// Each such term `+=` its own `mg · phase + eg · eg_phase` contribution;
    /// scatter writes that term's own param slots.
    /// Lets new tapered bonuses land as one-line `register_terms!`
    /// additions rather than bucket expansions.
    pub bonus: T,
    /// Raw king-safety score from "us"'s perspective, untapered.
    /// Combiner applies `phase / TOTAL_PHASE` to the `us - them + xray`
    /// differential so the whole king-safety block shares one taper.
    pub safety_us: T,
    /// Raw king-safety score from "them"'s perspective, untapered.
    pub safety_them: T,
    /// X-ray king-ring differential, untapered.
    /// Sits in the same taper block as `safety_us`/`safety_them`.
    /// Separate bucket because its feature (orthogonal x-ray count)
    /// is unrelated to the safety metrics, so future terms can route
    /// elsewhere without disturbing the safety path.
    pub xray: T,
}

/// Strategy for collapsing [`Accumulators`] into a final evaluation scalar.
///
/// A combiner consumes only bucket values and combiner-owned params; it
/// never reads raw features. Non-linearities over bucket values (sigmoid
/// king danger, quadratic pressure, eg scale factors) belong here.
pub trait Combiner {
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T) -> T;

    /// Produce per-term upstream derivatives given the loss derivative
    /// (STM sign folded in) and the game phase. Future non-linear combiners
    /// also write gradients for their own tunable params into `grads`;
    /// [`LinearCombiner`] has none, so `grads` is unused here.
    fn backward(phase: f64, d_loss: f64, grads: &mut [f64]) -> BucketUpstreams;
}

/// Soul's current combiner — linear sum with a single multiplicative taper
/// over the king-safety block.
///
/// ```rust
/// final = mg_eg
///       + mobility
///       + trunc((safety_us − safety_them + xray) · phase / TOTAL_PHASE)
/// ```
///
/// "Linear" refers to parameter dependence; phase is position-derived
/// non-tunable data, so `safety_tapered` is a fixed fraction of a
/// parameter-linear sum.
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
