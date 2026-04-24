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

use crate::{core::defs::TOTAL_PHASE, engine::autograd::EvalMath};

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
        buckets.mg_eg + buckets.mobility + safety_tapered
    }
}
