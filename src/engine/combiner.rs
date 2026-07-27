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

/// Pressure ceilings at 465 (`weak` maxes at 8, `ATTACKER[5]` at 582) and an
/// ordinary attack sits at 50 to 150, so this divisor is worth ~10% there and
/// ~45% at the ceiling: near-nothing until a king is genuinely under siege.
const KING_DANGER_DIVISOR: i32 = 1024;

/// The one non-linearity in the eval: pressure accelerates instead of scaling
/// with the weight table alone.
#[inline(always)]
fn king_danger<T: EvalMath<Scalar = T>>(pressure: T) -> T {
    pressure + ((pressure * pressure) / T::from_i32(KING_DANGER_DIVISOR)).trunc()
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
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T) -> T;

    /// Returns per-bucket upstream gradients given the loss derivative and game phase.
    /// Non-linear combiners also write their own param gradients into `grads`, and
    /// read `buckets`, since their partials depend on the values they collapse.
    fn backward(buckets: &Accumulators<f64>, phase: f64, d_loss: f64, grads: &mut [f64]) -> BucketUpstreams;
}

pub struct LinearCombiner;

impl Combiner for LinearCombiner {
    #[inline(always)]
    fn forward<T: EvalMath<Scalar = T>>(buckets: &Accumulators<T>, phase: T) -> T {
        let danger = king_danger(buckets.danger_us) - king_danger(buckets.danger_them);
        let safety_diff = buckets.safety_us - buckets.safety_them - danger + buckets.xray;
        let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
        let safety = taper(safety_diff, T::zero(), phase);

        buckets.mg_eg + buckets.mobility + bonus + safety
    }

    #[inline]
    fn backward(buckets: &Accumulators<f64>, phase: f64, d_loss: f64, _grads: &mut [f64]) -> BucketUpstreams {
        let t_mg = phase / f64::from(TOTAL_PHASE);
        let t_eg = 1.0 - t_mg;
        let taper = TaperPair { d_mg: d_loss * t_mg, d_eg: d_loss * t_eg };
        let safety_block = d_loss * t_mg;

        // d/dp of `p + p²/D`, and the block subtracts our danger and adds theirs.
        let slope = |p: f64| 1.0 + 2.0 * p / f64::from(KING_DANGER_DIVISOR);
        let king_safety = KingSafetyUpstream {
            shelter: safety_block,
            danger_us: -safety_block * slope(buckets.danger_us),
            danger_them: safety_block * slope(buckets.danger_them),
        };

        BucketUpstreams { mg_eg: taper, mobility: taper, bonus: taper, king_safety, xray: safety_block }
    }
}
