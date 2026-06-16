//! Per-term forward/backward trait and registration macro.
//!
//! Every eval term implements [`LinearTerm`]:
//!
//! - `apply` reads [`SharedFeatures`] and writes the term's bucket(s) on
//!   [`Accumulators`], the forward contribution.
//! - `scatter` consumes the combiner-derived upstream for those bucket(s)
//!   and writes parameter gradients, the backward pass.
//!
//! The [`register_terms!`] macro stitches impls into two unrolled
//! dispatch functions:
//!
//! - `apply_all_terms`: forward pass, called by
//!   [`crate::engine::eval::fill_accumulators`].
//! - `scatter_all_terms`: backward pass, called by the tuner's linear-
//!   grad tape.
//!
//! Adding a term is one macro line plus the impl, no plumbing edits
//! across `eval.rs` / `tape.rs`.
//!
//! `LinearTerm` assumes `∂bucket/∂param` is a pure feature coefficient:
//! the term is linear in its parameters. All non-linearities
//! (sigmoid gates, quadratic pressure curves, winnable scale factors)
//! belong in the [`crate::engine::combiner::Combiner`] layer, which
//! composes bucket values into the final score. Exotic per-term shapes
//! (param-param products, per-term activations) should land in a future
//! `NonlinearTerm` escape hatch rather than bending this trait.

use crate::engine::{
    autograd::EvalMath,
    combiner::Accumulators,
    eval::{EvalParams, SharedFeatures},
};

/// MG/EG upstream derivative pair for a bucket whose value is pre-tapered
/// inside the term (mobility's openness·phase blend, the PSQT accumulator's
/// `tapered` call). The combiner hands each term its MG and EG multipliers
/// pre-multiplied, so scatter code walks parameter slots in one pass
/// without re-deriving the phase fractions.
#[derive(Clone, Copy)]
pub struct TaperPair {
    pub d_mg: f64,
    pub d_eg: f64,
}

/// Per-term upstream derivatives emitted by the combiner's backward pass.
/// Each field is consumed by the term(s) [`register_terms!`] routes to it.
///
/// `king_safety` and `xray` carry equal values under [`LinearCombiner`]
/// today but sit in separate fields so a future combiner (sigmoid king
/// danger, independent xray scale) can diverge them without touching
/// any term.
///
/// `mg_eg` is consumed out-of-band by the tuner's PSQT / material
/// scatter, since PSQT lives in the accumulator rather than in any
/// registered term.
///
/// [`LinearCombiner`]: crate::engine::combiner::LinearCombiner
pub struct BucketUpstreams {
    pub mg_eg: TaperPair,
    pub mobility: TaperPair,
    pub bonus: TaperPair,
    pub king_safety: f64,
    pub xray: f64,
}

/// Parameter-linear evaluation term.
pub trait LinearTerm {
    /// Upstream derivative this term's scatter consumes. Typical shapes:
    ///
    /// - [`TaperPair`]: bucket is pre-tapered inside the term (e.g. mobility).
    /// - `f64`: bucket runs through one scalar combiner multiplier (e.g.
    ///   king-safety's single MG taper).
    type Upstream: Copy;

    /// Forward contribution; read features and params, write bucket(s).
    /// A term may write more than one bucket if they share the same feature
    /// extraction (e.g. `KingSafetyTerm` fills both `safety_us` and
    /// `safety_them` from one `SafetyMetrics::score` pass per side).
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>);

    /// Gradient scatter: `∂loss/∂bucket · ∂bucket/∂param` written into the
    /// term's owned slot range in `grads`.
    fn scatter(features: &SharedFeatures, upstream: Self::Upstream, grads: &mut [f64]);
}

/// Expand to `apply_all_terms` and `scatter_all_terms` dispatch functions
/// covering every registered term.
///
/// Each entry pairs a term type with the [`BucketUpstreams`] field whose
/// value is handed to that term's `scatter`; the compiler type-checks
/// the pairing against the term's associated `Upstream` type.
#[macro_export]
macro_rules! register_terms {
    ( $( $term:path => $upstream_field:ident ),* $(,)? ) => {
        #[inline]
        pub fn apply_all_terms<T: $crate::engine::autograd::EvalMath<Scalar = T>>(
            features: &$crate::engine::eval::SharedFeatures,
            params: &$crate::engine::eval::EvalParams<T>,
            phase: T,
            acc: &mut $crate::engine::combiner::Accumulators<T>,
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::apply::<T>(features, params, phase, acc); )*
        }

        #[inline]
        pub fn scatter_all_terms(
            features: &$crate::engine::eval::SharedFeatures,
            upstreams: &$crate::engine::term::BucketUpstreams,
            grads: &mut [f64],
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::scatter(features, upstreams.$upstream_field, grads); )*
        }
    };
}
