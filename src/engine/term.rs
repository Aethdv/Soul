//! Per-term forward/backward trait and registration macro.
//!
//! The [`LinearTerm`] trait connects a term's forward pass (`apply`) to its
//! gradient pass (`scatter`). [`register_terms!`] collects all impls into
//! `apply_all_terms` / `scatter_all_terms` dispatch. Adding a term is one
//! macro line plus the impl. No plumbing edits across files.

use crate::engine::{
    autograd::EvalMath,
    combiner::Accumulators,
    eval::{EvalParams, SharedFeatures},
};

/// MG/EG scalars pre-multiplied by the combiner so scatter skips the taper split.
#[derive(Clone, Copy)]
pub struct TaperPair {
    pub d_mg: f64,
    pub d_eg: f64,
}

/// [`register_terms!`] dispatches each field to the term(s) that consume it.
/// `mg_eg` is read by the tuner's PSQT/material scatter; PSQT is not a registered term.
pub struct BucketUpstreams {
    pub mg_eg: TaperPair,
    pub mobility: TaperPair,
    pub bonus: TaperPair,
    pub king_safety: KingSafetyUpstream,
    pub xray: f64,
}

/// The danger halves are per side because the combiner curves each one before
/// differencing them, so `∂block/∂danger` depends on which king it belongs to.
#[derive(Clone, Copy)]
pub struct KingSafetyUpstream {
    pub shelter: f64,
    pub danger_us: f64,
    pub danger_them: f64,
}

/// Provides [`LinearTerm::Input`] for a term.
/// Implement on [`SharedFeatures`] (engine) and [`FeatureRecord`] (tuner).
pub trait TermSource<Term> {
    type Input;
    fn extract(&self) -> Self::Input;
}

/// Forward `apply` and backward `scatter`.
/// `Input` connects the term to its [`TermSource`]; [`register_terms!`]
/// derives it so callers never spell the type.
pub trait LinearTerm {
    /// What the combiner's backward pass hands to this term's scatter.
    type Upstream: Copy;
    /// Every [`TermSource`] for this term provides exactly `Self::Input`.
    type Input;

    /// Write one or more buckets on [`Accumulators`] from features and params.
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>);

    /// The same buckets from extracted features, for sources with no board.
    /// Reads the flat vector at the layout offsets [`Self::scatter`] writes.
    fn apply_input(input: Self::Input, values: &[f64], phase: f64, acc: &mut Accumulators<f64>);

    /// Write parameter gradients from extracted features.
    fn scatter(input: Self::Input, upstream: Self::Upstream, grads: &mut [f64]);
}

/// Each entry maps a term type to its [`BucketUpstreams`] field;
/// `<LinearTerm::Input>` constrains the source side, so callers never write the type.
#[macro_export]
macro_rules! register_terms {
    (
        $( $term:path => $upstream:ident ),* $(,)?
    ) => {
        #[inline(always)]
        pub fn apply_all_terms<T: $crate::engine::autograd::EvalMath<Scalar = T>>(
            features: &$crate::engine::eval::SharedFeatures,
            params: &$crate::engine::eval::EvalParams<T>,
            phase: T,
            acc: &mut $crate::engine::combiner::Accumulators<T>,
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::apply::<T>(features, params, phase, acc); )*
        }

        /// Forward pass for a source that isn't a board, same terms in the same order.
        #[inline(always)]
        pub fn apply_all_inputs<S>(
            source: &S,
            values: &[f64],
            phase: f64,
            acc: &mut $crate::engine::combiner::Accumulators<f64>,
        ) where
            $( S: $crate::engine::term::TermSource<$term, Input = <$term as $crate::engine::term::LinearTerm>::Input> ),*
        {
            $(
                <$term as $crate::engine::term::LinearTerm>::apply_input(
                    $crate::engine::term::TermSource::<$term>::extract(source),
                    values,
                    phase,
                    acc,
                );
            )*
        }

        /// Input type inferred from each term's [`LinearTerm::Input`].
        #[inline(always)]
        pub fn scatter_all_terms<S>(
            source: &S,
            upstreams: &$crate::engine::term::BucketUpstreams,
            grads: &mut [f64],
        ) where
            $( S: $crate::engine::term::TermSource<$term, Input = <$term as $crate::engine::term::LinearTerm>::Input> ),*
        {
            $(
                <$term as $crate::engine::term::LinearTerm>::scatter(
                    $crate::engine::term::TermSource::<$term>::extract(source),
                    upstreams.$upstream,
                    grads,
                );
            )*
        }
    };
}
