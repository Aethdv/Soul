//! [`LinearTerm`] defines one eval term's forward pass and gradient scatter;
//! `register_terms!` generates the dispatch over the registered impls.

#![cfg_attr(not(feature = "dataset"), expect(dead_code, reason = "the gradient half of each term is the tuner's"))]

use crate::engine::{
    autograd::EvalMath,
    combiner::Accumulators,
    eval::{EvalParams, SharedFeatures},
};

#[derive(Clone, Copy)]
pub(crate) struct TaperPair {
    pub(crate) d_mg: f64,
    pub(crate) d_eg: f64,
}

pub(crate) struct BucketUpstreams {
    #[cfg_attr(
        all(feature = "dataset", not(test)),
        expect(dead_code, reason = "Material and PSQT scatter from the phase weights; only the gradient oracle reads this")
    )]
    pub(crate) material_psqt: TaperPair,
    pub(crate) mobility: TaperPair,
    pub(crate) bonus: TaperPair,
    pub(crate) king_safety: KingSafetyUpstream,
    pub(crate) xray: f64,
}

#[derive(Clone, Copy)]
pub(crate) struct KingSafetyUpstream {
    pub(crate) safety: f64,
    pub(crate) pressure_us: f64,
    pub(crate) pressure_them: f64,
}

pub(crate) trait TermSource<Term> {
    type Input;
    fn extract(&self) -> Self::Input;
}

pub(crate) trait LinearTerm {
    type Upstream: Copy;
    type Input;
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>);
    fn apply_input(input: Self::Input, values: &[f64], phase: f64, acc: &mut Accumulators<f64>);
    fn scatter(input: Self::Input, upstream: Self::Upstream, grads: &mut [f64]);
}

#[macro_export]
macro_rules! register_terms {
    (
        $( $term:path => $upstream:ident ),* $(,)?
    ) => {
        #[inline(always)]
        pub(crate) fn apply_all_terms<T: $crate::engine::autograd::EvalMath<Scalar = T>>(
            features: &$crate::engine::eval::SharedFeatures,
            params: &$crate::engine::eval::EvalParams<T>,
            phase: T,
            acc: &mut $crate::engine::combiner::Accumulators<T>,
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::apply::<T>(features, params, phase, acc); )*
        }

        #[cfg(feature = "dataset")]
        #[inline(always)]
        pub(crate) fn apply_all_inputs<S>(
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

        #[cfg(feature = "dataset")]
        #[inline(always)]
        pub(crate) fn scatter_all_terms<S>(
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
