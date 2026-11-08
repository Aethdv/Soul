//! [`LinearTerm`] defines one eval term's forward pass and gradient scatter;
//! `register_terms!` generates the dispatch over the registered impls.

use crate::engine::{
    autograd::EvalMath,
    combiner::Accumulators,
    eval::{EvalParams, SharedFeatures},
};
#[cfg(feature = "dataset")]
use crate::tools::dataset::FeatureRecord;

#[cfg(feature = "dataset")]
#[derive(Clone, Copy)]
pub(crate) struct TaperPair {
    pub(crate) d_mg: f64,
    pub(crate) d_eg: f64,
}

#[cfg(feature = "dataset")]
#[derive(Clone, Copy)]
pub(crate) struct KingSafetyUpstream {
    pub(crate) safety: f64,
    pub(crate) pressure_us: f64,
    pub(crate) pressure_them: f64,
}

#[cfg(feature = "dataset")]
pub(crate) struct BucketUpstreams {
    pub(crate) material_psqt: TaperPair,
    pub(crate) mobility: TaperPair,
    pub(crate) bonus: TaperPair,
    pub(crate) king_safety: KingSafetyUpstream,
    pub(crate) xray: f64,
}

pub(crate) trait LinearTerm {
    #[cfg(feature = "dataset")]
    type Upstream: Copy;

    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>);

    #[cfg(feature = "dataset")]
    fn apply_record(record: &FeatureRecord, values: &[f64], phase: f64, acc: &mut Accumulators<f64>);

    #[cfg(feature = "dataset")]
    fn scatter_record(record: &FeatureRecord, upstream: Self::Upstream, grads: &mut [f64]);

    #[cfg(all(test, feature = "dataset"))]
    fn scatter_shared(features: &SharedFeatures, upstream: Self::Upstream, grads: &mut [f64]);
}

#[macro_export]
macro_rules! register_terms {
    ($( $term:path => $upstream:ident ),* $(,)?) => {
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
        pub(crate) fn apply_all_records(
            record: &$crate::tools::dataset::FeatureRecord,
            values: &[f64],
            phase: f64,
            acc: &mut $crate::engine::combiner::Accumulators<f64>,
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::apply_record(record, values, phase, acc); )*
        }

        #[cfg(feature = "dataset")]
        #[inline(always)]
        pub(crate) fn scatter_all_records(
            record: &$crate::tools::dataset::FeatureRecord,
            upstreams: &$crate::engine::term::BucketUpstreams,
            grads: &mut [f64],
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::scatter_record(record, upstreams.$upstream, grads); )*
        }

        #[cfg(all(test, feature = "dataset"))]
        #[inline(always)]
        pub(crate) fn scatter_all_shared(
            features: &$crate::engine::eval::SharedFeatures,
            upstreams: &$crate::engine::term::BucketUpstreams,
            grads: &mut [f64],
        ) {
            $( <$term as $crate::engine::term::LinearTerm>::scatter_shared(features, upstreams.$upstream, grads); )*
        }
    };
}
