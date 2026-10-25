#[cfg(test)] pub(crate) mod dual;
pub(crate) mod traits;

pub(crate) use traits::EvalMath;
#[cfg(feature = "dataset")] pub(crate) use traits::EvalVec8;

#[cfg(test)] mod tests;
