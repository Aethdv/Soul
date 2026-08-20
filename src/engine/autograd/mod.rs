//! Autograd: the differentiation layer the tuner shares with the eval.

pub mod dual;
pub mod traits;

pub use dual::DualNode;
pub use traits::{EnvVec4, EnvVec8, EvalMath};

#[cfg(test)] mod tests;
