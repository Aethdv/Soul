//! Autograd: the differentiation layer the tuner shares with the eval.
//!
//! The same evaluation code runs at zero cost on plain integers and SIMD vectors
//! for search, and on `DualNode` for tuning, where forward-mode dual numbers carry
//! each value's partials alongside it. One forward pass yields the exact gradient
//! with respect to every parameter, no tape and no backward pass.

pub mod dual;
pub mod traits;

pub use dual::DualNode;
pub use traits::{EnvVec4, EnvVec8, EvalMath};

#[cfg(test)] mod tests;
