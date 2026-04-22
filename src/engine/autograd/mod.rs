//! The Autograd framework for tuning.
//!
//! This module allows the engine's internal evaluation logic to run at zero-cost
//! using standard integers or SIMD vectors, while doubling as a dynamically
//! recorded computational graph (`AutogradNode`) when tuning via backpropagation.

pub mod dual;
pub mod traits;

pub use dual::DualNode;
pub use traits::{EnvVec4, EnvVec8, EvalMath};

/// A swappable evaluation node type.
///
/// PLANNED: This is an abstraction hook for future differentiable
/// search experiments. By monomorphizing the search loop over `TraceNode`, we
/// can run the same negamax code in "play mode" (i32) or "tune mode" (DualNode),
/// allowing the engine to calculate gradients of the search result itself with
/// respect to its pruning and reduction constants.
#[cfg(feature = "searchtune")]
pub type TraceNode = DualNode;

#[cfg(not(feature = "searchtune"))]
pub type TraceNode = i32;

#[cfg(test)] mod tests;
