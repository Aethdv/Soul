//! Hardware-specific SIMD intrinsics and vectorization routines.

#[cfg(not(target_feature = "avx2"))]
compile_error!(
    "Soul architecture relies on AVX2 SIMD intrinsics. \
     You must compile with the `avx2` target feature enabled. \
     Use the provided Makefile (e.g., `make native` or `make pgo`) instead of `cargo build`."
);

mod avx2;
mod fills;
mod mask;
mod sse;

pub use avx2::*;
pub use fills::*;
pub use mask::*;
pub use sse::*;
