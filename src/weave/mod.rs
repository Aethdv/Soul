//! Hardware-specific SIMD intrinsics and vectorization routines.

#[cfg(not(target_feature = "avx2"))]
compile_error!(concat!(
    "Soul needs AVX2.\n",
    "\n",
    "    make native      builds for this machine's CPU\n",
    "    make pgo         the recommended release build\n",
    "\n",
    "Or set the flag directly:\n",
    "\n",
    "    RUSTFLAGS=\"-C target-cpu=native\" cargo build --release",
));

/// `impl Trait for Ty` straight from one intrinsic, one row per operator.
macro_rules! binops {
    ($ty:ident: $($trait:ident $method:ident $intrin:ident),+ $(,)?) => {
        $(
            impl core::ops::$trait for $ty {
                type Output = Self;
                #[inline(always)]
                fn $method(self, rhs: Self) -> Self {
                    // SAFETY: mod.rs gates the build on avx2, so every intrinsic here exists.
                    Self(unsafe { $intrin(self.0, rhs.0) })
                }
            }
        )+
    };
}

/// The op-assign forms, in terms of the operators above.
macro_rules! assign_ops {
    ($ty:ident: $($trait:ident $method:ident $op:tt),+ $(,)?) => {
        $(
            impl core::ops::$trait for $ty {
                #[inline(always)]
                fn $method(&mut self, rhs: Self) { *self = *self $op rhs; }
            }
        )+
    };
}

mod avx2;
mod fills;
mod mask;
mod sse;

pub use avx2::*;
pub use fills::*;
pub use mask::*;
pub use sse::*;
