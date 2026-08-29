//! SIMD predication masks for branchless operations.

use core::{arch::x86_64::*, ops::Not};

use super::*;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Mask(pub __m256i);

impl Mask {
    #[inline(always)]
    pub fn all_true() -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsics. Hardware support guaranteed by target features.
        unsafe {
            let z = _mm256_setzero_si256();
            Self(_mm256_cmpeq_epi64(z, z))
        }
    }

    #[inline(always)]
    pub fn all_false() -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm256_setzero_si256() })
    }

    /// Returns true if any lane is set.
    #[inline(always)]
    pub fn any(self) -> bool {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
    }

    /// Returns true if no lane is set.
    #[inline(always)]
    pub fn none(self) -> bool {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        unsafe { _mm256_testz_si256(self.0, self.0) != 0 }
    }

    /// Branchless select; mask ? a : b
    #[inline(always)]
    pub fn select(self, a: U64x4, b: U64x4) -> U64x4 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        U64x4(unsafe { _mm256_blendv_epi8(b.0, a.0, self.0) })
    }

    #[inline(always)]
    pub fn movemask_epi8(self) -> u32 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        unsafe { _mm256_movemask_epi8(self.0) as u32 }
    }
}

impl Not for Mask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { self ^ Self::all_true() }
}

binops! { Mask:
    BitAnd bitand   _mm256_and_si256,
    BitOr  bitor    _mm256_or_si256,
    BitXor bitxor   _mm256_xor_si256,
}
