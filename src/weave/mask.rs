//! SIMD predication masks for branchless operations.

#![rustfmt::skip]
// Typed SIMD wrappers — not CamelCase by design.
#![allow(non_camel_case_types)]

use core::arch::x86_64::*;
use core::ops::{BitAnd, BitOr, BitXor, Not};
use super::*;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct VMask(pub __m256i);

// ──────── VMask — Predication Mask ────────

impl VMask {
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
    pub fn select(self, a: Vu64x4, b: Vu64x4) -> Vu64x4 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Vu64x4(unsafe { _mm256_blendv_epi8(b.0, a.0, self.0) })
    }

    #[inline(always)]
    pub fn movemask_epi8(self) -> u32 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        unsafe { _mm256_movemask_epi8(self.0) as u32 }
    }
}

impl BitAnd for VMask {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
    }
}

impl BitOr for VMask {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
    }
}

impl Not for VMask {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self {
        self ^ Self::all_true()
    }
}

impl BitXor for VMask {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
    }
}
