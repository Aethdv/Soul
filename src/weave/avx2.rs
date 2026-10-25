//! 256-bit vectors. The build is gated on AVX2 in `weave/mod.rs`, so every intrinsic here
//! exists on any binary that reaches this module.

use core::{
    arch::x86_64::*,
    ops::{Neg, Not},
};

use super::*;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct U64x4(pub(crate) __m256i);

impl U64x4 {
    #[inline(always)]
    pub(crate) fn splat(v: u64) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_set1_epi64x(v as i64) })
    }

    #[inline(always)]
    fn ones() -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        unsafe {
            let z = _mm256_setzero_si256();
            Self(_mm256_cmpeq_epi64(z, z))
        }
    }

    #[inline(always)]
    pub(crate) fn from_lanes(a: u64, b: u64, c: u64, d: u64) -> Self {
        // _mm256_set_epi64x takes lanes in reverse (3, 2, 1, 0); re-reversed here so a..d map to lanes 0..3.
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_set_epi64x(d as i64, c as i64, b as i64, a as i64) })
    }

    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub(crate) unsafe fn load(ptr: *const u64) -> Self {
        // SAFETY: Caller guarantees the pointer is valid for an unaligned 256-bit read.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    pub(crate) fn shl<const N: i32>(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        Self(unsafe { _mm256_slli_epi64::<N>(self.0) })
    }

    #[inline(always)]
    pub(crate) fn shr<const N: i32>(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        Self(unsafe { _mm256_srli_epi64::<N>(self.0) })
    }

    #[inline(always)]
    pub(crate) fn andnot(self, rhs: Self) -> Self {
        // Computes (!self) & rhs, following the Intel intrinsic.
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_andnot_si256(self.0, rhs.0) })
    }

    #[inline(always)]
    pub(crate) fn cmp_eq(self, rhs: Self) -> Mask {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Mask(unsafe { _mm256_cmpeq_epi64(self.0, rhs.0) })
    }

    /// Vertical popcount: PSHUFB nibble LUT + SAD trick.
    /// Returns a U64x4 where each lane holds the popcount of the corresponding input lane.
    #[rustfmt::skip]
    #[inline]
    pub(crate) fn popcount(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        unsafe {
            let lut = _mm256_setr_epi8(
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
                0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            );

            let mask_lo = _mm256_set1_epi8(0x0F);
            let zero = _mm256_setzero_si256();

            let lo_nibbles = _mm256_and_si256(self.0, mask_lo);
            let hi_nibbles = _mm256_and_si256(_mm256_srli_epi16::<4>(self.0), mask_lo);

            let lo_counts = _mm256_shuffle_epi8(lut, lo_nibbles);
            let hi_counts = _mm256_shuffle_epi8(lut, hi_nibbles);

            let byte_counts = _mm256_add_epi8(lo_counts, hi_counts);

            Self(_mm256_sad_epu8(byte_counts, zero))
        }
    }

    #[inline(always)]
    pub(crate) fn extract<const IDX: i32>(self) -> u64 {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        unsafe { _mm256_extract_epi64::<IDX>(self.0) as u64 }
    }
}

impl Not for U64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { self ^ Self::ones() }
}

#[cfg_attr(not(test), expect(dead_code, reason = "only the dual-number oracle uses it"))]
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct F32x8(pub(crate) __m256);

#[cfg_attr(not(test), expect(dead_code, reason = "only the dual-number oracle uses it"))]
impl F32x8 {
    #[inline(always)]
    pub(crate) fn zero() -> Self { Self(unsafe { _mm256_setzero_ps() }) }

    #[inline(always)]
    pub(crate) fn splat(v: f32) -> Self { Self(unsafe { _mm256_set1_ps(v) }) }

    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub(crate) unsafe fn loadu(ptr: *const f32) -> Self { Self(unsafe { _mm256_loadu_ps(ptr) }) }

    /// # Safety
    /// `ptr` must be valid for writing 32 bytes (256 bits).
    #[inline(always)]
    pub(crate) unsafe fn storeu(self, ptr: *mut f32) { unsafe { _mm256_storeu_ps(ptr, self.0) }; }
}

impl Neg for F32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) }) }
}

impl From<[f32; 8]> for F32x8 {
    #[inline(always)]
    fn from(arr: [f32; 8]) -> Self { unsafe { Self::loadu(arr.as_ptr()) } }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct F64x4(pub(crate) __m256d);

#[expect(dead_code, reason = "completeness API: lane operations stay with their type")]
impl F64x4 {
    #[inline(always)]
    fn zero() -> Self { Self(unsafe { _mm256_setzero_pd() }) }

    #[inline(always)]
    pub(crate) fn splat(v: f64) -> Self { Self(unsafe { _mm256_set1_pd(v) }) }

    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub(crate) unsafe fn loadu(ptr: *const f64) -> Self { Self(unsafe { _mm256_loadu_pd(ptr) }) }

    /// # Safety
    /// `ptr` must be valid for writing 32 bytes (256 bits).
    #[inline(always)]
    pub(crate) unsafe fn storeu(self, ptr: *mut f64) { unsafe { _mm256_storeu_pd(ptr, self.0) }; }
}

impl Neg for F64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) }) }
}

impl From<[f64; 4]> for F64x4 {
    #[inline(always)]
    fn from(arr: [f64; 4]) -> Self { unsafe { Self::loadu(arr.as_ptr()) } }
}

impl From<F64x4> for [f64; 4] {
    #[inline(always)]
    fn from(v: F64x4) -> Self {
        let mut arr = [0.0; 4];
        unsafe { v.storeu(arr.as_mut_ptr()) };
        arr
    }
}

binops! { U64x4:
    Add     add     _mm256_add_epi64,
    Sub     sub     _mm256_sub_epi64,
    BitAnd  bitand  _mm256_and_si256,
    BitOr   bitor   _mm256_or_si256,
    BitXor  bitxor  _mm256_xor_si256,
}

assign_ops! { U64x4:
    BitAndAssign bitand_assign  &,
    BitOrAssign  bitor_assign   |,
    BitXorAssign bitxor_assign  ^,
}

binops! { F32x8:
    Add add   _mm256_add_ps,
    Sub sub   _mm256_sub_ps,
    Mul mul   _mm256_mul_ps,
    Div div   _mm256_div_ps,
}

assign_ops! { F32x8:
    AddAssign add_assign    +,
    SubAssign sub_assign    -,
    MulAssign mul_assign    *,
    DivAssign div_assign    /,
}

binops! { F64x4:
    Add add   _mm256_add_pd,
    Sub sub   _mm256_sub_pd,
    Mul mul   _mm256_mul_pd,
    Div div   _mm256_div_pd,
}

assign_ops! { F64x4:
    AddAssign add_assign    +,
    SubAssign sub_assign    -,
    MulAssign mul_assign    *,
    DivAssign div_assign    /,
}
