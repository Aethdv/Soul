//! SSE (128-bit) SIMD intrinsics wrapper.
//!
//! Provides safely typed wrappers around `__m128i` for parallel evaluation math.
//!
//! # Safety
//! Builds without `target_feature = "avx2"` are rejected at compile time by
//! `weave/mod.rs`. AVX2 is a superset of SSE, so all SSE intrinsics here
//! are available on any binary that reaches this module.

use core::{
    arch::x86_64::*,
    ops::{Neg, Shr},
};
use std::mem;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I16x8(pub __m128i);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct I32x4(pub __m128i);

impl I16x8 {
    #[inline(always)]
    pub fn new(v: [i16; 8]) -> Self {
        // SAFETY: The array is passed by value and lives on the stack. The pointer is valid for 16 bytes.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set1_epi16(v) })
    }

    /// Loads 8 contiguous i16s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn load(ptr: *const i16) -> Self {
        // SAFETY: The caller of this unsafe function must guarantee the pointer is valid for 16 bytes.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    /// Stores 8 contiguous i16s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut i16) {
        // SAFETY: The caller of this unsafe function must guarantee the pointer is valid for 16 bytes.
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    pub fn to_array(self) -> [i16; 8] {
        // SAFETY: Transmuting a 128-bit SIMD vector to an array of 8x 16-bit integers.
        // The sizes match exactly and alignment constraints are satisfied.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    pub fn extract<const N: i32>(self) -> i16 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        unsafe { _mm_extract_epi16::<N>(self.0) as i16 }
    }

    #[inline(always)]
    pub fn adds(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_adds_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn subs(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_subs_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_max_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_min_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn mullo(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn mulhi(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_mulhi_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn madd(self, rhs: Self) -> I32x4 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I32x4(unsafe { _mm_madd_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn clamp_relu(self, ceil: Self) -> Self { self.max(Self::zero()).min(ceil) }

    #[inline(always)]
    pub fn srai<const N: i32>(self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        Self(unsafe { _mm_srai_epi16::<N>(self.0) })
    }

    /// Extends the first 4 signed i16 lanes to i32 and returns them as a I32x4.
    /// Uses `vpmovsxwd` (SSE4.1).
    #[inline(always)]
    pub fn load_i32_4(self) -> I32x4 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I32x4(unsafe { _mm_cvtepi16_epi32(self.0) })
    }
}

impl From<[i16; 8]> for I16x8 {
    #[inline(always)]
    fn from(arr: [i16; 8]) -> Self { Self::new(arr) }
}

impl I32x4 {
    #[inline(always)]
    pub const fn new(v: [i32; 4]) -> Self {
        // SAFETY: The array is passed by value and lives on the stack. The pointer is valid for 16 bytes.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set1_epi32(v) })
    }

    #[inline(always)]
    pub fn from_lanes(a: i32, b: i32, c: i32, d: i32) -> Self {
        // _mm_set_epi32 takes lanes in reverse (3, 2, 1, 0); re-reversed here so a..d map to lanes 0..3.
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set_epi32(d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 4]) -> Self { Self::new(arr) }

    /// Horizontal sum: unpack-add then shuffle-add, dodging the slower `_mm_hadd_epi32`.
    #[inline]
    pub fn reduce_sum(self) -> i32 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        unsafe {
            let v = self.0;
            let hi64 = _mm_unpackhi_epi64(v, v);
            let sum64 = _mm_add_epi32(v, hi64);
            let hi32 = _mm_shuffle_epi32::<0b_00_00_00_01>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);
            _mm_cvtsi128_si32(sum32)
        }
    }

    /// Pack to i16 with signed saturation.
    ///
    /// Relies on the 128-bit `_mm_packs_epi32` appending `hi`'s 4 lanes after
    /// `self`'s 4, which keeps the [MG, EG] and [Diff, Diff] lanes aligned for the
    /// horizontal dot product in `evaluate_score_diff`.
    ///
    /// Porting trap: the 256-bit `_mm256_packs_epi32` packs each 128-bit lane
    /// independently instead of appending, emitting `[A0..3, B0..3, A4..7, B4..7]`
    /// rather than the sequential `[A0..7, B0..7]` and wrecking the dot-product order.
    /// A port has to follow it with `_mm256_permute4x64_epi64` to restore sequential lanes.
    #[inline(always)]
    pub fn pack_i16(self, hi: Self) -> I16x8 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I16x8(unsafe { _mm_packs_epi32(self.0, hi.0) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        // SAFETY: Transmuting a 128-bit SIMD vector to an array of 4x 32-bit integers.
        // The sizes match exactly and alignment constraints are satisfied.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    pub fn extract<const N: i32>(self) -> i32 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        unsafe { _mm_extract_epi32::<N>(self.0) }
    }

    #[inline(always)]
    pub fn srai<const N: i32>(self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        Self(unsafe { _mm_srai_epi32::<N>(self.0) })
    }
}

impl From<[i32; 4]> for I32x4 {
    #[inline(always)]
    fn from(arr: [i32; 4]) -> Self { Self::new(arr) }
}

impl Shr<i32> for I32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        // _mm_srai_epi32 requires an immediate, so we can't use it directly if rhs is variable.
        // But for mobility usage, rhs is a constant 10.
        // We act like portable_simd and use _mm_sra_epi32 which takes an XMM count.
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(_mm_sra_epi32(self.0, count))
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F32x4(pub __m128);

impl F32x4 {
    #[inline(always)]
    pub fn zero() -> Self { Self(unsafe { _mm_setzero_ps() }) }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(unsafe { _mm_set1_ps(v) }) }

    /// Load 4 contiguous f32s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f32) -> Self { Self(unsafe { _mm_loadu_ps(ptr) }) }

    /// Store 4 contiguous f32s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f32) { unsafe { _mm_storeu_ps(ptr, self.0) }; }
}

impl Neg for F32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_sub_ps(_mm_setzero_ps(), self.0) }) }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct F64x2(pub __m128d);

impl F64x2 {
    #[inline(always)]
    pub fn zero() -> Self { Self(unsafe { _mm_setzero_pd() }) }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self(unsafe { _mm_set1_pd(v) }) }

    /// Load 2 contiguous f64s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f64) -> Self { Self(unsafe { _mm_loadu_pd(ptr) }) }

    /// Store 2 contiguous f64s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f64) { unsafe { _mm_storeu_pd(ptr, self.0) }; }
}

impl Neg for F64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm_sub_pd(_mm_setzero_pd(), self.0) }) }
}

binops! { I16x8:
    Add     add      _mm_add_epi16,
    Sub     sub      _mm_sub_epi16,
    BitAnd  bitand   _mm_and_si128,
    BitOr   bitor    _mm_or_si128,
    BitXor  bitxor   _mm_xor_si128,
}

assign_ops! { I16x8:
    AddAssign add_assign    +,
    SubAssign sub_assign    -,
}

binops! { I32x4:
    Add add   _mm_add_epi32,
    Sub sub   _mm_sub_epi32,
    Mul mul   _mm_mullo_epi32,
}

assign_ops! { I32x4:
    MulAssign mul_assign    *,
}

binops! { F32x4:
    Add add   _mm_add_ps,
    Sub sub   _mm_sub_ps,
    Mul mul   _mm_mul_ps,
    Div div   _mm_div_ps,
}

assign_ops! { F32x4:
    AddAssign add_assign    +,
    SubAssign sub_assign    -,
    MulAssign mul_assign    *,
    DivAssign div_assign    /,
}

binops! { F64x2:
    Add add   _mm_add_pd,
    Sub sub   _mm_sub_pd,
    Mul mul   _mm_mul_pd,
    Div div   _mm_div_pd,
}

assign_ops! { F64x2:
    AddAssign add_assign    +,
    SubAssign sub_assign    -,
    MulAssign mul_assign    *,
    DivAssign div_assign    /,
}
