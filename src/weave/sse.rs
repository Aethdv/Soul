//! SSE (128-bit) SIMD intrinsics wrapper.
//!
//! Provides safely typed wrappers around `__m128i` for parallel evaluation math.

#![allow(non_camel_case_types)]

use core::{
    arch::x86_64::*,
    ops::{Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Shr, Sub, SubAssign},
};
use std::mem;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vi16x8(pub __m128i);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vi32x4(pub __m128i);

impl Vi16x8 {
    #[inline(always)]
    pub fn new(v: [i16; 8]) -> Self {
        // SAFETY: The array is passed by value and lives on the stack. The pointer is valid for 16 bytes.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
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
        // SAFETY: The lane index N is constrained by const generics. Hardware support is guaranteed.
        unsafe { _mm_extract_epi16::<N>(self.0) as i16 }
    }

    #[inline(always)]
    pub fn adds(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_adds_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn subs(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_subs_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_max_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_min_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn mullo(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn mulhi(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_mulhi_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn madd(self, rhs: Self) -> Vi32x4 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Vi32x4(unsafe { _mm_madd_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn clamp_relu(self, ceil: Self) -> Self {
        self.max(Self::zero()).min(ceil)
    }

    #[inline(always)]
    pub fn srai<const N: i32>(self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Const generic N ensures valid shift count.
        Self(unsafe { _mm_srai_epi16::<N>(self.0) })
    }

    /// Extends the first 4 signed i16 lanes to i32 and returns them as a Vi32x4.
    /// Uses `vpmovsxwd` (SSE4.1).
    #[inline(always)]
    pub fn load_i32_4(self) -> Vi32x4 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Vi32x4(unsafe { _mm_cvtepi16_epi32(self.0) })
    }
}

impl From<[i16; 8]> for Vi16x8 {
    #[inline(always)]
    fn from(arr: [i16; 8]) -> Self {
        Self::new(arr)
    }
}

impl Add for Vi16x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_add_epi16(self.0, rhs.0) })
    }
}

impl Sub for Vi16x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_sub_epi16(self.0, rhs.0) })
    }
}

impl AddAssign for Vi16x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for Vi16x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl BitAnd for Vi16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_and_si128(self.0, rhs.0) })
    }
}

impl BitOr for Vi16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_or_si128(self.0, rhs.0) })
    }
}

impl BitXor for Vi16x8 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_xor_si128(self.0, rhs.0) })
    }
}

impl Vi32x4 {
    #[inline(always)]
    pub const fn new(v: [i32; 4]) -> Self {
        // SAFETY: The array is passed by value and lives on the stack. The pointer is valid for 16 bytes.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub fn splat(v: i32) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_set1_epi32(v) })
    }

    #[inline(always)]
    pub fn from_lanes(a: i32, b: i32, c: i32, d: i32) -> Self {
        // NOTE: _mm_set_epi32 takes elements in reverse order (3, 2, 1, 0).
        // This wrapper re-reverses them so arguments map to lanes 0..3.
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_set_epi32(d, c, b, a) })
    }

    #[inline(always)]
    pub fn from_array(arr: [i32; 4]) -> Self {
        Self::new(arr)
    }

    /// Horizontal sum (2x hadd + extract)
    #[inline]
    pub fn reduce_sum(self) -> i32 {
        // SAFETY: Pure SIMD arithmetic intrinsics. Hardware support guaranteed by target features.
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
    /// NOTE: This relies on the 128-bit SSE behavior of `_mm_packs_epi32`,
    /// which cleanly appends the 4 lanes of `hi` to the 4 lanes of `self`.
    /// This keeps the [MG, EG] and [Diff, Diff] lanes correctly aligned
    /// for the subsequent horizontal dot product in `evaluate_score_diff`.
    ///
    /// WARNING: AVX2 PORTING TRAP
    /// If you upgrade this operation to AVX2 (256-bit `_mm256_packs_epi32`),
    /// the instruction interleaves per 128-bit lane rather than appending
    /// the full vectors sequentially! It will output `[A0, A1, B0, B1, A2, A3, B2, B3]`,
    /// instantly destroying the horizontal dot product math. You must issue an
    /// explicit `_mm256_permute4x64_epi64` after packing to restore sequential lane order.
    #[inline(always)]
    pub fn pack_i16(self, hi: Self) -> Vi16x8 {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Vi16x8(unsafe { _mm_packs_epi32(self.0, hi.0) })
    }

    #[inline(always)]
    pub fn to_array(self) -> [i32; 4] {
        // SAFETY: Transmuting a 128-bit SIMD vector to an array of 4x 32-bit integers.
        // The sizes match exactly and alignment constraints are satisfied.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    pub fn extract<const N: i32>(self) -> i32 {
        // SAFETY: The lane index N is constrained by const generics. Hardware support is guaranteed.
        unsafe { _mm_extract_epi32::<N>(self.0) }
    }

    #[inline(always)]
    pub fn srai<const N: i32>(self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Const generic N ensures valid shift count.
        Self(unsafe { _mm_srai_epi32::<N>(self.0) })
    }
}

impl From<[i32; 4]> for Vi32x4 {
    #[inline(always)]
    fn from(arr: [i32; 4]) -> Self {
        Self::new(arr)
    }
}

impl Add for Vi32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_add_epi32(self.0, rhs.0) })
    }
}

impl Sub for Vi32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_sub_epi32(self.0, rhs.0) })
    }
}

impl Mul for Vi32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        // SAFETY: Pure SIMD arithmetic intrinsic. Hardware support guaranteed by target features.
        Self(unsafe { _mm_mullo_epi32(self.0, rhs.0) })
    }
}

impl MulAssign for Vi32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Shr<i32> for Vi32x4 {
    type Output = Self;
    #[inline(always)]
    fn shr(self, rhs: i32) -> Self {
        // _mm_srai_epi32 requires an immediate, so we can't use it directly if rhs is variable.
        // But for mobility usage, rhs is a constant 10.
        // We act like portable_simd and use _mm_sra_epi32 which takes an XMM count.
        // SAFETY: Pure SIMD arithmetic intrinsics. Hardware support guaranteed by target features.
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(_mm_sra_epi32(self.0, count))
        }
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vf32x4(pub __m128);

impl Vf32x4 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm_setzero_ps() })
    }

    #[inline(always)]
    pub fn splat(v: f32) -> Self {
        Self(unsafe { _mm_set1_ps(v) })
    }

    /// Load 4 contiguous f32s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f32) -> Self {
        Self(unsafe { _mm_loadu_ps(ptr) })
    }

    /// Store 4 contiguous f32s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f32) {
        unsafe { _mm_storeu_ps(ptr, self.0) };
    }
}

impl Add for Vf32x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_add_ps(self.0, rhs.0) })
    }
}

impl AddAssign for Vf32x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vf32x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_sub_ps(self.0, rhs.0) })
    }
}

impl SubAssign for Vf32x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Vf32x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mul_ps(self.0, rhs.0) })
    }
}

impl MulAssign for Vf32x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Vf32x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { _mm_div_ps(self.0, rhs.0) })
    }
}

impl DivAssign for Vf32x4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Vf32x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_sub_ps(_mm_setzero_ps(), self.0) })
    }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vf64x2(pub __m128d);

impl Vf64x2 {
    #[inline(always)]
    pub fn zero() -> Self {
        Self(unsafe { _mm_setzero_pd() })
    }

    #[inline(always)]
    pub fn splat(v: f64) -> Self {
        Self(unsafe { _mm_set1_pd(v) })
    }

    /// Load 2 contiguous f64s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f64) -> Self {
        Self(unsafe { _mm_loadu_pd(ptr) })
    }

    /// Store 2 contiguous f64s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f64) {
        unsafe { _mm_storeu_pd(ptr, self.0) };
    }
}

impl Add for Vf64x2 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        Self(unsafe { _mm_add_pd(self.0, rhs.0) })
    }
}

impl AddAssign for Vf64x2 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for Vf64x2 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        Self(unsafe { _mm_sub_pd(self.0, rhs.0) })
    }
}

impl SubAssign for Vf64x2 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for Vf64x2 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        Self(unsafe { _mm_mul_pd(self.0, rhs.0) })
    }
}

impl MulAssign for Vf64x2 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for Vf64x2 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        Self(unsafe { _mm_div_pd(self.0, rhs.0) })
    }
}

impl DivAssign for Vf64x2 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for Vf64x2 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        Self(unsafe { _mm_sub_pd(_mm_setzero_pd(), self.0) })
    }
}
