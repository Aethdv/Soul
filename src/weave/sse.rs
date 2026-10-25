//! 128-bit vectors. AVX2 implies every SSE level, and the build is gated on AVX2 in
//! `weave/mod.rs`.

use core::{arch::x86_64::*, ops::Shr};
use std::mem;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct I16x8(pub(crate) __m128i);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct I32x4(pub(crate) __m128i);

#[expect(dead_code, reason = "completeness API: lane operations stay with their type")]
impl I16x8 {
    #[inline(always)]
    pub(crate) fn new(v: [i16; 8]) -> Self {
        // SAFETY: v provides 16 initialized bytes; the load is unaligned.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub(crate) fn zero() -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub(crate) fn splat(v: i16) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set1_epi16(v) })
    }

    /// # Safety
    /// `ptr` must be valid for reading 16 bytes (128 bits).
    #[inline(always)]
    pub(crate) unsafe fn load(ptr: *const i16) -> Self {
        // SAFETY: caller guarantees 16 readable bytes; the load is unaligned.
        Self(unsafe { _mm_loadu_si128(ptr as *const __m128i) })
    }

    /// # Safety
    /// `ptr` must be valid for writing 16 bytes (128 bits).
    #[inline(always)]
    pub(crate) unsafe fn store(self, ptr: *mut i16) {
        // SAFETY: caller guarantees 16 writable bytes; the store is unaligned.
        unsafe { _mm_storeu_si128(ptr as *mut __m128i, self.0) }
    }

    #[inline(always)]
    pub(crate) fn to_array(self) -> [i16; 8] {
        // SAFETY: Equal size, with every bit pattern valid for the integer lanes.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    pub(crate) fn extract<const N: i32>(self) -> i16 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        unsafe { _mm_extract_epi16::<N>(self.0) as i16 }
    }

    #[inline(always)]
    fn adds(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_adds_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    fn subs(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_subs_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub(crate) fn max(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_max_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub(crate) fn min(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_min_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mullo(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_mullo_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    fn mulhi(self, rhs: Self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_mulhi_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub(crate) fn madd(self, rhs: Self) -> I32x4 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I32x4(unsafe { _mm_madd_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    fn clamp_relu(self, ceil: Self) -> Self { self.max(Self::zero()).min(ceil) }

    #[inline(always)]
    fn srai<const N: i32>(self) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        Self(unsafe { _mm_srai_epi16::<N>(self.0) })
    }

    /// Extends the first 4 signed i16 lanes to i32 and returns them as a I32x4.
    /// Uses `vpmovsxwd` (SSE4.1).
    #[inline(always)]
    fn load_i32_4(self) -> I32x4 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I32x4(unsafe { _mm_cvtepi16_epi32(self.0) })
    }
}

impl From<[i16; 8]> for I16x8 {
    #[inline(always)]
    fn from(arr: [i16; 8]) -> Self { Self::new(arr) }
}

#[expect(dead_code, reason = "completeness API: lane operations stay with their type")]
impl I32x4 {
    #[inline(always)]
    pub(crate) const fn new(v: [i32; 4]) -> Self {
        // SAFETY: v provides 16 initialized bytes; the load is unaligned.
        Self(unsafe { _mm_loadu_si128(v.as_ptr() as *const __m128i) })
    }

    #[inline(always)]
    pub(crate) fn zero() -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_setzero_si128() })
    }

    #[inline(always)]
    pub(crate) fn splat(v: i32) -> Self {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set1_epi32(v) })
    }

    #[inline(always)]
    pub(crate) fn from_lanes(a: i32, b: i32, c: i32, d: i32) -> Self {
        // _mm_set_epi32 takes lanes in reverse (3, 2, 1, 0); re-reversed here so a..d map to lanes 0..3.
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        Self(unsafe { _mm_set_epi32(d, c, b, a) })
    }

    #[inline(always)]
    pub(crate) fn from_array(arr: [i32; 4]) -> Self { Self::new(arr) }

    /// Horizontal sum: unpack-add then shuffle-add, dodging the slower `_mm_hadd_epi32`.
    #[inline]
    fn reduce_sum(self) -> i32 {
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
    pub(crate) fn pack_i16(self, hi: Self) -> I16x8 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        I16x8(unsafe { _mm_packs_epi32(self.0, hi.0) })
    }

    #[inline(always)]
    fn to_array(self) -> [i32; 4] {
        // SAFETY: Equal size, with every bit pattern valid for the integer lanes.
        unsafe { mem::transmute(self.0) }
    }

    #[inline(always)]
    pub(crate) fn extract<const N: i32>(self) -> i32 {
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics. Const generic enforces bounds.
        unsafe { _mm_extract_epi32::<N>(self.0) }
    }

    #[inline(always)]
    pub(crate) fn srai<const N: i32>(self) -> Self {
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
        // A runtime count goes through _mm_sra_epi32, which reads it from a register.
        // SAFETY: AVX2 gate in mod.rs guarantees all SSE intrinsics.
        unsafe {
            let count = _mm_cvtsi32_si128(rhs);
            Self(_mm_sra_epi32(self.0, count))
        }
    }
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
