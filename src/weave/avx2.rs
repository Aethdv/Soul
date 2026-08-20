//! AVX2 (256-bit) SIMD intrinsics wrapper.
//!
//! Provides safely typed wrappers around `__m256i` for parallel bitboard and
//! accumulator operations.
//!
//! # Safety
//! Builds without `target_feature = "avx2"` are rejected at compile time by
//! `weave/mod.rs`, so every intrinsic here is available on any binary that
//! reaches this module.

use core::{
    arch::x86_64::*,
    ops::{
        Add, AddAssign, BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Div, DivAssign, Mul, MulAssign, Neg, Not,
        Sub, SubAssign,
    },
};

use super::*;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vu64x4(pub __m256i);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vi16x16(pub __m256i);

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vi32x8(pub __m256i);

impl Vi32x8 {
    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn from_raw(v: __m256i) -> Self { Self(v) }

    #[inline(always)]
    pub fn as_u64x4(self) -> Vu64x4 { Vu64x4(self.0) }

    #[inline(always)]
    pub fn as_i16x16(self) -> Vi16x16 { Vi16x16(self.0) }

    /// Horizontal sum to scalar i32.
    #[inline]
    pub fn hsum(self) -> i32 {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        unsafe {
            let v = self.0;
            let hi128 = _mm256_extracti128_si256::<1>(v);
            let lo128 = _mm256_castsi256_si128(v);
            let sum128 = _mm_add_epi32(lo128, hi128);
            let hi64 = _mm_unpackhi_epi64(sum128, sum128);
            let sum64 = _mm_add_epi32(sum128, hi64);
            let hi32 = _mm_shuffle_epi32::<0b_00_00_00_01>(sum64);
            let sum32 = _mm_add_epi32(sum64, hi32);
            _mm_cvtsi128_si32(sum32)
        }
    }
}

impl Add for Vi32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_add_epi32(self.0, rhs.0) })
    }
}

impl Vu64x4 {
    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn splat(v: u64) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_set1_epi64x(v as i64) })
    }

    #[inline(always)]
    pub fn ones() -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        unsafe {
            let z = _mm256_setzero_si256();
            Self(_mm256_cmpeq_epi64(z, z))
        }
    }

    #[inline(always)]
    pub fn from_lanes(a: u64, b: u64, c: u64, d: u64) -> Self {
        // _mm256_set_epi64x takes lanes in reverse (3, 2, 1, 0); re-reversed here so a..d map to lanes 0..3.
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_set_epi64x(d as i64, c as i64, b as i64, a as i64) })
    }

    /// Load 4 contiguous u64s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn load(ptr: *const u64) -> Self {
        // SAFETY: Caller guarantees the pointer is valid for an unaligned 256-bit read.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    #[inline(always)]
    pub fn shl<const N: i32>(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        Self(unsafe { _mm256_slli_epi64::<N>(self.0) })
    }

    #[inline(always)]
    pub fn shr<const N: i32>(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        Self(unsafe { _mm256_srli_epi64::<N>(self.0) })
    }

    #[inline(always)]
    pub fn andnot(self, rhs: Self) -> Self {
        // Computes (!self) & rhs, following the Intel intrinsic.
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_andnot_si256(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmp_eq(self, rhs: Self) -> VMask {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        VMask(unsafe { _mm256_cmpeq_epi64(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn cmp_gt(self, rhs: Self) -> VMask {
        // Bias both comparisons to make signed comparison behave like unsigned
        let bias = Self::splat(0x8000_0000_0000_0000);
        let a = self ^ bias;
        let b = rhs ^ bias;
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        VMask(unsafe { _mm256_cmpgt_epi64(a.0, b.0) })
    }

    /// Vertical popcount: PSHUFB nibble LUT + SAD trick.
    /// Returns a Vu64x4 where each lane holds the popcount of the corresponding input lane.
    #[rustfmt::skip]
    #[inline]
    pub fn popcount(self) -> Self {
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
    pub fn extract<const IDX: i32>(self) -> u64 {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        unsafe { _mm256_extract_epi64::<IDX>(self.0) as u64 }
    }

    #[inline(always)]
    pub fn as_i16x16(self) -> Vi16x16 { Vi16x16(self.0) }

    #[inline(always)]
    pub fn as_i32x8(self) -> Vi32x8 { Vi32x8(self.0) }
}

impl Add for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_add_epi64(self.0, rhs.0) })
    }
}

impl Sub for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_sub_epi64(self.0, rhs.0) })
    }
}

impl BitAnd for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
    }
}

impl BitAndAssign for Vu64x4 {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: Self) { *self = *self & rhs; }
}

impl BitOr for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
    }
}

impl BitOrAssign for Vu64x4 {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: Self) { *self = *self | rhs; }
}

impl BitXor for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
    }
}

impl BitXorAssign for Vu64x4 {
    #[inline(always)]
    fn bitxor_assign(&mut self, rhs: Self) { *self = *self ^ rhs; }
}

impl Not for Vu64x4 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { self ^ Self::ones() }
}

impl Vi16x16 {
    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_setzero_si256() })
    }

    #[inline(always)]
    pub fn splat(v: i16) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_set1_epi16(v) })
    }

    #[inline(always)]
    pub fn from_raw(v: __m256i) -> Self { Self(v) }

    /// Loads 16 contiguous i16s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn load(ptr: *const i16) -> Self {
        // SAFETY: Caller guarantees the pointer is valid for an unaligned 256-bit read.
        Self(unsafe { _mm256_loadu_si256(ptr as *const __m256i) })
    }

    /// Stores 16 contiguous i16s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn store(self, ptr: *mut i16) {
        // SAFETY: Caller guarantees the pointer is valid for an unaligned 256-bit write.
        unsafe { _mm256_storeu_si256(ptr as *mut __m256i, self.0) }
    }

    #[inline(always)]
    pub fn max(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_max_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn min(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_min_epi16(self.0, rhs.0) })
    }

    /// Clamp to [0, max]: ReLU then cap.
    #[inline(always)]
    pub fn clamp_relu(self, ceil: Self) -> Self { self.max(Self::zero()).min(ceil) }

    #[inline(always)]
    pub fn mulhi(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_mulhi_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn mullo(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_mullo_epi16(self.0, rhs.0) })
    }

    /// Horizontal pairwise add of i16 products, producing i32 lanes.
    #[inline(always)]
    pub fn madd(self, rhs: Self) -> Vi32x8 {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Vi32x8(unsafe { _mm256_madd_epi16(self.0, rhs.0) })
    }

    #[inline(always)]
    pub fn srai<const N: i32>(self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate. Const generic enforces bounds.
        Self(unsafe { _mm256_srai_epi16::<N>(self.0) })
    }

    #[inline(always)]
    pub fn blend(self, rhs: Self, mask: VMask) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_blendv_epi8(self.0, rhs.0, mask.0) })
    }

    #[inline(always)]
    pub fn as_u64x4(self) -> Vu64x4 { Vu64x4(self.0) }

    #[inline(always)]
    pub fn as_i32x8(self) -> Vi32x8 { Vi32x8(self.0) }
}

impl Add for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_add_epi16(self.0, rhs.0) })
    }
}

impl Sub for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_sub_epi16(self.0, rhs.0) })
    }
}

impl BitAnd for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitand(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_and_si256(self.0, rhs.0) })
    }
}

impl BitOr for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitor(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_or_si256(self.0, rhs.0) })
    }
}

impl BitXor for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        // SAFETY: AVX2 available per mod.rs compile_error gate.
        Self(unsafe { _mm256_xor_si256(self.0, rhs.0) })
    }
}

impl Not for Vi16x16 {
    type Output = Self;
    #[inline(always)]
    fn not(self) -> Self { self ^ Self::splat(-1) }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vf32x8(pub __m256);

impl Vf32x8 {
    #[inline(always)]
    pub fn zero() -> Self { Self(unsafe { _mm256_setzero_ps() }) }

    #[inline(always)]
    pub fn splat(v: f32) -> Self { Self(unsafe { _mm256_set1_ps(v) }) }

    /// Load 8 contiguous f32s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f32) -> Self { Self(unsafe { _mm256_loadu_ps(ptr) }) }

    /// Store 8 contiguous f32s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f32) { unsafe { _mm256_storeu_ps(ptr, self.0) }; }
}

impl Add for Vf32x8 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self(unsafe { _mm256_add_ps(self.0, rhs.0) }) }
}

impl AddAssign for Vf32x8 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl Sub for Vf32x8 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm256_sub_ps(self.0, rhs.0) }) }
}

impl SubAssign for Vf32x8 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl Mul for Vf32x8 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { Self(unsafe { _mm256_mul_ps(self.0, rhs.0) }) }
}

impl MulAssign for Vf32x8 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl Div for Vf32x8 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self { Self(unsafe { _mm256_div_ps(self.0, rhs.0) }) }
}

impl DivAssign for Vf32x8 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl Neg for Vf32x8 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm256_sub_ps(_mm256_setzero_ps(), self.0) }) }
}

impl From<[f32; 8]> for Vf32x8 {
    #[inline(always)]
    fn from(arr: [f32; 8]) -> Self { unsafe { Self::loadu(arr.as_ptr()) } }
}

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct Vf64x4(pub __m256d);

impl Vf64x4 {
    #[inline(always)]
    pub fn zero() -> Self { Self(unsafe { _mm256_setzero_pd() }) }

    #[inline(always)]
    pub fn splat(v: f64) -> Self { Self(unsafe { _mm256_set1_pd(v) }) }

    /// Load 4 contiguous f64s directly from memory.
    ///
    /// # Safety
    /// `ptr` must be valid for reading 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn loadu(ptr: *const f64) -> Self { Self(unsafe { _mm256_loadu_pd(ptr) }) }

    /// Store 4 contiguous f64s directly to memory.
    ///
    /// # Safety
    /// `ptr` must be valid for writing 32 bytes (256 bits).
    #[inline(always)]
    pub unsafe fn storeu(self, ptr: *mut f64) { unsafe { _mm256_storeu_pd(ptr, self.0) }; }
}

impl Add for Vf64x4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self { Self(unsafe { _mm256_add_pd(self.0, rhs.0) }) }
}

impl AddAssign for Vf64x4 {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) { *self = *self + rhs; }
}

impl Sub for Vf64x4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self { Self(unsafe { _mm256_sub_pd(self.0, rhs.0) }) }
}

impl SubAssign for Vf64x4 {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) { *self = *self - rhs; }
}

impl Mul for Vf64x4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self { Self(unsafe { _mm256_mul_pd(self.0, rhs.0) }) }
}

impl MulAssign for Vf64x4 {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) { *self = *self * rhs; }
}

impl Div for Vf64x4 {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self { Self(unsafe { _mm256_div_pd(self.0, rhs.0) }) }
}

impl DivAssign for Vf64x4 {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) { *self = *self / rhs; }
}

impl Neg for Vf64x4 {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self { Self(unsafe { _mm256_sub_pd(_mm256_setzero_pd(), self.0) }) }
}

impl From<[f64; 4]> for Vf64x4 {
    #[inline(always)]
    fn from(arr: [f64; 4]) -> Self { unsafe { Self::loadu(arr.as_ptr()) } }
}

impl From<Vf64x4> for [f64; 4] {
    #[inline(always)]
    fn from(v: Vf64x4) -> Self {
        let mut arr = [0.0; 4];
        unsafe { v.storeu(arr.as_mut_ptr()) };
        arr
    }
}
