//! Mathematical traits for generic evaluation and automatic differentiation.

use std::{
    arch::x86_64::_mm_cvtsi32_si128,
    ops,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use crate::{
    core::defs::TOTAL_PHASE,
    weave::{Vi16x8, Vi32x4},
};

/// The interface `evaluate` is generic over: `i32` with `Vi16x8`/`Vi32x4` for search,
/// `DualNode` for the tuner, `f64` for the oracle. The module doc has the why.
///
/// `to_i32` and `to_f64` hand back a raw value to branch on, and the gradient stops
/// there: whatever the branch decides is invisible to the derivative.
pub trait EvalMath:
    Sized
    + Copy
    + Clone
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + SubAssign
    + Mul<Output = Self>
    + MulAssign
    + Div<Output = Self>
    + DivAssign
    + Neg<Output = Self>
{
    type Scalar;
    type Vec4: EnvVec4<Scalar = Self::Scalar, Vec8 = Self::Vec8>;
    type Vec8: EnvVec8<Scalar = Self::Scalar, Vec4 = Self::Vec4>;
    type Array4: ops::Index<usize, Output = Self>;
    type Array6: ops::Index<usize, Output = Self>;

    fn load_scalar(values: &[f64], offset: usize, slot: &mut usize) -> Self;
    fn load_vec4(values: &[f64], offset: usize, slot: &mut usize) -> Self::Vec4;
    fn load_array4(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array4;
    fn load_array6(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array6;

    fn zero() -> Self;
    fn new(val: f64) -> Self;

    fn max(self, other: Self) -> Self;
    fn min(self, other: Self) -> Self;

    fn from_i32(val: i32) -> Self;
    fn from_vi32x4(v: crate::weave::Vi32x4) -> Self::Vec4;
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4;

    fn abs(self) -> Self;

    fn to_i32(self) -> i32;
    fn to_f64(self) -> f64;

    /// Truncate the fractional part (acts as identity for integers).
    fn trunc(self) -> Self;

    /// `clamp` is taken by the inherent methods on both `i32` and `f64`, hence the name.
    fn math_clamp(self, min: Self, max: Self) -> Self;

    /// Tapered interpolation: (MG · phase + EG · (TOTAL_PHASE - phase)) / TOTAL_PHASE.
    /// Specialized per-type to allow SIMD `madd` in the engine hot path.
    fn tapered(acc: &Self::Vec8, phase: Self) -> Self;
}

pub trait EnvVec4: Sized + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    type Scalar;
    type Vec8;

    fn zero() -> Self;
    fn splat(val: i32) -> Self;
    fn from_lanes(a: Self::Scalar, b: Self::Scalar, c: Self::Scalar, d: Self::Scalar) -> Self;
    fn extract<const N: i32>(self) -> Self::Scalar;
    fn srai<const N: i32>(self) -> Self;
    fn pack_i16(self, hi: Self) -> Self::Vec8;
}

pub trait EnvVec8: Sized + Copy {
    type Scalar;
    type Vec4;

    fn zero() -> Self;
    fn splat(val: i16) -> Self;
    fn madd(self, rhs: Self) -> Self::Vec4;
    fn load_i32_4(self) -> Self::Vec4;
    fn extract<const N: i32>(self) -> Self::Scalar;
}

impl EvalMath for i32 {
    type Scalar = i32;
    type Vec4 = Vi32x4;
    type Vec8 = Vi16x8;
    type Array4 = [i32; 4];
    type Array6 = [i32; 6];

    #[inline(always)]
    fn load_scalar(values: &[f64], offset: usize, _slot: &mut usize) -> Self { values[offset] as i32 }

    #[inline(always)]
    fn load_vec4(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Vec4 {
        Self::Vec4::from_lanes(
            values[offset] as i32,
            values[offset + 1] as i32,
            values[offset + 2] as i32,
            values[offset + 3] as i32,
        )
    }

    #[inline(always)]
    fn load_array4(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array4 {
        [values[offset] as i32, values[offset + 1] as i32, values[offset + 2] as i32, values[offset + 3] as i32]
    }

    #[inline(always)]
    fn load_array6(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array6 {
        [
            values[offset] as i32,
            values[offset + 1] as i32,
            values[offset + 2] as i32,
            values[offset + 3] as i32,
            values[offset + 4] as i32,
            values[offset + 5] as i32,
        ]
    }

    #[inline(always)]
    fn zero() -> Self { 0 }

    #[inline(always)]
    fn new(val: f64) -> Self { val as i32 }

    #[inline(always)]
    fn from_i32(val: i32) -> Self { val }

    #[inline(always)]
    fn max(self, other: Self) -> Self { std::cmp::max(self, other) }

    #[inline(always)]
    fn min(self, other: Self) -> Self { std::cmp::min(self, other) }

    #[inline(always)]
    fn abs(self) -> Self { i32::abs(self) }

    #[inline(always)]
    fn to_i32(self) -> i32 { self }

    #[inline(always)]
    fn from_vi32x4(v: Vi32x4) -> Self::Vec4 { v }

    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 { Vi32x4::from_array(arr) }

    #[inline(always)]
    fn to_f64(self) -> f64 { self as f64 }

    #[inline(always)]
    fn trunc(self) -> Self { self }

    #[inline(always)]
    fn math_clamp(self, min: Self, max: Self) -> Self { Ord::clamp(self, min, max) }

    #[inline(always)]
    fn tapered(acc: &Self::Vec8, phase: Self) -> Self {
        let eg_p = TOTAL_PHASE - phase;
        // Both halves sit in [0, TOTAL_PHASE], so they pack into one i32 with no sign bit
        // bleeding from the low lane into the high.
        let packed = (phase as u32) | ((eg_p as u32) << 16);
        // SAFETY: AVX2 per the compile_error gate in weave/mod.rs; a register move with no
        // memory operand and no further precondition.
        let weights = Vi16x8(unsafe { _mm_cvtsi32_si128(packed.cast_signed()) });
        // madd's pairwise dot product folds mg·phase + eg·eg_phase into one instruction.
        acc.madd(weights).extract::<0>() / TOTAL_PHASE
    }
}

impl EvalMath for f64 {
    type Scalar = f64;
    type Vec4 = F64Vec4;
    type Vec8 = F64Vec8;
    type Array4 = [f64; 4];
    type Array6 = [f64; 6];

    #[inline(always)]
    fn load_scalar(values: &[f64], offset: usize, _slot: &mut usize) -> Self { values[offset] }

    #[inline(always)]
    fn load_vec4(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Vec4 {
        Self::Vec4::from_lanes(values[offset], values[offset + 1], values[offset + 2], values[offset + 3])
    }

    #[inline(always)]
    fn load_array4(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array4 {
        [values[offset], values[offset + 1], values[offset + 2], values[offset + 3]]
    }

    #[inline(always)]
    fn load_array6(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array6 {
        [
            values[offset],
            values[offset + 1],
            values[offset + 2],
            values[offset + 3],
            values[offset + 4],
            values[offset + 5],
        ]
    }

    #[inline(always)]
    fn zero() -> Self { 0.0 }

    #[inline(always)]
    fn new(val: f64) -> Self { val }

    #[inline(always)]
    fn from_i32(val: i32) -> Self { f64::from(val) }

    #[inline(always)]
    fn max(self, other: Self) -> Self { f64::max(self, other) }

    #[inline(always)]
    fn min(self, other: Self) -> Self { f64::min(self, other) }

    #[inline(always)]
    fn abs(self) -> Self { f64::abs(self) }

    #[inline(always)]
    fn to_i32(self) -> i32 { self as i32 }

    #[inline(always)]
    fn from_vi32x4(v: Vi32x4) -> Self::Vec4 {
        let arr = v.to_array();
        F64Vec4([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64])
    }

    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 { F64Vec4([arr[0] as f64, arr[1] as f64, arr[2] as f64, arr[3] as f64]) }

    #[inline(always)]
    fn to_f64(self) -> f64 { self }

    #[inline(always)]
    fn trunc(self) -> Self { f64::trunc(self) }

    #[inline(always)]
    fn math_clamp(self, min: Self, max: Self) -> Self { f64::clamp(self, min, max) }

    #[inline(always)]
    fn tapered(acc: &Self::Vec8, phase: Self) -> Self {
        let mg = acc.0[0];
        let eg = acc.0[1];
        let p = phase;
        let tot = f64::from(TOTAL_PHASE);
        ((mg * p + eg * (tot - p)) / tot).trunc()
    }
}

#[derive(Clone, Copy)]
pub struct F64Vec4(pub [f64; 4]);

#[derive(Clone, Copy)]
pub struct F64Vec8(pub [f64; 8]);

impl Add for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        F64Vec4([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1], self.0[2] + rhs.0[2], self.0[3] + rhs.0[3]])
    }
}

impl Sub for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        F64Vec4([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1], self.0[2] - rhs.0[2], self.0[3] - rhs.0[3]])
    }
}

impl Mul for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        F64Vec4([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1], self.0[2] * rhs.0[2], self.0[3] * rhs.0[3]])
    }
}

impl EnvVec4 for F64Vec4 {
    type Scalar = f64;
    type Vec8 = F64Vec8;

    #[inline(always)]
    fn zero() -> Self { F64Vec4([0.0; 4]) }

    #[inline(always)]
    fn splat(val: i32) -> Self { F64Vec4([f64::from(val); 4]) }

    #[inline(always)]
    fn from_lanes(a: f64, b: f64, c: f64, d: f64) -> Self { F64Vec4([a, b, c, d]) }

    #[inline(always)]
    fn srai<const SHIFT: i32>(self) -> Self {
        // floor, not trunc: a signed arithmetic right shift rounds toward negative
        // infinity, where truncation toward zero would diverge on negative values.
        let div = f64::from(1 << SHIFT);
        F64Vec4([(self.0[0] / div).floor(), (self.0[1] / div).floor(), (self.0[2] / div).floor(), (self.0[3] / div).floor()])
    }

    #[inline(always)]
    fn pack_i16(self, other: Self) -> F64Vec8 {
        F64Vec8([self.0[0], self.0[1], self.0[2], self.0[3], other.0[0], other.0[1], other.0[2], other.0[3]])
    }

    #[inline(always)]
    fn extract<const N: i32>(self) -> f64 { self.0[N as usize] }
}

impl EnvVec8 for F64Vec8 {
    type Scalar = f64;
    type Vec4 = F64Vec4;

    #[inline(always)]
    fn zero() -> Self { F64Vec8([0.0; 8]) }

    #[inline(always)]
    fn splat(val: i16) -> Self { F64Vec8([f64::from(val); 8]) }

    #[inline(always)]
    fn madd(self, rhs: Self) -> Self::Vec4 {
        F64Vec4([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2] + self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4] + self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6] + self.0[7] * rhs.0[7],
        ])
    }

    #[inline(always)]
    fn load_i32_4(self) -> Self::Vec4 { F64Vec4([self.0[0], self.0[1], self.0[2], self.0[3]]) }

    #[inline(always)]
    fn extract<const N: i32>(self) -> Self::Scalar { self.0[N as usize] }
}

impl EnvVec4 for Vi32x4 {
    type Scalar = i32;
    type Vec8 = Vi16x8;

    #[inline(always)]
    fn zero() -> Self { Self::zero() }

    #[inline(always)]
    fn splat(val: i32) -> Self { Self::splat(val) }

    #[inline(always)]
    fn from_lanes(a: i32, b: i32, c: i32, d: i32) -> Self { Self::from_lanes(a, b, c, d) }

    #[inline(always)]
    fn extract<const N: i32>(self) -> i32 { self.extract::<N>() }

    #[inline(always)]
    fn srai<const N: i32>(self) -> Self { self.srai::<N>() }

    #[inline(always)]
    fn pack_i16(self, hi: Self) -> Self::Vec8 { self.pack_i16(hi) }
}

impl EnvVec8 for Vi16x8 {
    type Scalar = i32;
    type Vec4 = Vi32x4;

    #[inline(always)]
    fn zero() -> Self { Self::zero() }

    #[inline(always)]
    fn splat(val: i16) -> Self { Self::splat(val) }

    #[inline(always)]
    fn madd(self, rhs: Self) -> Self::Vec4 { self.madd(rhs) }

    #[inline(always)]
    fn load_i32_4(self) -> Self::Vec4 { self.load_i32_4() }

    #[inline(always)]
    fn extract<const N: i32>(self) -> Self::Scalar { i32::from(self.extract::<N>()) }
}
