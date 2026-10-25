//! Mathematical traits for generic evaluation and automatic differentiation.

#![cfg_attr(not(feature = "dataset"), expect(dead_code, reason = "parameter loading is the tuner's"))]

use std::{
    arch::x86_64::_mm_cvtsi32_si128,
    ops::{Add, AddAssign, Div, Mul, Neg, Sub},
};

use crate::{
    core::defs::TOTAL_PHASE,
    weave::{I16x8, I32x4},
};

/// `i32` with `I16x8`/`I32x4` for search.
/// `f64` for the tuner.
/// `DualNode` for the gradient oracle.
#[rustfmt::skip]
pub(crate) trait EvalMath:
    Sized
    + Copy
    + Add<Output = Self>
    + AddAssign
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
{
    type Scalar;
    type Vec4: EvalVec4<Scalar = Self::Scalar, Vec8 = Self::Vec8>;
    type Vec8: EvalVec8<Vec4 = Self::Vec4>;
    type Array6: core::ops::Index<usize, Output = Self>;

    fn load_scalar(values: &[f64], offset: usize, slot: &mut usize) -> Self;
    fn load_vec4(values: &[f64], offset: usize, slot: &mut usize) -> Self::Vec4;
    fn load_array6(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array6;

    fn zero() -> Self;
    fn from_i32(val: i32) -> Self;
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4;
    fn trunc(self) -> Self;
    fn taper_acc(acc: &Self::Vec8, phase: Self) -> Self;
}

pub(crate) trait EvalVec4: Sized + Copy + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> {
    type Scalar;
    type Vec8;

    fn splat(val: i32) -> Self;
    fn from_lanes(a: Self::Scalar, b: Self::Scalar, c: Self::Scalar, d: Self::Scalar) -> Self;
    fn extract<const N: i32>(self) -> Self::Scalar;
    fn srai<const N: i32>(self) -> Self;
    fn pack_i16(self, hi: Self) -> Self::Vec8;
}

pub(crate) trait EvalVec8: Sized + Copy {
    type Vec4;

    fn zero() -> Self;
    fn madd(self, rhs: Self) -> Self::Vec4;
}

impl EvalMath for i32 {
    type Scalar = i32;
    type Vec4 = I32x4;
    type Vec8 = I16x8;
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
    fn load_array6(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array6 {
        core::array::from_fn(|i| values[offset + i] as i32)
    }

    #[inline(always)]
    fn zero() -> Self { 0 }

    #[inline(always)]
    fn from_i32(val: i32) -> Self { val }

    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 { I32x4::from_array(arr) }

    #[inline(always)]
    fn trunc(self) -> Self { self }

    #[inline(always)]
    fn taper_acc(acc: &Self::Vec8, phase: Self) -> Self {
        let eg_p = TOTAL_PHASE - phase;
        let packed = (phase as u32) | ((eg_p as u32) << 16);
        // SAFETY: SSE2 is guaranteed on x86_64.
        let weights = I16x8(unsafe { _mm_cvtsi32_si128(packed.cast_signed()) });
        // One madd computes mg · phase + eg · eg_p.
        acc.madd(weights).extract::<0>() / TOTAL_PHASE
    }
}

impl EvalMath for f64 {
    type Scalar = f64;
    type Vec4 = F64Vec4;
    type Vec8 = F64Vec8;
    type Array6 = [f64; 6];

    #[inline(always)]
    fn load_scalar(values: &[f64], offset: usize, _slot: &mut usize) -> Self { values[offset] }

    #[inline(always)]
    fn load_vec4(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Vec4 {
        Self::Vec4::from_lanes(values[offset], values[offset + 1], values[offset + 2], values[offset + 3])
    }

    #[inline(always)]
    fn load_array6(values: &[f64], offset: usize, _slot: &mut usize) -> Self::Array6 {
        core::array::from_fn(|i| values[offset + i])
    }

    #[inline(always)]
    fn zero() -> Self { 0.0 }

    #[inline(always)]
    fn from_i32(val: i32) -> Self { f64::from(val) }

    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 { F64Vec4(arr.map(f64::from)) }

    #[inline(always)]
    fn trunc(self) -> Self { f64::trunc(self) }

    #[inline(always)]
    fn taper_acc(acc: &Self::Vec8, phase: Self) -> Self {
        let mg = acc.0[0];
        let eg = acc.0[1];
        let tot = f64::from(TOTAL_PHASE);
        ((mg * phase + eg * (tot - phase)) / tot).trunc()
    }
}

#[derive(Clone, Copy)]
pub(crate) struct F64Vec4(pub(crate) [f64; 4]);

#[derive(Clone, Copy)]
pub(crate) struct F64Vec8(pub(crate) [f64; 8]);

impl Add for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output { F64Vec4(std::array::from_fn(|i| self.0[i] + rhs.0[i])) }
}

impl Sub for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output { F64Vec4(std::array::from_fn(|i| self.0[i] - rhs.0[i])) }
}

impl Mul for F64Vec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output { F64Vec4(std::array::from_fn(|i| self.0[i] * rhs.0[i])) }
}

impl EvalVec4 for F64Vec4 {
    type Scalar = f64;
    type Vec8 = F64Vec8;

    #[inline(always)]
    fn splat(val: i32) -> Self { F64Vec4([f64::from(val); 4]) }

    #[inline(always)]
    fn from_lanes(a: f64, b: f64, c: f64, d: f64) -> Self { F64Vec4([a, b, c, d]) }

    #[inline(always)]
    fn srai<const SHIFT: i32>(self) -> Self {
        // Negative values must round down, as an arithmetic shift does.
        let div = f64::from(1 << SHIFT);
        F64Vec4(self.0.map(|value| (value / div).floor()))
    }

    #[inline(always)]
    fn pack_i16(self, other: Self) -> F64Vec8 {
        F64Vec8([self.0[0], self.0[1], self.0[2], self.0[3], other.0[0], other.0[1], other.0[2], other.0[3]])
    }

    #[inline(always)]
    fn extract<const N: i32>(self) -> f64 { self.0[N as usize] }
}

impl EvalVec8 for F64Vec8 {
    type Vec4 = F64Vec4;

    #[inline(always)]
    fn zero() -> Self { F64Vec8([0.0; 8]) }

    #[inline(always)]
    fn madd(self, rhs: Self) -> Self::Vec4 {
        F64Vec4([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2] + self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4] + self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6] + self.0[7] * rhs.0[7],
        ])
    }
}

impl EvalVec4 for I32x4 {
    type Scalar = i32;
    type Vec8 = I16x8;

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

impl EvalVec8 for I16x8 {
    type Vec4 = I32x4;

    #[inline(always)]
    fn zero() -> Self { Self::zero() }

    #[inline(always)]
    fn madd(self, rhs: Self) -> Self::Vec4 { self.madd(rhs) }
}
