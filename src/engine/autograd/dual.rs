//! Forward-mode AD via dual numbers for the HCE eval graph.
//!
//! The eval has 30 tunable inputs (2 accumulator lanes + 28 EvalParams weights)
//! producing 1 scalar. Each dual number carries partial derivatives in `[f32; 32]`
//! (padded for AVX2: exactly 4 · ymm256 registers).
//!
//! Because the current eval is linear in its parameters, the production training
//! loop uses `eval_linear_grad` — direct feature extraction that's "~30× cheaper".
//! This dual path serves as a correctness oracle:
//! run both, compare, and any disagreement means the hand-derived gradient has a bug.
//!
//! An `active` bitmask tracks which gradient slots are non-zero. Constants
//! have `active = 0`, enabling short-circuit paths that skip gradient
//! computation entirely when one operand is constant.

use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::traits::{EnvVec4, EnvVec8, EvalMath};
use crate::weave::Vf32x8;

/// 30 slots (2 acc lanes + 28 params) padded to 32 for alignment.
pub const DUAL_N: usize = 32;

/// A dual number: value + gradient vector + active-slot bitmask.
///
/// The bitmask enables 𝒪(1) constant detection: when `active == 0`,
/// all gradient slots are zero and we can skip gradient work entirely.
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DualNode {
    pub grad: [f32; DUAL_N],
    pub val: f64,
    pub active: u32,
}

impl DualNode {
    #[inline(always)]
    pub fn floor(self) -> Self {
        // Straight-Through Estimator (STE): passes gradient through unmodified
        // to traverse quantization operations during training.
        Self { val: self.val.floor(), grad: self.grad, active: self.active }
    }

    /// Constant — zero gradient, no active slots.
    #[inline(always)]
    pub fn constant(val: f64) -> Self {
        Self { grad: [0.0; DUAL_N], val, active: 0 }
    }

    /// Seed a dual variable: `grad[idx] = 1.0`, active bit set.
    #[inline(always)]
    pub fn seed(val: f64, idx: usize) -> Self {
        let mut grad = [0.0f32; DUAL_N];
        grad[idx] = 1.0;
        Self { grad, val, active: 1 << idx }
    }

    /// Zero value, zero gradient.
    #[inline(always)]
    pub fn zero() -> Self {
        Self { grad: [0.0; DUAL_N], val: 0.0, active: 0 }
    }
}

impl fmt::Debug for DualNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({:.4}, active={:#010x})", self.val, self.active)
    }
}

// ──────── Operators ────────
//
// Each operator checks `active == 0` (constant) to skip gradient work.
// Ideally for the ~50% of eval operations involving a constant,
// this eliminates the gradient computation entirely.
//
// The dense `for i in 0..DUAL_N` loops auto-vectorize into 4 AVX2
// instructions (32 f32s = 4 · ymm256).

impl Add for DualNode {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let val = self.val + rhs.val;
        let active = self.active | rhs.active;

        if active == 0 {
            return Self { grad: self.grad, val, active };
        }

        let mut grad = [0.0f32; DUAL_N];
        if self.active != 0 && rhs.active != 0 {
            // SAFETY: DualNode grad arrays are exactly DUAL_N (32) f32s.
            // 4 iterations · stride 8 = offsets 0, 8, 16, 24; each load/store
            // touches 8 f32s (32 bytes). Max access: 24 + 8 = 32 = DUAL_N.
            unsafe {
                let pa = self.grad.as_ptr();
                let pb = rhs.grad.as_ptr();
                let po = grad.as_mut_ptr();
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    let gb = Vf32x8::loadu(pb.add(off));
                    (ga + gb).storeu(po.add(off));
                }
            }
        } else if self.active != 0 {
            grad.copy_from_slice(&self.grad);
        } else {
            grad.copy_from_slice(&rhs.grad);
        }

        Self { grad, val, active }
    }
}

impl AddAssign for DualNode {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Sub for DualNode {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let val = self.val - rhs.val;
        let active = self.active | rhs.active;

        if active == 0 {
            return Self { grad: self.grad, val, active };
        }

        let mut grad = [0.0f32; DUAL_N];
        if self.active != 0 && rhs.active != 0 {
            unsafe {
                let pa = self.grad.as_ptr();
                let pb = rhs.grad.as_ptr();
                let po = grad.as_mut_ptr();
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    let gb = Vf32x8::loadu(pb.add(off));
                    (ga - gb).storeu(po.add(off));
                }
            }
        } else if self.active != 0 {
            grad.copy_from_slice(&self.grad);
        } else {
            unsafe {
                let pb = rhs.grad.as_ptr();
                let po = grad.as_mut_ptr();
                let z = Vf32x8::zero();
                for i in 0..4 {
                    let off = i * 8;
                    let gb = Vf32x8::loadu(pb.add(off));
                    (z - gb).storeu(po.add(off));
                }
            }
        }

        Self { grad, val, active }
    }
}

impl SubAssign for DualNode {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl Mul for DualNode {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let val = self.val * rhs.val;
        let active = self.active | rhs.active;

        if active == 0 {
            return Self { grad: self.grad, val, active };
        }

        let a = self.val as f32;
        let b = rhs.val as f32;
        let mut grad = [0.0f32; DUAL_N];

        // SAFETY: grad arrays are DUAL_N (32) f32s; see Add impl.
        unsafe {
            let pa = self.grad.as_ptr();
            let pb = rhs.grad.as_ptr();
            let po = grad.as_mut_ptr();

            if self.active != 0 && rhs.active != 0 {
                let va = Vf32x8::splat(a);
                let vb = Vf32x8::splat(b);
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    let gb = Vf32x8::loadu(pb.add(off));
                    (vb * ga + va * gb).storeu(po.add(off));
                }
            } else if self.active != 0 {
                let vb = Vf32x8::splat(b);
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    (vb * ga).storeu(po.add(off));
                }
            } else {
                let va = Vf32x8::splat(a);
                for i in 0..4 {
                    let off = i * 8;
                    let gb = Vf32x8::loadu(pb.add(off));
                    (va * gb).storeu(po.add(off));
                }
            }
        }

        Self { grad, val, active }
    }
}

impl MulAssign for DualNode {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Div for DualNode {
    type Output = Self;
    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        let active = self.active | rhs.active;

        if active == 0 {
            return Self { grad: self.grad, val, active };
        }

        let b = rhs.val as f32;
        let b2 = b * b;
        let a = self.val as f32;
        let mut grad = [0.0f32; DUAL_N];

        // SAFETY: grad arrays are DUAL_N (32) f32s; see Add impl.
        unsafe {
            let pa = self.grad.as_ptr();
            let pb = rhs.grad.as_ptr();
            let po = grad.as_mut_ptr();

            if self.active != 0 && rhs.active != 0 {
                let vb = Vf32x8::splat(b);
                let va = Vf32x8::splat(a);
                let vb2 = Vf32x8::splat(b2);
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    let gb = Vf32x8::loadu(pb.add(off));
                    let num = vb * ga - va * gb;
                    (num / vb2).storeu(po.add(off));
                }
            } else if self.active != 0 {
                let vb = Vf32x8::splat(b);
                for i in 0..4 {
                    let off = i * 8;
                    let ga = Vf32x8::loadu(pa.add(off));
                    (ga / vb).storeu(po.add(off));
                }
            } else {
                let va = Vf32x8::splat(-a);
                let vb2 = Vf32x8::splat(b2);
                for i in 0..4 {
                    let off = i * 8;
                    let gb = Vf32x8::loadu(pb.add(off));
                    let num = va * gb;
                    (num / vb2).storeu(po.add(off));
                }
            }
        }

        Self { grad, val, active }
    }
}

impl DivAssign for DualNode {
    #[inline(always)]
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Neg for DualNode {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        if self.active == 0 {
            return Self { grad: self.grad, val: -self.val, active: 0 };
        }

        let mut grad = [0.0f32; DUAL_N];
        // SAFETY: grad arrays are DUAL_N (32) f32s; see Add impl.
        unsafe {
            let pa = self.grad.as_ptr();
            let po = grad.as_mut_ptr();
            let z = Vf32x8::zero();
            for i in 0..4 {
                let off = i * 8;
                let ga = Vf32x8::loadu(pa.add(off));
                (z - ga).storeu(po.add(off));
            }
        }
        Self { grad, val: -self.val, active: self.active }
    }
}

// ──────── EvalMath ────────

impl EvalMath for DualNode {
    type Scalar = DualNode;
    type Vec4 = DualVec4;
    type Vec8 = DualVec8;
    type Array4 = [DualNode; 4];
    type Array6 = [DualNode; 6];

    #[inline(always)]
    fn load_scalar(values: &[f64], offset: usize, slot: &mut usize) -> Self {
        let node = DualNode::seed(values[offset], *slot);
        *slot += 1;
        node
    }

    #[inline(always)]
    fn load_vec4(values: &[f64], offset: usize, slot: &mut usize) -> Self::Vec4 {
        let mut out = [DualNode::zero(); 4];
        for i in 0..4 {
            out[i] = DualNode::seed(values[offset + i], *slot);
            *slot += 1;
        }
        DualVec4(out)
    }

    #[allow(dead_code)]
    #[inline(always)]
    fn load_array4(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array4 {
        let mut out = [DualNode::zero(); 4];
        for i in 0..4 {
            out[i] = DualNode::seed(values[offset + i], *slot);
            *slot += 1;
        }
        out
    }

    #[inline(always)]
    fn load_array6(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array6 {
        let mut out = [DualNode::zero(); 6];
        for i in 0..6 {
            out[i] = DualNode::seed(values[offset + i], *slot);
            *slot += 1;
        }
        out
    }

    #[inline(always)]
    fn zero() -> Self {
        DualNode::zero()
    }

    #[inline(always)]
    fn new(val: f64) -> Self {
        DualNode::constant(val)
    }

    #[inline(always)]
    fn from_i32(val: i32) -> Self {
        DualNode::constant(f64::from(val))
    }

    #[inline(always)]
    fn max(self, other: Self) -> Self {
        if self.val >= other.val { self } else { other }
    }

    #[inline(always)]
    fn min(self, other: Self) -> Self {
        if self.val <= other.val { self } else { other }
    }

    #[inline(always)]
    fn abs(self) -> Self {
        if self.val < 0.0 { -self } else { self }
    }

    #[inline(always)]
    fn to_i32(self) -> i32 {
        self.val as i32
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self.val
    }

    #[inline(always)]
    fn trunc(self) -> Self {
        Self { val: self.val.trunc(), grad: self.grad, active: self.active }
    }

    #[inline(always)]
    fn math_clamp(self, min: Self, max: Self) -> Self {
        if self.val <= min.val {
            return Self { val: min.val, grad: [0.0; DUAL_N], active: 0 };
        }
        if self.val >= max.val {
            return Self { val: max.val, grad: [0.0; DUAL_N], active: 0 };
        }
        self
    }

    #[inline(always)]
    fn tapered(acc: &Self::Vec8, phase: Self) -> Self {
        let mg = acc.0[0];
        let eg = acc.0[1];
        let p = phase;
        let eg_p = Self::from_i32(24) - phase;
        let tot = Self::from_i32(24);
        ((mg * p + eg * eg_p) / tot).trunc()
    }

    #[inline(always)]
    fn from_vi32x4(v: crate::weave::Vi32x4) -> Self::Vec4 {
        let arr = v.to_array();
        DualVec4([Self::from_i32(arr[0]), Self::from_i32(arr[1]), Self::from_i32(arr[2]), Self::from_i32(arr[3])])
    }

    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 {
        DualVec4([Self::from_i32(arr[0]), Self::from_i32(arr[1]), Self::from_i32(arr[2]), Self::from_i32(arr[3])])
    }
}

// ──────── DualVec4 ────────

#[derive(Clone, Copy)]
pub struct DualVec4(pub [DualNode; 4]);

impl Add for DualVec4 {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        DualVec4([self.0[0] + rhs.0[0], self.0[1] + rhs.0[1], self.0[2] + rhs.0[2], self.0[3] + rhs.0[3]])
    }
}

impl Sub for DualVec4 {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        DualVec4([self.0[0] - rhs.0[0], self.0[1] - rhs.0[1], self.0[2] - rhs.0[2], self.0[3] - rhs.0[3]])
    }
}

impl Mul for DualVec4 {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        DualVec4([self.0[0] * rhs.0[0], self.0[1] * rhs.0[1], self.0[2] * rhs.0[2], self.0[3] * rhs.0[3]])
    }
}

impl EnvVec4 for DualVec4 {
    type Scalar = DualNode;
    type Vec8 = DualVec8;

    #[inline(always)]
    fn zero() -> Self {
        DualVec4([DualNode::zero(); 4])
    }

    #[inline(always)]
    fn splat(val: i32) -> Self {
        let n = DualNode::constant(f64::from(val));
        DualVec4([n, n, n, n])
    }

    #[inline(always)]
    fn from_lanes(a: DualNode, b: DualNode, c: DualNode, d: DualNode) -> Self {
        DualVec4([a, b, c, d])
    }

    #[inline(always)]
    fn extract<const N: i32>(self) -> DualNode {
        self.0[N as usize]
    }

    #[inline(always)]
    fn srai<const N: i32>(self) -> Self {
        let divisor = DualNode::constant((1_u32 << N) as f64);
        DualVec4([
            (self.0[0] / divisor).floor(),
            (self.0[1] / divisor).floor(),
            (self.0[2] / divisor).floor(),
            (self.0[3] / divisor).floor(),
        ])
    }

    #[inline(always)]
    fn pack_i16(self, hi: Self) -> DualVec8 {
        let min = DualNode::constant(-32768.0);
        let max = DualNode::constant(32767.0);
        DualVec8([
            self.0[0].math_clamp(min, max),
            self.0[1].math_clamp(min, max),
            self.0[2].math_clamp(min, max),
            self.0[3].math_clamp(min, max),
            hi.0[0].math_clamp(min, max),
            hi.0[1].math_clamp(min, max),
            hi.0[2].math_clamp(min, max),
            hi.0[3].math_clamp(min, max),
        ])
    }
}

// ──────── DualVec8 ────────

#[derive(Clone, Copy)]
pub struct DualVec8(pub [DualNode; 8]);

impl EnvVec8 for DualVec8 {
    type Scalar = DualNode;
    type Vec4 = DualVec4;

    #[inline(always)]
    fn zero() -> Self {
        DualVec8([DualNode::zero(); 8])
    }

    #[inline(always)]
    fn splat(val: i16) -> Self {
        let n = DualNode::constant(f64::from(val));
        DualVec8([n, n, n, n, n, n, n, n])
    }

    #[inline(always)]
    fn madd(self, rhs: Self) -> DualVec4 {
        DualVec4([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2] + self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4] + self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6] + self.0[7] * rhs.0[7],
        ])
    }

    #[inline(always)]
    fn load_i32_4(self) -> DualVec4 {
        DualVec4([self.0[0], self.0[1], self.0[2], self.0[3]])
    }

    #[inline(always)]
    fn extract<const N: i32>(self) -> DualNode {
        self.0[N as usize]
    }
}
