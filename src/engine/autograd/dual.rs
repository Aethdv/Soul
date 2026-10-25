//! Forward-mode automatic differentiation for the HCE evaluation graph.
//!
//! Differentiates with respect to both accumulator lanes and every
//! crate::engine::eval::EvalParams weight. Each dual number's derivative
//! vector is `[f32; DUAL_N]`.
//!
//! The gradient oracle compares `eval_linear_grad` (~30× cheaper) against
//! this implementation, including the combiner's nonlinear king-danger curve.

use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::traits::{EvalMath, EvalVec4, EvalVec8};
use crate::{core::defs::TOTAL_PHASE, engine::eval_params, weave::F32x8};

/// Gradient slots, rounded to eight-lane SIMD chunks.
pub(crate) const DUAL_N: usize = (eval_params::DUAL_SLOTS + 7) & !7;

#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub(crate) struct DualNode {
    pub(crate) grad: [f32; DUAL_N],
    pub(crate) val: f64,
    tracked: bool,
}

impl DualNode {
    #[inline(always)]
    fn floor(self) -> Self { Self { val: self.val.floor(), grad: self.grad, tracked: self.tracked } }

    #[inline(always)]
    pub(crate) fn constant(val: f64) -> Self { Self { grad: [0.0; DUAL_N], val, tracked: false } }

    /// Independent variable differentiated with respect to slot `slot`.
    #[inline(always)]
    pub(crate) fn seed(val: f64, slot: usize) -> Self {
        let mut grad = [0.0f32; DUAL_N];
        grad[slot] = 1.0;
        Self { grad, val, tracked: true }
    }

    #[inline(always)]
    pub(crate) fn zero() -> Self { Self::constant(0.0) }

    #[inline(always)]
    fn clamp_i16(self) -> Self {
        if self.val <= i16::MIN as f64 {
            return Self::constant(i16::MIN as f64);
        }
        if self.val >= i16::MAX as f64 {
            return Self::constant(i16::MAX as f64);
        }
        self
    }
}

impl fmt::Debug for DualNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Dual({:.4}, tracked={})", self.val, self.tracked) }
}

#[inline(always)]
fn grad_map1(a: &[f32; DUAL_N], out: &mut [f32; DUAL_N], f: impl Fn(F32x8) -> F32x8) {
    // SAFETY: a, out are each exactly DUAL_N f32s, DUAL_N a multiple of 8; see grad_map2.
    unsafe {
        let (pa, po) = (a.as_ptr(), out.as_mut_ptr());
        for i in 0..DUAL_N / 8 {
            let off = i * 8;
            f(F32x8::loadu(pa.add(off))).storeu(po.add(off));
        }
    }
}

#[inline(always)]
fn grad_map2(a: &[f32; DUAL_N], b: &[f32; DUAL_N], out: &mut [f32; DUAL_N], f: impl Fn(F32x8, F32x8) -> F32x8) {
    // SAFETY: DUAL_N is rounded up to a multiple of 8, so every SIMD access is in bounds.
    unsafe {
        let (pa, pb, po) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        for i in 0..DUAL_N / 8 {
            let off = i * 8;
            let ga = F32x8::loadu(pa.add(off));
            let gb = F32x8::loadu(pb.add(off));
            f(ga, gb).storeu(po.add(off));
        }
    }
}

/// Forward a compound-assignment operator to its binary counterpart.
macro_rules! impl_assign {
    ($trait:ident, $method:ident, $op:tt) => {
        impl $trait for DualNode {
            #[inline(always)]
            fn $method(&mut self, rhs: Self) {
                *self = *self $op rhs;
            }
        }
    };
}

impl Add for DualNode {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        let val = self.val + rhs.val;
        let tracked = self.tracked || rhs.tracked;

        if !tracked {
            return Self { grad: self.grad, val, tracked };
        }

        let mut grad = [0.0f32; DUAL_N];

        if self.tracked && rhs.tracked {
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| ga + gb);
        } else if self.tracked {
            grad.copy_from_slice(&self.grad);
        } else {
            grad.copy_from_slice(&rhs.grad);
        }

        Self { grad, val, tracked }
    }
}

impl_assign!(AddAssign, add_assign, +);

impl Sub for DualNode {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let val = self.val - rhs.val;
        let tracked = self.tracked || rhs.tracked;

        if !tracked {
            return Self { grad: self.grad, val, tracked };
        }

        let mut grad = [0.0f32; DUAL_N];

        if self.tracked && rhs.tracked {
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| ga - gb);
        } else if self.tracked {
            grad.copy_from_slice(&self.grad);
        } else {
            grad_map1(&rhs.grad, &mut grad, |gb| F32x8::zero() - gb);
        }

        Self { grad, val, tracked }
    }
}

impl_assign!(SubAssign, sub_assign, -);

impl Mul for DualNode {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let val = self.val * rhs.val;
        let tracked = self.tracked || rhs.tracked;

        if !tracked {
            return Self { grad: self.grad, val, tracked };
        }

        let a = self.val as f32;
        let b = rhs.val as f32;

        let mut grad = [0.0f32; DUAL_N];

        if self.tracked && rhs.tracked {
            let (va, vb) = (F32x8::splat(a), F32x8::splat(b));
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| vb * ga + va * gb);
        } else if self.tracked {
            let vb = F32x8::splat(b);
            grad_map1(&self.grad, &mut grad, |ga| vb * ga);
        } else {
            let va = F32x8::splat(a);
            grad_map1(&rhs.grad, &mut grad, |gb| va * gb);
        }

        Self { grad, val, tracked }
    }
}

impl_assign!(MulAssign, mul_assign, *);

impl Div for DualNode {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        let tracked = self.tracked || rhs.tracked;

        if !tracked {
            return Self { grad: self.grad, val, tracked };
        }

        let b = rhs.val as f32;
        let b2 = b * b;
        let a = self.val as f32;

        let mut grad = [0.0f32; DUAL_N];

        if self.tracked && rhs.tracked {
            let (va, vb, vb2) = (F32x8::splat(a), F32x8::splat(b), F32x8::splat(b2));
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| (vb * ga - va * gb) / vb2);
        } else if self.tracked {
            let vb = F32x8::splat(b);
            grad_map1(&self.grad, &mut grad, |ga| ga / vb);
        } else {
            let (va, vb2) = (F32x8::splat(-a), F32x8::splat(b2));
            grad_map1(&rhs.grad, &mut grad, |gb| (va * gb) / vb2);
        }

        Self { grad, val, tracked }
    }
}

impl_assign!(DivAssign, div_assign, /);

impl Neg for DualNode {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        if !self.tracked {
            return Self { grad: self.grad, val: -self.val, tracked: false };
        }

        let mut grad = [0.0f32; DUAL_N];
        grad_map1(&self.grad, &mut grad, |ga| F32x8::zero() - ga);
        Self { grad, val: -self.val, tracked: self.tracked }
    }
}

impl EvalMath for DualNode {
    type Scalar = DualNode;
    type Vec4 = DualVec4;
    type Vec8 = DualVec8;
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

    #[inline(always)]
    fn load_array6(values: &[f64], offset: usize, slot: &mut usize) -> Self::Array6 {
        let out = core::array::from_fn(|i| DualNode::seed(values[offset + i], *slot + i));
        *slot += 6;
        out
    }

    #[inline(always)]
    fn zero() -> Self { DualNode::zero() }

    #[inline(always)]
    fn from_i32(val: i32) -> Self { DualNode::constant(f64::from(val)) }

    #[inline(always)]
    fn trunc(self) -> Self { Self { val: self.val.trunc(), grad: self.grad, tracked: self.tracked } }

    #[inline(always)]
    fn taper_acc(acc: &Self::Vec8, phase: Self) -> Self {
        let mg = acc.0[0];
        let eg = acc.0[1];
        let eg_p = Self::from_i32(TOTAL_PHASE) - phase;
        let tot = Self::from_i32(TOTAL_PHASE);
        ((mg * phase + eg * eg_p) / tot).trunc()
    }

    // array::map adds a 1256-byte stack frame and three memcpy calls in the conversion probe.
    #[inline(always)]
    fn from_i32_array(arr: [i32; 4]) -> Self::Vec4 {
        DualVec4([Self::from_i32(arr[0]), Self::from_i32(arr[1]), Self::from_i32(arr[2]), Self::from_i32(arr[3])])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DualVec4([DualNode; 4]);

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

impl EvalVec4 for DualVec4 {
    type Scalar = DualNode;
    type Vec8 = DualVec8;

    #[inline(always)]
    fn splat(val: i32) -> Self {
        let n = DualNode::constant(f64::from(val));
        DualVec4([n, n, n, n])
    }

    #[inline(always)]
    fn from_lanes(a: DualNode, b: DualNode, c: DualNode, d: DualNode) -> Self { DualVec4([a, b, c, d]) }

    #[inline(always)]
    fn extract<const N: i32>(self) -> DualNode { self.0[N as usize] }

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
        DualVec8([
            self.0[0].clamp_i16(),
            self.0[1].clamp_i16(),
            self.0[2].clamp_i16(),
            self.0[3].clamp_i16(),
            hi.0[0].clamp_i16(),
            hi.0[1].clamp_i16(),
            hi.0[2].clamp_i16(),
            hi.0[3].clamp_i16(),
        ])
    }
}

#[derive(Clone, Copy)]
pub(crate) struct DualVec8(pub(crate) [DualNode; 8]);

impl EvalVec8 for DualVec8 {
    type Vec4 = DualVec4;

    #[inline(always)]
    fn zero() -> Self { DualVec8([DualNode::zero(); 8]) }

    #[inline(always)]
    fn madd(self, rhs: Self) -> DualVec4 {
        DualVec4([
            self.0[0] * rhs.0[0] + self.0[1] * rhs.0[1],
            self.0[2] * rhs.0[2] + self.0[3] * rhs.0[3],
            self.0[4] * rhs.0[4] + self.0[5] * rhs.0[5],
            self.0[6] * rhs.0[6] + self.0[7] * rhs.0[7],
        ])
    }
}
