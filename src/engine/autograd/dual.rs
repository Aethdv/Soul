//! Forward-mode automatic differentiation via dual numbers for the HCE evaluation graph.
//!
//! # Parameters
//! Differentiates with respect to the two accumulator lanes and all [`EvalParams`] weights.
//! Dual numbers store partial derivatives in a `[f32; DUAL_N]` array, sized at compile time
//! from the total parameter count.
//!
//! # Purpose
//! Serves as a reference implementation (oracle) for testing. Because the evaluation is
//! linear in its parameters, production training uses `eval_linear_grad` (direct feature
//! extraction, ~30× cheaper). Any divergence between the two indicates a bug in the
//! analytical gradient.
//!
//! # Implementation Details
//! Values track an `active` flag to indicate non-zero derivatives. Operations with constant
//! operands bypass gradient propagation entirely.

use std::{
    fmt,
    ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

use super::traits::{EnvVec4, EnvVec8, EvalMath};
use crate::{core::defs::TOTAL_PHASE, engine::eval_params, weave::Vf32x8};

/// Partials per dual number; the tunable input count rounded up to a multiple
/// of 8 for the AVX2 chunk loop. Auto-grows when an eval term is added.
pub const DUAL_N: usize = (eval_params::DUAL_SLOTS + 7) & !7;

/// `active` is `false` exactly when every gradient slot is zero (a constant),
/// which lets the operators skip gradient work in 𝒪(1).
#[derive(Clone, Copy)]
#[repr(C, align(32))]
pub struct DualNode {
    pub grad: [f32; DUAL_N],
    pub val: f64,
    pub active: bool,
}

impl DualNode {
    #[inline(always)]
    pub fn floor(self) -> Self {
        Self { val: self.val.floor(), grad: self.grad, active: self.active }
    }

    #[inline(always)]
    pub fn constant(val: f64) -> Self {
        Self { grad: [0.0; DUAL_N], val, active: false }
    }

    /// Seed a dual variable; `grad[idx] = 1.0`, active bit set.
    #[inline(always)]
    pub fn seed(val: f64, idx: usize) -> Self {
        let mut grad = [0.0f32; DUAL_N];
        grad[idx] = 1.0;
        Self { grad, val, active: true }
    }

    #[inline(always)]
    pub fn zero() -> Self {
        Self { grad: [0.0; DUAL_N], val: 0.0, active: false }
    }
}

impl fmt::Debug for DualNode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Dual({:.4}, active={})", self.val, self.active)
    }
}

/// Map `f` lane-wise across two gradient vectors into `out`, in `DUAL_N / 8`
/// chunks of 8 f32. The single unsafe surface every binary operator's gradient shares.
#[inline(always)]
fn grad_map2(a: &[f32; DUAL_N], b: &[f32; DUAL_N], out: &mut [f32; DUAL_N], f: impl Fn(Vf32x8, Vf32x8) -> Vf32x8) {
    // SAFETY: a, b, out are each exactly DUAL_N f32s, and DUAL_N is a multiple
    // of 8. Each iteration's stride-8 load/store touches offsets [off, off+8),
    // and the last is [DUAL_N-8, DUAL_N), never out of bounds.
    unsafe {
        let (pa, pb, po) = (a.as_ptr(), b.as_ptr(), out.as_mut_ptr());
        for i in 0..DUAL_N / 8 {
            let off = i * 8;
            let ga = Vf32x8::loadu(pa.add(off));
            let gb = Vf32x8::loadu(pb.add(off));
            f(ga, gb).storeu(po.add(off));
        }
    }
}

/// Unary counterpart of [`grad_map2`] for single-input gradients (Neg, scale).
#[inline(always)]
fn grad_map1(a: &[f32; DUAL_N], out: &mut [f32; DUAL_N], f: impl Fn(Vf32x8) -> Vf32x8) {
    // SAFETY: a, out are each exactly DUAL_N f32s, DUAL_N a multiple of 8; see grad_map2.
    unsafe {
        let (pa, po) = (a.as_ptr(), out.as_mut_ptr());
        for i in 0..DUAL_N / 8 {
            let off = i * 8;
            f(Vf32x8::loadu(pa.add(off))).storeu(po.add(off));
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
        let active = self.active || rhs.active;
        if !active {
            return Self { grad: self.grad, val, active };
        }

        let mut grad = [0.0f32; DUAL_N];
        if self.active && rhs.active {
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| ga + gb);
        } else if self.active {
            grad.copy_from_slice(&self.grad);
        } else {
            grad.copy_from_slice(&rhs.grad);
        }
        Self { grad, val, active }
    }
}

impl_assign!(AddAssign, add_assign, +);

impl Sub for DualNode {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        let val = self.val - rhs.val;
        let active = self.active || rhs.active;
        if !active {
            return Self { grad: self.grad, val, active };
        }

        let mut grad = [0.0f32; DUAL_N];
        if self.active && rhs.active {
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| ga - gb);
        } else if self.active {
            grad.copy_from_slice(&self.grad);
        } else {
            grad_map1(&rhs.grad, &mut grad, |gb| Vf32x8::zero() - gb);
        }
        Self { grad, val, active }
    }
}

impl_assign!(SubAssign, sub_assign, -);

impl Mul for DualNode {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        let val = self.val * rhs.val;
        let active = self.active || rhs.active;
        if !active {
            return Self { grad: self.grad, val, active };
        }

        let a = self.val as f32;
        let b = rhs.val as f32;

        let mut grad = [0.0f32; DUAL_N];
        if self.active && rhs.active {
            let (va, vb) = (Vf32x8::splat(a), Vf32x8::splat(b));
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| vb * ga + va * gb);
        } else if self.active {
            let vb = Vf32x8::splat(b);
            grad_map1(&self.grad, &mut grad, |ga| vb * ga);
        } else {
            let va = Vf32x8::splat(a);
            grad_map1(&rhs.grad, &mut grad, |gb| va * gb);
        }
        Self { grad, val, active }
    }
}

impl_assign!(MulAssign, mul_assign, *);

impl Div for DualNode {
    type Output = Self;

    #[inline(always)]
    fn div(self, rhs: Self) -> Self {
        let val = self.val / rhs.val;
        let active = self.active || rhs.active;
        if !active {
            return Self { grad: self.grad, val, active };
        }

        let b = rhs.val as f32;
        let b2 = b * b;
        let a = self.val as f32;

        let mut grad = [0.0f32; DUAL_N];
        if self.active && rhs.active {
            let (va, vb, vb2) = (Vf32x8::splat(a), Vf32x8::splat(b), Vf32x8::splat(b2));
            grad_map2(&self.grad, &rhs.grad, &mut grad, |ga, gb| (vb * ga - va * gb) / vb2);
        } else if self.active {
            let vb = Vf32x8::splat(b);
            grad_map1(&self.grad, &mut grad, |ga| ga / vb);
        } else {
            let (va, vb2) = (Vf32x8::splat(-a), Vf32x8::splat(b2));
            grad_map1(&rhs.grad, &mut grad, |gb| (va * gb) / vb2);
        }
        Self { grad, val, active }
    }
}

impl_assign!(DivAssign, div_assign, /);

impl Neg for DualNode {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self {
        if !self.active {
            return Self { grad: self.grad, val: -self.val, active: false };
        }

        let mut grad = [0.0f32; DUAL_N];
        grad_map1(&self.grad, &mut grad, |ga| Vf32x8::zero() - ga);
        Self { grad, val: -self.val, active: self.active }
    }
}

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
            return Self { val: min.val, grad: [0.0; DUAL_N], active: false };
        }
        if self.val >= max.val {
            return Self { val: max.val, grad: [0.0; DUAL_N], active: false };
        }
        self
    }

    #[inline(always)]
    fn tapered(acc: &Self::Vec8, phase: Self) -> Self {
        let mg = acc.0[0];
        let eg = acc.0[1];
        let p = phase;
        let eg_p = Self::from_i32(TOTAL_PHASE) - phase;
        let tot = Self::from_i32(TOTAL_PHASE);
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
