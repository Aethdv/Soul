//! Lion: "Evolved Sign Momentum".
//!
//! Adam scales every step by tracked variance; Lion takes the sign of a blend of momentum
//! and gradient, so every parameter moves the same distance whatever the gradient's size.
//! This implementation holds that step where momentum and gradient disagree.
//!
//! Chen et al., Symbolic Discovery of Optimization Algorithms. NeurIPS 2023.
//! <https://arxiv.org/abs/2302.06675v4>

use std::ops::Range;

use crate::engine::pct;

/// Magnitude below which a blend or a gradient is floating-point residue rather than a direction.
const DEAD_ZONE: f64 = 1e-9;

/// Momentum below this cannot outvote a fresh gradient.
const MOMENTUM_EPSILON: f64 = 1e-6;

pub struct Lion {
    beta1: f64,
    lr: f64,
    wd: f64,
    /// Exponential moving average of past gradients per parameter slot.
    momentum: Vec<f64>,
    /// Cumulative boundary clipping events per parameter slot.
    clipped: Vec<u64>,
    /// Cumulative L1 distance traversed by active sign updates since the last query.
    step_l1: f64,
}

/// The per-parameter masks an update reads, one entry per slot in every field.
pub struct Masks {
    pub decay: Vec<f64>,
    pub fixed: Vec<bool>,
    pub beta2: Vec<f64>,
    pub lr: Vec<f64>,
    pub clip: Vec<(f64, f64)>,
}

impl Masks {
    /// Every slot at the same setting: none fixed, none clipped, unit decay and rate.
    #[must_use]
    pub fn uniform(slots: usize, beta2: f64) -> Self {
        Self {
            decay: vec![1.0; slots],
            fixed: vec![false; slots],
            beta2: vec![beta2; slots],
            lr: vec![1.0; slots],
            clip: vec![(f64::NEG_INFINITY, f64::INFINITY); slots],
        }
    }
}

/// Gating diagnostic counts aggregated over parameter-update events.
#[derive(Clone, Copy, Default)]
pub struct GateCensus {
    pub total: u64,
    /// Gated: gradient opposes momentum above the noise floor (`m · g ≤ 0` and `|m| > 1e-6`).
    pub skipped: u64,
    /// Gated: update magnitude below the deadband threshold (`|c| < 1e-9`).
    pub dead: u64,
    /// Stepped: gradient opposes momentum, but low momentum (`|m| ≤ 1e-6`) waived the gate.
    pub epsilon_waived: u64,
    /// Canonical cautious criterion would have held the step (`c · g ≤ 0`).
    pub canonical: u64,
    /// Gated by momentum conflict (`m · g ≤ 0`) where the canonical criterion would have stepped (`c · g > 0`).
    pub band: u64,
    /// Held by canonical condition (`c · g ≤ 0`), but stepped under momentum waiver (`|m| ≤ 1e-6`).
    pub canonical_only: u64,
    /// Zero or sub-threshold gradient (`|g| < 1e-9`).
    pub absent: u64,
}

impl Lion {
    #[must_use]
    pub fn new(slots: usize, beta1: f64, lr: f64, wd: f64) -> Self {
        Self { beta1, lr, wd, momentum: vec![0.0; slots], clipped: vec![0; slots], step_l1: 0.0 }
    }

    #[must_use]
    pub fn momentum(&self) -> &[f64] { &self.momentum }

    #[must_use]
    pub fn clipped(&self) -> &[u64] { &self.clipped }

    /// Returns and resets the accumulated L1 distance of sign updates since the previous call.
    /// Decoupled weight decay is excluded.
    pub fn take_step_l1(&mut self) -> f64 { std::mem::take(&mut self.step_l1) }

    pub fn restore_momentum(&mut self, momentum: &[f64]) {
        debug_assert_eq!(self.momentum.len(), momentum.len());
        self.momentum.copy_from_slice(momentum);
    }

    /// Divides momentum by the factor its parameters were scaled by. Momentum is carried in
    /// parameter units, so skipping it leaves every later step wrong by that factor.
    pub fn rescale<F: Fn(usize) -> f64>(&mut self, slot_scale: F) {
        for (i, m) in self.momentum.iter_mut().enumerate() {
            *m /= slot_scale(i);
        }
    }

    #[inline]
    pub const fn set_lr(&mut self, lr: f64) { self.lr = lr; }

    /// Tallies gating decisions over a parameter slice without modifying state.
    #[must_use]
    pub fn census(&self, range: Range<usize>, gradients: &[f64], fixed_mask: &[bool]) -> GateCensus {
        let momentum = &self.momentum[range.clone()];
        let gradients = &gradients[range.clone()];
        let fixed_mask = &fixed_mask[range];

        let mut census = GateCensus::default();

        for i in 0..momentum.len() {
            if fixed_mask[i] {
                continue;
            }

            let (m, g) = (momentum[i], gradients[i]);
            let (c, verdict) = self.gate(m, g);
            let disagrees = m * g <= 0.0;

            census.total += 1;
            census.absent += u64::from(g.abs() < DEAD_ZONE);
            census.canonical += u64::from(c * g <= 0.0);

            match verdict {
                Gate::Dead => census.dead += 1,
                Gate::Held => {
                    census.skipped += 1;
                    census.band += u64::from(c * g > 0.0);
                },
                Gate::Step => {
                    census.epsilon_waived += u64::from(disagrees);
                    census.canonical_only += u64::from(c * g <= 0.0);
                },
            }
        }
        census
    }

    pub fn update(&mut self, params: &mut [f64], gradients: &[f64], masks: &Masks) {
        debug_assert_eq!(params.len(), self.momentum.len());
        debug_assert_eq!(params.len(), gradients.len());
        debug_assert_eq!(params.len(), masks.decay.len());
        debug_assert_eq!(params.len(), masks.fixed.len());
        debug_assert_eq!(params.len(), masks.beta2.len());
        debug_assert_eq!(params.len(), masks.lr.len());
        debug_assert_eq!(params.len(), masks.clip.len());

        for i in 0..params.len() {
            if masks.fixed[i] {
                continue;
            }

            let param = params[i];
            let momentum = self.momentum[i];
            let gradient = gradients[i];
            let decay = masks.decay[i];
            let eff_lr = self.lr * masks.lr[i];

            // 1. Blended search direction and gate decision.
            let (c, verdict) = self.gate(momentum, gradient);

            // 2. Decoupled weight decay and directional sign step.
            let decayed = eff_lr.mul_add(-self.wd * decay * param, param);
            let updated = if verdict == Gate::Step {
                self.step_l1 += eff_lr;
                decayed - eff_lr * c.signum()
            } else {
                decayed
            };

            // 3. Bounds, counting the truncations: a pinned parameter otherwise reads as converged.
            let (min, max) = masks.clip[i];
            let clamped = updated.clamp(min, max);
            self.clipped[i] += u64::from(clamped != updated);
            params[i] = clamped;

            // 4. Momentum tracking: m = β₂ · m + (1 − β₂) · g.
            // Flush near-zero state to prevent accumulation of floating-point residue.
            self.momentum[i] = if gradient.abs() < DEAD_ZONE && momentum.abs() < DEAD_ZONE {
                0.0
            } else {
                masks.beta2[i].mul_add(momentum, (1.0 - masks.beta2[i]) * gradient)
            };
        }
    }

    /// Evaluates blended direction `c = β₁ · m + (1 − β₁) · g` and gating status.
    ///
    /// - `Dead`: `|c| < 1e-9` prevents updates driven by floating-point sign ambiguity.
    /// - `Held`: `m · g ≤ 0` and `|m| > 1e-6` skips updates during local oscillation.
    /// - `Step`: Direction confirmed, or momentum too small to outvote the gradient (`|m| ≤ 1e-6`).
    #[inline(always)]
    fn gate(&self, momentum: f64, gradient: f64) -> (f64, Gate) {
        let c = self.beta1.mul_add(momentum, (1.0 - self.beta1) * gradient);
        let verdict = if c.abs() < DEAD_ZONE {
            Gate::Dead
        } else if momentum * gradient <= 0.0 && momentum.abs() > MOMENTUM_EPSILON {
            Gate::Held
        } else {
            Gate::Step
        };
        (c, verdict)
    }
}

impl GateCensus {
    pub fn absorb(&mut self, other: Self) {
        self.total += other.total;
        self.skipped += other.skipped;
        self.dead += other.dead;
        self.epsilon_waived += other.epsilon_waived;
        self.canonical += other.canonical;
        self.band += other.band;
        self.canonical_only += other.canonical_only;
        self.absent += other.absent;
    }

    #[must_use]
    pub fn share(&self, count: u64) -> f64 { if self.total == 0 { 0.0 } else { count as f64 / self.total as f64 } }
    #[must_use]
    pub fn percent(&self, count: u64) -> f64 { pct(count, self.total) }
    /// Fraction of updates that executed a sign step (`1 - (skipped + dead) / total`).
    #[must_use]
    pub fn active_share(&self) -> f64 { self.share(self.total.saturating_sub(self.skipped + self.dead)) }
}

/// Gating verdict for a single coordinate update.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gate {
    /// Update magnitude below noise floor (`|c| < 1e-9`).
    Dead,
    /// Momentum and gradient directionally conflict (`m · g ≤ 0`) with `|m| > 1e-6`.
    Held,
    /// Step allowed along `sign(c)`.
    Step,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_separates_every_gate_outcome() {
        let momentum = [0.5, 0.5, 0.001, 1e-9, 0.0, -1e-6, 0.5];
        let gradients = [1.0, -1.0, -1.0, -1.0, 0.0, 1e-6, -1.0];
        let fixed_mask = [false, false, false, false, false, false, true];
        let mut lion = Lion::new(momentum.len(), 0.9, 1.0, 0.0);
        lion.restore_momentum(&momentum);
        let census = lion.census(0..momentum.len(), &gradients, &fixed_mask);
        assert_eq!(census.total, 6, "fixed parameter must be excluded");
        assert_eq!(census.skipped, 2, "gated: momentum above threshold opposing gradient");
        assert_eq!(census.band, 1, "in-band: gated by m·g <= 0 while c·g > 0");
        assert_eq!(census.epsilon_waived, 2, "momentum under threshold steps despite conflict");
        assert_eq!(census.canonical_only, 1, "waived coordinate where c·g <= 0");
        assert_eq!(census.dead, 1);
        assert_eq!(census.absent, 1);
        assert_eq!(census.canonical, 3);
    }

    #[test]
    fn lion_clipping_works() {
        let mut params = vec![1.5];
        let grads_neg = vec![-1.0];
        let masks = Masks { decay: vec![0.0], clip: vec![(-2.0, 2.0)], ..Masks::uniform(1, 0.99) };
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);
        opt.restore_momentum(&[-0.5]);
        opt.update(&mut params, &grads_neg, &masks);
        assert!((params[0] - 2.0).abs() < 1e-9, "parameter must clamp to upper bound: {}", params[0]);
        let mut params_sparse = vec![10.0];
        let grads_sparse = vec![0.0];
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);
        opt.update(&mut params_sparse, &grads_sparse, &masks);
        assert!((params_sparse[0] - 2.0).abs() < 1e-9, "sparse update must enforce clamp bounds: {}", params_sparse[0]);
    }

    #[test]
    fn lion_zero_gradient_does_not_step() {
        let masks = Masks { decay: vec![0.0], ..Masks::uniform(1, 0.99) };
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);
        for m0 in [0.5, -0.5] {
            let mut params = vec![1.0];
            opt.restore_momentum(&[m0]);
            opt.update(&mut params, &[0.0], &masks);
            assert!((params[0] - 1.0).abs() < 1e-12, "zero gradient must not take sign step (m={m0}): {}", params[0]);
        }
    }

    #[test]
    fn lion_no_clipping_by_default() {
        let mut params = vec![100.0];
        let grads = vec![0.0];
        let masks = Masks { decay: vec![0.0], ..Masks::uniform(1, 0.99) };
        let mut opt = Lion::new(1, 0.9, 0.1, 0.0);
        opt.update(&mut params, &grads, &masks);
        assert!((params[0] - 100.0).abs() < 0.01, "unbounded parameter must not clamp: {}", params[0]);
    }

    #[test]
    fn lion_zero_gradient_still_applies_weight_decay() {
        let mut params = vec![10.0];
        let grads = vec![0.0];
        // decay = lr · wd · d · p = 1.0 · 0.05 · 1.0 · 10.0 = 0.5
        let mut opt = Lion::new(1, 0.9, 1.0, 0.05);
        opt.update(&mut params, &grads, &Masks::uniform(1, 0.99));
        let expected = 10.0 - 0.5;
        assert!((params[0] - expected).abs() < 1e-6, "expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_sign_update_unchanged() {
        let mut params = vec![1.0];
        let grads = vec![1.0]; // g = 1.0, m = 0.0 → c = 0.1
        let masks = Masks { decay: vec![0.5], ..Masks::uniform(1, 0.99) };
        // decay = 0.2 · 0.01 · 0.5 · 1.0 = 0.001
        // sign update = 0.2 · sign(0.1) = 0.2
        // net: 1.0 - 0.001 - 0.2 = 0.799
        let mut opt = Lion::new(1, 0.9, 0.2, 0.01);
        opt.update(&mut params, &grads, &masks);
        let expected = 1.0 - 0.001 - 0.2;
        assert!((params[0] - expected).abs() < 1e-6, "expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_skips_sign_update_on_momentum_gradient_disagreement() {
        let mut params = vec![5.0];
        let grads = vec![-1.0];
        let masks = Masks { decay: vec![0.0], ..Masks::uniform(1, 0.99) };
        // c = 0.9 · 0.5 + 0.1 · (-1.0) = 0.35 (> 0), standard Lion would step.
        let mut opt = Lion::new(1, 0.9, 0.1, 0.0);
        opt.restore_momentum(&[0.5]);
        opt.update(&mut params, &grads, &masks);
        assert!((params[0] - 5.0).abs() < 1e-9, "directional conflict must hold step: {}", params[0]);
    }
}
