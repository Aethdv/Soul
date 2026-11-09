//! Lion ("Evolved Sign Momentum") with a momentum-gradient disagreement gate.
//!
//! AdamW carries first and second gradient moments (`m` and `v`) and scales each coordinate by
//! `m / √v`. Lion drops `v` and moves every active parameter by `±lr` along `sign(c)`, where
//! `c = β₁·m + (1 − β₁)·g` blends stored momentum with the current batch gradient before
//! `m ← β₂·m + (1 − β₂)·g` folds the gradient into history. Updating `m` second is what keeps
//! the two betas distinct: a smaller `β₁` lets one batch flip the step's sign while `β₂` holds
//! a longer window. Dropping `1 / √v` is also why sparse or wide-range groups have explicit
//! `lr` multipliers in [`Masks`].
//!
//! Cautious AdamW and cautious Lion mask the step when the update opposes the gradient (`c · g ≤ 0`).
//! Here the gate tests pre-blend momentum instead (`m · g ≤ 0` once `|m| > 1e-6`): a noisy batch with
//! `|g| > (β₁ / (1 − β₁))·|m|` pulls `c` onto `g`'s sign and would step under `c · g ≤ 0`, whereas
//! `m · g ≤ 0` withholds the sign step (decoupled weight decay still runs) until `m` turns with it.
//!
//! Chen et al. Symbolic Discovery of Optimization Algorithms. NeurIPS 2023.
//! <https://arxiv.org/abs/2302.06675v4>
//!
//! Liang et al. Cautious Optimizers: Improving Training with One Line of Code. 2024.
//! <https://arxiv.org/abs/2411.16085>

use std::ops::Range;

use crate::engine::pct;

const DEAD_ZONE: f64 = 1e-9;
const MOMENTUM_EPSILON: f64 = 1e-6;

pub(crate) struct Lion {
    beta1: f64,
    lr: f64,
    wd: f64,
    momentum: Vec<f64>,
    clipped: Vec<u64>,
    step_l1: f64,
}

pub(crate) struct Masks {
    pub(crate) decay: Vec<f64>,
    pub(crate) fixed: Vec<bool>,
    pub(crate) beta2: Vec<f64>,
    pub(crate) lr: Vec<f64>,
    pub(crate) clip: Vec<(f64, f64)>,
}

#[cfg(test)]
impl Masks {
    pub(crate) fn uniform(slots: usize, beta2: f64) -> Self {
        Self {
            decay: vec![1.0; slots],
            fixed: vec![false; slots],
            beta2: vec![beta2; slots],
            lr: vec![1.0; slots],
            clip: vec![(f64::NEG_INFINITY, f64::INFINITY); slots],
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(crate) struct GateTally {
    pub(crate) total: u64,
    /// Held because `m · g ≤ 0` with `|m| > 1e-6`.
    pub(crate) skipped: u64,
    /// Held because `|c| < 1e-9`.
    pub(crate) dead: u64,
    /// Stepped despite `m · g ≤ 0` because `|m| ≤ 1e-6`.
    pub(crate) epsilon_waived: u64,
    /// Liang's cautious mask (`c · g ≤ 0`) would have held the step.
    pub(crate) canonical: u64,
    /// Held by `m · g ≤ 0` where `c · g > 0` would have stepped.
    pub(crate) band: u64,
    /// Held by `c · g ≤ 0` but stepped under the `|m| ≤ 1e-6` waiver.
    pub(crate) canonical_only: u64,
    /// `|g| < 1e-9`.
    pub(crate) absent: u64,
}

impl Lion {
    pub(crate) fn new(slots: usize, beta1: f64, lr: f64, wd: f64) -> Self {
        Self { beta1, lr, wd, momentum: vec![0.0; slots], clipped: vec![0; slots], step_l1: 0.0 }
    }

    pub(crate) fn momentum(&self) -> &[f64] { &self.momentum }
    pub(crate) fn clipped(&self) -> &[u64] { &self.clipped }

    /// Drains accumulated L1 sign-step travel (excluding decoupled weight decay).
    pub(crate) fn take_step_l1(&mut self) -> f64 { std::mem::take(&mut self.step_l1) }

    pub(crate) fn restore_momentum(&mut self, momentum: &[f64]) {
        debug_assert_eq!(self.momentum.len(), momentum.len());
        self.momentum.copy_from_slice(momentum);
    }

    /// Rescales momentum contravariantly when parameter slot `i` is multiplied by `slot_scale(i)`
    /// at fixed `K · score`, since `∂L / ∂(s · θ) = (1 / s) · ∂L / ∂θ`.
    pub(crate) fn rescale<F: Fn(usize) -> f64>(&mut self, slot_scale: F) {
        for (i, m) in self.momentum.iter_mut().enumerate() {
            *m /= slot_scale(i);
        }
    }

    #[inline]
    pub(crate) const fn set_lr(&mut self, lr: f64) { self.lr = lr; }

    pub(crate) fn tally(&self, range: Range<usize>, grads: &[f64], fixed: &[bool]) -> GateTally {
        let mut c = GateTally::default();

        for i in range {
            if fixed[i] {
                continue;
            }

            let (m, g) = (self.momentum[i], grads[i]);
            let (blend, verdict) = self.gate(m, g);

            c.total += 1;
            c.absent += u64::from(g.abs() < DEAD_ZONE);
            c.canonical += u64::from(blend * g <= 0.0);

            match verdict {
                Gate::Dead => c.dead += 1,
                Gate::Held => {
                    c.skipped += 1;
                    c.band += u64::from(blend * g > 0.0);
                },
                Gate::Step => {
                    c.epsilon_waived += u64::from(m * g <= 0.0);
                    c.canonical_only += u64::from(blend * g <= 0.0);
                },
            }
        }
        c
    }

    pub(crate) fn update(&mut self, values: &mut [f64], grads: &[f64], masks: &Masks) {
        let n = values.len();
        debug_assert!(
            n == self.momentum.len()
                && n == grads.len()
                && n == masks.decay.len()
                && n == masks.fixed.len()
                && n == masks.beta2.len()
                && n == masks.lr.len()
                && n == masks.clip.len()
        );

        for i in 0..n {
            if masks.fixed[i] {
                continue;
            }

            let (v, m, g, eff_lr) = (values[i], self.momentum[i], grads[i], self.lr * masks.lr[i]);
            let (c, verdict) = self.gate(m, g);
            let decayed = eff_lr.mul_add(-self.wd * masks.decay[i] * v, v);

            let updated = if verdict == Gate::Step {
                self.step_l1 += eff_lr;
                decayed - eff_lr * c.signum()
            } else {
                decayed
            };

            let (min, max) = masks.clip[i];
            let clamped = updated.clamp(min, max);
            self.clipped[i] += u64::from(clamped != updated);
            values[i] = clamped;

            // Flush sub-threshold residue so untouched slots stay at exact 0.0 instead of subnormals.
            self.momentum[i] = if g.abs() < DEAD_ZONE && m.abs() < DEAD_ZONE {
                0.0
            } else {
                masks.beta2[i].mul_add(m, (1.0 - masks.beta2[i]) * g)
            };
        }
    }

    #[inline(always)]
    fn gate(&self, m: f64, g: f64) -> (f64, Gate) {
        let c = self.beta1.mul_add(m, (1.0 - self.beta1) * g);
        let verdict = if c.abs() < DEAD_ZONE {
            Gate::Dead
        } else if m * g <= 0.0 && m.abs() > MOMENTUM_EPSILON {
            Gate::Held
        } else {
            Gate::Step
        };
        (c, verdict)
    }
}

impl GateTally {
    pub(crate) fn absorb(&mut self, other: Self) {
        self.total += other.total;
        self.skipped += other.skipped;
        self.dead += other.dead;
        self.epsilon_waived += other.epsilon_waived;
        self.canonical += other.canonical;
        self.band += other.band;
        self.canonical_only += other.canonical_only;
        self.absent += other.absent;
    }

    pub(crate) fn percent(&self, count: u64) -> f64 { pct(count, self.total) }

    pub(crate) fn active_share(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.total.saturating_sub(self.skipped + self.dead) as f64 / self.total as f64
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Gate {
    Dead,
    Held,
    Step,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::training::l2_norm;

    #[test]
    fn tally_separates_gate_outcomes() {
        let momentum = [0.5, 0.5, 0.001, 1e-9, 0.0, -1e-6, 0.5];
        let grads = [1.0, -1.0, -1.0, -1.0, 0.0, 1e-6, -1.0];
        let fixed = [false, false, false, false, false, false, true];
        let mut lion = Lion::new(momentum.len(), 0.9, 1.0, 0.0);
        lion.restore_momentum(&momentum);
        let c = lion.tally(0..momentum.len(), &grads, &fixed);
        assert_eq!(
            (c.total, c.skipped, c.band, c.epsilon_waived, c.canonical_only, c.dead, c.absent, c.canonical),
            (6, 2, 1, 2, 1, 1, 1, 3)
        );
    }

    #[test]
    fn update_invariants() {
        let mut p = [1.0];
        Lion::new(1, 0.9, 0.2, 0.01).update(&mut p, &[1.0], &Masks { decay: vec![0.5], ..Masks::uniform(1, 0.99) });
        assert!((p[0] - 0.799).abs() < 1e-12, "got {}", p[0]);

        let no_decay = Masks { decay: vec![0.0], ..Masks::uniform(1, 0.99) };
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);
        for m0 in [0.5, -0.5] {
            let mut p = [1.0];
            opt.restore_momentum(&[m0]);
            opt.update(&mut p, &[0.0], &no_decay);
            assert_eq!(p[0], 1.0, "zero gradient stepped at m={m0}");
        }
        let mut p = [10.0];
        Lion::new(1, 0.9, 1.0, 0.05).update(&mut p, &[0.0], &Masks::uniform(1, 0.99));
        assert!((p[0] - 9.5).abs() < 1e-12);

        // Opposing gradient holds the step even when c = 0.9 * 0.5 - 0.1 = 0.35 > 0.
        let mut p = [5.0];
        let mut opt = Lion::new(1, 0.9, 0.1, 0.0);
        opt.restore_momentum(&[0.5]);
        opt.update(&mut p, &[-1.0], &no_decay);
        assert_eq!(p[0], 5.0);

        let mut p = [1.5, 10.0, 100.0, 2.0];
        let masks = Masks {
            decay: vec![0.0; 4],
            fixed: vec![false, false, false, true],
            clip: vec![(-2.0, 2.0), (-2.0, 2.0), (f64::NEG_INFINITY, f64::INFINITY), (-2.0, 2.0)],
            ..Masks::uniform(4, 0.99)
        };
        let mut opt = Lion::new(4, 0.9, 1.0, 0.0);
        opt.restore_momentum(&[-0.5, 0.0, 0.0, 0.0]);
        opt.update(&mut p, &[-1.0, 0.0, 0.0, 1.0], &masks);
        assert_eq!((p, opt.clipped()), ([2.0, 2.0, 100.0, 2.0], &[1, 1, 0, 0][..]));
    }

    #[test]
    fn rescale_transforms_momentum_contravariantly() {
        let mut lion = Lion::new(3, 0.9, 0.1, 0.0);
        lion.restore_momentum(&[1.0, 2.0, 3.0]);
        lion.rescale(|i| match i {
            0 => 2.0,
            1 => 0.5,
            _ => 1.0,
        });
        assert_eq!(lion.momentum(), [0.5, 4.0, 3.0]);
    }

    #[test]
    fn converges_on_quadratic() {
        let (mut params, masks, mut opt) = (vec![10.0; 10], Masks::uniform(10, 0.99), Lion::new(10, 0.9, 0.1, 0.0));
        for iter in 0..200 {
            let grads: Vec<f64> = params.iter().map(|&p| 2.0 * p).collect();
            opt.set_lr(0.1 * (1.0 - iter as f64 / 200.0).max(0.05));
            opt.update(&mut params, &grads, &masks);
        }
        let norm = l2_norm(&params);
        assert!(norm < 0.5, "did not converge from norm 31.6: {norm}");
    }
}
