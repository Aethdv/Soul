//! Lion: "Evolved Sign Momentum".
//!
//! Unlike Adam, which carefully tracks variance to scale every step,
//! Lion takes the sign of a blend of momentum and gradient to step
//! parameters at a uniform rate. Ours asks a further question first,
//! "Where do momentum and gradient agree?", holding the step where they
//! clash; the exact rule is on [`Lion::gate`].
//!
//! *Ref: Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang,
//! Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*
//! Symbolic Discovery of Optimization Algorithms. NeurIPS 2023.
//! <https://arxiv.org/abs/2302.06675v4>

use std::ops::Range;

use super::groups::{ParamGroup, param_group};

pub struct Lion {
    interp: f64, // β₁ (interpolation weight)
    lr: f64,
    wd: f64,
    /// One EMA of past gradients per slot. Owned here rather than by the caller:
    /// it is the algorithm's state, and what it holds decides how it rescales.
    momentum: Vec<f64>,
    /// Updates the clip bound truncated, per slot. Without the count a pinned parameter
    /// reads as a converged one, and `Δp` agrees.
    clipped: Vec<u64>,
    /// Σ of the `eff_lr` every sign step spent, since the last read. Gate width and step
    /// length are the same lever, `‖Δθ‖₁ = eff_lr · (stepping count)`.
    step_l1: f64,
}

/// Why the coordinates of one `update` call did or did not take their sign step.
///
/// Counted over parameter-updates, not parameters: a 500-batch epoch votes 500 times per
/// parameter. The counts are also the step length, since `‖Δθ‖₁ = eff_lr · (total − skipped −
/// dead)` and every stepping coordinate moves the same distance, so any gate that skips less
/// also steps further. That coupling is what made the cautious-mask retunes unreadable, and
/// these counts are what price a correction for it.
#[derive(Clone, Copy, Default)]
pub struct GateCensus {
    pub total: u64,
    /// Skipped: the gradient disagrees with momentum that clears the epsilon.
    pub skipped: u64,
    /// Skipped: `|c|` under the dead zone, no direction to take.
    pub dead: u64,
    /// Stepped only because `|m|` sat under the gate's epsilon, gradient disagreeing.
    pub epsilon_waived: u64,
    /// Liang's canonical mask would skip here, whatever ours did.
    pub canonical: u64,
    /// Ours skips and Liang's does not.
    pub band: u64,
    /// Liang's skips and ours steps, the other and larger direction of the same difference.
    pub canonical_only: u64,
    /// No gradient reached this parameter in this batch.
    pub absent: u64,
}

impl Lion {
    #[must_use]
    pub fn new(slots: usize, interp: f64, lr: f64, wd: f64) -> Self {
        Self { interp, lr, wd, momentum: vec![0.0; slots], clipped: vec![0; slots], step_l1: 0.0 }
    }

    #[must_use]
    pub fn momentum(&self) -> &[f64] {
        &self.momentum
    }

    #[must_use]
    pub fn clipped(&self) -> &[u64] {
        &self.clipped
    }

    /// The L1 distance the sign steps travelled since the last call, and zero it.
    /// Weight decay is excluded: it moves θ under every gate alike.
    pub fn take_step_l1(&mut self) -> f64 {
        std::mem::take(&mut self.step_l1)
    }

    pub fn restore_momentum(&mut self, momentum: &[f64]) {
        debug_assert_eq!(self.momentum.len(), momentum.len());
        self.momentum.copy_from_slice(momentum);
    }

    /// Rescale the trail when the parameter vector is rescaled under it.
    ///
    /// A slot's gradient scales by the reciprocal of whatever the slot took,
    /// and momentum is an EMA of gradients, so it follows or the gate reads
    /// stale signs. An optimizer holding squared gradients would take the square.
    pub fn rescale<F: Fn(usize) -> f64>(&mut self, slot_factor: F) {
        for (i, m) in self.momentum.iter_mut().enumerate() {
            *m /= slot_factor(i);
        }
    }

    #[inline]
    pub const fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Tallies what [`Lion::update`] is about to decide, over the same momentum and gradients.
    ///
    /// Call it before the update, while `momentum` still holds the values the gate will read.
    /// Groups are contiguous in the parameter layout, so a per-group tally is this over a
    /// subslice.
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
            census.absent += u64::from(g.abs() < 1e-9);
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

    pub fn update(
        &mut self,
        params: &mut [f64],
        gradients: &[f64],
        decay_mask: &[f64],
        fixed_mask: &[bool],
        beta2: &[f64],
        lr_mask: &[f64],
        clip_mask: &[(f64, f64)],
    ) {
        debug_assert_eq!(params.len(), self.momentum.len());
        debug_assert_eq!(params.len(), gradients.len());
        debug_assert_eq!(params.len(), decay_mask.len());
        debug_assert_eq!(params.len(), fixed_mask.len());
        debug_assert_eq!(params.len(), beta2.len());
        debug_assert_eq!(params.len(), lr_mask.len());
        debug_assert_eq!(params.len(), clip_mask.len());

        for i in 0..params.len() {
            if fixed_mask[i] {
                continue;
            }

            let p = params[i];
            let m = self.momentum[i];
            let g = gradients[i];
            let d = decay_mask[i];
            let eff_lr = self.lr * lr_mask[i];

            // 1. Interpolation, the search direction: c = β₁ · m + (1 - β₁) · g, and the
            //    gate's verdict on it.
            let (c, verdict) = self.gate(m, g);

            // 2. Parameter update: θ = θ - lr · (sign(c) + wd · θ). The magnitude is lr in
            //    every dimension, the sign alone decides direction. Decay fires whatever the
            //    gate says: a converged parameter without gradient signal should not lose its
            //    regularization pressure along with its step.
            let decayed = eff_lr.mul_add(-self.wd * d * p, p);

            let updated = if verdict == Gate::Step {
                self.step_l1 += eff_lr;
                decayed - eff_lr * c.signum()
            } else {
                decayed
            };

            // 3. Weight clipping, into the slot's own range: unbounded for most parameters
            //    and ±100 for mobility, whose features have no ceiling of their own.
            let (min, max) = clip_mask[i];
            let clamped = updated.clamp(min, max);

            self.clipped[i] += u64::from(clamped != updated);
            params[i] = clamped;

            // 4. Momentum tracking for the next step: m = β₂ · m + (1 - β₂) · g, β₂ per
            //    parameter. Hard-zero when both are essentially zero: floating-point residue
            //    in m otherwise accumulates and triggers sign steps on converged parameters.
            self.momentum[i] = if g.abs() < 1e-9 && m.abs() < 1e-9 { 0.0 } else { beta2[i].mul_add(m, (1.0 - beta2[i]) * g) };
        }
    }

    /// The blended direction `c = β₁·m + (1−β₁)·g`, and what the gate does with it.
    ///
    /// Three outcomes. Dead is `|c|` under the zone where a direction stops meaning
    /// anything, and it has to be a case of its own because `sign(0.0)` is `1.0`, which
    /// would walk every quiet parameter positive forever. Held is momentum and gradient
    /// disagreeing, local oscillation worth sitting out; an absent gradient counts as
    /// disagreement, where a signum test would let one momentum sign coast. The
    /// `|m| > 1e-6` clause is what stops a zero-momentum parameter from deferring its
    /// own first step.
    ///
    /// Ref: Kaizhao Liang, Lizhang Chen, Bo Liu & Qiang Liu (2024).
    /// Cautious Optimizers: Improving Training with One Line of Code.
    /// <https://arxiv.org/abs/2411.16085v4>
    ///
    /// Liang's canonical mask skips on c·g ≤ 0, which at β₁ = 0.9 reads m·g ≤ -g²/9:
    /// a strict subset of ours, so it steps on reversals we hold. Attempted at 490 HCE
    /// parameters, two retunes on separate seeds:
    ///
    /// ```text
    ///   Elo   | -6.24 ± 6.43 (95%)
    ///   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
    ///   LLR   | -2.54 (-2.47, 2.91) [0.00, 5.00]
    ///   Games | N: 5286 W: 1492 L: 1587 D: 2207
    ///   <https://asylum.red/test/5761/>
    ///
    ///   Elo   | -1.53 ± 4.13 (95%)
    ///   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
    ///   LLR   | -2.50 (-2.47, 2.91) [0.00, 5.00]
    ///   Games | N: 12734 W: 3658 L: 3714 D: 5362
    ///   <https://asylum.red/test/5762/>
    /// ```
    ///
    /// Liang pairs the mask with a φ/mean(φ) rescale, which we skip: it would set the
    /// surviving step to lr·dim/nnz, forfeiting the uniform magnitude Lion is built on
    /// and pricing every coordinate off a global statistic. Skipping it is not free.
    /// Gate width sets ‖Δθ‖₁ directly, so a wider gate is also a longer step, and the
    /// two runs above differ in step length as well as in mask shape. Any retry pins
    /// one of the two, or it buys another confounded result.
    ///
    /// Ref: Taejong Joo, Wenhan Xia, Cheolmin Kim, Ming Zhang & Eugene Ie (2026).
    /// On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
    /// <https://arxiv.org/abs/2602.15322v1>
    ///
    /// Magma scores per parameter block. Ours collapsed that to one global cossim over
    /// 430 HCE parameters, 384 of them PSQT, which set the gate for everything else.
    ///
    /// ```text
    ///   Elo   | -0.67 ± 5.39 (95%)
    ///   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
    ///   LLR   | -1.17 (-2.47, 2.91) [0.00, 5.00]
    ///   Games | N: 8240 W: 2499 L: 2515 D: 3226
    ///   <https://asylum.red/test/4378/>
    /// ```
    ///
    /// TODO: Revisit per-group, with more HCE terms or at NNUE scale.
    #[inline(always)]
    fn gate(&self, m: f64, g: f64) -> (f64, Gate) {
        let c = self.interp.mul_add(m, (1.0 - self.interp) * g);

        let verdict = if c.abs() < 1e-9 {
            Gate::Dead
        } else if m * g <= 0.0 && m.abs() > 1e-6 {
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

    /// Fraction of a count against the parameter-updates counted, zero on an empty census.
    #[must_use]
    pub fn share(&self, count: u64) -> f64 {
        if self.total == 0 { 0.0 } else { count as f64 / self.total as f64 }
    }

    #[must_use]
    pub fn percent(&self, count: u64) -> f64 {
        100.0 * self.share(count)
    }

    /// The share of updates that stepped, which is Liang's φ. Theirs falls to about 0.55
    /// on a 100M-parameter model at batch 4096; ours sits near 1.0.
    ///
    /// Read ours carefully. The `|m| > 1e-6` clause waives a coordinate whose momentum
    /// never clears it, so those step and count as active however loudly they disagree,
    /// and φ under this gate reports how often momentum was large enough to be judged.
    /// Under the epsilon-free `c·g` form it reports disagreement itself.
    #[must_use]
    pub fn active_share(&self) -> f64 {
        self.share(self.total.saturating_sub(self.skipped + self.dead))
    }
}

/// Per-group momentum decay mask.
///
/// Different parameter groups have different natural gradient timescales.
/// - PSQT (0.995): squares only see updates when a piece of that type lands
///   there: longer momentum smooths sparse signal across positions.
/// - Mobility (0.95): features are computed every position; shorter momentum
///   lets weights track the faster dynamics without lag.
/// - Everything else: the configured default.
pub fn build_beta2_mask(slots: usize, default_beta2: f64) -> Vec<f64> {
    (0..slots)
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => 0.995,
            ParamGroup::Mobility => 0.95,
            ParamGroup::Material | ParamGroup::Other => default_beta2,
        })
        .collect()
}

/// What the gate decided for one coordinate.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Gate {
    /// `|c|` under the dead zone: no direction to take.
    Dead,
    /// The gradient disagrees with momentum that clears the epsilon.
    Held,
    /// Move by `sign(c)`.
    Step,
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unbounded parameter, for every test whose subject is not the clamp.
    const OPEN: [(f64, f64); 1] = [(f64::NEG_INFINITY, f64::INFINITY)];

    #[test]
    fn census_separates_every_gate_outcome() {
        // One parameter per outcome, in order: agreeing step, gated skip, skip inside the band
        // where Liang would have stepped, epsilon waiver, dead zone, an epsilon waiver Liang
        // would have caught, and a fixed parameter that must not be counted at all.
        let momentum = [0.5, 0.5, 0.001, 1e-9, 0.0, -1e-6, 0.5];
        let gradients = [1.0, -1.0, -1.0, -1.0, 0.0, 1e-6, -1.0];
        let fixed_mask = [false, false, false, false, false, false, true];

        let mut lion = Lion::new(momentum.len(), 0.9, 1.0, 0.0);
        lion.restore_momentum(&momentum);

        let c = lion.census(0..momentum.len(), &gradients, &fixed_mask);

        assert_eq!(c.total, 6, "the fixed parameter must not be counted");
        assert_eq!(c.skipped, 2, "gated: momentum over the epsilon disagreeing with the gradient");
        assert_eq!(c.band, 1, "of those, one has c·g > 0 and would have stepped under Liang's mask");
        assert_eq!(c.epsilon_waived, 2, "momentum under 1e-6 steps against a disagreeing gradient");
        assert_eq!(c.canonical_only, 1, "one of those waivers has c·g ≤ 0 and Liang's mask would hold it");
        assert_eq!(c.dead, 1);
        assert_eq!(c.absent, 1);
        assert_eq!(c.canonical, 3, "the out-of-band skip, the dead zone, and the caught waiver");
    }

    #[test]
    fn lion_clipping_works() {
        let mut params = vec![1.5];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];
        let clip_mask = vec![(-2.0, 2.0)];

        let grads_neg = vec![-1.0];
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);
        opt.restore_momentum(&[-0.5]);
        let beta2 = vec![0.99];

        let lr_mask = vec![1.0; params.len()];
        opt.update(&mut params, &grads_neg, &decay_mask, &fixed_mask, &beta2, &lr_mask, &clip_mask);
        assert!((params[0] - 2.0).abs() < 1e-9, "Should be clipped to max: {}", params[0]);

        // Test sparse path clipping
        let mut params_sparse = vec![10.0];
        let grads_sparse = vec![0.0];
        let lr_mask2 = vec![1.0; params_sparse.len()];
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);

        opt.update(&mut params_sparse, &grads_sparse, &decay_mask, &fixed_mask, &beta2, &lr_mask2, &clip_mask);

        assert!((params_sparse[0] - 2.0).abs() < 1e-9, "Sparse update should still clip: {}", params_sparse[0]);
    }

    #[test]
    fn lion_zero_gradient_does_not_step() {
        // g = 0 must not step either momentum sign.
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];
        let beta2 = vec![0.99];
        let lr_mask = vec![1.0];
        let mut opt = Lion::new(1, 0.9, 1.0, 0.0);

        for m0 in [0.5, -0.5] {
            let mut params = vec![1.0];
            opt.restore_momentum(&[m0]);
            opt.update(&mut params, &[0.0], &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);
            assert!((params[0] - 1.0).abs() < 1e-12, "g=0 must not step (m={m0}): {}", params[0]);
        }
    }

    #[test]
    fn lion_no_clipping_by_default() {
        let mut params = vec![100.0];
        let grads = vec![0.0];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        let mut opt = Lion::new(1, 0.9, 0.1, 0.0);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        // c = β₁ · m + (1 - β₁) · g = 0.9 · 0.0 + 0.1 · 0.0 = 0
        assert!((params[0] - 100.0).abs() < 0.01, "No clipping: {}", params[0]);
    }

    #[test]
    fn lion_zero_gradient_still_applies_weight_decay() {
        // c ≈ 0, g ≈ 0: no sign update, but weight decay must still fire.
        let mut params = vec![10.0];
        let grads = vec![0.0];
        let decay_mask = vec![1.0]; // Full decay on this slot
        let fixed_mask = vec![false];

        // lr=1.0, wd=0.05, d=1.0 → decay = 1.0 · 0.05 · 1.0 · 10.0 = 0.5
        let mut opt = Lion::new(1, 0.9, 1.0, 0.05);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        let expected = 10.0 - 0.5;
        assert!((params[0] - expected).abs() < 1e-6, "Expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_sign_update_unchanged() {
        // When c is nonzero, behavior should be identical to before the fix.
        let mut params = vec![1.0];
        let grads = vec![1.0]; // g=1.0, m=0 → c = 0.9·0 + 0.1·1 = 0.1
        let decay_mask = vec![0.5];
        let fixed_mask = vec![false];

        // lr=0.2, wd=0.01, d=0.5 → decay = 0.2·0.01·0.5·1.0 = 0.001
        // sign update = 0.2 · sign(0.1) = 0.2
        // net: 1.0 - 0.001 - 0.2 = 0.799
        let mut opt = Lion::new(1, 0.9, 0.2, 0.01);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        let expected = 1.0 - 0.001 - 0.2;
        assert!((params[0] - expected).abs() < 1e-6, "Expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_skips_sign_update_on_momentum_gradient_disagreement() {
        // momentum and gradient point in opposite directions → skip.
        let mut params = vec![5.0];
        let grads = vec![-1.0];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        // c = 0.9·0.5 + 0.1·(-1.0) = 0.45 - 0.1 = 0.35 > 0 → would normally update
        // but m.signum() ≠ g.signum() → skip sign step
        let mut opt = Lion::new(1, 0.9, 0.1, 0.0);
        opt.restore_momentum(&[0.5]);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        // No weight decay (wd=0, d=1.0), sign update skipped → no change.
        assert!((params[0] - 5.0).abs() < 1e-9, "Expected no change, got {}", params[0]);
    }
}
