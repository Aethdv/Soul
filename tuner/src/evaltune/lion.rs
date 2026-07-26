//! Lion: "Evolved Sign Momentum".
//!
//! Unlike Adam, which carefully tracks variance to scale every step,
//! Lion asks a simpler question: "Where do momentum and gradient agree?"
//! It takes the sign of their blend to step parameters at a uniform rate,
//! damping noise whenever the signals clash.
//!
//! *Ref: Xiangning Chen, Chen Liang, Da Huang, Esteban Real, Kaiyuan Wang,
//! Yao Liu, Hieu Pham, Xuanyi Dong, Thang Luong, Cho-Jui Hsieh, Yifeng Lu, Quoc V. Le*
//! Symbolic Discovery of Optimization Algorithms. NeurIPS 2023.
//! <https://arxiv.org/abs/2302.06675v4>
//!
//! The update rule consists of four steps per iteration t.
//!
//! 1. Interpolation (Search Direction):
//!    Combine current gradient (g) and previous momentum (m).
//!    c = β₁ · m + (1 - β₁) · g
//!
//! 2. Parameter Update:
//!    Update parameters (θ) using the sign of c.
//!    θ = θ - lr · (sign(c) + wd · θ)
//!
//!    NOTE: The update magnitude is uniform (lr) across all dimensions, determined
//!    solely by the sign.
//!
//! 3. Weight clipping:
//!    Clamp θ into its own configured range, unbounded for most parameters
//!    and ±100 for mobility, whose features have no ceiling of their own.
//!
//! 4. Momentum Tracking:
//!    Update the stored momentum (m) for the next step.
//!    m = β₂ · m + (1 - β₂) · g,
//!    where β₂ is per-parameter (PSQT 0.995, mobility 0.95, default 0.99).

pub struct Lion {
    interp: f64, // β₁ (interpolation weight)
    lr: f64,
    wd: f64,
}

impl Lion {
    #[must_use]
    pub const fn new(interp: f64, lr: f64, wd: f64) -> Self {
        Self { interp, lr, wd }
    }

    #[inline]
    pub const fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    pub fn update(
        &self,
        params: &mut [f64],
        momentum: &mut [f64],
        gradients: &[f64],
        decay_mask: &[f64],
        fixed_mask: &[bool],
        beta2: &[f64],
        lr_mask: &[f64],
        clip_mask: &[(f64, f64)],
    ) {
        debug_assert_eq!(params.len(), momentum.len());
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
            let m = momentum[i];
            let g = gradients[i];
            let d = decay_mask[i];
            let eff_lr = self.lr * lr_mask[i];

            // 1. Interpolation: c = β₁ · m + (1 - β₁) · g
            let c = self.interp.mul_add(m, (1.0 - self.interp) * g);

            // 2. Parameter update with sign gate and weight decay.
            //
            // When |c| is negligible the sign step is skipped, but weight decay
            // still fires: a converged parameter without gradient signal should
            // not lose its regularization pressure.
            //
            // Per-parameter disagreement gate (m · g ≤ 0) catches local oscillation:
            // if momentum and gradient disagree, the sign update is skipped for this
            // parameter. An absent gradient (g = 0) is disagreement; a signum test
            // would let one momentum sign coast.
            //
            // Ref: Kaizhao Liang, Lizhang Chen, Bo Liu & Qiang Liu (2024).
            // Cautious Optimizers: Improving Training with One Line of Code.
            // <https://arxiv.org/abs/2411.16085v4>
            //
            // Liang's canonical mask skips on c·g ≤ 0, which at β₁ = 0.9 reads m·g ≤ -g²/9:
            // a strict subset of ours, so it steps on reversals we hold. Attempted at
            // 490 HCE parameters, two retunes on separate seeds:
            //
            //   Elo   | -6.24 ± 6.43 (95%)
            //   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
            //   LLR   | -2.54 (-2.47, 2.91) [0.00, 5.00]
            //   Games | N: 5286 W: 1492 L: 1587 D: 2207
            //   <https://asylum.red/test/5761/>
            //
            //   Elo   | -1.53 ± 4.13 (95%)
            //   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
            //   LLR   | -2.50 (-2.47, 2.91) [0.00, 5.00]
            //   Games | N: 12734 W: 3658 L: 3714 D: 5362
            //   <https://asylum.red/test/5762/>
            //
            // Liang pairs the mask with a φ/mean(φ) rescale, which we skip: it would set the
            // surviving step to lr·dim/nnz, forfeiting the uniform magnitude Lion is built on
            // and pricing every coordinate off a global statistic. Skipping it is not free.
            // Gate width sets ‖Δθ‖₁ directly, so a wider gate is also a longer step, and the
            // two runs above differ in step length as well as in mask shape. Any retry pins
            // one of the two, or it buys another confounded result.
            //
            // Ref: Taejong Joo, Wenhan Xia, Cheolmin Kim, Ming Zhang & Eugene Ie (2026).
            // On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.
            // <https://arxiv.org/abs/2602.15322v1>
            //
            // Magma scores per parameter block. Ours collapsed that to one global cossim over
            // 430 HCE parameters, 384 of them PSQT, which set the gate for everything else.
            // However, I do admit that I was a bit too impatient that day.
            //
            //   Elo   | -0.67 ± 5.39 (95%)
            //   SPRT  | 8.0+0.08s Threads=1 Hash=16MB
            //   LLR   | -1.17 (-2.47, 2.91) [0.00, 5.00]
            //   Games | N: 8240 W: 2499 L: 2515 D: 3226
            //   <https://asylum.red/test/4378/>
            //
            // TODO: Revisit per-group, with more HCE terms or at NNUE scale.
            let decayed = eff_lr.mul_add(-self.wd * d * p, p);
            // Skip the Lion sign step when the correlation gate is open (c≈0)
            // or momentum and gradient disagree: either way, decay only.
            // The m.abs() guard prevents zero-momentum from deferring every parameter's first step.
            let updated = if c.abs() < 1e-9 || (m * g <= 0.0 && m.abs() > 1e-6) { decayed } else { decayed - eff_lr * c.signum() };

            // 3. Weight clipping
            let (min, max) = clip_mask[i];
            params[i] = updated.clamp(min, max);

            // 4. Momentum: m = β₂ · m + (1 - β₂) · g
            // Hard-zero momentum when both gradient and current momentum are essentially zero.
            // Without this, floating-point residuals in m can accumulate and trigger sign updates
            // on perfectly converged parameters: a kind of ghost-gradient effect.
            momentum[i] = if g.abs() < 1e-9 && m.abs() < 1e-9 { 0.0 } else { beta2[i].mul_add(m, (1.0 - beta2[i]) * g) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// One unbounded parameter, for every test whose subject is not the clamp.
    const OPEN: [(f64, f64); 1] = [(f64::NEG_INFINITY, f64::INFINITY)];

    #[test]
    fn lion_clipping_works() {
        let mut params = vec![1.5];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];
        let clip_mask = vec![(-2.0, 2.0)];

        let grads_neg = vec![-1.0];
        let opt = Lion::new(0.9, 1.0, 0.0);
        let mut momentum_neg = vec![-0.5];
        let beta2 = vec![0.99];

        let lr_mask = vec![1.0; params.len()];
        opt.update(&mut params, &mut momentum_neg, &grads_neg, &decay_mask, &fixed_mask, &beta2, &lr_mask, &clip_mask);
        assert!((params[0] - 2.0).abs() < 1e-9, "Should be clipped to max: {}", params[0]);

        // Test sparse path clipping
        let mut params_sparse = vec![10.0];
        let mut momentum_sparse = vec![0.0];
        let grads_sparse = vec![0.0];
        let lr_mask2 = vec![1.0; params_sparse.len()];

        opt.update(
            &mut params_sparse, &mut momentum_sparse, &grads_sparse, &decay_mask, &fixed_mask, &beta2, &lr_mask2, &clip_mask,
        );

        assert!((params_sparse[0] - 2.0).abs() < 1e-9, "Sparse update should still clip: {}", params_sparse[0]);
    }

    #[test]
    fn lion_zero_gradient_does_not_step() {
        // g = 0 must not step either momentum sign.
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];
        let beta2 = vec![0.99];
        let lr_mask = vec![1.0];
        let opt = Lion::new(0.9, 1.0, 0.0);

        for m0 in [0.5, -0.5] {
            let mut params = vec![1.0];
            let mut momentum = vec![m0];
            opt.update(&mut params, &mut momentum, &[0.0], &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);
            assert!((params[0] - 1.0).abs() < 1e-12, "g=0 must not step (m={m0}): {}", params[0]);
        }
    }

    #[test]
    fn lion_no_clipping_by_default() {
        let mut params = vec![100.0];
        let mut momentum = vec![0.0];
        let grads = vec![0.0];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        let opt = Lion::new(0.9, 0.1, 0.0);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        // c = β₁ · m + (1 - β₁) · g = 0.9 · 0.0 + 0.1 · 0.0 = 0
        assert!((params[0] - 100.0).abs() < 0.01, "No clipping: {}", params[0]);
    }

    #[test]
    fn lion_zero_gradient_still_applies_weight_decay() {
        // c ≈ 0, g ≈ 0: no sign update, but weight decay must still fire.
        let mut params = vec![10.0];
        let mut momentum = vec![0.0];
        let grads = vec![0.0];
        let decay_mask = vec![1.0]; // Full decay on this slot
        let fixed_mask = vec![false];

        // lr=1.0, wd=0.05, d=1.0 → decay = 1.0 · 0.05 · 1.0 · 10.0 = 0.5
        let opt = Lion::new(0.9, 1.0, 0.05);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        let expected = 10.0 - 0.5;
        assert!((params[0] - expected).abs() < 1e-6, "Expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_sign_update_unchanged() {
        // When c is nonzero, behavior should be identical to before the fix.
        let mut params = vec![1.0];
        let mut momentum = vec![0.0];
        let grads = vec![1.0]; // g=1.0, m=0 → c = 0.9·0 + 0.1·1 = 0.1
        let decay_mask = vec![0.5];
        let fixed_mask = vec![false];

        // lr=0.2, wd=0.01, d=0.5 → decay = 0.2·0.01·0.5·1.0 = 0.001
        // sign update = 0.2 · sign(0.1) = 0.2
        // net: 1.0 - 0.001 - 0.2 = 0.799
        let opt = Lion::new(0.9, 0.2, 0.01);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        let expected = 1.0 - 0.001 - 0.2;
        assert!((params[0] - expected).abs() < 1e-6, "Expected {expected}, got {}", params[0]);
    }

    #[test]
    fn lion_skips_sign_update_on_momentum_gradient_disagreement() {
        // momentum and gradient point in opposite directions → skip.
        let mut params = vec![5.0];
        let mut momentum = vec![0.5]; // positive momentum
        let grads = vec![-1.0]; // negative gradient
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        // c = 0.9·0.5 + 0.1·(-1.0) = 0.45 - 0.1 = 0.35 > 0 → would normally update
        // but m.signum() ≠ g.signum() → skip sign step
        let opt = Lion::new(0.9, 0.1, 0.0);
        let lr_mask = vec![1.0; params.len()];
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2, &lr_mask, &OPEN);

        // No weight decay (wd=0, d=1.0), sign update skipped → no change.
        assert!((params[0] - 5.0).abs() < 1e-9, "Expected no change, got {}", params[0]);
    }
}
