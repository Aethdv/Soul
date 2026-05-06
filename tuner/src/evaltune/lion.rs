//! Lion: "Evolved Sign Momentum".
//! In human terms: gradient descent with mood swings.
//!
//! Unlike Adam, which carefully tracks variance and bias,
//! Lion just goes "is the gradient positive?"
//! and shoves the parameter one step in that direction.
//!
//! *Ref: Chen, X., Liang, C., Huang, D., Real, E., Wang, K., Liu, Y., Pham, H.,
//! Dong, X., Luong, T., Hsieh, C.-J., Lu, Y., & Le, Q. V. (2023). Symbolic
//! Discovery of Optimization Algorithms. NeurIPS 2023.*
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
//! 3. Optional weight clipping:
//!    If configured, clamp θ to a physical range (e.g., -100 to 100 for mobility).
//!
//! 4. Momentum Tracking:
//!    Update the stored momentum (m) for the next step.
//!    m = β₂ · m + (1 - β₂) · g,
//!    where β₂ is per-parameter (PSQT 0.995, mobility 0.95, default 0.99).

pub struct Lion {
    interp: f64, // β₁ (interpolation weight)
    lr: f64,
    wd: f64,
    clip: Option<(f64, f64)>,
    /// Momentum-Aligned Gradient Masking temperature.
    /// 0.0 = disabled. 0.05–0.3 effective range.
    /// Scales the sign update magnitude by sigmoid(cossim(momentum_vec, gradient_vec) / tau).
    /// Induces implicit Δᵀ·H·Δ geometric regularization toward flatter minima.
    magma_tau: f64,
}

impl Lion {
    #[must_use]
    pub const fn new(interp: f64, lr: f64, wd: f64) -> Self {
        Self { interp, lr, wd, clip: None, magma_tau: 0.0 }
    }

    /// Create Lion with weight clipping enabled.
    #[must_use]
    pub const fn with_clipping(interp: f64, lr: f64, wd: f64, min: f64, max: f64) -> Self {
        Self { interp, lr, wd, clip: Some((min, max)), magma_tau: 0.0 }
    }

    /// Enable weight clipping on an existing instance.
    #[must_use]
    pub const fn clipped(mut self, min: f64, max: f64) -> Self {
        self.clip = Some((min, max));
        self
    }

    /// Enable Momentum-Aligned Gradient Masking with the given temperature.
    /// τ controls gating sharpness: 0.05 (aggressive) to 0.3 (permissive).
    #[must_use]
    pub const fn with_magma(mut self, tau: f64) -> Self {
        self.magma_tau = tau;
        self
    }

    #[inline]
    pub const fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
    }

    /// Performs a single optimization step
    pub fn update(
        &self,
        params: &mut [f64],
        momentum: &mut [f64],
        gradients: &[f64],
        decay_mask: &[f64],
        fixed_mask: &[bool],
        beta2: &[f64],
    ) {
        debug_assert_eq!(params.len(), momentum.len());
        debug_assert_eq!(params.len(), gradients.len());
        debug_assert_eq!(params.len(), decay_mask.len());
        debug_assert_eq!(params.len(), fixed_mask.len());
        debug_assert_eq!(params.len(), beta2.len());

        // Magma; global momentum-gradient alignment gate.
        // Accumulate cosine similarity across all parameters,
        // then scale the sign-update magnitude by sigmoid(cossim / tau).
        // Biases toward flatter minima by suppressing updates when the
        // optimizer oscillation indicates high local curvature.
        let magma_scale = if self.magma_tau > 0.0 {
            let mut dot = 0.0f64;
            let mut norm_m = 0.0f64;
            let mut norm_g = 0.0f64;

            for i in 0..params.len() {
                if !fixed_mask[i] {
                    dot = momentum[i].mul_add(gradients[i], dot);
                    norm_m = momentum[i].mul_add(momentum[i], norm_m);
                    norm_g = gradients[i].mul_add(gradients[i], norm_g);
                }
            }
            let cossim = if norm_m > 1e-12 && norm_g > 1e-12 { dot / (norm_m.sqrt() * norm_g.sqrt()) } else { 0.0 };
            1.0 / (1.0 + (-cossim / self.magma_tau).exp())
        } else {
            1.0
        };

        for i in 0..params.len() {
            if fixed_mask[i] {
                continue;
            }

            let p = params[i];
            let m = momentum[i];
            let g = gradients[i];
            let d = decay_mask[i];

            // 1. Interpolation: c = β₁ · m + (1 - β₁) · g
            let c = self.interp.mul_add(m, (1.0 - self.interp) * g);

            // 2. Parameter update with sign gate and weight decay.
            //
            // When |c| is negligible the sign step is skipped, but weight decay
            // still fires — a converged parameter without gradient signal should
            // not lose its regularisation pressure.
            //
            // Per-parameter disagreement gate (m.signum() ≠ g.signum()) is local
            // oscillation suppression. Magma's global cossim gate is a curvature-
            // dependent scaling layer on top — when the optimizer as a whole is
            // aligned, full step; when oscillating, suppressed step.
            //
            // *Ref: Joo, T., Xia, W., Kim, C., Zhang, M., & Ie, E. (2026).
            // On Surprising Effectiveness of Masking Updates in Adaptive Optimizers.*
            // <https://arxiv.org/abs/2602.15322v1>
            let decayed = self.lr.mul_add(-self.wd * d * p, p);
            let updated = if c.abs() < 1e-9 {
                decayed
            } else if m.signum() != g.signum() && m.abs() > 1e-6 {
                decayed
            } else {
                let sign = c.signum();
                decayed - self.lr * sign * magma_scale
            };

            // 3. Optional weight clipping
            params[i] = match self.clip {
                Some((min, max)) => updated.clamp(min, max),
                None => updated,
            };

            // 4. Momentum: m = β₂ · m + (1 - β₂) · g
            // Hard-zero momentum when both gradient and current momentum are essentially zero.
            // Without this, floating-point residuals in m can accumulate and trigger sign updates
            // on perfectly converged parameters — a kind of ghost-gradient effect.
            momentum[i] = if g.abs() < 1e-9 && m.abs() < 1e-9 { 0.0 } else { beta2[i].mul_add(m, (1.0 - beta2[i]) * g) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lion_clipping_works() {
        let mut params = vec![1.5]; // Within range
        let _momentum = vec![0.5]; // Strong existing momentum
        let _grads = vec![1.0]; // Strong gradient (opposite direction)
        let decay_mask = vec![0.0]; // No weight decay
        let fixed_mask = vec![false];

        let grads_neg = vec![-1.0];
        let opt = Lion::with_clipping(0.9, 1.0, 0.0, -2.0, 2.0);
        let mut momentum_neg = vec![-0.5];
        let beta2 = vec![0.99];

        opt.update(&mut params, &mut momentum_neg, &grads_neg, &decay_mask, &fixed_mask, &beta2);
        assert!((params[0] - 2.0).abs() < 1e-9, "Should be clipped to max: {}", params[0]);

        // Test sparse path clipping
        let mut params_sparse = vec![10.0];
        let mut momentum_sparse = vec![0.0];
        let grads_sparse = vec![0.0];
        opt.update(&mut params_sparse, &mut momentum_sparse, &grads_sparse, &decay_mask, &fixed_mask, &beta2);
        assert!((params_sparse[0] - 2.0).abs() < 1e-9, "Sparse update should still clip: {}", params_sparse[0]);
    }

    #[test]
    fn lion_no_clipping_by_default() {
        let mut params = vec![100.0];
        let mut momentum = vec![0.0];
        let grads = vec![0.0];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        let opt = Lion::new(0.9, 0.1, 0.0);
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2);

        // c = β₁ · m + (1 - β₁) · g = 0.9 · 0.0 + 0.1 · 0.0 = 0 :p
        assert!((params[0] - 100.0).abs() < 0.01, "No clipping: {}", params[0]);
    }

    #[test]
    fn lion_dead_zone_still_applies_weight_decay() {
        // c ≈ 0, g ≈ 0: no sign update, but weight decay must still fire.
        let mut params = vec![10.0];
        let mut momentum = vec![0.0];
        let grads = vec![0.0];
        let decay_mask = vec![1.0]; // Full decay on this slot
        let fixed_mask = vec![false];

        // lr=1.0, wd=0.05, d=1.0 → decay = 1.0 · 0.05 · 1.0 · 10.0 = 0.5
        let opt = Lion::new(0.9, 1.0, 0.05);
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2);

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
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2);

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
        let beta2 = vec![0.99];
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2);

        // No weight decay (wd=0, d=1.0), sign update skipped → no change.
        assert!((params[0] - 5.0).abs() < 1e-9, "Expected no change, got {}", params[0]);
    }
}
