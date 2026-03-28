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
//!    m = β₂ · m + (1 - β₂) · g
pub struct Lion {
    interp:   f64, // β₁ (interpolation weight)
    momentum: f64, // β₂ (momentum decay)
    lr:       f64,
    wd:       f64,
    clip:     Option<(f64, f64)>,
}

impl Lion {
    #[must_use]
    pub const fn new(interp: f64, momentum: f64, lr: f64, wd: f64) -> Self {
        Self {
            interp,
            momentum,
            lr,
            wd,
            clip: None,
        }
    }

    /// Create Lion with weight clipping enabled.
    #[must_use]
    pub const fn with_clipping(interp: f64, momentum: f64, lr: f64, wd: f64, min: f64, max: f64) -> Self {
        Self {
            interp,
            momentum,
            lr,
            wd,
            clip: Some((min, max)),
        }
    }

    /// Enable weight clipping on an existing instance.
    #[must_use]
    pub const fn clipped(mut self, min: f64, max: f64) -> Self {
        self.clip = Some((min, max));
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
    ) {
        debug_assert_eq!(params.len(), momentum.len());
        debug_assert_eq!(params.len(), gradients.len());
        debug_assert_eq!(params.len(), decay_mask.len());
        debug_assert_eq!(params.len(), fixed_mask.len());

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

            // 2. Guard against signumless updates:
            // When |c| is negligible, sign(c) ≈ 0 but weight decay on p would still fire,
            // silently eroding a parameter without any gradient signal.
            // Skip the update but clipping still applies below.
            //
            // TODO: Consider Momentum-Aligned Gradient Masking (Magma).
            // Maintaining dense momentum updates while gating the sign-update magnitude
            // by sigmoid(cossim(m, g) / τ) induces a curvature-dependent geometric
            // regularization term proportional to Δᵀ·H·Δ. This implicitly biases the
            // tuner toward flatter, more generalizable minima and suppresses noisy
            // updates in high-curvature directions of the evaluation landscape.
            // *Ref: Joo, T., Xia, W., Kim, C., Zhang, M., & Ie, E. (2026). On
            // Surprising Effectiveness of Masking Updates in Adaptive Optimizers.*
            // <https://arxiv.org/abs/2602.15322v1>
            let updated = if c.abs() < 1e-9 {
                p // No significant gradient or momentum
            } else {
                let sign = c.signum();
                self.lr.mul_add(-(self.wd * d).mul_add(p, sign), p)
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
            momentum[i] = if g.abs() < 1e-9 && m.abs() < 1e-9 {
                0.0
            } else {
                self.momentum.mul_add(m, (1.0 - self.momentum) * g)
            };
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
        let opt = Lion::with_clipping(0.9, 0.99, 1.0, 0.0, -2.0, 2.0);
        let mut momentum_neg = vec![-0.5];

        opt.update(&mut params, &mut momentum_neg, &grads_neg, &decay_mask, &fixed_mask);
        assert!((params[0] - 2.0).abs() < 1e-9, "Should be clipped to max: {}", params[0]);

        // Test sparse path clipping
        let mut params_sparse = vec![10.0];
        let mut momentum_sparse = vec![0.0];
        let grads_sparse = vec![0.0];
        opt.update(
            &mut params_sparse,
            &mut momentum_sparse,
            &grads_sparse,
            &decay_mask,
            &fixed_mask,
        );
        assert!(
            (params_sparse[0] - 2.0).abs() < 1e-9,
            "Sparse update should still clip: {}",
            params_sparse[0]
        );
    }

    #[test]
    fn lion_no_clipping_by_default() {
        let mut params = vec![100.0];
        let mut momentum = vec![0.0];
        let grads = vec![0.0];
        let decay_mask = vec![0.0];
        let fixed_mask = vec![false];

        let opt = Lion::new(0.9, 0.99, 0.1, 0.0);
        opt.update(&mut params, &mut momentum, &grads, &decay_mask, &fixed_mask);

        // c = β₁ · m + (1 - β₁) · g = 0.9 · 0.0 + 0.1 · 0.0 = 0 :p
        assert!((params[0] - 100.0).abs() < 0.01, "No clipping: {}", params[0]);
    }
}
