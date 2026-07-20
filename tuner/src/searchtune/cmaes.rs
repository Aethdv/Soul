//! Covariance Matrix Adaptation Evolution Strategy (CMA-ES)
//!
//! Implements a hybrid variant of CMA-ES specifically engineered for the extreme
//! stochastic noise of chess engine self-play: Separable, Soft Active CMA-ES with
//! Orthogonal Mirrored Sampling, SNR-Adaptive Learning Rates, and Dynamic Budgeting.
//!
//! 1. Separable CMA-ES (Sep-CMA-ES)
//!    Standard CMA-ES maintains a full `O(N²)` covariance matrix to model parameter
//!    interactions. In highly stochastic environments, off-diagonal correlations tend
//!    to overfit to noise rather than to true signal. We restrict the covariance matrix
//!    to its diagonal `O(N)`, sacrificing rotational invariance for numerical stability
//!    and enabling the algorithm to scale to high-dimensional parameter sets without
//!    premature convergence.
//!    *Ref: Ros, R., & Hansen, N. (2008). A Simple Modification in CMA-ES Achieving
//!    Linear Time and Space Complexity. PPSN X, pp. 296–305.*
//!    <https://inria.hal.science/inria-00287367/document>
//!
//! 2. Soft Active CMA-ES
//!    Active CMA-ES assigns negative recombination weights to the worst-performing
//!    candidates, explicitly contracting the covariance in directions associated with
//!    fitness degradation. This increases convergence pressure toward the true optimum.
//!    In chess self-play, however, a bad result often reflects unlucky game pairings
//!    rather than a genuinely inferior parameter set. The `active_softness` scalar
//!    uniformly dampens all negative weights, reducing the risk of the search space
//!    collapsing over stochastic noise.
//!    *Ref: Jastrebski, G., & Arnold, D. V. (2006). Improving Evolution Strategies
//!    through Active Covariance Matrix Adaptation. 2006 IEEE Congress on Evolutionary
//!    Computation (CEC 2006), pp. 9719–9726.*
//!    <https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=8f7ccc273a844f585289159793e00dbce0365207>
//!
//! 3. Orthogonal Mirrored Sampling
//!    Instead of independent standard normal samples, we generate an orthogonal basis
//!    via Gram-Schmidt and evaluate each direction as a mirrored pair (`+z` and `-z`).
//!    Orthogonality maximizes the volume covered by the population in parameter space,
//!    reducing redundant sampling. Mirroring cancels the odd-order bias in the mean
//!    update, halving the variance of the mean shift estimate for the same evaluation
//!    budget.
//!    *Ref: Wang, H., Emmerich, M., & Bäck, T. (2019). Mirrored Orthogonal Sampling
//!    for Covariance Matrix Adaptation Evolution Strategies. Evolutionary Computation,
//!    27(4), 699–725.* <https://doi.org/10.1162/evco_a_00251>
//!    <https://scholarlypublications.universiteitleiden.nl/access/item%3A3243901/view>
//!
//! 4. SNR-Gated Covariance Adaptation (Domain-Specific Heuristic)
//!    Active CMA-ES is vulnerable to structural match noise: a candidate may lose a
//!    short match due to opening book draw-rates rather than genuine parameter
//!    inferiority. If the covariance shrinks in response to such noise, the search
//!    space can collapse prematurely around noise.
//!    We apply the Law of Total Variance (`Var(Obs) = Var(True) + Var(Noise)`) using
//!    the average per-candidate match variance (mean of `SE²`, not `mean(SE)²`;
//!    Jensen's inequality would systematically under-state noise the other way).
//!    The Reliability Coefficient `R = Var(True) / Var(Obs)`, the classical
//!    psychometric measure of signal fraction, is smoothed across generations, and
//!    a cubic smoothstep of the EMA gates the negative covariance updates only.
//!    Positive updates pass through at full strength so noisy gens still move toward
//!    observed winners; only the destructive (search-space contracting) updates are
//!    throttled, since their failure mode (premature collapse) is unrecoverable.
//!    *Ref: Spearman, C. (1904). The Proof and Measurement of Association between
//!    Two Things. The American Journal of Psychology, 15(1), 72–101.*
//!    *Note: applying this classical coefficient to gate Active CMA-ES negative-weight
//!    updates is, to our knowledge, unused elsewhere in chess engine tuning. Related
//!    to LRA-CMA-ES's SNR-maintenance but acts on individual covariance updates rather
//!    than the global learning rate.*
//!
//! 5. SNR-Adaptive Learning Rate (LRA-CMA-ES)
//!    The global update scale `η` is adapted each generation to maintain a stable
//!    signal-to-noise ratio in the mean update. An exponential moving average tracks
//!    the per-dimension mean shift (signal) against the total update energy (noise).
//!    When the SNR is high the learning rate stays near 1.0; when the SNR degrades
//!    due to match variance, `η` is reduced, slowing the mean and covariance updates
//!    proportionally to keep adaptation aligned with the true gradient.
//!    *Ref: Nomura, M., Akimoto, Y., & Ono, I. (2023). CMA-ES with Learning Rate
//!    Adaptation: Can CMA-ES with Default Population Size Solve Multimodal and Noisy
//!    Problems? GECCO 2023, pp. 839–847.*
//!    <https://doi.org/10.1145/3583131.3590358>
//!
//! 6. Adaptive Evaluation Budgeting (AR-CMA-ES)
//!    The match budget per candidate is dynamically scaled based on the lower bound of
//!    Expected Improvement. By observing the local gradient magnitude and per-candidate
//!    match variance, the tuner allocates more games only when the ranking among
//!    candidates is sufficiently ambiguous to warrant the extra evaluation cost.
//!    *Ref: Dinu, C.-V., Patel, Y. J., Bonet-Monroig, X., & Wang, H. (2024).
//!    An Adaptive Re-evaluation Method for Evolution Strategy under Additive Noise.*
//!    <https://arxiv.org/abs/2409.16757>

/// Separable (Diagonal) Soft Active CMA-ES with SNR-adaptive learning rates.
///
/// Computes a diagonal covariance matrix to prevent noise overfitting, actively
/// contracts search volume around failed candidates, and gates updates based on
/// empirical match variance to prevent premature convergence.
///
/// Note on Learning Rate Adaptation (LRA):
/// This implementation provides the metrics needed for LRA (via `update_snr()`), but
/// does not automatically adjust its internal learning rate (`eta`). The caller
/// must call `update_snr()` after each generation and pass the result to `set_lr()`
/// to activate adaptive scaling.
#[derive(Clone)]
pub struct CmaEs {
    n: usize,
    lambda: usize,
    mu: usize,
    generation: usize,
    mean: Vec<f64>,
    sigma: f64,
    variances: Vec<f64>,
    p_sigma: Vec<f64>,
    p_c: Vec<f64>,
    weights: Vec<f64>,
    mu_eff: f64,
    c_sigma: f64,
    d_sigma: f64,
    c_c: f64,
    c_1: f64,
    c_mu: f64,
    chi_n: f64,
    mu_eff_neg: f64,
    active_softness: f64,

    // ── Adaptation State
    eta: f64,             // Global learning rate factor
    lra_e: Vec<f64>,      // Moving average of updates (Signal)
    lra_v: f64,           // Moving average of update magnitudes (Noise)
    g_norm: f64,          // Current natural gradient norm
    reliability_ema: f64, // Smoothed reliability coefficient for negative-weight gating
}

/// The smoothing factor for the SNR-adaptive learning rate moving averages.
const LRA_BETA: f64 = 0.1;

/// Smoothing factor for the reliability EMA.
/// Lower = more stable, slower to react to genuine noise regime changes.
const RELIABILITY_BETA: f64 = 0.2;

/// Clamps normalized values to [0, 1].
///
/// CMA-ES assumes an infinite optimization plane, but engine parameters inhabit a
/// strict bounded hypercube. We hard-clamp to prevent the search from hallucinating
/// physically impossible parameters, acting as a reflecting boundary for straying samples.
#[inline]
pub fn clamp_normalized(values: &[f64]) -> Vec<f64> {
    clamp_normalized_iter(values).collect()
}

/// Zero-cost iterator mapping normalized values to [0, 1].
#[inline]
pub fn clamp_normalized_iter(values: &[f64]) -> impl Iterator<Item = f64> + '_ {
    values.iter().map(|v| v.clamp(0.0, 1.0))
}

/// Default CMA-ES population size for `n` parameters at the given scaling factor.
///
/// NOTE: This deviates slightly from Hansen's formula (4 + floor(3·ln(n)))
/// by using a ceiling and a minimum floor of 16 to ensure enough statistical
/// support for diagonal covariance estimation in high-noise landscapes.
pub fn default_lambda(n: usize, scale: f64) -> usize {
    (3.0f64.mul_add((n as f64).ln(), 4.0) * scale)
        .ceil()
        // A floor of 16 guarantees at least 8 positive and 8 negative weights.
        // This provides enough statistical support to estimate the diagonal covariance
        // even in very noisy landscapes.
        .max(16.0) as usize
}

impl CmaEs {
    #[must_use]
    pub fn new(n: usize, lambda_multiplier: f64) -> Self {
        Self::new_with_lambda(n, default_lambda(n, lambda_multiplier), 0.5)
    }

    /// Create CMA-ES with a specific population size (for IPOP restarts)
    #[must_use]
    pub fn new_with_lambda(n: usize, lambda: usize, active_softness: f64) -> Self {
        // Round up to nearest even: if odd, +1 makes it even; clears LSB (& !1) ensures even.
        // Mirrored sampling generates +z/-z pairs, so λ must be even.
        let lambda = (lambda + 1) & !1;
        let mu = lambda / 2;
        let nf = n as f64;

        // Positive weights (top μ)
        let mut weights = vec![0.0; lambda];
        for (i, weight) in weights.iter_mut().enumerate().take(mu) {
            *weight = (mu as f64 + 0.5).ln() - (i as f64 + 1.0).ln();
        }

        // Negative weights (bottom half)
        // Hansen (2016) Table 1: w_i^raw = ln((λ+1)/2) − ln(i) for i = μ+1, ..., λ.
        // The worst candidates (largest rank) must carry the most negative penalty.
        for (i, weight) in weights[mu..lambda].iter_mut().enumerate() {
            let rank = (mu - i) as f64;
            *weight = -((lambda as f64 + 1.0) / (2.0 * rank)).ln();
        }

        // Normalize positive weights
        let pos_sum: f64 = weights[..mu].iter().sum();
        for weight in weights.iter_mut().take(mu) {
            *weight /= pos_sum;
        }

        // μ_eff and learning rates
        let sum_sq_pos: f64 = weights[..mu].iter().map(|w| w * w).sum();
        let mu_eff = 1.0 / sum_sq_pos;

        let c_sigma = (mu_eff + 2.0) / (nf + mu_eff + 5.0);
        let d_sigma = 2.0f64.mul_add((0.0_f64).max((mu_eff - 1.0) / (nf + 1.0)).sqrt(), 1.0) + c_sigma;
        let c_c = (4.0 + mu_eff / nf) / (nf + 4.0 + 2.0 * mu_eff / nf);
        let c_1 = 2.0 / ((nf + 1.3).mul_add(nf + 1.3, mu_eff));
        let c_mu = (2.0 * (mu_eff - 2.0 + 1.0 / mu_eff) / ((nf + 2.0).mul_add(nf + 2.0, mu_eff))).min(1.0 - c_1);

        // Scale negative weights
        let neg_sum_raw: f64 = weights[mu..].iter().map(|w| w.abs()).sum();
        let sum_sq_neg: f64 = weights[mu..].iter().map(|w| w * w).sum();

        let mu_eff_neg = if sum_sq_neg > 0.0 {
            neg_sum_raw.powi(2) / sum_sq_neg
        } else {
            // Safety fallback: if no negative weights exist, mu_eff_neg is
            // functionally 1.0 (though this branch is unreachable due to the lambda floor).
            1.0
        };

        // Alpha 1: Rank-one/rank-mu balance
        let alpha_mu = 1.0 + c_1 / c_mu;

        // Alpha 2: Statistical stability
        let alpha_mueff = 1.0 + (2.0 * mu_eff_neg) / (mu_eff + 2.0);

        // Alpha 3: Positive definiteness (geometric safeguard)
        let alpha_pos_def = (1.0 - c_1 - c_mu) / (nf * c_mu);

        // Take MINIMUM (most restrictive constraint)
        let alpha_limit = alpha_mu.min(alpha_mueff).min(alpha_pos_def);

        if neg_sum_raw > 1e-10 {
            // Safety scale from alpha constraints
            let safety_scale = alpha_limit / neg_sum_raw;

            // Final scale: min(1, safety_scale) · softness
            //
            // We scale negative updates by active_softness. In a deterministic function,
            // a terrible result means "never look here again". In chess self-play, a terrible
            // result often just means "we lost a few coin flips". Dampening negative weights
            // stops the optimizer from shrinking the search space over noise cliffs.
            let final_scale = safety_scale.min(1.0) * active_softness;

            for weight in weights.iter_mut().take(lambda).skip(mu) {
                *weight *= final_scale;
            }
        }

        // Recalculate mu_eff_neg after scaling
        let scaled_neg_sum: f64 = weights[mu..].iter().map(|w| w.abs()).sum();
        let scaled_sq_neg: f64 = weights[mu..].iter().map(|w| w * w).sum();
        let mu_eff_neg = if scaled_sq_neg > 0.0 {
            scaled_neg_sum.powi(2) / scaled_sq_neg
        } else {
            // 0.0, not 1.0: post-scale, if all negatives vanished, there's nothing to account for
            0.0
        };

        let chi_n = nf.sqrt() * (1.0 - 1.0 / (4.0 * nf) + 1.0 / (21.0 * nf * nf));

        Self {
            n,
            lambda,
            mu,
            generation: 0,
            mean: vec![0.5; n],
            sigma: 0.25,
            variances: vec![1.0; n],
            p_sigma: vec![0.0; n],
            p_c: vec![0.0; n],
            weights,
            mu_eff,
            c_sigma,
            d_sigma,
            c_c,
            c_1,
            c_mu,
            chi_n,
            mu_eff_neg,
            active_softness,
            eta: 1.0,
            lra_e: vec![0.0; n],
            lra_v: 0.0,
            g_norm: 0.0,
            reliability_ema: 0.0,
        }
    }

    /// Generates λ candidates using mirrored sampling
    ///
    /// Instead of independent random samples, we evaluate opposite pairs (+z and -z).
    /// This structural anti-correlation cancels out odd-order bias in the mean update,
    /// halving the variance of the mean shift estimate for the same evaluation budget.
    pub fn sample_population(&self, rng: &mut fastrand::Rng) -> Vec<Vec<f64>> {
        let half = self.lambda / 2;
        let mut population = Vec::with_capacity(self.lambda);
        let z_matrix = sample_orthogonal_z_matrix(self.n, half, rng);

        for z in z_matrix {
            let x: Vec<f64> = self
                .mean
                .iter()
                .zip(&self.variances)
                .zip(&z)
                .map(|((&m, &var), &zi)| (self.sigma * var.sqrt()).mul_add(zi, m))
                .collect();

            population.push(x);

            let x_neg: Vec<f64> = self
                .mean
                .iter()
                .zip(&self.variances)
                .zip(&z)
                .map(|((&m, &var), &zi)| (self.sigma * var.sqrt()).mul_add(-zi, m))
                .collect();
            population.push(x_neg);
        }

        population
    }

    /// Core update: shifts the mean, adapts step size (σ), and sculpts the covariance matrix.
    ///
    /// The elite subset (μ) tugs the mean toward victory. The entire population (λ) expands
    /// or shrinks our uncertainty (covariance) along their respective vectors.
    /// Crucially, negative update vectors are gated by Signal-to-Noise Ratio (SNR).
    pub fn update(&mut self, population_normalized: &[Vec<f64>], penalized_elo: &[f64], raw_elo: &[f64], avg_var_noise: f64) {
        let mut indices: Vec<usize> = (0..self.lambda).collect();
        indices.sort_unstable_by(|&a, &b| penalized_elo[b].total_cmp(&penalized_elo[a]));

        // ── SNR Gating (Reliability Coefficient)
        // Raw match variance combines true engine strength (signal) and match luck (noise).
        // Gating on raw variance alone punishes candidates for unlucky pairings, so we
        // decompose: Var(Observed) = Var(True) + Var(Noise). The noise term is the
        // average per-candidate match variance (mean of SE², not SE-of-mean squared;
        // Jensen would systematically under-state noise the other way).
        // R = Var(True) / Var(Observed) is the reliability coefficient (Spearman, 1904).
        let mean_raw = raw_elo.iter().sum::<f64>() / self.lambda as f64;
        let var_observed = raw_elo.iter().map(|e| (e - mean_raw).powi(2)).sum::<f64>() / (self.lambda.max(2) - 1) as f64;
        let var_signal = (var_observed - avg_var_noise).max(0.0);
        let reliability = if var_observed > 1e-6 { (var_signal / var_observed).clamp(0.0, 1.0) } else { 0.0 };

        // Smooth across generations: a single quiet gen shouldn't mute all negative
        // updates. Lazy init on the first call so we don't bias toward 1.0 (trust)
        // or 0.0 (paranoia) before we've measured anything.
        self.reliability_ema = if self.generation == 0 {
            reliability
        } else {
            (1.0 - RELIABILITY_BETA).mul_add(self.reliability_ema, RELIABILITY_BETA * reliability)
        };

        // Cubic smoothstep the EMA reliability to gracefully fade out noise-driven negative updates
        let r = self.reliability_ema;
        let neg_scale = r * r * (3.0 - 2.0 * r);

        self.generation += 1;

        let old_mean = self.mean.clone();
        let std_devs: Vec<f64> = self.variances.iter().map(|v| v.sqrt()).collect();

        // Recombination mean at full rate (eta = 1). The evolution paths must see
        // the unscaled natural-gradient step: eta scales only the final mean and
        // covariance deltas, never the paths. An eta-scaled step biases ||p_sigma||
        // toward eta·chi_n, so CSA shrinks sigma with no signal, and since eta falls
        // as the search converges the decay compounds.
        let mut recomb = vec![0.0; self.n];

        for (i, &idx) in indices.iter().enumerate().take(self.mu) {
            let w = self.weights[i];

            for (r, &x) in recomb.iter_mut().zip(&population_normalized[idx]) {
                *r = w.mul_add(x, *r);
            }
        }

        let y: Vec<f64> = recomb.iter().zip(&old_mean).map(|(&r, &om)| (r - om) / self.sigma).collect();
        let z_mean: Vec<f64> = y.iter().zip(&std_devs).map(|(&yi, &si)| yi / si).collect();

        // Mean update at the LRA rate: m ← old_mean + eta · (recomb - old_mean).
        for ((m, &r), &om) in self.mean.iter_mut().zip(&recomb).zip(&old_mean) {
            *m = self.eta.mul_add(r - om, om);
        }

        for (ps, &zi) in self.p_sigma.iter_mut().zip(&z_mean) {
            *ps = (1.0 - self.c_sigma).mul_add(*ps, (self.c_sigma * (2.0 - self.c_sigma) * self.mu_eff).sqrt() * zi);
        }

        let ps_norm: f64 = self.p_sigma.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.sigma *= ((self.c_sigma / self.d_sigma) * (ps_norm / self.chi_n - 1.0)).exp();

        // Clamp sigma to [1e-6, 10.0] to prevent search space collapse while
        // allowing for sufficient exploration in noisy landscapes.
        // A ceiling of 10.0 is used as a hard safety boundary since engine
        // parameters are normalized to [0, 1], meaning 10.0 covers several search space widths.
        self.sigma = self.sigma.clamp(1e-6, 10.0);

        // ── h_sigma: Rank-One Update Guard
        // Compare ||p_sigma|| against its expected length under a random walk
        // (Hansen, 2016), with the startup correction for a path still filling up.
        // An unusually LONG path means sigma is far too small for the landscape:
        // consecutive steps all point the same way, and the path direction is a
        // transient of that correction, not converged curvature. h_sigma = 0 then
        // stalls the rank-one input so the covariance doesn't inflate along a
        // temporary direction; the (1 - h_sigma) term in old_cov_weight repays
        // the variance the frozen path update would have carried.
        let expected_ps = self.chi_n * (1.0 - (1.0 - self.c_sigma).powi(2 * self.generation as i32)).sqrt();

        let h_sigma_cond = ps_norm < (1.4 + 2.0 / (self.n as f64 + 1.0)) * expected_ps;
        let h_sigma = if h_sigma_cond { 1.0 } else { 0.0 };

        for (pc, &yi) in self.p_c.iter_mut().zip(&y) {
            *pc = (1.0 - self.c_c).mul_add(*pc, h_sigma * (self.c_c * (2.0 - self.c_c) * self.mu_eff).sqrt() * yi);
        }

        // Precalculate dynamic weights (SNR scaling + Mahalanobis normalization)
        let mut dyn_weights = self.weights.clone();
        let mut neg_sum_abs = 0.0;
        let mut neg_sum_sq = 0.0;

        for k in 0..self.lambda {
            let idx = indices[k];
            if dyn_weights[k] < 0.0 {
                dyn_weights[k] *= neg_scale;

                // Calculate: ||C^(-1/2) · z_j||² = sum_j (z_j²).
                // Do NOT divide by variance again, since z_j is already scaled by std_dev.
                let mut mahal_norm_sq = 0.0;
                for j in 0..self.n {
                    let z_j = (population_normalized[idx][j] - old_mean[j]) / (self.sigma * std_devs[j]);
                    mahal_norm_sq += z_j * z_j;
                }

                // Normalize: w_effective = w · n / ||C^(-1/2) · y||²
                // This ensures each negative sample contributes equal "variance removal"
                if mahal_norm_sq > 1e-10 {
                    dyn_weights[k] *= self.n as f64 / mahal_norm_sq;
                }

                neg_sum_abs += dyn_weights[k].abs();
                neg_sum_sq += dyn_weights[k] * dyn_weights[k];
            }
        }

        // Update mu_eff_neg for diagnostics (reflects dynamic scaling)
        if neg_sum_sq > 0.0 {
            self.mu_eff_neg = neg_sum_abs.powi(2) / neg_sum_sq;
        }

        let sum_w: f64 = dyn_weights.iter().sum();
        let c_1_eff = self.c_1 * self.eta;
        let c_mu_eff = self.c_mu * self.eta;

        // Covariance matrix learning (Diagonal)
        // Update the variances using rank-1 and rank-μ updates.
        // Formula: (1 - c₁ - c_μ·Σw)·C + c₁·(p_c·p_cᵀ) + c_μ·Σ(w·y·yᵀ)
        // With h_σ correction: (1 - c₁ - c_μ·Σw + (1-h_σ)·c₁·c_c·(2-c_c))·C
        // Ref: Hansen (2016) Eq. 38.
        let old_cov_weight = c_mu_eff.mul_add(-sum_w, 1.0 - c_1_eff) + (1.0 - h_sigma) * c_1_eff * self.c_c * (2.0 - self.c_c);

        for i in 0..self.n {
            let rank_one = c_1_eff * self.p_c[i] * self.p_c[i];
            let mut rank_mu = 0.0;

            for k in 0..self.lambda {
                let idx = indices[k];
                // Covariance update requires standard (X-m)/σ!
                // Not divided by std_dev, because it scales proportional to variance!
                let y_i = (population_normalized[idx][i] - old_mean[i]) / self.sigma;
                rank_mu = dyn_weights[k].mul_add(y_i * y_i, rank_mu);
            }

            self.variances[i] = self.variances[i].mul_add(old_cov_weight, rank_one + c_mu_eff * rank_mu);
            self.variances[i] = self.variances[i].max(1e-6);
            self.variances[i] = self.variances[i].min(100.0);
        }

        // ── Adaptation Metrics
        // 1. Mean Shift Norm
        // Measure the magnitude of the update in the current coordinate system.
        // g = (m_new - m_old) / sigma, normalized by diagonal variances.
        let mut g_sq_norm = 0.0;

        for (i, &v) in self.variances.iter().enumerate() {
            let gi = (self.mean[i] - old_mean[i]) / self.sigma;
            g_sq_norm += gi * gi / v;
        }
        self.g_norm = g_sq_norm.sqrt();

        // 2. Signal/Noise Estimators
        // Track the raw jitter in parameter updates to estimate the reliable signal.
        let mut delta_norm_sq = 0.0;

        for (e, (&m, &om)) in self.lra_e.iter_mut().zip(self.mean.iter().zip(&old_mean)) {
            let delta_i = m - om;
            *e = (1.0_f64 - LRA_BETA).mul_add(*e, LRA_BETA * delta_i);
            delta_norm_sq += delta_i * delta_i;
        }
        self.lra_v = (1.0_f64 - LRA_BETA).mul_add(self.lra_v, LRA_BETA * delta_norm_sq);
    }

    #[must_use]
    pub fn current_mean(&self) -> Vec<f64> {
        self.mean.clone()
    }

    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    #[must_use]
    pub const fn sigma(&self) -> f64 {
        self.sigma
    }

    pub fn set_sigma(&mut self, s: f64) {
        self.sigma = s;
    }

    #[must_use]
    pub const fn lambda(&self) -> usize {
        self.lambda
    }

    #[must_use]
    pub const fn mu_eff_neg(&self) -> f64 {
        self.mu_eff_neg
    }

    #[must_use]
    pub fn variances(&self) -> &[f64] {
        &self.variances
    }

    #[must_use]
    pub fn p_sigma(&self) -> &[f64] {
        &self.p_sigma
    }

    #[must_use]
    pub fn p_c(&self) -> &[f64] {
        &self.p_c
    }

    /// Returns the norm of the last update's mean shift.
    #[must_use]
    pub fn mean_shift_norm(&self) -> f64 {
        self.g_norm
    }

    /// Estimates the current Signal-to-Noise Ratio (SNR) of parameter updates.
    #[must_use]
    pub fn update_snr(&self) -> f64 {
        // Steady-state identity for EMAs: E[e²] ≈ β/(2−β) · E[x²] for uncorrelated x_t.
        // Isolates the genuine directional signal by subtracting the expected noise floor.
        let e_sq_norm: f64 = self.lra_e.iter().map(|&x| x * x).sum();
        let bias_corr = LRA_BETA / (2.0 - LRA_BETA);

        let num = (e_sq_norm - bias_corr * self.lra_v).max(0.0);
        let den = (self.lra_v - e_sq_norm).max(1e-10);
        num / den
    }

    /// Sum of squared positive weights. Used for signal scaling.
    #[must_use]
    pub fn sum_sq_weights(&self) -> f64 {
        self.weights[..self.mu].iter().map(|&w| w * w).sum()
    }

    #[must_use]
    pub fn learning_rate(&self) -> f64 {
        self.eta
    }

    pub fn set_lr(&mut self, lr: f64) {
        // A floor of 0.05 prevents the optimizer from stalling completely
        // during transient noise spikes.
        self.eta = lr.clamp(0.05, 1.0);
    }

    /// Directly overwrites the search mean, bypassing the normal optimizer update.
    /// Used for warm-starting from a prior run's best parameters.
    ///
    /// # Panics
    /// if `mean.len() != self.n`.
    pub fn set_mean(&mut self, mean: &[f64]) {
        self.mean.copy_from_slice(mean);
    }

    /// Restores full optimizer state from a checkpoint.
    /// Silently ignores fields whose length doesn't match `n`.
    /// A mismatch typically means the parameter schema changed between runs.
    pub fn restore_state(&mut self, variances: Vec<f64>, p_sigma: Vec<f64>, p_c: Vec<f64>) {
        if variances.len() == self.n {
            self.variances = variances;
        }

        if p_sigma.len() == self.n {
            self.p_sigma = p_sigma;
        }

        if p_c.len() == self.n {
            self.p_c = p_c;
        }
    }

    /// Resets to a fresh optimizer centered at `start`, with `new_lambda` candidates and
    /// step size `new_sigma`. Fully decoupled from `SearchParams`: only the statistical
    /// state is reset, not the parameter definitions.
    ///
    /// If σ collapses into a local minimum, an IPOP restart (doubling λ and resetting σ)
    /// helps the optimizer escape.
    ///
    /// This implementation preserves learned variance ratios: dimensions that were
    /// "settled" (low variance) in the previous run are used to seed the new run,
    /// giving the restart a "sensitivity hint" from the previous search.
    pub fn restart_from(&mut self, start: Vec<f64>, new_lambda: usize, new_sigma: f64) {
        let saved_variances = self.variances.clone();
        let max_var = saved_variances.iter().copied().fold(f64::NEG_INFINITY, f64::max);

        let n = self.n;
        let softness = self.active_softness;
        *self = Self::new_with_lambda(n, new_lambda, softness);
        self.mean = start;
        self.sigma = new_sigma;

        // Preserve relative variance ratios from the last run, renormalized to [1, 2].
        // This tells the restart "these dimensions were already settled; explore the others."
        if max_var > 1e-10 {
            for (v_new, &v_old) in self.variances.iter_mut().zip(&saved_variances) {
                *v_new = 1.0 + (v_old / max_var);
            }
        }
    }
}

/// Generates one standard normal deviate via Box-Muller.
///
/// Box-Muller naturally produces two independent normals per call (`cos` and `sin`);
/// only `cos` is returned here since this is used only in the rare linear-dependence
/// fallback where we rebuild an individual vector component at a time.
/// For the main sampling loop, pairs are generated directly to avoid the waste.
fn rand_norm(rng: &mut fastrand::Rng) -> f64 {
    let u1: f64 = rng.f64().max(1e-9);
    let u2: f64 = rng.f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Generates an orthogonal basis matrix for sampling.
/// Uses Gram-Schmidt process to ensure rotational symmetry in the search space.
fn sample_orthogonal_z_matrix(n: usize, k: usize, rng: &mut fastrand::Rng) -> Vec<Vec<f64>> {
    let mut matrix = Vec::with_capacity(k);
    let mut norms = Vec::with_capacity(k);

    for i in 0..k {
        // 1. Sample standard normal
        let mut z = Vec::with_capacity(n);

        for _ in (0..n).step_by(2) {
            let u1: f64 = rng.f64().max(1e-9);
            let u2: f64 = rng.f64();
            let radius = (-2.0 * u1.ln()).sqrt();
            let theta = 2.0 * std::f64::consts::PI * u2;
            let (sin, cos) = theta.sin_cos();
            z.push(radius * cos);

            if z.len() < n {
                z.push(radius * sin);
            }
        }

        // save original radius to maintain the chi distribution
        let radius = z.iter().map(|x| x * x).sum::<f64>().sqrt();
        norms.push(radius);

        // 2. Gram-Schmidt Orthogonalization
        if i < n {
            for u in &matrix {
                let dot: f64 = z.iter().zip(u).map(|(a, b)| a * b).sum();

                for (zi, &ui) in z.iter_mut().zip(u) {
                    *zi -= dot * ui;
                }
            }

            // Normalize the direction vector
            let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();

            if norm > 1e-10 {
                for x in &mut z {
                    *x /= norm;
                }
                matrix.push(z);
            } else {
                // Fallback for linear dependence
                for x in &mut z {
                    *x = rand_norm(rng);
                }

                let norm: f64 = z.iter().map(|x| x * x).sum::<f64>().sqrt();

                for x in &mut z {
                    *x /= norm;
                }
                matrix.push(z);
            }
        } else {
            // Past n vectors, no independent directions remain in n-space:
            // Gram-Schmidt would annihilate the sample entirely.
            // Just normalize to the original chi-distribution radius.
            if radius > 1e-10 {
                for x in &mut z {
                    *x /= radius;
                }
            }
            // Edge case: if radius is somehow negligible (astronomically unlikely for
            // standard normals in practice), the resulting sample is near-zero and will
            // act as a neutral point in the population. Not worth guarding more aggressively.
            matrix.push(z);
        }
    }

    // 3. Re-scale by original radii to maintain the proper Mahalanobis volume
    for (z, norm) in matrix.iter_mut().zip(norms) {
        for x in z {
            *x *= norm;
        }
    }

    matrix
}
