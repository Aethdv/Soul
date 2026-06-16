//! Elo estimation and uncertainty quantification for the Pentanomial model.
//!
//! Three standard-error estimators are provided, in ascending order of sample-size trust:
//! - [`Pentanomial::fisher_std_err`]: theory-based (Davidson model), good for small N.
//! - [`Pentanomial::std_err`]: empirical Welford variance, accurate for large N.
//! - [`Pentanomial::std_err_hybrid`]: Hermite-interpolated blend; use this one in practice.

use std::f64::consts::LN_10;

/// We track matches in pairs (swapping colors) to cancel out first-move advantage.
/// We distinguish between Win-Loss (WL) and Draw-Draw (DD) pairs.
/// Both yield 1.0 points, but WL indicates high variance (volatility) while DD
/// indicates low variance (deadlock). Collapsing them into a single bucket discards
/// the variance data needed for accurate Maximum Likelihood Estimation.
#[derive(Clone, Copy, Default, Debug)]
pub struct Pentanomial {
    pub ll: u32,
    pub ld: u32,
    pub dd: u32,
    pub wl: u32,
    pub wd: u32,
    pub ww: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameResult {
    Win,
    Draw,
    Loss,
}

impl GameResult {
    #[must_use]
    pub const fn flip(self) -> Self {
        match self {
            Self::Win => Self::Loss,
            Self::Draw => Self::Draw,
            Self::Loss => Self::Win,
        }
    }
}

impl Pentanomial {
    #[inline]
    #[must_use]
    pub fn total(&self) -> u32 {
        self.ll + self.ld + self.dd + self.wl + self.wd + self.ww
    }

    #[must_use]
    pub fn score(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            // No data = even
            return 0.5;
        }

        let points = (f64::from(self.ww) * 2.0)
            + (f64::from(self.wd) * 1.5)
            + (f64::from(self.dd) * 1.0)
            + (f64::from(self.wl) * 1.0)
            + (f64::from(self.ld) * 0.5);

        points / (f64::from(total) * 2.0)
    }

    /// Derives the Maximum Likelihood Elo (MLE) using the Davidson model.
    /// Uses Newton-Raphson to maximize log-likelihood, separating skill
    /// differential from the draw rate.
    #[must_use]
    pub fn mle_elo(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.0;
        }

        let score = self.score();
        if score <= 0.001 {
            return -1000.0;
        }
        if score >= 0.999 {
            return 1000.0;
        }

        // Initial guess from simple Elo formula
        let mut delta = -400.0 * (1.0 / score - 1.0).log10();
        if !delta.is_finite() {
            delta = 0.0;
        }

        // nu models the draw tendency. Small samples are unreliable: we anchor to empirical.
        let draw_rate = self.draw_rate();
        let nu = (2.0 * draw_rate / (1.0 - draw_rate)).clamp(0.1, 10.0);

        // Laplace / Dirichlet smoothing: inject a prior of (0.5, 0.5, 2.0, 0.5, 0.5)
        // across the five buckets. The inflated center prior treats draws as the
        // neutral baseline, anchoring the MLE toward 0 Elo from small samples.
        let n = [
            f64::from(self.ll) + 0.5,
            f64::from(self.ld) + 0.5,
            f64::from(self.dd) + f64::from(self.wl) + 2.0,
            f64::from(self.wd) + 0.5,
            f64::from(self.ww) + 0.5,
        ];

        // Newton-Raphson on log-likelihood
        for _ in 0..15 {
            let gamma = (delta / 800.0 * LN_10).exp();
            let inv_gamma = 1.0 / gamma;
            let z = gamma + inv_gamma + nu;

            // Single game probabilities (Davidson model)
            let pw = gamma / z;
            let pl = inv_gamma / z;
            let pd = nu / z;

            // Pair probabilities (convolution)
            let p = [pl * pl, 2.0 * pl * pd, 2.0f64.mul_add(pw * pl, pd * pd), 2.0 * pw * pd, pw * pw];

            // Davidson model derivatives for Newton-Raphson
            let k = LN_10 / 800.0;
            let q = pw - pl;
            let dpw = k * pw * (1.0 - q);
            let dpl = k * pl * (-1.0 - q);
            let dpd = -k * pd * q;

            let dp = [
                2.0 * pl * dpl,
                2.0 * (dpl * pd + pl * dpd),
                2.0f64.mul_add(dpw * pl + pw * dpl, 2.0 * pd * dpd),
                2.0 * (dpw * pd + pw * dpd),
                2.0 * pw * dpw,
            ];

            let mut grad = 0.0;
            let mut hess = 0.0;
            for i in 0..5 {
                if p[i] > 1e-10 && n[i] > 0.0 {
                    grad += n[i] * dp[i] / p[i];
                    // Hessian approximation: -n · (dp/p)²
                    hess -= n[i] * (dp[i] / p[i]).powi(2);
                }
            }

            if hess.abs() < 1e-10 {
                break;
            }

            let step = -grad / hess;
            delta += step.clamp(-100.0, 100.0);

            if step.abs() < 0.01 {
                break;
            }
        }

        delta.clamp(-1000.0, 1000.0)
    }

    /// Computes theoretical standard error using Fisher Information for the Davidson
    /// model. Preferred for small sample sizes where empirical variance is unstable.
    #[must_use]
    pub fn fisher_std_err(&self) -> f64 {
        let total = self.total();
        if total < 2 {
            return 100.0;
        }

        let elo = self.mle_elo();
        let draw_rate = self.draw_rate();
        let nu = (2.0 * draw_rate / (1.0 - draw_rate)).clamp(0.1, 10.0);

        // Fisher information for Davidson model
        let gamma = (elo / 800.0 * LN_10).exp();
        let inv_gamma = 1.0 / gamma;
        let z = gamma + inv_gamma + nu;

        let pw = gamma / z;
        let pl = inv_gamma / z;
        let pd = nu / z;

        // Pair probabilities (convolution)
        // P(pair) = [P_ll, P_ld, P_dd+wl, P_wd, P_ww]
        let p = [pl * pl, 2.0 * pl * pd, 2.0f64.mul_add(pw * pl, pd * pd), 2.0 * pw * pd, pw * pw];

        // Davidson model derivatives for pairs
        let k = LN_10 / 800.0;
        let q = pw - pl;
        let dpw = k * pw * (1.0 - q);
        let dpl = k * pl * (-1.0 - q);
        let dpd = -k * pd * q;

        let dp = [
            2.0 * pl * dpl,
            2.0 * (dpl * pd + pl * dpd),
            2.0f64.mul_add(dpw * pl + pw * dpl, 2.0 * pd * dpd),
            2.0 * (dpw * pd + pw * dpd),
            2.0 * pw * dpw,
        ];

        // Fisher information: I(δ) = Σ (1/P_i) · (dP_i/dδ)²
        let mut fisher_info = 0.0;
        for i in 0..5 {
            if p[i] > 1e-10 {
                fisher_info += (1.0 / p[i]) * dp[i] * dp[i];
            }
        }

        // Standard error: 1 / sqrt(N_pairs · I(δ))
        let n_pairs = f64::from(total);
        1.0 / (fisher_info * n_pairs).sqrt()
    }

    /// Blends Fisher information (small samples) with empirical Welford (large samples).
    #[must_use]
    pub fn std_err_hybrid(&self) -> f64 {
        let total = self.total();
        if total < 2 {
            return 100.0;
        }

        let fisher = self.fisher_std_err();
        if total < 10 {
            return fisher;
        }

        let empirical = self.std_err();

        // Agreement-based blending: how much do we trust the empirical measurement?
        // If the theoretical model (Fisher) and reality (Empirical) diverge,
        // we default to the model until they reconcile.
        let ratio = (fisher / empirical.max(1e-9)).clamp(0.5, 2.0);
        let agreement = 1.0 / (1.0 + (ratio - 1.0).abs() * 5.0);

        // We also apply a sample-size based confidence ramp.
        // We don't fully trust empirical until the sample is large enough to be stable.
        let trust_ramp = (f64::from(total) / 100.0).min(1.0);
        let alpha = agreement * trust_ramp;

        (1.0 - alpha).mul_add(fisher, alpha * empirical)
    }

    /// Lower Confidence Bound: `mle_elo - k · std_err`.
    ///
    /// Used as a pessimistic fitness signal that penalizes statistically uncertain
    /// estimates, preventing the optimizer from chasing lucky outliers.
    #[must_use]
    pub fn lcb_elo(&self, k: f64) -> f64 {
        k.mul_add(-self.std_err_hybrid(), self.mle_elo())
    }

    /// Welford-based variance estimate
    #[must_use]
    pub fn std_err(&self) -> f64 {
        let n = f64::from(self.total());
        if n < 2.0 {
            return 100.0;
        }

        // We explicitly multiply by 0.0 for the LL (Loss-Loss) case.
        // While mathematically redundant, it documents the scoring mapping and
        // ensures the weight of every Pentanomial bucket is accounted for.
        let sum_sq = (f64::from(self.ll) * 0.0) // LL: 0.0 pts
            + (f64::from(self.ld) * 0.25)       // LD: 0.5 pts
            + (f64::from(self.dd) * 1.0)        // DD: 1.0 pts
            + (f64::from(self.wl) * 1.0)        // WL: 1.0 pts
            + (f64::from(self.wd) * 2.25)       // WD: 1.5 pts
            + (f64::from(self.ww) * 4.0); // WW: 2.0 pts

        let mean_pair = self.score() * 2.0;
        let rss = (n * mean_pair).mul_add(-mean_pair, sum_sq);
        let var_pair = (rss / (n - 1.0)).max(0.0);

        let se_pair = (var_pair / n).sqrt();
        let se_p = se_pair / 2.0;

        // SE(Elo) = |dElo/dp| · SE(p)
        let p = self.score().clamp(0.01, 0.99);
        (400.0 / (LN_10 * p * (1.0 - p))) * se_p
    }

    pub const fn merge(&mut self, other: &Pentanomial) {
        self.ll += other.ll;
        self.ld += other.ld;
        self.dd += other.dd;
        self.wl += other.wl;
        self.wd += other.wd;
        self.ww += other.ww;
    }

    /// Extracts the raw geometric draw tendency.
    fn draw_rate(&self) -> f64 {
        let total = self.total();
        if total == 0 {
            return 0.5;
        }

        // Count games with at least one draw
        let draw_games = f64::from(self.ld + self.dd * 2 + self.wd);
        let total_games = f64::from(total * 2);
        (draw_games / total_games).clamp(0.01, 0.99)
    }
}

#[must_use]
pub fn pair_to_pentanomial(first: GameResult, second: GameResult) -> Pentanomial {
    use GameResult::{Draw, Loss, Win};

    let mut p = Pentanomial::default();
    match (first, second) {
        (Win, Win) => p.ww = 1,
        (Win, Draw) | (Draw, Win) => p.wd = 1,
        (Draw, Draw) => p.dd = 1,
        (Win, Loss) | (Loss, Win) => p.wl = 1,
        (Draw, Loss) | (Loss, Draw) => p.ld = 1,
        (Loss, Loss) => p.ll = 1,
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_perfect_score() {
        let p = Pentanomial { ww: 10, ..Default::default() };
        assert!((p.score() - 1.0).abs() < 0.001);
        assert!(p.mle_elo() > 500.0);
    }

    #[test]
    fn test_zero_score() {
        let p = Pentanomial { ll: 10, ..Default::default() };
        assert!(p.score() < 0.001);
        assert!(p.mle_elo() < -500.0);
    }

    #[test]
    fn test_even_score() {
        let p = Pentanomial { dd: 10, ..Default::default() };
        assert!((p.score() - 0.5).abs() < 0.001);
        assert!(p.mle_elo().abs() < 10.0);
    }

    #[test]
    fn test_wl_distinct_from_dd() {
        // WL and DD are tracked separately even though they have same point value
        let wl = Pentanomial { wl: 10, ..Default::default() };
        let dd = Pentanomial { dd: 10, ..Default::default() };

        // Same MLE (same point total)
        assert!((wl.mle_elo() - dd.mle_elo()).abs() < 5.0);
        assert!((wl.score() - dd.score()).abs() < 0.001);

        assert_eq!(wl.wl, 10);
        assert_eq!(wl.dd, 0);
        assert_eq!(dd.dd, 10);
        assert_eq!(dd.wl, 0);

        assert_eq!(wl.total(), dd.total());
    }
}
