//! Elo estimation and consensus merging for the SA-CMA-ES tuner.
//!
//! Provides the [`EloCache`] for spatial memoization of match results, allowing
//! the tuner to "look up" nearby historical evaluations instead of re-playing
//! expensive matches.

pub const PRIOR_WEIGHT: f64 = 0.05;

/// A spatial memory of all historically evaluated Elo measurements.
///
/// Playing test matches yields Elo estimates with massive error bars.
/// Instead of treating each generation in a vacuum, we treat prior evaluations
/// as spatial anchors. We use a Gaussian kernel to blend nearby historical
/// measurements. This radically reduces variance from small sample sizes
/// at the cost of a slight inertial drag toward the past.
pub struct EloEntry {
    pub candidate: Vec<f64>,
    pub opponent: Vec<f64>,
    pub elo: f64,
    pub weight: f64,
}

#[derive(Default)]
pub struct EloCache {
    entries: Vec<EloEntry>,
}

impl EloCache {
    pub fn add(&mut self, params: Vec<f64>, opponent: Vec<f64>, elo: f64, weight: f64) {
        self.entries.push(EloEntry { candidate: params, opponent, elo, weight });
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Adaptive bandwidth via Silverman's rule of thumb: h = 1.06 · σ_params · n^(-1/5)
    /// where σ_params is the standard deviation of the candidate dimension vectors.
    ///
    /// This allows the Gaussian kernel to naturally "zoom in" as the population
    /// converges, maintaining high resolution in the most relevant parameter regions.
    #[must_use]
    pub fn silverman_bandwidth(&self) -> f64 {
        let n_entries = self.entries.len();
        if n_entries < 2 {
            return 0.1;
        }

        let n_params = self.entries[0].candidate.len();
        let nf = n_entries as f64;

        // 1. Compute mean candidate vector
        let mut mean = vec![0.0; n_params];
        for entry in &self.entries {
            for (i, &v) in entry.candidate.iter().enumerate() {
                mean[i] += v;
            }
        }
        for m in &mut mean {
            *m /= nf;
        }

        // 2. Compute average variance across dimensions
        let mut total_var = 0.0;
        for entry in &self.entries {
            for (i, &v) in entry.candidate.iter().enumerate() {
                total_var += (v - mean[i]).powi(2);
            }
        }
        let avg_var = total_var / (nf * n_params as f64);

        // 3. Silverman's Rule: h = 1.06 · std_dev · n^(-1/5)
        // We floor the bandwidth to prevent the kernel from vanishing entirely.
        (1.06 * avg_var.sqrt() * nf.powf(-0.2)).max(0.01)
    }

    /// Evaluates the Bayesian consensus using Mahalanobis distance.
    ///
    /// We calculate the distance between the candidates PLUS the distance between the
    /// opponents. If the Polyak mean has barely moved, old measurements are safely
    /// merged. If the anchor has jumped, the distance explodes and the stale data
    /// is naturally ignored.
    pub fn weighted_elo(
        &self,
        params: &[f64],
        opponent: &[f64],
        variances: &[f64],
        sigma: f64,
        smoothing_radius: f64,
        temperature: f64,
    ) -> Option<(f64, f64)> {
        if self.entries.is_empty() {
            return None;
        }

        let r_sq = smoothing_radius * smoothing_radius;
        let mut sum_weight = 0.0;
        let mut sum_elo = 0.0;

        let base_var = sigma * sigma;

        let nf = params.len() as f64;

        for entry in &self.entries {
            let mut d_mah_sq = 0.0;
            for i in 0..params.len() {
                let cand_diff = entry.candidate[i] - params[i];
                let opp_diff = entry.opponent[i] - opponent[i];

                d_mah_sq += (cand_diff.powi(2) + opp_diff.powi(2)) / (base_var * variances[i].max(1e-9));
            }

            let w = (-d_mah_sq / (2.0 * nf * r_sq * temperature)).exp() * entry.weight;
            sum_weight += w;
            sum_elo += w * entry.elo;
        }

        // 0.05 corresponds to a very weak belief that the true Elo is 0.0.
        let avg = sum_elo / (sum_weight + PRIOR_WEIGHT);
        Some((avg, sum_weight))
    }

    /// A brute-force spatial average for console reporting.
    ///
    /// Unlike the Gaussian kernel used for optimization gradients, this simply
    /// draws a hard circle and computes the unweighted mean of everything inside.
    /// Useful for diagnostics to confidently say "Yes, this neighborhood is actually +15 Elo".
    /// Uses raw Euclidean distance (not Mahalanobis-normalized): suitable only for diagnostics where
    /// the covariance scale doesn't matter.
    /// NOTE: This is an O(N) scan over the entire history.
    pub fn denoised_elo(&self, params: &[f64], radius: f64) -> Option<(f64, usize)> {
        let nearby: Vec<f64> = self
            .entries
            .iter()
            .filter(|e| Self::distance(&e.candidate, params) < radius)
            .map(|e| e.elo)
            .collect();

        if nearby.is_empty() {
            None
        } else {
            let avg = nearby.iter().sum::<f64>() / nearby.len() as f64;
            Some((avg, nearby.len()))
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Euclidean distance in normalized space
    #[inline(always)]
    fn distance(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>().sqrt()
    }
}
