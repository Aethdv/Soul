//! Per-parameter bounds reporter.
//!
//! Each generation we record raw (pre-clamp) normalized samples to detect when the
//! tuner repeatedly proposes values outside `[min, max]`: that's the signal that a
//! declared bound is too tight. We also track elite mean/variance and compare the
//! observed clamping rate against the expected clamp rate under the current CMA-ES
//! proposal (Gaussian with σ = sigma · √variance_i, mean = mean_i). Observed
//! substantially > expected → the tuner is genuinely fighting the bound, not just
//! Gaussian tail noise → recommend widening.
//!
//! Reports are appended to a text file every N epochs and on SIGINT, preserving
//! per-epoch history so you can scrub back through the run.

use std::{
    collections::VecDeque,
    fs::OpenOptions,
    io::{self, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use soul::engine::search_params::ParamDef;

use super::{cmaes::CmaEs, optimizer::normal_cdf};

/// Per-generation per-parameter raw observations (pre-clamp).
#[derive(Clone, Default)]
struct GenSample {
    lo_hits: u32,
    hi_hits: u32,
    n_samples: u32,
}

#[derive(Clone)]
pub struct BoundsConfig {
    /// Generations retained for sliding-window clamp-rate analysis.
    pub window_gens: usize,
    /// Observed rate must exceed `expected * multiplier + floor` to flag widen.
    pub alarm_multiplier: f64,
    /// Absolute floor for alarm; prevents flags when expected ≈ 0 and a single sample slips out.
    pub alarm_floor: f64,
    /// EMA smoothing for elite mean / variance tracking.
    pub elite_beta: f64,
}

impl Default for BoundsConfig {
    fn default() -> Self {
        Self { window_gens: 20, alarm_multiplier: 2.0, alarm_floor: 0.02, elite_beta: 0.2 }
    }
}

pub struct BoundsTracker {
    n: usize,
    config: BoundsConfig,

    observed_lo: Vec<f64>, // smallest raw normalized value ever proposed
    observed_hi: Vec<f64>, // largest

    window: VecDeque<Vec<GenSample>>,

    elite_mean_ema: Vec<f64>, // denormalized (raw param units)
    elite_var_ema: Vec<f64>,
    have_elite_init: bool,
}

impl BoundsTracker {
    #[must_use]
    pub fn new(n: usize, config: BoundsConfig) -> Self {
        Self {
            n,
            observed_lo: vec![f64::INFINITY; n],
            observed_hi: vec![f64::NEG_INFINITY; n],
            window: VecDeque::with_capacity(config.window_gens.max(1)),
            elite_mean_ema: vec![0.0; n],
            elite_var_ema: vec![0.0; n],
            have_elite_init: false,
            config,
        }
    }

    /// Record one generation. `population_normalized` is the raw proposal (may stray
    /// outside `[0, 1]`); `elite_indices` are the top-μ winners by penalized fitness.
    pub fn observe(&mut self, population_normalized: &[Vec<f64>], elite_indices: &[usize], params: &[&ParamDef]) {
        let mut cur = vec![GenSample::default(); self.n];

        for sample in population_normalized {
            for (i, &v) in sample.iter().enumerate() {
                cur[i].n_samples += 1;

                if v < 0.0 {
                    cur[i].lo_hits += 1;
                }
                if v > 1.0 {
                    cur[i].hi_hits += 1;
                }

                if v < self.observed_lo[i] {
                    self.observed_lo[i] = v;
                }
                if v > self.observed_hi[i] {
                    self.observed_hi[i] = v;
                }
            }
        }

        self.window.push_back(cur);
        while self.window.len() > self.config.window_gens {
            self.window.pop_front();
        }

        // Elite denormalized mean + variance (population stats over the elite subset)
        let n_elite = elite_indices.len().max(1) as f64;
        let mut means = vec![0.0; self.n];

        for &idx in elite_indices {
            for (i, &v) in population_normalized[idx].iter().enumerate() {
                means[i] += params[i].denormalize(v.clamp(0.0, 1.0)) / n_elite;
            }
        }

        let mut vars = vec![0.0; self.n];

        for &idx in elite_indices {
            for (i, &v) in population_normalized[idx].iter().enumerate() {
                let raw = params[i].denormalize(v.clamp(0.0, 1.0));

                vars[i] += (raw - means[i]).powi(2) / n_elite;
            }
        }

        if !self.have_elite_init {
            self.elite_mean_ema = means;
            self.elite_var_ema = vars;
            self.have_elite_init = true;
        } else {
            let b = self.config.elite_beta;

            for i in 0..self.n {
                self.elite_mean_ema[i] = (1.0 - b).mul_add(self.elite_mean_ema[i], b * means[i]);
                self.elite_var_ema[i] = (1.0 - b).mul_add(self.elite_var_ema[i], b * vars[i]);
            }
        }
    }

    /// Append one report snapshot to `path`.
    ///
    /// # Errors
    /// Forwards any filesystem error from opening or writing.
    pub fn write_report(&self, path: &Path, params: &[&ParamDef], cmaes: &CmaEs, epoch: usize, label: &str) -> io::Result<()> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;
        let ts = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);

        writeln!(file)?;
        writeln!(file, "{}", "=".repeat(150))?;
        writeln!(file, "TUNER BOUNDS REPORT | {label} | epoch {epoch} | unix={ts}")?;
        writeln!(file, "{}", "=".repeat(150))?;
        writeln!(file)?;
        writeln!(
            file,
            "{:<22} {:>8} {:>16} {:>18} {:>16} {:>9} {:>9} {:>20}  verdict",
            "param", "default", "cur[min,max]", "observed[lo,hi]", "elite[μ±σ]", "lo_clmp", "hi_clmp", "expected[lo,hi]"
        )?;
        writeln!(file, "{}", "-".repeat(150))?;

        let mean = cmaes.mean();
        let variances = cmaes.variances();
        let sigma = cmaes.sigma();

        for (i, p) in params.iter().enumerate() {
            let (window_lo, window_hi, window_n) = self.window.iter().fold((0u64, 0u64, 0u64), |(lo, hi, n), g| {
                (lo + u64::from(g[i].lo_hits), hi + u64::from(g[i].hi_hits), n + u64::from(g[i].n_samples))
            });

            let lo_rate = if window_n > 0 { window_lo as f64 / window_n as f64 } else { 0.0 };
            let hi_rate = if window_n > 0 { window_hi as f64 / window_n as f64 } else { 0.0 };

            let proposal_sigma = (sigma * variances[i].sqrt()).max(1e-9);
            let expected_lo = normal_cdf(-mean[i] / proposal_sigma);
            let expected_hi = 1.0 - normal_cdf((1.0 - mean[i]) / proposal_sigma);

            let obs_lo_raw = p.denormalize(self.observed_lo[i].clamp(0.0, 1.0));
            let obs_hi_raw = p.denormalize(self.observed_hi[i].clamp(0.0, 1.0));
            let elite_std = self.elite_var_ema[i].sqrt();

            let verdict = self.verdict(p, i, lo_rate, hi_rate, expected_lo, expected_hi, obs_lo_raw, obs_hi_raw, elite_std);

            writeln!(
                file,
                "{:<22} {:>8.0} {:>16} {:>18} {:>16} {:>8.1}% {:>8.1}% {:>20}  {}",
                p.name,
                p.default,
                format!("[{:.0}, {:.0}]", p.min, p.max),
                format!("[{:.0}, {:.0}]", obs_lo_raw, obs_hi_raw),
                format!("{:.1}±{:.1}", self.elite_mean_ema[i], elite_std),
                lo_rate * 100.0,
                hi_rate * 100.0,
                format!("[{:.2}%, {:.2}%]", expected_lo * 100.0, expected_hi * 100.0),
                verdict,
            )?;
        }

        writeln!(file)?;
        file.flush()?;
        Ok(())
    }

    fn verdict(
        &self,
        p: &ParamDef,
        i: usize,
        lo_rate: f64,
        hi_rate: f64,
        expected_lo: f64,
        expected_hi: f64,
        obs_lo_raw: f64,
        obs_hi_raw: f64,
        elite_std: f64,
    ) -> String {
        let alarm = self.config.alarm_multiplier;
        let floor = self.config.alarm_floor;
        let range = (p.max - p.min).max(1.0);
        let step = p.step.max(1.0);

        if lo_rate > alarm.mul_add(expected_lo, floor) {
            let suggest = (p.min - step * (lo_rate / 0.05).ceil()).max(0.0);
            return format!("LOWER MIN -> ~{suggest:.0}");
        }

        if hi_rate > alarm.mul_add(expected_hi, floor) {
            let suggest = step.mul_add((hi_rate / 0.05).ceil(), p.max);
            return format!("RAISE MAX -> ~{suggest:.0}");
        }

        if !self.have_elite_init {
            return "warmup".to_string();
        }

        let near_default = (self.elite_mean_ema[i] - p.default).abs() < 0.05 * range;

        if elite_std < 0.02 * range && near_default {
            return "LOW SIGNAL".to_string();
        }

        if elite_std < 0.02 * range {
            return format!("CONVERGED -> {:.1}", self.elite_mean_ema[i]);
        }

        let interior_lo = obs_lo_raw > 0.10f64.mul_add(range, p.min);
        let interior_hi = obs_hi_raw < p.max - 0.10 * range;

        if interior_lo && interior_hi {
            return format!("TIGHTEN -> [{:.0}, {:.0}]", obs_lo_raw - 0.05 * range, 0.05f64.mul_add(range, obs_hi_raw));
        }
        "ok".to_string()
    }
}
