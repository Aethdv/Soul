//! What the epoch loop needs besides the gradient: the target it trains toward, the weights it
//! samples with, and the trails it watches for a stall or a split.

use serde::{Deserialize, Deserializer, Serialize};

pub use crate::engine::sigmoid;
use crate::{
    engine::{EpdEntry, FeatureRecord, LAYOUT, SoulEntry, TOTAL_PHASE, eval_f64, eval_params, flip_wdl},
    report::report_phase_balance,
};

/// Fast EMA window in epochs.
pub const TREND_FAST: usize = 10;
/// Slow EMA window in epochs; also defines the warmup period before divergence checks activate.
const TREND_SLOW: usize = 40;

pub const A_FAST: f64 = 2.0 / (TREND_FAST as f64 + 1.0);
const A_SLOW: f64 = 2.0 / (TREND_SLOW as f64 + 1.0);

/// Noise multiplier threshold for divergence detection.
///
/// Tested against the mean absolute epoch-to-epoch validation delta `E[|Δval|] = 2σ / √(π) ≈ 1.13σ`.
/// Because `val_fast` and `val_slow` share inputs, their covariance reduces `sd(fast − slow)` to ≈ 0.21σ.
/// A unit threshold provides a ~5.3σ barrier against random plateau oscillations while remaining
/// sensitive to genuine drift.
const TREND_NOISE_K: f64 = 1.0;

/// Online 95th-percentile gradient norm estimator using log-space stochastic approximation.
pub struct GradientStats {
    p95: f64,
    alpha: f64,
    count: usize,
}

pub trait TunableData: Sync + Send {
    fn eval(&self, values: &[f64]) -> f64;
    fn result(&self) -> f64;
}

/// Computes the blended WDL training target.
///
/// Search scores near zero blend toward the empirical game outcome; high-magnitude scores
/// saturate toward `sigmoid(score, k)`. When `wdl_blend >= 1.0`, instance-level score scaling
/// is bypassed.
///
/// Returns `(target, d(target)/dk)`.
pub fn wdl_target(record: &FeatureRecord, k: f64, wdl_blend: f64) -> (f64, f64) {
    const CONFIDENCE_THRESHOLD: f64 = 400.0;

    if record.score == SoulEntry::NO_SCORE {
        return (f64::from(record.result) / 2.0, 0.0);
    }

    let score = f64::from(record.score);
    let instance_blend = if wdl_blend >= 1.0 { 1.0 } else { wdl_blend * (score.abs() / CONFIDENCE_THRESHOLD).min(1.0) };

    let expected = sigmoid(score, k);
    let outcome = f64::from(record.result) / 2.0;
    let target = (1.0 - instance_blend).mul_add(outcome, instance_blend * expected);
    let d_target_dk = instance_blend * expected * (1.0 - expected) * score;

    (target, d_target_dk)
}

/// Overfitting detector comparing fast and slow loss trajectories.
///
/// Evaluates dual-EMA smoothed trends rather than running minima to avoid trough-selection
/// bias on noisy series. Divergence triggers when training loss is trending down while
/// validation loss is climbing beyond the estimated noise floor.
#[derive(Default)]
pub struct DivergenceMonitor {
    train_fast: f64,
    train_slow: f64,
    val_fast: f64,
    val_slow: f64,
    noise: f64,
    prev_val: f64,
    seen: usize,
}

/// Checkpoint loss records selected via smoothed trails.
///
/// Model selection tracks EMA-smoothed loss rather than raw per-epoch values to prevent
/// overfitting to stochastic troughs in evaluation noise.
#[derive(Clone, Serialize, Deserialize)]
pub struct Progress {
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    #[serde(default = "unset_smooth", deserialize_with = "unset_if_null")]
    pub val_smooth: f64,
    #[serde(default = "unset_best")]
    pub best_val_smooth: f64,
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    #[serde(default = "unset_smooth", deserialize_with = "unset_if_null")]
    pub train_smooth: f64,
    #[serde(default = "unset_best")]
    pub best_train_smooth: f64,
    pub plateau_count: usize,
}

/// Returns game-phase weights per piece type in standard layout order.
pub fn phase_weights() -> [f64; 6] {
    let params = eval_params::collect_parameters();
    let offset = LAYOUT.phase_offset;
    std::array::from_fn(|pt| params[offset + pt].value)
}

/// Computes the discrete game phase `[0..=TOTAL_PHASE]` for a position record.
pub fn phase_of(record: &FeatureRecord, phase_w: &[f64; 6]) -> usize {
    let raw: f64 = (0..6).map(|pt| f64::from(record.phase_counts[pt]) * phase_w[pt]).sum();
    raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc() as usize
}

/// Computes importance weights to balance sample density across game phases.
///
/// If `target` is `None`, reweights toward a uniform phase distribution (inverse frequency).
/// Clamps raw weights to `[1/cap, cap]` and normalizes to mean 1.0.
pub fn build_phase_weights(records: &[FeatureRecord], cap: f64, target: Option<&[f64]>) -> Vec<f64> {
    let cap = cap.max(1.0);
    let phase_w = phase_weights();

    let mut hist = vec![0u64; TOTAL_PHASE as usize + 1];
    for record in records {
        hist[phase_of(record, &phase_w)] += 1;
    }

    let active_bins = hist.iter().filter(|&&c| c > 0).count().max(1);
    let avg_count = records.len() as f64 / active_bins as f64;
    let n = records.len() as f64;
    let target_sum: f64 = target.map_or(1.0, |t| t.iter().sum::<f64>().max(1e-12));
    let (lo, hi) = (1.0 / cap, cap);

    let mut clamped = 0usize;
    let mut weights: Vec<f64> = records
        .iter()
        .map(|record| {
            let p = phase_of(record, &phase_w);
            // This record occupies its own bin, so `hist[p]` is at least one.
            let raw = match target {
                None => avg_count / hist[p] as f64,
                Some(t) => (t.get(p).copied().unwrap_or(0.0) / target_sum) / (hist[p] as f64 / n),
            };

            if raw < lo || raw > hi {
                clamped += 1;
            }
            raw.clamp(lo, hi)
        })
        .collect();

    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    for w in &mut weights {
        *w /= mean;
    }

    report_phase_balance(&hist, &weights, cap, clamped);
    weights
}

/// Multiplies per-sample dataset weights into phase weights, normalizing the result to mean 1.0.
pub fn merge_weights(phase_weights: Vec<f64>, sample_weights: &[f32]) -> Vec<f64> {
    if sample_weights.is_empty() {
        return phase_weights;
    }

    let mut weights = if phase_weights.is_empty() { vec![1.0; sample_weights.len()] } else { phase_weights };

    for (w, &s) in weights.iter_mut().zip(sample_weights) {
        *w *= f64::from(s);
    }

    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    if mean > 0.0 {
        for w in &mut weights {
            *w /= mean;
        }
    }
    weights
}

impl GradientStats {
    #[must_use]
    pub fn new(window: usize) -> Self { Self { p95: 1.0, alpha: 2.0 / (window as f64 + 1.0), count: 0 } }

    /// Updates the quantile estimate with an observed gradient norm.
    ///
    /// Steps up by `0.95` when exceeding the estimate and down by `0.05` below it,
    /// equilibrating at the 95th percentile (`0.05 · 0.95 + 0.95 · (−0.05) = 0`).
    pub fn update(&mut self, norm: f64) {
        if self.count == 0 {
            self.p95 = norm.max(1e-6);
            self.count = 1;
            return;
        }
        self.count += 1;

        let step = if norm > self.p95 { 0.95 } else { -0.05 };
        self.p95 *= (self.alpha * step).exp();
    }

    /// Returns the clipping threshold, falling back to `default` until sufficient samples accumulate.
    ///
    /// Floored at 0.1, since an estimate that decays toward zero would clip every later gradient
    /// down to nothing.
    #[must_use]
    pub fn clip_threshold(&self, default: f64) -> f64 {
        if self.count < 10 {
            return default;
        }
        self.p95.max(0.1)
    }
}

impl DivergenceMonitor {
    /// Call once per epoch, or the trails advance at the wrong rate.
    /// Returns `true` once they split.
    pub fn update(&mut self, train_loss: f64, val_loss: f64) -> bool {
        if self.seen == 0 {
            self.train_fast = train_loss;
            self.train_slow = train_loss;
            self.val_fast = val_loss;
            self.val_slow = val_loss;
        } else {
            self.train_fast += A_FAST * (train_loss - self.train_fast);
            self.train_slow += A_SLOW * (train_loss - self.train_slow);
            self.val_fast += A_FAST * (val_loss - self.val_fast);
            self.val_slow += A_SLOW * (val_loss - self.val_slow);
            self.noise += A_SLOW * ((val_loss - self.prev_val).abs() - self.noise);
        }

        self.prev_val = val_loss;
        self.seen += 1;

        self.seen > TREND_SLOW && self.train_fast < self.train_slow && (self.val_fast - self.val_slow) > TREND_NOISE_K * self.noise
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            best_val_loss: f64::MAX,
            best_val_epoch: 0,
            val_smooth: unset_smooth(),
            best_val_smooth: unset_best(),
            best_train_loss: f64::MAX,
            best_train_epoch: 0,
            train_smooth: unset_smooth(),
            best_train_smooth: unset_best(),
            plateau_count: 0,
        }
    }
}

impl Progress {
    /// Updates smoothed training loss and records if a new minimum was reached.
    pub fn record_train(&mut self, epoch: usize, loss: f64) -> bool {
        self.train_smooth = smooth(self.train_smooth, loss);
        if self.train_smooth >= self.best_train_smooth {
            return false;
        }
        self.best_train_smooth = self.train_smooth;
        self.best_train_loss = loss;
        self.best_train_epoch = epoch;
        true
    }

    /// Updates smoothed validation loss and plateau counter, returning `true` on a new record.
    pub fn record_val(&mut self, epoch: usize, loss: f64) -> bool {
        self.val_smooth = smooth(self.val_smooth, loss);
        if self.val_smooth >= self.best_val_smooth {
            self.plateau_count += 1;
            return false;
        }
        self.best_val_smooth = self.val_smooth;
        self.best_val_loss = loss;
        self.best_val_epoch = epoch;
        self.plateau_count = 0;
        true
    }
}

impl TunableData for SoulEntry {
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 { eval_f64(&self.to_board(), values) }

    #[inline]
    fn result(&self) -> f64 { f64::from(self.result) / 2.0 }
}

impl TunableData for EpdEntry {
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 { eval_f64(&self.board, values) }

    #[inline]
    fn result(&self) -> f64 { flip_wdl(self.result, self.board.stm) }
}

/// Applies an EMA step, initializing directly to `loss` if the trail is unseeded (`NaN`).
fn smooth(trail: f64, loss: f64) -> f64 { if trail.is_finite() { A_FAST.mul_add(loss - trail, trail) } else { loss } }

fn unset_smooth() -> f64 { f64::NAN }

fn unset_best() -> f64 { f64::MAX }

/// Deserializes JSON `null` as `f64::NAN` to represent uninitialized EMA trails.
fn unset_if_null<'de, D: Deserializer<'de>>(de: D) -> Result<f64, D::Error> {
    Ok(Option::<f64>::deserialize(de)?.unwrap_or_else(unset_smooth))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wobble(i: usize, amp: f64) -> f64 {
        let x = (i as f64 * 12.9898).sin() * 43_758.545_312;
        (x - x.floor()).mul_add(2.0, -1.0) * amp
    }

    #[test]
    fn divergence_quiet_on_a_noisy_plateau() {
        let mut d = DivergenceMonitor::default();
        let mut fired = 0;
        for e in 0..600 {
            let train = 0.4041 + wobble(e, 4e-6);
            let val = 0.4053 + wobble(e + 977, 20e-6);
            if d.update(train, val) {
                fired += 1;
            }
        }
        assert_eq!(fired, 0, "flat plateau must not trigger divergence");
    }

    #[test]
    fn divergence_fires_on_a_real_split() {
        let mut d = DivergenceMonitor::default();
        let mut fired = 0;
        for e in 0..600 {
            let t = e as f64;
            let train = 0.4041 - t * 2e-6 + wobble(e, 4e-6);
            let val = 0.4053 + t * 2e-6 + wobble(e + 977, 20e-6);
            if d.update(train, val) {
                fired += 1;
            }
        }
        assert!(fired > 400, "sustained divergence must flag, fired {fired} of 600");
    }

    #[test]
    fn divergence_stays_quiet_through_a_restart() {
        let mut d = DivergenceMonitor::default();
        for e in 0..200 {
            d.update(0.4041 + wobble(e, 4e-6), 0.4053 + wobble(e + 977, 20e-6));
        }

        let mut fired = 0;
        for e in 0..160 {
            let bump = 0.0012 * (-f64::from(i32::try_from(e).unwrap()) / 25.0).exp();
            if d.update(0.4041 + bump + wobble(e, 4e-6), 0.4053 + bump + wobble(e + 977, 20e-6)) {
                fired += 1;
            }
        }
        assert_eq!(fired, 0, "learning rate restart must not trigger divergence");
    }

    #[test]
    fn divergence_warms_up_before_reporting() {
        let mut d = DivergenceMonitor::default();
        for e in 0..TREND_SLOW {
            let t = e as f64;
            assert!(!d.update(0.5 - t * 1e-3, 0.5 + t * 1e-3), "reported at epoch {e}, inside warmup span");
        }
        assert!(d.update(0.5 - 0.04, 0.5 + 0.04), "must report once warmup period completes");
    }
}
