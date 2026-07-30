//! What a run measures: the WDL target and its sigmoid, phase weighting, the
//! gradient-norm percentile the clip reads, and the two trails a run keeps,
//! [`Progress`] for the records it ships and [`DivergenceMonitor`] for the one
//! it only warns on.

use serde::{Deserialize, Serialize};

pub use super::engine::sigmoid;
use super::engine::{FeatureRecord, LAYOUT, Position, TOTAL_PHASE, eval_f64, eval_params, flip_wdl};
use crate::evaltune::{loader, report::report_phase_balance};

// EMA spans in epochs. Their difference is the trend; the slow span also gates warmup,
// since a trend read before that span has filled is reading its own seed.
pub const TREND_FAST: usize = 10;
const TREND_SLOW: usize = 40;

pub const A_FAST: f64 = 2.0 / (TREND_FAST as f64 + 1.0);
const A_SLOW: f64 = 2.0 / (TREND_SLOW as f64 + 1.0);

/// Multiple of the observed per-epoch noise a rise must clear to count as divergence.
///
/// Every figure here is in units of σ, the raw per-epoch validation noise. What gets tested is
/// the smoothed difference rather than a raw value: both trails smooth the same input, so their
/// covariance leaves sd(fast − slow) at 0.21σ, well under the 0.47σ that summing their
/// deviations suggests. It is tested against the noise estimate E|Δval| = 2σ/√π ≈ 1.13σ, so one
/// unit of that is a 5.3σ bar on a 0.21σ quantity. A flat plateau stays quiet under it, and
/// drift twenty times under the epoch wobble still trips it. Raw-value intuition suggests 2 or
/// 3, which lands at 11σ here and never fires at all.
const TREND_NOISE_K: f64 = 1.0;

/// Online 95th percentile estimation of gradient norms via SGD.
///
/// The estimation happens in log-space to ensure the threshold remains positive and
/// tracks the order-of-magnitude changes in norms as the learning rate decays.
pub struct GradientStats {
    p95: f64,
    alpha: f64,
    count: usize,
}

/// Read-only evaluation trait: can compute a score and knows the game result.
///
/// Implemented by both `Entry` (raw EPD) and `SoulEntry` (encoded); the ablation
/// tool is generic over it.
pub trait TunableData: Sync + Send {
    fn eval(&self, values: &[f64]) -> f64;
    fn result(&self) -> f64;
}

/// WDL-blended training target. Near-zero search scores (low engine confidence)
/// fall back to the game result; high-magnitude scores trust the eval fully.
/// `wdl_blend >= 1.0` bypasses instance scaling for random-restart data: the
/// target is pure `sigmoid(score)`. 400 cp is the empirical saturation point.
pub fn wdl_target(entry: &loader::SoulEntry, k: f64, wdl_blend: f64) -> f64 {
    const CONFIDENCE_THRESHOLD: f64 = 400.0;

    if entry.score == loader::SoulEntry::NO_SCORE {
        return f64::from(entry.result) / 2.0;
    }
    let score = f64::from(entry.score);

    let instance_blend = if wdl_blend >= 1.0 { 1.0 } else { wdl_blend * (score.abs() / CONFIDENCE_THRESHOLD).min(1.0) };

    let expected = sigmoid(score, k);
    // result in {0,1,2} → normalize to [0.0, 1.0] for sigmoid target.
    (1.0 - instance_blend).mul_add(f64::from(entry.result) / 2.0, instance_blend * expected)
}

/// Overfitting detector: fit still improving while generalization degrades.
///
/// Neither loss is compared to its own running minimum. A running minimum over a noisy series
/// settles at the deepest trough it has seen and never recovers, so it sits below the true mean
/// by roughly the noise amplitude and every ordinary epoch afterward reads as a regression
/// against it. A trend carries no such bias, and it needs no special case at an LR restart:
/// a restart lifts both losses at once, and divergence needs train falling. Clearing the
/// trails there would only blind the detector for a slow span, so nothing clears them.
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

/// A run's best-so-far records, one per series.
///
/// Both select on a smoothed trail rather than the raw loss. A running minimum over a
/// noisy series carries the bias described on [`DivergenceMonitor`], and here it decides
/// which epoch's parameters get saved, so it would save whichever epoch the noise dug
/// deepest. The training series gets no exemption: a fixed-magnitude sign step orbits a
/// minimum rather than settling into it, so train loss stays as noisy as val until the
/// schedule decays.
#[derive(Clone, Serialize, Deserialize)]
pub struct Progress {
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    #[serde(default = "unset_smooth")]
    pub val_smooth: f64,
    #[serde(default = "unset_best")]
    pub best_val_smooth: f64,
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    #[serde(default = "unset_smooth")]
    pub train_smooth: f64,
    #[serde(default = "unset_best")]
    pub best_train_smooth: f64,
    pub plateau_count: usize,
}

/// The eval's own `PHASE`, in piece-type order.
pub fn phase_weights() -> [f64; 6] {
    let params = eval_params::collect_parameters();
    let woff = LAYOUT.phase_offset;

    std::array::from_fn(|pt| params[woff + pt].value)
}

/// Game phase of a record, `0..=TOTAL_PHASE`. Fixed for the life of a run, since `PHASE`
/// are constants rather than tunables.
pub fn phase_of(rec: &FeatureRecord, phase_w: &[f64; 6]) -> usize {
    let raw: f64 = (0..6).map(|pt| f64::from(rec.phase_counts[pt]) * phase_w[pt]).sum();

    raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc() as usize
}

/// Reweights toward `target` phase distribution, clamped to `[1/cap, cap]`.
/// `None` is uniform: inverse bucket frequency, lifting sparse phases toward
/// even representation. `Some(t)` is `target[phase] / observed[phase]`, toward
/// the density `t`. Mean-1 keeps gradient scale equal to unweighted.
pub fn build_phase_weights(records: &[FeatureRecord], cap: f64, target: Option<&[f64]>) -> Vec<f64> {
    let cap = cap.max(1.0);
    let phase_w = phase_weights();

    let mut hist = vec![0u64; TOTAL_PHASE as usize + 1];

    for rec in records {
        hist[phase_of(rec, &phase_w)] += 1;
    }

    let used = hist.iter().filter(|&&c| c > 0).count().max(1);
    let avg = records.len() as f64 / used as f64;
    let n = records.len() as f64;
    let target_sum: f64 = target.map_or(1.0, |t| t.iter().sum::<f64>().max(1e-12));
    let (lo, hi) = (1.0 / cap, cap);

    let mut clamped = 0usize;
    let mut weights: Vec<f64> = records
        .iter()
        .map(|rec| {
            let p = phase_of(rec, &phase_w);
            let raw = match target {
                // Uniform: inverse frequency, lifting sparse phases toward even weight.
                None => avg / hist[p] as f64,
                // Custom: importance weight toward the target density `t`.
                Some(t) => {
                    let observed = hist[p] as f64 / n;
                    if observed > 0.0 { (t.get(p).copied().unwrap_or(0.0) / target_sum) / observed } else { 0.0 }
                },
            };

            if raw < lo || raw > hi {
                clamped += 1;
            }

            raw.clamp(lo, hi)
        })
        .collect();

    // Mean-1 normalization keeps the gradient scale equal to an unweighted run.
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;

    for w in &mut weights {
        *w /= mean;
    }

    report_phase_balance(&hist, &weights, cap, clamped);
    weights
}

impl GradientStats {
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self { p95: 1.0, alpha: 2.0 / (window as f64 + 1.0), count: 0 }
    }

    pub fn update(&mut self, norm: f64) {
        if self.count == 0 {
            self.p95 = norm.max(1e-6);
            self.count = 1;
            return;
        }
        self.count += 1;

        // ── Online 95th percentile estimation.
        // We move up by (1-p) when norm > p95, and down by p when norm < p95.
        // Balancing: 0.05 · 0.95 (up) + 0.95 · -0.05 (down) = 0.
        let step = if norm > self.p95 { 0.95 } else { -0.05 };

        // Update in log-space. Stays positive regardless of how
        // small norms get, and the multiplicative step scales with
        // the current magnitude. This eliminates the need for a floor
        // at initialization or late in training.
        self.p95 *= (self.alpha * step).exp();
    }

    /// Returns the estimated 95th percentile of recent gradient norms.
    ///
    /// Falls back to `default` until 10 observations are collected:
    /// estimating a distribution from fewer points is noise, not signal.
    #[must_use]
    pub fn clip_threshold(&self, default: f64) -> f64 {
        if self.count < 10 {
            return default;
        }
        self.p95.max(0.1)
    }
}

impl DivergenceMonitor {
    /// Feeds one epoch, reporting whether the run is diverging.
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

        self.seen > TREND_SLOW && self.train_fast < self.train_slow && self.val_fast - self.val_slow > TREND_NOISE_K * self.noise
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
    /// Reports whether this epoch set a training record.
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

    /// Reports whether this epoch set a validation record, clearing the plateau
    /// counter if it did and advancing it otherwise.
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

impl TunableData for loader::SoulEntry {
    /// Evaluation via FEN round-trip: valid but slow.
    /// Production code uses `eval_record` with a packed `FeatureRecord`.
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 {
        let board = Position::from_fen(&self.to_fen());
        eval_f64(&board, values)
    }

    #[inline]
    fn result(&self) -> f64 {
        f64::from(self.result) / 2.0
    }
}

impl TunableData for loader::Entry {
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 {
        eval_f64(&self.board, values)
    }

    #[inline]
    fn result(&self) -> f64 {
        // EPD gives the result White-relative.
        flip_wdl(self.result, self.board.stm)
    }
}

/// A non-finite trail is the unseeded state, which is what `unset_smooth` writes.
fn smooth(trail: f64, loss: f64) -> f64 {
    if trail.is_finite() { A_FAST.mul_add(loss - trail, trail) } else { loss }
}

/// A checkpoint written before smoothed selection carries no trail; it re-seeds from the
/// first epoch after resume.
fn unset_smooth() -> f64 {
    f64::NAN
}

/// Serde's own f64 default would be 0.0, a record no smoothed loss can ever beat, which
/// would freeze the matching best-params vector at whatever the checkpoint happened to hold.
fn unset_best() -> f64 {
    f64::MAX
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic stand-in for epoch noise, so the assertions below cannot flake.
    fn wobble(i: usize, amp: f64) -> f64 {
        let x = (i as f64 * 12.9898).sin() * 43758.545_312;
        (x - x.floor()).mul_add(2.0, -1.0) * amp
    }

    #[test]
    fn divergence_quiet_on_a_noisy_plateau() {
        // Both losses flat with val wobbling 20e-6 an epoch: the shape a running-minimum
        // comparison flags on roughly every other epoch.
        let mut d = DivergenceMonitor::default();
        let mut fired = 0;

        for e in 0..600 {
            let train = 0.4041 + wobble(e, 4e-6);
            let val = 0.4053 + wobble(e + 977, 20e-6);

            if d.update(train, val) {
                fired += 1;
            }
        }

        assert_eq!(fired, 0, "flat plateau must not read as divergence");
    }

    #[test]
    fn divergence_fires_on_a_real_split() {
        // Train descending, val climbing, both under the same noise as the plateau case.
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
        // Nothing clears the trails at an LR restart, so this test carries the whole guarantee:
        // neither the jump nor the recovery that follows it may read as divergence.
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

        assert_eq!(fired, 0, "a restart cycle must not read as divergence");
    }

    #[test]
    fn divergence_warms_up_before_reporting() {
        let mut d = DivergenceMonitor::default();

        // Maximally divergent input, so warmup is the only thing holding the flag down.
        for e in 0..TREND_SLOW {
            let t = e as f64;
            assert!(!d.update(0.5 - t * 1e-3, 0.5 + t * 1e-3), "reported at epoch {e}, inside the warmup span");
        }

        assert!(d.update(0.5 - 0.04, 0.5 + 0.04), "must report once the slow span has filled");
    }
}
