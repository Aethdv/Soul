//! Parameter-group masks, WDL target blending, phase-balance weights, and run trackers.

use std::ops::Range;

use serde::{Deserialize, Deserializer, Serialize};

pub(crate) use crate::engine::sigmoid;
use crate::{
    config::EvalTuneConfig,
    engine::{FeatureRecord, LAYOUT, SoulEntry, TABLE_SQUARES, TOTAL_PHASE, Tunable, eval_params},
    lion::Masks,
    report::print_phase_balance,
};

pub(crate) const GROUP_NAMES: [&str; 4] = ["psqt", "material", "mobility", "other"];

// Mobility lanes multiply raw square and piece counts rather than 0/1 indicators.
const MOBILITY_CLAMP: f64 = 100.0;

// Keeps p + c · p² / SCALE monotonic; c < 0 puts a turnover inside reachable king pressure.
const DANGER_CURVE_CLAMP: (f64, f64) = (0.0, 256.0);

const PSQT_BETA2: f64 = 0.995;
const MOBILITY_BETA2: f64 = 0.95;

const TREND_FAST: usize = 10;
const TREND_SLOW: usize = 40;
const ALPHA_FAST: f64 = 2.0 / (TREND_FAST as f64 + 1.0);
const ALPHA_SLOW: f64 = 2.0 / (TREND_SLOW as f64 + 1.0);

// Tested against E[|Δval|] = 2σ / √π ≈ 1.13σ; shared inputs put sd(fast − slow) ≈ 0.21σ,
// so 1.0 gives a ~5.3σ barrier against plateau jitter.
const TREND_NOISE_K: f64 = 1.0;

pub(crate) struct GradientStats {
    p95:   f64,
    alpha: f64,
    count: usize,
}

impl GradientStats {
    pub(crate) fn new(window: usize) -> Self { Self { p95: 1.0, alpha: 2.0 / (window as f64 + 1.0), count: 0 } }

    /// Log-space stochastic quantile step (`0.05 · 0.95 + 0.95 · (−0.05) = 0` at p95).
    pub(crate) fn update(&mut self, norm: f64) {
        if self.count == 0 {
            self.p95 = norm.max(1e-6);
            self.count = 1;
            return;
        }
        self.count += 1;
        self.p95 *= (self.alpha * if norm > self.p95 { 0.95 } else { -0.05 }).exp();
    }

    pub(crate) fn clip_threshold(&self, default: f64) -> f64 { if self.count < 10 { default } else { self.p95.max(0.1) } }
}

#[derive(Default)]
pub(crate) struct DivergenceMonitor {
    train_fast: f64,
    train_slow: f64,
    val_fast:   f64,
    val_slow:   f64,
    noise:      f64,
    prev_val:   f64,
    seen:       usize,
}

impl DivergenceMonitor {
    /// Call once per epoch; returns `true` when train falls while val climbs past the noise bar.
    pub(crate) fn update(&mut self, train_loss: f64, val_loss: f64) -> bool {
        if self.seen == 0 {
            (self.train_fast, self.train_slow, self.val_fast, self.val_slow) = (train_loss, train_loss, val_loss, val_loss);
        } else {
            self.train_fast += ALPHA_FAST * (train_loss - self.train_fast);
            self.train_slow += ALPHA_SLOW * (train_loss - self.train_slow);
            self.val_fast += ALPHA_FAST * (val_loss - self.val_fast);
            self.val_slow += ALPHA_SLOW * (val_loss - self.val_slow);
            self.noise += ALPHA_SLOW * ((val_loss - self.prev_val).abs() - self.noise);
        }
        self.prev_val = val_loss;
        self.seen += 1;
        self.seen > TREND_SLOW && self.train_fast < self.train_slow && (self.val_fast - self.val_slow) > TREND_NOISE_K * self.noise
    }
}

/// Checkpoint selection tracks EMA-smoothed loss rather than raw troughs.
#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct Progress {
    pub(crate) best_val_loss:     f64,
    pub(crate) best_val_epoch:    usize,
    #[serde(default = "unset_smooth", deserialize_with = "unset_if_null")]
    pub(crate) val_smooth:        f64,
    #[serde(default = "unset_best")]
    pub(crate) best_val_smooth:   f64,
    pub(crate) best_train_loss:   f64,
    pub(crate) best_train_epoch:  usize,
    #[serde(default = "unset_smooth", deserialize_with = "unset_if_null")]
    pub(crate) train_smooth:      f64,
    #[serde(default = "unset_best")]
    pub(crate) best_train_smooth: f64,
    pub(crate) plateau_count:     usize,
}

impl Progress {
    pub(crate) fn record_train(&mut self, epoch: usize, loss: f64) -> bool {
        self.train_smooth = smooth(self.train_smooth, loss);
        if self.train_smooth >= self.best_train_smooth {
            return false;
        }
        (self.best_train_smooth, self.best_train_loss, self.best_train_epoch) = (self.train_smooth, loss, epoch);
        true
    }

    pub(crate) fn record_val(&mut self, epoch: usize, loss: f64) -> bool {
        self.val_smooth = smooth(self.val_smooth, loss);
        if self.val_smooth >= self.best_val_smooth {
            self.plateau_count += 1;
            return false;
        }
        (self.best_val_smooth, self.best_val_loss, self.best_val_epoch, self.plateau_count) = (self.val_smooth, loss, epoch, 0);
        true
    }
}

impl Default for Progress {
    fn default() -> Self {
        Self {
            best_val_loss:     f64::MAX,
            best_val_epoch:    0,
            val_smooth:        unset_smooth(),
            best_val_smooth:   unset_best(),
            best_train_loss:   f64::MAX,
            best_train_epoch:  0,
            train_smooth:      unset_smooth(),
            best_train_smooth: unset_best(),
            plateau_count:     0,
        }
    }
}

pub(crate) fn group_ranges(slots: usize) -> [Range<usize>; GROUP_NAMES.len()] {
    const {
        assert!(
            LAYOUT.psqt_offset + LAYOUT.psqt_len == LAYOUT.material_offset
                && LAYOUT.material_offset + LAYOUT.material_len == LAYOUT.mobility_open_offset
                && LAYOUT.mobility_open_offset + LAYOUT.mobility_open_len == LAYOUT.mobility_closed_offset
        );
    }
    let psqt = LAYOUT.material_offset.min(slots);
    let mat = LAYOUT.mobility_open_offset.min(slots);
    let mob = (LAYOUT.mobility_closed_offset + LAYOUT.mobility_closed_len).min(slots);
    [0..psqt, psqt..mat, mat..mob, mob..slots]
}

pub(crate) fn describe_beta2(config: &EvalTuneConfig) -> String {
    if config.beta2_tracks_lr {
        format!("1 - β₂ = {:.4}·lr^(2/3), one window for every group", config.beta2_lr_coefficient)
    } else {
        format!("{} constant, PSQT {PSQT_BETA2}, Mobility {MOBILITY_BETA2}", config.beta2)
    }
}

/// `1 - β₂ = c · lr^(2/3)`, balancing batch variance against curvature drift (`evaltune momentum`).
pub(crate) fn beta2_for_lr(coefficient: f64, lr: f64) -> f64 { 1.0 - (coefficient * lr.powf(2.0 / 3.0)).clamp(1e-6, 1.0) }

pub(crate) fn slot_masks(params: &[Tunable], config: &EvalTuneConfig) -> Masks {
    let slots = params.len();
    let [psqt, _mat, mob, _other] = group_ranges(slots);

    let mut decay = vec![1.0; slots];
    for i in psqt.clone() {
        let sq = i % TABLE_SQUARES;
        // Half-table files 2..=3 are c-d (mirrored to e-f), giving the c3-f6 central box.
        if (2..=5).contains(&(sq / 4)) && (2..=3).contains(&(sq % 4)) {
            decay[i] = 0.5;
        }
    }
    decay[mob.clone()].fill(1.5);

    let mut beta2 = vec![config.beta2; slots];
    beta2[psqt].fill(PSQT_BETA2);
    beta2[mob.clone()].fill(MOBILITY_BETA2);

    let mut clip = vec![(f64::NEG_INFINITY, f64::INFINITY); slots];
    clip[mob].fill((-MOBILITY_CLAMP, MOBILITY_CLAMP));
    if LAYOUT.king_danger_offset < slots {
        clip[LAYOUT.king_danger_offset] = DANGER_CURVE_CLAMP;
    }

    Masks { decay, fixed: params.iter().map(|p| p.is_fixed).collect(), beta2, lr: lr_mask(slots, config), clip }
}

pub(crate) fn lr_mask(slots: usize, config: &EvalTuneConfig) -> Vec<f64> {
    let mut lr = vec![config.lr_other; slots];
    let [psqt, mat, mob, _] = group_ranges(slots);
    lr[psqt].fill(config.lr_psqt);
    lr[mat].fill(config.lr_material);
    lr[mob].fill(config.lr_mobility);
    lr
}

/// Returns `(target, ∂target/∂k)` when `with_k_grad` is set (otherwise `0.0`), scaling `wdl_blend` by `|score| / 400` when `< 1.0`.
pub(crate) fn wdl_target(record: &FeatureRecord, k: f64, wdl_blend: f64, with_k_grad: bool) -> (f64, f64) {
    let outcome = f64::from(record.result) / 2.0;

    if wdl_blend == 0.0 || record.score == SoulEntry::NO_SCORE {
        return (outcome, 0.0);
    }

    let score = f64::from(record.score);
    let blend = if wdl_blend >= 1.0 { 1.0 } else { wdl_blend * (score.abs() / 400.0).min(1.0) };
    let expected = sigmoid(score, k);
    let target = (1.0 - blend).mul_add(outcome, blend * expected);
    let dt_dk = if with_k_grad { blend * expected * (1.0 - expected) * score } else { 0.0 };

    (target, dt_dk)
}

pub(crate) fn piece_phase_weights() -> [f64; 6] {
    let params = eval_params::collect_parameters();
    std::array::from_fn(|pt| params[LAYOUT.phase_offset + pt].value)
}

pub(crate) fn phase_of(record: &FeatureRecord, phase_w: &[f64; 6]) -> usize {
    let raw: f64 = (0..6).map(|pt| f64::from(record.phase_counts[pt]) * phase_w[pt]).sum();
    raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc() as usize
}

/// Reweights samples toward `target` (or uniform across populated phases), clamped to `[1/cap, cap]` and normalized to mean 1.0.
pub(crate) fn phase_balance_weights(records: &[FeatureRecord], cap: f64, target: Option<&[f64]>) -> Vec<f64> {
    let cap = cap.max(1.0);
    let phase_w = piece_phase_weights();
    let mut hist = vec![0u64; TOTAL_PHASE as usize + 1];
    for record in records {
        hist[phase_of(record, &phase_w)] += 1;
    }

    let avg_count = records.len() as f64 / hist.iter().filter(|&&c| c > 0).count().max(1) as f64;
    let n = records.len() as f64;
    let target_sum: f64 = target.map_or(1.0, |t| t.iter().sum::<f64>().max(1e-12));
    let (lo, hi) = (1.0 / cap, cap);

    let mut clamped = 0usize;
    let mut weights: Vec<f64> = records
        .iter()
        .map(|record| {
            let p = phase_of(record, &phase_w);
            let raw = match target {
                None => avg_count / hist[p] as f64,
                Some(t) => (t.get(p).copied().unwrap_or(0.0) / target_sum) / (hist[p] as f64 / n),
            };
            clamped += usize::from(raw < lo || raw > hi);
            raw.clamp(lo, hi)
        })
        .collect();

    normalize_unit_mean(&mut weights);
    print_phase_balance(&hist, &weights, cap, clamped);
    weights
}

pub(crate) fn merge_sample_weights(balance_weights: Vec<f64>, replay_weights: &[f32]) -> Vec<f64> {
    if replay_weights.is_empty() {
        return balance_weights;
    }
    let mut weights = if balance_weights.is_empty() { vec![1.0; replay_weights.len()] } else { balance_weights };
    for (w, &s) in weights.iter_mut().zip(replay_weights) {
        *w *= f64::from(s);
    }
    normalize_unit_mean(&mut weights);
    weights
}

#[inline]
pub(crate) fn add_slice<T: std::ops::AddAssign + Copy>(dst: &mut [T], src: &[T]) {
    debug_assert_eq!(dst.len(), src.len());
    for (a, &b) in dst.iter_mut().zip(src) {
        *a += b;
    }
}

#[inline]
pub(crate) fn l2_norm(xs: &[f64]) -> f64 { xs.iter().map(|x| x * x).sum::<f64>().sqrt() }

fn normalize_unit_mean(weights: &mut [f64]) {
    let mean = weights.iter().sum::<f64>() / weights.len() as f64;
    if mean > 0.0 {
        for w in weights {
            *w /= mean;
        }
    }
}

fn smooth(trail: f64, loss: f64) -> f64 { if trail.is_finite() { ALPHA_FAST.mul_add(loss - trail, trail) } else { loss } }
fn unset_smooth() -> f64 { f64::NAN }
fn unset_best() -> f64 { f64::MAX }
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
    fn beta2_window_shortens_with_lr() {
        assert!((beta2_for_lr(2.9403, 0.03) - 0.7161).abs() < 1e-4);
        assert!((beta2_for_lr(2.9403, 0.0001) - 0.9937).abs() < 1e-4);
        // 8^(2/3) == 4, so 8x the learning rate quadruples 1 - β₂.
        let (slow, fast) = (beta2_for_lr(2.9403, 0.001), beta2_for_lr(2.9403, 0.008));
        assert!((1.0 - fast - 4.0 * (1.0 - slow)).abs() < 1e-12, "{slow} vs {fast}");
    }

    #[test]
    fn divergence_separates_split_from_plateau_restart_and_warmup() {
        let mut d = DivergenceMonitor::default();
        assert_eq!(
            (0..600)
                .filter(|&e| d.update(0.4041 + wobble(e, 4e-6), 0.4053 + wobble(e + 977, 20e-6)))
                .count(),
            0
        );

        for e in 0..160 {
            let bump = 0.0012 * (-(e as f64) / 25.0).exp();
            assert!(!d.update(0.4041 + bump + wobble(e, 4e-6), 0.4053 + bump + wobble(e + 977, 20e-6)));
        }

        let mut d = DivergenceMonitor::default();
        for e in 0..TREND_SLOW {
            assert!(!d.update(0.5 - e as f64 * 1e-3, 0.5 + e as f64 * 1e-3), "fired during warmup at epoch {e}");
        }
        assert!(d.update(0.5 - 0.04, 0.5 + 0.04));

        let mut d = DivergenceMonitor::default();
        let fired = (0..600)
            .filter(|&e| d.update(0.4041 - e as f64 * 2e-6 + wobble(e, 4e-6), 0.4053 + e as f64 * 2e-6 + wobble(e + 977, 20e-6)))
            .count();
        assert!(fired > 400, "sustained split fired only {fired}/600 times");
    }
}
