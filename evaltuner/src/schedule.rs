//! Learning rate and WDL-blend schedules, and the combinators that compose them.
//!
//! A schedule maps the epoch onto a rate. The TOML config names one directly;
//! [`Warmup`] and [`Sequence`] wrap one from code alone.

use std::f64::consts::PI;

pub trait LrScheduler: Send + Sync {
    /// Learning rate for `epoch`, which is 1-indexed and runs to `total`.
    #[must_use]
    fn rate(&self, epoch: usize, total: usize) -> f64;
    fn describe(&self) -> String;

    /// Wrap this scheduler with linear warmup over the first `epochs` epochs.
    fn with_warmup(self, epochs: usize) -> Warmup<Self>
    where Self: Sized {
        Warmup { inner: self, warmup_epochs: epochs }
    }

    /// Chain `other` after this one, from `switch_epoch` on.
    fn then<S: LrScheduler>(self, other: S, switch_epoch: usize) -> Sequence<Self, S>
    where Self: Sized {
        Sequence { first: self, second: other, switch_epoch }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Constant {
    pub value: f64,
}

impl Constant {
    #[must_use]
    pub const fn new(value: f64) -> Self { Self { value } }
}

impl LrScheduler for Constant {
    #[inline]
    fn rate(&self, _epoch: usize, _total: usize) -> f64 { self.value }
    fn describe(&self) -> String { format!("Constant ({})", self.value) }
}

#[derive(Clone, Copy, Debug)]
pub struct Linear {
    pub start: f64,
    pub end: f64,
}

impl Linear {
    #[must_use]
    pub const fn new(start: f64, end: f64) -> Self { Self { start, end } }
}

impl LrScheduler for Linear {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        let t = (epoch - 1) as f64 / (total - 1).max(1) as f64;
        self.start + t * (self.end - self.start)
    }

    fn describe(&self) -> String { format!("Linear ({} → {})", self.start, self.end) }
}

/// Exponential decay: LR = start · gamma^epoch.
#[derive(Clone, Copy, Debug)]
pub struct Exponential {
    pub start: f64,
    pub gamma: f64,
}

impl Exponential {
    #[must_use]
    pub const fn new(start: f64, gamma: f64) -> Self { Self { start, gamma } }
}

impl LrScheduler for Exponential {
    #[inline]
    fn rate(&self, epoch: usize, _total: usize) -> f64 { self.start * self.gamma.powi((epoch - 1) as i32) }
    fn describe(&self) -> String { format!("Exponential ({} × {}^n)", self.start, self.gamma) }
}

/// Multiplicative step decay: LR = start · gamma^(epoch / step).
#[derive(Clone, Copy, Debug)]
pub struct StepDecay {
    pub start: f64,
    pub gamma: f64,
    pub step_epochs: usize,
}

impl StepDecay {
    #[must_use]
    pub const fn new(start: f64, gamma: f64, step_epochs: usize) -> Self { Self { start, gamma, step_epochs } }
}

impl LrScheduler for StepDecay {
    #[inline]
    fn rate(&self, epoch: usize, _total: usize) -> f64 {
        let steps = (epoch - 1) / self.step_epochs.max(1);
        self.start * self.gamma.powi(steps as i32)
    }

    fn describe(&self) -> String { format!("StepDecay ({} × {}^(n/{}))", self.start, self.gamma, self.step_epochs) }
}

/// Cosine annealing with configurable cycle count (SGDR when cycles > 1).
#[derive(Clone, Copy, Debug)]
pub struct CosineAnnealing {
    pub base: f64,
    pub min: f64,
    pub warmup_ratio: f64,
    pub cycles: usize,
}

impl CosineAnnealing {
    #[must_use]
    pub const fn new(base: f64, min: f64) -> Self { Self { base, min, warmup_ratio: 0.0, cycles: 1 } }

    #[must_use]
    pub const fn warmup_ratio(mut self, ratio: f64) -> Self {
        self.warmup_ratio = ratio;
        self
    }

    #[must_use]
    pub const fn cycles(mut self, n: usize) -> Self {
        self.cycles = n;
        self
    }
}

impl LrScheduler for CosineAnnealing {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        let cycle_len = total / self.cycles.max(1);
        let cycle_pos = (epoch - 1) % cycle_len.max(1);
        let warmup_len = (cycle_len as f64 * self.warmup_ratio) as usize;
        if cycle_pos < warmup_len {
            let t = cycle_pos as f64 / warmup_len.max(1) as f64;
            self.min + t * (self.base - self.min)
        } else {
            let t = (cycle_pos - warmup_len) as f64 / (cycle_len - warmup_len).max(1) as f64;
            self.min + 0.5 * (self.base - self.min) * (1.0 + (PI * t).cos())
        }
    }

    fn describe(&self) -> String {
        if self.cycles > 1 {
            format!("CosineAnnealing ({} → {}, {} cycles)", self.base, self.min, self.cycles)
        } else {
            format!("CosineAnnealing ({} → {})", self.base, self.min)
        }
    }
}

/// Warmup-Stable-Decay: LR ramps up, stays flat, then decays linearly to min.
#[derive(Clone, Copy, Debug)]
pub struct WarmupStableDecay {
    pub base: f64,
    pub min: f64,
    pub warmup_ratio: f64,
    pub stable_ratio: f64,
}

impl WarmupStableDecay {
    #[must_use]
    pub const fn new(base: f64, min: f64, warmup: f64, stable: f64) -> Self {
        Self { base, min, warmup_ratio: warmup, stable_ratio: stable }
    }
}

impl LrScheduler for WarmupStableDecay {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        let warmup_len = (total as f64 * self.warmup_ratio) as usize;
        let stable_len = (total as f64 * self.stable_ratio) as usize;
        let decay_len = total.saturating_sub(warmup_len + stable_len);
        if epoch <= warmup_len {
            let t = epoch as f64 / warmup_len.max(1) as f64;
            self.min + t * (self.base - self.min)
        } else if epoch <= warmup_len + stable_len {
            self.base
        } else {
            let t = ((epoch - warmup_len - stable_len) as f64 / decay_len.max(1) as f64).min(1.0);
            self.base - t * (self.base - self.min)
        }
    }

    fn describe(&self) -> String {
        format!(
            "WarmupStableDecay ({} → {}, warm: {:.0}%, stable: {:.0}%)",
            self.base,
            self.min,
            self.warmup_ratio * 100.0,
            self.stable_ratio * 100.0
        )
    }
}

/// Stable-Decay: LR stays flat, then decays linearly to min.
/// Lion's sign-step benefits from a prolonged exploration window at full
/// step size before settling: warmup is unnecessary for non-adaptive optimizers.
#[derive(Clone, Copy, Debug)]
pub struct StableDecay {
    pub base: f64,
    pub min: f64,
    pub stable_ratio: f64,
}

impl StableDecay {
    #[must_use]
    pub const fn new(base: f64, min: f64, stable: f64) -> Self { Self { base, min, stable_ratio: stable } }
}

impl LrScheduler for StableDecay {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        let stable_len = (total as f64 * self.stable_ratio) as usize;
        if epoch <= stable_len {
            self.base
        } else {
            let t = ((epoch - stable_len) as f64 / total.saturating_sub(stable_len).max(1) as f64).min(1.0);
            self.base - t * (self.base - self.min)
        }
    }

    fn describe(&self) -> String {
        format!("StableDecay ({} → {}, stable: {:.0}%)", self.base, self.min, self.stable_ratio * 100.0)
    }
}

/// Linear warmup wrapper: scales the inner scheduler from 0 to 1 over the first N epochs.
/// The config-reachable form is the `warmup_ratio` field on `Cosine` or `WarmupStableDecay`.
#[derive(Clone, Debug)]
pub struct Warmup<S> {
    pub inner: S,
    pub warmup_epochs: usize,
}

impl<S: LrScheduler> LrScheduler for Warmup<S> {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        let base_rate = self.inner.rate(epoch, total);
        if epoch <= self.warmup_epochs { base_rate * (epoch as f64 / self.warmup_epochs as f64) } else { base_rate }
    }
    fn describe(&self) -> String { format!("Warmup ({} epochs, {})", self.warmup_epochs, self.inner.describe()) }
}

/// Sequence two schedulers: `first` until `switch_epoch`, then `second`.
#[derive(Clone, Debug)]
pub struct Sequence<A, B> {
    pub first: A,
    pub second: B,
    pub switch_epoch: usize,
}

impl<A: LrScheduler, B: LrScheduler> LrScheduler for Sequence<A, B> {
    #[inline]
    fn rate(&self, epoch: usize, total: usize) -> f64 {
        if epoch <= self.switch_epoch {
            self.first.rate(epoch, self.switch_epoch)
        } else {
            self.second.rate(epoch - self.switch_epoch, total - self.switch_epoch)
        }
    }

    fn describe(&self) -> String {
        format!("Sequence ({} → {} @{})", self.first.describe(), self.second.describe(), self.switch_epoch)
    }
}

/// WDL blend scheduler: controls blend between game result (0.0) and WDL probs (1.0).
pub trait WdlScheduler: Send + Sync {
    #[must_use]
    fn blend(&self, epoch: usize, total: usize) -> f64;
    fn describe(&self) -> String;
}

#[derive(Clone, Copy, Debug)]
pub struct ConstantWdl {
    pub value: f64,
}

impl ConstantWdl {
    #[must_use]
    pub const fn new(value: f64) -> Self { Self { value } }
}

impl WdlScheduler for ConstantWdl {
    #[inline]
    fn blend(&self, _epoch: usize, _total: usize) -> f64 { self.value }
    fn describe(&self) -> String { format!("ConstantWDL ({})", self.value) }
}

#[derive(Clone, Copy, Debug)]
pub struct LinearWdl {
    pub start: f64,
    pub end: f64,
}

impl LinearWdl {
    #[must_use]
    pub const fn new(start: f64, end: f64) -> Self { Self { start, end } }
}

impl WdlScheduler for LinearWdl {
    #[inline]
    fn blend(&self, epoch: usize, total: usize) -> f64 {
        let t = (epoch - 1) as f64 / (total - 1).max(1) as f64;
        self.start + t * (self.end - self.start)
    }

    fn describe(&self) -> String { format!("LinearWDL ({} → {})", self.start, self.end) }
}

#[derive(Clone, Copy, Debug)]
pub struct CosineWdl {
    pub start: f64,
    pub end: f64,
}

impl CosineWdl {
    #[must_use]
    pub const fn new(start: f64, end: f64) -> Self { Self { start, end } }
}

impl WdlScheduler for CosineWdl {
    #[inline]
    fn blend(&self, epoch: usize, total: usize) -> f64 {
        let t = (epoch - 1) as f64 / (total - 1).max(1) as f64;
        self.end + 0.5 * (self.start - self.end) * (1.0 + (PI * t).cos())
    }

    fn describe(&self) -> String { format!("CosineWDL ({} → {})", self.start, self.end) }
}

/// WDL blend that stays stable for a portion of training, then cosine decays to an end value.
#[derive(Clone, Copy, Debug)]
pub struct StableDecayWdl {
    pub start: f64,
    pub end: f64,
    pub stable_ratio: f64,
}

impl StableDecayWdl {
    #[must_use]
    pub const fn new(start: f64, end: f64, stable_ratio: f64) -> Self { Self { start, end, stable_ratio } }
}

impl WdlScheduler for StableDecayWdl {
    #[inline]
    fn blend(&self, epoch: usize, total: usize) -> f64 {
        let stable_epochs = (total as f64 * self.stable_ratio).round() as usize;
        if epoch <= stable_epochs {
            self.start
        } else {
            let decay_epochs = total - stable_epochs;
            let t = (epoch - stable_epochs - 1) as f64 / decay_epochs.max(1) as f64;
            self.end + 0.5 * (self.start - self.end) * (1.0 + (PI * t).cos())
        }
    }

    fn describe(&self) -> String {
        format!("StableDecayWDL ({} → {}, stable: {:.0}%)", self.start, self.end, self.stable_ratio * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn constant_scheduler() {
        let s = Constant::new(0.1);
        assert!((s.rate(1, 100) - 0.1).abs() < 1e-10);
        assert!((s.rate(50, 100) - 0.1).abs() < 1e-10);
        assert!((s.rate(100, 100) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn linear_scheduler() {
        let s = Linear::new(1.0, 0.0);
        assert!((s.rate(1, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(100, 100) - 0.0).abs() < 1e-10);
        assert!((s.rate(50, 100) - 0.5).abs() < 0.02);
    }

    #[test]
    fn exponential_scheduler() {
        let s = Exponential::new(1.0, 0.9);
        assert!((s.rate(1, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(2, 100) - 0.9).abs() < 1e-10);
        assert!((s.rate(3, 100) - 0.81).abs() < 1e-10);
    }

    #[test]
    fn cosine_scheduler_endpoints() {
        let s = CosineAnnealing::new(1.0, 0.0);
        assert!((s.rate(1, 100) - 1.0).abs() < 0.05);
        assert!((s.rate(100, 100) - 0.0).abs() < 0.05);
    }

    #[test]
    fn cosine_with_cycles() {
        let s = CosineAnnealing::new(1.0, 0.0).cycles(2);
        assert!(s.rate(51, 100) > 0.8, "cycle 2 must restart at peak learning rate");
    }

    #[test]
    fn step_decay_scheduler() {
        let s = StepDecay::new(1.0, 0.5, 10);
        assert!((s.rate(1, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(10, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(11, 100) - 0.5).abs() < 1e-10);
        assert!((s.rate(21, 100) - 0.25).abs() < 1e-10);
    }

    #[test]
    fn warmup_combinator() {
        let s = Constant::new(1.0).with_warmup(10);
        assert!((s.rate(1, 100) - 0.1).abs() < 1e-10);
        assert!((s.rate(5, 100) - 0.5).abs() < 1e-10);
        assert!((s.rate(10, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(11, 100) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn sequence_combinator() {
        let s = Constant::new(1.0).then(Constant::new(0.1), 50);
        assert!((s.rate(1, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(50, 100) - 1.0).abs() < 1e-10);
        assert!((s.rate(51, 100) - 0.1).abs() < 1e-10);
    }

    #[test]
    fn stable_decay_phase_boundaries() {
        let s = StableDecay::new(0.03, 0.0001, 0.75);
        assert!((s.rate(1, 100) - 0.03).abs() < 1e-10);
        assert!((s.rate(75, 100) - 0.03).abs() < 1e-10, "last stable epoch must still sit at base");
        assert!(s.rate(76, 100) < 0.03, "decay must begin the epoch after");
        assert!((s.rate(100, 100) - 0.0001).abs() < 1e-10, "final epoch must land on min");
    }

    #[test]
    fn warmup_stable_decay_phase_boundaries() {
        let s = WarmupStableDecay::new(1.0, 0.0, 0.1, 0.4);
        assert!(s.rate(1, 100) < 1.0, "warmup must start below base");
        assert!((s.rate(10, 100) - 1.0).abs() < 1e-10, "warmup must reach base at its last epoch");
        assert!((s.rate(50, 100) - 1.0).abs() < 1e-10, "stable phase must hold base to its last epoch");
        assert!(s.rate(51, 100) < 1.0, "decay must begin the epoch after");
        assert!((s.rate(100, 100) - 0.0).abs() < 1e-10, "final epoch must land on min");
    }

    #[test]
    fn wdl_constant() {
        let s = ConstantWdl::new(0.3);
        assert!((s.blend(1, 100) - 0.3).abs() < 1e-10);
        assert!((s.blend(100, 100) - 0.3).abs() < 1e-10);
    }

    #[test]
    fn wdl_linear() {
        let s = LinearWdl::new(0.0, 1.0);
        assert!((s.blend(1, 100) - 0.0).abs() < 1e-10);
        assert!((s.blend(100, 100) - 1.0).abs() < 1e-10);
        assert!((s.blend(50, 100) - 0.5).abs() < 0.02);
    }
}
