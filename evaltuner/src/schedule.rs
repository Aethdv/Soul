//! Learning-rate and WDL-blend schedules over 1-indexed epochs `1..=total`.

use std::f64::consts::PI;

use serde::Deserialize;

const DEFAULT_WDL_END: f64 = 0.3;

#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum LrSchedule {
    Constant { value: f64 },
    Linear { start: f64, end: f64 },
    Exponential { start: f64, gamma: f64 },
    StepDecay { start: f64, gamma: f64, step_epochs: usize },
    Cosine { base: f64, min: f64, warmup_ratio: f64, cycles: usize },
    WarmupStableDecay { base: f64, min: f64, warmup_ratio: f64, stable_ratio: f64 },
    StableDecay { base: f64, min: f64, stable_ratio: f64 },
}

impl LrSchedule {
    #[inline]
    pub(crate) fn rate(self, epoch: usize, total: usize) -> f64 {
        match self {
            Self::Constant { value } => value,
            Self::Linear { start, end } => lerp(start, end, epoch - 1, total - 1),
            Self::Exponential { start, gamma } => start * gamma.powi((epoch - 1) as i32),
            Self::StepDecay { start, gamma, step_epochs } => start * gamma.powi(((epoch - 1) / step_epochs.max(1)) as i32),
            Self::Cosine { base, min, warmup_ratio, cycles } => {
                let cycle_len = total / cycles.max(1);
                let cycle_pos = (epoch - 1) % cycle_len.max(1);
                let warmup_len = (cycle_len as f64 * warmup_ratio) as usize;
                if cycle_pos < warmup_len {
                    lerp(min, base, cycle_pos, warmup_len)
                } else {
                    cosine_decay(base, min, cycle_pos - warmup_len, cycle_len - warmup_len)
                }
            },
            Self::WarmupStableDecay { base, min, warmup_ratio, stable_ratio } => {
                wsd_rate(base, min, warmup_ratio, stable_ratio, epoch, total)
            },
            Self::StableDecay { base, min, stable_ratio } => wsd_rate(base, min, 0.0, stable_ratio, epoch, total),
        }
    }

    pub(crate) fn is_restart(self, epoch: usize, total: usize) -> bool {
        matches!(self, Self::Cosine { cycles, .. } if epoch > 1 && (epoch - 1).is_multiple_of((total / cycles.max(1)).max(1)))
    }

    pub(crate) fn describe(self) -> String {
        match self {
            Self::Constant { value } => format!("Constant ({value})"),
            Self::Linear { start, end } => format!("Linear ({start} → {end})"),
            Self::Exponential { start, gamma } => format!("Exponential ({start} × {gamma}^n)"),
            Self::StepDecay { start, gamma, step_epochs } => format!("StepDecay ({start} × {gamma}^(n/{step_epochs}))"),
            Self::Cosine { base, min, cycles, .. } if cycles > 1 => format!("CosineAnnealing ({base} → {min}, {cycles} cycles)"),
            Self::Cosine { base, min, .. } => format!("CosineAnnealing ({base} → {min})"),
            Self::WarmupStableDecay { base, min, warmup_ratio, stable_ratio } => {
                format!(
                    "WarmupStableDecay ({base} → {min}, warm: {:.0}%, stable: {:.0}%)",
                    warmup_ratio * 100.0,
                    stable_ratio * 100.0
                )
            },
            Self::StableDecay { base, min, stable_ratio } => {
                format!("StableDecay ({base} → {min}, stable: {:.0}%)", stable_ratio * 100.0)
            },
        }
    }

    pub(crate) fn from_cli(
        kind: &str,
        lr: Option<f64>,
        min_lr: Option<f64>,
        warmup: Option<f64>,
        cycles: Option<usize>,
    ) -> Option<Self> {
        match kind {
            "constant" => Some(Self::Constant { value: lr.unwrap_or(0.1) }),
            "linear" => Some(Self::Linear { start: lr.unwrap_or(0.1), end: min_lr.unwrap_or(0.0) }),
            "cosine" => Some(Self::Cosine {
                base: lr.unwrap_or(0.1),
                min: min_lr.unwrap_or(0.0001),
                warmup_ratio: warmup.unwrap_or(0.1),
                cycles: cycles.unwrap_or(1),
            }),
            "warmup-stable-decay" | "wsd" => Some(Self::WarmupStableDecay {
                base: lr.unwrap_or(0.1),
                min: min_lr.unwrap_or(0.0),
                warmup_ratio: warmup.unwrap_or(0.1),
                stable_ratio: 0.5,
            }),
            "stable-decay" | "sd" => {
                Some(Self::StableDecay { base: lr.unwrap_or(0.03), min: min_lr.unwrap_or(0.0001), stable_ratio: 0.5 })
            },
            _ => None,
        }
    }

    /// `--cycles` doubles as `StepDecay::step_epochs` on the command line.
    pub(crate) fn apply_overrides(&mut self, lr: Option<f64>, min_lr: Option<f64>, warmup: Option<f64>, cycles: Option<usize>) {
        match self {
            Self::Constant { value } | Self::Exponential { start: value, .. } => set_if_some(value, lr),
            Self::Linear { start, end } | Self::StableDecay { base: start, min: end, .. } => {
                set_if_some(start, lr);
                set_if_some(end, min_lr);
            },
            Self::Cosine { base, min, warmup_ratio, cycles: count } => {
                set_if_some(base, lr);
                set_if_some(min, min_lr);
                set_if_some(warmup_ratio, warmup);
                set_if_some(count, cycles);
            },
            Self::WarmupStableDecay { base, min, warmup_ratio, .. } => {
                set_if_some(base, lr);
                set_if_some(min, min_lr);
                set_if_some(warmup_ratio, warmup);
            },
            Self::StepDecay { start, step_epochs, .. } => {
                set_if_some(start, lr);
                set_if_some(step_epochs, cycles);
            },
        }
    }
}

/// Blends the training target between game outcome (`0.0`) and search WDL (`1.0`).
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub(crate) enum WdlSchedule {
    Constant { value: f64 },
    Linear { start: f64, end: f64 },
    Cosine { start: f64, end: f64 },
    StableDecay { start: f64, end: f64, stable_ratio: f64 },
}

impl WdlSchedule {
    #[inline]
    pub(crate) fn blend(self, epoch: usize, total: usize) -> f64 {
        match self {
            Self::Constant { value } => value,
            Self::Linear { start, end } => lerp(start, end, epoch - 1, total - 1),
            Self::Cosine { start, end } => cosine_decay(start, end, epoch - 1, total - 1),
            Self::StableDecay { start, end, stable_ratio } => {
                let stable_epochs = (total as f64 * stable_ratio).round() as usize;
                if epoch <= stable_epochs {
                    start
                } else {
                    cosine_decay(start, end, epoch - stable_epochs - 1, total - stable_epochs)
                }
            },
        }
    }

    pub(crate) fn describe(self) -> String {
        match self {
            Self::Constant { value } => format!("ConstantWDL ({value})"),
            Self::Linear { start, end } => format!("LinearWDL ({start} → {end})"),
            Self::Cosine { start, end } => format!("CosineWDL ({start} → {end})"),
            Self::StableDecay { start, end, stable_ratio } => {
                format!("StableDecayWDL ({start} → {end}, stable: {:.0}%)", stable_ratio * 100.0)
            },
        }
    }

    pub(crate) fn defaults(self) -> (f64, f64) {
        match self {
            Self::Cosine { start, end } | Self::Linear { start, end } | Self::StableDecay { start, end, .. } => (start, end),
            Self::Constant { value } => (value, DEFAULT_WDL_END),
        }
    }

    pub(crate) fn from_cli(
        kind: &str,
        blend: Option<f64>,
        start: Option<f64>,
        end: Option<f64>,
        (d_start, d_end): (f64, f64),
    ) -> Option<Self> {
        let (start, end) = (start.unwrap_or(d_start), end.unwrap_or(d_end));
        match kind {
            "constant" => Some(Self::Constant { value: blend.unwrap_or(d_start) }),
            "linear" => Some(Self::Linear { start, end }),
            "cosine" => Some(Self::Cosine { start, end }),
            "stable-decay" => Some(Self::StableDecay { start, end, stable_ratio: 0.35 }),
            _ => None,
        }
    }

    /// `--blend` flattens any configured curve to a constant ratio.
    pub(crate) fn apply_overrides(&mut self, blend: Option<f64>, start: Option<f64>, end: Option<f64>) {
        if let Some(v) = blend {
            *self = Self::Constant { value: v };
            return;
        }
        if let Self::Linear { start: from, end: to }
        | Self::Cosine { start: from, end: to }
        | Self::StableDecay { start: from, end: to, .. } = self
        {
            set_if_some(from, start);
            set_if_some(to, end);
        }
    }
}

#[inline(always)]
pub(crate) fn set_if_some<T>(dst: &mut T, src: Option<T>) {
    if let Some(v) = src {
        *dst = v;
    }
}

#[inline]
fn lerp(start: f64, end: f64, step: usize, span: usize) -> f64 { start + (step as f64 / span.max(1) as f64) * (end - start) }

#[inline]
fn cosine_decay(start: f64, end: f64, step: usize, span: usize) -> f64 {
    let t = step as f64 / span.max(1) as f64;
    end + 0.5 * (start - end) * (1.0 + (PI * t).cos())
}

#[inline]
fn wsd_rate(base: f64, min: f64, warmup_ratio: f64, stable_ratio: f64, epoch: usize, total: usize) -> f64 {
    let warmup_len = (total as f64 * warmup_ratio) as usize;
    let stable_len = (total as f64 * stable_ratio) as usize;
    let decay_len = total.saturating_sub(warmup_len + stable_len);
    if epoch <= warmup_len {
        lerp(min, base, epoch, warmup_len)
    } else if epoch <= warmup_len + stable_len {
        base
    } else {
        base - ((epoch - warmup_len - stable_len) as f64 / decay_len.max(1) as f64).min(1.0) * (base - min)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lr_and_wdl_schedule_curves() {
        for (sched, epoch, want, tol) in [
            (LrSchedule::Constant { value: 0.1 }, 50, 0.1, 1e-10),
            (LrSchedule::Linear { start: 1.0, end: 0.0 }, 1, 1.0, 1e-10),
            (LrSchedule::Linear { start: 1.0, end: 0.0 }, 50, 0.5, 0.02),
            (LrSchedule::Linear { start: 1.0, end: 0.0 }, 100, 0.0, 1e-10),
            (LrSchedule::Exponential { start: 1.0, gamma: 0.9 }, 3, 0.81, 1e-10),
            (LrSchedule::StepDecay { start: 1.0, gamma: 0.5, step_epochs: 10 }, 10, 1.0, 1e-10),
            (LrSchedule::StepDecay { start: 1.0, gamma: 0.5, step_epochs: 10 }, 11, 0.5, 1e-10),
            (LrSchedule::StepDecay { start: 1.0, gamma: 0.5, step_epochs: 10 }, 21, 0.25, 1e-10),
            (LrSchedule::Cosine { base: 1.0, min: 0.0, warmup_ratio: 0.0, cycles: 1 }, 1, 1.0, 0.05),
            (LrSchedule::Cosine { base: 1.0, min: 0.0, warmup_ratio: 0.0, cycles: 1 }, 100, 0.0, 0.05),
            (LrSchedule::Cosine { base: 1.0, min: 0.0, warmup_ratio: 0.0, cycles: 2 }, 51, 1.0, 1e-10),
            (LrSchedule::StableDecay { base: 0.03, min: 0.0001, stable_ratio: 0.75 }, 75, 0.03, 1e-10),
            (LrSchedule::StableDecay { base: 0.03, min: 0.0001, stable_ratio: 0.75 }, 100, 0.0001, 1e-10),
            (LrSchedule::WarmupStableDecay { base: 1.0, min: 0.0, warmup_ratio: 0.1, stable_ratio: 0.4 }, 10, 1.0, 1e-10),
            (LrSchedule::WarmupStableDecay { base: 1.0, min: 0.0, warmup_ratio: 0.1, stable_ratio: 0.4 }, 50, 1.0, 1e-10),
            (LrSchedule::WarmupStableDecay { base: 1.0, min: 0.0, warmup_ratio: 0.1, stable_ratio: 0.4 }, 100, 0.0, 1e-10),
        ] {
            let got = sched.rate(epoch, 100);
            assert!((got - want).abs() <= tol, "{sched:?} at {epoch}/100: got {got}, want {want}");
        }
        let wsd = LrSchedule::WarmupStableDecay { base: 1.0, min: 0.0, warmup_ratio: 0.1, stable_ratio: 0.4 };
        assert!(wsd.rate(1, 100) < 1.0 && wsd.rate(51, 100) < 1.0);

        let single = LrSchedule::Cosine { base: 0.1, min: 0.0001, warmup_ratio: 0.1, cycles: 1 };
        assert!((2..=12).all(|e| !single.is_restart(e, 100)));
        let sgdr = LrSchedule::Cosine { base: 0.1, min: 0.0001, warmup_ratio: 0.1, cycles: 2 };
        assert_eq!((2..=100).filter(|&e| sgdr.is_restart(e, 100)).collect::<Vec<_>>(), [51]);

        for (sched, epoch, want, tol) in [
            (WdlSchedule::Constant { value: 0.3 }, 50, 0.3, 1e-10),
            (WdlSchedule::Linear { start: 0.0, end: 1.0 }, 1, 0.0, 1e-10),
            (WdlSchedule::Linear { start: 0.0, end: 1.0 }, 50, 0.5, 0.02),
            (WdlSchedule::Linear { start: 0.0, end: 1.0 }, 100, 1.0, 1e-10),
            (WdlSchedule::Cosine { start: 0.8, end: 0.2 }, 1, 0.8, 1e-10),
            (WdlSchedule::Cosine { start: 0.8, end: 0.2 }, 100, 0.2, 1e-10),
            (WdlSchedule::StableDecay { start: 0.8, end: 0.2, stable_ratio: 0.35 }, 35, 0.8, 1e-10),
        ] {
            let got = sched.blend(epoch, 100);
            assert!((got - want).abs() <= tol, "{sched:?} at {epoch}/100: got {got}, want {want}");
        }
    }
}
