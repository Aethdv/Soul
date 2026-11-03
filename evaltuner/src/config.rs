//! TOML configuration schema and loss functions.

use std::{error::Error, fmt, fs, path::Path, str::FromStr};

use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};

use crate::{
    alarm,
    schedule::{LrSchedule, WdlSchedule},
};

pub(crate) const RANDOM_INIT_SPREAD: f64 = 16.0;

const PROB_EPS: f64 = 1e-7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub(crate) enum Init {
    #[default]
    Default,
    Zero,
    Random,
}

impl FromStr for Init {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "default" => Ok(Self::Default),
            "zero" => Ok(Self::Zero),
            "random" => Ok(Self::Random),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) enum LossFn {
    #[default]
    CrossEntropy,
    MeanSquaredError,
    Focal {
        gamma: f64,
    },
    SmoothedCE {
        epsilon: f64,
    },
}

impl<'de> Deserialize<'de> for LossFn {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = LossFn;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("\"ce\", \"mse\", \"focal\", \"sce\", or a map specifying { gamma } / { epsilon }")
            }

            fn visit_str<E: de::Error>(self, value: &str) -> Result<LossFn, E> {
                value.parse().map_err(|_| de::Error::unknown_variant(value, LossFn::NAMES))
            }

            fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<LossFn, M::Error> {
                let loss = match map.next_entry::<String, f64>()? {
                    Some((k, v)) if k == "gamma" => LossFn::Focal { gamma: v },
                    Some((k, v)) if k == "epsilon" => LossFn::SmoothedCE { epsilon: v },
                    Some((k, _)) => return Err(de::Error::unknown_field(&k, &["gamma", "epsilon"])),
                    None => return Err(de::Error::custom("expected 'gamma' (Focal) or 'epsilon' (SmoothedCE)")),
                };

                if let Some((extra, _)) = map.next_entry::<String, f64>()? {
                    return Err(de::Error::custom(format!("conflicting parameter '{extra}'; specify only 'gamma' or 'epsilon'")));
                }

                Ok(loss)
            }
        }

        deserializer.deserialize_any(V)
    }
}

impl FromStr for LossFn {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, ()> {
        match s {
            "ce" => Ok(Self::CrossEntropy),
            "mse" => Ok(Self::MeanSquaredError),
            "focal" => Ok(Self::Focal { gamma: 2.0 }),
            "sce" => Ok(Self::SmoothedCE { epsilon: 0.01 }),
            _ => Err(()),
        }
    }
}

impl fmt::Display for LossFn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::CrossEntropy => f.write_str("cross-entropy"),
            Self::MeanSquaredError => f.write_str("mean squared error"),
            Self::Focal { gamma } => write!(f, "focal (γ={gamma})"),
            Self::SmoothedCE { epsilon } => write!(f, "label-smoothed cross-entropy (ε={epsilon})"),
        }
    }
}

impl LossFn {
    pub(crate) const NAMES: &'static [&'static str] = &["ce", "mse", "focal", "sce"];

    pub(crate) fn loss(self, pred: f64, target: f64) -> f64 {
        match self {
            Self::MeanSquaredError => (pred - target).powi(2),
            Self::CrossEntropy => {
                let p = clamp_prob(pred);
                -(target * p.ln() + (1.0 - target) * (1.0 - p).ln())
            },
            Self::Focal { gamma } => (clamp_prob(pred) - target).abs().powf(gamma) * Self::CrossEntropy.loss(pred, target),
            Self::SmoothedCE { epsilon } => Self::CrossEntropy.loss(pred, smooth_target(target, epsilon)),
        }
    }

    /// `∂L / ∂score`.
    pub(crate) fn grad_scale(self, pred: f64, target: f64, k: f64) -> f64 {
        let err = pred - target;
        match self {
            Self::MeanSquaredError => 2.0 * err * pred * (1.0 - pred) * k,
            Self::CrossEntropy => err * k,
            Self::Focal { gamma } => {
                let p = clamp_prob(pred);
                let ce = Self::CrossEntropy.loss(pred, target);
                let diff = p - target;
                let base = diff.abs().max(1e-12);
                base.powf(gamma) * diff * k + gamma * base.powf(gamma - 1.0) * diff.signum() * k * p * (1.0 - p) * ce
            },
            Self::SmoothedCE { epsilon } => Self::CrossEntropy.grad_scale(pred, smooth_target(target, epsilon), k),
        }
    }

    /// `∂²L / ∂score²`, scaling `Σ w_i · H_i · a_i a_iᵀ` in `curvature`.
    pub(crate) fn hessian_scale(self, pred: f64, target: f64, k: f64) -> f64 {
        match self {
            Self::MeanSquaredError => {
                let u = pred * (1.0 - pred);
                2.0 * k * k * u * (u + (pred - target) * (1.0 - 2.0 * pred))
            },
            Self::CrossEntropy => k * k * pred * (1.0 - pred),
            // Differentiating b^γ · CE twice yields three product-rule terms;
            // a CE-shaped approximation drops the two from |S − T|.
            Self::Focal { gamma } => {
                let p = clamp_prob(pred);
                let (u, ce, diff) = (p * (1.0 - p), Self::CrossEntropy.loss(pred, target), p - target);
                let base = diff.abs().max(1e-12);
                let h = (2.0 * gamma + 1.0) * u * base.powf(gamma)
                    + gamma * (gamma - 1.0) * u * u * ce * base.powf(gamma - 2.0)
                    + gamma * diff.signum() * u * (1.0 - 2.0 * p) * ce * base.powf(gamma - 1.0);
                k * k * h
            },
            Self::SmoothedCE { epsilon } => Self::CrossEntropy.hessian_scale(pred, smooth_target(target, epsilon), k),
        }
    }

    /// `∂L / ∂target`, needed for `∂L / ∂K` when a WDL blend moves the target with K.
    pub(crate) fn grad_target(self, pred: f64, target: f64) -> f64 {
        match self {
            Self::MeanSquaredError => -2.0 * (pred - target),
            Self::CrossEntropy => {
                let p = clamp_prob(pred);
                (1.0 - p).ln() - p.ln()
            },
            Self::Focal { gamma } => {
                let (p, ce) = (clamp_prob(pred), Self::CrossEntropy.loss(pred, target));
                let diff = p - target;
                let base = diff.abs();
                // Floor only for γ ≤ 1 where γ − 1 ≤ 0; for γ > 1 the derivative is 0 at p == target.
                let powered = if gamma <= 1.0 { base.max(1e-12) } else { base };
                base.powf(gamma) * Self::CrossEntropy.grad_target(pred, target)
                    - gamma * powered.powf(gamma - 1.0) * diff.signum() * ce
            },
            Self::SmoothedCE { epsilon } => Self::CrossEntropy.grad_target(pred, smooth_target(target, epsilon)) * (1.0 - epsilon),
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub(crate) enum KMode {
    Sweep {
        #[serde(default = "default_k_sweep_interval")]
        interval: usize,
    },
    Learned {
        #[serde(default = "default_k_lr_mult")]
        lr_mult: f64,
    },
    Fixed {
        value: f64,
    },
}

impl Default for KMode {
    fn default() -> Self { Self::Sweep { interval: default_k_sweep_interval() } }
}

impl fmt::Display for KMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Sweep { interval } => write!(f, "Sweep ({interval})"),
            Self::Learned { lr_mult } => write!(f, "Learned ({lr_mult})"),
            Self::Fixed { value } => write!(f, "Fixed ({value})"),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct TunerConfig {
    pub(crate) evaltune: EvalTuneConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub(crate) struct EvalTuneConfig {
    pub(crate) beta1: f64,
    pub(crate) beta2: f64,
    #[serde(default)]
    pub(crate) beta2_tracks_lr: bool,
    /// `cbrt(4 · drift² / variance)` from `evaltune momentum`.
    #[serde(default = "default_beta2_lr_coefficient")]
    pub(crate) beta2_lr_coefficient: f64,
    pub(crate) weight_decay: f64,
    #[serde(default = "default_ema_decay")]
    pub(crate) ema_decay: f64,
    pub(crate) grad_clip: f64,
    pub(crate) k_min: f64,
    pub(crate) k_max: f64,
    #[serde(default)]
    pub(crate) k_mode: KMode,
    #[serde(default)]
    pub(crate) loss: LossFn,
    pub(crate) batch_size: usize,
    pub(crate) epochs: usize,
    #[serde(default)]
    pub(crate) epoch_sample: Option<f64>,
    #[serde(default)]
    pub(crate) unfreeze_epoch: usize,
    /// Block size for blocked permutation; `0` runs a full shuffle.
    #[serde(default)]
    pub(crate) shuffle_block: usize,
    #[serde(default = "default_log_path")]
    pub(crate) log_path: String,
    #[serde(default = "default_patience")]
    pub(crate) patience: usize,
    pub(crate) lr_schedule: LrSchedule,
    pub(crate) wdl_schedule: WdlSchedule,
    #[serde(default = "default_one")]
    pub(crate) lr_psqt: f64,
    #[serde(default = "default_one")]
    pub(crate) lr_material: f64,
    #[serde(default = "default_one")]
    pub(crate) lr_mobility: f64,
    #[serde(default = "default_one")]
    pub(crate) lr_other: f64,
    #[serde(default)]
    pub(crate) seed: Option<u64>,
    #[serde(default)]
    pub(crate) split_seed: Option<u64>,
    #[serde(default)]
    pub(crate) init: Init,
    #[serde(default)]
    pub(crate) gate_tally: bool,
    #[serde(default = "default_true")]
    pub(crate) auto_freeze: bool,
    #[serde(default = "default_freeze_start")]
    pub(crate) freeze_start_epoch: usize,
    #[serde(default = "default_freeze_cadence")]
    pub(crate) freeze_cadence: usize,
    #[serde(default = "default_freeze_threshold")]
    pub(crate) freeze_threshold: f64,
    #[serde(default = "default_freeze_consecutive")]
    pub(crate) freeze_consecutive: usize,
    #[serde(default)]
    pub(crate) replay_filter: Option<String>,
    /// Centipawn cutoff on `|static - search|`; `0` disables filtering.
    #[serde(default)]
    pub(crate) volatility_threshold: i16,
    #[serde(default = "default_true")]
    pub(crate) volatility_adaptive: bool,
    #[serde(default)]
    pub(crate) phase_balance: bool,
    /// Clamps phase-balance weights to `1/c ..= c`.
    #[serde(default = "default_phase_balance_cap")]
    pub(crate) phase_balance_cap: f64,
    #[serde(default)]
    pub(crate) phase_target: Option<Vec<f64>>,
    #[serde(default)]
    pub(crate) val_max: Option<usize>,
}

impl TunerConfig {
    /// Resolves a relative `replay_filter` path against the config file's own directory.
    pub(crate) fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(path).inspect_err(|e| alarm!("Failed to read config file '{path}': {e}"))?;
        let mut config: Self = toml::from_str(&contents).inspect_err(|e| alarm!("Failed to parse TOML from '{path}': {e}"))?;

        if let (Some(dir), Some(filter)) = (Path::new(path).parent(), config.evaltune.replay_filter.as_ref())
            && Path::new(filter).is_relative()
        {
            config.evaltune.replay_filter = Some(dir.join(filter).to_string_lossy().into_owned());
        }

        Ok(config)
    }
}

#[inline]
fn clamp_prob(pred: f64) -> f64 { pred.clamp(PROB_EPS, 1.0 - PROB_EPS) }

#[inline]
fn smooth_target(target: f64, eps: f64) -> f64 { target * (1.0 - eps) + 0.5 * eps }

const fn default_patience() -> usize { 100 }
const fn default_ema_decay() -> f64 { 0.999 }
const fn default_true() -> bool { true }
const fn default_freeze_start() -> usize { 500 }
const fn default_freeze_cadence() -> usize { 100 }
const fn default_freeze_threshold() -> f64 { 1e-7 }
const fn default_freeze_consecutive() -> usize { 2 }
const fn default_one() -> f64 { 1.0 }
const fn default_beta2_lr_coefficient() -> f64 { 2.9403 }
const fn default_phase_balance_cap() -> f64 { 8.0 }
const fn default_k_sweep_interval() -> usize { 200 }
const fn default_k_lr_mult() -> f64 { 0.001 }

fn default_log_path() -> String { "evaltune.jsonl".into() }

#[cfg(test)]
impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            evaltune: EvalTuneConfig {
                lr_schedule: LrSchedule::Cosine { base: 0.1, min: 0.0001, warmup_ratio: 0.1, cycles: 1 },
                wdl_schedule: WdlSchedule::Constant { value: 0.3 },
                beta1: 0.9,
                beta2: 0.99,
                beta2_tracks_lr: false,
                beta2_lr_coefficient: default_beta2_lr_coefficient(),
                weight_decay: 0.00001,
                loss: LossFn::CrossEntropy,
                batch_size: 65536,
                epochs: 4000,
                shuffle_block: 0,
                log_path: default_log_path(),
                grad_clip: 1.0,
                k_min: 0.003,
                k_max: 0.010,
                k_mode: KMode::default(),
                patience: default_patience(),
                ema_decay: default_ema_decay(),
                unfreeze_epoch: 0,
                seed: None,
                split_seed: None,
                init: Init::Default,
                gate_tally: false,
                auto_freeze: true,
                freeze_start_epoch: default_freeze_start(),
                freeze_cadence: default_freeze_cadence(),
                freeze_threshold: default_freeze_threshold(),
                freeze_consecutive: default_freeze_consecutive(),
                replay_filter: None,
                volatility_threshold: 0,
                volatility_adaptive: true,
                phase_balance: false,
                phase_balance_cap: default_phase_balance_cap(),
                phase_target: None,
                epoch_sample: None,
                val_max: None,
                lr_psqt: 1.0,
                lr_material: 1.0,
                lr_mobility: 1.0,
                lr_other: 1.0,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::sigmoid;

    fn parse_loss(spec: &str) -> Result<LossFn, toml::de::Error> {
        #[derive(Deserialize)]
        struct Wrap {
            loss: LossFn,
        }
        toml::from_str::<Wrap>(&format!("loss = {spec}")).map(|w| w.loss)
    }

    #[test]
    fn loss_derivatives_match_finite_differences() {
        for loss in [
            LossFn::CrossEntropy,
            LossFn::MeanSquaredError,
            LossFn::Focal { gamma: 0.5 },
            LossFn::Focal { gamma: 1.5 },
            LossFn::Focal { gamma: 2.0 },
            LossFn::SmoothedCE { epsilon: 0.1 },
        ] {
            for &k in &[0.002, 0.005] {
                for &s in &[-400.0, 0.0, 400.0] {
                    let sig = sigmoid(s, k);
                    for &t in &[0.0, 0.3, 1.0] {
                        // Focal's |S − T| factor is non-differentiable at S == T.
                        if (sig - t).abs() < 0.05 {
                            continue;
                        }
                        let (ds, sig_hi, sig_lo) = (1e-3, sigmoid(s + 1e-3, k), sigmoid(s - 1e-3, k));
                        let (g, g_fd) = (loss.grad_scale(sig, t, k), (loss.loss(sig_hi, t) - loss.loss(sig_lo, t)) / (2.0 * ds));
                        assert!((g - g_fd).abs() <= 1e-4 * g.abs().max(1e-9) + 1e-12, "grad {loss:?} {s} {t} {k}: {g} vs {g_fd}");
                        let (h, h_fd) = (
                            loss.hessian_scale(sig, t, k),
                            (loss.grad_scale(sig_hi, t, k) - loss.grad_scale(sig_lo, t, k)) / (2.0 * ds),
                        );
                        assert!((h - h_fd).abs() <= 1e-4 * h.abs().max(1e-9) + 1e-12, "hess {loss:?} {s} {t} {k}: {h} vs {h_fd}");
                    }
                }
            }

            for (pred, t) in [(0.3_f64, 0.2_f64), (0.5, 0.5), (0.8, 0.9), (0.1, 0.9), (0.9999, 0.5)] {
                if matches!(loss, LossFn::Focal { gamma } if gamma <= 1.0) && (pred - t).abs() < 1e-6 {
                    continue;
                }
                let fd = (loss.loss(pred, t + 1e-7) - loss.loss(pred, t - 1e-7)) / 2e-7;
                let closed = loss.grad_target(pred, t);
                assert!((fd - closed).abs() < 1e-6, "grad_target {loss:?} ({pred}, {t}): {closed} vs {fd}");
            }
        }
    }

    #[test]
    fn loss_and_config_toml_parsing() {
        for (spec, want) in [
            ("\"ce\"", LossFn::CrossEntropy),
            ("\"mse\"", LossFn::MeanSquaredError),
            ("\"focal\"", LossFn::Focal { gamma: 2.0 }),
            ("\"sce\"", LossFn::SmoothedCE { epsilon: 0.01 }),
            ("{ gamma = 2.5 }", LossFn::Focal { gamma: 2.5 }),
            ("{ epsilon = 0.02 }", LossFn::SmoothedCE { epsilon: 0.02 }),
        ] {
            assert_eq!(parse_loss(spec).unwrap(), want, "{spec}");
        }
        for (bad, needle) in [
            ("\"crossentropy\"", "crossentropy"),
            ("{ delta = 1.0 }", "delta"),
            ("{ gamma = 2.0, epsilon = 0.01 }", "epsilon"),
        ] {
            assert!(parse_loss(bad).unwrap_err().to_string().contains(needle), "{bad}");
        }

        let raw = fs::read_to_string("config.toml").expect("read config.toml");
        TunerConfig::from_file("config.toml").expect("config.toml must parse");
        assert!(
            toml::from_str::<TunerConfig>(&format!("log_levle = \"info\"\n{raw}"))
                .unwrap_err()
                .to_string()
                .contains("log_levle")
        );
    }
}
