//! Pipeline configuration schemas, loss function definitions, and schedule builders.

use std::{error::Error, fs, path::Path};

use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};

use crate::{
    engine::color::{ALARM_PEN, RESET},
    schedule::{self, LrScheduler, WdlScheduler},
};

pub const DEFAULT_WDL_END: f64 = 0.3;
pub const DEFAULT_K_LR_MULT: f64 = 0.001;
pub const DEFAULT_K_SWEEP_INTERVAL: usize = 200;

/// Half-width of the uniform draw for [`Init::Random`].
pub const RANDOM_INIT_SPREAD: f64 = 16.0;

/// Weight initialization strategy prior to optimization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    /// Load baseline engine weights.
    #[default]
    Default,
    /// Zero all parameter vectors to test dataset identifiability.
    Zero,
    /// Sample uniformly in `-spread..spread` around the default weights.
    Random,
}

/// Objective loss functions for evaluation parameter optimization.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum LossFn {
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

struct LossFnVisitor;

impl<'de> Visitor<'de> for LossFnVisitor {
    type Value = LossFn;

    fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        f.write_str("\"ce\", \"mse\", \"focal\", \"sce\", or a map specifying { gamma } / { epsilon }")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<LossFn, E> {
        value.parse().map_err(|_| de::Error::unknown_variant(value, LossFn::NAMES))
    }

    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<LossFn, M::Error> {
        let loss = match map.next_entry::<String, f64>()? {
            Some((key, value)) if key == "gamma" => LossFn::Focal { gamma: value },
            Some((key, value)) if key == "epsilon" => LossFn::SmoothedCE { epsilon: value },
            Some((key, _)) => return Err(de::Error::unknown_field(&key, &["gamma", "epsilon"])),
            None => return Err(de::Error::custom("expected 'gamma' (Focal) or 'epsilon' (SmoothedCE)")),
        };

        if let Some((extra, _)) = map.next_entry::<String, f64>()? {
            return Err(de::Error::custom(format!("conflicting parameter '{extra}'; specify only 'gamma' or 'epsilon'")));
        }

        Ok(loss)
    }
}

impl<'de> Deserialize<'de> for LossFn {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where D: serde::Deserializer<'de> {
        deserializer.deserialize_any(LossFnVisitor)
    }
}

impl std::str::FromStr for LossFn {
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

impl std::fmt::Display for LossFn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::CrossEntropy => f.write_str("cross-entropy"),
            Self::MeanSquaredError => f.write_str("mean squared error"),
            Self::Focal { gamma } => write!(f, "focal (γ={gamma})"),
            Self::SmoothedCE { epsilon } => write!(f, "label-smoothed cross-entropy (ε={epsilon})"),
        }
    }
}

impl LossFn {
    pub const NAMES: &'static [&'static str] = &["ce", "mse", "focal", "sce"];

    /// Loss of the sigmoid prediction against a target outcome in `0.0..=1.0`.
    pub fn loss(self, sig: f64, target: f64) -> f64 {
        match self {
            Self::MeanSquaredError => {
                let err = sig - target;
                err * err
            },
            Self::CrossEntropy => {
                let s = sig.clamp(1e-7, 1.0 - 1e-7);
                -(target * s.ln() + (1.0 - target) * (1.0 - s).ln())
            },
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let ce = Self::CrossEntropy.loss(sig, target);
                let base = (prob - target).abs();
                base.powf(gamma) * ce
            },
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.loss(sig, t)
            },
        }
    }

    /// First derivative of the loss with respect to the evaluation score.
    pub fn grad_scale(self, sig: f64, target: f64, k: f64) -> f64 {
        let err = sig - target;
        match self {
            Self::MeanSquaredError => 2.0 * err * sig * (1.0 - sig) * k,
            Self::CrossEntropy => err * k,
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let ce = Self::CrossEntropy.loss(sig, target);
                let diff = prob - target;
                let base = diff.abs().max(1e-12);
                let ce_grad = base.powf(gamma) * diff * k;
                let focal_grad = gamma * base.powf(gamma - 1.0) * diff.signum() * k * prob * (1.0 - prob) * ce;
                ce_grad + focal_grad
            },
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.grad_scale(sig, t, k)
            },
        }
    }

    /// Second derivative of the loss with respect to the evaluation score.
    ///
    /// The curvature probes build the parameter Hessian from it, summing `w_i · H · a_i a_iᵀ`.
    pub fn hessian_scale(self, sig: f64, target: f64, k: f64) -> f64 {
        match self {
            Self::MeanSquaredError => {
                let u = sig * (1.0 - sig);
                2.0 * k * k * u * (u + (sig - target) * (1.0 - 2.0 * sig))
            },
            Self::CrossEntropy => k * k * sig * (1.0 - sig),
            // FL = b^γ·CE; differentiating the product twice by the product rule
            // yields three terms, and the two that come from the |S−T| factor's
            // own second derivative are the ones a CE-shaped guess would drop.
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let u = prob * (1.0 - prob);
                let ce = Self::CrossEntropy.loss(sig, target);
                let diff = prob - target;
                let base = diff.abs().max(1e-12);
                let b_gamma = base.powf(gamma);
                let h = (2.0 * gamma + 1.0) * u * b_gamma
                    + gamma * (gamma - 1.0) * u * u * ce * base.powf(gamma - 2.0)
                    + gamma * diff.signum() * u * (1.0 - 2.0 * prob) * ce * base.powf(gamma - 1.0);
                k * k * h
            },
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.hessian_scale(sig, t, k)
            },
        }
    }

    /// Derivative of the loss with respect to the target outcome.
    ///
    /// Needed to optimize K under a WDL schedule, where the target moves with K.
    pub fn grad_target(self, sig: f64, target: f64) -> f64 {
        match self {
            Self::MeanSquaredError => -2.0 * (sig - target),
            Self::CrossEntropy => {
                let s = sig.clamp(1e-7, 1.0 - 1e-7);
                (1.0 - s).ln() - s.ln()
            },
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let ce = Self::CrossEntropy.loss(sig, target);
                let diff = prob - target;
                // The floor exists for γ ≤ 1, where the exponent γ − 1 is
                // non-positive and a zero base would blow up. For γ > 1 the
                // derivative is genuinely 0 at the coincidence point, and the
                // floor would fake a step there.
                let base = diff.abs();
                let powered = if gamma <= 1.0 { base.max(1e-12) } else { base };
                let ce_grad = base.powf(gamma) * Self::CrossEntropy.grad_target(sig, target);
                let focal_grad = -gamma * powered.powf(gamma - 1.0) * diff.signum() * ce;

                ce_grad + focal_grad
            },
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.grad_target(sig, t) * (1.0 - epsilon)
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum KMode {
    /// Re-estimates K by golden-section search every `interval` epochs.
    Sweep {
        #[serde(default = "default_k_sweep_interval")]
        interval: usize,
    },
    /// Learns K alongside the parameters by gradient descent.
    Learned {
        #[serde(default = "default_k_lr_mult")]
        lr_mult: f64,
    },
    /// Holds K constant.
    Fixed { value: f64 },
}

impl Default for KMode {
    fn default() -> Self { Self::Sweep { interval: DEFAULT_K_SWEEP_INTERVAL } }
}

impl std::fmt::Display for KMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            KMode::Sweep { interval } => write!(f, "Sweep ({interval})"),
            KMode::Learned { lr_mult } => write!(f, "Learned ({lr_mult})"),
            KMode::Fixed { value } => write!(f, "Fixed ({value})"),
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum LrScheduleConfig {
    Constant {
        value: f64,
    },
    Linear {
        start: f64,
        end: f64,
    },
    Exponential {
        start: f64,
        gamma: f64,
    },
    StepDecay {
        start: f64,
        gamma: f64,
        step_epochs: usize,
    },
    Cosine {
        base: f64,
        min: f64,
        warmup_ratio: f64,
        cycles: usize,
    },
    #[serde(rename = "wsd")]
    WarmupStableDecay {
        base: f64,
        min: f64,
        warmup_ratio: f64,
        stable_ratio: f64,
    },
    #[serde(rename = "sd")]
    StableDecay {
        base: f64,
        min: f64,
        stable_ratio: f64,
    },
}

impl LrScheduleConfig {
    pub fn into_scheduler(self) -> Box<dyn LrScheduler> {
        match self {
            Self::Constant { value } => Box::new(schedule::Constant::new(value)),
            Self::Linear { start, end } => Box::new(schedule::Linear::new(start, end)),
            Self::Exponential { start, gamma } => Box::new(schedule::Exponential::new(start, gamma)),
            Self::StepDecay { start, gamma, step_epochs } => Box::new(schedule::StepDecay::new(start, gamma, step_epochs)),
            Self::Cosine { base, min, warmup_ratio, cycles } => {
                Box::new(schedule::CosineAnnealing::new(base, min).warmup_ratio(warmup_ratio).cycles(cycles))
            },
            Self::WarmupStableDecay { base, min, warmup_ratio, stable_ratio } => {
                Box::new(schedule::WarmupStableDecay::new(base, min, warmup_ratio, stable_ratio))
            },
            Self::StableDecay { base, min, stable_ratio } => Box::new(schedule::StableDecay::new(base, min, stable_ratio)),
        }
    }

    /// Mutates schedule parameters in-place from CLI flags.
    pub fn apply_overrides(&mut self, lr: Option<f64>, min_lr: Option<f64>, warmup: Option<f64>, cycles: Option<usize>) {
        match self {
            Self::Constant { value } => set(value, lr),
            Self::Linear { start, end } => {
                set(start, lr);
                set(end, min_lr);
            },
            Self::Cosine { base, min, warmup_ratio, cycles: count, .. } => {
                set(base, lr);
                set(min, min_lr);
                set(warmup_ratio, warmup);
                set(count, cycles);
            },
            Self::WarmupStableDecay { base, min, warmup_ratio, .. } => {
                set(base, lr);
                set(min, min_lr);
                set(warmup_ratio, warmup);
            },
            Self::StableDecay { base, min, .. } => {
                set(base, lr);
                set(min, min_lr);
            },
            Self::Exponential { start, .. } => set(start, lr),
            Self::StepDecay { start, step_epochs, .. } => {
                set(start, lr);
                set(step_epochs, cycles);
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum WdlScheduleConfig {
    Constant { value: f64 },
    Linear { start: f64, end: f64 },
    Cosine { start: f64, end: f64 },
    StableDecay { start: f64, end: f64, stable_ratio: f64 },
}

impl WdlScheduleConfig {
    pub fn is_active(&self) -> bool {
        match self {
            Self::Constant { value } => *value > 0.0,
            Self::Linear { start, end } => *start > 0.0 || *end > 0.0,
            Self::Cosine { start, end } => *start > 0.0 || *end > 0.0,
            Self::StableDecay { start, end, .. } => *start > 0.0 || *end > 0.0,
        }
    }

    pub fn into_scheduler(self) -> Box<dyn WdlScheduler> {
        match self {
            Self::Constant { value } => Box::new(schedule::ConstantWdl::new(value)),
            Self::Linear { start, end } => Box::new(schedule::LinearWdl::new(start, end)),
            Self::Cosine { start, end } => Box::new(schedule::CosineWdl::new(start, end)),
            Self::StableDecay { start, end, stable_ratio } => Box::new(schedule::StableDecayWdl::new(start, end, stable_ratio)),
        }
    }

    pub fn defaults(&self) -> (f64, f64) {
        match self {
            Self::Cosine { start, end } | Self::Linear { start, end } | Self::StableDecay { start, end, .. } => (*start, *end),
            Self::Constant { value } => (*value, DEFAULT_WDL_END),
        }
    }

    pub fn apply_overrides(&mut self, blend: Option<f64>, start: Option<f64>, end: Option<f64>) {
        if let Some(v) = blend {
            *self = Self::Constant { value: v };
            return;
        }

        match self {
            Self::Linear { start: from, end: to }
            | Self::Cosine { start: from, end: to }
            | Self::StableDecay { start: from, end: to, .. } => {
                set(from, start);
                set(to, end);
            },
            Self::Constant { .. } => {},
        }
    }
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct TunerConfig {
    pub evaltune: EvalTuneConfig,
    pub searchtune: SearchTuneConfig,
    pub general: GeneralConfig,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct EvalTuneConfig {
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    /// Polyak exponential moving average decay per batch update.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    pub grad_clip: f64,
    pub k_min: f64,
    pub k_max: f64,
    #[serde(default)]
    pub k_mode: KMode,
    #[serde(default)]
    pub loss: LossFn,
    pub batch_size: usize,
    pub epochs: usize,
    /// Subsampling fraction drawn randomly per epoch. `None` samples the full split.
    #[serde(default)]
    pub epoch_sample: Option<f64>,
    /// Warmup epoch at which non-material parameters (mobility, king safety) unfreeze.
    #[serde(default)]
    pub unfreeze_epoch: usize,
    /// Chunk size for blocked sequential shuffling. 0 enforces a full random permutation.
    #[serde(default)]
    pub shuffle_block: usize,
    /// Path for appending JSONL training telemetry.
    #[serde(default = "default_log_path")]
    pub log_path: String,
    /// Validation plateau patience (in epochs) before halving learning rate.
    #[serde(default = "default_patience")]
    pub patience: usize,
    pub lr_schedule: LrScheduleConfig,
    pub wdl_schedule: WdlScheduleConfig,
    #[serde(default = "default_one")]
    pub lr_psqt: f64,
    #[serde(default = "default_one")]
    pub lr_material: f64,
    #[serde(default = "default_one")]
    pub lr_mobility: f64,
    #[serde(default = "default_one")]
    pub lr_other: f64,
    /// Seed for training batch ordering. Randomly initialized if `None`.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Seed for the validation holdout split.
    #[serde(default)]
    pub split_seed: Option<u64>,
    #[serde(default)]
    pub init: Init,
    /// Collects per-group agreement telemetry for the Lion optimizer.
    #[serde(default)]
    pub gate_census: bool,
    /// Freezes updates to parameters whose gradient magnitude remains stagnant.
    #[serde(default = "default_true")]
    pub auto_freeze: bool,
    /// Earliest epoch to activate parameter stagnation checks.
    #[serde(default = "default_freeze_start")]
    pub freeze_start_epoch: usize,
    /// Stagnation check cadence (in epochs).
    #[serde(default = "default_freeze_cadence")]
    pub freeze_cadence: usize,
    /// Gradient EMA floor below which a parameter is considered stagnant.
    #[serde(default = "default_freeze_threshold")]
    pub freeze_threshold: f64,
    /// Required consecutive stagnant checks before locking a parameter.
    #[serde(default = "default_freeze_consecutive")]
    pub freeze_consecutive: usize,
    /// Optional viriformat replay filter file path.
    #[serde(default)]
    pub replay_filter: Option<String>,
    /// Prunes a label when `|static - search|` in centipawns exceeds this. 0 disables.
    #[serde(default)]
    pub volatility_threshold: i16,
    /// Scales volatility threshold dynamically with remaining piece count.
    #[serde(default = "default_true")]
    pub volatility_adaptive: bool,
    /// Reweights samples inversely proportional to game-phase frequency.
    #[serde(default)]
    pub phase_balance: bool,
    /// Ceiling on a phase-balancing sample weight, which clamps to `1/c ..= c`.
    #[serde(default = "default_phase_balance_cap")]
    pub phase_balance_cap: f64,
    /// Target game-phase density distribution (`0..=TOTAL_PHASE`, 25 buckets). `None` targets uniform.
    #[serde(default)]
    pub phase_target: Option<Vec<f64>>,
    /// Hard cap on validation slice size.
    #[serde(default)]
    pub val_max: Option<usize>,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct SearchTuneConfig {
    pub population_scale: f64,
    pub sigma_init: f64,
    #[serde(default)]
    pub sigma_restart: f64,
    pub min_sigma: f64,
    pub max_restarts: usize,
    pub stagnation_threshold: usize,
    #[serde(default)]
    pub h2h_pairs: usize,
    #[serde(default)]
    pub validation_pairs: usize,
    pub reeval_interval: usize,
    pub confidence_factor: f64,
    #[serde(default = "default_smoothing_radius")]
    pub smoothing_radius: f64,
    pub epochs: usize,
    pub pairs: usize,
    pub tc: Option<String>,
    #[serde(default)]
    pub h2h_tc: Option<String>,
    #[serde(default)]
    pub val_tc: Option<String>,
    pub speed_penalty: f64,
    pub centering_penalty: f64,
    pub active_softness: f64,
    pub sigma_boost_factor: f64,

    /// Snapshot interval (in generations) for parameter boundary saturation reports.
    #[serde(default = "default_bounds_report_interval")]
    pub bounds_report_interval: usize,
    /// Sliding window length (generations) for boundary clamp rate monitoring.
    #[serde(default = "default_bounds_window_gens")]
    pub bounds_window_gens: usize,
    /// Multiplier on expected boundary hit rate required to flag a widening warning.
    #[serde(default = "default_bounds_alarm_multiplier")]
    pub bounds_alarm_multiplier: f64,
    /// Minimum absolute boundary hit rate required to flag a widening warning.
    #[serde(default = "default_bounds_alarm_floor")]
    pub bounds_alarm_floor: f64,
    /// Output file path for parameter boundary reports.
    #[serde(default = "default_bounds_report_path")]
    pub bounds_report_path: String,
}

#[derive(Debug, Deserialize, Clone)]
#[serde(deny_unknown_fields)]
pub struct GeneralConfig {
    pub checkpoint_interval: usize,
    pub log_level: String,
    pub log_file: String,
    pub tensorboard_dir: String,
}

impl TunerConfig {
    /// Parses a configuration file from TOML.
    ///
    /// Resolves relative filter paths relative to the configuration file's directory.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
        let contents = fs::read_to_string(path).map_err(|e| {
            eprintln!("{ALARM_PEN}[!] Failed to read config file '{path}': {e}{RESET}");
            e
        })?;

        let mut config: Self = toml::from_str(&contents).map_err(|e| {
            eprintln!("{ALARM_PEN}[!] Failed to parse TOML from '{path}': {e}{RESET}");
            e
        })?;

        if let (Some(dir), Some(filter)) = (Path::new(path).parent(), config.evaltune.replay_filter.as_ref())
            && Path::new(filter).is_relative()
        {
            config.evaltune.replay_filter = Some(dir.join(filter).to_string_lossy().into_owned());
        }

        Ok(config)
    }
}

impl Default for TunerConfig {
    fn default() -> Self {
        Self {
            evaltune: EvalTuneConfig {
                lr_schedule: LrScheduleConfig::Cosine { base: 0.1, min: 0.0001, warmup_ratio: 0.1, cycles: 1 },
                wdl_schedule: WdlScheduleConfig::Constant { value: 0.3 },
                beta1: 0.9,
                beta2: 0.99,
                weight_decay: 0.00001,
                loss: LossFn::CrossEntropy,
                batch_size: 65536,
                epochs: 4000,
                shuffle_block: 0,
                log_path: default_log_path(),
                grad_clip: 1.0,
                k_min: 0.003,
                k_max: 0.010,
                k_mode: KMode::Sweep { interval: DEFAULT_K_SWEEP_INTERVAL },
                patience: 100,
                ema_decay: 0.999,
                unfreeze_epoch: 0,
                seed: None,
                split_seed: None,
                init: Init::Default,
                gate_census: false,
                auto_freeze: true,
                freeze_start_epoch: 500,
                freeze_cadence: 100,
                freeze_threshold: 1e-7,
                freeze_consecutive: 2,
                replay_filter: None,
                volatility_threshold: 0,
                volatility_adaptive: true,
                phase_balance: false,
                phase_balance_cap: 8.0,
                phase_target: None,
                epoch_sample: None,
                val_max: None,
                lr_psqt: 1.0,
                lr_material: 1.0,
                lr_mobility: 1.0,
                lr_other: 1.0,
            },
            searchtune: SearchTuneConfig {
                population_scale: 2.0,
                sigma_init: 0.25,
                sigma_restart: 0.4,
                min_sigma: 0.02,
                max_restarts: 3,
                stagnation_threshold: 15,
                h2h_pairs: 300,
                validation_pairs: 1000,
                reeval_interval: 8,
                confidence_factor: 2.0,
                smoothing_radius: 0.1,
                epochs: 200,
                pairs: 128,
                tc: Some("4+0.04".to_string()),
                h2h_tc: Some("1.0+0.01".to_string()),
                val_tc: Some("1.0+0.01".to_string()),
                speed_penalty: 115.0,
                centering_penalty: 100.0,
                active_softness: 0.5,
                sigma_boost_factor: 2.0,
                bounds_report_interval: 5,
                bounds_window_gens: 20,
                bounds_alarm_multiplier: 2.0,
                bounds_alarm_floor: 0.02,
                bounds_report_path: "bounds_report.txt".to_string(),
            },
            general: GeneralConfig {
                checkpoint_interval: 50,
                log_level: "info".into(),
                log_file: "tuner.log".into(),
                tensorboard_dir: "runs".into(),
            },
        }
    }
}

const fn default_bounds_report_interval() -> usize { 5 }
const fn default_bounds_window_gens() -> usize { 20 }
const fn default_bounds_alarm_multiplier() -> f64 { 2.0 }
const fn default_bounds_alarm_floor() -> f64 { 0.02 }
const fn default_smoothing_radius() -> f64 { 0.1 }
const fn default_patience() -> usize { 100 }
const fn default_ema_decay() -> f64 { 0.999 }
const fn default_true() -> bool { true }
const fn default_freeze_start() -> usize { 500 }
const fn default_freeze_cadence() -> usize { 100 }
const fn default_freeze_threshold() -> f64 { 1e-7 }
const fn default_freeze_consecutive() -> usize { 2 }
const fn default_one() -> f64 { 1.0 }
const fn default_phase_balance_cap() -> f64 { 8.0 }
const fn default_k_sweep_interval() -> usize { DEFAULT_K_SWEEP_INTERVAL }
const fn default_k_lr_mult() -> f64 { DEFAULT_K_LR_MULT }

fn default_bounds_report_path() -> String { "bounds_report.txt".to_string() }
fn default_log_path() -> String { "evaltune.jsonl".into() }

#[inline(always)]
fn set<T>(field: &mut T, from: Option<T>) {
    if let Some(v) = from {
        *field = v;
    }
}

#[cfg(test)]
mod tests {
    use super::{LossFn, TunerConfig};

    fn parse_loss(spec: &str) -> Result<LossFn, toml::de::Error> {
        #[derive(serde::Deserialize)]
        struct Wrap {
            loss: LossFn,
        }
        toml::from_str::<Wrap>(&format!("loss = {spec}")).map(|w| w.loss)
    }

    fn sigmoid(score: f64, k: f64) -> f64 { 1.0 / (1.0 + (-k * score).exp()) }

    /// Anchors the derivative chain to the loss itself. Without it the Hessian test
    /// compares one derivative against another, and a wrong `grad_scale` would be
    /// confirmed by a Hessian consistent with it.
    #[test]
    fn grad_scale_matches_finite_difference_of_loss() {
        let losses = [
            LossFn::CrossEntropy,
            LossFn::MeanSquaredError,
            LossFn::Focal { gamma: 0.5 },
            LossFn::Focal { gamma: 1.5 },
            LossFn::Focal { gamma: 2.0 },
            LossFn::SmoothedCE { epsilon: 0.1 },
        ];

        for loss in losses {
            for &k in &[0.002, 0.005] {
                for &s in &[-400.0, 0.0, 400.0] {
                    for &t in &[0.0, 0.3, 1.0] {
                        // Focal's |S − T| factor is not differentiable where the two coincide.
                        if (sigmoid(s, k) - t).abs() < 0.05 {
                            continue;
                        }

                        let delta = 1e-3;
                        let g = loss.grad_scale(sigmoid(s, k), t, k);
                        let fd = (loss.loss(sigmoid(s + delta, k), t) - loss.loss(sigmoid(s - delta, k), t)) / (2.0 * delta);
                        let tol = 1e-4 * g.abs().max(1e-9) + 1e-12;
                        assert!((g - fd).abs() <= tol, "{loss:?} at s={s} t={t} k={k}: closed {g:.6e} vs fd {fd:.6e}");
                    }
                }
            }
        }
    }

    #[test]
    fn hessian_matches_finite_difference_of_gradient() {
        let losses = [
            LossFn::CrossEntropy,
            LossFn::MeanSquaredError,
            LossFn::Focal { gamma: 0.5 },
            LossFn::Focal { gamma: 1.5 },
            LossFn::Focal { gamma: 2.0 },
            LossFn::SmoothedCE { epsilon: 0.1 },
        ];

        for loss in losses {
            for &k in &[0.002, 0.005] {
                for &s in &[-400.0, 0.0, 400.0] {
                    for &t in &[0.0, 0.3, 1.0] {
                        let sig = sigmoid(s, k);

                        // Skip the singular/discontinuous point for Focal loss where gamma < 2.
                        if (sig - t).abs() < 0.05 {
                            continue;
                        }

                        let delta = 1e-3;
                        let h = loss.hessian_scale(sig, t, k);
                        let fd = (loss.grad_scale(sigmoid(s + delta, k), t, k) - loss.grad_scale(sigmoid(s - delta, k), t, k))
                            / (2.0 * delta);

                        let tol = 1e-4 * h.abs().max(1e-9) + 1e-12;
                        assert!((h - fd).abs() <= tol, "{loss:?} at s={s} t={t} k={k}: closed {h:.6e} vs fd {fd:.6e}");
                    }
                }
            }
        }
    }

    #[test]
    fn parses_all_loss_variants() {
        assert_eq!(parse_loss("\"ce\"").unwrap(), LossFn::CrossEntropy);
        assert_eq!(parse_loss("\"mse\"").unwrap(), LossFn::MeanSquaredError);
        assert_eq!(parse_loss("\"focal\"").unwrap(), LossFn::Focal { gamma: 2.0 });
        assert_eq!(parse_loss("\"sce\"").unwrap(), LossFn::SmoothedCE { epsilon: 0.01 });
        assert_eq!(parse_loss("{ gamma = 2.5 }").unwrap(), LossFn::Focal { gamma: 2.5 });
        assert_eq!(parse_loss("{ epsilon = 0.02 }").unwrap(), LossFn::SmoothedCE { epsilon: 0.02 });
    }

    #[test]
    fn rejects_malformed_loss_specifications() {
        for (doc, wanted) in [
            ("\"crossentropy\"", "crossentropy"),
            ("{ delta = 1.0 }", "delta"),
            ("{ gamma = 2.0, epsilon = 0.01 }", "epsilon"),
        ] {
            let error = parse_loss(doc).expect_err("{doc} must not parse");
            assert!(error.to_string().contains(wanted), "error must name '{wanted}': {error}");
        }
    }

    #[test]
    fn loads_default_config_file() { TunerConfig::from_file("evaltune.toml").expect("evaltune.toml must parse"); }

    #[test]
    fn rejects_unknown_configuration_keys() {
        let doc = format!("log_levle = \"info\"\n{}", std::fs::read_to_string("evaltune.toml").unwrap());
        let error = toml::from_str::<TunerConfig>(&doc).expect_err("unknown key must fail deserialization");
        assert!(error.to_string().contains("log_levle"), "error must identify misnamed key: {error}");
    }

    #[test]
    fn grad_target_matches_finite_difference_of_loss() {
        let losses = [LossFn::CrossEntropy, LossFn::MeanSquaredError, LossFn::Focal { gamma: 1.5 }, LossFn::SmoothedCE {
            epsilon: 0.01,
        }];

        for loss in losses {
            for (sig, target) in [(0.3, 0.2), (0.5, 0.5), (0.8, 0.9), (0.1, 0.9), (0.9999, 0.5)] {
                let h = 1e-7;
                let numeric = (loss.loss(sig, target + h) - loss.loss(sig, target - h)) / (2.0 * h);
                let closed = loss.grad_target(sig, target);
                assert!((numeric - closed).abs() < 1e-6, "{loss:?} at ({sig}, {target}): {closed} vs fd {numeric}");
            }
        }
    }
}
