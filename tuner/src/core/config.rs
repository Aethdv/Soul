//! Global configuration structures for the tuning pipeline.

use std::fs;

use serde::Deserialize;

use crate::core::schedule::{self, LrScheduler, WdlScheduler};

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
}

#[derive(Debug, Deserialize, Clone)]
#[serde(tag = "type", rename_all = "kebab-case")]
pub enum WdlScheduleConfig {
    Constant {
        value: f64,
    },
    Linear {
        start: f64,
        end: f64,
    },
    Cosine {
        start: f64,
        end: f64,
    },
    StableDecay {
        start: f64,
        end: f64,
        stable_ratio: f64,
    },
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
}

#[derive(Debug, Deserialize, Clone)]
pub struct TunerConfig {
    pub evaltune: EvalTuneConfig,
    pub searchtune: SearchTuneConfig,
    pub general: GeneralConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct EvalTuneConfig {
    pub lr_schedule: LrScheduleConfig,
    pub wdl_schedule: WdlScheduleConfig,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    /// Empirically, 262,144 works well with ~8M positions; smaller datasets benefit from smaller
    /// batches (e.g., 8192 for ~2M). Gradient stability is highly batch-size dependent.
    pub batch_size: usize,
    pub epochs: usize,
    pub grad_clip: f64,
    pub k_min: f64,
    pub k_max: f64,
    /// Plateau patience epochs before halving lr_scale. Default: 100.
    #[serde(default = "default_patience")]
    pub patience: usize,
    /// Decay rate for Polyak averaging (Chronological EMA). Default: 0.999.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    /// Epoch at which to unfreeze non-material parameters (mobility, king safety, etc.).
    /// During epochs 0..unfreeze_epoch only PSQT and material values train; the rest
    /// are held at their initial values. After unfreeze_epoch, all trainable parameters
    /// participate normally. 0 disables progressive unfreeze. Default: 0.
    #[serde(default)]
    pub unfreeze_epoch: usize,
    /// Fixed RNG seed for deterministic batch ordering. When None (default), the seed
    /// is randomly generated at startup. Set to any u64 for reproducible training runs.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Magma temperature for Lion sign-update scaling.
    /// 0.0 = disabled. 0.05–0.3 effective range.
    /// Controls sigmoid(cossim(momentum, gradient) / tau) gate sharpness.
    #[serde(default)]
    pub magma_tau: f64,
    /// Enable auto-freeze of stagnant parameters. Default: true.
    #[serde(default = "default_true")]
    pub auto_freeze: bool,
    /// Delay auto-freeze activation until this epoch. Default: 500.
    #[serde(default = "default_freeze_start")]
    pub freeze_start_epoch: usize,
    /// Check for stagnant parameters every N epochs. Default: 100.
    #[serde(default = "default_freeze_cadence")]
    pub freeze_cadence: usize,
    /// Grad EMA below this value is considered stagnant. Default: 1e-7.
    #[serde(default = "default_freeze_threshold")]
    pub freeze_threshold: f64,
    /// Number of consecutive checks before freezing. Default: 2.
    #[serde(default = "default_freeze_consecutive")]
    pub freeze_consecutive: usize,
}

fn default_patience() -> usize {
    100
}
fn default_ema_decay() -> f64 {
    0.999
}
fn default_true() -> bool {
    true
}
fn default_freeze_start() -> usize {
    500
}
fn default_freeze_cadence() -> usize {
    100
}
fn default_freeze_threshold() -> f64 {
    1e-7
}
fn default_freeze_consecutive() -> usize {
    2
}

#[derive(Debug, Deserialize, Clone)]
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
}

fn default_smoothing_radius() -> f64 {
    0.1
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeneralConfig {
    pub checkpoint_interval: usize,
    pub log_level: String,
    pub log_file: String,
    pub tensorboard_dir: String,
}

impl TunerConfig {
    /// # Errors
    /// if file verification fails or TOML parsing fails.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let contents = fs::read_to_string(path).map_err(|e| {
            eprintln!("\x1b[31m[!] Failed to read config file '{}': {}\x1b[0m", path, e);
            e
        })?;
        let config: Self = toml::from_str(&contents).map_err(|e| {
            eprintln!("\x1b[31m[!] Failed to parse TOML from '{}': {}\x1b[0m", path, e);
            e
        })?;
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
                batch_size: 32768,
                epochs: 8000,
                grad_clip: 1.0,
                k_min: 0.003,
                k_max: 0.010,
                patience: 100,
                ema_decay: 0.999,
                unfreeze_epoch: 0,
                seed: None,
                magma_tau: 0.0,
                auto_freeze: true,
                freeze_start_epoch: 500,
                freeze_cadence: 100,
                freeze_threshold: 1e-7,
                freeze_consecutive: 2,
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
