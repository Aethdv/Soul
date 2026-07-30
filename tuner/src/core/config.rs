//! Global configuration structures for the tuning pipeline.

use std::{error::Error, fs};

use serde::{
    Deserialize,
    de::{self, MapAccess, Visitor},
};

use crate::core::schedule::{self, LrScheduler, WdlScheduler};

pub const DEFAULT_WDL_END: f64 = 0.3;
pub const DEFAULT_K_LR_MULT: f64 = 0.01;
pub const DEFAULT_K_SWEEP_INTERVAL: usize = 200;

/// Half-width of the uniform draw for [`Init::Random`]. Small against every
/// weight that matters, large enough that two seeds start meaningfully apart.
pub const RANDOM_INIT_SPREAD: f64 = 16.0;

/// `Zero` whether the data determines a good eval at all.
/// `Random` whether two seeds arrive at the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Init {
    #[default]
    Default,
    Zero,
    Random,
}

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
        f.write_str("\"ce\", \"mse\", \"focal\", \"sce\", or a map { gamma/epsilon: f64 }")
    }

    fn visit_str<E: de::Error>(self, value: &str) -> Result<LossFn, E> {
        match value {
            "ce" => Ok(LossFn::CrossEntropy),
            "mse" => Ok(LossFn::MeanSquaredError),
            "focal" => Ok(LossFn::Focal { gamma: 2.0 }),
            "sce" => Ok(LossFn::SmoothedCE { epsilon: 0.01 }),
            _ => Err(de::Error::unknown_variant(value, &["ce", "mse", "focal", "sce"])),
        }
    }

    /// The parameter names the variant, so two of them name two losses.
    fn visit_map<M: MapAccess<'de>>(self, mut map: M) -> Result<LossFn, M::Error> {
        let loss = match map.next_entry::<String, f64>()? {
            Some((key, value)) if key == "gamma" => LossFn::Focal { gamma: value },
            Some((key, value)) if key == "epsilon" => LossFn::SmoothedCE { epsilon: value },
            Some((key, _)) => return Err(de::Error::unknown_field(&key, &["gamma", "epsilon"])),
            None => return Err(de::Error::custom("expected 'gamma' (Focal) or 'epsilon' (SmoothedCE)")),
        };

        if let Some((extra, _)) = map.next_entry::<String, f64>()? {
            return Err(de::Error::custom(format!("{extra} names a second loss; give one of 'gamma' or 'epsilon'")));
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

impl LossFn {
    /// Returns `L(sig, target)` for the configured loss function.
    pub fn loss(self, sig: f64, target: f64) -> f64 {
        match self {
            // L = (S − target)²
            Self::MeanSquaredError => {
                let err = sig - target;
                err * err
            },
            // L = −target·ln(S) − (1−target)·ln(1−S)
            Self::CrossEntropy => {
                let s = sig.clamp(1e-7, 1.0 - 1e-7);
                -(target * s.ln() + (1.0 - target) * (1.0 - s).ln())
            },
            // FL = |s-T|^γ · CE
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let ce = Self::CrossEntropy.loss(sig, target);
                let base = (prob - target).abs();
                base.powf(gamma) * ce
            },
            // SCE = CE(s, T·(1-ε) + 0.5·ε)
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.loss(sig, t)
            },
        }
    }

    /// Outer derivative ∂L/∂score. `k` is the sigmoid scaling constant.
    pub fn grad_scale(self, sig: f64, target: f64, k: f64) -> f64 {
        let err = sig - target;
        match self {
            // dJ/dx = 2·(S − target)·K·S·(1 − S)
            Self::MeanSquaredError => 2.0 * err * sig * (1.0 - sig) * k,
            // dL/dx = (S − target)·K
            Self::CrossEntropy => err * k,
            // dFL/dx = |s-T|^γ·(s-T)·K + γ·|s-T|^(γ-1)·sign(s-T)·K·s·(1-s)·CE
            Self::Focal { gamma } => {
                let prob = sig.clamp(1e-7, 1.0 - 1e-7);
                let ce = Self::CrossEntropy.loss(sig, target);
                let diff = prob - target;
                let base = diff.abs().max(1e-12);
                let ce_grad = base.powf(gamma) * diff * k;
                let focal_grad = gamma * base.powf(gamma - 1.0) * diff.signum() * k * prob * (1.0 - prob) * ce;

                ce_grad + focal_grad
            },
            // ∂SCE/∂x = CE_grad(s, T·(1-ε) + 0.5·ε)
            Self::SmoothedCE { epsilon } => {
                let t = target * (1.0 - epsilon) + 0.5 * epsilon;
                Self::CrossEntropy.grad_scale(sig, t, k)
            },
        }
    }
}

#[derive(Debug, Deserialize, Clone, Copy, PartialEq)]
#[serde(tag = "type", rename_all = "kebab-case", deny_unknown_fields)]
pub enum KMode {
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
    fn default() -> Self {
        Self::Sweep { interval: DEFAULT_K_SWEEP_INTERVAL }
    }
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

    /// Apply CLI overrides to the schedule's numeric fields without changing its type.
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

    /// Extract default start/end values regardless of variant.
    pub fn defaults(&self) -> (f64, f64) {
        match self {
            Self::Cosine { start, end } | Self::Linear { start, end } | Self::StableDecay { start, end, .. } => (*start, *end),
            Self::Constant { value } => (*value, DEFAULT_WDL_END),
        }
    }

    /// Apply CLI overrides: `blend` replaces the entire schedule with a constant;
    /// `start`/`end` update the current type's fields in place.
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
    pub lr_schedule: LrScheduleConfig,
    pub wdl_schedule: WdlScheduleConfig,
    pub beta1: f64,
    pub beta2: f64,
    pub weight_decay: f64,
    /// Default is MSE.
    #[serde(default)]
    pub loss: LossFn,
    /// Smaller batches = more updates/epoch = better generalization
    /// at the cost of slower per-epoch wall time.
    /// 32k-131k is the sweet spot for 2-15M positions.
    /// Handwavy; Below 50: overfit. Above 300: diminishing returns.
    pub batch_size: usize,
    pub epochs: usize,
    /// Records per shuffled block, or 0 for a full permutation. Blocks trade the
    /// fresh partition an epoch's reshuffle buys for sequential reads.
    #[serde(default)]
    pub shuffle_block: usize,
    /// Where the run appends its JSONL. Two runs pointed at one path land in one
    /// file, and whatever reads it then has to choose between them.
    #[serde(default = "default_log_path")]
    pub log_path: String,
    pub grad_clip: f64,
    pub k_min: f64,
    pub k_max: f64,
    #[serde(default)]
    pub k_mode: KMode,
    /// Plateau patience epochs before halving lr_scale. Default: 100.
    #[serde(default = "default_patience")]
    pub patience: usize,
    /// Decay rate for Polyak averaging (Chronological EMA). Default: 0.999.
    ///
    /// Applied once per batch, so the window is 1000 updates rather than 1000 epochs: about ten
    /// epochs of a 7.1M-position set at batch 65536, about two of a 32.8M one. Updates are the
    /// right unit, since one Lion step displaces `eff_lr` whatever the epoch length, but it does
    /// mean the tail average that decides what ships covers fewer epochs as a dataset grows.
    #[serde(default = "default_ema_decay")]
    pub ema_decay: f64,
    /// Epoch at which to unfreeze non-material parameters (mobility, king safety, etc.).
    /// During epochs 0..unfreeze_epoch only PSQT and material values train; the rest
    /// are held at their initial values. After unfreeze_epoch, all trainable parameters
    /// participate normally. 0 disables progressive unfreeze. Default: 0.
    #[serde(default)]
    pub unfreeze_epoch: usize,
    /// Per-group LR multipliers for Lion sign-step scaling.
    #[serde(default = "default_one")]
    pub lr_psqt: f64,
    #[serde(default = "default_lr_material")]
    pub lr_material: f64,
    #[serde(default = "default_lr_mobility")]
    pub lr_mobility: f64,
    #[serde(default = "default_one")]
    pub lr_other: f64,
    /// Fixed RNG seed for deterministic batch ordering. When None (default), the seed
    /// is randomly generated at startup. Set to any u64 for reproducible training runs.
    #[serde(default)]
    pub seed: Option<u64>,
    /// Seed for the shuffle that carves out the validation slice. When None (default), a fixed
    /// seed holds out the same positions in every run, which is what makes two runs' validation
    /// losses comparable. Set it to measure how much a result owes to which tenth got held out.
    #[serde(default)]
    pub split_seed: Option<u64>,
    #[serde(default)]
    pub init: Init,
    /// Report how Lion's disagreement gate votes, per epoch and per parameter group. Costs a
    /// pass over the parameters per batch, so it is off outside a diagnostic run. Default: false.
    #[serde(default)]
    pub gate_census: bool,
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
    /// Path to a viriformat filter file, gating which replayed positions load.
    /// Absent loads every position, so changing this moves the dataset
    /// fingerprint and a checkpoint from another setting will not resume.
    #[serde(default)]
    pub replay_filter: Option<String>,
    /// Volatility filter threshold in centipawns. 0 = disabled.
    /// Positions where |static_eval - search_eval| exceeds this are skipped.
    #[serde(default)]
    pub volatility_threshold: i16,
    /// Scale the threshold with piece count (higher in complex positions).
    #[serde(default = "default_true")]
    pub volatility_adaptive: bool,
    /// Phase-stratified gradient balancing: reweight each training sample by the
    /// inverse frequency of its game-phase bucket, so endgame params aren't
    /// drowned by midgame-heavy data. Off by default.
    #[serde(default)]
    pub phase_balance: bool,
    /// Cap on the per-sample phase-balancing weight (clamped to `[1/cap, cap]`).
    #[serde(default = "default_phase_balance_cap")]
    pub phase_balance_cap: f64,
    /// Target phase distribution to reweight toward: one density per phase
    /// (`0..=TOTAL_PHASE`, 25 values; need not sum to 1). The per-sample weight
    /// becomes `target[phase] / observed[phase]`. None = uniform (inverse
    /// frequency, the default). Applies only when `phase_balance` is on.
    #[serde(default)]
    pub phase_target: Option<Vec<f64>>,
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

    /// Generations between bounds-report snapshots (also written on SIGINT).
    #[serde(default = "default_bounds_report_interval")]
    pub bounds_report_interval: usize,
    /// Sliding-window size (generations) for clamp-rate detection.
    #[serde(default = "default_bounds_window_gens")]
    pub bounds_window_gens: usize,
    /// Observed clamp rate must exceed `expected_rate * multiplier + floor` to flag a widen.
    #[serde(default = "default_bounds_alarm_multiplier")]
    pub bounds_alarm_multiplier: f64,
    /// Absolute floor for alarm, prevents flagging when expected rate ≈ 0.
    #[serde(default = "default_bounds_alarm_floor")]
    pub bounds_alarm_floor: f64,
    /// File path for the bounds report, relative to CWD.
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
    /// # Errors
    /// if file verification fails or TOML parsing fails.
    pub fn from_file(path: &str) -> Result<Self, Box<dyn Error>> {
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

/// Overwrite a schedule field only where the CLI supplied one,
/// so a flag the caller left off keeps whatever the config file set.
fn set<T>(field: &mut T, from: Option<T>) {
    if let Some(v) = from {
        *field = v;
    }
}

fn default_bounds_report_interval() -> usize {
    5
}
fn default_bounds_window_gens() -> usize {
    20
}
fn default_bounds_alarm_multiplier() -> f64 {
    2.0
}
fn default_bounds_alarm_floor() -> f64 {
    0.02
}
fn default_bounds_report_path() -> String {
    "bounds_report.txt".to_string()
}
fn default_smoothing_radius() -> f64 {
    0.1
}

fn default_log_path() -> String {
    "evaltune.jsonl".into()
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
fn default_one() -> f64 {
    1.0
}
fn default_lr_material() -> f64 {
    0.3
}
fn default_lr_mobility() -> f64 {
    0.5
}
fn default_phase_balance_cap() -> f64 {
    8.0
}

fn default_k_sweep_interval() -> usize {
    DEFAULT_K_SWEEP_INTERVAL
}

fn default_k_lr_mult() -> f64 {
    DEFAULT_K_LR_MULT
}

#[cfg(test)]
mod tests {
    use super::{LossFn, TunerConfig};

    /// A bare scalar is not a TOML document, so the spec goes under a key or it never
    /// reaches the visitor.
    fn loss(spec: &str) -> Result<LossFn, toml::de::Error> {
        #[derive(serde::Deserialize)]
        struct Wrap {
            loss: LossFn,
        }

        toml::from_str::<Wrap>(&format!("loss = {spec}")).map(|w| w.loss)
    }

    #[test]
    fn every_loss_spelling_parses_to_its_variant() {
        assert_eq!(loss("\"ce\"").unwrap(), LossFn::CrossEntropy);
        assert_eq!(loss("\"mse\"").unwrap(), LossFn::MeanSquaredError);
        assert_eq!(loss("\"focal\"").unwrap(), LossFn::Focal { gamma: 2.0 });
        assert_eq!(loss("\"sce\"").unwrap(), LossFn::SmoothedCE { epsilon: 0.01 });
        assert_eq!(loss("{ gamma = 2.5 }").unwrap(), LossFn::Focal { gamma: 2.5 });
        assert_eq!(loss("{ epsilon = 0.02 }").unwrap(), LossFn::SmoothedCE { epsilon: 0.02 });
    }

    #[test]
    fn an_ambiguous_or_unknown_loss_is_refused() {
        for (doc, wanted) in [
            ("\"crossentropy\"", "crossentropy"),
            ("{ delta = 1.0 }", "delta"),
            ("{ gamma = 2.0, epsilon = 0.01 }", "epsilon"),
        ] {
            let error = loss(doc).expect_err("{doc} must not parse");
            assert!(error.to_string().contains(wanted), "the error must name {wanted}: {error}");
        }
    }

    #[test]
    fn the_shipped_config_parses() {
        TunerConfig::from_file("tuner_config.toml").expect("tuner_config.toml must parse");
    }

    #[test]
    fn a_misspelled_key_is_refused() {
        // The key must go in the root table. Adding a section instead redeclares
        // a table, which TOML refuses on its own, and this passes with the
        // attribute gone.
        let doc = format!("log_levle = \"info\"\n{}", std::fs::read_to_string("tuner_config.toml").unwrap());
        let error = toml::from_str::<TunerConfig>(&doc).expect_err("an unknown key must fail the load");

        assert!(error.to_string().contains("log_levle"), "the error must name the offending key: {error}");
    }
}
