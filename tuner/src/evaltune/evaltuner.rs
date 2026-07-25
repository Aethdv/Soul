//! HCE training: gradient descent on Soul's weights against WDL-labeled positions.
//!
//! Features are extracted to `FeatureRecord`s once at startup; epochs are sequential
//! reads over the cached SoA path (`eval_record` + `accumulate_record_grad`, not the
//! dual-number oracle the tuner tests use). An initial golden-section search fixes
//! the sigmoid's centipawn→winrate scale; `k_ref` stays frozen so the reference loss
//! is comparable across runs.
//!
//! Targeted best-loss captures track the validation split; separate records
//! let you inspect both the best-validating and best-training parameters.

use std::{
    fs,
    io::{self, BufWriter, Write},
    path,
    time::Instant,
};

use palette::{CLEAR_LINE, RESET};
use rayon::prelude::*;
use soul::{
    color,
    core::{defs::TOTAL_PHASE, psqt},
    engine::eval_params::{self, Tunable},
    tools::dataset::FeatureRecord,
};

use super::{lion::Lion, loader, palette, report::*, storage::*, training::*};
use crate::core::{
    config::{EvalTuneConfig, KMode, LossFn, LrScheduleConfig},
    fnv::Fnv1a,
    logger::JsonLogger,
};

/// Hard clamp for mobility parameters to prevent drift from unbounded features.
const MOB_CLAMP: f64 = 100.0;

// EMA spans in epochs. Their difference is the trend; the slow span also gates warmup,
// since a trend read before that span has filled is reading its own seed.
const TREND_FAST: usize = 10;
const TREND_SLOW: usize = 40;

const A_FAST: f64 = 2.0 / (TREND_FAST as f64 + 1.0);
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

/// State machine for K.
struct KController {
    k: f64,
    k_ref: f64,
    mode: KMode,
    k_min: f64,
    k_max: f64,
    beta1: f64,
    beta2: f64,
    momentum: f64,
}

impl KController {
    fn bootstrap(
        config: &EvalTuneConfig,
        ctx: &TrainerContext,
        values: &[f64],
        init_blend: f64,
        resume: Option<&CheckpointData>,
    ) -> Self {
        let (k, k_ref, k_momentum) = match resume {
            Some(d) => (d.k, d.k_ref, d.k_momentum),
            None => match config.k_mode {
                KMode::Fixed { value } => (value, value, 0.0),
                _ => {
                    println!("Optimizing K...");

                    let k = golden_search_k(config.k_min, config.k_max, 1e-6 * (config.k_max - config.k_min), |kk| {
                        ctx.val_eval(values, kk, init_blend)
                    });

                    (k, k, 0.0)
                },
            },
        };

        Self {
            k,
            k_ref,
            mode: config.k_mode,
            k_min: config.k_min,
            k_max: config.k_max,
            beta1: config.beta1,
            beta2: config.beta2,
            momentum: k_momentum,
        }
    }

    fn k(&self) -> f64 {
        self.k
    }
    fn k_ref(&self) -> f64 {
        self.k_ref
    }

    fn on_epoch(&mut self, epoch: usize, ctx: &TrainerContext, ema_values: &[f64], blend: f64) -> Option<f64> {
        let KMode::Sweep { interval } = self.mode else { return None };
        if !epoch.is_multiple_of(interval.max(1)) {
            return None;
        }

        self.k =
            golden_search_k(self.k_min, self.k_max, 1e-6 * (self.k_max - self.k_min), |kk| ctx.val_eval(ema_values, kk, blend));

        Some(self.k)
    }

    fn on_batch(&mut self, k_grad: f64, batch_count: usize, lr: f64, scale: f64, weight_decay: f64) {
        let KMode::Learned { lr_mult } = self.mode else { return };

        let n = batch_count.max(1) as f64;
        let kg = k_grad / n * scale;
        let eff_lr = lr * lr_mult;
        let c = self.beta1.mul_add(self.momentum, (1.0 - self.beta1) * kg);

        self.k -= eff_lr * (c.signum() + weight_decay * self.k);
        self.momentum = self.beta2.mul_add(self.momentum, (1.0 - self.beta2) * kg);
        self.k = self.k.clamp(self.k_min, self.k_max);
    }
}

/// Overfitting detector: fit still improving while generalization degrades.
///
/// Neither loss is compared to its own running minimum. A running minimum over a noisy series
/// settles at the deepest trough it has seen and never recovers, so it sits below the true mean
/// by roughly the noise amplitude and every ordinary epoch afterward reads as a regression
/// against it. A trend carries no such bias, and it needs no special case at an LR restart:
/// a restart lifts both losses at once, and divergence needs train falling. Clearing the
/// trails there would only blind the detector for a slow span, so nothing clears them.
struct DivergenceMonitor {
    train_fast: f64,
    train_slow: f64,
    val_fast: f64,
    val_slow: f64,
    noise: f64,
    prev_val: f64,
    seen: usize,
}

impl DivergenceMonitor {
    const fn new() -> Self {
        Self { train_fast: 0.0, train_slow: 0.0, val_fast: 0.0, val_slow: 0.0, noise: 0.0, prev_val: 0.0, seen: 0 }
    }

    /// Feeds one epoch, reporting whether the run is diverging.
    fn update(&mut self, train_loss: f64, val_loss: f64) -> bool {
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

pub fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>) -> f64 {
    let total_start = Instant::now();

    // Enable FTZ/DAZ on the MAIN thread immediately.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        // SAFETY: MXCSR manipulation has no memory precondition; the call is always sound.
        enable_ftz_daz();
    }

    // Configure the Rayon thread pool (catches new worker threads).
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            // SAFETY: MXCSR manipulation has no memory precondition; the call is always sound.
            #[cfg(target_arch = "x86_64")]
            enable_ftz_daz();
        })
        .build_global()
        .ok();

    let effective_dataset: String = match (dataset_path, resume_path) {
        (Some(p), _) => p.to_string(),
        (None, Some(rp)) => peek_checkpoint(rp)
            .ok()
            .map(|cp| cp.dataset_path)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "default".to_string()),
        (None, None) => "default".to_string(),
    };

    let paths = resolve_dataset_paths(&effective_dataset);
    let Some(paths) = paths else {
        eprintln!(
            "{}[!] Error: No dataset found. Use --dataset <path> or place .soul.zst files in data/.{RESET}",
            color::ansi_fg((225, 89, 91)),
        );

        return f64::MAX;
    };

    let all_entries = loader::load_datasets(&paths);

    if all_entries.is_empty() {
        eprintln!("Error: No positions loaded.");
        return f64::MAX;
    }

    let dataset_label = paths.join(", ");
    let best_val = train_entries(all_entries, &dataset_label, config, resume_path);

    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n{}Done in {elapsed:.2}s{RESET}", palette::fg(palette::BRAND));
    best_val
}

/// Enable Flush-to-Zero and Denormals-are-Zero for performance.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn enable_ftz_daz() {
    use std::arch::asm;
    let mut mxcsr: u32 = 0;
    // SAFETY: stmxcsr/ldmxcsr read and write the 4-byte MXCSR to/from mxcsr, a valid
    // aligned local u32; the ops only toggle FP denormal handling, never memory.
    unsafe {
        asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        mxcsr |= 0x8040; // FTZ | DAZ
        asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack, preserves_flags));
    }
}

struct TrainerContext<'a> {
    train: &'a [loader::SoulEntry],
    val: &'a [loader::SoulEntry],
    records: &'a [FeatureRecord],
    train_count: usize,
    phase_weights: &'a [f64],
    loss_fn: LossFn,
    vol_threshold: i16,
    vol_adaptive: bool,
}

impl TrainerContext<'_> {
    fn passes_vol_filter(&self, entry: &loader::SoulEntry, static_eval: i16) -> bool {
        if self.vol_threshold == 0 || entry.score == i16::MAX {
            return true;
        }

        let t = if self.vol_adaptive {
            let short = 10i16.saturating_sub(entry.occupancy.count_ones() as i16);
            self.vol_threshold + short.saturating_mul(2)
        } else {
            self.vol_threshold
        };

        (i32::from(static_eval) - i32::from(entry.score)).abs() <= t as i32
    }

    fn batch_grad(&self, batch_indices: &[usize], values: &[f64], k: f64, blend: f64) -> (Vec<f64>, f64, f64, usize) {
        batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(mut g, mut k_g, mut loss, mut count), chunk| {
                    for &i in chunk {
                        let entry = &self.train[i];
                        let record = &self.records[i];

                        if !self.passes_vol_filter(entry, record.static_eval) {
                            continue;
                        }

                        let target = wdl_target(entry, k, blend);
                        let score = loader::eval_record(record, values);
                        let sig = sigmoid(score, k);
                        let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[i] };
                        let gs = self.loss_fn.grad_scale(sig, target, k);

                        loss += w * self.loss_fn.loss(sig, target);
                        loader::accumulate_record_grad(record, values, gs * w, &mut g);

                        // gs is ∂L/∂score = K · (sig - target) · dσ/dscore.
                        // We need ∂L/∂K = score · (sig - target) · dσ/dscore.
                        // So ∂L/∂K = (gs / K) · score.
                        k_g += (gs / k) * score * w;

                        count += 1;
                    }

                    (g, k_g, loss, count)
                },
            )
            .reduce(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(g1, kg1, l1, c1), (g2, kg2, l2, c2)| {
                    let (g, l) = grad_combine((g1, l1), (g2, l2));
                    (g, kg1 + kg2, l, c1 + c2)
                },
            )
    }

    fn val_eval(&self, values: &[f64], k: f64, blend: f64) -> f64 {
        let (wsum, weight) = self
            .val
            .par_iter()
            .enumerate()
            .fold(
                || (0.0_f64, 0.0_f64),
                |(mut wsum, mut weight), (idx, entry)| {
                    let record = &self.records[self.train_count + idx];

                    if !self.passes_vol_filter(entry, record.static_eval) {
                        return (wsum, weight);
                    }

                    let score = loader::eval_record(record, values);
                    let sig = sigmoid(score, k);
                    let target = wdl_target(entry, k, blend);
                    let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[self.train_count + idx] };

                    wsum += w * self.loss_fn.loss(sig, target);
                    weight += w;

                    (wsum, weight)
                },
            )
            .reduce(|| (0.0_f64, 0.0_f64), |(s1, w1), (s2, w2)| (s1 + s2, w1 + w2));

        if weight > 0.0 { wsum / weight } else { 0.0 }
    }
}

/// Golden-section search for K.
///
/// Maintains two interior probes `a` and `b`; the losing probe becomes the new
/// boundary, requiring only one fresh eval per iteration. The probe offset is
/// `C · range` (not `range`) because the surviving probe already sits at `C²`
/// of the old width, placing the new probe at `C` of the new width.
///
/// Assumes `eval` is unimodal on `[lo, hi]`; otherwise the result is not
/// guaranteed to be a global minimum.
pub fn golden_search_k<F: Fn(f64) -> f64>(lo: f64, hi: f64, tol: f64, eval: F) -> f64 {
    assert!(lo < hi, "golden_search_k: lo ({lo}) must be < hi ({hi})");
    assert!(tol > 0.0, "golden_search_k: tol ({tol}) must be positive");

    if hi - lo <= tol {
        return (lo + hi) / 2.0;
    }

    const C: f64 = 0.618_033_988_749_894_9; // (√5 − 1) / 2

    let mut lo = lo;
    let mut hi = hi;
    let mut width = hi - lo;
    let mut a = hi - C * width;
    let mut b = lo + C * width;
    let mut fa = eval(a);
    let mut fb = eval(b);

    while width > tol {
        width *= C;

        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - C * width;
            fa = eval(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + C * width;
            fb = eval(b);
        }
    }

    (lo + hi) / 2.0
}

fn train_entries(
    mut entries: Vec<loader::SoulEntry>,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
) -> f64 {
    let dataset_fnv = dataset_fingerprint(&entries);

    // A resume must shuffle under the checkpoint's seed, or the train/val
    // split moves and old training positions leak into val.
    let rng_seed = match resume_path {
        Some(path) => {
            let cp = peek_checkpoint(path).unwrap_or_else(|e| {
                eprintln!("Failed to read checkpoint: {e}");
                std::process::exit(1);
            });

            if let Some(s) = config.seed
                && s != cp.rng_seed
            {
                println!("--seed {s} ignored: resume reuses the checkpoint's shuffle seed {}", cp.rng_seed);
            }

            // The seed replays the same shuffle only over the same entries.
            if cp.dataset != dataset_fnv {
                eprintln!(
                    "{}[!] Warning: dataset does not match the checkpoint's fingerprint.\n\
                     [!] The train/val split will differ from the original run: positions the\n\
                     [!] checkpoint trained on may now sit in val, making its loss optimistic.{RESET}",
                    color::ansi_fg((225, 89, 91)),
                );
            }

            cp.rng_seed
        },

        None => config.seed.unwrap_or_else(|| fastrand::u64(..)),
    };

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    rng.shuffle(&mut entries);

    // One-time cost: training reads FeatureRecords straight through.
    // Parallel because entries are independent.
    println!("Extracting features ({} entries)...", entries.len());
    let records: Vec<FeatureRecord> = entries.par_iter().map(FeatureRecord::from_entry).collect();

    let val_count = entries.len() / 10;
    let train_count = entries.len() - val_count;
    let (train, val) = entries.split_at(train_count);

    print_dataset_stats(train, val, entries.len(), |e: &loader::SoulEntry| {
        let stm_white = (e.stm_and_ep & 0x80) == 0;
        let r = f64::from(e.result) / 2.0;
        if stm_white { r } else { 1.0 - r }
    });

    // Weight loss too, not just gradient, or selection fights training.
    let phase_weights = if config.phase_balance {
        build_phase_weights(&records, config.phase_balance_cap, config.phase_target.as_deref())
    } else {
        Vec::new()
    };

    let ctx = TrainerContext {
        train,
        val,
        records: &records,
        train_count,
        phase_weights: &phase_weights,
        loss_fn: config.loss,
        vol_threshold: config.volatility_threshold,
        vol_adaptive: config.volatility_adaptive,
    };

    train_loop(train.len(), "SoulEntry", dataset_label, config, resume_path, rng_seed, dataset_fnv, &ctx)
}

/// Hashed before shuffle: identifies loaded contents, not a permutation.
/// A checkpoint's seed replays the same split only over the same entries.
fn dataset_fingerprint(entries: &[loader::SoulEntry]) -> u64 {
    let mut fnv = Fnv1a::new();
    fnv.write_bytes(&(entries.len() as u64).to_le_bytes());

    let stride = (entries.len() / 1024).max(1);

    for e in entries.iter().step_by(stride) {
        fnv.write_bytes(&e.occupancy.to_le_bytes());
        fnv.write_bytes(&e.score.to_le_bytes());
        fnv.write_bytes(&[e.result, e.stm_and_ep]);
    }

    fnv.digest()
}

/// Reweights toward `target` phase distribution, clamped to `[1/cap, cap]`.
/// `None` is uniform: inverse bucket frequency, lifting sparse phases toward
/// even representation. `Some(t)` is `target[phase] / observed[phase]`, toward
/// the density `t`. Mean-1 keeps gradient scale equal to unweighted.
fn build_phase_weights(records: &[FeatureRecord], cap: f64, target: Option<&[f64]>) -> Vec<f64> {
    let cap = cap.max(1.0);
    let params = eval_params::collect_parameters();
    let woff = psqt::LAYOUT.weight_offset;
    let phase_w: [f64; 6] = std::array::from_fn(|pt| params[woff + pt].value);
    let total = f64::from(TOTAL_PHASE);

    // Phase is fixed (PHASE_WEIGHTS are constant), so a single startup pass suffices.
    let phase_of = |rec: &FeatureRecord| -> usize {
        let raw: f64 = (0..6).map(|pt| f64::from(rec.phase_counts[pt]) * phase_w[pt]).sum();
        raw.clamp(0.0, total).trunc() as usize
    };

    let mut hist = vec![0u64; TOTAL_PHASE as usize + 1];

    for rec in records {
        hist[phase_of(rec)] += 1;
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
            let p = phase_of(rec);
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

/// Set `phase_balance_cap` toward the printed imbalance to fully correct it,
/// or lower to spare the sparse buckets their variance.
fn report_phase_balance(hist: &[u64], weights: &[f64], cap: f64, clamped: usize) {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    let max_pop = hist.iter().copied().max().unwrap_or(0);
    let min_pop = hist.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
    let imbalance = if min_pop > 0 { max_pop as f64 / min_pop as f64 } else { f64::INFINITY };

    let bars: String = hist
        .iter()
        .map(|&c| if c == 0 { ' ' } else { BLOCKS[(((c as f64 / max_pop.max(1) as f64) * 7.0).round() as usize).min(7)] })
        .collect();

    let wmin = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let wmax = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let clamp_pct = 100.0 * clamped as f64 / weights.len().max(1) as f64;

    let lab = palette::fg(palette::LABEL);
    let v = palette::fg(palette::VALUE);

    println!("{lab}Phase balance:{RESET} {v}{bars}{RESET} {lab}(phase 0..{}){RESET}", hist.len() - 1);
    println!(
        "  {lab}imbalance{RESET} {v}{imbalance:.0}×{RESET} {lab}vs cap{RESET} {v}{cap:.0}×{RESET}  \
         {lab}weights{RESET} {v}{wmin:.2}–{wmax:.2}×{RESET}  {lab}clamped{RESET} {v}{clamp_pct:.1}%{RESET}"
    );
}

fn grad_combine((mut g1, l1): (Vec<f64>, f64), (g2, l2): (Vec<f64>, f64)) -> (Vec<f64>, f64) {
    for (a, b) in g1.iter_mut().zip(g2) {
        *a += b;
    }

    (g1, l1 + l2)
}

fn print_dataset_stats<T, F: Fn(&T) -> f64>(train: &[T], val: &[T], total: usize, result_fn: F) {
    let lab = palette::fg(palette::LABEL);
    let c = palette::fg(palette::COUNT);
    println!("{lab}Positions:{RESET}  {c}{total}{RESET} ({} train / {} val)", train.len(), val.len());

    let (ww, bw, dr) = train.iter().fold((0, 0, 0), |(w, b, d), entry| {
        let r = result_fn(entry);

        if (r - 1.0).abs() < 1e-4 {
            (w + 1, b, d)
        } else if r.abs() < 1e-4 {
            (w, b + 1, d)
        } else {
            (w, b, d + 1)
        }
    });

    println!("  {lab}White wins:{RESET} {c}{ww}{RESET}");
    println!("  {lab}Black wins:{RESET} {c}{bw}{RESET}");
    println!("  {lab}Draws:{RESET}      {c}{dr}{RESET}");
}

/// Loss history as a sparkline: lower loss → shorter block.
fn loss_sparkline(history: &[f64]) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    if history.is_empty() {
        return String::new();
    }

    let lo = history.iter().copied().fold(f64::MAX, f64::min);
    let hi = history.iter().copied().fold(f64::MIN, f64::max);
    let span = (hi - lo).max(1e-12);

    let mut out = String::with_capacity(history.len() * 20);

    for &v in history {
        let frac = (v - lo) / span; // 0 = best (lowest), 1 = worst (highest)
        let level = (frac * 8.0).min(7.0) as usize;

        out.push_str(&palette::fg(color::advantage(1.0 - 2.0 * frac)));
        out.push(BLOCKS[level]);
    }

    out.push_str(RESET);
    out
}

fn train_loop(
    train_len: usize,
    mode_label: &str,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    rng_seed: u64,
    dataset_fnv: u64,
    ctx: &TrainerContext,
) -> f64 {
    let all_params = eval_params::collect_parameters();
    let default_values: Vec<f64> = all_params.iter().map(|p| p.value).collect();

    let resume = resume_path.map(|path| {
        println!("Resuming from checkpoint: {}{path}{RESET}", palette::fg(palette::VALUE));
        load_checkpoint(path, &all_params, &default_values).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {e}");
            std::process::exit(1);
        })
    });

    let (start_epoch, mut lr_scale) = resume.as_ref().map_or((1, 1.0), |d| (d.epoch, d.lr_scale));
    let mut values = resume.as_ref().map_or_else(|| default_values.clone(), |d| d.values.clone());
    let mut momentum = resume.as_ref().map_or_else(|| vec![0.0; default_values.len()], |d| d.momentum.clone());

    let is_constant_schedule = matches!(config.lr_schedule, LrScheduleConfig::Constant { .. });
    let lr_scheduler = config.lr_schedule.clone().into_scheduler();
    let wdl_scheduler = config.wdl_schedule.clone().into_scheduler();

    let init_blend = wdl_scheduler.blend(1, config.epochs);
    let mut k_ctrl = KController::bootstrap(config, ctx, &values, init_blend, resume.as_ref());

    let v = palette::fg(palette::VALUE);
    let lab = palette::fg(palette::LABEL);
    let k = k_ctrl.k();
    let win_rate_100cp = sigmoid(100.0, k);

    println!("{lab}K Factor:{RESET}   {v}{k:.6}{RESET} (100cp -> {:.1}%)", win_rate_100cp * 100.0);
    println!("{lab}K Mode:{RESET}     {}", match config.k_mode {
        KMode::Fixed { value } => format!("{v}Fixed{RESET} ({value})"),
        KMode::Learned { lr_mult } => format!("{v}Learned{RESET} ({lr_mult})"),
        KMode::Sweep { interval } => format!("{v}Sweep{RESET} ({interval})"),
    });

    let seed_label = if resume.is_some() {
        " (checkpoint)"
    } else if config.seed.is_some() {
        " (deterministic)"
    } else {
        ""
    };

    println!("{lab}Seed:{RESET}       {v}{rng_seed}{RESET}{seed_label}");

    let initial_values = values.clone();

    // Setup optimizer state and convergence tracking
    let mut fixed_mask: Vec<bool> = all_params.iter().map(|p| p.is_fixed).collect();
    let decay_mask = build_decay_mask(&all_params);
    let beta2_mask = build_beta2_mask(&all_params, config.beta2);
    let lr_mask = build_lr_mask(&all_params, config);

    // Zero init: 0.99 EMA decay washes out any seed within a few batches.
    let mut grad_ema_per_param = resume.as_ref().map_or_else(|| vec![0.0_f64; values.len()], |d| d.grad_ema.clone());
    let mut stagnant_epochs = resume.as_ref().map_or_else(|| vec![0usize; values.len()], |d| d.stagnant.clone());

    println!("{lab}Parameters:{RESET} {v}{}{RESET}", all_params.len());
    println!("{lab}Mode:{RESET}       {v}{mode_label}{RESET}");
    {
        let d = lr_scheduler.describe();
        let d = d.find('(').map_or_else(
            || format!("{v}{d}{RESET}"),
            |op| {
                let name = &d[..op].trim_end();
                let inner = &d[op + 1..d.len() - 1];
                format!("{v}{name}{RESET} ({inner})")
            },
        );
        println!("{lab}LR Sched:{RESET}   {d}");
    }
    {
        let d = wdl_scheduler.describe();
        let d = d.find('(').map_or_else(
            || format!("{v}{d}{RESET}"),
            |op| {
                let name = &d[..op].trim_end();
                let inner = &d[op + 1..d.len() - 1];
                format!("{v}{name}{RESET} ({inner})")
            },
        );
        println!("{lab}WDL Sched:{RESET}  {d}");
    }
    println!("{lab}Optimizer:{RESET}  {v}Lion{RESET} (Batch: {}, WD: {})", config.batch_size, config.weight_decay);

    let log_file = fs::OpenOptions::new().create(true).append(true).open("evaltune_log.txt").ok();
    let mut logger = log_file.map(BufWriter::new);

    if let Some(ref mut w) = logger {
        writeln!(w).ok();
        writeln!(w, "Seed:      {rng_seed}").ok();
        writeln!(w, "Mode:      {mode_label}").ok();
        writeln!(w, "Dataset:   {dataset_label}").ok();
        writeln!(w, "K:         {k:.6} (100cp → {:.1}%)", win_rate_100cp * 100.0).ok();
        writeln!(w, "K mode:    {}", match config.k_mode {
            KMode::Fixed { value } => format!("Fixed ({value})"),
            KMode::Learned { lr_mult } => format!("Learned ({lr_mult})"),
            KMode::Sweep { interval } => format!("Sweep ({interval})"),
        })
        .ok();
        writeln!(w, "Epochs:    {}", config.epochs).ok();
        writeln!(w, "Params:    {}", all_params.len()).ok();
        writeln!(w, "LR:        {}", lr_scheduler.describe()).ok();
        writeln!(w, "WDL:       {}", wdl_scheduler.describe()).ok();
        writeln!(w, "Optimizer: Lion (batch: {}, WD: {})", config.batch_size, config.weight_decay).ok();
        writeln!(w).ok();
        writeln!(w, "{:>5}  {:>11}  {:>11}  {:>11}  {:>8}", "epoch", "L_train", "L_val", "L_ref", "LR").ok();
    }

    let mut json_logger = JsonLogger::new("evaltune.jsonl").ok();

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    let mut optimizer = Lion::new(config.beta1, lr_scheduler.rate(start_epoch, config.epochs), config.weight_decay);

    let mut grad_stats = GradientStats::new(100);
    let mut indices: Vec<usize> = (0..train_len).collect();

    let mut ema_values = resume.as_ref().map_or_else(|| values.clone(), |d| d.ema.clone());
    let lr_peak = (1..=config.epochs).fold(0.0f64, |m, e| m.max(lr_scheduler.rate(e, config.epochs)));

    // Constant schedule has no tail → uniform Polyak instead of tail EMA.
    let mut ema_active = is_constant_schedule;
    let ema_threshold = if is_constant_schedule { 0.0 } else { 0.3 * lr_peak };

    let mut best_val_loss = resume.as_ref().map_or(f64::MAX, |d| d.best_val_loss);
    let mut best_val_epoch = resume.as_ref().map_or(0, |d| d.best_val_epoch);
    let mut val_smooth = resume.as_ref().map_or(f64::NAN, |d| d.val_smooth);
    let mut best_val_smooth = resume.as_ref().map_or(f64::MAX, |d| d.best_val_smooth);
    let mut best_train_loss = resume.as_ref().map_or(f64::MAX, |d| d.best_train_loss);
    let mut best_train_epoch = resume.as_ref().map_or(0, |d| d.best_train_epoch);
    let mut train_smooth = resume.as_ref().map_or(f64::NAN, |d| d.train_smooth);
    let mut best_train_smooth = resume.as_ref().map_or(f64::MAX, |d| d.best_train_smooth);
    let mut plateau_count = resume.as_ref().map_or(0, |d| d.plateau_count);

    let np = all_params.len();
    let mut best_val_params = resume.as_ref().map_or_else(|| vec![0.0; np], |d| d.best_val_params.clone());
    let mut best_train_params = resume.as_ref().map_or_else(|| vec![0.0; np], |d| d.best_train_params.clone());

    // Not restored on resume: sparklines are a display artifact, not state.
    let mut val_history: Vec<f64> = Vec::new();
    let mut train_history: Vec<f64> = Vec::new();
    let mut prev_val_loss = f64::NAN;

    // Also not restored: the detector re-warms within a slow span, and a warning does not
    // justify more checkpoint surface.
    let mut divergence = DivergenceMonitor::new();

    let psqt_end = psqt::LAYOUT.material_offset;
    let base_end = psqt_end + psqt::LAYOUT.material_len;
    let mob_start = psqt::LAYOUT.mobility_open_offset;
    let mob_end = psqt::LAYOUT.mobility_closed_offset + psqt::LAYOUT.mobility_closed_len;

    // ── Progressive unfreeze
    // Freeze non-psqt/mat for the first unfreeze_epoch epochs.
    // Resume restores the saved mask. It encodes this gate plus any auto-freeze.
    if let Some(d) = &resume {
        fixed_mask.copy_from_slice(&d.frozen);
    } else if config.unfreeze_epoch > 0 {
        for f in &mut fixed_mask[base_end..] {
            *f = true;
        }

        println!("Progressive unfreeze: params {base_end}+ frozen until epoch {}", config.unfreeze_epoch);
    }

    for epoch in start_epoch..=config.epochs {
        let t0 = Instant::now();
        let blend = wdl_scheduler.blend(epoch, config.epochs);

        if let Some(new_k) = k_ctrl.on_epoch(epoch, ctx, &ema_values, blend) {
            println!("  Reoptimized K: {new_k:.6}");

            if let Some(ref mut w) = logger {
                writeln!(w, "# K re-opt @ epoch {epoch}: {new_k:.6}").ok();
            }
        }

        let scheduled_lr = lr_scheduler.rate(epoch, config.epochs) * lr_scale;
        let lr = scheduled_lr.max(0.00001);

        if !ema_active && lr < ema_threshold {
            println!("  EMA activated at epoch {epoch} (lr {lr:.6} < {ema_threshold:.6})");
            ema_active = true;
        }

        let is_restart = epoch > 1 && {
            let prev_scheduled_lr = lr_scheduler.rate(epoch - 1, config.epochs) * lr_scale;
            // ≥50% LR jump = scheduler restart (cosine SGDR cycle boundary).
            // Correct for cosine-with-cycles. Would false-fire on warmup.
            // TODO: gate on scheduler type or add LrScheduler::is_restart_boundary.
            scheduled_lr > prev_scheduled_lr * 1.5
        };

        optimizer.set_lr(lr);

        if is_restart {
            plateau_count = 0;
        }

        rng.shuffle(&mut indices);

        let mut train_loss = 0.0;
        let mut train_count = 0usize;
        let mut total_grads = vec![0.0; values.len()];

        for batch in indices.chunks(config.batch_size) {
            let (mut grads, k_grad, batch_loss, batch_count) = ctx.batch_grad(batch, &values, k_ctrl.k(), blend);

            train_loss += batch_loss;
            train_count += batch_count;

            let n = batch_count.max(1) as f64;
            let norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
            let avg_norm = norm / n;

            grad_stats.update(avg_norm);

            // ── Dynamic Gradient Clipping
            // Clips outliers based on distribution of recent batch norms.
            let clip_thresh = grad_stats.clip_threshold(config.grad_clip);
            let threshold = clip_thresh * n;

            let scale = if norm > threshold { threshold / norm } else { 1.0 };

            for (i, g) in grads.iter_mut().enumerate() {
                *g = *g / n * scale;
                total_grads[i] += *g;
            }

            optimizer.update(&mut values, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2_mask, &lr_mask);

            for value in &mut values[mob_start..mob_end] {
                *value = value.clamp(-MOB_CLAMP, MOB_CLAMP);
            }

            k_ctrl.on_batch(k_grad, batch_count, lr, scale, config.weight_decay);

            // ── Per-parameter Convergence Tracking
            // Freeze parameters that have statistically converged to reduce noise.
            for i in 0..values.len() {
                if !fixed_mask[i] {
                    grad_ema_per_param[i] = 0.99_f64.mul_add(grad_ema_per_param[i], 0.01 * grads[i].abs());
                }
            }

            // ── Tail-only EMA
            // Skip the noisy high-LR phase; only average once LR has
            // decayed below 30 % of its peak. Before that, snapshot
            // the live weights directly.
            if ema_active {
                for i in 0..values.len() {
                    ema_values[i] = config.ema_decay.mul_add(ema_values[i], (1.0 - config.ema_decay) * values[i]);
                }
            } else {
                ema_values.copy_from_slice(&values);
            }
        }

        // Progressive unfreeze; lift the material-only gate.
        if config.unfreeze_epoch > 0 && epoch == config.unfreeze_epoch {
            for (i, p) in all_params.iter().enumerate() {
                fixed_mask[i] = p.is_fixed;
            }

            println!("  Unfrozen all remaining parameters at epoch {epoch}");
        }

        // Auto-freeze stagnant parameters
        if config.auto_freeze && epoch > config.freeze_start_epoch && epoch % config.freeze_cadence == 0 {
            let mut frozen = 0;

            for i in 0..values.len() {
                if !fixed_mask[i] && grad_ema_per_param[i] < config.freeze_threshold && !all_params[i].freeze_resistant {
                    stagnant_epochs[i] += 1;

                    if stagnant_epochs[i] >= config.freeze_consecutive {
                        fixed_mask[i] = true;
                        frozen += 1;
                    }
                } else {
                    stagnant_epochs[i] = 0;
                }
            }

            if frozen > 0 {
                println!("  Auto-frozen {frozen} stagnant parameters.");
            }
        }

        let val_loss = ctx.val_eval(&ema_values, k_ctrl.k(), blend);
        let ref_loss = ctx.val_eval(&ema_values, k_ctrl.k_ref(), 0.0);
        let train_loss = train_loss / train_count.max(1) as f64;

        // Both records select on a smoothed trail. A running minimum over the raw series carries
        // the bias described on DivergenceMonitor, and here it decides which epoch's parameters
        // get saved, so it saves whichever epoch the noise dug deepest. The training series gets
        // no exemption: a fixed-magnitude sign step orbits a minimum rather than settling into
        // it, so train loss stays as noisy as val until the schedule decays.
        train_smooth = if train_smooth.is_finite() { train_smooth + A_FAST * (train_loss - train_smooth) } else { train_loss };

        if train_smooth < best_train_smooth {
            best_train_smooth = train_smooth;
            best_train_loss = train_loss;
            best_train_epoch = epoch;
            best_train_params.copy_from_slice(&ema_values);
        }

        // ── Validation Plateau Detection
        // Reduce LR if validation loss stalls for Constant schedule.
        val_smooth = if val_smooth.is_finite() { val_smooth + A_FAST * (val_loss - val_smooth) } else { val_loss };

        let improved_val = val_smooth < best_val_smooth;

        if improved_val {
            best_val_smooth = val_smooth;
            best_val_loss = val_loss;
            best_val_epoch = epoch;
            best_val_params.copy_from_slice(&ema_values);
            plateau_count = 0;
        } else {
            plateau_count += 1;
            // Plateau LR halving is gated to constant schedules only.
            // Cosine/WSD/etc don't need a separate stall-response mechanism.
            // Reducing LR when the scheduler is already responsible for decay would overcorrect.
            if is_constant_schedule && plateau_count >= config.patience {
                lr_scale *= 0.5;
                plateau_count = 0;
                println!("  Plateau detected, LR scale → {lr_scale:.3}");
            }
        }

        let is_best = improved_val;
        let overfit = divergence.update(train_loss, val_loss);

        // Group-wise gradient norms for diagnostics
        let psqt_norm = total_grads[..psqt_end].iter().map(|g| g * g).sum::<f64>().sqrt();
        let mob_norm = total_grads[mob_start..mob_end].iter().map(|g| g * g).sum::<f64>().sqrt();

        if let Some(ref mut w) = logger {
            writeln!(
                w,
                "{:>3}     {:.6}    {:.6}    {:.6}    {:.4}{}",
                epoch,
                train_loss,
                val_loss,
                ref_loss,
                lr,
                if is_best { " *" } else { "" }
            )
            .ok();
        }

        if let Some(ref mut l) = json_logger {
            if is_restart {
                l.log("restart", &serde_json::json!({ "epoch": epoch }));
            }

            l.log(
                "epoch",
                &serde_json::json!({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "ref_loss": ref_loss,
                    "lr": lr,
                    "is_best": is_best,
                    "psqt_norm": psqt_norm,
                    "mob_norm": mob_norm,
                    "overfit": overfit
                }),
            );
        }

        let elapsed = t0.elapsed().as_secs_f32();

        // Loss has no absolute scale, so color the live value by its per-epoch
        // trend; dropped from last epoch → green ▼, rose → red ▲.
        let (arrow, trend) = if !prev_val_loss.is_finite() || (val_loss - prev_val_loss).abs() < 1e-7 {
            ('·', palette::fg(palette::LABEL))
        } else if val_loss < prev_val_loss {
            ('▼', palette::fg(color::advantage(0.7)))
        } else {
            ('▲', palette::fg(color::advantage(-0.7)))
        };

        let lab = palette::fg(palette::LABEL);
        let dim = palette::fg(palette::DIM);

        let (mark, epoch_c) = if is_best { ("✦ ", palette::fg(palette::BRAND)) } else { ("  ", dim.clone()) };
        let warn = if overfit { format!("  {}⚠ overfit{RESET}", palette::fg(color::advantage(-1.0))) } else { String::new() };

        #[rustfmt::skip]
        println!(
            "{mark}{epoch_c}Epoch {epoch:>3}/{}{RESET}  \
             {lab}val{RESET} {trend}{val_loss:.6}{RESET} {trend}{arrow}{RESET}  \
             {lab}train{RESET} {dim}{train_loss:.6}{RESET}  \
             {lab}ref{RESET} {dim}{ref_loss:.6}{RESET}  \
             {lab}lr{RESET} {}{lr:.4}{RESET}  {dim}{elapsed:.2}s{RESET}{warn}{CLEAR_LINE}",
            config.epochs,
            palette::fg(palette::VALUE),
        );

        val_history.push(val_loss);
        train_history.push(train_loss);
        prev_val_loss = val_loss;

        if epoch % 20 == 0 || epoch == config.epochs {
            let val_tail = &val_history[val_history.len().saturating_sub(40)..];
            let train_tail = &train_history[train_history.len().saturating_sub(40)..];

            println!("\n  {lab}L_val{RESET}    {}", loss_sparkline(val_tail));
            println!("\n  {lab}L_train{RESET}  {}", loss_sparkline(train_tail));

            if epoch != config.epochs {
                print_params(&all_params, &initial_values, &ema_values);
            }

            if let Err(e) = save_checkpoint("evaltune_checkpoint.json", &all_params, &TrainerState {
                epoch: epoch + 1, // resume starts here; the current epoch is already done
                lr_scale,
                k: k_ctrl.k(),
                k_ref: k_ctrl.k_ref(),
                k_momentum: k_ctrl.momentum,
                best_val_loss,
                best_val_epoch,
                val_smooth,
                best_val_smooth,
                best_train_loss,
                best_train_epoch,
                train_smooth,
                best_train_smooth,
                plateau_count,
                rng_seed,
                dataset: dataset_fnv,
                dataset_path: dataset_label,
                values: &values,
                momentum: &momentum,
                ema: &ema_values,
                grad_ema: &grad_ema_per_param,
                stagnant: &stagnant_epochs,
                frozen: &fixed_mask,
                best_val_params: &best_val_params,
                best_train_params: &best_train_params,
            }) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }
    }

    // Flush the log before writing final reports,
    // since print_results opens its own handle to the same file.
    drop(logger);

    sensitivity_report(&all_params, &grad_ema_per_param, &fixed_mask);
    let last_val = val_history.last().copied().unwrap_or(0.0);
    let last_train = train_history.last().copied().unwrap_or(0.0);
    print_results(
        &all_params,
        &initial_values,
        &ema_values,
        &BestEpochs {
            best_val_params: &best_val_params,
            best_val_loss,
            best_val_epoch,
            best_train_params: &best_train_params,
            best_train_loss,
            best_train_epoch,
            last_val,
            last_train,
        },
        config.epochs,
    );

    best_val_loss
}

fn sensitivity_report(params: &[Tunable], grad_ema: &[f64], fixed_mask: &[bool]) {
    let Ok(mut f) = fs::File::create("sensitivity-report.txt") else { return };
    let mut w = io::BufWriter::new(&mut f);

    writeln!(w, "Sensitivity Analysis").ok();
    writeln!(w).ok();

    let mut sensitivities = Vec::new();
    let mut frozen = Vec::new();

    for p in params {
        let delta = grad_ema[p.idx];

        if p.is_fixed || fixed_mask[p.idx] {
            frozen.push((delta, p.idx, p.name.as_str()));
        } else {
            sensitivities.push((delta, p.idx, p.name.as_str()));
        }
    }

    sensitivities.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    frozen.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let max_width = |list: &[(f64, usize, &str)]| list.iter().take(10).map(|r| r.2.len()).max().unwrap_or(20);
    let active_width = max_width(&sensitivities) + 1;
    let frozen_width = frozen.iter().take(10).map(|r| r.2.len()).max().unwrap_or(20) + 1;

    writeln!(w, "  Top Load-Bearing Parameters:").ok();

    for (i, (delta, _, name)) in sensitivities.iter().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    writeln!(w).ok();
    writeln!(w, "  Lowest-Impact Parameters:").ok();

    for (i, (delta, _, name)) in sensitivities.iter().rev().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    if !frozen.is_empty() {
        writeln!(w).ok();
        writeln!(w, "  Highest Sensitivity Auto-Frozen/Fixed Parameters:").ok();

        for (i, (delta, _, name)) in frozen.iter().take(10).enumerate() {
            writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = frozen_width).ok();
        }
    }
}

fn resolve_dataset_paths(input: &str) -> Option<Vec<String>> {
    if input == "default" {
        let mut paths = Vec::new();

        if let Ok(entries) = fs::read_dir("data") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.to_string_lossy();

                if name.ends_with(".soul.zst") || name.ends_with(".soul") {
                    paths.push(name.to_string());
                }
            }
        }

        if paths.is_empty() {
            eprintln!("{}Error: No default dataset found in data/ directory.{RESET}", color::ansi_fg((225, 89, 91)));
            eprintln!("Please provide a dataset path using --dataset <path>");
            None
        } else {
            println!("Auto-discovered datasets: {}", paths.join(", "));
            Some(paths)
        }
    } else {
        let paths: Vec<String> = input
            .split(',')
            .map(str::trim)
            .map(|s| {
                if path::Path::new(s).exists() {
                    s.to_string()
                } else {
                    let data_prefixed = format!("data/{s}");
                    if path::Path::new(&data_prefixed).exists() { data_prefixed } else { s.to_string() }
                }
            })
            .collect();

        Some(paths)
    }
}

/// Parameter group by position in the layout, the axis the per-group optimizer
/// masks (decay, momentum, learning rate) all key off.
enum ParamGroup {
    Psqt,
    Material,
    Mobility,
    Other,
}

/// Classify a parameter index into its layout group: the single source of the
/// group boundaries the masks below share.
fn param_group(i: usize) -> ParamGroup {
    if i < psqt::LAYOUT.material_offset {
        ParamGroup::Psqt
    } else if i < psqt::LAYOUT.mobility_open_offset {
        ParamGroup::Material
    } else if i < psqt::LAYOUT.mobility_closed_offset + psqt::LAYOUT.mobility_closed_len {
        ParamGroup::Mobility
    } else {
        ParamGroup::Other
    }
}

/// Weight decay mask: not all parameters deserve equal punishment.
///
/// - PSQT center squares decay at 0.5× (central values are more structurally
///   significant; aggressive decay risks flattening critical gradients).
/// - Mobility weights decay at 1.5× (these can drift without bound since their
///   features are unbounded integer counts).
/// - Everything else decays at 1.0×.
fn build_decay_mask(params: &[Tunable]) -> Vec<f64> {
    (0..params.len())
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => {
                let sq = i % 32;
                let (row, col) = (sq / 4, sq % 4);
                let is_center = (2..=5).contains(&row) && (2..=3).contains(&col);
                if is_center { 0.5 } else { 1.0 }
            },

            ParamGroup::Mobility => 1.5,
            ParamGroup::Material | ParamGroup::Other => 1.0,
        })
        .collect()
}

/// Per-group momentum decay mask.
///
/// Different parameter groups have different natural gradient timescales.
/// - PSQT (0.995): squares only see updates when a piece of that type lands
///   there: longer momentum smooths sparse signal across positions.
/// - Mobility (0.95): features are computed every position; shorter momentum
///   lets weights track the faster dynamics without lag.
/// - Everything else (0.99): the existing default from the config.
fn build_beta2_mask(params: &[Tunable], default_beta2: f64) -> Vec<f64> {
    (0..params.len())
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => 0.995,
            ParamGroup::Mobility => 0.95,
            ParamGroup::Material | ParamGroup::Other => default_beta2,
        })
        .collect()
}

/// Per-group learning-rate mask: PSQT, material, mobility, and the rest each scale
/// by their configured rate, so groups on different gradient scales tune independently.
fn build_lr_mask(params: &[Tunable], config: &EvalTuneConfig) -> Vec<f64> {
    (0..params.len())
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => config.lr_psqt,
            ParamGroup::Material => config.lr_material,
            ParamGroup::Mobility => config.lr_mobility,
            ParamGroup::Other => config.lr_other,
        })
        .collect()
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
        let mut d = DivergenceMonitor::new();
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
        let mut d = DivergenceMonitor::new();
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
        let mut d = DivergenceMonitor::new();

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
        let mut d = DivergenceMonitor::new();

        // Maximally divergent input, so warmup is the only thing holding the flag down.
        for e in 0..TREND_SLOW {
            let t = e as f64;
            assert!(!d.update(0.5 - t * 1e-3, 0.5 + t * 1e-3), "reported at epoch {e}, inside the warmup span");
        }

        assert!(d.update(0.5 - 0.04, 0.5 + 0.04), "must report once the slow span has filled");
    }
}
