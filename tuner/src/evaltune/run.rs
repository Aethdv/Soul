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
    io::{BufWriter, Write},
    time::Instant,
};

use palette::{CLEAR_LINE, RESET};
use rayon::prelude::*;

use super::{
    engine::{FeatureRecord, LAYOUT, Tunable, color, eval_params},
    groups::{GROUP_NAMES, build_clip_mask, build_decay_mask, build_lr_mask, group_ranges},
    lion::{GateCensus, Lion, build_beta2_mask},
    loader::{self, dataset_fingerprint, resolve_dataset_paths},
    palette,
    probes::{curvature_report, gather_cost, val_cost},
    report::*,
    scale::{GAUGE_PROBE, Gauge, KController, canonicalize},
    storage::*,
    training::*,
};
use crate::core::{
    config::{EvalTuneConfig, Init, KMode, LossFn, LrScheduleConfig, RANDOM_INIT_SPREAD},
    logger::JsonLogger,
    shuffle::{self, Shuffler},
};

/// Fixes which tenth of a dataset is held out, so two runs over one dataset are scored on the
/// same positions. The value is arbitrary and permanent: changing it renumbers every
/// `best_val_loss` ever recorded on every dataset.
const VAL_SPLIT_SEED: u64 = 0x5350_4C49_5432_3736;

/// What a loaded dataset is for.
///
/// Both arms want every step up to `TrainerContext`, feature extraction included, so the choice
/// rides through the loader rather than growing a second copy of it.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Task {
    Train,
    /// Time the gradient pass over shuffled indices against sequential ones. Identical arithmetic
    /// over identical records, so the difference is the cost of the random gathers, which is the
    /// measurement that says whether the epoch loop is bound by memory or by math.
    GatherCost,
    /// Build the objective's exact Hessian at the shipped parameters and report what the data
    /// constrains, what it leaves free, and which parameters merely restate each other.
    Curvature,
    /// Time one fused validation traversal against the two separate ones it replaced, in a
    /// single process, since the val column's run-to-run spread swallows the difference.
    ValCost,
}

pub struct TrainerContext<'a> {
    pub train: &'a [loader::SoulEntry],
    pub val: &'a [loader::SoulEntry],
    pub records: &'a [FeatureRecord],
    pub train_count: usize,
    pub phase_weights: &'a [f64],
    loss_fn: LossFn,
    vol_threshold: i16,
    vol_adaptive: bool,
}

impl TrainerContext<'_> {
    pub fn passes_vol_filter(&self, entry: &loader::SoulEntry, static_eval: i16) -> bool {
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

    pub fn batch_grad(&self, batch_indices: &[u32], values: &[f64], k: f64, blend: f64) -> (Vec<f64>, f64, f64, usize) {
        batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(mut g, mut k_g, mut loss, mut count), chunk| {
                    for &i in chunk {
                        let i = i as usize;
                        let entry = &self.train[i];
                        let record = &self.records[i];

                        if !self.passes_vol_filter(entry, record.static_eval) {
                            continue;
                        }

                        let target = wdl_target(entry, k, blend);
                        let eval = loader::eval_record_full(record, values);
                        let sig = sigmoid(eval.score, k);
                        let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[i] };
                        let gs = self.loss_fn.grad_scale(sig, target, k);

                        loss += w * self.loss_fn.loss(sig, target);
                        loader::accumulate_record_grad(record, &eval, gs * w, &mut g);

                        // gs is ∂L/∂score = K · (sig - target) · dσ/dscore.
                        // We need ∂L/∂K = score · (sig - target) · dσ/dscore.
                        // So ∂L/∂K = (gs / K) · score.
                        k_g += (gs / k) * eval.score * w;

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

    /// Validation loss at each `(k, blend)` probe, over one pass of the split.
    ///
    /// The eval is the whole cost of that pass and depends on neither, so a second probe rides
    /// along for a sigmoid and a target more. The epoch report wants two, its live loss and the
    /// frozen-`k_ref` reference; the K search wants one per probe.
    pub fn val_eval<const N: usize>(&self, values: &[f64], probes: [(f64, f64); N]) -> [f64; N] {
        let (wsum, weight) = self
            .val
            .par_iter()
            .enumerate()
            .fold(
                || ([0.0_f64; N], 0.0_f64),
                |(mut wsum, mut weight), (idx, entry)| {
                    let record = &self.records[self.train_count + idx];

                    if !self.passes_vol_filter(entry, record.static_eval) {
                        return (wsum, weight);
                    }

                    let score = loader::eval_record(record, values);
                    let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[self.train_count + idx] };

                    for (sum, &(k, blend)) in wsum.iter_mut().zip(&probes) {
                        let sig = sigmoid(score, k);
                        let target = wdl_target(entry, k, blend);

                        *sum += w * self.loss_fn.loss(sig, target);
                    }

                    weight += w;

                    (wsum, weight)
                },
            )
            .reduce(
                || ([0.0_f64; N], 0.0_f64),
                |(mut sums, w1), (rhs, w2)| {
                    for (sum, add) in sums.iter_mut().zip(&rhs) {
                        *sum += add;
                    }

                    (sums, w1 + w2)
                },
            );

        if weight > 0.0 { wsum.map(|sum| sum / weight) } else { [0.0; N] }
    }
}

/// Batch order and the validation holdout draw from separate seeds,
/// so a retune moves one without moving the other.
#[derive(Clone, Copy)]
struct Seeds {
    rng_seed: u64,
    split_seed: u64,
}

/// The starting parameter vector for a fresh run; a resume overrides it with the
/// checkpoint's. Fixed slots hold their declared values under every mode.
pub fn seed_values(params: &[Tunable], init: Init, seed: u64) -> Vec<f64> {
    let mut rng = fastrand::Rng::with_seed(seed);

    params
        .iter()
        .map(|p| match init {
            _ if p.is_fixed => p.value,
            Init::Default => p.value,
            Init::Zero => 0.0,
            Init::Random => (rng.f64() * 2.0 - 1.0) * RANDOM_INIT_SPREAD,
        })
        .collect()
}

pub fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>, task: Task) -> f64 {
    let total_start = Instant::now();

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
    let best_val = train_entries(all_entries, &dataset_label, config, resume_path, task);

    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n{}Done in {elapsed:.2}s{RESET}", palette::fg(palette::BRAND));
    best_val
}

fn train_entries(
    mut entries: Vec<loader::SoulEntry>,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    task: Task,
) -> f64 {
    let dataset_fnv = dataset_fingerprint(&entries);

    let (rng_seed, split_seed) = match resume_path {
        Some(path) => {
            let cp = peek_checkpoint(path).unwrap_or_else(|e| {
                eprintln!("Failed to read checkpoint: {e}");
                std::process::exit(1);
            });

            if let Some(s) = config.seed
                && s != cp.rng_seed
            {
                println!("--seed {s} ignored: resume reuses the checkpoint's batch-order seed {}", cp.rng_seed);
            }

            // The split seed replays its shuffle only over the same entries.
            if cp.dataset != dataset_fnv {
                eprintln!(
                    "{}[!] Warning: dataset does not match the checkpoint's fingerprint.\n\
                     [!] The train/val split will differ from the original run: positions the\n\
                     [!] checkpoint trained on may now sit in val, making its loss optimistic.{RESET}",
                    color::ansi_fg((225, 89, 91)),
                );
            }

            (cp.rng_seed, cp.split_seed.unwrap_or(cp.rng_seed))
        },

        None => (config.seed.unwrap_or_else(|| fastrand::u64(..)), config.split_seed.unwrap_or(VAL_SPLIT_SEED)),
    };

    // Games sit contiguously in the file, so the split needs a shuffle rather than a cut, and the
    // shuffle needs a seed of its own: under the training seed, each run holds out a different
    // tenth and no two validation losses compare.
    fastrand::Rng::with_seed(split_seed).shuffle(&mut entries);

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

    match task {
        Task::GatherCost => {
            gather_cost(&ctx, config);
            return 0.0;
        },

        Task::Curvature => {
            curvature_report(&ctx, config);
            return 0.0;
        },

        Task::ValCost => {
            val_cost(&ctx, config);
            return 0.0;
        },

        Task::Train => {},
    }

    let seeds = Seeds { rng_seed, split_seed };

    train_loop(train.len(), "SoulEntry", dataset_label, config, resume_path, seeds, dataset_fnv, &ctx)
}

fn grad_combine((mut g1, l1): (Vec<f64>, f64), (g2, l2): (Vec<f64>, f64)) -> (Vec<f64>, f64) {
    for (a, b) in g1.iter_mut().zip(g2) {
        *a += b;
    }

    (g1, l1 + l2)
}

fn train_loop(
    train_len: usize,
    mode_label: &str,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    seeds: Seeds,
    dataset_fnv: u64,
    ctx: &TrainerContext,
) -> f64 {
    let Seeds { rng_seed, split_seed } = seeds;

    let all_params = eval_params::collect_parameters();
    let default_values: Vec<f64> = all_params.iter().map(|p| p.value).collect();

    let resume = resume_path.map(|path| {
        println!("Resuming from checkpoint: {}{path}{RESET}", palette::fg(palette::VALUE));
        let data = load_checkpoint(path, &all_params, &default_values).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {e}");
            std::process::exit(1);
        });

        if data.fresh_params > 0 {
            println!(
                "{}{}{RESET} parameter(s) are newer than the checkpoint, starting from code defaults",
                palette::fg(palette::VALUE),
                data.fresh_params,
            );
        }

        data
    });

    let (start_epoch, mut lr_scale) = resume.as_ref().map_or((1, 1.0), |d| (d.epoch, d.lr_scale));
    let mut values = match resume.as_ref() {
        Some(d) => d.values.clone(),
        None => seed_values(&all_params, config.init, rng_seed),
    };

    let is_constant_schedule = matches!(config.lr_schedule, LrScheduleConfig::Constant { .. });
    let lr_scheduler = config.lr_schedule.clone().into_scheduler();
    let wdl_scheduler = config.wdl_schedule.clone().into_scheduler();

    let init_blend = wdl_scheduler.blend(1, config.epochs);
    let mut k_ctrl = KController::bootstrap(config, ctx, &values, &default_values, init_blend, resume.as_ref());

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

    if split_seed != VAL_SPLIT_SEED {
        println!("{lab}Split seed:{RESET} {v}{split_seed}{RESET} (L_val does not compare to default-split runs)");
    }

    if config.init != Init::Default && resume.is_none() {
        println!("{lab}Init:{RESET}       {v}{:?}{RESET} (cold start; K is meaningless until material grows)", config.init);
    }

    let initial_values = values.clone();

    // Train split only, strided to span it: nothing is fitted to the probe, but the
    // val column keeps its own positions.
    let train = &ctx.records[..ctx.train_count];
    let stride = (train.len() / GAUGE_PROBE).max(1);
    let probe: Vec<&FeatureRecord> = train.iter().step_by(stride).take(GAUGE_PROBE).collect();

    let mut gauge = Gauge::new(probe, &default_values);
    let hold_scale = gauge.holds(&values);

    // Setup optimizer state and convergence tracking
    let mut fixed_mask: Vec<bool> = all_params.iter().map(|p| p.is_fixed).collect();
    let decay_mask = build_decay_mask(all_params.len());
    let beta2_mask = build_beta2_mask(all_params.len(), config.beta2);
    let lr_mask = build_lr_mask(all_params.len(), config);
    let clip_mask = build_clip_mask(all_params.len());

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

        if split_seed != VAL_SPLIT_SEED {
            writeln!(w, "Split:     {split_seed}").ok();
        }

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

    let mut json_logger = JsonLogger::new(&config.log_path).ok();

    // Where a run starts. Inferring it from the epoch number going backwards is
    // wrong for a resume, which continues the count.
    if let Some(ref l) = json_logger {
        l.log(
            "run",
            &serde_json::json!({
                "seed": rng_seed,
                "split_seed": split_seed,
                "epochs": config.epochs,
                "start_epoch": start_epoch,
                "dataset": dataset_label,
                "init": format!("{:?}", config.init),
                "params": all_params.len(),
            }),
        );
    }

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    let mut optimizer =
        Lion::new(all_params.len(), config.beta1, lr_scheduler.rate(start_epoch, config.epochs), config.weight_decay);

    if let Some(d) = &resume {
        optimizer.restore_momentum(&d.momentum);
    }

    let mut grad_stats = GradientStats::new(100);

    // u32 halves the working set of the per-epoch shuffle and of the index stream through the
    // batch loop, worth 12% of epoch time at 32.8M positions.
    let mut indices = vec![0u32; train_len];
    let mut shuffler = Shuffler::new(train_len);

    let mut ema_values = resume.as_ref().map_or_else(|| values.clone(), |d| d.ema.clone());
    let lr_peak = (1..=config.epochs).fold(0.0f64, |m, e| m.max(lr_scheduler.rate(e, config.epochs)));

    // Constant schedule has no tail → uniform Polyak instead of tail EMA.
    let mut ema_active = is_constant_schedule;
    let ema_threshold = if is_constant_schedule { 0.0 } else { 0.3 * lr_peak };

    let mut progress = resume.as_ref().map_or_else(Progress::default, |d| d.progress.clone());

    let slots = all_params.len();
    let mut best_val_params = resume.as_ref().map_or_else(|| vec![0.0; slots], |d| d.best_val_params.clone());
    let mut best_train_params = resume.as_ref().map_or_else(|| vec![0.0; slots], |d| d.best_train_params.clone());

    // Not restored on resume: sparklines are a display artifact, not state.
    let mut val_history: Vec<f64> = Vec::new();
    let mut train_history: Vec<f64> = Vec::new();
    let mut prev_val_loss = f64::NAN;

    // Also not restored: the detector re-warms within a slow span, and a warning does not
    // justify more checkpoint surface.
    let mut divergence = DivergenceMonitor::default();

    // Not restored on resume: re-seeding from the resumed EMA is the right baseline anyway.
    let mut prev_quantized = vec![0i32; slots];
    let mut quantized = vec![0i32; slots];
    quantize(&ema_values, &mut prev_quantized);

    let mut epochs_run = 0usize;
    let mut epoch_seconds = 0.0f64;
    let mut grad_seconds = 0.0f64;
    let mut shuffle_seconds = 0.0f64;
    let mut val_seconds = 0.0f64;
    let mut epoch_positions = 0u64;

    let psqt_end = LAYOUT.material_offset;
    let base_end = psqt_end + LAYOUT.material_len;
    let mob_start = LAYOUT.mobility_open_offset;
    let mob_end = LAYOUT.mobility_closed_offset + LAYOUT.mobility_closed_len;

    let group_ranges = group_ranges(slots);
    let mut run_census = [GateCensus::default(); GROUP_NAMES.len()];

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
        let mut epoch_census = [GateCensus::default(); GROUP_NAMES.len()];

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
            progress.plateau_count = 0;
            // A latched EMA would average the new cycle's peak-LR weights.
            ema_active = is_constant_schedule;
        }

        let t_shuffle = Instant::now();
        if config.shuffle_block > 0 {
            shuffle::fill_blocked(&mut indices, rng.u64(..), config.shuffle_block);
        } else {
            shuffler.fill(&mut indices, rng.u64(..));
        }
        let shuffle_secs = t_shuffle.elapsed().as_secs_f32();

        let mut train_loss = 0.0;
        let mut train_count = 0usize;
        let mut total_grads = vec![0.0; values.len()];

        let t_grad = Instant::now();

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

            // Before the update, while momentum still holds what the gate is about to read.
            if config.gate_census {
                for (census, range) in epoch_census.iter_mut().zip(&group_ranges) {
                    census.absorb(optimizer.census(range.clone(), &grads, &fixed_mask));
                }
            }

            optimizer.update(&mut values, &grads, &decay_mask, &fixed_mask, &beta2_mask, &lr_mask, &clip_mask);

            k_ctrl.on_batch(k_grad, batch_count, lr, scale, config.weight_decay);
            canonicalize(&mut values, &all_params);

            if hold_scale {
                gauge.restore(&mut values, &mut optimizer, &mut k_ctrl);
            }

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

        let grad_secs = t_grad.elapsed().as_secs_f32();

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

        let t_val = Instant::now();

        // ref_loss reads the same scores at frozen k_ref against the pure outcome,
        // which is what makes one run's number comparable to another's.
        let [val_loss, ref_loss] = ctx.val_eval(&ema_values, [(k_ctrl.k(), blend), (k_ctrl.k_ref(), 0.0)]);
        let val_secs = t_val.elapsed().as_secs_f32();

        let train_loss = train_loss / train_count.max(1) as f64;

        if progress.record_train(epoch, train_loss) {
            best_train_params.copy_from_slice(&ema_values);
        }

        let improved_val = progress.record_val(epoch, val_loss);

        if improved_val {
            best_val_params.copy_from_slice(&ema_values);
        } else {
            // A decaying schedule already answers a stall; halving on top would correct twice.
            if is_constant_schedule && progress.plateau_count >= config.patience {
                lr_scale *= 0.5;
                progress.plateau_count = 0;
                println!("  Plateau detected, LR scale → {lr_scale:.3}");
            }
        }

        let is_best = improved_val;
        let overfit = divergence.update(train_loss, val_loss);

        // Parameters that crossed an integer boundary this epoch. Zero from here on means the
        // run is still descending in f64 and shipping nothing, which is where a budget ends.
        quantize(&ema_values, &mut quantized);
        let moved = quantized.iter().zip(&prev_quantized).filter(|(q, p)| q != p).count();
        prev_quantized.copy_from_slice(&quantized);

        // Group-wise gradient norms for diagnostics
        let psqt_norm = total_grads[..psqt_end].iter().map(|g| g * g).sum::<f64>().sqrt();
        let mob_norm = total_grads[mob_start..mob_end].iter().map(|g| g * g).sum::<f64>().sqrt();

        if let Some(ref mut w) = logger {
            writeln!(
                w,
                "{:>5}  {:>11.6}  {:>11.6}  {:>11.6}  {:>8.4}{}",
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
                    "overfit": overfit,
                    "moved": moved,
                    "gauge": gauge.applied
                }),
            );
        }

        let elapsed = t0.elapsed().as_secs_f32();

        // Denominator is the gradient pass, not the epoch: every epoch trains the same
        // position count, so an epoch-timed rate would only restate the timer.
        let mpos = train_count as f32 / grad_secs.max(1e-6) / 1e6;

        epochs_run += 1;
        epoch_seconds += f64::from(elapsed);
        grad_seconds += f64::from(grad_secs);
        shuffle_seconds += f64::from(shuffle_secs);
        val_seconds += f64::from(val_secs);
        epoch_positions += train_count as u64;

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
             {lab}lr{RESET} {}{lr:.4}{RESET}  {lab}Δp{RESET} {dim}{moved:>3}{RESET}  \
             {dim}{elapsed:.2}s{RESET}  {dim}{mpos:.1}M pos/s{RESET}{warn}{CLEAR_LINE}",
            config.epochs,
            palette::fg(palette::VALUE),
        );

        if config.gate_census {
            let mut all = GateCensus::default();

            for (total, epoch) in run_census.iter_mut().zip(&epoch_census) {
                total.absorb(*epoch);
                all.absorb(*epoch);
            }

            println!(
                "  {lab}gate{RESET} skip {v}{:.1}%{RESET}  canonical {v}{:.1}%{RESET}  band {v}{:.2}%{RESET}  \
                 c-only {v}{:.1}%{RESET}  waived {v}{:.1}%{RESET}  dead {v}{:.1}%{RESET}  no grad {v}{:.1}%{RESET}",
                100.0 * all.share(all.skipped),
                100.0 * all.share(all.canonical),
                100.0 * all.share(all.band),
                100.0 * all.share(all.canonical_only),
                100.0 * all.share(all.epsilon_waived),
                100.0 * all.share(all.dead),
                100.0 * all.share(all.absent),
            );
        }

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
                progress: &progress,
                rng_seed,
                split_seed,
                dataset: dataset_fnv,
                dataset_path: dataset_label,
                values: &values,
                momentum: optimizer.momentum(),
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

    // The JSON log opens in append mode, so a seed sweep writes every run's final params into
    // one file for reading the spread directly.
    if let Some(ref l) = json_logger {
        quantize(&best_val_params, &mut quantized);

        l.log(
            "final",
            &serde_json::json!({
                "seed": rng_seed,
                "split_seed": split_seed,
                "epochs": epochs_run,
                "best_val_loss": progress.best_val_loss,
                "best_val_epoch": progress.best_val_epoch,
                "best_train_loss": progress.best_train_loss,
                "best_train_epoch": progress.best_train_epoch,
                "params": quantized,
                "sensitivity": grad_ema_per_param,
            }),
        );
    }

    // A cold start was left to find its own scale, so its output is normalized
    // here instead: the search reads centipawns, and nothing else would put the
    // run's eval back on the scale `search_params` was written against.
    let landed = if hold_scale {
        1.0 / gauge.applied
    } else {
        gauge.normalize(&mut ema_values);
        gauge.normalize(&mut best_train_params);
        1.0 / gauge.normalize(&mut best_val_params)
    };

    let lab = palette::fg(palette::LABEL);
    let v = palette::fg(palette::VALUE);
    let how = if hold_scale { "held through the run" } else { "normalized on the way out" };

    let gauge_line = format!("\n{lab}Gauge:{RESET}      {v}{landed:.3}×{RESET} pull on the eval's scale, {how}\n");
    let off_scale = off_scale_warning(Gauge::measure(&gauge.probe, &ema_values) / gauge.reference);
    let clamped_k = clamped_k_warning(config, k_ctrl.k());
    let calibration = calibration_report(ctx, &best_val_params, k_ctrl.k());
    let census = if config.gate_census { gate_census_report(&run_census) } else { String::new() };

    print!("{gauge_line}");
    eprint!("{off_scale}{clamped_k}");
    print!("{calibration}{census}");

    if let Some(ref mut w) = logger {
        for part in [&gauge_line, &off_scale, &clamped_k, &calibration, &census] {
            write!(w, "{}", color::strip(part)).ok();
        }
    }

    // Flushed before the reports below, so a panic in one cannot cost the epoch history.
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
            best_val_loss: progress.best_val_loss,
            best_val_epoch: progress.best_val_epoch,
            best_train_params: &best_train_params,
            best_train_loss: progress.best_train_loss,
            best_train_epoch: progress.best_train_epoch,
            last_val,
            last_train,
        },
        config.epochs,
    );

    if epochs_run > 0 {
        // Epoch timers stop before the checkpoint write, so these totals fall short of the
        // process wall clock by that plus feature extraction.
        let avg_mpos = epoch_positions as f64 / grad_seconds.max(1e-6) / 1e6;
        let rest_seconds = epoch_seconds - grad_seconds - shuffle_seconds - val_seconds;
        let lab = palette::fg(palette::LABEL);
        let dim = palette::fg(palette::DIM);

        println!(
            "\n{lab}Trained{RESET} {epochs_run} epochs in {epoch_seconds:.2}s  \
             {dim}grad {grad_seconds:.2}s · shuffle {shuffle_seconds:.2}s · val {val_seconds:.2}s · rest {rest_seconds:.2}s{RESET}  \
             {avg_mpos:.1}M pos/s"
        );
    }

    progress.best_val_loss
}

#[cfg(test)]
mod tests {
    use super::{
        super::engine::{Position, SoulEntry},
        *,
    };

    #[test]
    fn a_cold_start_leaves_the_fixed_slots_alone() {
        let params = eval_params::collect_parameters();

        for init in [Init::Zero, Init::Random] {
            let values = seed_values(&params, init, 7);

            for (p, v) in params.iter().zip(&values) {
                if p.is_fixed {
                    assert_eq!(*v, p.value, "{init:?} moved fixed slot {}", p.name);
                } else {
                    assert!(v.abs() <= RANDOM_INIT_SPREAD, "{init:?} left {} at {v}", p.name);
                }
            }
        }

        let phase = LAYOUT.phase_offset;
        let zeroed = seed_values(&params, Init::Zero, 7);
        assert!(zeroed[phase..phase + LAYOUT.phase_len].iter().any(|w| *w > 0.0), "phase taper zeroed");
    }
}
