//! Gradient descent on weights against WDL-labeled positions.
//!
//! Features are extracted to `FeatureRecord`s once at startup; epochs read the cached
//! SoA path (`eval_record` + `accumulate_record_grad`). An initial golden-section search
//! fixes the sigmoid's centipawn-to-winrate scale; `k_ref` stays frozen so reference loss
//! compares across runs.

use std::{
    env, fs,
    io::{BufWriter, Write},
    path::PathBuf,
    process,
    time::Instant,
};

use rayon::prelude::*;

use crate::{
    alarm,
    config::{EvalTuneConfig, Init, LossFn, RANDOM_INIT_SPREAD},
    engine::{
        FeatureRecord, ReplayFilter, SoulEntry, Tunable, accumulate_record_grad, color, eval_params, eval_record, eval_record_full,
        flip_wdl,
    },
    lion::{GateTally, Lion},
    loader::{LoadedDataset, dataset_fingerprint, is_viri_path, load_datasets, resolve_dataset_paths},
    probes::{batch_size_report, curvature_report, gather_cost, momentum_report, val_cost},
    report::{
        BRAND, BestSnapshots, DIM, EpochStats, LAB, RESET, VAL, calibration_report, clamped_k_warning, clip_report, gate_columns,
        gate_tally_report, off_scale_warning, print_dataset_stats, print_loss_history, print_params, print_results, quantize,
        sensitivity_report,
    },
    scale::{GAUGE_PROBE, Gauge, KController, canonicalize},
    schedule::LrSchedule,
    shuffle::Shuffler,
    storage::{JsonLog, RunState, TrainerState, load_checkpoint, peek_checkpoint, save_checkpoint},
    training::{
        DivergenceMonitor, GROUP_NAMES, GradientStats, Progress, add_slice, beta2_for_lr, describe_beta2, group_ranges, l2_norm,
        merge_sample_weights, phase_balance_weights, sigmoid, slot_masks, wdl_target,
    },
};

pub(crate) const ARTIFACT_DIR: &str = "EVALTUNE_ARTIFACT_DIR";

const VAL_SPLIT_SEED: u64 = 0x5350_4C49_5432_3736;
const K_FIT_PROBE: usize = 65_536;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Task {
    Train,
    GatherCost,
    Curvature,
    ValCost,
    BatchSize,
    Momentum,
}

pub(crate) struct Objective<'a> {
    pub(crate) train: &'a [FeatureRecord],
    pub(crate) val: &'a [FeatureRecord],
    pub(crate) sample_weights: &'a [f64],
    loss_fn: LossFn,
    volatility_threshold: i16,
    volatility_adaptive: bool,
}

impl Objective<'_> {
    #[inline]
    pub(crate) const fn loss_fn(&self) -> LossFn { self.loss_fn }

    pub(crate) fn passes_volatility_filter(&self, record: &FeatureRecord) -> bool {
        if self.volatility_threshold == 0 || record.score == SoulEntry::NO_SCORE {
            return true;
        }

        let threshold = if self.volatility_adaptive {
            let occ = i16::from(record.piece_count);
            self.volatility_threshold.saturating_add((occ - 10).saturating_mul(2))
        } else {
            self.volatility_threshold
        }
        .max(0);

        (i32::from(record.static_eval) - i32::from(record.score)).abs() <= i32::from(threshold)
    }

    pub(crate) fn batch_grad(&self, batch_indices: &[u32], values: &[f64], k: f64, blend: f64) -> (Vec<f64>, f64, f64, usize) {
        batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(mut grad_acc, mut k_grad_acc, mut loss_acc, mut count), chunk| {
                    for &idx in chunk {
                        let i = idx as usize;
                        let record = &self.train[i];

                        if !self.passes_volatility_filter(record) {
                            continue;
                        }

                        let (target, dt_dk) = wdl_target(record, k, blend);
                        let eval = eval_record_full(record, values);
                        let p = sigmoid(eval.score, k);
                        let w = if self.sample_weights.is_empty() { 1.0 } else { self.sample_weights[i] };
                        let gs = self.loss_fn.grad_scale(p, target, k);
                        let k_term = if dt_dk == 0.0 { 0.0 } else { self.loss_fn.grad_target(p, target) * dt_dk };

                        loss_acc += w * self.loss_fn.loss(p, target);
                        accumulate_record_grad(record, &eval, gs * w, &mut grad_acc);
                        k_grad_acc += ((gs / k) * eval.score + k_term) * w;
                        count += 1;
                    }
                    (grad_acc, k_grad_acc, loss_acc, count)
                },
            )
            .reduce(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(mut g1, kg1, l1, c1), (g2, kg2, l2, c2)| {
                    add_slice(&mut g1, &g2);
                    (g1, kg1 + kg2, l1 + l2, c1 + c2)
                },
            )
    }

    pub(crate) fn val_eval<const N: usize>(&self, values: &[f64], probes: [(f64, f64); N]) -> [f64; N] {
        self.split_eval(self.val, self.train.len(), values, probes)
    }

    pub(crate) fn k_fit_eval(&self, values: &[f64], k: f64, blend: f64) -> f64 {
        if self.val.is_empty() {
            let probe = &self.train[..self.train.len().min(K_FIT_PROBE)];
            return self.split_eval(probe, 0, values, [(k, blend)])[0];
        }
        self.val_eval(values, [(k, blend)])[0]
    }

    fn split_eval<const N: usize>(
        &self,
        records: &[FeatureRecord],
        offset: usize,
        values: &[f64],
        probes: [(f64, f64); N],
    ) -> [f64; N] {
        let (weighted_loss_sums, total_weight) = records
            .par_iter()
            .enumerate()
            .fold(
                || ([0.0_f64; N], 0.0_f64),
                |(mut sums, mut weight_acc), (idx, record)| {
                    if !self.passes_volatility_filter(record) {
                        return (sums, weight_acc);
                    }

                    let score = eval_record(record, values);
                    let w = if self.sample_weights.is_empty() { 1.0 } else { self.sample_weights[offset + idx] };

                    for (sum, &(k, blend)) in sums.iter_mut().zip(&probes) {
                        let p = sigmoid(score, k);
                        let (target, _) = wdl_target(record, k, blend);
                        *sum += w * self.loss_fn.loss(p, target);
                    }

                    weight_acc += w;
                    (sums, weight_acc)
                },
            )
            .reduce(
                || ([0.0_f64; N], 0.0_f64),
                |(mut sums1, w1), (sums2, w2)| {
                    add_slice(&mut sums1, &sums2);
                    (sums1, w1 + w2)
                },
            );

        if total_weight > 0.0 { weighted_loss_sums.map(|sum| sum / total_weight) } else { [0.0; N] }
    }
}

pub(crate) fn artifact(name: &str) -> String {
    match env::var_os(ARTIFACT_DIR) {
        Some(dir) => PathBuf::from(dir).join(name).display().to_string(),
        None => name.to_string(),
    }
}

pub(crate) fn replay_filter(config: &EvalTuneConfig) -> ReplayFilter {
    let Some(path) = config.replay_filter.as_deref() else {
        return ReplayFilter::UNRESTRICTED;
    };
    let text = fs::read_to_string(path).unwrap_or_else(|e| {
        alarm!("Cannot read replay filter {path}: {e}");
        process::exit(1);
    });
    let filter: ReplayFilter = toml::from_str(&text).unwrap_or_else(|e| {
        alarm!("Cannot parse replay filter {path}: {e}");
        process::exit(1);
    });
    println!("Replay filter: {path}");
    filter
}

pub(crate) fn seed_values(params: &[Tunable], init: Init, seed: u64) -> Vec<f64> {
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

pub(crate) fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>, task: Task) -> bool {
    let total_start = Instant::now();
    let effective_dataset = match (dataset_path, resume_path) {
        (Some(path), _) => path.to_string(),
        (None, Some(checkpoint)) => peek_checkpoint(checkpoint)
            .ok()
            .map(|cp| cp.dataset_path)
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "default".to_string()),
        (None, None) => "default".to_string(),
    };

    let Some(paths) = resolve_dataset_paths(&effective_dataset) else {
        alarm!("Error: No dataset found. Use --dataset <path> or place .vf files in data/.");
        return false;
    };

    let filter = replay_filter(config);
    let is_replay_dataset = paths.iter().any(|p| is_viri_path(p));
    if is_replay_dataset && config.replay_filter.is_none() {
        alarm!("No replay_filter named: every replayed position trains, tactical and in-check ones included.");
    }

    let loaded = load_datasets(&paths, &filter);
    if loaded.entries.is_empty() {
        eprintln!("Error: No positions loaded.");
        return false;
    }

    let dataset_label = paths.join(", ");
    let keep_fraction = epoch_keep_fraction(config, &filter, is_replay_dataset);
    let trained = train_entries(loaded, &dataset_label, config, resume_path, task, keep_fraction);
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n{BRAND}Done in {elapsed:.2}s{RESET}");
    trained
}

#[derive(Clone, Copy)]
struct Seeds {
    rng_seed: u64,
    split_seed: u64,
}

fn shuffle_groups(
    mut entries: Vec<SoulEntry>,
    mut weights: Vec<f32>,
    groups: &[u32],
    seed: u64,
) -> (Vec<SoulEntry>, Vec<f32>, Vec<u32>) {
    if groups.len() == entries.len() {
        fastrand::Rng::with_seed(seed).shuffle(&mut entries);
        fastrand::Rng::with_seed(seed).shuffle(&mut weights);
        return (entries, weights, groups.to_vec());
    }

    let mut starts = Vec::with_capacity(groups.len());
    let mut offset = 0usize;

    for &size in groups {
        starts.push(offset);
        offset += size as usize;
    }
    debug_assert_eq!(offset, entries.len(), "the loader left positions outside every group");

    let mut order: Vec<u32> = (0..groups.len() as u32).collect();
    fastrand::Rng::with_seed(seed).shuffle(&mut order);

    let mut shuffled_entries = Vec::with_capacity(entries.len());
    let mut shuffled_weights = Vec::with_capacity(weights.len());
    let mut shuffled_sizes = Vec::with_capacity(groups.len());

    for &group_idx in &order {
        let start = starts[group_idx as usize];
        let size = groups[group_idx as usize] as usize;

        shuffled_entries.extend_from_slice(&entries[start..start + size]);
        if !weights.is_empty() {
            shuffled_weights.extend_from_slice(&weights[start..start + size]);
        }
        shuffled_sizes.push(size as u32);
    }

    (shuffled_entries, shuffled_weights, shuffled_sizes)
}

// Takes whole groups off the tail while always leaving at least one group to train on.
fn holdout(sizes: &[u32], total: usize, target: usize) -> (usize, usize) {
    let mut positions = 0usize;
    let mut groups = 0usize;

    for &size in sizes.iter().rev() {
        let taken = positions + size as usize;
        if positions >= target || taken >= total {
            break;
        }

        if taken > target && taken - target > target - positions {
            break;
        }

        positions = taken;
        groups += 1;
    }

    (positions, groups)
}

fn epoch_keep_fraction(config: &EvalTuneConfig, filter: &ReplayFilter, is_replay_dataset: bool) -> f64 {
    if let Some(sample) = config.epoch_sample {
        return sample.clamp(0.0, 1.0);
    }
    if is_replay_dataset && filter.random_fen_skipping {
        1.0 - filter.random_fen_skip_probability.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

fn epoch_sample_len(train_len: usize, keep: f64) -> usize {
    if train_len == 0 {
        return 0;
    }
    ((train_len as f64 * keep).round() as usize).clamp(1, train_len)
}

fn train_entries(
    loaded: LoadedDataset,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    task: Task,
    epoch_keep: f64,
) -> bool {
    let LoadedDataset { entries, sample_weights, group_sizes } = loaded;
    let dataset_fnv = dataset_fingerprint(&entries);
    let (rng_seed, split_seed) = match resume_path {
        Some(path) => {
            let cp = peek_checkpoint(path).unwrap_or_else(|e| {
                eprintln!("Failed to read checkpoint: {e}");
                process::exit(1);
            });

            if let Some(seed) = config.seed
                && seed != cp.rng_seed
            {
                println!("--seed {seed} ignored: resume reuses checkpoint batch-order seed {}", cp.rng_seed);
            }

            if cp.dataset != dataset_fnv {
                alarm!(
                    "Warning: dataset does not match the checkpoint's fingerprint.\n\
                     [!] The train/val split will differ from the original run: positions the\n\
                     [!] checkpoint trained on may now sit in val, making its loss optimistic."
                );
            }
            (cp.rng_seed, cp.split_seed.unwrap_or(cp.rng_seed))
        },
        None => (config.seed.unwrap_or_else(|| fastrand::u64(..)), config.split_seed.unwrap_or(VAL_SPLIT_SEED)),
    };

    let (entries, sample_weights, sizes) = shuffle_groups(entries, sample_weights, &group_sizes, split_seed);
    let total = entries.len();

    println!("Extracting features ({total} entries)...");
    let records: Vec<FeatureRecord> = entries.par_iter().map(FeatureRecord::from_entry).collect();

    let target = (entries.len() / 10).min(config.val_max.unwrap_or(usize::MAX));
    let (mut val_count, val_groups) = holdout(&sizes, entries.len(), target);

    if val_groups < val_count {
        println!("Holding out {val_groups} whole games ({val_count} positions)");
    }

    if val_count == 0 && target > 0 {
        alarm!(
            "No game can leave without taking the file with it. Falling back to a position cut,\n\
             [!] so train and validation will share games and L_val will read low."
        );
        val_count = target;
    }
    if val_count == 0 {
        alarm!(
            "No validation split: {} positions cannot spare a tenth (or val_max = 0 requested none).\n\
             [!] The run trains and reports train loss only; nothing will rank its outcome,\n\
             [!] and K is fitted on a sample of the positions it trains on.",
            entries.len(),
        );
    }

    let train_count = total - val_count;
    if val_count * 2 < target {
        alarm!("The holdout came to {val_count} positions against a target of {target}: too few games to cut a tenth.");
    }

    let (entry_train, entry_val) = entries.split_at(train_count);
    print_dataset_stats(entry_train, entry_val, total, |e: &SoulEntry| flip_wdl(f64::from(e.result) / 2.0, e.stm()));
    drop(entries);

    let (train, val) = records.split_at(train_count);
    let balance_weights = if config.phase_balance {
        phase_balance_weights(&records, config.phase_balance_cap, config.phase_target.as_deref())
    } else {
        Vec::new()
    };
    let sample_weights = merge_sample_weights(balance_weights, &sample_weights);
    let obj = Objective {
        train,
        val,
        sample_weights: &sample_weights,
        loss_fn: config.loss,
        volatility_threshold: config.volatility_threshold,
        volatility_adaptive: config.volatility_adaptive,
    };

    match task {
        Task::GatherCost => {
            gather_cost(&obj, config);
            return false;
        },
        Task::Curvature => {
            curvature_report(&obj, config);
            return false;
        },
        Task::ValCost => {
            val_cost(&obj, config);
            return false;
        },
        Task::BatchSize => {
            batch_size_report(&obj, config);
            return false;
        },
        Task::Momentum => {
            momentum_report(&obj, config);
            return false;
        },
        Task::Train => {},
    }

    let seeds = Seeds { rng_seed, split_seed };
    train_loop(epoch_keep, dataset_label, config, resume_path, seeds, dataset_fnv, &obj);
    true
}

fn format_labeled_value(text: &str) -> String {
    if let Some(inner) = text.strip_suffix(')')
        && let Some((head, detail)) = inner.split_once(" (")
    {
        format!("{VAL}{head}{RESET} ({detail})")
    } else {
        format!("{VAL}{text}{RESET}")
    }
}

fn train_loop(
    epoch_keep: f64,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    seeds: Seeds,
    dataset_fnv: u64,
    obj: &Objective,
) {
    let Seeds { rng_seed, split_seed } = seeds;
    let train_len = obj.train.len();

    let has_val = !obj.val.is_empty();
    let params = eval_params::collect_parameters();
    let defaults = eval_params::default_values(&params);
    let slots = params.len();

    let resume = resume_path.map(|path| {
        println!("Resuming from checkpoint: {VAL}{path}{RESET}");
        let data = load_checkpoint(path, &params, &defaults).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {e}");
            process::exit(1);
        });
        if data.fresh_params > 0 {
            println!("{VAL}{}{RESET} parameters are newer than the checkpoint, starting from code defaults", data.fresh_params);
        }
        data
    });

    let (start_epoch, mut lr_scale) = resume.as_ref().map_or((1, 1.0), |d| (d.run.epoch, d.run.lr_scale));
    let mut values = match resume.as_ref() {
        Some(d) => d.values.clone(),
        None => seed_values(&params, config.init, rng_seed),
    };

    let lr_schedule = config.lr_schedule;
    let wdl_schedule = config.wdl_schedule;
    let is_constant_schedule = matches!(lr_schedule, LrSchedule::Constant { .. });
    let init_blend = wdl_schedule.blend(1, config.epochs);

    let mut k_ctrl = KController::bootstrap(config, obj, &values, &defaults, init_blend, resume.as_ref());
    let k = k_ctrl.k();
    let win_rate_100cp = sigmoid(100.0, k);

    println!("{LAB}K Factor:{RESET}   {VAL}{k:.6}{RESET} (100cp -> {:.1}%)", win_rate_100cp * 100.0);
    println!("{LAB}K Mode:{RESET}     {}", format_labeled_value(&config.k_mode.to_string()));

    let seed_label = if resume.is_some() {
        " (checkpoint)"
    } else if config.seed.is_some() {
        " (deterministic)"
    } else {
        ""
    };
    println!("{LAB}Seed:{RESET}       {VAL}{rng_seed}{RESET}{seed_label}");
    if split_seed != VAL_SPLIT_SEED {
        println!("{LAB}Split seed:{RESET} {VAL}{split_seed}{RESET} (L_val not comparable to default-split runs)");
    }

    if config.init != Init::Default && resume.is_none() {
        println!(
            "{LAB}Init:{RESET}       {VAL}{:?}{RESET} (cold start; K meaningless until material scale emerges)",
            config.init
        );
    }

    let initial_values = values.clone();
    let stride = (train_len / GAUGE_PROBE).max(1);
    let probe: Vec<&FeatureRecord> = obj.train.iter().step_by(stride).take(GAUGE_PROBE).collect();
    let mut gauge = Gauge::new(probe, &defaults);
    let hold_scale = gauge.holds(&values);

    let mut masks = slot_masks(&params, config);
    let mut grad_ema = resume.as_ref().map_or_else(|| vec![0.0_f64; slots], |d| d.grad_ema.clone());
    let mut stagnant_counts = resume.as_ref().map_or_else(|| vec![0usize; slots], |d| d.stagnant.clone());

    println!("{LAB}Parameters:{RESET} {VAL}{slots}{RESET}");
    println!("{LAB}Mode:{RESET}       {VAL}SoulEntry{RESET}");
    println!("{LAB}LR Sched:{RESET}   {}", format_labeled_value(&lr_schedule.describe()));
    println!("{LAB}WDL Sched:{RESET}  {}", format_labeled_value(&wdl_schedule.describe()));
    println!("{LAB}Beta2:{RESET}      {}", format_labeled_value(&describe_beta2(config)));
    println!("{LAB}Optimizer:{RESET}  {VAL}Lion{RESET} (Batch: {}, WD: {})", config.batch_size, config.weight_decay);

    let log_file = fs::OpenOptions::new().create(true).append(true).open(artifact("evaltune_log.txt")).ok();
    let mut text_log = log_file.map(BufWriter::new);

    if let Some(ref mut w) = text_log {
        writeln!(w).ok();
        writeln!(w, "Seed:      {rng_seed}").ok();
        if split_seed != VAL_SPLIT_SEED {
            writeln!(w, "Split:     {split_seed}").ok();
        }
        writeln!(w, "Mode:      SoulEntry").ok();
        writeln!(w, "Dataset:   {dataset_label}").ok();
        writeln!(w, "K:         {k:.6} (100cp → {:.1}%)", win_rate_100cp * 100.0).ok();
        writeln!(w, "K mode:    {}", config.k_mode).ok();
        writeln!(w, "Epochs:    {}", config.epochs).ok();
        writeln!(w, "Params:    {slots}").ok();
        writeln!(w, "LR:        {}", lr_schedule.describe()).ok();
        writeln!(w, "WDL:       {}", wdl_schedule.describe()).ok();
        writeln!(w, "Beta2:     {}", describe_beta2(config)).ok();
        writeln!(w, "Optimizer: Lion (batch: {}, WD: {})", config.batch_size, config.weight_decay).ok();
        writeln!(w).ok();
        writeln!(w, "{}", EpochStats::log_header()).ok();
    }

    let mut json_log = JsonLog::open(&config.log_path)
        .inspect_err(|e| alarm!("Cannot open the run log at {}: {e}", config.log_path))
        .ok();

    if let Some(ref mut l) = json_log {
        l.write(
            "run",
            &serde_json::json!({
                "seed": rng_seed,
                "split_seed": split_seed,
                "epochs": config.epochs,
                "start_epoch": start_epoch,
                "dataset": dataset_label,
                "init": format!("{:?}", config.init),
                "params": slots,
            }),
        );
    }

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    let mut optimizer = Lion::new(slots, config.beta1, lr_schedule.rate(start_epoch, config.epochs), config.weight_decay);

    if let Some(d) = &resume {
        optimizer.restore_momentum(&d.momentum);
    }

    let mut grad_stats = GradientStats::new(100);
    // Gathering through shuffled indices costs 12% of epoch time at 32.8M positions.
    let mut indices = vec![0u32; train_len];
    let mut shuffler = Shuffler::new(train_len);

    let epoch_sample = epoch_sample_len(train_len, epoch_keep);
    let steps_per_epoch = epoch_sample.div_ceil(config.batch_size);

    if epoch_sample < train_len {
        println!("Sampling {epoch_sample} of {train_len} training positions per epoch, redrawn each one");
    }

    println!(
        "{LAB}Steps:{RESET}      {VAL}{}{RESET} over {} epochs, {steps_per_epoch} per epoch",
        steps_per_epoch * config.epochs,
        config.epochs,
    );

    let mut ema_values = resume.as_ref().map_or_else(|| values.clone(), |d| d.ema.clone());
    let lr_peak = (1..=config.epochs).fold(0.0f64, |m, e| m.max(lr_schedule.rate(e, config.epochs)));

    // Constant schedule has no tail → uniform Polyak instead of tail EMA.
    let mut ema_active = is_constant_schedule;
    let ema_threshold = if is_constant_schedule { 0.0 } else { 0.3 * lr_peak };

    let mut progress = resume.as_ref().map_or_else(Progress::default, |d| d.run.progress.clone());
    let mut best_val_params = resume.as_ref().map_or_else(|| vec![0.0; slots], |d| d.best_val_params.clone());
    let mut best_train_params = resume.as_ref().map_or_else(|| vec![0.0; slots], |d| d.best_train_params.clone());
    let mut val_history: Vec<f64> = Vec::new();
    let mut train_history: Vec<f64> = Vec::new();
    let mut prev_val_loss = f64::NAN;
    let mut divergence = DivergenceMonitor::default();
    let mut prev_quantized = vec![0i32; slots];
    let mut quantized = vec![0i32; slots];

    quantize(&ema_values, &mut prev_quantized);

    let mut epochs_run = 0usize;
    let mut batches_run = 0u64;
    let mut clipped_seen = 0u64;
    let mut epoch_seconds = 0.0f64;
    let mut grad_seconds = 0.0f64;
    let mut shuffle_seconds = 0.0f64;
    let mut val_seconds = 0.0f64;
    let mut epoch_positions = 0u64;

    let group_ranges = group_ranges(slots);
    let mut run_tally = [GateTally::default(); GROUP_NAMES.len()];

    if let Some(d) = &resume {
        masks.fixed.copy_from_slice(&d.frozen);
    } else if config.unfreeze_epoch > 0 {
        let base_end = group_ranges[1].end;
        masks.fixed[base_end..].fill(true);
        println!("Progressive unfreeze: params {base_end}+ frozen until epoch {}", config.unfreeze_epoch);
    }

    for epoch in start_epoch..=config.epochs {
        let t0 = Instant::now();
        let blend = wdl_schedule.blend(epoch, config.epochs);
        let mut epoch_tally = [GateTally::default(); GROUP_NAMES.len()];

        if let Some(new_k) = k_ctrl.on_epoch(epoch, obj, &ema_values, blend) {
            println!("  Reoptimized K: {new_k:.6}");
            if let Some(ref mut w) = text_log {
                writeln!(w, "# K re-opt @ epoch {epoch}: {new_k:.6}").ok();
            }
        }

        let lr = (lr_schedule.rate(epoch, config.epochs) * lr_scale).max(0.00001);
        if !ema_active && lr < ema_threshold {
            println!("  EMA activated at epoch {epoch} (lr {lr:.6} < {ema_threshold:.6})");
            ema_active = true;
        }

        let is_restart = lr_schedule.is_restart(epoch, config.epochs);
        optimizer.set_lr(lr);

        if config.beta2_tracks_lr {
            masks.beta2.fill(beta2_for_lr(config.beta2_lr_coefficient, lr));
        }

        if is_restart {
            progress.plateau_count = 0;
            ema_active = is_constant_schedule;
        }

        let t_shuffle = Instant::now();

        if config.shuffle_block > 0 {
            shuffler.fill_blocked(&mut indices, rng.u64(..), config.shuffle_block, epoch_sample);
        } else {
            shuffler.fill(&mut indices, rng.u64(..));
        }

        let shuffle_secs = t_shuffle.elapsed().as_secs_f32();

        let mut train_loss = 0.0;
        let mut train_count = 0usize;
        let mut epoch_grads = vec![0.0; slots];
        let t_grad = Instant::now();

        for batch in indices[..epoch_sample].chunks(config.batch_size) {
            batches_run += 1;

            let (mut grads, k_grad, batch_loss, batch_count) = obj.batch_grad(batch, &values, k_ctrl.k(), blend);
            train_loss += batch_loss;
            train_count += batch_count;

            let n = batch_count.max(1) as f64;
            let norm = l2_norm(&grads);
            grad_stats.update(norm / n);

            let threshold = grad_stats.clip_threshold(config.grad_clip) * n;
            let clip_scale = if norm > threshold { threshold / norm } else { 1.0 };
            for (i, g) in grads.iter_mut().enumerate() {
                *g = *g / n * clip_scale;
                epoch_grads[i] += *g;
            }

            // Before the update, while momentum still holds what the gate is about to read.
            if config.gate_tally {
                for (tally, range) in epoch_tally.iter_mut().zip(&group_ranges) {
                    tally.absorb(optimizer.tally(range.clone(), &grads, &masks.fixed));
                }
            }

            optimizer.update(&mut values, &grads, &masks);
            k_ctrl.on_batch(k_grad, batch_count, lr, clip_scale, config.weight_decay);
            canonicalize(&mut values, &params);

            if hold_scale {
                gauge.restore(&mut values, &mut optimizer, &mut k_ctrl);
            }

            for i in 0..slots {
                if !masks.fixed[i] {
                    grad_ema[i] = 0.99_f64.mul_add(grad_ema[i], 0.01 * grads[i].abs());
                }
            }

            // Skip the noisy high-LR phase; only average once LR has decayed below 30% of its
            // peak. Before that, snapshot the live weights directly.
            if ema_active {
                for i in 0..slots {
                    ema_values[i] = config.ema_decay.mul_add(ema_values[i], (1.0 - config.ema_decay) * values[i]);
                }
            } else {
                ema_values.copy_from_slice(&values);
            }
        }

        let grad_secs = t_grad.elapsed().as_secs_f32();
        let step_l1 = optimizer.take_step_l1();
        let clipped_total: u64 = optimizer.clipped().iter().sum();
        let clipped_epoch = clipped_total - clipped_seen;
        clipped_seen = clipped_total;

        if config.unfreeze_epoch > 0 && epoch == config.unfreeze_epoch {
            for (i, p) in params.iter().enumerate() {
                masks.fixed[i] = p.is_fixed;
            }
            println!("  Unfrozen all remaining parameters at epoch {epoch}");
        }

        if config.auto_freeze
            && epoch > config.freeze_start_epoch
            && config.freeze_cadence > 0
            && epoch.is_multiple_of(config.freeze_cadence)
        {
            let mut frozen = 0;
            for i in 0..slots {
                if !masks.fixed[i] && grad_ema[i] < config.freeze_threshold && !params[i].freeze_resistant {
                    stagnant_counts[i] += 1;
                    if stagnant_counts[i] >= config.freeze_consecutive {
                        masks.fixed[i] = true;
                        frozen += 1;
                    }
                } else {
                    stagnant_counts[i] = 0;
                }
            }
            if frozen > 0 {
                println!("  Auto-frozen {frozen} stagnant parameters.");
            }
        }

        let t_val = Instant::now();
        let [val_loss, ref_loss] = if has_val {
            obj.val_eval(&ema_values, [(k_ctrl.k(), blend), (k_ctrl.k_ref(), 0.0)])
        } else {
            [f64::NAN, f64::NAN]
        };
        let val_secs = t_val.elapsed().as_secs_f32();
        let train_loss = train_loss / train_count.max(1) as f64;
        let improved_train = progress.record_train(epoch, train_loss);

        if improved_train {
            best_train_params.copy_from_slice(&ema_values);
        }

        let is_best = if has_val {
            let improved = progress.record_val(epoch, val_loss);
            if improved {
                best_val_params.copy_from_slice(&ema_values);
            } else if is_constant_schedule && progress.plateau_count >= config.patience {
                lr_scale *= 0.5;
                progress.plateau_count = 0;
                println!("  Plateau detected, LR scale → {lr_scale:.3}");
            }
            improved
        } else {
            if improved_train {
                best_val_params.copy_from_slice(&ema_values);
            }
            improved_train
        };

        let overfit = has_val && divergence.update(train_loss, val_loss);
        let drifted = val_history.first().is_some_and(|&opening| progress.val_smooth > opening);

        quantize(&ema_values, &mut quantized);
        let moved = quantized.iter().zip(&prev_quantized).filter(|(q, p)| q != p).count();
        prev_quantized.copy_from_slice(&quantized);

        let stats = EpochStats {
            epoch,
            train_loss,
            val_loss,
            ref_loss,
            lr,
            is_best,
            psqt_norm: l2_norm(&epoch_grads[group_ranges[0].clone()]),
            mobility_norm: l2_norm(&epoch_grads[group_ranges[2].clone()]),
            overfit,
            drifted,
            moved,
            gauge: gauge.applied,
            step_l1,
            clipped: clipped_epoch,
        };

        if let Some(ref mut w) = text_log {
            writeln!(w, "{}", stats.log_row()).ok();
        }
        if let Some(ref mut l) = json_log {
            if is_restart {
                l.write("restart", &serde_json::json!({ "epoch": epoch }));
            }
            l.write("epoch", &stats);
        }

        let elapsed = t0.elapsed().as_secs_f32();
        let mpos = train_count as f32 / grad_secs.max(1e-6) / 1e6;

        epochs_run += 1;
        epoch_seconds += f64::from(elapsed);
        grad_seconds += f64::from(grad_secs);
        shuffle_seconds += f64::from(shuffle_secs);
        val_seconds += f64::from(val_secs);
        epoch_positions += train_count as u64;

        println!("{}", stats.status_line(config.epochs, prev_val_loss, elapsed, mpos));

        if config.gate_tally {
            let mut total_tally = GateTally::default();
            for (total, epoch_stat) in run_tally.iter_mut().zip(&epoch_tally) {
                total.absorb(*epoch_stat);
                total_tally.absorb(*epoch_stat);
            }
            println!("  {LAB}gate{RESET}      {}  {DIM}step{RESET} {VAL}{step_l1:.1}{RESET}", gate_columns(&total_tally));
        }

        if has_val {
            val_history.push(val_loss);
            prev_val_loss = val_loss;
        }

        train_history.push(train_loss);

        if epoch.is_multiple_of(20) || epoch == config.epochs {
            print_loss_history(&val_history, &train_history);
            if epoch != config.epochs {
                print_params(&params, &initial_values, &ema_values);
            }

            if let Err(e) = save_checkpoint(&artifact("evaltune_checkpoint.json"), &params, &TrainerState {
                run: RunState {
                    epoch: epoch + 1,
                    lr_scale,
                    k: k_ctrl.k(),
                    k_ref: k_ctrl.k_ref(),
                    k_momentum: k_ctrl.momentum,
                    progress: progress.clone(),
                },
                rng_seed,
                split_seed,
                dataset: dataset_fnv,
                dataset_path: dataset_label,
                values: &values,
                momentum: optimizer.momentum(),
                ema: &ema_values,
                grad_ema: &grad_ema,
                stagnant: &stagnant_counts,
                frozen: &masks.fixed,
                best_val_params: &best_val_params,
                best_train_params: &best_train_params,
            }) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }
    }

    let off_scale_ratio = Gauge::measure(&gauge.probe, &ema_values) / gauge.reference;
    let (gauge_scale, report_k) = if hold_scale {
        (1.0 / gauge.applied, k_ctrl.k())
    } else {
        gauge.normalize(&mut ema_values);
        gauge.normalize(&mut best_train_params);
        let factor = gauge.normalize(&mut best_val_params);
        (1.0 / factor, k_ctrl.k() / factor)
    };

    if let Some(ref mut l) = json_log {
        quantize(&best_val_params, &mut quantized);
        l.write(
            "final",
            &serde_json::json!({
                "seed": rng_seed,
                "split_seed": split_seed,
                "epochs": epochs_run,
                "steps": batches_run,
                "best_val_loss": if has_val { serde_json::Value::from(progress.best_val_loss) } else { serde_json::Value::Null },
                "best_val_epoch": if has_val { serde_json::Value::from(progress.best_val_epoch) } else { serde_json::Value::Null },
                "best_train_loss": progress.best_train_loss,
                "best_train_epoch": progress.best_train_epoch,
                "best_val_params": quantized,
                "sensitivity": grad_ema,
            }),
        );
    }

    let how = if hold_scale { "held through the run" } else { "normalized on the way out" };
    let gauge_line = format!("\n{LAB}Gauge:{RESET}      {VAL}{gauge_scale:.3}×{RESET} pull on the eval's scale, {how}\n");
    let off_scale = off_scale_warning(off_scale_ratio);
    let clamped_k = clamped_k_warning(config, k_ctrl.k());
    let calibration = if has_val { calibration_report(obj, &best_val_params, report_k) } else { String::new() };
    let tally = if config.gate_tally { gate_tally_report(&run_tally) } else { String::new() };
    let clip = clip_report(&params, optimizer.clipped(), batches_run);

    print!("{gauge_line}");
    eprint!("{off_scale}{clamped_k}");
    print!("{calibration}{tally}{clip}");

    if let Some(ref mut w) = text_log {
        for part in [&gauge_line, &off_scale, &clamped_k, &calibration, &tally, &clip] {
            write!(w, "{}", color::strip(part)).ok();
        }
    }

    drop(text_log);
    sensitivity_report(&params, &grad_ema, &masks.fixed);

    print_results(
        &params,
        &initial_values,
        &ema_values,
        &BestSnapshots {
            best_val_params: &best_val_params,
            best_val_loss: if has_val { Some(progress.best_val_loss) } else { None },
            best_val_epoch: progress.best_val_epoch,
            best_train_params: &best_train_params,
            best_train_loss: progress.best_train_loss,
            best_train_epoch: progress.best_train_epoch,
            last_val: val_history.last().copied(),
            last_train: train_history.last().copied(),
        },
        config.epochs,
    );

    if epochs_run > 0 {
        let avg_mpos = epoch_positions as f64 / grad_seconds.max(1e-6) / 1e6;
        let rest_seconds = epoch_seconds - grad_seconds - shuffle_seconds - val_seconds;
        println!(
            "\n{LAB}Trained{RESET} {epochs_run} epochs in {epoch_seconds:.2}s  \
             {DIM}grad {grad_seconds:.2}s · shuffle {shuffle_seconds:.2}s · val {val_seconds:.2}s · rest {rest_seconds:.2}s{RESET}  \
             {avg_mpos:.1}M pos/s"
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::TunerConfig,
        engine::{LAYOUT, Position},
    };

    fn vol_obj(threshold: i16, adaptive: bool) -> Objective<'static> {
        Objective {
            train: &[],
            val: &[],
            sample_weights: &[],
            loss_fn: LossFn::CrossEntropy,
            volatility_threshold: threshold,
            volatility_adaptive: adaptive,
        }
    }

    #[test]
    fn holdout_takes_whole_groups_nearest_target() {
        for (sizes, total, target, want) in [
            (&[100u32, 80, 40][..], 220, 100, (120, 2)),
            (&[100, 200, 40], 340, 100, (40, 1)),
            (&[1; 100], 100, 10, (10, 10)),
            (&[95, 5], 100, 10, (5, 1)),
            (&[18, 6], 24, 3, (6, 1)),
        ] {
            assert_eq!(holdout(sizes, total, target), want);
        }
    }

    #[test]
    fn merged_weights_preserve_ratios_and_unit_mean() {
        let sample = vec![0.25f32, 0.75, 1.0, 0.5];
        let merged = merge_sample_weights(Vec::new(), &sample);
        let mean = merged.iter().sum::<f64>() / merged.len() as f64;
        assert!((mean - 1.0).abs() < 1e-12);
        assert!((merged[1] / merged[0] - 3.0).abs() < 1e-12);

        let phase = vec![2.0, 2.0, 1.0, 1.0];
        let both = merge_sample_weights(phase.clone(), &sample);
        assert!((both.iter().sum::<f64>() / both.len() as f64 - 1.0).abs() < 1e-12);
        assert_eq!(merge_sample_weights(phase.clone(), &[]), phase);
        assert!(merge_sample_weights(Vec::new(), &[]).is_empty());
    }

    #[test]
    fn decimation_thins_epoch_without_emptying_it() {
        let mut config = TunerConfig::default().evaltune;
        let ninety = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };
        let off = ReplayFilter { random_fen_skipping: false, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };
        assert!((epoch_keep_fraction(&config, &ninety, true) - 0.1).abs() < 1e-12);
        assert!((epoch_keep_fraction(&config, &off, true) - 1.0).abs() < 1e-12);
        assert!((epoch_keep_fraction(&config, &ninety, false) - 1.0).abs() < 1e-12);
        config.epoch_sample = Some(0.25);
        for replays in [true, false] {
            assert!((epoch_keep_fraction(&config, &ninety, replays) - 0.25).abs() < 1e-12);
        }
        assert_eq!(epoch_sample_len(32_800_000, 0.1), 3_280_000);
        assert_eq!(epoch_sample_len(32_800_000, 1.0), 32_800_000);
        assert_eq!(epoch_sample_len(4, 0.0), 1);
        assert_eq!(epoch_sample_len(0, 0.5), 0);
    }

    #[test]
    fn replay_filter_toml_parses_shipped_and_partial_files() {
        let text = fs::read_to_string("replay_filter.toml").expect("replay_filter.toml must exist");
        let shipped: toml::Table = toml::from_str(&text).unwrap();
        let known: toml::Table = toml::from_str(&toml::to_string(&ReplayFilter::default()).unwrap()).unwrap();
        let _: ReplayFilter = toml::from_str(&text).unwrap();
        for key in shipped.keys() {
            assert!(known.contains_key(key), "unknown ReplayFilter field {key}");
        }

        let partial: ReplayFilter = toml::from_str("min_ply = 24\nfilter_castling = true\n").unwrap();
        assert_eq!(partial.min_ply, 24);
        assert!(partial.filter_castling);
        assert_eq!(partial.min_pieces, 4);
        assert!(partial.filter_tactical);
    }

    #[test]
    fn volatility_filter_bounds() {
        for (thresh, adaptive, pieces, score, static_eval, want) in [
            (40i16, true, 32u8, 0i16, 80i16, true),
            (40, true, 32, 0, 100, false),
            (40, true, 31, 0, 80, true),
            (40, true, 31, 0, 100, false),
            (40, true, 8, 0, 30, true),
            (40, true, 8, 0, 50, false),
            (40, false, 32, 0, 40, true),
            (40, false, 32, 0, 41, false),
            (8, true, 6, 0, 0, true),
            (8, true, 6, 0, 1, false),
            (i16::MAX, true, 2, 0, 30_000, true),
            (i16::MAX, true, 32, 0, 30_000, true),
            (i16::MAX, true, 32, -1, 32_767, false),
        ] {
            let mut record = FeatureRecord::default();
            record.piece_count = pieces;
            record.score = score;
            record.static_eval = static_eval;
            assert_eq!(
                vol_obj(thresh, adaptive).passes_volatility_filter(&record),
                want,
                "thresh={thresh} adaptive={adaptive} pieces={pieces} score={score} static={static_eval}"
            );
        }
    }

    #[test]
    fn k_gradient_matches_batch_loss_finite_difference() {
        let params = eval_params::collect_parameters();
        let values = eval_params::default_values(&params);
        let entry = SoulEntry::from_board(
            &Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            1.0,
            Some(55),
        );
        let record = FeatureRecord::from_entry(&entry);
        let obj = Objective {
            train: std::slice::from_ref(&record),
            val: &[],
            sample_weights: &[],
            loss_fn: LossFn::CrossEntropy,
            volatility_threshold: 0,
            volatility_adaptive: true,
        };
        let (k, blend, h) = (0.005, 0.3, 1e-6);
        let (_, k_g, ..) = obj.batch_grad(&[0], &values, k, blend);
        let loss_at = |kk: f64| {
            let (target, _) = wdl_target(&record, kk, blend);
            obj.loss_fn.loss(sigmoid(eval_record_full(&record, &values).score, kk), target)
        };
        let numeric = (loss_at(k + h) - loss_at(k - h)) / (2.0 * h);
        assert!((k_g - numeric).abs() < 1e-6 * (1.0 + numeric.abs()), "k_g {k_g} vs FD {numeric}");
    }

    #[test]
    fn cold_start_preserves_fixed_slots() {
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
        assert!(zeroed[phase..phase + LAYOUT.phase_len].iter().any(|w| *w > 0.0));
    }
}
