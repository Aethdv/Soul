//! HCE training: gradient descent on Soul's weights against WDL-labeled positions.
//!
//! Features are extracted to `FeatureRecord`s once at startup; epochs are sequential
//! reads over the cached SoA path (`eval_record` + `accumulate_record_grad`, not the
//! dual-number oracle the tuner tests use). An initial golden-section search fixes
//! the sigmoid's centipawn→winrate scale; `k_ref` stays frozen so the reference loss
//! is comparable across runs.
//!
//! The split is cut on whole games, so a held-out ply cannot sit one move from its
//! own siblings in train. Best-loss captures track validation and training separately,
//! so both parameter vectors are there to inspect.

use std::{
    fs,
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use palette::{DIM, LAB, RESET, VAL};
use rayon::prelude::*;

use crate::{
    alarm,
    config::{EvalTuneConfig, Init, LossFn, LrScheduleConfig, RANDOM_INIT_SPREAD},
    engine::{
        Color, FeatureRecord, LAYOUT, ReplayFilter, SoulEntry, Tunable, accumulate_record_grad, color, eval_params, eval_record,
        eval_record_full, flip_wdl,
    },
    groups::{GROUP_NAMES, build_masks, group_ranges},
    lion::{GateCensus, Lion},
    loader::{dataset_fingerprint, load_datasets, resolve_dataset_paths},
    logger::JsonLogger,
    palette,
    probes::{batch_size_report, curvature_report, gather_cost, momentum_report, val_cost},
    report::*,
    scale::{GAUGE_PROBE, Gauge, KController, canonicalize},
    shuffle::Shuffler,
    storage::*,
    training::*,
};

/// Fixes which tenth of a dataset is held out.
const VAL_SPLIT_SEED: u64 = 0x5350_4C49_5432_3736;

/// Positions a K fit reads when the run holds nothing out.
/// Bounded because the golden search reads it some thirty times.
const K_FIT_PROBE: usize = 65_536;

pub const ARTIFACT_DIR: &str = "EVALTUNE_ARTIFACT_DIR";

/// What a loaded dataset is for.
///
/// Both arms want every step up to `TrainerContext`, feature extraction included, so the choice
/// travels through the loader rather than growing a second copy of it.
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
    /// What one step needs: the batch size past which averaging stops buying signal, and how
    /// much of a batch that size points the wrong way. From one large draw, not a run sweep.
    BatchSize,
    /// What the step direction should remember: the β₂ balancing momentum's variance against the
    /// staleness of the gradients it still holds.
    Momentum,
}

pub struct TrainerContext<'a> {
    pub train: &'a [FeatureRecord],
    pub val: &'a [FeatureRecord],
    pub train_count: usize,
    pub phase_weights: &'a [f64],
    loss_fn: LossFn,
    vol_threshold: i16,
    vol_adaptive: bool,
}

impl TrainerContext<'_> {
    /// The configured loss, for probes that read its Hessian.
    #[inline]
    #[must_use]
    pub const fn loss_fn(&self) -> LossFn { self.loss_fn }

    pub fn passes_vol_filter(&self, record: &FeatureRecord) -> bool {
        if self.vol_threshold == 0 || record.score == SoulEntry::NO_SCORE {
            return true;
        }

        let threshold = if self.vol_adaptive {
            let occ = i16::from(record.piece_count);
            self.vol_threshold.saturating_add((occ - 10).saturating_mul(2))
        } else {
            self.vol_threshold
        }
        .max(0);

        (i32::from(record.static_eval) - i32::from(record.score)).abs() <= threshold as i32
    }

    pub fn batch_grad(&self, batch_indices: &[u32], values: &[f64], k: f64, blend: f64) -> (Vec<f64>, f64, f64, usize) {
        batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(mut grad_acc, mut k_grad_acc, mut loss_acc, mut count), chunk| {
                    for &idx in chunk {
                        let i = idx as usize;
                        let record = &self.train[i];
                        if !self.passes_vol_filter(record) {
                            continue;
                        }

                        let (target, dt_dk) = wdl_target(record, k, blend);
                        let eval = eval_record_full(record, values);
                        let p = sigmoid(eval.score, k);
                        let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[i] };
                        let gs = self.loss_fn.grad_scale(p, target, k);

                        loss_acc += w * self.loss_fn.loss(p, target);
                        accumulate_record_grad(record, &eval, gs * w, &mut grad_acc);
                        k_grad_acc += ((gs / k) * eval.score + self.loss_fn.grad_target(p, target) * dt_dk) * w;
                        count += 1;
                    }
                    (grad_acc, k_grad_acc, loss_acc, count)
                },
            )
            .reduce(
                || (vec![0.0; values.len()], 0.0f64, 0.0f64, 0usize),
                |(g1, kg1, l1, c1), (g2, kg2, l2, c2)| {
                    let (g, l) = combine_grads((g1, l1), (g2, l2));
                    (g, kg1 + kg2, l, c1 + c2)
                },
            )
    }

    /// Validation loss at each `(k, blend)` probe, over one pass of the split.
    pub fn val_eval<const N: usize>(&self, values: &[f64], probes: [(f64, f64); N]) -> [f64; N] {
        self.split_eval(self.val, self.train_count, values, probes)
    }

    /// Loss at one probe over whichever split a K fit can read.
    pub fn k_fit_eval(&self, values: &[f64], k: f64, blend: f64) -> f64 {
        if self.val.is_empty() {
            let probe = &self.train[..self.train.len().min(K_FIT_PROBE)];
            return self.split_eval(probe, 0, values, [(k, blend)])[0];
        }
        self.val_eval(values, [(k, blend)])[0]
    }

    /// One weighted pass over `records`, whose phase weights start at `offset`.
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
                    if !self.passes_vol_filter(record) {
                        return (sums, weight_acc);
                    }

                    let score = eval_record(record, values);
                    let w = if self.phase_weights.is_empty() { 1.0 } else { self.phase_weights[offset + idx] };
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
                    for (s1, s2) in sums1.iter_mut().zip(&sums2) {
                        *s1 += s2;
                    }
                    (sums1, w1 + w2)
                },
            );

        if total_weight > 0.0 { weighted_loss_sums.map(|sum| sum / total_weight) } else { [0.0; N] }
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

/// A run's artifact by name, under [`ARTIFACT_DIR`] when a parent named one.
pub fn artifact(name: &str) -> String {
    match std::env::var_os(ARTIFACT_DIR) {
        Some(dir) => PathBuf::from(dir).join(name).display().to_string(),
        None => name.to_string(),
    }
}

/// What a run leaves behind for a caller that ranks runs.
pub enum Outcome {
    Ranked(f64),
    /// Trained, on a dataset with no tenth to spare, so no figure ranks it.
    Unranked,
    /// Nothing trained: the dataset never loaded, or the task reports to stdout instead.
    NotTrained,
}

impl Outcome {
    #[inline]
    #[must_use]
    pub const fn trained(&self) -> bool { !matches!(self, Self::NotTrained) }
}

/// Trains on `dataset_path`, or on the dataset the `resume_path` checkpoint names.
pub fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>, task: Task) -> Outcome {
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
        return Outcome::NotTrained;
    };

    let filter = replay_filter(config);
    let is_replay_dataset = paths.iter().any(|p| p.ends_with(".vf") || p.ends_with(".viri"));
    if is_replay_dataset && config.replay_filter.is_none() {
        alarm!("No replay_filter named: every replayed position trains, tactical and in-check ones included.");
    }

    let (all_entries, sample_weights, groups) = load_datasets(&paths, &filter);
    if all_entries.is_empty() {
        eprintln!("Error: No positions loaded.");
        return Outcome::NotTrained;
    }

    let dataset_label = paths.join(", ");
    let keep_fraction = epoch_keep_fraction(config, &filter, is_replay_dataset);
    let outcome = train_entries(all_entries, sample_weights, groups, &dataset_label, config, resume_path, task, keep_fraction);
    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n{}Done in {elapsed:.2}s{RESET}", palette::BRAND);
    outcome
}

/// Permutes whole groups, so a game leaves for the holdout entire,
/// and returns the group sizes in their new order.
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

/// Holdout positions and the groups they came from, taking whole groups off the tail.
/// One group always stays behind, since a run needs something to train on.
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

/// The share of the training split one epoch draws.
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

pub fn replay_filter(config: &EvalTuneConfig) -> ReplayFilter {
    let Some(path) = config.replay_filter.as_deref() else {
        return ReplayFilter::UNRESTRICTED;
    };
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        alarm!("Cannot read replay filter {path}: {e}");
        std::process::exit(1);
    });
    let filter: ReplayFilter = toml::from_str(&text).unwrap_or_else(|e| {
        alarm!("Cannot parse replay filter {path}: {e}");
        std::process::exit(1);
    });
    println!("Replay filter: {path}");
    filter
}

fn train_entries(
    entries: Vec<SoulEntry>,
    sample_weights: Vec<f32>,
    groups: Vec<u32>,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    task: Task,
    epoch_keep: f64,
) -> Outcome {
    let dataset_fnv = dataset_fingerprint(&entries);
    let (rng_seed, split_seed) = match resume_path {
        Some(path) => {
            let cp = peek_checkpoint(path).unwrap_or_else(|e| {
                eprintln!("Failed to read checkpoint: {e}");
                std::process::exit(1);
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

    let (entries, sample_weights, sizes) = shuffle_groups(entries, sample_weights, &groups, split_seed);
    let total = entries.len();

    // One time cost
    println!("Extracting features ({} entries)...", entries.len());
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
    print_dataset_stats(entry_train, entry_val, total, |e: &SoulEntry| {
        let stm = if (e.stm_and_ep & 0x80) == 0 { Color::White } else { Color::Black };
        flip_wdl(f64::from(e.result) / 2.0, stm)
    });
    drop(entries);

    let (train, val) = records.split_at(train_count);

    let phase_weights = if config.phase_balance {
        build_phase_weights(&records, config.phase_balance_cap, config.phase_target.as_deref())
    } else {
        Vec::new()
    };

    let phase_weights = merge_weights(phase_weights, &sample_weights);
    let ctx = TrainerContext {
        train,
        val,
        train_count,
        phase_weights: &phase_weights,
        loss_fn: config.loss,
        vol_threshold: config.volatility_threshold,
        vol_adaptive: config.volatility_adaptive,
    };

    match task {
        Task::GatherCost => {
            gather_cost(&ctx, config);
            return Outcome::NotTrained;
        },
        Task::Curvature => {
            curvature_report(&ctx, config);
            return Outcome::NotTrained;
        },
        Task::ValCost => {
            val_cost(&ctx, config);
            return Outcome::NotTrained;
        },
        Task::BatchSize => {
            batch_size_report(&ctx, config);
            return Outcome::NotTrained;
        },
        Task::Momentum => {
            momentum_report(&ctx, config);
            return Outcome::NotTrained;
        },
        Task::Train => {},
    }

    let seeds = Seeds { rng_seed, split_seed };
    train_loop(train.len(), epoch_keep, "SoulEntry", dataset_label, config, resume_path, seeds, dataset_fnv, &ctx)
}

/// `Name (detail)` with the name in the value color. Both schedulers and `KMode`
/// describe themselves in that shape, so the header paints them the same way.
fn paint_head(text: &str) -> String {
    match text.find('(') {
        Some(op) => format!("{VAL}{}{RESET} ({})", text[..op].trim_end(), &text[op + 1..text.len() - 1]),
        None => format!("{VAL}{text}{RESET}"),
    }
}

fn combine_grads((mut g1, l1): (Vec<f64>, f64), (g2, l2): (Vec<f64>, f64)) -> (Vec<f64>, f64) {
    for (a, b) in g1.iter_mut().zip(g2) {
        *a += b;
    }
    (g1, l1 + l2)
}

fn train_loop(
    train_len: usize,
    epoch_keep: f64,
    mode_label: &str,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    seeds: Seeds,
    dataset_fnv: u64,
    ctx: &TrainerContext,
) -> Outcome {
    let Seeds { rng_seed, split_seed } = seeds;

    let has_val = !ctx.val.is_empty();
    let all_params = eval_params::collect_parameters();
    let default_values = eval_params::default_values(&all_params);
    let slots = all_params.len();

    let resume = resume_path.map(|path| {
        println!("Resuming from checkpoint: {}{path}{RESET}", palette::VAL);
        let data = load_checkpoint(path, &all_params, &default_values).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {e}");
            std::process::exit(1);
        });
        if data.fresh_params > 0 {
            println!(
                "{}{}{RESET} parameters are newer than the checkpoint, starting from code defaults",
                palette::VAL,
                data.fresh_params
            );
        }
        data
    });

    let (start_epoch, mut lr_scale) = resume.as_ref().map_or((1, 1.0), |d| (d.run.epoch, d.run.lr_scale));
    let mut values = match resume.as_ref() {
        Some(d) => d.values.clone(),
        None => seed_values(&all_params, config.init, rng_seed),
    };

    let is_constant_schedule = matches!(config.lr_schedule, LrScheduleConfig::Constant { .. });
    let lr_scheduler = config.lr_schedule.clone().into_scheduler();
    let wdl_scheduler = config.wdl_schedule.clone().into_scheduler();
    let init_blend = wdl_scheduler.blend(1, config.epochs);

    let mut k_ctrl = KController::bootstrap(config, ctx, &values, &default_values, init_blend, resume.as_ref());
    let k = k_ctrl.k();
    let win_rate_100cp = sigmoid(100.0, k);

    println!("{LAB}K Factor:{RESET}   {VAL}{k:.6}{RESET} (100cp -> {:.1}%)", win_rate_100cp * 100.0);
    println!("{LAB}K Mode:{RESET}     {}", paint_head(&config.k_mode.to_string()));

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
    let stride = (ctx.train.len() / GAUGE_PROBE).max(1);
    let probe: Vec<&FeatureRecord> = ctx.train.iter().step_by(stride).take(GAUGE_PROBE).collect();
    let mut gauge = Gauge::new(probe, &default_values);
    let hold_scale = gauge.holds(&values);

    let mut masks = build_masks(&all_params, config);

    let mut grad_ema = resume.as_ref().map_or_else(|| vec![0.0_f64; slots], |d| d.grad_ema.clone());
    let mut stagnant_counts = resume.as_ref().map_or_else(|| vec![0usize; slots], |d| d.stagnant.clone());

    println!("{LAB}Parameters:{RESET} {VAL}{slots}{RESET}");
    println!("{LAB}Mode:{RESET}       {VAL}{mode_label}{RESET}");
    println!("{LAB}LR Sched:{RESET}   {}", paint_head(&lr_scheduler.describe()));
    println!("{LAB}WDL Sched:{RESET}  {}", paint_head(&wdl_scheduler.describe()));
    println!("{LAB}Optimizer:{RESET}  {VAL}Lion{RESET} (Batch: {}, WD: {})", config.batch_size, config.weight_decay);

    let log_file = fs::OpenOptions::new().create(true).append(true).open(artifact("evaltune_log.txt")).ok();
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
        writeln!(w, "K mode:    {}", config.k_mode).ok();
        writeln!(w, "Epochs:    {}", config.epochs).ok();
        writeln!(w, "Params:    {slots}").ok();
        writeln!(w, "LR:        {}", lr_scheduler.describe()).ok();
        writeln!(w, "WDL:       {}", wdl_scheduler.describe()).ok();
        writeln!(w, "Optimizer: Lion (batch: {}, WD: {})", config.batch_size, config.weight_decay).ok();
        writeln!(w).ok();
        writeln!(w, "{}", EpochStats::log_header()).ok();
    }

    let mut json_logger = JsonLogger::new(&config.log_path)
        .inspect_err(|e| eprintln!("{}[!] Cannot open the run log at {}: {e}{RESET}", palette::ALARM, config.log_path))
        .ok();

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
                "params": slots,
            }),
        );
    }

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    let mut optimizer = Lion::new(slots, config.beta1, lr_scheduler.rate(start_epoch, config.epochs), config.weight_decay);

    if let Some(d) = &resume {
        optimizer.restore_momentum(&d.momentum);
    }

    let mut grad_stats = GradientStats::new(100);
    // worth 12% of epoch time at 32.8M positions.
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
    let lr_peak = (1..=config.epochs).fold(0.0f64, |m, e| m.max(lr_scheduler.rate(e, config.epochs)));
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

    let psqt_end = LAYOUT.material_offset;
    let base_end = psqt_end + LAYOUT.material_len;
    let mob_start = LAYOUT.mobility_open_offset;
    let mob_end = LAYOUT.mobility_closed_offset + LAYOUT.mobility_closed_len;
    let group_ranges = group_ranges(slots);
    let mut run_census = [GateCensus::default(); GROUP_NAMES.len()];

    if let Some(d) = &resume {
        masks.fixed.copy_from_slice(&d.frozen);
    } else if config.unfreeze_epoch > 0 {
        for f in &mut masks.fixed[base_end..] {
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

        let is_restart = lr_scheduler.is_restart_boundary(epoch, config.epochs);

        optimizer.set_lr(lr);

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

            let (mut grads, k_grad, batch_loss, batch_count) = ctx.batch_grad(batch, &values, k_ctrl.k(), blend);

            train_loss += batch_loss;
            train_count += batch_count;

            let n = batch_count.max(1) as f64;
            let norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
            grad_stats.update(norm / n);

            let threshold = grad_stats.clip_threshold(config.grad_clip) * n;
            let scale = if norm > threshold { threshold / norm } else { 1.0 };
            for (i, g) in grads.iter_mut().enumerate() {
                *g = *g / n * scale;
                epoch_grads[i] += *g;
            }

            // Before the update, while momentum still holds what the gate is about to read.
            if config.gate_census {
                for (census, range) in epoch_census.iter_mut().zip(&group_ranges) {
                    census.absorb(optimizer.census(range.clone(), &grads, &masks.fixed));
                }
            }

            optimizer.update(&mut values, &grads, &masks);
            k_ctrl.on_batch(k_grad, batch_count, lr, scale, config.weight_decay);
            canonicalize(&mut values, &all_params);

            if hold_scale {
                gauge.restore(&mut values, &mut optimizer, &mut k_ctrl);
            }

            // The |gradient| trail the auto-freeze below reads.
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
            for (i, p) in all_params.iter().enumerate() {
                masks.fixed[i] = p.is_fixed;
            }
            println!("  Unfrozen all remaining parameters at epoch {epoch}");
        }

        if config.auto_freeze && epoch > config.freeze_start_epoch && epoch % config.freeze_cadence == 0 {
            let mut frozen = 0;
            for i in 0..slots {
                if !masks.fixed[i] && grad_ema[i] < config.freeze_threshold && !all_params[i].freeze_resistant {
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
            ctx.val_eval(&ema_values, [(k_ctrl.k(), blend), (k_ctrl.k_ref(), 0.0)])
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
            } else {
                if is_constant_schedule && progress.plateau_count >= config.patience {
                    lr_scale *= 0.5;
                    progress.plateau_count = 0;
                    println!("  Plateau detected, LR scale → {lr_scale:.3}");
                }
            }
            improved
        } else {
            if improved_train {
                best_val_params.copy_from_slice(&ema_values);
            }
            improved_train
        };

        let overfit = if has_val { divergence.update(train_loss, val_loss) } else { false };
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
            psqt_norm: epoch_grads[..psqt_end].iter().map(|g| g * g).sum::<f64>().sqrt(),
            mob_norm: epoch_grads[mob_start..mob_end].iter().map(|g| g * g).sum::<f64>().sqrt(),
            overfit,
            drifted,
            moved,
            gauge: gauge.applied,
            step_l1,
            clipped: clipped_epoch,
        };

        if let Some(ref mut w) = logger {
            writeln!(w, "{}", stats.log_row()).ok();
        }

        if let Some(ref mut l) = json_logger {
            if is_restart {
                l.log("restart", &serde_json::json!({ "epoch": epoch }));
            }
            l.log("epoch", &stats);
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

        if config.gate_census {
            let mut total_census = GateCensus::default();
            for (total, epoch_stat) in run_census.iter_mut().zip(&epoch_census) {
                total.absorb(*epoch_stat);
                total_census.absorb(*epoch_stat);
            }
            println!("  {LAB}gate{RESET}      {}  {DIM}step{RESET} {VAL}{step_l1:.1}{RESET}", gate_columns(&total_census));
        }

        if has_val {
            val_history.push(val_loss);
            prev_val_loss = val_loss;
        }
        train_history.push(train_loss);

        if epoch % 20 == 0 || epoch == config.epochs {
            print_loss_history(&val_history, &train_history);
            if epoch != config.epochs {
                print_params(&all_params, &initial_values, &ema_values);
            }

            if let Err(e) = save_checkpoint(&artifact("evaltune_checkpoint.json"), &all_params, &TrainerState {
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
    let (landed, report_k) = if hold_scale {
        (1.0 / gauge.applied, k_ctrl.k())
    } else {
        gauge.normalize(&mut ema_values);
        gauge.normalize(&mut best_train_params);
        let factor = gauge.normalize(&mut best_val_params);
        (1.0 / factor, k_ctrl.k() / factor)
    };

    if let Some(ref l) = json_logger {
        quantize(&best_val_params, &mut quantized);
        l.log(
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
    let gauge_line = format!("\n{LAB}Gauge:{RESET}      {VAL}{landed:.3}×{RESET} pull on the eval's scale, {how}\n");
    let off_scale = off_scale_warning(off_scale_ratio);
    let clamped_k = clamped_k_warning(config, k_ctrl.k());

    // Without a split a calibration table would print empty.
    let calibration = if has_val { calibration_report(ctx, &best_val_params, report_k) } else { String::new() };
    let census = if config.gate_census { gate_census_report(&run_census) } else { String::new() };
    let clip = clip_report(&all_params, optimizer.clipped(), batches_run);

    print!("{gauge_line}");
    eprint!("{off_scale}{clamped_k}");
    print!("{calibration}{census}{clip}");

    if let Some(ref mut w) = logger {
        for part in [&gauge_line, &off_scale, &clamped_k, &calibration, &census, &clip] {
            write!(w, "{}", color::strip(part)).ok();
        }
    }

    drop(logger);
    sensitivity_report(&all_params, &grad_ema, &masks.fixed);

    let last_val = val_history.last().copied();
    let last_train = train_history.last().copied();
    print_results(
        &all_params,
        &initial_values,
        &ema_values,
        &BestEpochs {
            best_val_params: &best_val_params,
            best_val_loss: if has_val { Some(progress.best_val_loss) } else { None },
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
        let avg_mpos = epoch_positions as f64 / grad_seconds.max(1e-6) / 1e6;
        let rest_seconds = epoch_seconds - grad_seconds - shuffle_seconds - val_seconds;
        println!(
            "\n{LAB}Trained{RESET} {epochs_run} epochs in {epoch_seconds:.2}s  \
             {DIM}grad {grad_seconds:.2}s · shuffle {shuffle_seconds:.2}s · val {val_seconds:.2}s · rest {rest_seconds:.2}s{RESET}  \
             {avg_mpos:.1}M pos/s"
        );
    }

    if has_val { Outcome::Ranked(progress.best_val_loss) } else { Outcome::Unranked }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_holdout_takes_whole_groups_and_lands_nearest_the_target() {
        assert_eq!(holdout(&[100, 80, 40], 220, 100), (120, 2));
        assert_eq!(holdout(&[100, 200, 40], 340, 100), (40, 1));
        assert_eq!(holdout(&[1; 100], 100, 10), (10, 10));
        assert_eq!(holdout(&[95, 5], 100, 10), (5, 1));
        assert_eq!(holdout(&[18, 6], 24, 3), (6, 1));
    }

    #[test]
    fn one_seed_permutes_two_slices_alike() {
        let mut left: Vec<u32> = (0..5000).collect();
        let mut right: Vec<u32> = (0..5000).collect();
        fastrand::Rng::with_seed(0xA11CE).shuffle(&mut left);
        fastrand::Rng::with_seed(0xA11CE).shuffle(&mut right);
        assert_eq!(left, right);
        assert_ne!(left, (0..5000).collect::<Vec<u32>>(), "the shuffle did nothing");
    }

    #[test]
    fn merged_weights_keep_the_gradient_scale() {
        let sample = vec![0.25f32, 0.75, 1.0, 0.5];
        let merged = merge_weights(Vec::new(), &sample);
        let mean = merged.iter().sum::<f64>() / merged.len() as f64;
        assert!((mean - 1.0).abs() < 1e-12, "mean {mean} is not one");
        assert!((merged[1] / merged[0] - 3.0).abs() < 1e-12, "the ratio between two weights moved");
        let phase = vec![2.0, 2.0, 1.0, 1.0];
        let both = merge_weights(phase.clone(), &sample);
        let both_mean = both.iter().sum::<f64>() / both.len() as f64;
        assert!((both_mean - 1.0).abs() < 1e-12);
        assert_eq!(merge_weights(phase.clone(), &[]), phase, "no sampling weights leaves the phase ones alone");
        assert!(merge_weights(Vec::new(), &[]).is_empty());
    }

    #[test]
    fn decimation_thins_the_epoch_and_not_the_dataset() {
        let mut config = crate::config::TunerConfig::default().evaltune;
        let ninety = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };
        let off = ReplayFilter { random_fen_skipping: false, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };
        assert!((epoch_keep_fraction(&config, &ninety, true) - 0.1).abs() < 1e-12);
        assert!((epoch_keep_fraction(&config, &off, true) - 1.0).abs() < 1e-12, "the flag gates the probability");
        assert!((epoch_keep_fraction(&config, &ninety, false) - 1.0).abs() < 1e-12);
        config.epoch_sample = Some(0.25);
        for replays in [true, false] {
            assert!((epoch_keep_fraction(&config, &ninety, replays) - 0.25).abs() < 1e-12, "the config knob wins");
        }
        assert_eq!(epoch_sample_len(32_800_000, 0.1), 3_280_000);
        assert_eq!(epoch_sample_len(32_800_000, 1.0), 32_800_000);
        assert_eq!(epoch_sample_len(4, 0.0), 1, "rounding must not empty an epoch");
        assert_eq!(epoch_sample_len(0, 0.5), 0);
    }

    #[test]
    fn the_shipped_filter_file_names_only_real_fields() {
        let text = std::fs::read_to_string("replay_filter.toml").expect("replay_filter.toml must exist");
        let shipped: toml::Table = toml::from_str(&text).expect("replay_filter.toml must parse");
        let rendered = toml::to_string(&ReplayFilter::default()).expect("the filter must serialize");
        let known: toml::Table = toml::from_str(&rendered).expect("its own render must parse");
        let _: ReplayFilter = toml::from_str(&text).expect("replay_filter.toml must parse as a filter");
        for key in shipped.keys() {
            assert!(known.contains_key(key), "`{key}` is not a ReplayFilter field");
        }
    }

    #[test]
    fn a_partial_filter_file_fills_from_the_defaults() {
        let filter: ReplayFilter = toml::from_str("min_ply = 24\nfilter_castling = true\n").expect("a partial filter file parses");
        assert_eq!(filter.min_ply, 24);
        assert!(filter.filter_castling);
        assert_eq!(filter.min_pieces, 4, "an unnamed field keeps viriformat's default");
        assert!(filter.filter_tactical, "and so does an unnamed flag");
    }

    fn vol_ctx(threshold: i16, adaptive: bool) -> TrainerContext<'static> {
        TrainerContext {
            train: &[],
            val: &[],
            train_count: 0,
            phase_weights: &[],
            loss_fn: LossFn::CrossEntropy,
            vol_threshold: threshold,
            vol_adaptive: adaptive,
        }
    }

    #[test]
    fn the_volatility_filter_scales_with_complexity() {
        let rec = |pieces: u8, static_eval: i16| FeatureRecord { piece_count: pieces, static_eval, ..Default::default() };
        let ctx = vol_ctx(40, true);
        assert!(ctx.passes_vol_filter(&rec(32, 80)), "occ 32: t = 84 accepts an 80 cp disagreement");
        assert!(!ctx.passes_vol_filter(&rec(32, 100)), "occ 32: t = 84 rejects a 100 cp disagreement");
        assert!(ctx.passes_vol_filter(&rec(8, 30)), "occ 8: t = 36 accepts a 30 cp disagreement");
        assert!(!ctx.passes_vol_filter(&rec(8, 50)), "occ 8: t = 36 rejects a 50 cp disagreement");
        let ctx_fixed = vol_ctx(40, false);
        assert!(ctx_fixed.passes_vol_filter(&rec(32, 40)), "fixed: the threshold ignores occupancy");
        assert!(!ctx_fixed.passes_vol_filter(&rec(32, 41)), "fixed: the threshold ignores occupancy");
    }

    #[test]
    fn the_volatility_filter_floor_never_rejects_a_whole_position_class() {
        let ctx = vol_ctx(8, true);
        let rec = |pieces: u8, static_eval: i16| FeatureRecord { piece_count: pieces, static_eval, ..Default::default() };
        assert!(ctx.passes_vol_filter(&rec(6, 0)), "a floored threshold passes a perfect match");
        assert!(!ctx.passes_vol_filter(&rec(6, 1)), "a floored threshold rejects any disagreement");
        let ctx_dense = TrainerContext { vol_threshold: 40, ..ctx };
        assert!(ctx_dense.passes_vol_filter(&rec(31, 0)), "a dense position passes a perfect match");
        assert!(ctx_dense.passes_vol_filter(&rec(31, 80)), "occ 31: t = 82 accepts an 80 cp disagreement");
        assert!(!ctx_dense.passes_vol_filter(&rec(31, 100)), "occ 31: t = 82 rejects a 100 cp disagreement");
    }

    #[test]
    fn a_huge_volatility_threshold_saturates_instead_of_overflowing() {
        let ctx = vol_ctx(i16::MAX, true);
        let rec = |pieces: u8, score: i16, static_eval: i16| FeatureRecord {
            piece_count: pieces,
            score,
            static_eval,
            ..Default::default()
        };
        assert!(ctx.passes_vol_filter(&rec(2, 0, 30_000)), "a sparse position saturates the add");
        assert!(ctx.passes_vol_filter(&rec(32, 0, 30_000)), "a dense position saturates the add");
        assert!(!ctx.passes_vol_filter(&rec(32, -1, 32_767)), "32768 is past the saturated 32767");
    }

    #[test]
    fn the_k_gradient_matches_a_finite_difference_of_the_batch_loss() {
        use super::super::engine::{Position, SoulEntry};

        let params = eval_params::collect_parameters();
        let values = eval_params::default_values(&params);
        let entry = SoulEntry::from_board(
            &Position::from_fen("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
            2.0,
            Some(55),
        );
        let record = FeatureRecord::from_entry(&entry);
        let ctx = TrainerContext {
            train: std::slice::from_ref(&record),
            val: &[],
            train_count: 1,
            phase_weights: &[],
            loss_fn: LossFn::CrossEntropy,
            vol_threshold: 0,
            vol_adaptive: true,
        };
        let k = 0.005;
        let blend = 0.3;
        let (_, k_g, ..) = ctx.batch_grad(&[0], &values, k, blend);

        let loss_at = |kk: f64| {
            let (target, _) = wdl_target(&record, kk, blend);
            let eval = eval_record_full(&record, &values);
            ctx.loss_fn.loss(sigmoid(eval.score, kk), target)
        };
        let h = 1e-6;
        let numeric = (loss_at(k + h) - loss_at(k - h)) / (2.0 * h);
        assert!((k_g - numeric).abs() < 1e-6 * (1.0 + numeric.abs()), "batch k_g {k_g} against FD {numeric}");
    }

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
