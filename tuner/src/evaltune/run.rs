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

use palette::{BRAND, CLEAR_LINE, DIM, LAB, RESET, VAL};
use rayon::prelude::*;

use super::{
    engine::{Color, FeatureRecord, LAYOUT, Tunable, color, eval_params},
    groups::{GROUP_NAMES, build_clip_mask, build_decay_mask, build_lr_mask, group_ranges},
    lion::{GateCensus, Lion, build_beta2_mask},
    loader::{self, ReplayFilter, dataset_fingerprint, flip_wdl, resolve_dataset_paths},
    palette,
    probes::{curvature_report, gather_cost, val_cost},
    report::*,
    scale::{GAUGE_PROBE, Gauge, KController, canonicalize},
    storage::*,
    training::*,
};
use crate::core::{
    config::{EvalTuneConfig, Init, LossFn, LrScheduleConfig, RANDOM_INIT_SPREAD},
    logger::JsonLogger,
    shuffle::Shuffler,
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
        if self.vol_threshold == 0 || entry.score == loader::SoulEntry::NO_SCORE {
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

/// The best validation loss a training run reached, or `None` when there is no such figure:
/// a dataset that never loaded, or a diagnostic task, which reports to stdout and ranks nothing.
pub fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>, task: Task) -> Option<f64> {
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
            palette::ALARM,
        );

        return None;
    };

    let filter = replay_filter(config);
    let replays = paths.iter().any(|p| p.ends_with(".vf") || p.ends_with(".viri"));

    if replays && config.replay_filter.is_none() {
        eprintln!(
            "{}[!] No replay_filter named: every replayed position trains, tactical and in-check ones included.{RESET}",
            palette::ALARM,
        );
    }

    let (all_entries, sample_weights, groups) = loader::load_datasets(&paths, &filter);

    if all_entries.is_empty() {
        eprintln!("Error: No positions loaded.");
        return None;
    }

    let dataset_label = paths.join(", ");
    let keep = epoch_keep_fraction(config, &filter, replays);
    let best_val = train_entries(all_entries, sample_weights, groups, &dataset_label, config, resume_path, task, keep);

    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n{}Done in {elapsed:.2}s{RESET}", palette::BRAND);
    best_val
}

/// Permutes whole groups, so a game leaves for the holdout entire, and returns the group sizes in
/// their new order.
///
/// The weights ride the same permutation, so a position keeps the weight computed for it.
fn shuffle_groups(
    mut entries: Vec<loader::SoulEntry>,
    mut weights: Vec<f32>,
    groups: &[u32],
    seed: u64,
) -> (Vec<loader::SoulEntry>, Vec<f32>, Vec<u32>) {
    // One group per position means nothing here records games, so the entries shuffle directly and
    // such a set keeps the holdout its earlier runs measured against.
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

    let mut shuffled = Vec::with_capacity(entries.len());
    let mut shuffled_weights = Vec::with_capacity(weights.len());
    let mut sizes = Vec::with_capacity(groups.len());

    for &group in &order {
        let start = starts[group as usize];
        let size = groups[group as usize] as usize;

        shuffled.extend_from_slice(&entries[start..start + size]);

        if !weights.is_empty() {
            shuffled_weights.extend_from_slice(&weights[start..start + size]);
        }

        sizes.push(size as u32);
    }

    (shuffled, shuffled_weights, sizes)
}

/// Holdout positions and the groups they came from, taking whole groups off the tail.
///
/// A group is indivisible, so the target is a wish: the last one is taken when it lands the holdout
/// no farther from the target than leaving it would, which centers the miss instead of always
/// overshooting by part of a game. Picking groups by size to hit the target exactly would hold out
/// short games in preference to long ones, and a holdout drawn on length is not a sample of the set.
///
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
///
/// The filter's drop chance is a statement about a replay, so it reaches only a
/// dataset that came from one; `epoch_sample` is the general knob and wins
/// wherever it is set.
fn epoch_keep_fraction(config: &EvalTuneConfig, filter: &ReplayFilter, replays: bool) -> f64 {
    if let Some(sample) = config.epoch_sample {
        return sample.clamp(0.0, 1.0);
    }

    if replays && filter.random_fen_skipping {
        1.0 - filter.random_fen_skip_probability.clamp(0.0, 1.0)
    } else {
        1.0
    }
}

/// The permutation is fresh each epoch, so a prefix of it is a fresh draw. Under
/// `shuffle_block` it permutes blocks, and the prefix takes whole ones.
fn epoch_sample_len(train_len: usize, keep: f64) -> usize {
    if train_len == 0 {
        return 0;
    }

    ((train_len as f64 * keep).round() as usize).clamp(1, train_len)
}

/// The filter file, or every position when none was named. A named file that will
/// not read is fatal: training on the whole set instead of the asked-for subset
/// would silently answer a different question.
pub fn replay_filter(config: &EvalTuneConfig) -> ReplayFilter {
    let Some(path) = config.replay_filter.as_deref() else {
        return ReplayFilter::UNRESTRICTED;
    };

    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        eprintln!("{}[!] Cannot read replay filter {path}: {e}{RESET}", palette::ALARM);
        std::process::exit(1);
    });

    let filter: ReplayFilter = toml::from_str(&text).unwrap_or_else(|e| {
        eprintln!("{}[!] Cannot parse replay filter {path}: {e}{RESET}", palette::ALARM);
        std::process::exit(1);
    });

    println!("Replay filter: {path}");
    filter
}

fn train_entries(
    entries: Vec<loader::SoulEntry>,
    sample_weights: Vec<f32>,
    groups: Vec<u32>,
    dataset_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    task: Task,
    epoch_keep: f64,
) -> Option<f64> {
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
                    palette::ALARM,
                );
            }

            (cp.rng_seed, cp.split_seed.unwrap_or(cp.rng_seed))
        },

        None => (config.seed.unwrap_or_else(|| fastrand::u64(..)), config.split_seed.unwrap_or(VAL_SPLIT_SEED)),
    };

    // Games sit contiguously in the file, so the split needs a shuffle rather than a cut, and the
    // shuffle needs a seed of its own: under the training seed, each run holds out a different
    // tenth and no two validation losses compare.
    let (entries, sample_weights, sizes) = shuffle_groups(entries, sample_weights, &groups, split_seed);

    // One-time cost: training reads FeatureRecords straight through.
    // Parallel because entries are independent.
    println!("Extracting features ({} entries)...", entries.len());
    let records: Vec<FeatureRecord> = entries.par_iter().map(FeatureRecord::from_entry).collect();

    let target = (entries.len() / 10).min(config.val_max.unwrap_or(usize::MAX));
    let (mut val_count, val_groups) = holdout(&sizes, entries.len(), target);

    if val_groups < val_count {
        println!("Holding out {val_groups} whole games, {val_count} positions");
    }

    // A file of one game has none to spare, and an empty holdout is worse than a contaminated one:
    // `val_eval` averages nothing, reports zero, and `best_val` never improves on it again. The
    // position cut is the honest fallback, since it says what it costs.
    if val_count == 0 && target > 0 {
        eprintln!(
            "{}[!] No game can leave without taking the file with it. Falling back to a position cut,\n\
             [!] so train and validation will share games and L_val will read low.{RESET}",
            palette::ALARM,
        );

        val_count = target;
    }

    let train_count = entries.len() - val_count;
    let (train, val) = entries.split_at(train_count);

    // A file with a handful of long games cannot spare a tenth of itself without spending most of
    // its independent samples, so the split keeps them and says so: `L_val` is then an estimate
    // over that many games, and `best_val` selection is choosing on it.
    if val_count * 2 < target {
        eprintln!(
            "{}[!] The holdout came to {val_count} positions against a target of {target}: too few games to cut a tenth.{RESET}",
            palette::ALARM,
        );
    }

    print_dataset_stats(train, val, entries.len(), |e: &loader::SoulEntry| {
        let stm = if (e.stm_and_ep & 0x80) == 0 { Color::White } else { Color::Black };

        flip_wdl(f64::from(e.result) / 2.0, stm)
    });

    // Weight loss too, not just gradient, or selection fights training.
    let phase_weights = if config.phase_balance {
        build_phase_weights(&records, config.phase_balance_cap, config.phase_target.as_deref())
    } else {
        Vec::new()
    };

    let phase_weights = merge_weights(phase_weights, &sample_weights);

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
            return None;
        },

        Task::Curvature => {
            curvature_report(&ctx, config);
            return None;
        },

        Task::ValCost => {
            val_cost(&ctx, config);
            return None;
        },

        Task::Train => {},
    }

    let seeds = Seeds { rng_seed, split_seed };

    Some(train_loop(train.len(), epoch_keep, "SoulEntry", dataset_label, config, resume_path, seeds, dataset_fnv, &ctx))
}

/// `Name (detail)` with the name in the value color. Both schedulers and `KMode`
/// describe themselves in that shape, so the header paints them the same way.
fn paint_head(text: &str) -> String {
    match text.find('(') {
        Some(op) => format!("{VAL}{}{RESET} ({})", text[..op].trim_end(), &text[op + 1..text.len() - 1]),
        None => format!("{VAL}{text}{RESET}"),
    }
}

fn grad_combine((mut g1, l1): (Vec<f64>, f64), (g2, l2): (Vec<f64>, f64)) -> (Vec<f64>, f64) {
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
) -> f64 {
    let Seeds { rng_seed, split_seed } = seeds;

    let all_params = eval_params::collect_parameters();
    let default_values = eval_params::default_values(&all_params);

    let resume = resume_path.map(|path| {
        println!("Resuming from checkpoint: {}{path}{RESET}", palette::VAL);
        let data = load_checkpoint(path, &all_params, &default_values).unwrap_or_else(|e| {
            eprintln!("Failed to load checkpoint: {e}");
            std::process::exit(1);
        });

        if data.fresh_params > 0 {
            println!(
                "{}{}{RESET} parameter(s) are newer than the checkpoint, starting from code defaults",
                palette::VAL,
                data.fresh_params,
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
        println!("{LAB}Split seed:{RESET} {VAL}{split_seed}{RESET} (L_val does not compare to default-split runs)");
    }

    if config.init != Init::Default && resume.is_none() {
        println!("{LAB}Init:{RESET}       {VAL}{:?}{RESET} (cold start; K is meaningless until material grows)", config.init);
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

    println!("{LAB}Parameters:{RESET} {VAL}{}{RESET}", all_params.len());
    println!("{LAB}Mode:{RESET}       {VAL}{mode_label}{RESET}");
    println!("{LAB}LR Sched:{RESET}   {}", paint_head(&lr_scheduler.describe()));
    println!("{LAB}WDL Sched:{RESET}  {}", paint_head(&wdl_scheduler.describe()));
    println!("{LAB}Optimizer:{RESET}  {VAL}Lion{RESET} (Batch: {}, WD: {})", config.batch_size, config.weight_decay);

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
        writeln!(w, "K mode:    {}", config.k_mode).ok();
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
    let epoch_sample = epoch_sample_len(train_len, epoch_keep);
    let steps_per_epoch = epoch_sample.div_ceil(config.batch_size);

    if epoch_sample < train_len {
        println!("Sampling {epoch_sample} of {train_len} training positions per epoch, redrawn each one");
    }

    // An epoch is not a unit of training: the share drawn and the batch size decide what one is
    // worth, and a skip probability moves the share by an order of magnitude. Two runs compare only
    // at equal steps.
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
            shuffler.fill_blocked(&mut indices, rng.u64(..), config.shuffle_block, epoch_sample);
        } else {
            shuffler.fill(&mut indices, rng.u64(..));
        }
        let shuffle_secs = t_shuffle.elapsed().as_secs_f32();

        let mut train_loss = 0.0;
        let mut train_count = 0usize;
        let mut total_grads = vec![0.0; values.len()];

        let t_grad = Instant::now();

        for batch in indices[..epoch_sample].chunks(config.batch_size) {
            batches_run += 1;

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
        let step_l1 = optimizer.take_step_l1();

        // Per epoch, not per run. A run total averages the high-`lr` opening into the decayed
        // tail, and it is the tail that says whether the floor is a preference or an artifact.
        let clipped_total: u64 = optimizer.clipped().iter().sum();
        let clipped_epoch = clipped_total - clipped_seen;
        clipped_seen = clipped_total;

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

        // The overfit flag tests a slope, so a run that simply walks back uphill never
        // trips it. This tests displacement, against the only bar the run itself supplies:
        // a smoothed loss above the opening one has given back everything it ever gained.
        let drifted = val_history.first().is_some_and(|opening| progress.val_smooth > *opening);

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
                    "drifted": drifted,
                    "moved": moved,
                    "gauge": gauge.applied,
                    "step_l1": step_l1,
                    "clipped": clipped_epoch
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
            ('·', LAB.to_string())
        } else if val_loss < prev_val_loss {
            ('▼', palette::fg(color::advantage(0.7)))
        } else {
            ('▲', palette::fg(color::advantage(-0.7)))
        };

        let (mark, epoch_c) = if is_best { ("✦ ", BRAND) } else { ("  ", DIM) };
        let alarm = palette::fg(color::advantage(-1.0));
        let warn = match (overfit, drifted) {
            (true, _) => format!("  {alarm}⚠ overfit{RESET}"),
            (false, true) => format!("  {alarm}⚠ drift{RESET}"),
            (false, false) => String::new(),
        };

        #[rustfmt::skip]
        println!(
            "{mark}{epoch_c}Epoch {epoch:>3}/{}{RESET}  \
             {LAB}val{RESET} {trend}{val_loss:.6}{RESET} {trend}{arrow}{RESET}  \
             {LAB}train{RESET} {DIM}{train_loss:.6}{RESET}  \
             {LAB}ref{RESET} {DIM}{ref_loss:.6}{RESET}  \
             {LAB}lr{RESET} {}{lr:.4}{RESET}  {LAB}Δp{RESET} {DIM}{moved:>3}{RESET}  \
             {DIM}{elapsed:.2}s{RESET}  {DIM}{mpos:.1}M pos/s{RESET}{warn}{CLEAR_LINE}",
            config.epochs,
            palette::VAL,
        );

        if config.gate_census {
            let mut all = GateCensus::default();

            for (total, epoch) in run_census.iter_mut().zip(&epoch_census) {
                total.absorb(*epoch);
                all.absorb(*epoch);
            }

            println!(
                "  {LAB}gate{RESET} φ {VAL}{:.4}{RESET}  step {VAL}{step_l1:.1}{RESET}  skip {VAL}{:.1}%{RESET}  canonical {VAL}{:.1}%{RESET}  band {VAL}{:.2}%{RESET}  \
                 c-only {VAL}{:.1}%{RESET}  waived {VAL}{:.1}%{RESET}  dead {VAL}{:.1}%{RESET}  no grad {VAL}{:.1}%{RESET}",
                all.active_share(),
                all.percent(all.skipped),
                all.percent(all.canonical),
                all.percent(all.band),
                all.percent(all.canonical_only),
                all.percent(all.epsilon_waived),
                all.percent(all.dead),
                all.percent(all.absent),
            );
        }

        val_history.push(val_loss);
        train_history.push(train_loss);
        prev_val_loss = val_loss;

        if epoch % 20 == 0 || epoch == config.epochs {
            let val_tail = &val_history[val_history.len().saturating_sub(40)..];
            let train_tail = &train_history[train_history.len().saturating_sub(40)..];

            println!("\n  {LAB}L_val{RESET}    {}", loss_sparkline(val_tail));
            println!("\n  {LAB}L_train{RESET}  {}", loss_sparkline(train_tail));

            if epoch != config.epochs {
                print_params(&all_params, &initial_values, &ema_values);
            }

            if let Err(e) = save_checkpoint("evaltune_checkpoint.json", &all_params, &TrainerState {
                run: RunState {
                    epoch: epoch + 1, // resume starts here; the current epoch is already done
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

    // A cold start was left to find its own scale, so its output is normalized
    // here instead: the search reads centipawns, and nothing else would put the
    // run's eval back on the scale `search_params` was written against.
    //
    // K is the other half of that scale and takes the same correction: a report
    // pairing moved parameters with the K they moved away from reads the mismatch
    // rather than the model. `Gauge::restore` pays it on a held run.
    //
    // The off-scale ratio is measured before the cold-start normalization that
    // follows: it reads the scale the run actually found, and measured afterwards
    // it would sit at exactly 1.0, silencing the warning on the one path whose
    // scale can wander.
    let off_scale_ratio = Gauge::measure(&gauge.probe, &ema_values) / gauge.reference;

    let (landed, report_k) = if hold_scale {
        (1.0 / gauge.applied, k_ctrl.k())
    } else {
        gauge.normalize(&mut ema_values);
        gauge.normalize(&mut best_train_params);

        let factor = gauge.normalize(&mut best_val_params);

        (1.0 / factor, k_ctrl.k() / factor)
    };

    // The JSON log opens in append mode, so a seed sweep writes every run's final params into
    // one file for reading the spread directly. It has to sit below the normalization: a cold
    // start's vector is off the centipawn scale until then, and the spread would be read in a unit
    // the run invented for itself.
    if let Some(ref l) = json_logger {
        quantize(&best_val_params, &mut quantized);

        l.log(
            "final",
            &serde_json::json!({
                "seed": rng_seed,
                "split_seed": split_seed,
                "epochs": epochs_run,
                "steps": batches_run,
                "best_val_loss": progress.best_val_loss,
                "best_val_epoch": progress.best_val_epoch,
                "best_train_loss": progress.best_train_loss,
                "best_train_epoch": progress.best_train_epoch,
                // Named for the vector it is: the run also ships `ema_values`,
                // and a spread measured over one of them says nothing about the other.
                "best_val_params": quantized,
                "sensitivity": grad_ema_per_param,
            }),
        );
    }

    let how = if hold_scale { "held through the run" } else { "normalized on the way out" };

    let gauge_line = format!("\n{LAB}Gauge:{RESET}      {VAL}{landed:.3}×{RESET} pull on the eval's scale, {how}\n");
    let off_scale = off_scale_warning(off_scale_ratio);
    let clamped_k = clamped_k_warning(config, k_ctrl.k());
    let calibration = calibration_report(ctx, &best_val_params, report_k);
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

        println!(
            "\n{LAB}Trained{RESET} {epochs_run} epochs in {epoch_seconds:.2}s  \
             {DIM}grad {grad_seconds:.2}s · shuffle {shuffle_seconds:.2}s · val {val_seconds:.2}s · rest {rest_seconds:.2}s{RESET}  \
             {avg_mpos:.1}M pos/s"
        );
    }

    progress.best_val_loss
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_holdout_takes_whole_groups_and_lands_nearest_the_target() {
        // Tail first: 40 alone misses 100 by 60, and 40 + 80 overshoots it by 20, so both go.
        assert_eq!(holdout(&[100, 80, 40], 220, 100), (120, 2));

        // Taking the 200 would miss by 140 where stopping misses by 60, so it stays.
        assert_eq!(holdout(&[100, 200, 40], 340, 100), (40, 1));

        // Positions of their own hit the target exactly, as they did before groups existed.
        assert_eq!(holdout(&[1; 100], 100, 10), (10, 10));

        // Two games, and the one that would take the whole file with it cannot leave.
        assert_eq!(holdout(&[95, 5], 100, 10), (5, 1));

        // Equally far either way, and a holdout of six beats a holdout of nothing.
        assert_eq!(holdout(&[18, 6], 24, 3), (6, 1));
    }

    /// Positions and weights are shuffled in two calls, and must walk the same swaps.
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
        assert!(merged[1] / merged[0] - 3.0 < 1e-12, "the ratio between two weights moved");

        let phase = vec![2.0, 2.0, 1.0, 1.0];
        let both = merge_weights(phase.clone(), &sample);
        let both_mean = both.iter().sum::<f64>() / both.len() as f64;

        assert!((both_mean - 1.0).abs() < 1e-12);
        assert_eq!(merge_weights(phase.clone(), &[]), phase, "no sampling weights leaves the phase ones alone");
        assert!(merge_weights(Vec::new(), &[]).is_empty());
    }

    #[test]
    fn decimation_thins_the_epoch_and_not_the_dataset() {
        let mut config = crate::core::config::TunerConfig::default().evaltune;
        let ninety = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };
        let off = ReplayFilter { random_fen_skipping: false, random_fen_skip_probability: 0.9, ..ReplayFilter::UNRESTRICTED };

        assert!((epoch_keep_fraction(&config, &ninety, true) - 0.1).abs() < 1e-12);
        assert!((epoch_keep_fraction(&config, &off, true) - 1.0).abs() < 1e-12, "the flag gates the probability");

        // The drop chance describes a replay, so an EPD or .soul dataset never sees it.
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

    /// A key the struct does not have parses as nothing, since every absent field
    /// fills from the defaults, so a rename on either side would go unnoticed until
    /// a run quietly ignored half the file. Values are the run's to choose; names
    /// are not.
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

    /// Matching viriformat's field names means its files parse here unchanged,
    /// partial ones included.
    #[test]
    fn a_partial_filter_file_fills_from_the_defaults() {
        let filter: ReplayFilter = toml::from_str("min_ply = 24\nfilter_castling = true\n").expect("a partial filter file parses");

        assert_eq!(filter.min_ply, 24);
        assert!(filter.filter_castling);
        assert_eq!(filter.min_pieces, 4, "an unnamed field keeps viriformat's default");
        assert!(filter.filter_tactical, "and so does an unnamed flag");
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
                    assert!(v.abs() <= RANDOM_INIT_SPREAD, "{init:?} left {} at {VAL}", p.name);
                }
            }
        }

        let phase = LAYOUT.phase_offset;
        let zeroed = seed_values(&params, Init::Zero, 7);
        assert!(zeroed[phase..phase + LAYOUT.phase_len].iter().any(|w| *w > 0.0), "phase taper zeroed");
    }
}
