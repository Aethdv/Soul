use std::process;

use clap::{Parser, Subcommand};
use evaltuner::{
    ablation,
    assay::{self, Assay},
    config::{Init, KMode, LossFn, LrScheduleConfig, TunerConfig, WdlScheduleConfig},
    correlation,
    engine::Help,
    run::{self, Task, replay_filter},
    seeds,
};

/// Where the tuner looks when `--config` is absent.
const DEFAULT_CONFIG: &str = "evaltuner/config.toml";

#[derive(Parser)]
#[command(
    name = "evaltune",
    about = "Tunes evaluation parameters via logistic regression & Lion optimizer.",
    disable_help_flag = true,
    disable_help_subcommand = true
)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,
    #[arg(short, long, value_delimiter = ',', num_args = 1..)]
    dataset: Option<Vec<String>>,
    #[arg(short, long, default_value = DEFAULT_CONFIG)]
    config: String,
    #[arg(short, long)]
    epochs: Option<usize>,
    #[arg(short, long)]
    blend: Option<f64>,
    #[arg(short, long)]
    resume: Option<String>,
    #[arg(long)]
    lr: Option<f64>,
    #[arg(long)]
    min_lr: Option<f64>,
    #[arg(long)]
    warmup: Option<f64>,
    #[arg(long)]
    cycles: Option<usize>,
    #[arg(long)]
    lr_schedule: Option<String>,
    #[arg(long)]
    wdl_start: Option<f64>,
    #[arg(long)]
    wdl_end: Option<f64>,
    #[arg(long)]
    wdl_schedule: Option<String>,
    #[arg(long)]
    seed: Option<u64>,
    #[arg(long)]
    split_seed: Option<u64>,
    #[arg(long)]
    init: Option<String>,
    #[arg(long)]
    lr_mult: Option<f64>,
    #[arg(long)]
    log: Option<String>,
    #[arg(long)]
    shuffle_block: Option<usize>,
    #[arg(long, action = clap::ArgAction::SetTrue)]
    help: bool,
}

/// What every probe takes; they differ only in the [`Task`] they dispatch.
#[derive(clap::Args)]
struct ProbeArgs {
    #[arg(short, long, default_value = DEFAULT_CONFIG)]
    config: String,
    dataset: String,
}

#[derive(Subcommand)]
enum Commands {
    #[command(name = "ablation")]
    Ablation {
        #[arg(short, long, value_delimiter = ',', num_args = 1..)]
        data: Vec<String>,
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
    },
    #[command(name = "correlation")]
    Correlation,
    #[command(name = "seed-spread")]
    SeedSpread {
        #[arg(long, default_value_t = 8)]
        count: usize,
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,
        #[arg(long, default_value = "evaltune.jsonl")]
        log: String,
        dataset: String,
    },
    #[command(name = "gather-cost")]
    GatherCost(ProbeArgs),
    #[command(name = "curvature")]
    Curvature(ProbeArgs),
    #[command(name = "val-cost")]
    ValCost(ProbeArgs),
    #[command(name = "batch-size")]
    BatchSize(ProbeArgs),
    #[command(name = "momentum")]
    Momentum(ProbeArgs),
    #[command(name = "score")]
    Score {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(short, long)]
        params: Vec<String>,
        #[arg(short, long)]
        loss: Option<String>,
        #[arg(long, default_value = "shipped")]
        shipped: String,
        #[arg(required = true, num_args = 1..)]
        datasets: Vec<String>,
    },
    #[command(name = "material")]
    Material {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(long, default_value = "shipped")]
        shipped: String,
        #[arg(required = true, num_args = 1..)]
        datasets: Vec<String>,
    },
    #[command(name = "spread")]
    Spread {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(required = true, num_args = 1..)]
        datasets: Vec<String>,
    },
    #[command(name = "profile")]
    Profile {
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(required = true, num_args = 1..)]
        datasets: Vec<String>,
    },
    #[command(name = "sweep-lr-mult")]
    SweepLrMult {
        #[arg(short, long, value_delimiter = ',', num_args = 0..)]
        values: Option<Vec<f64>>,
        #[arg(long, default_value_t = 0.001)]
        min: f64,
        #[arg(long, default_value_t = 0.3)]
        max: f64,
        #[arg(long, default_value_t = 6)]
        count: usize,
        #[arg(short, long, default_value = DEFAULT_CONFIG)]
        config: String,
        #[arg(short, long)]
        epochs: Option<usize>,
        #[arg(long, default_value_t = 1)]
        refine_rounds: usize,
        #[arg(long)]
        seed: Option<u64>,
        dataset: String,
    },
    Help,
}

fn main() {
    let args = Args::parse();

    if args.help {
        print_help();
        return;
    }

    match args.command {
        Some(Commands::Help) => print_help(),
        Some(Commands::Ablation { data, config: config_path }) => {
            ablation::run_ablation(&data, &replay_filter(&load_config(&config_path).evaltune));
        },
        Some(Commands::Correlation) => {
            correlation::run_correlation();
        },
        Some(Commands::SeedSpread { count, config: config_path, epochs, log, dataset }) => {
            seeds::run_seed_spread(&dataset, &config_path, epochs, count, &log);
        },
        Some(Commands::GatherCost(args)) => probe(&args, Task::GatherCost),
        Some(Commands::Curvature(args)) => probe(&args, Task::Curvature),
        Some(Commands::ValCost(args)) => probe(&args, Task::ValCost),
        Some(Commands::BatchSize(args)) => probe(&args, Task::BatchSize),
        Some(Commands::Momentum(args)) => probe(&args, Task::Momentum),
        Some(Commands::Score { config: config_path, sample, params, loss, shipped, datasets }) => {
            let Some(loss) = parse_loss(loss.as_deref(), &config_path) else {
                return;
            };
            assay(&config_path, &datasets, Assay::Score { params, loss, shipped }, sample);
        },
        Some(Commands::Material { config: config_path, sample, shipped, datasets }) => {
            assay(&config_path, &datasets, Assay::Material { shipped }, sample);
        },
        Some(Commands::Spread { config: config_path, sample, datasets }) => {
            assay(&config_path, &datasets, Assay::Spread, sample);
        },
        Some(Commands::Profile { config: config_path, sample, datasets }) => {
            assay(&config_path, &datasets, Assay::Profile, sample);
        },
        Some(Commands::SweepLrMult { values, min, max, count, config: config_path, epochs, refine_rounds, seed, dataset }) => {
            let grid = match values {
                Some(v) if !v.is_empty() => v,
                _ => log_space(min, max, count),
            };

            sweep_lr_mult(&dataset, &config_path, grid, (min, max), epochs.unwrap_or(100), refine_rounds, seed);
        },
        None => {
            if !run_evaltune(args) {
                process::exit(1);
            }
        },
    }
}

/// One row per dataset, so an assay runs beside the trainer rather than through it.
fn assay(config_path: &str, datasets: &[String], report: Assay, sample: Option<usize>) {
    assay::run(&report, datasets, &load_config(config_path).evaltune, sample);
}

fn load_config(path: &str) -> TunerConfig {
    TunerConfig::from_file(path).unwrap_or_else(|_| {
        eprintln!("Error: cannot read the config at '{path}'. Fix the path, or run from the repo root.");
        process::exit(1);
    })
}

/// Cross-entropy unless asked otherwise.
fn parse_loss(name: Option<&str>, config_path: &str) -> Option<LossFn> {
    match name {
        None => Some(LossFn::CrossEntropy),
        Some("config") => Some(load_config(config_path).evaltune.loss),
        Some(other) => other.parse().ok().or_else(|| {
            eprintln!("Unknown --loss '{other}'; expected 'config' or one of {:?}", LossFn::NAMES);
            None
        }),
    }
}

/// One diagnostic pass over a dataset.
fn probe(args: &ProbeArgs, task: Task) { run::run(Some(&args.dataset), &load_config(&args.config).evaltune, None, task); }

fn log_space(min: f64, max: f64, points: usize) -> Vec<f64> {
    // Otherwise i / (n - 1) is 0/0 and every point on the grid comes out NaN.
    if points <= 1 {
        return vec![min];
    }

    let log_min = min.ln();
    let log_max = max.ln();

    (0..points)
        .map(|i| {
            let t = i as f64 / (points - 1) as f64;
            (log_min * (1.0 - t) + log_max * t).exp()
        })
        .collect()
}

/// Each round doubles the epoch budget and re-centers a three-point grid on the winner.
fn sweep_lr_mult(
    dataset: &str,
    config_path: &str,
    mut grid: Vec<f64>,
    (min, max): (f64, f64),
    base_epochs: usize,
    refine_rounds: usize,
    seed: Option<u64>,
) {
    let mut results: Vec<(f64, f64, f32)> = Vec::new();
    let mut best = (f64::MAX, grid[0]);

    let print_row = |lr_mult: f64, loss: f64, elapsed: f32| {
        let loss_str = if loss == f64::MAX { "FAILED".to_string() } else { format!("{loss:>10.6}") };
        println!("  {lr_mult:>7.4}    {loss_str}    {elapsed:.1}s");
    };

    for round in 0..=refine_rounds {
        let epochs = base_epochs * (1 << round);
        let log_step = if grid.len() > 1 { (grid[grid.len() - 1].ln() - grid[0].ln()) / (grid.len() - 1) as f64 } else { 0.5 };

        println!("── Round {round}: {epochs} epochs");
        println!("  lr_mult    Best L_val    Time");
        println!("  -------    ----------    ----");

        for &lr_mult in &grid {
            let (loss, duration) = run_sweep_trial(lr_mult, dataset, config_path, epochs, seed);

            print_row(lr_mult, loss, duration);
            results.push((lr_mult, loss, duration));

            if loss < best.0 {
                best = (loss, lr_mult);
            }
        }

        if round < refine_rounds && grid.len() > 1 {
            let half_step = log_step / 2.0;
            let log_best = best.1.ln();

            grid = vec![
                (log_best - half_step).exp().clamp(min, max),
                log_best.exp().clamp(min, max),
                (log_best + half_step).exp().clamp(min, max),
            ];
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));

    println!("\n  Sorted by L_val:");
    println!("  lr_mult    Best L_val    Time");
    println!("  -------    ----------    ----");

    for &(lr_mult, loss, duration) in &results {
        print_row(lr_mult, loss, duration);
    }

    println!("\nBest lr_mult = {:.4} (L_val = {:.6})", best.1, best.0);
}

/// One trial's best validation loss, or `f64::MAX` if it never reported one.
fn run_sweep_trial(lr_mult: f64, dataset: &str, config_path: &str, epochs: usize, seed: Option<u64>) -> (f64, f32) {
    let log_path = std::env::temp_dir().join(format!("sweep_lr_mult_{}.jsonl", std::process::id()));
    let log_str = log_path.to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&log_str);
    let mut extra_args = vec![("--lr-mult", lr_mult.to_string())];
    if let Some(s) = seed {
        extra_args.push(("--seed", s.to_string()));
    }

    let start = std::time::Instant::now();
    let success = seeds::spawn_trial(dataset, config_path, epochs, &log_str, &extra_args);
    let elapsed = start.elapsed().as_secs_f32();
    let val_loss = if success { seeds::last_best_val(&log_str).unwrap_or(f64::MAX) } else { f64::MAX };
    let _ = std::fs::remove_file(&log_str);
    (val_loss, elapsed)
}

fn run_evaltune(args: Args) -> bool {
    let mut tuner_config = load_config(&args.config);
    if let Some(epochs) = args.epochs {
        tuner_config.evaltune.epochs = epochs;
    }
    if let Some(seed) = args.seed {
        tuner_config.evaltune.seed = Some(seed);
    }
    if let Some(ref path) = args.log {
        tuner_config.evaltune.log_path = path.clone();
    }
    if let Some(block) = args.shuffle_block {
        tuner_config.evaltune.shuffle_block = block;
    }
    if let Some(seed) = args.split_seed {
        tuner_config.evaltune.split_seed = Some(seed);
    }
    if let Some(mode) = args.init {
        tuner_config.evaltune.init = match mode.as_str() {
            "default" => Init::Default,
            "zero" => Init::Zero,
            "random" => Init::Random,
            other => {
                eprintln!("Unknown --init '{other}'; expected 'default', 'zero', or 'random'");
                return false;
            },
        };
    }

    if let Some(lr_mult) = args.lr_mult {
        tuner_config.evaltune.k_mode = KMode::Learned { lr_mult };
    }

    // LR schedule: type override or field overrides
    if let Some(schedule_type) = args.lr_schedule {
        let schedule = match schedule_type.as_str() {
            "constant" => Some(LrScheduleConfig::Constant { value: args.lr.unwrap_or(0.1) }),
            "linear" => Some(LrScheduleConfig::Linear { start: args.lr.unwrap_or(0.1), end: args.min_lr.unwrap_or(0.0) }),
            "cosine" => Some(LrScheduleConfig::Cosine {
                base: args.lr.unwrap_or(0.1),
                min: args.min_lr.unwrap_or(0.0001),
                warmup_ratio: args.warmup.unwrap_or(0.1),
                cycles: args.cycles.unwrap_or(1),
            }),
            "wsd" => Some(LrScheduleConfig::WarmupStableDecay {
                base: args.lr.unwrap_or(0.1),
                min: args.min_lr.unwrap_or(0.0),
                warmup_ratio: args.warmup.unwrap_or(0.1),
                stable_ratio: 0.5,
            }),
            "sd" => Some(LrScheduleConfig::StableDecay {
                base: args.lr.unwrap_or(0.03),
                min: args.min_lr.unwrap_or(0.0001),
                stable_ratio: 0.5,
            }),
            _ => {
                eprintln!("Warning: Unknown LR schedule '{schedule_type}', ignoring.");
                None
            },
        };

        if let Some(s) = schedule {
            tuner_config.evaltune.lr_schedule = s;
        }
    } else {
        tuner_config
            .evaltune
            .lr_schedule
            .apply_overrides(args.lr, args.min_lr, args.warmup, args.cycles);
    }

    // WDL schedule: type override or field overrides
    let (default_start, default_end) = tuner_config.evaltune.wdl_schedule.defaults();

    if let Some(schedule_type) = args.wdl_schedule {
        let schedule = match schedule_type.as_str() {
            "constant" => Some(WdlScheduleConfig::Constant { value: args.blend.unwrap_or(default_start) }),
            "linear" => Some(WdlScheduleConfig::Linear {
                start: args.wdl_start.unwrap_or(default_start),
                end: args.wdl_end.unwrap_or(default_end),
            }),
            "cosine" => Some(WdlScheduleConfig::Cosine {
                start: args.wdl_start.unwrap_or(default_start),
                end: args.wdl_end.unwrap_or(default_end),
            }),
            "stable-decay" => Some(WdlScheduleConfig::StableDecay {
                start: args.wdl_start.unwrap_or(default_start),
                end: args.wdl_end.unwrap_or(default_end),
                stable_ratio: 0.35,
            }),
            _ => {
                eprintln!("Warning: Unknown WDL schedule '{schedule_type}', ignoring.");
                None
            },
        };

        if let Some(s) = schedule {
            tuner_config.evaltune.wdl_schedule = s;
        }
    } else {
        tuner_config
            .evaltune
            .wdl_schedule
            .apply_overrides(args.blend, args.wdl_start, args.wdl_end);
    }

    let dataset_str = args.dataset.map(|v| v.join(","));
    run::run(dataset_str.as_deref(), &tuner_config.evaltune, args.resume.as_deref(), Task::Train).trained()
}

fn print_help() {
    let h = Help::new(36);

    h.header("Evaluation Parameter Tuning via Evolved Sign Momentum");
    h.separator();

    h.header("Commands");
    h.command_args("ablation", "-d <path,...>", "Zero term groups, report ΔL_val");
    h.command_args("correlation", "", "Analyze PSQT square adjacency roughness");
    h.command_args("curvature", "<dataset>", "Report what the data determines about the weights");
    h.command_args("gather-cost", "<dataset>", "Time the gradient pass, sequential vs blocked vs shuffled");
    h.command_args("val-cost", "<dataset>", "Time the fused validation pass against two separate ones");
    h.command_args("batch-size", "<dataset>", "What one step needs: the noise scale, and a batch's sign error");
    h.command_args("momentum", "<dataset>", "What the step remembers: β₂ against gradient staleness");
    h.command_args("seed-spread", "<dataset> [options]", "Run N seeds of one config, report where they land");
    h.command_args("sweep-lr-mult", "<dataset> [options]", "Sweep lr_mult with auto-grid + refinement");
    h.separator();

    h.header("Options");
    h.option("-d, --dataset", "<path,...>", "Paths to .epd, .txt or .vf files");
    h.option("-e, --epochs", "<N>", "Number of training epochs, overriding the config");
    h.option("-b, --blend", "<ratio>", "Target: 0 = game result, 1 = search score; flattens any WDL schedule");
    h.option("-r, --resume", "<path>", "Resume from a JSON checkpoint");
    h.option("--lr", "<value>", "Base learning rate");
    h.option("--min-lr", "<value>", "Minimum learning rate (cosine/linear)");
    h.option_default("--warmup", "<ratio>", "Warmup fraction", "0.1");
    h.option_default("--cycles", "<N>", "Cosine restarts (SGDR), or StepDecay's period in epochs", "1");
    h.option("--lr-schedule", "<type>", "[cosine|linear|constant|wsd|sd]");
    h.option("--wdl-start", "<ratio>", "WDL blend start (scheduled)");
    h.option("--wdl-end", "<ratio>", "WDL blend end (scheduled)");
    h.option("--wdl-schedule", "<type>", "[constant|linear|cosine|stable-decay]");
    h.option("--seed", "<u64>", "Fixed RNG seed for reproducible training");
    h.option("--split-seed", "<u64>", "Reseed the validation holdout, fixed by default");
    h.option("--lr-mult", "<f64>", "Learning-rate multiplier");
    h.option_default("--init", "<mode>", "Starting weights [default|zero|random]", "default");
    h.option("--log", "<path>", "JSON-lines run log, read back by seed-spread and sweeps");
    h.option("--shuffle-block", "<N>", "Permute blocks of N consecutive positions, 0 for a full shuffle");
    h.separator();

    h.header("Assays: what a dataset says before anything trains on it");
    h.command_args("score", "<dataset...>", "Loss of each parameter vector on each set, at its own K");
    h.subcommand("-p, --params", "<checkpoint>", "Score this run too, at its best-validation vector");
    h.subcommand_default("-l, --loss", "<name>", "[ce|sce|mse|focal], or config for the one training uses", "ce");
    h.subcommand_default("--shipped", "<label>", "Name the reference row", "shipped");
    h.command_args("material", "<dataset...>", "Ten piece values fitted to the labels alone");
    h.subcommand_default("--shipped", "<label>", "Name the reference row", "shipped");
    h.command_args("profile", "<dataset...>", "Labels, imbalances, phase, and how the games ended");
    h.separator();

    h.header("Assay options, on all three");
    h.option("--sample", "<N>", "Every nth position, for sets too large to hold whole");
    h.option("-c, --config", "<path>", "Where the replay filter for a .vf set is named");
    h.separator();

    h.header("Examples");
    h.example("./eval material <data1> <data2>");
    h.example("./eval score -p evaltune_checkpoint.json <data>");
    h.example("./eval profile --sample 400000 <data>");
}
