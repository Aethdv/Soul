use std::process;

use clap::{Parser, Subcommand};
use soul::cli::Help;
use tuner::{
    core::config::{Init, KMode, LossFn, LrScheduleConfig, TunerConfig, WdlScheduleConfig},
    evaltune,
    evaltune::{
        ablation,
        assay::Assay,
        correlation, loader,
        run::{Task, replay_filter},
        seeds,
    },
};

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
    #[arg(short, long, default_value = "tuner/tuner_config.toml")]
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

#[derive(Subcommand)]
enum Commands {
    #[command(name = "encode")]
    Encode {
        input: String,
        output: String,
    },
    #[command(name = "ablation")]
    Ablation {
        #[arg(short, long, value_delimiter = ',', num_args = 1..)]
        data: Vec<String>,
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
    },
    #[command(name = "correlation")]
    Correlation,
    #[command(name = "seed-spread")]
    SeedSpread {
        #[arg(long, default_value_t = 8)]
        count: usize,
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
        #[arg(short, long, default_value_t = 100)]
        epochs: usize,
        #[arg(long, default_value = "evaltune.jsonl")]
        log: String,
        dataset: String,
    },
    #[command(name = "gather-cost")]
    GatherCost {
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
        dataset: String,
    },
    #[command(name = "curvature")]
    Curvature {
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
        dataset: String,
    },
    #[command(name = "val-cost")]
    ValCost {
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
        dataset: String,
    },
    #[command(name = "score")]
    Score {
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
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
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
        config: String,
        #[arg(long)]
        sample: Option<usize>,
        #[arg(long, default_value = "shipped")]
        shipped: String,
        #[arg(required = true, num_args = 1..)]
        datasets: Vec<String>,
    },
    #[command(name = "profile")]
    Profile {
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
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
        #[arg(short, long, default_value = "tuner/tuner_config.toml")]
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
        Some(Commands::Encode { input, output }) => {
            if let Err(e) = loader::encode_epd(&input, &output) {
                eprintln!("Error: {e}");
                process::exit(1);
            }
        },
        Some(Commands::Ablation { data, config: config_path }) => {
            ablation::run_ablation(&data, &replay_filter(&load_config(&config_path).evaltune));
        },
        Some(Commands::Correlation) => {
            correlation::run_correlation();
        },
        Some(Commands::SeedSpread { count, config: config_path, epochs, log, dataset }) => {
            seeds::run_seed_spread(&dataset, &config_path, epochs, count, &log);
        },
        Some(Commands::GatherCost { config: config_path, dataset }) => {
            probe(&config_path, &dataset, Task::GatherCost);
        },
        Some(Commands::Curvature { config: config_path, dataset }) => {
            probe(&config_path, &dataset, Task::Curvature);
        },
        Some(Commands::ValCost { config: config_path, dataset }) => {
            probe(&config_path, &dataset, Task::ValCost);
        },
        Some(Commands::Score { config: config_path, sample, params, loss, shipped, datasets }) => {
            let Some(loss) = parse_loss(loss.as_deref(), &config_path) else {
                return;
            };

            assay(&config_path, &datasets, Assay::Score { params, loss, shipped }, sample);
        },
        Some(Commands::Material { config: config_path, sample, shipped, datasets }) => {
            assay(&config_path, &datasets, Assay::Material { shipped }, sample);
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

/// One row per dataset, so an assay runs beside the trainer rather than through it: the split, the
/// sample weights and the merge into one pool all describe a training run.
fn assay(config_path: &str, datasets: &[String], report: Assay, sample: Option<usize>) {
    evaltune::assay::run(&report, datasets, &load_config(config_path).evaltune, sample);
}

fn load_config(path: &str) -> TunerConfig {
    TunerConfig::from_file(path).unwrap_or_else(|e| {
        eprintln!("Warning: Failed to load config '{path}': {e}. Using defaults.");
        TunerConfig::default()
    })
}

/// The yardstick's own loss, `config` for whatever the config file trains under, or a name.
fn parse_loss(name: Option<&str>, config_path: &str) -> Option<LossFn> {
    match name {
        None => Some(LossFn::CrossEntropy),
        Some("config") => Some(TunerConfig::from_file(config_path).map(|cfg| cfg.evaltune.loss).unwrap_or_default()),
        Some(other) => other.parse().ok().or_else(|| {
            eprintln!("Unknown --loss '{other}'; expected config or one of {:?}", LossFn::NAMES);
            None
        }),
    }
}

/// One diagnostic pass over a dataset: everything up to the trainer, then the probe instead of it.
fn probe(config_path: &str, dataset: &str, task: Task) {
    evaltune::run(Some(dataset), &load_config(config_path).evaltune, None, task);
}

fn log_space(lo: f64, hi: f64, n: usize) -> Vec<f64> {
    // Otherwise `i / (n - 1)` is 0/0 and every point on the grid comes out NaN.
    if n <= 1 {
        return vec![lo];
    }

    let e0 = lo.ln();
    let e1 = hi.ln();

    (0..n)
        .map(|i| {
            let t = i as f64 / (n - 1) as f64;
            (e0 * (1.0 - t) + e1 * t).exp()
        })
        .collect()
}

/// Each round doubles the epoch budget and re-centers a three-point grid on the winner,
/// so the cheap rounds place the bracket and the expensive one only has to split it.
fn sweep_lr_mult(
    dataset: &str,
    config_path: &str,
    mut grid: Vec<f64>,
    (min, max): (f64, f64),
    base_epochs: usize,
    refine_rounds: usize,
    seed: Option<u64>,
) {
    let mut all_results: Vec<(f64, f64, f32)> = Vec::new();
    let mut best = (f64::MAX, grid[0]);

    let row = |lr_mult: f64, val: f64, t: f32| {
        let label = if val == f64::MAX { "FAILED".to_string() } else { format!("{val:>10.6}") };
        println!("  {lr_mult:>7.4}    {label}    {t:.1}s");
    };

    for round in 0..=refine_rounds {
        let ep = base_epochs * (1 << round);
        let lstep = if grid.len() > 1 { (grid[grid.len() - 1].ln() - grid[0].ln()) / (grid.len() - 1) as f64 } else { 0.5 };

        println!("── Round {round}: {ep} epochs");
        println!("  lr_mult    Best L_val    Time");
        println!("  -------    ----------    ----");

        for &lr_mult in &grid {
            let (val, t) = run_sweep_trial(lr_mult, dataset, config_path, ep, seed);

            row(lr_mult, val, t);
            all_results.push((lr_mult, val, t));

            if val < best.0 {
                best = (val, lr_mult);
            }
        }

        if round < refine_rounds && grid.len() > 1 {
            let half = lstep / 2.0;
            let b = best.1.ln();

            grid = vec![(b - half).exp().clamp(min, max), b.exp().clamp(min, max), (b + half).exp().clamp(min, max)];
        }
    }

    all_results.sort_by(|a, b| a.1.total_cmp(&b.1));

    println!("\n  Sorted by L_val:");
    println!("  lr_mult    Best L_val    Time");
    println!("  -------    ----------    ----");

    for &(lr_mult, val, t) in &all_results {
        row(lr_mult, val, t);
    }

    println!("\nBest lr_mult = {:.4} (L_val = {:.6})", best.1, best.0);
}

/// One trial's best validation loss, or `f64::MAX` if it never reported one.
///
/// The child logs to a scratch file of its own rather than the run log, so a sweep of
/// short trials does not bury a real run in the file the plotter reads.
fn run_sweep_trial(lr_mult: f64, dataset: &str, config_path: &str, epochs: usize, seed: Option<u64>) -> (f64, f32) {
    let log = std::env::temp_dir().join(format!("sweep_lr_mult_{}.jsonl", std::process::id()));
    let log = log.to_string_lossy().into_owned();
    let _ = std::fs::remove_file(&log);

    let mut extra = vec![("--lr-mult", lr_mult.to_string())];

    if let Some(s) = seed {
        extra.push(("--seed", s.to_string()));
    }

    let start = std::time::Instant::now();
    let ok = seeds::spawn_trial(dataset, config_path, epochs, &log, &extra);
    let elapsed = start.elapsed().as_secs_f32();

    let val = if ok { seeds::last_best_val(&log).unwrap_or(f64::MAX) } else { f64::MAX };
    let _ = std::fs::remove_file(&log);

    (val, elapsed)
}

fn run_evaltune(args: Args) -> bool {
    let mut tuner_config = TunerConfig::from_file(&args.config).unwrap_or_else(|e| {
        eprintln!("Warning: Failed to load config '{}': {e}. Using defaults.", args.config);
        TunerConfig::default()
    });

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
                eprintln!("Unknown --init '{other}'; expected default, zero or random");
                return false;
            },
        };
    }

    if let Some(lr_mult) = args.lr_mult {
        tuner_config.evaltune.k_mode = KMode::Learned { lr_mult };
    }

    // LR schedule: type override or field overrides
    if let Some(stype) = args.lr_schedule {
        let s = match stype.as_str() {
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
                eprintln!("Warning: Unknown LR schedule '{}', ignoring.", stype);
                None
            },
        };

        if let Some(s) = s {
            tuner_config.evaltune.lr_schedule = s;
        }
    } else {
        tuner_config
            .evaltune
            .lr_schedule
            .apply_overrides(args.lr, args.min_lr, args.warmup, args.cycles);
    }

    // WDL schedule: type override or field overrides
    let (def_start, def_end) = tuner_config.evaltune.wdl_schedule.defaults();

    if let Some(stype) = args.wdl_schedule {
        let s = match stype.as_str() {
            "constant" => Some(WdlScheduleConfig::Constant { value: args.blend.unwrap_or(def_start) }),
            "linear" => {
                Some(WdlScheduleConfig::Linear { start: args.wdl_start.unwrap_or(def_start), end: args.wdl_end.unwrap_or(def_end) })
            },
            "cosine" => {
                Some(WdlScheduleConfig::Cosine { start: args.wdl_start.unwrap_or(def_start), end: args.wdl_end.unwrap_or(def_end) })
            },
            "stable-decay" => Some(WdlScheduleConfig::StableDecay {
                start: args.wdl_start.unwrap_or(def_start),
                end: args.wdl_end.unwrap_or(def_end),
                stable_ratio: 0.35,
            }),
            _ => {
                eprintln!("Warning: Unknown WDL schedule '{}', ignoring.", stype);
                None
            },
        };

        if let Some(s) = s {
            tuner_config.evaltune.wdl_schedule = s;
        }
    } else {
        tuner_config
            .evaltune
            .wdl_schedule
            .apply_overrides(args.blend, args.wdl_start, args.wdl_end);
    }

    let dataset_str = args.dataset.map(|v| v.join(","));
    evaltune::run(dataset_str.as_deref(), &tuner_config.evaltune, args.resume.as_deref(), Task::Train).is_some()
}

fn print_help() {
    let h = Help::new(36);

    h.header("Evaluation Parameter Tuning via Evolved Sign Momentum");
    h.separator();

    h.header("Commands");
    h.command_args("encode", "<in> <out>", "Pre-encode EPD → .soul.zst");
    h.command_args("ablation", "-d <path,...>", "Zero term groups, report ΔL_val");
    h.command_args("correlation", "", "Analyze PSQT square adjacency roughness");
    h.command_args("curvature", "<dataset>", "Report what the data determines about the weights");
    h.command_args("score", "<dataset...> [-p <ckpt>]", "Loss of each parameter vector on each set, at its own K");
    h.command_args("material", "<dataset...>", "Material-only logistic fit against the shipped table");
    h.command_args("profile", "<dataset...>", "Labels, imbalances, phase, and how the games ended");
    h.command_args("gather-cost", "<dataset>", "Time the gradient pass, sequential vs shuffled");
    h.command_args("val-cost", "<dataset>", "Time the fused validation pass against two separate ones");
    h.command_args("seed-spread", "<dataset> [options]", "Run N seeds of one config, report where they land");
    h.command_args("sweep-lr-mult", "<dataset> [options]", "Sweep lr_mult with auto-grid + refinement");
    h.separator();

    h.header("Options");
    h.option("-d, --dataset", "<path,...>", "Paths to .epd or .soul.zst files");
    h.option_default("-e, --epochs", "<N>", "Number of training epochs", "4000");
    h.option_default("-b, --blend", "<ratio>", "Target: 0 = game result, 1 = search score", "0.3");
    h.option("-r, --resume", "<path>", "Resume from a JSON checkpoint");
    h.option("--lr", "<value>", "Base learning rate");
    h.option("--min-lr", "<value>", "Minimum learning rate (cosine/linear)");
    h.option_default("--warmup", "<ratio>", "Warmup fraction", "0.1");
    h.option("--lr-schedule", "<type>", "[cosine|linear|constant|wsd|sd]");
    h.option("--wdl-start", "<ratio>", "WDL blend start (scheduled)");
    h.option("--wdl-end", "<ratio>", "WDL blend end (scheduled)");
    h.option("--wdl-schedule", "<type>", "[constant|linear|cosine]");
    h.option("--seed", "<u64>", "Fixed RNG seed for reproducible training");
    h.option("--split-seed", "<u64>", "Reseed the validation holdout, fixed by default");
    h.option("--lr-mult", "<f64>", "Learning-rate multiplier");
    h.option_default("--init", "<mode>", "Starting weights [default|zero|random]", "default");
    h.option("--sample", "<N>", "Assay N positions taken evenly, not the whole set");
    h.option("--loss", "<name>", "Score under [ce|sce|mse|focal] or config");
}
