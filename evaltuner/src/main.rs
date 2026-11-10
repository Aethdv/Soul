// Linear-algebra routines use standard single-letter notation (x, y, w, m, k, p).
#![allow(clippy::many_single_char_names)]

mod ablation;
mod assay;
mod config;
mod correlation;
mod curvature;
mod engine;
mod lion;
mod loader;
mod probes;
mod report;
mod run;
mod scale;
mod schedule;
mod seeds;
mod shuffle;
mod storage;
mod training;

use std::process;

use clap::{Parser, Subcommand};

use crate::{
    assay::Assay,
    config::{KMode, LossFn, TunerConfig},
    engine::Help,
    run::{Task, replay_filter},
};

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

#[derive(clap::Args)]
struct ProbeArgs {
    #[arg(short, long, default_value = DEFAULT_CONFIG)]
    config: String,
    dataset: String,
}

#[derive(clap::Args)]
struct AssayArgs {
    #[arg(short, long, default_value = DEFAULT_CONFIG)]
    config: String,
    #[arg(long)]
    sample: Option<usize>,
    #[arg(required = true, num_args = 1..)]
    datasets: Vec<String>,
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
        #[command(flatten)]
        assay: AssayArgs,
        #[arg(short, long)]
        params: Vec<String>,
        #[arg(short, long)]
        loss: Option<String>,
        #[arg(long, default_value = "shipped")]
        shipped: String,
    },
    #[command(name = "material")]
    Material {
        #[command(flatten)]
        assay: AssayArgs,
        #[arg(long, default_value = "shipped")]
        shipped: String,
    },
    #[command(name = "spread")]
    Spread(AssayArgs),
    #[command(name = "profile")]
    Profile(AssayArgs),
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
        Some(Commands::Ablation { data, config }) => {
            ablation::run_ablation(&data, &replay_filter(&load_config(&config).evaltune));
        },
        Some(Commands::Correlation) => correlation::run_correlation(),
        Some(Commands::SeedSpread { count, config, epochs, log, dataset }) => {
            seeds::run_seed_spread(&dataset, &config, epochs, count, &log);
        },
        Some(Commands::GatherCost(args)) => probe(&args, Task::GatherCost),
        Some(Commands::Curvature(args)) => probe(&args, Task::Curvature),
        Some(Commands::ValCost(args)) => probe(&args, Task::ValCost),
        Some(Commands::BatchSize(args)) => probe(&args, Task::BatchSize),
        Some(Commands::Momentum(args)) => probe(&args, Task::Momentum),
        Some(Commands::Score { assay: a, params, loss, shipped }) => {
            let Some(loss) = parse_loss(loss.as_deref(), &a.config) else {
                process::exit(1);
            };
            assay(&a.config, &a.datasets, Assay::Score { params, loss, shipped }, a.sample);
        },
        Some(Commands::Material { assay: a, shipped }) => {
            assay(&a.config, &a.datasets, Assay::Material { shipped }, a.sample);
        },
        Some(Commands::Spread(a)) => assay(&a.config, &a.datasets, Assay::Spread, a.sample),
        Some(Commands::Profile(a)) => assay(&a.config, &a.datasets, Assay::Profile, a.sample),
        Some(Commands::SweepLrMult { values, min, max, count, config, epochs, refine_rounds, seed, dataset }) => {
            seeds::sweep_lr_mult(&dataset, &config, values, (min, max, count), epochs.unwrap_or(100), refine_rounds, seed);
        },
        None => {
            if !run_evaltune(args) {
                process::exit(1);
            }
        },
    }
}

fn assay(config_path: &str, datasets: &[String], report: Assay, sample: Option<usize>) {
    assay::run(&report, datasets, &load_config(config_path).evaltune, sample);
}

fn load_config(path: &str) -> TunerConfig {
    TunerConfig::from_file(path).unwrap_or_else(|_| {
        eprintln!("Error: cannot read the config at '{path}'. Fix the path, or run from the repo root.");
        process::exit(1);
    })
}

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

fn probe(args: &ProbeArgs, task: Task) { run::run(Some(&args.dataset), &load_config(&args.config).evaltune, None, task); }

fn run_evaltune(args: Args) -> bool {
    let mut tuner_config = load_config(&args.config);
    let cfg = &mut tuner_config.evaltune;

    if let Some(epochs) = args.epochs {
        cfg.epochs = epochs;
    }
    if let Some(seed) = args.seed {
        cfg.seed = Some(seed);
    }
    if let Some(path) = args.log {
        cfg.log_path = path;
    }
    if let Some(block) = args.shuffle_block {
        cfg.shuffle_block = block;
    }
    if let Some(seed) = args.split_seed {
        cfg.split_seed = Some(seed);
    }
    if let Some(mode) = args.init {
        let Ok(init) = mode.parse() else {
            eprintln!("Unknown --init '{mode}'; expected 'default', 'zero', or 'random'");
            return false;
        };
        cfg.init = init;
    }
    if let Some(lr_mult) = args.lr_mult {
        cfg.k_mode = KMode::Learned { lr_mult };
    }
    if let Err(err) = cfg
        .lr_schedule
        .apply_cli(args.lr_schedule.as_deref(), args.lr, args.min_lr, args.warmup, args.cycles)
    {
        eprintln!("{err}");
        return false;
    }

    if let Err(err) = cfg
        .wdl_schedule
        .apply_cli(args.wdl_schedule.as_deref(), args.blend, args.wdl_start, args.wdl_end)
    {
        eprintln!("{err}");
        return false;
    }

    let dataset_str = args.dataset.map(|v| v.join(","));
    run::run(dataset_str.as_deref(), cfg, args.resume.as_deref(), Task::Train)
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
    h.command_args("spread", "<dataset...>", "Mobility counts, openness, and rank-7 passer phase");
    h.command_args("profile", "<dataset...>", "Labels, imbalances, phase, and how the games ended");
    h.separator();

    h.header("Assay options");
    h.option("--sample", "<N>", "Every nth position, for sets too large to hold whole");
    h.option("-c, --config", "<path>", "Where the replay filter for a .vf set is named");
    h.separator();

    h.header("Examples");
    h.example("./eval material <data1> <data2>");
    h.example("./eval score -p evaltune_checkpoint.json <data>");
    h.example("./eval profile --sample 400000 <data>");
}
