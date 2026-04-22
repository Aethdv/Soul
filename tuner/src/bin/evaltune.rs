use clap::{Parser, Subcommand};
use tuner::evaltune;

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
    restarts: Option<usize>,

    #[arg(long)]
    lr_schedule: Option<String>,

    #[arg(long)]
    wdl_start: Option<f64>,

    #[arg(long)]
    wdl_end: Option<f64>,

    #[arg(long)]
    wdl_schedule: Option<String>,

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
            if let Err(e) = evaltune::loader::encode_epd(&input, &output) {
                eprintln!("Error: {e}");
                std::process::exit(1);
            }
        },
        None => {
            let mut tuner_config = tuner::core::config::TunerConfig::from_file(&args.config).unwrap_or_else(|e| {
                eprintln!("Warning: Failed to load config '{}': {e}. Using defaults.", args.config);
                tuner::core::config::TunerConfig::default()
            });

            if let Some(epochs) = args.epochs {
                tuner_config.evaltune.epochs = epochs;
            }

            // Apply LR schedule overrides
            if let Some(stype) = args.lr_schedule {
                match stype.as_str() {
                    "constant" => {
                        tuner_config.evaltune.lr_schedule =
                            tuner::core::config::LrScheduleConfig::Constant { value: args.lr.unwrap_or(0.1) }
                    },
                    "linear" => {
                        tuner_config.evaltune.lr_schedule = tuner::core::config::LrScheduleConfig::Linear {
                            start: args.lr.unwrap_or(0.1),
                            end: args.min_lr.unwrap_or(0.0),
                        }
                    },
                    "cosine" => {
                        tuner_config.evaltune.lr_schedule = tuner::core::config::LrScheduleConfig::Cosine {
                            base: args.lr.unwrap_or(0.1),
                            min: args.min_lr.unwrap_or(0.0001),
                            warmup_ratio: args.warmup.unwrap_or(0.1),
                            restarts: args.restarts.unwrap_or(1),
                        }
                    },
                    "wsd" => {
                        tuner_config.evaltune.lr_schedule = tuner::core::config::LrScheduleConfig::WarmupStableDecay {
                            base: args.lr.unwrap_or(0.1),
                            min: args.min_lr.unwrap_or(0.0),
                            warmup_ratio: args.warmup.unwrap_or(0.1),
                            stable_ratio: 0.5,
                        }
                    },
                    _ => eprintln!("Warning: Unknown LR schedule '{}', ignoring.", stype),
                }
            } else if let tuner::core::config::LrScheduleConfig::Cosine {
                ref mut base,
                ref mut min,
                ref mut warmup_ratio,
                ref mut restarts,
            } = tuner_config.evaltune.lr_schedule
            {
                if let Some(v) = args.lr {
                    *base = v;
                }
                if let Some(v) = args.min_lr {
                    *min = v;
                }
                if let Some(v) = args.warmup {
                    *warmup_ratio = v;
                }
                if let Some(v) = args.restarts {
                    *restarts = v;
                }
            }

            // Apply WDL schedule overrides
            let (def_start, def_end) = match tuner_config.evaltune.wdl_schedule {
                tuner::core::config::WdlScheduleConfig::Cosine { start, end }
                | tuner::core::config::WdlScheduleConfig::Linear { start, end }
                | tuner::core::config::WdlScheduleConfig::StableDecay { start, end, .. } => (start, end),
                tuner::core::config::WdlScheduleConfig::Constant { value } => (value, 0.3),
            };

            if let Some(stype) = args.wdl_schedule {
                match stype.as_str() {
                    "constant" => {
                        tuner_config.evaltune.wdl_schedule =
                            tuner::core::config::WdlScheduleConfig::Constant { value: args.blend.unwrap_or(def_start) }
                    },
                    "linear" => {
                        tuner_config.evaltune.wdl_schedule = tuner::core::config::WdlScheduleConfig::Linear {
                            start: args.wdl_start.unwrap_or(def_start),
                            end: args.wdl_end.unwrap_or(def_end),
                        }
                    },
                    "cosine" => {
                        tuner_config.evaltune.wdl_schedule = tuner::core::config::WdlScheduleConfig::Cosine {
                            start: args.wdl_start.unwrap_or(def_start),
                            end: args.wdl_end.unwrap_or(def_end),
                        }
                    },
                    "stable-decay" => {
                        tuner_config.evaltune.wdl_schedule = tuner::core::config::WdlScheduleConfig::StableDecay {
                            start: args.wdl_start.unwrap_or(def_start),
                            end: args.wdl_end.unwrap_or(def_end),
                            stable_ratio: 0.35,
                        }
                    },
                    _ => eprintln!("Warning: Unknown WDL schedule '{}', ignoring.", stype),
                }
            } else {
                if let Some(blend) = args.blend {
                    tuner_config.evaltune.wdl_schedule = tuner::core::config::WdlScheduleConfig::Constant { value: blend };
                }
                match tuner_config.evaltune.wdl_schedule {
                    tuner::core::config::WdlScheduleConfig::Linear { ref mut start, ref mut end }
                    | tuner::core::config::WdlScheduleConfig::Cosine { ref mut start, ref mut end }
                    | tuner::core::config::WdlScheduleConfig::StableDecay { ref mut start, ref mut end, .. } => {
                        if let Some(v) = args.wdl_start {
                            *start = v;
                        }
                        if let Some(v) = args.wdl_end {
                            *end = v;
                        }
                    },
                    _ => {},
                }
            }

            let dataset_str = args.dataset.map(|v| v.join(","));
            evaltune::run(dataset_str.as_deref(), &tuner_config.evaltune, args.resume.as_deref());
        },
    }
}

fn print_help() {
    let h = soul::cli::Help::new(28);

    h.header("Evaluation Parameter Tuning via Evolved Sign Momentum");
    h.separator();

    h.header("Commands");
    h.command_args("encode", "<in> <out>", "Pre-encode EPD → .soul.zst");
    h.separator();

    h.header("Options");
    h.option("-d, --dataset", "<path,...>", "Paths to .epd or .soul.zst files");
    h.option_default("-e, --epochs", "<N>", "Number of training epochs", "8000");
    h.option_default("-b, --blend", "<ratio>", "Constant WDL blend factor", "0.3");
    h.option("-r, --resume", "<path>", "Resume from a JSON checkpoint");
    h.option("--lr", "<value>", "Base learning rate");
    h.option("--min-lr", "<value>", "Minimum learning rate (cosine/linear)");
    h.option_default("--warmup", "<ratio>", "Warmup fraction", "0.1");
    h.option("--lr-schedule", "<type>", "[cosine|linear|constant|wsd]");
    h.option("--wdl-start", "<ratio>", "WDL blend start (scheduled)");
    h.option("--wdl-end", "<ratio>", "WDL blend end (scheduled)");
    h.option("--wdl-schedule", "<type>", "[constant|linear|cosine]");
}
