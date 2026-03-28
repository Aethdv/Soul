#![feature(custom_inner_attributes)]
#![rustfmt::skip]

use clap::{Parser, Subcommand};
use tuner::searchtune;

#[derive(Parser)]
#[command(
    name = "searchtune",
    about = "Tunes search parameters via CMA-ES",
    disable_help_flag = true,
    disable_help_subcommand = true
)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    #[arg(short, long, default_value = "UHO_Lichess_4852_v1.epd")]
    openings: String,

    #[arg(short, long, default_value = "tuner/tuner_config.toml")]
    config: String,

    #[arg(short, long)]
    epochs: Option<usize>,

    #[arg(short, long)]
    pairs: Option<usize>,

    #[arg(short, long)]
    tc: Option<String>,

    #[arg(long)]
    h2h_tc: Option<String>,

    #[arg(long)]
    val_tc: Option<String>,

    #[arg(long)]
    h2h_pairs: Option<usize>,

    #[arg(long)]
    val_pairs: Option<usize>,

    #[arg(short, long)]
    resume: bool,

    #[arg(long)]
    centering_penalty: Option<f64>,

    #[arg(long)]
    speed_penalty: Option<f64>,

    #[arg(long)]
    active_softness: Option<f64>,

    #[arg(long, action = clap::ArgAction::SetTrue)]
    help: bool,
}

#[derive(Subcommand)]
enum Commands {
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
        None => {
            let mut tuner_config = tuner::core::config::TunerConfig::from_file(&args.config)
                .unwrap_or_else(|e| {
                    eprintln!("Warning: Failed to load config '{}': {e}. Using defaults.", args.config);
                    tuner::core::config::TunerConfig::default()
                });

            if let Some(epochs) = args.epochs {
                tuner_config.searchtune.epochs = epochs;
            }
            if let Some(pairs) = args.pairs {
                tuner_config.searchtune.pairs = pairs;
            }
            if let Some(tc) = args.tc {
                tuner_config.searchtune.tc = Some(tc);
            }
            if let Some(h2h_tc) = args.h2h_tc {
                tuner_config.searchtune.h2h_tc = Some(h2h_tc);
            }
            if let Some(val_tc) = args.val_tc {
                tuner_config.searchtune.val_tc = Some(val_tc);
            }
            if let Some(h2h_pairs) = args.h2h_pairs {
                tuner_config.searchtune.h2h_pairs = h2h_pairs;
            }
            if let Some(val_pairs) = args.val_pairs {
                tuner_config.searchtune.validation_pairs = val_pairs;
            }
            if let Some(cp) = args.centering_penalty {
                tuner_config.searchtune.centering_penalty = cp;
            }
            if let Some(nps) = args.speed_penalty {
                tuner_config.searchtune.speed_penalty = nps;
            }
            if let Some(soft) = args.active_softness {
                tuner_config.searchtune.active_softness = soft;
            }

            searchtune::run(
                &args.openings,
                &tuner_config.searchtune,
                args.resume
            );
        },
    }
}

/// Manual help text. MUST be kept in sync with `Args` struct above.
/// If you add/remove a flag there, mirror it here or nobody will ever find it.
fn print_help() {
    let h = soul::cli::Help::new(34);

    h.header("Search Parameter Tuning via Soft Active CMA-ES");
    h.separator();

    h.header("Options");
    h.option_default("-o, --openings", "<path>", "Openings EPD file", "UHO_Lichess_4852_v1.epd");
    h.option_default("-e, --epochs", "<N>", "CMA-ES generations", "50");
    h.option_default("-p, --pairs", "<N>", "Game pairs per candidate", "16");
    h.option("-t, --tc", "<time>", "Time control (nodes=N / 4+0.04 / depth=N)");
    h.option_default("--h2h-tc", "<time>", "H2H validation time control", "1.0+0.01");
    h.option_default("--val-tc", "<time>", "Ensemble validation time control", "1.0+0.01");
    h.option_default("--h2h-pairs", "<N>", "Pairs for H2H validation", "300");
    h.option_default("--val-pairs", "<N>", "Pairs for ensemble validation", "1000");
    h.option("-r, --resume", "", "Resume from checkpoint");
    h.option_default("--centering-penalty", "<val>", "Centering penalty", "100.0");
    h.option_default("--speed-penalty", "<val>", "Speed penalty", "115.0");
    h.option_default("--active-softness", "<val>", "Active CMA-ES softness", "0.5");
}
