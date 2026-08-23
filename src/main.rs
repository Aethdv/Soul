//! CLI entry point and command router.
//!
//! Dispatches commands to development tools (bench, perft, dataset)
//! or enters an interactive protocol loop (UCI/XBoard) on stdin.

use std::{
    env::args,
    fs,
    io::{self, BufRead, IsTerminal},
    process,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use soul::{
    core::{
        board::{Position, STARTPOS},
        defs::{MAX_DEPTH, Protocol},
    },
    engine::{
        history::History,
        search::{Limits, SearchConfig, SearchDisplay, Searcher, ThreadResult},
        search_params::{SearchParams, spsa_table},
        tt::TranspositionTable,
    },
    protocols, tools,
};

const SPSA_SCREENFUL: usize = 70;

#[allow(clippy::too_many_lines)]
fn main() {
    let args: Vec<String> = args().collect();
    if args.len() > 1 {
        match args[1].as_str() {
            "help" | "--help" | "-h" => {
                protocols::uci::print_help(true);
            },
            "license" | "--license" => {
                protocols::uci::print_license();
            },
            "version" | "--version" | "-V" => {
                println!("soul {}", env!("CARGO_PKG_VERSION"));
            },
            "bench" => {
                let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(12);
                let hash_mb = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
                tools::bench::run(depth, hash_mb);
            },
            "perft" => {
                let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
                let board = Position::from_fen(STARTPOS);
                tools::perft::run(&board, depth, false);
            },
            "divide" => {
                let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
                let board = Position::from_fen(STARTPOS);
                tools::perft::run(&board, depth, true);
            },
            "spsa" => {
                let table = spsa_table();
                if table.lines().count() > SPSA_SCREENFUL && io::stdout().is_terminal() {
                    match fs::write("spsa.txt", &table) {
                        Ok(()) => println!("{} params written to spsa.txt", table.lines().count()),
                        Err(e) => {
                            eprintln!("spsa.txt: {e}");
                            process::exit(1);
                        },
                    }
                } else {
                    print!("{table}");
                }
            },
            #[cfg(feature = "rigs")]
            "speedtest" => {
                let limit = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                tools::speedtest::run(limit);
            },
            #[cfg(feature = "datagen")]
            "datagen" => {
                let stop = Arc::new(AtomicBool::new(false));
                let datagen_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::datagen::run(&datagen_args, &stop);
            },
            #[cfg(feature = "datagen")]
            "genfens" => {
                let genfens_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::genfens::run(&genfens_args);
            },
            #[cfg(feature = "rigs")]
            "measure" => {
                let measure_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::measure::run(&measure_args);
            },
            #[cfg(feature = "dataset")]
            "dataset" => {
                let dataset_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::dataset::cli::run(&dataset_args);
            },
            "go" => {
                let limit_args: Vec<String> = args[2..].iter().map(String::to_string).collect();
                protocols::uci::run_cli_go(&limit_args);
            },
            "gopretty" => {
                let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(MAX_DEPTH);

                let board = Position::from_fen(STARTPOS);
                let limits = Limits { depth, protocol: Protocol::Uci, ..Default::default() };

                let stop = Arc::new(AtomicBool::new(false));
                let display =
                    SearchDisplay { show_wdl: true, go_pretty: true, pretty_print: false, show_currmove: true, use_ansi: true };
                let cfg = SearchConfig::new_full(limits, Instant::now(), stop, 0, display, SearchParams::default());
                let mut history_table = History::new();
                let mut searcher = Searcher::new(&cfg, &board, &[], Arc::new(TranspositionTable::new(16, 1)));
                searcher.iterative_deepening(&mut history_table);

                // The search publishes instead of printing; one thread means slot 0.
                let result = ThreadResult::unpack(cfg.result_slots[0].load(Ordering::Relaxed));
                println!("bestmove {}", result.mv.to_uci(board.is_frc));
            },
            _ if args[1].contains(' ') => {
                protocols::uci::run_commands(&args[1..]);
            },
            _ => {
                eprintln!("Unknown command. Try 'help' for usage.");
                process::exit(1);
            },
        }
    } else {
        let first_line = {
            let stdin = io::stdin();
            let mut lines = stdin.lock().lines();
            lines.next().and_then(Result::ok)
        };

        if let Some(first_line) = first_line {
            let trimmed = first_line.trim();
            if trimmed == "xboard" {
                let stdin = io::stdin();
                let mut lines = stdin.lock().lines();
                protocols::xboard::main_loop(&mut lines);
            } else if trimmed == "uci" {
                protocols::uci::main_loop(Some("uci".to_string()));
            } else {
                protocols::uci::main_loop(Some(trimmed.to_string()));
            }
        }
    }
}
