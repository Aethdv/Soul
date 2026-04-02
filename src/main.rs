//! CLI entry point and command router.
//!
//! Dispatches commands to development tools (bench, perft, dataset)
//! or enters an interactive protocol loop (UCI/XBoard) on stdin.

use std::{
    io::{self, BufRead},
    sync::Arc,
};

use soul::{
    core::board::{Position, STARTPOS},
    engine, protocols, tools,
};

#[allow(clippy::too_many_lines)]
fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Handle commands
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
                let depth = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(16);
                tools::bench::run(depth);
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
            "speedtest" => {
                let limit = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(0);
                tools::speedtest::run(limit);
            },
            "genfens" => {
                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let genfens_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::genfens::run(&genfens_args, &stop);
            },
            "dataset" => {
                let dataset_args: Vec<&str> = args[2..].iter().map(String::as_str).collect();
                tools::dataset::cli::run(&dataset_args);
            },
            "go" => {
                let limit_args: Vec<String> = args[2..].iter().map(String::to_string).collect();
                protocols::uci::run_cli_go(&limit_args);
            },
            "gopretty" => {
                let depth = args
                    .get(2)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or_else(|| <u8>::try_from(soul::core::defs::MAX_PLY).unwrap());

                let board = Position::from_fen(STARTPOS);
                let limits = engine::search::Limits {
                    depth,
                    protocol: soul::core::defs::Protocol::Uci,
                    ..Default::default()
                };

                let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
                let display = engine::search::SearchDisplay {
                    show_wdl:      true,
                    go_pretty:     true,
                    pretty_print:  false,
                    show_currmove: true,
                    use_ansi:      true,
                };
                let cfg = engine::search::SearchConfig::new_full(
                    limits,
                    std::time::Instant::now(),
                    stop,
                    0,
                    display,
                    engine::search_params::SearchParams::default(),
                );
                let mut searcher = engine::search::Searcher::new(
                    &cfg,
                    &board,
                    &[],
                    engine::history::History::new(),
                    Arc::new(engine::tt::TranspositionTable::new(16)),
                );
                searcher.iterative_deepening();
                if let Some(best_move) = searcher.best_move() {
                    println!("bestmove {}", best_move.to_uci(board.is_frc));
                }
            },
            _ => {
                eprintln!("Unknown command. Try 'help' for usage.");
                std::process::exit(1);
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
