//! CLI entry point and protocol autodetection.

#[cfg(feature = "datagen")]
use std::sync::{Arc, atomic::AtomicBool};
use std::{
    env,
    io::{self, BufRead},
    process,
    str::FromStr,
};

use soul::{
    core::board::{Position, STARTPOS},
    protocols, tools,
};

fn main() {
    let args: Vec<String> = env::args().collect();
    let Some(command) = args.get(1) else {
        return protocol_session();
    };

    let rest: Vec<&str> = args[2..].iter().map(String::as_str).collect();

    match command.as_str() {
        "help" | "--help" | "-h" => protocols::uci::print_help(true),
        "license" | "--license" => protocols::uci::print_license(),
        "version" | "--version" | "-V" => println!("soul v{}", soul::VERSION),

        "bench" => {
            soul::cli::banner();
            tools::bench::run(arg(&rest, 0).unwrap_or(tools::bench::DEFAULT_DEPTH), arg(&rest, 1).unwrap_or(16));
        },
        "perft" | "divide" => {
            let bulk = rest.iter().any(|a| a.eq_ignore_ascii_case("bulk"));
            tools::perft::run(&Position::from_fen(STARTPOS), arg(&rest, 0).unwrap_or(5), command == "divide", bulk);
        },

        "spsa" => tools::spsa::run(&rest),
        #[cfg(feature = "rigs")]
        "speedtest" => tools::speedtest::run(arg(&rest, 0).unwrap_or(0)),
        #[cfg(feature = "datagen")]
        "datagen" => {
            if !tools::datagen::run(&rest, &Arc::new(AtomicBool::new(false))) {
                process::exit(1);
            }
        },
        "genfens" => tools::genfens::run(&rest),
        #[cfg(feature = "rigs")]
        "atb" => tools::attack_table_bench::run(&rest),
        #[cfg(feature = "dataset")]
        "dataset" => {
            if !tools::dataset::cli::run(&rest) {
                process::exit(1);
            }
        },

        "go" => protocols::uci::run_cli_go(&rest),
        "gopretty" => protocols::uci::run_cli_pretty(arg(&rest, 0)),

        _ if command.contains(' ') => protocols::uci::run_commands(&args[1..]),
        _ => {
            eprintln!("Unknown command. Try 'help' for usage.");
            process::exit(1);
        },
    }
}

fn arg<T: FromStr>(rest: &[&str], i: usize) -> Option<T> { rest.get(i).and_then(|s| s.parse().ok()) }

fn protocol_session() {
    soul::cli::banner();

    let stdin = io::stdin();
    let mut lines = stdin.lock().lines();
    let Some(Ok(first)) = lines.next() else { return };

    if first.trim() == "xboard" {
        protocols::xboard::main_loop(&mut lines);
        return;
    }

    // UCI starts its own stdin listener.
    drop(lines);
    protocols::uci::main_loop(Some(first));
}
