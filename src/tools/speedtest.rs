//! Performance test utility with adaptive time limits.
//!
//! Measures search throughput against a predefined suite of positions,
//! using move-time limits to simulate realistic game conditions.

use std::{
    io::{self, Write},
    process::Command,
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

use crate::{
    core::{board::Position, defs::Protocol},
    engine::{
        history,
        search::{Limits, Searcher},
    },
};

const DIM: &str = "\x1b[38;2;108;112;134m";
const BOLD: &str = "\x1b[1m";
const LAVENDER: &str = "\x1b[38;2;180;190;254m";
const TEXT: &str = "\x1b[38;2;205;214;244m";
const PEACH: &str = "\x1b[38;2;250;179;135m";
const RESET: &str = "\x1b[0m";

const SPEEDTEST_FENS: &str = include_str!("../data/speedtest.fens");
/// Run speedtest with adaptive time per position.
pub fn run(limit: usize) {
    use crate::engine::{search::SearchConfig, search_params::SearchParams};

    let fens: Vec<&str> = if limit > 0 {
        SPEEDTEST_FENS.lines().take(limit).collect()
    } else {
        SPEEDTEST_FENS.lines().collect()
    };
    let mut total_nodes: u64 = 0;
    let total = fens.len();
    let start = Instant::now();
    let stop_signal = Arc::new(AtomicBool::new(false));

    // Print header
    println!();
    println!("  Running speedtest on {total} positions...");
    println!();
    io::stdout().flush().ok();

    for (i, fen) in fens.iter().enumerate() {
        let board = Position::from_fen(fen);
        let ply = fen
            .rsplit_once(' ')
            .and_then(|(_, s)| s.parse::<u64>().ok())
            .unwrap_or(1)
            .saturating_mul(2);

        let move_time = 50000 / (ply + 15);

        let limits = Limits {
            movetime: move_time,
            silent: true,
            protocol: Protocol::Uci,
            ..Default::default()
        };

        let history = vec![board.hash];
        stop_signal.store(false, std::sync::atomic::Ordering::Release);

        let cfg = SearchConfig::new(
            limits.clone(),
            Instant::now(),
            stop_signal.clone(),
            0,
            SearchParams::default(),
        );
        let mut searcher = Searcher::new(&cfg, &board, &history, history::History::new());

        searcher.iterative_deepening();
        total_nodes += searcher.nodes;

        let bar_width = 40;
        let progress = (i + 1) as f32 / total as f32;
        let filled = (bar_width as f32 * progress) as usize;
        let bar: String = "=".repeat(filled) + &" ".repeat(bar_width - filled);
        print!("\r  [{}] {:>5.1}%  ({}/{})", bar, progress * 100.0, i + 1, total);
        io::stdout().flush().ok();
    }

    let elapsed = start.elapsed();
    let nps = (total_nodes as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64;

    let nps_formatted = format_with_separators(nps);
    let nodes_formatted = format_with_separators(total_nodes);

    let binary_name = std::env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Soul".to_string());

    let rustc_version = get_rustc_version();
    let arch = get_arch_string();
    let features = get_feature_flags();

    println!("\n");
    println!("  {DIM}────────────────────────────────────────────────{RESET}");
    println!("   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}", "Binary", binary_name);
    println!(
        "   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}",
        "Version",
        env!("CARGO_PKG_VERSION")
    );
    println!("   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}", "Rust", rustc_version);
    println!("   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}", "Arch", arch);
    println!("   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}", "Features", features);
    println!("  {DIM}────────────────────────────────────────────────{RESET}");
    println!("   {LAVENDER}{:<15}{RESET} {TEXT}{}{RESET}", "Positions", total);
    println!("   {LAVENDER}{:<15}{RESET} {BOLD}{TEXT}{}{RESET}", "Nodes", nodes_formatted);
    println!(
        "   {LAVENDER}{:<15}{RESET} {BOLD}{TEXT}{:.3} s{RESET}",
        "Time",
        elapsed.as_secs_f64()
    );
    println!("   {PEACH}{:<15}{RESET} {BOLD}{PEACH}{}{RESET}", "NPS", nps_formatted);
    println!("  {DIM}────────────────────────────────────────────────{RESET}");
    println!();
}

/// Build architecture string based on target features
const fn get_arch_string() -> &'static str {
    #[cfg(target_feature = "avx512f")]
    {
        "x86-64-avx512"
    }
    #[cfg(all(target_feature = "bmi2", not(target_feature = "avx512f")))]
    {
        "x86-64-bmi2"
    }
    #[cfg(all(target_feature = "avx2", not(target_feature = "bmi2")))]
    {
        "x86-64-avx2"
    }
    #[cfg(all(target_feature = "sse4.1", not(target_feature = "avx2")))]
    {
        "x86-64-sse41"
    }
    #[cfg(all(target_feature = "popcnt", not(target_feature = "sse4.1")))]
    {
        "x86-64-popcnt"
    }
    #[cfg(not(any(
        target_feature = "popcnt",
        target_feature = "sse4.1",
        target_feature = "avx2",
        target_feature = "bmi2",
        target_feature = "avx512f"
    )))]
    {
        "generic"
    }
}

/// Get a compact string of enabled CPU features
fn get_feature_flags() -> String {
    #[allow(unused_mut)]
    let mut parts: Vec<&str> = Vec::new();

    #[cfg(target_feature = "avx512f")]
    parts.push("AVX512");
    #[cfg(target_feature = "bmi2")]
    parts.push("BMI2");
    #[cfg(target_feature = "avx2")]
    parts.push("AVX2");
    #[cfg(target_feature = "sse4.1")]
    parts.push("SSE4.1");
    #[cfg(target_feature = "ssse3")]
    parts.push("SSSE3");
    #[cfg(target_feature = "popcnt")]
    parts.push("POPCNT");

    if parts.is_empty() {
        "none".to_string()
    } else {
        parts.join(" ")
    }
}

fn format_with_separators(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}

/// Get rustc version at runtime.
fn get_rustc_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map_or_else(
            || "N/A".to_string(),
            |s| {
                s.strip_prefix("rustc ")
                    .unwrap_or(&s)
                    .split_whitespace()
                    .next()
                    .unwrap_or("unknown")
                    .to_string()
            },
        )
}
