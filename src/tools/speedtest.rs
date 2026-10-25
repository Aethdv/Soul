//! Search throughput over a larger fixed suite of positions.

use std::{
    env,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::{
    color::{self, BOLD, RESET, Rgb},
    core::{board::Position, defs::Protocol, util::fmt_comma},
    engine::{
        history::History,
        search::{Limits, SearchConfig, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
};

const DIM: Rgb = (108, 112, 134);
const LAVENDER: Rgb = (180, 190, 254);
const TEXT: Rgb = (205, 214, 244);
const PEACH: Rgb = (250, 179, 135);

const FENS: &str = include_str!("../data/speedtest.fens");

pub fn run(positions: usize) {
    let lines = FENS.lines();
    let fens: Vec<&str> = if positions > 0 { lines.take(positions).collect() } else { lines.collect() };
    let total = fens.len();
    let start = Instant::now();
    let stop_signal = Arc::new(AtomicBool::new(false));

    let mut total_nodes: u64 = 0;

    // Cleared between positions rather than reallocated.
    let tt = Arc::new(TranspositionTable::new(16, 1));

    println!();
    println!("  Running speedtest on {total} positions...");
    println!();

    io::stdout().flush().ok();

    for (i, fen) in fens.iter().enumerate() {
        let board = Position::from_fen(fen);

        // Budgets shrink as the game progresses.
        let move_number = u64::from(board.fullmove_number);
        let movetime_ms = 50_000 / (2 * move_number + 15);
        let limits = Limits { movetime: movetime_ms, silent: true, protocol: Protocol::Uci, ..Default::default() };

        let trail = vec![board.hash];
        stop_signal.store(false, Ordering::Relaxed);

        // SAFETY: the searches run inline on this thread, so none is in flight.
        unsafe { tt.clear(1) };

        let cfg = SearchConfig::new(limits, Instant::now(), stop_signal.clone(), 0, SearchParams::default());
        let mut history = History::new();
        let mut searcher = Searcher::new(&cfg, &board, &trail, tt.clone());

        searcher.iterative_deepening(&mut history);
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

    let nps_formatted = fmt_comma(nps);
    let nodes_formatted = fmt_comma(total_nodes);

    let binary_name = env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Soul".to_string());

    let arch = arch_string();
    let features = feature_flags();

    let (dim, label, text, peach) = (color::ansi_fg(DIM), color::ansi_fg(LAVENDER), color::ansi_fg(TEXT), color::ansi_fg(PEACH));
    let rule = format!("  {dim}──────────────────────────────────────────────────{RESET}");
    let row = |name: &str, value: &str| println!("   {label}{name:<15}{RESET} {text}{value}{RESET}");

    println!("\n");
    println!("{rule}");
    row("Binary", &binary_name);
    row("Version", crate::VERSION);
    row("Rust", env!("SOUL_RUSTC"));
    row("Arch", arch);
    row("Features", &features);
    println!("{rule}");
    row("Positions", &total.to_string());
    row("Nodes", &format!("{BOLD}{nodes_formatted}"));
    row("Time", &format!("{BOLD}{:.3} s", elapsed.as_secs_f64()));
    println!("   {peach}{:<15}{RESET} {BOLD}{peach}{nps_formatted}{RESET}", "NPS");
    println!("{rule}");
    println!();
}

// AVX2 is the build minimum.
const fn arch_string() -> &'static str {
    if cfg!(target_feature = "avx512f") {
        "x86-64-avx512"
    } else if cfg!(target_feature = "bmi2") {
        "x86-64-bmi2"
    } else {
        "x86-64-avx2"
    }
}

fn feature_flags() -> String {
    let parts: &[&str] = &[
        #[cfg(target_feature = "avx512f")]
        "AVX512F",
        #[cfg(target_feature = "avx512bw")]
        "AVX512BW",
        #[cfg(target_feature = "avx512vl")]
        "AVX512VL",
        #[cfg(target_feature = "avx512dq")]
        "AVX512DQ",
        #[cfg(target_feature = "avx512cd")]
        "AVX512CD",
        #[cfg(target_feature = "avx512vbmi")]
        "AVX512VBMI",
        #[cfg(target_feature = "avx512vbmi2")]
        "AVX512VBMI2",
        #[cfg(target_feature = "avx512vnni")]
        "AVX512VNNI",
        #[cfg(target_feature = "avx512bitalg")]
        "AVX512BITALG",
        #[cfg(target_feature = "avx512vpopcntdq")]
        "AVX512VPOPCNTDQ",
        #[cfg(target_feature = "bmi2")]
        "BMI2",
        #[cfg(target_feature = "avx2")]
        "AVX2",
        #[cfg(target_feature = "sse4.1")]
        "SSE4.1",
        #[cfg(target_feature = "ssse3")]
        "SSSE3",
        #[cfg(target_feature = "popcnt")]
        "POPCNT",
    ];

    if parts.is_empty() { "none".to_string() } else { parts.join(" ") }
}
