//! Search throughput over a fixed suite of middlegame positions.
//!
//! Each position gets a movetime derived from its own move number, so a position
//! nine moves in searches longer than one seventy-five moves in, the way a clock
//! thins out over a game. The report ends with the machine and build it ran on,
//! since a node rate means nothing without them.

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
    core::{board::Position, defs::Protocol, util::format_comma},
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

const SPEEDTEST_FENS: &str = include_str!("../data/speedtest.fens");

/// Search the first `positions` of the suite, or all of them when it is zero.
pub fn run(positions: usize) {
    let all = SPEEDTEST_FENS.lines();
    let fens: Vec<&str> = if positions > 0 { all.take(positions).collect() } else { all.collect() };
    let total = fens.len();
    let start = Instant::now();
    let stop_signal = Arc::new(AtomicBool::new(false));

    let mut total_nodes: u64 = 0;

    // One table, cleared between positions. A fresh one per FEN would fault in
    // sixteen megabytes each time, and that cost lands in the elapsed time the
    // node rate is divided by.
    let tt = Arc::new(TranspositionTable::new(16, 1));

    println!();
    println!("  Running speedtest on {total} positions...");
    println!();

    io::stdout().flush().ok();

    for (i, fen) in fens.iter().enumerate() {
        let board = Position::from_fen(fen);

        // Thinner budgets deeper into the game, taken from the position rather than
        // re-read off the string: a FEN whose last field will not parse would
        // otherwise fall back to move one and take the longest search in the suite.
        let ply = u64::from(board.fullmove_number).saturating_mul(2);
        let move_time = 50000 / (ply + 15);
        let limits = Limits { movetime: move_time, silent: true, protocol: Protocol::Uci, ..Default::default() };

        let history = vec![board.hash];
        // The previous position's search raised the flag when it hit its movetime,
        // so it has to come back down or this one aborts on entry.
        stop_signal.store(false, Ordering::Relaxed);
        // SAFETY: the searches run inline on this thread, so none is in flight.
        unsafe { tt.clear(1) };

        let cfg = SearchConfig::new(limits, Instant::now(), stop_signal.clone(), 0, SearchParams::default());
        let mut history_table = History::new();
        let mut searcher = Searcher::new(&cfg, &board, &history, tt.clone());

        searcher.iterative_deepening(&mut history_table);
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

    let nps_formatted = format_comma(nps);
    let nodes_formatted = format_comma(total_nodes);

    let binary_name = env::current_exe()
        .ok()
        .and_then(|p| p.file_name().map(|s| s.to_string_lossy().into_owned()))
        .unwrap_or_else(|| "Soul".to_string());

    let arch = get_arch_string();
    let features = get_feature_flags();

    let (dim, label, text, peach) = (color::ansi_fg(DIM), color::ansi_fg(LAVENDER), color::ansi_fg(TEXT), color::ansi_fg(PEACH));
    let rule = format!("  {dim}──────────────────────────────────────────────────{RESET}");
    let row = |name: &str, value: &str| println!("   {label}{name:<15}{RESET} {text}{value}{RESET}");

    println!("\n");
    println!("{rule}");
    row("Binary", &binary_name);
    row("Version", env!("CARGO_PKG_VERSION"));
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

/// The target features this binary was built with, subsets and all.
fn get_feature_flags() -> String {
    macro_rules! enabled {
        ($($feature:literal => $label:literal),* $(,)?) => {{
            #[allow(unused_mut)]
            let mut parts: Vec<&str> = Vec::new();
            $(
                #[cfg(target_feature = $feature)]
                parts.push($label);
            )*
            parts
        }};
    }

    let parts = enabled! {
        "avx512f" => "AVX512F",
        "avx512bw" => "AVX512BW",
        "avx512vl" => "AVX512VL",
        "avx512dq" => "AVX512DQ",
        "avx512cd" => "AVX512CD",
        "avx512vbmi" => "AVX512VBMI",
        "avx512vbmi2" => "AVX512VBMI2",
        "avx512vnni" => "AVX512VNNI",
        "avx512bitalg" => "AVX512BITALG",
        "avx512vpopcntdq" => "AVX512VPOPCNTDQ",
        "bmi2" => "BMI2",
        "avx2" => "AVX2",
        "sse4.1" => "SSE4.1",
        "ssse3" => "SSSE3",
        "popcnt" => "POPCNT",
    };

    if parts.is_empty() { "none".to_string() } else { parts.join(" ") }
}

