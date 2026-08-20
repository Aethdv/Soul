//! Dataset CLI tool for inspecting, analyzing, and encoding training data.

use std::io::{self, BufWriter, Write};

use crate::{
    cli::Help,
    core::defs::Color,
    engine::wdl::wdl_model,
    tools::dataset::{self, SoulEntry, flip_result},
};

// SAFETY: Byte arrays contain only valid 7-bit ASCII characters.
const SEP_THICK: &str = unsafe { std::str::from_utf8_unchecked(&[b'='; 80]) };
const SEP_THIN: &str = unsafe { std::str::from_utf8_unchecked(&[b'-'; 80]) };

/// Loads a dataset or prints the error to stderr and returns from the caller.
macro_rules! load_or_bail {
    ($path:expr) => {
        match load_any_dataset($path) {
            Ok(entries) => entries,
            Err(e) => {
                eprintln!("Error loading dataset: {e}");
                return;
            },
        }
    };
}

/// Entry point for dataset subcommands.
pub fn run(args: &[&str]) {
    match args {
        [] | ["help" | "--help" | "-h", ..] => help(),

        ["inspect", path, rest @ ..] => {
            let count = rest.first().and_then(|s| s.parse().ok()).unwrap_or(10);
            inspect(path, count);
        },
        ["inspect"] => eprintln!("Usage: soul dataset inspect <path> [count]"),

        ["info", path, ..] => info(path),
        ["info"] => eprintln!("Usage: soul dataset info <path>"),

        ["deltas", path, ..] => {
            for p in path.split(',').map(str::trim) {
                dump_scores(p);
            }
        },
        ["deltas"] => eprintln!("Usage: soul dataset deltas <path>"),

        [unknown, ..] => {
            eprintln!("Unknown dataset command: {unknown}");
            help();
        },
    }
}

/// Replays a viriformat file into the entries every command here reads.
fn load_any_dataset(path: &str) -> io::Result<Vec<SoulEntry>> {
    dataset::parse_viri_file(path, &dataset::ReplayFilter::UNRESTRICTED).map(|(entries, ..)| entries)
}

fn help() {
    let h = Help::new(28);

    h.header("Usage:");
    println!("  soul dataset <command> [options]");
    h.separator();

    h.header("Commands:");
    h.subcommand_default("inspect", "<path> [count]", "Show first N entries as readable FENs", "10");
    h.subcommand("info", "<path>", "Show dataset statistics");
    h.subcommand("deltas", "<path>", "Dump (delta, result, static, search) CSV for analysis");
    h.separator();

    h.header("Examples:");
    h.example("soul dataset inspect data.vf 20");
    h.example("soul dataset info data.vf");
}

/// Dumps the first `count` positions as FEN strings alongside their search and outcome labels.
fn inspect(path: &str, count: usize) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);
    let display_count = count.min(entries.len());

    // Lock stdout and buffer writes to prevent per-line syscall and mutex acquisition overhead.
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let _ = writeln!(out, "Inspecting first {display_count} entries of {}:", entries.len());
    let _ = writeln!(out, "{SEP_THICK}");

    for (i, entry) in entries.iter().take(count).enumerate() {
        let fen_raw = entry.to_fen();
        let fen = fen_raw.split_once(';').map(|(f, _)| f).unwrap_or(&fen_raw).trim();
        let score = entry.score;
        let result = entry.result;
        let piece_count = entry.occupancy.count_ones();
        let (win, draw, loss) = wdl_model(i32::from(score), entry.material_count());
        let wdl_str = format!("W:{:.1}% D:{:.1}% L:{:.1}%", win * 100.0, draw * 100.0, loss * 100.0);
        let _ = writeln!(out, "[{i:05}] {fen}");
        let _ = writeln!(out, "       Search: {:+5}  Result: {}  {}  Pieces: {}", score, result, wdl_str, piece_count);
        let _ = writeln!(out, "{SEP_THIN}");
    }
}

/// Computes a single-pass statistical summary over dataset evaluations and outcomes.
fn info(path: &str) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);

    let entry_size = std::mem::size_of::<SoulEntry>();
    let raw_size = entries.len() * entry_size;

    // 64-bit accumulators prevent overflow on large datasets (100M+ entries).
    let mut total_search = 0i64;
    let mut scored_count = 0u64;
    let mut white_wins = 0u64;
    let mut draw_count = 0u64;
    let mut black_wins = 0u64;

    for entry in &entries {
        let score = entry.score;
        let result = entry.result;
        let is_white_stm = (entry.stm_and_ep & 0x80) == 0;

        if score != SoulEntry::NO_SCORE {
            total_search += i64::from(score);
            scored_count += 1;
        }

        let white_result = flip_result(result, if is_white_stm { Color::White } else { Color::Black });
        if white_result >= 2 {
            white_wins += 1;
        } else if white_result == 0 {
            black_wins += 1;
        } else {
            draw_count += 1;
        }
    }

    let total_entries = entries.len() as f64;

    println!("Dataset: {path}");
    println!("Entries: {}", entries.len());
    println!("Entry size: {entry_size} bytes");
    println!("Raw size: {:.2} MB", (raw_size as f64) / 1024.0 / 1024.0);
    println!();
    println!("Results:");
    println!("  White wins: {white_wins} ({:.1}%)", (white_wins as f64) / total_entries * 100.0);
    println!("  Black wins: {black_wins} ({:.1}%)", (black_wins as f64) / total_entries * 100.0);
    println!("  Draws:      {draw_count} ({:.1}%)", (draw_count as f64) / total_entries * 100.0);
    println!();

    println!("Average Scores:");
    match (scored_count, entries.len() as u64 - scored_count) {
        (0, _) => println!("  Search:  none (all records unscored)"),
        (s, 0) => println!("  Search:  {:+.2} cp", (total_search as f64) / s as f64),
        (s, missing) => println!("  Search:  {:+.2} cp ({missing} entries unscored)", (total_search as f64) / s as f64),
    }
}

/// Dumps position results and search evaluations to a CSV file.
fn dump_scores(path: &str) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);
    let out_path = format!("{path}.scores.csv");
    let file = std::fs::File::create(&out_path).expect("Failed to create output file");
    let mut out = BufWriter::new(file);

    let _ = writeln!(out, "result,score");

    for entry in &entries {
        let _ = writeln!(out, "{:.1},{}", f64::from(entry.result) / 2.0, entry.score);
    }
    println!("Saved -> {out_path}");
}
