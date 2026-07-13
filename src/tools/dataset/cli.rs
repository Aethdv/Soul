//! Dataset CLI - inspect, analyze, and encode chess training data.
//!
//! The `.soul` binary format packs board states, scores, and WDL labels
//! into a compact layout optimized for training pipelines.

use std::{
    fs::File,
    io::{self, BufRead, BufReader, BufWriter, Write},
    path::Path,
};

use crate::{
    cli::Help,
    engine::wdl::wdl_model,
    tools::dataset::{self, SoulEntry},
};

// SAFETY: Both byte arrays consist entirely of valid 7-bit ASCII characters (= and -).
const SEP_THICK: &str = unsafe { std::str::from_utf8_unchecked(&[b'='; 80]) };
const SEP_THIN: &str = unsafe { std::str::from_utf8_unchecked(&[b'-'; 80]) };

/// Loads a dataset (Soul binary or viriformat) or prints the error and returns.
/// Must be a macro: a function's `return` can't unwind a foreign scope.
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

/// Slice-pattern dispatch: exhaustive, zero-cost, no manual bounds checks.
/// The `encode` arm is split into two patterns: the match itself proves
/// the argument count, replacing the original's `if args.len() < 3` guard.
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

        ["encode", input, output, ..] => encode(input, output),
        ["encode", ..] => eprintln!("Usage: soul dataset encode <input.epd> <output.soul>"),

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

fn load_any_dataset(path: &str) -> io::Result<Vec<SoulEntry>> {
    if path.ends_with(".viri") || path.ends_with(".vf") {
        dataset::parse_viri_file(path)
    } else {
        dataset::load_encoded(path)
    }
}

fn help() {
    let h = Help::new(28);

    h.header("Usage:");
    println!("  soul dataset <command> [options]");
    h.separator();

    h.header("Commands:");
    h.subcommand_default("inspect", "<path> [count]", "Show first N entries as readable FENs", "10");
    h.subcommand("info", "<path>", "Show dataset statistics");
    h.subcommand("encode", "<input> <output>", "Convert EPD/TXT/FEN file to .soul binary format");
    h.subcommand("deltas", "<path>", "Dump (delta, result, static, search) CSV for analysis");
    h.separator();

    h.header("Examples:");
    h.example("soul dataset inspect data.soul.zst 20");
    h.example("soul dataset info data.soul.zst");
    h.example("soul dataset encode books.epd books.soul.zst");
}

/// Dumps the first `count` positions as FENs alongside their
/// training labels. Run this first after generating a dataset: visual
/// inspection catches encoding bugs and label corruption before they
/// silently poison a multi-hour training run.
fn inspect(path: &str, count: usize) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);
    let show = count.min(entries.len());

    // Lock stdout once, wrap in BufWriter.
    // Without this, each writeln! independently acquires the mutex AND flushes on a tty,
    // measurably painful when dumping hundreds of entries through less.
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let _ = writeln!(out, "Inspecting first {show} entries of {}:", entries.len());
    let _ = writeln!(out, "{SEP_THICK}");

    for (i, entry) in entries.iter().take(count).enumerate() {
        let fen_raw = entry.to_fen();
        let fen = fen_raw.split_once(';').map(|(f, _)| f).unwrap_or(&fen_raw).trim();

        let score = entry.score;
        let result = entry.result;
        let piece_count = entry.occupancy.count_ones();

        let (w, d, l) = wdl_model(i32::from(score), piece_count);
        let wdl_str = format!("W:{:.1}% D:{:.1}% L:{:.1}%", w * 100.0, d * 100.0, l * 100.0);

        let _ = writeln!(out, "[{i:05}] {fen}");
        let _ = writeln!(out, "       Search: {:+5}  Result: {}  {}  Pieces: {}", score, result, wdl_str, piece_count,);
        let _ = writeln!(out, "{SEP_THIN}");
    }
}

/// Single-pass statistical summary.
/// The output surfaces data-quality problems at a glance:
/// heavily skewed W/D/L ratios usually point to broken adjudication,
/// a non-zero average static score suggests the generator has a systematic side-to-move bias.
fn info(path: &str) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);

    let entry_size = std::mem::size_of::<SoulEntry>();
    let raw_size = entries.len() * entry_size;

    // Widen accumulators to prevent precision loss on 100M+ entry datasets.
    // Scores use i64 to prevent overflow: WDL uses f64 because f32 loses
    // precision past 2²⁴ (~16.8M): adding 1.0 to 16_777_216.0f32 is a no-op.
    let mut total_search = 0i64;
    let mut white_wins = 0u64;
    let mut draw_count = 0u64;
    let mut black_wins = 0u64;

    for entry in &entries {
        let score = entry.score;
        let result = entry.result;
        let stm_white = (entry.stm_and_ep & 0x80) == 0;

        total_search += i64::from(score);

        // Result: 0=loss, 1=draw, 2=win from side-to-move perspective.
        // Convert to white-relative for the breakdown.
        let white_result = if stm_white { result } else { 2 - result };

        if white_result >= 2 {
            white_wins += 1;
        } else if white_result == 0 {
            black_wins += 1;
        } else {
            draw_count += 1;
        }
    }

    let n = entries.len() as f64;

    println!("Dataset: {path}");
    println!("Entries: {}", entries.len());
    println!("Entry size: {entry_size} bytes");
    println!("Raw size: {:.2} MB", (raw_size as f64) / 1024.0 / 1024.0);
    println!();
    println!("Results:");
    println!("  White wins: {white_wins} ({:.1}%)", (white_wins as f64) / n * 100.0);
    println!("  Black wins: {black_wins} ({:.1}%)", (black_wins as f64) / n * 100.0);
    println!("  Draws:      {draw_count} ({:.1}%)", (draw_count as f64) / n * 100.0);
    println!();
    println!("Average Scores:");
    println!("  Search:  {:+.2} cp", (total_search as f64) / n);
}

fn dump_scores(path: &str) {
    let entries: Vec<SoulEntry> = load_or_bail!(path);
    let out_path = format!("{path}.scores.csv");
    let file = std::fs::File::create(&out_path).expect("Failed to create output file");
    let mut out = BufWriter::new(file);

    let _ = writeln!(out, "result,score");

    for entry in &entries {
        let _ = writeln!(out, "{:.1},{}", f64::from(entry.result) / 2.0, entry.score);
    }
    println!("Saved → {out_path}");
}

/// Converts plaintext EPD/FEN into compact `.soul` binary format.
/// Handles raw and zstd-compressed inputs transparently.
///
/// Performance on large-scale conversion (100M+ positions):
/// - Line buffer is reused across iterations: zero per-line allocation.
/// - UTF-8 validation is skipped: EPD is pure 7-bit ASCII by definition.
/// - Entry vector is pre-sized from file metadata to minimize growth.
fn encode(input: &str, output: &str) {
    println!("Encoding {input} -> {output}...");

    let file = match File::open(input) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening input file: {e}");
            return;
        },
    };

    // Grab file size before the reader chain consumes the handle.
    let file_len = file.metadata().map_or(0, |m| m.len());

    let is_zst = Path::new(input).extension().is_some_and(|ext| ext.eq_ignore_ascii_case("zst"));

    let mut reader: Box<dyn BufRead> = if is_zst {
        match zstd::Decoder::new(file) {
            Ok(d) => Box::new(BufReader::new(d)),
            Err(e) => {
                eprintln!("Error creating zstd decoder: {e}");
                return;
            },
        }
    } else {
        Box::new(BufReader::new(file))
    };

    // EPD lines average 90 bytes. For zstd, assume ~4× compress ratio.
    let est_bytes = if is_zst { file_len.saturating_mul(4) } else { file_len };
    let est_entries = (est_bytes / 90).max(256) as usize;

    let mut entries = Vec::with_capacity(est_entries);
    let mut buf = Vec::with_capacity(128);
    let mut good = 0usize;
    let mut bad = 0usize;
    let mut line_idx = 0usize;

    loop {
        buf.clear();
        match reader.read_until(b'\n', &mut buf) {
            Ok(0) => break,
            Err(_) => {
                // I/O error: skip, matching lines() semantics.
                line_idx += 1;
                continue;
            },
            Ok(_) => {},
        }

        // Strip trailing line terminators.
        // Windows sends \r\n, Unix \n, and ancient Mac tools occasionally emit bare \r.
        while buf.last().is_some_and(|&b| b == b'\n' || b == b'\r') {
            buf.pop();
        }

        // SAFETY: EPD/FEN is strictly 7-bit ASCII: piece chars (KQRBNPkqrbnp),
        // coordinates (a-h, 1-8), digits, slashes, spaces, and annotation tokens.
        // Multi-byte UTF-8 codepoints are structurally impossible.
        let buf = buf.strip_prefix(b"\xEF\xBB\xBF").unwrap_or(&buf);
        let Ok(line) = std::str::from_utf8(buf) else {
            line_idx += 1;
            bad += 1;
            continue;
        };

        if let Some(entry) = dataset::parse_epd_entry(line) {
            entries.push(entry);
            good += 1;
        } else {
            if bad < 5 {
                eprintln!("Warning: failed to parse line {}: '{line}'", line_idx + 1);
            }

            bad += 1;
        }

        line_idx += 1;
    }

    println!("Parsed {good} entries. Failed lines: {bad}");

    if entries.is_empty() {
        println!("No entries found to save.");
    } else {
        match dataset::save_encoded(output, &entries) {
            Ok(()) => println!("\x1b[92mSuccess!\x1b[0m Saved to {output}"),
            Err(e) => eprintln!("Error saving output: {e}"),
        }
    }
}
