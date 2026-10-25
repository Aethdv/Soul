//! `soul dataset`: inspect a viriformat file, summarize it, or dump its scores.

use std::io::{self, BufWriter, Write};

use crate::{
    cli::Help,
    core::util::pct,
    engine::wdl::wdl_model,
    tools::dataset::{self, SoulEntry, flip_result},
};

pub fn run(args: &[&str]) -> bool {
    match args {
        [] | ["help" | "--help" | "-h", ..] => {
            help();
            true
        },
        ["inspect", path, rest @ ..] => {
            let Some(count) = rest.first().map_or(Some(10), |value| value.parse().ok()) else {
                eprintln!("Invalid count: {}", rest[0]);
                return false;
            };
            inspect(path, count)
        },
        ["inspect"] => {
            eprintln!("Usage: soul dataset inspect <path> [count]");
            false
        },
        ["info", path, ..] => info(path),
        ["info"] => {
            eprintln!("Usage: soul dataset info <path>");
            false
        },
        ["deltas" | "scores", path, ..] => {
            let mut ok = true;
            for path in path.split(',').map(str::trim) {
                if let Err(e) = write_scores_csv(path) {
                    eprintln!("Error writing {path}: {e}");
                    ok = false;
                }
            }
            ok
        },
        ["deltas" | "scores"] => {
            eprintln!("Usage: soul dataset scores <path,...>");
            false
        },
        [unknown, ..] => {
            eprintln!("Unknown dataset command: {unknown}");
            help();
            false
        },
    }
}

fn load_entries(path: &str) -> io::Result<Vec<SoulEntry>> {
    dataset::parse_viri_file(path, &dataset::ReplayFilter::UNRESTRICTED)
        .map(|(entries, ..)| entries)
        .map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")))
}

fn help() {
    let h = Help::new(28);

    h.header("Usage:");
    println!("  soul dataset <command> [options]");
    h.separator();

    h.header("Commands:");
    h.subcommand_default("inspect", "<path> [count]", "Show first N entries as readable FENs", "10");
    h.subcommand("info", "<path>", "Show dataset statistics");
    h.subcommand("scores", "<path,...>", "Write <path>.scores.csv with result and score");
    h.separator();

    h.header("Examples:");
    h.example("soul dataset inspect data.vf 20");
    h.example("soul dataset info data.vf");
}

fn inspect(path: &str, count: usize) -> bool {
    let entries = match load_entries(path) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Error loading dataset: {e}");
            return false;
        },
    };

    let display_count = count.min(entries.len());
    let stdout = io::stdout();
    let mut out = BufWriter::new(stdout.lock());

    let _ = writeln!(out, "Inspecting first {display_count} entries of {}:", entries.len());
    let _ = writeln!(out, "{:=<80}", "");

    for (i, entry) in entries.iter().take(count).enumerate() {
        let fen = entry.to_fen();
        let score = entry.score;
        let result = entry.result;
        let piece_count = entry.occupancy.count_ones();
        let (win, draw, loss) = wdl_model(i32::from(score), entry.material_count());
        let wdl = format!("W:{:.1}% D:{:.1}% L:{:.1}%", win * 100.0, draw * 100.0, loss * 100.0);
        let _ = writeln!(out, "[{i:05}] {fen}");
        let _ = writeln!(out, "       Search: {:+5}  Result: {}  {}  Pieces: {}", score, result, wdl, piece_count);
        let _ = writeln!(out, "{:-<80}", "");
    }

    true
}

fn info(path: &str) -> bool {
    let entries = match load_entries(path) {
        Ok(entries) => entries,
        Err(e) => {
            eprintln!("Error loading dataset: {e}");
            return false;
        },
    };

    let entry_size = std::mem::size_of::<SoulEntry>();
    let record_bytes = entries.len() * entry_size;

    let mut total_search = 0i64;
    let mut scored_count = 0u64;
    let mut white_wins = 0u64;
    let mut draws = 0u64;
    let mut black_wins = 0u64;

    for entry in &entries {
        if entry.score != SoulEntry::NO_SCORE {
            total_search += i64::from(entry.score);
            scored_count += 1;
        }

        match flip_result(entry.result, entry.stm()) {
            2 => white_wins += 1,
            0 => black_wins += 1,
            _ => draws += 1,
        }
    }

    let total = entries.len() as u64;

    println!("Dataset: {path}");
    println!("Entries: {}", entries.len());
    println!("Entry size: {entry_size} bytes");
    println!("Record bytes: {:.2} MB", (record_bytes as f64) / 1024.0 / 1024.0);
    println!();
    println!("Results:");
    println!("  White wins: {white_wins} ({:.1}%)", pct(white_wins, total));
    println!("  Black wins: {black_wins} ({:.1}%)", pct(black_wins, total));
    println!("  Draws:      {draws} ({:.1}%)", pct(draws, total));
    println!();
    println!("Average Score (STM):");

    match (scored_count, total - scored_count) {
        (0, _) => println!("  none (all records unscored)"),
        (count, 0) => println!("  {:+.2} cp", (total_search as f64) / count as f64),
        (count, missing) => println!("  {:+.2} cp ({missing} entries unscored)", (total_search as f64) / count as f64),
    }

    true
}

fn write_scores_csv(path: &str) -> io::Result<()> {
    let entries = load_entries(path)?;
    let out_path = format!("{path}.scores.csv");
    let mut out = BufWriter::new(std::fs::File::create(&out_path)?);

    writeln!(out, "result,score")?;
    for entry in &entries {
        writeln!(out, "{:.1},{}", f64::from(entry.result) / 2.0, entry.score)?;
    }

    out.flush()?;
    println!("Saved -> {out_path}");
    Ok(())
}
