//! Piece-Square Table (PSQT) spatial continuity analysis.
//!
//! Each 4-by-8 mirrored half-board is compared pair by adjacent pair against its own mean
//! step. A step far above the rest flags a square the data barely reached.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use crate::{
    engine::{TABLE_SQUARES, Tunable, eval_params},
    report::PIECE_NAMES,
};

const PHASES: [&str; 2] = ["MG", "EG"];
const RANKS: usize = 8;
const FILES: usize = 4;

const _: () = assert!(RANKS * FILES == TABLE_SQUARES);
const ADJACENT_PAIRS: usize = RANKS * (FILES - 1) + (RANKS - 1) * FILES;

const OUTLIER_MULT: f64 = 1.5;
// Centipawn floor so flat tables do not flag rounding jitter as outliers.
// TODO: We might need to rethink this, does not seem to work as intended and kind of goes to shit.
const OUTLIER_FLOOR: f64 = 5.0;
const REPORT_PATH: &str = "correlation-report.txt";

pub(crate) fn run_correlation() {
    let params = eval_params::collect_parameters();
    let values = eval_params::default_values(&params);
    let tables = analyze_tables(&values, &params);

    if let Err(e) = write_report(&tables) {
        eprintln!("Failed to write {REPORT_PATH}: {e}");
        return;
    }

    let total_outliers: usize = tables.iter().map(|t| t.outliers.len()).sum();
    if total_outliers > 0 {
        println!("{REPORT_PATH} written: {total_outliers} outlier pairs found.");
    } else {
        println!("{REPORT_PATH} written: PSQT surface is smooth.");
    }
}

struct AdjacentPair {
    sq_a: usize,
    sq_b: usize,
    diff: i32,
}

struct TableRoughness {
    piece: &'static str,
    phase: &'static str,
    mean: f64,
    max: i32,
    outliers: Vec<AdjacentPair>,
}

fn analyze_tables(values: &[f64], params: &[Tunable]) -> Vec<TableRoughness> {
    let values = values.as_chunks::<TABLE_SQUARES>().0;
    let params = params.as_chunks::<TABLE_SQUARES>().0;

    (0..PIECE_NAMES.len() * PHASES.len())
        .map(|i| analyze_table(&values[i], &params[i], PIECE_NAMES[i / PHASES.len()], PHASES[i % PHASES.len()]))
        .collect()
}

fn analyze_table(values: &[f64], params: &[Tunable], piece: &'static str, phase: &'static str) -> TableRoughness {
    let mut pairs = Vec::with_capacity(ADJACENT_PAIRS);
    let mut push = |sq_a: usize, sq_b: usize| {
        if !params[sq_a].is_fixed && !params[sq_b].is_fixed {
            pairs.push(AdjacentPair { sq_a, sq_b, diff: (values[sq_a] - values[sq_b]).round().abs() as i32 });
        }
    };

    for rank in 0..RANKS {
        for file in 0..FILES {
            let sq = rank * FILES + file;
            if file + 1 < FILES {
                push(sq, sq + 1);
            }
            if rank + 1 < RANKS {
                push(sq, sq + FILES);
            }
        }
    }

    let total: i64 = pairs.iter().map(|p| i64::from(p.diff)).sum();
    let mean = if pairs.is_empty() { 0.0 } else { total as f64 / pairs.len() as f64 };
    let max = pairs.iter().map(|p| p.diff).max().unwrap_or(0);
    let threshold = (mean * OUTLIER_MULT).max(OUTLIER_FLOOR) as i32;
    let outliers = pairs.into_iter().filter(|p| p.diff > threshold).collect();

    TableRoughness { piece, phase, mean, max, outliers }
}

fn write_report(tables: &[TableRoughness]) -> io::Result<()> {
    let file = File::create(REPORT_PATH)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "PSQT Adjacency Analysis")?;
    writeln!(w, "Outlier: a step over {OUTLIER_MULT}× the slice mean, and over {OUTLIER_FLOOR}cp\n")?;

    for t in tables {
        writeln!(w, "  {:>7} {}  mean_diff: {:5.1}  max_diff: {:3}", t.piece, t.phase, t.mean, t.max)?;
        for p in &t.outliers {
            writeln!(w, "    {}↔{}  {:4}", square_name(p.sq_a), square_name(p.sq_b), p.diff)?;
        }
    }

    if tables.iter().all(|t| t.outliers.is_empty()) {
        writeln!(w, "\nNo significant outliers: PSQT surface appears smooth.")?;
    }

    w.flush()
}

fn square_name(idx: usize) -> String {
    let file = (b'A' + (idx % FILES) as u8) as char;
    let rank = idx / FILES + 1;
    format!("{file}{rank}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn square_name_coordinates() {
        for (idx, want) in [(0, "A1"), (3, "D1"), (4, "A2"), (31, "D8")] {
            assert_eq!(square_name(idx), want);
        }
    }
}
