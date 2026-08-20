//! Piece-Square Table (PSQT) spatial continuity analysis.
//!
//! Each 4 by 8 mirrored half-board is compared pair by adjacent pair against its own mean
//! step. A step far above the rest can be a square the data barely reached.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use crate::{
    engine::{Tunable, eval_params},
    report::PIECE_NAMES,
};

const PHASES: [&str; 2] = ["MG", "EG"];
const RANKS: usize = 8;
const FILES: usize = 4; // Horizontally mirrored half-board (A–D files)
const HALF_BOARD: usize = RANKS * FILES; // 32
const ADJACENT_PAIRS_PER_SLICE: usize = RANKS * (FILES - 1) + (RANKS - 1) * FILES; // 52

/// Multiplier over the mean step size defining an adjacency outlier.
const OUTLIER_MULT: f64 = 1.5;
/// Minimum threshold floor in centipawns to prevent flagging negligible noise on flat tables.
const OUTLIER_FLOOR: f64 = 5.0;

const REPORT_PATH: &str = "correlation-report.txt";

/// Analyzes PSQT spatial continuity across all pieces and writes results to `REPORT_PATH`.
pub fn run_correlation() {
    let params = eval_params::collect_parameters();
    let values = eval_params::default_values(&params);
    let slices = analyze_all(&values, &params);

    if let Err(e) = write_report(&slices) {
        eprintln!("Failed to write {REPORT_PATH}: {e}");
        return;
    }

    let total_outliers: usize = slices.iter().map(|s| s.outliers.len()).sum();
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

struct PsqtSliceStats {
    piece: &'static str,
    phase: &'static str,
    mean: f64,
    max: i32,
    outliers: Vec<AdjacentPair>,
}

fn analyze_all(values: &[f64], params: &[Tunable]) -> Vec<PsqtSliceStats> {
    let mut slices = Vec::with_capacity(PIECE_NAMES.len() * PHASES.len());

    for (p_idx, &piece) in PIECE_NAMES.iter().enumerate() {
        let base = p_idx * PHASES.len() * HALF_BOARD;

        for (ph_idx, &phase) in PHASES.iter().enumerate() {
            let offset = base + ph_idx * HALF_BOARD;
            let range = offset..offset + HALF_BOARD;
            slices.push(analyze_slice(&values[range.clone()], &params[range], piece, phase));
        }
    }
    slices
}

fn analyze_slice(v: &[f64], params: &[Tunable], piece: &'static str, phase: &'static str) -> PsqtSliceStats {
    let mut pairs = Vec::with_capacity(ADJACENT_PAIRS_PER_SLICE);
    // The pawn ranks nobody can occupy are fixed, so a step into one measures an untrained
    // default rather than roughness.
    let mut push = |a: usize, b: usize| {
        if !params[a].is_fixed && !params[b].is_fixed {
            pairs.push(make_pair(v, a, b));
        }
    };

    for rank in 0..RANKS {
        for file in 0..FILES {
            let idx = rank * FILES + file;

            // Horizontal step (E-W)
            if file + 1 < FILES {
                push(idx, idx + 1);
            }
            // Vertical step (N-S)
            if rank + 1 < RANKS {
                push(idx, idx + FILES);
            }
        }
    }

    let total: i32 = pairs.iter().map(|p| p.diff).sum();
    let mean = f64::from(total) / pairs.len() as f64;
    let max = pairs.iter().map(|p| p.diff).max().unwrap_or(0);
    let threshold = (mean * OUTLIER_MULT).max(OUTLIER_FLOOR) as i32;
    let outliers: Vec<AdjacentPair> = pairs.into_iter().filter(|p| p.diff > threshold).collect();

    PsqtSliceStats { piece, phase, mean, max, outliers }
}

fn make_pair(v: &[f64], sq_a: usize, sq_b: usize) -> AdjacentPair {
    AdjacentPair { sq_a, sq_b, diff: (v[sq_a] - v[sq_b]).round().abs() as i32 }
}

fn write_report(slices: &[PsqtSliceStats]) -> io::Result<()> {
    let file = File::create(REPORT_PATH)?;
    let mut w = BufWriter::new(file);

    writeln!(w, "PSQT Adjacency Analysis")?;
    writeln!(w, "Outlier: a step over {OUTLIER_MULT}× the slice mean, and over {OUTLIER_FLOOR}cp\n")?;

    for s in slices {
        writeln!(w, "  {:>7} {}  mean_diff: {:5.1}  max_diff: {:3}", s.piece, s.phase, s.mean, s.max)?;

        for p in &s.outliers {
            writeln!(w, "    {}↔{}  {:4}", square_name(p.sq_a), square_name(p.sq_b), p.diff)?;
        }
    }

    if slices.iter().all(|s| s.outliers.is_empty()) {
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
        assert_eq!(square_name(0), "A1");
        assert_eq!(square_name(3), "D1");
        assert_eq!(square_name(4), "A2");
        assert_eq!(square_name(31), "D8");
    }
}
