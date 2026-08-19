//! Piece-Square Table (PSQT) spatial continuity analysis.
//!
//! Evaluates gradient smoothness across horizontally mirrored half-boards (4 by 8),
//! flagging adjacent square pairs whose evaluation difference deviates significantly
//! from the piece-phase average.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use crate::engine::eval_params;

const PIECES: [&str; 6] = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
const PHASES: [&str; 2] = ["MG", "EG"];
const RANKS: usize = 8;
const FILES: usize = 4; // Horizontally mirrored half-board (A–D files)
const HALF_BOARD: usize = RANKS * FILES; // 32
const ADJACENT_PAIRS_PER_SLICE: usize = RANKS * (FILES - 1) + (RANKS - 1) * FILES; // 52

/// Multiplier over the mean step size defining an adjacency outlier.
const OUTLIER_MULT: f64 = 1.5;
/// Minimum threshold floor in centipawns to prevent flagging negligible noise on flat tables.
const OUTLIER_FLOOR: f64 = 5.0;

/// Analyzes PSQT spatial continuity across all pieces and writes results to `correlation-report.txt`.
pub fn run_correlation() {
    let values = eval_params::default_values(&eval_params::collect_parameters());
    let slices = analyze_all(&values);

    if let Err(e) = write_report(&slices) {
        eprintln!("Failed to write correlation-report.txt: {e}");
        return;
    }

    let total_outliers: usize = slices.iter().map(|s| s.outliers.len()).sum();
    if total_outliers > 0 {
        println!("correlation-report.txt written: {total_outliers} outlier pairs found.");
    } else {
        println!("correlation-report.txt written: PSQT surface is smooth.");
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
    count: usize,
    mean: f64,
    max: i32,
    outliers: Vec<AdjacentPair>,
}

fn analyze_all(values: &[f64]) -> Vec<PsqtSliceStats> {
    let mut slices = Vec::with_capacity(PIECES.len() * PHASES.len());

    for (p_idx, &piece) in PIECES.iter().enumerate() {
        let base = p_idx * 64;

        for (ph_idx, &phase) in PHASES.iter().enumerate() {
            let offset = base + ph_idx * HALF_BOARD;
            let slice = &values[offset..offset + HALF_BOARD];
            slices.push(analyze_slice(slice, piece, phase));
        }
    }
    slices
}

fn analyze_slice(v: &[f64], piece: &'static str, phase: &'static str) -> PsqtSliceStats {
    let mut pairs = Vec::with_capacity(ADJACENT_PAIRS_PER_SLICE);

    for rank in 0..RANKS {
        for file in 0..FILES {
            let idx = rank * FILES + file;

            // Horizontal step (E-W)
            if file + 1 < FILES {
                pairs.push(make_pair(v, idx, idx + 1));
            }
            // Vertical step (N-S)
            if rank + 1 < RANKS {
                pairs.push(make_pair(v, idx, idx + FILES));
            }
        }
    }

    let total: i32 = pairs.iter().map(|p| p.diff).sum();
    let n = pairs.len();
    let mean = total as f64 / n.max(1) as f64;
    let max = pairs.iter().map(|p| p.diff).max().unwrap_or(0);
    let threshold = (mean * OUTLIER_MULT).max(OUTLIER_FLOOR) as i32;
    let outliers: Vec<AdjacentPair> = pairs.into_iter().filter(|p| p.diff > threshold).collect();

    PsqtSliceStats { piece, phase, count: n, mean, max, outliers }
}

#[inline(always)]
fn make_pair(v: &[f64], sq_a: usize, sq_b: usize) -> AdjacentPair {
    AdjacentPair { sq_a, sq_b, diff: (v[sq_a] - v[sq_b]).round().abs() as i32 }
}

fn write_report(slices: &[PsqtSliceStats]) -> io::Result<()> {
    let file = File::create("correlation-report.txt")?;
    let mut w = BufWriter::new(file);

    writeln!(w, "PSQT Adjacency Analysis\n")?;

    let mut summary_lines: Vec<String> = Vec::new();

    for s in slices {
        writeln!(w, "  {:>7} {}  mean_diff: {:5.1}  max_diff: {:3}  pairs: {}", s.piece, s.phase, s.mean, s.max, s.count,)?;

        for p in &s.outliers {
            let line = format!("{:>7} {}  {}↔{}  {:4}", s.piece, s.phase, square_name(p.sq_a), square_name(p.sq_b), p.diff,);
            writeln!(w, "    {line}")?;
            summary_lines.push(line);
        }
    }

    writeln!(w)?;
    if summary_lines.is_empty() {
        writeln!(w, "No significant outliers: PSQT surface appears smooth.")?;
    } else {
        writeln!(w, "Outlier summary (|diff| > {OUTLIER_MULT}× piece mean):")?;
        for line in &summary_lines {
            writeln!(w, "  {line}")?;
        }
    }

    w.flush()
}

// Converts (0..31) → (A1..D8).
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
