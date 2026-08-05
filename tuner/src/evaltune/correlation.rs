//! PSQT adjacency analysis: finds outlier differences between adjacent
//! squares in the mirrored half-board, written to correlation-report.txt.

use std::{
    fs::File,
    io,
    io::{BufWriter, Write},
};

use super::engine::eval_params;

const PIECES: [&str; 6] = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
const PHASES: [&str; 2] = ["MG", "EG"];
const RANKS: usize = 8;
const FILES: usize = 4; // half-board, mirrored
const HALF: usize = RANKS * FILES; // 32

// Outlier threshold: adjacent pairs whose difference exceeds
// 1.5× the piece-phase mean (floored at 5 cp) are flagged.
const OUTLIER_MULT: f64 = 1.5;
const OUTLIER_FLOOR: f64 = 5.0;

/// Run PSQT adjacency analysis on the current parameter values.
pub fn run_correlation() {
    let values = eval_params::default_values(&eval_params::collect_parameters());

    let slices = analyse_all(&values);

    if write_report(&slices).is_err() {
        eprintln!("Failed to write correlation-report.txt");
        return;
    }

    let total_outliers: usize = slices.iter().map(|s| s.outliers.len()).sum();
    if total_outliers > 0 {
        println!("correlation-report.txt written: {total_outliers} outlier pairs found.");
    } else {
        println!("correlation-report.txt written: PSQT surface is smooth.");
    }
}

struct Pair {
    a: usize,
    b: usize,
    diff: i32,
}

struct SliceStats {
    piece: &'static str,
    phase: &'static str,
    count: usize,
    mean: f64,
    max: i32,
    outliers: Vec<Pair>,
}

fn analyse_all(values: &[f64]) -> Vec<SliceStats> {
    let mut slices = Vec::with_capacity(PIECES.len() * PHASES.len());

    for (p_idx, &piece) in PIECES.iter().enumerate() {
        let base = p_idx * 64;

        for (ph, &phase) in PHASES.iter().enumerate() {
            let v = &values[base + ph * HALF..base + ph * HALF + HALF];
            slices.push(analyse_slice(v, piece, phase));
        }
    }
    slices
}

fn analyse_slice(v: &[f64], piece: &'static str, phase: &'static str) -> SliceStats {
    let mut pairs = Vec::new();

    for rank in 0..RANKS {
        for file in 0..FILES {
            let idx = rank * FILES + file;
            if file + 1 < FILES {
                let nb = idx + 1;
                pairs.push(make_pair(v, idx, nb));
            }
            if rank + 1 < RANKS {
                let nb = idx + FILES;
                pairs.push(make_pair(v, idx, nb));
            }
        }
    }

    let total: i32 = pairs.iter().map(|p| p.diff).sum();
    let n = pairs.len();
    let mean = total as f64 / n.max(1) as f64;
    let max = pairs.iter().map(|p| p.diff).max().unwrap_or(0);
    let threshold = (mean * OUTLIER_MULT).max(OUTLIER_FLOOR) as i32;
    let outliers: Vec<Pair> = pairs.into_iter().filter(|p| p.diff > threshold).collect();

    SliceStats { piece, phase, count: n, mean, max, outliers }
}

fn make_pair(v: &[f64], a: usize, b: usize) -> Pair {
    Pair { a, b, diff: (v[a] - v[b]).round().abs() as i32 }
}

fn write_report(slices: &[SliceStats]) -> io::Result<()> {
    let f = File::create("correlation-report.txt")?;
    let mut w = BufWriter::new(f);

    writeln!(w, "PSQT Adjacency Analysis\n")?;

    let mut all_outlier_lines: Vec<String> = Vec::new();

    for s in slices {
        writeln!(w, "  {:>7} {}  mean_diff: {:5.1}  max_diff: {:3}  pairs: {}", s.piece, s.phase, s.mean, s.max, s.count,)?;

        for p in &s.outliers {
            let line = format!("{:>7} {}  {}↔{}  {:4}", s.piece, s.phase, sq_name(p.a), sq_name(p.b), p.diff,);
            writeln!(w, "    {line}")?;
            all_outlier_lines.push(line);
        }
    }

    writeln!(w)?;
    if all_outlier_lines.is_empty() {
        writeln!(w, "No significant outliers: PSQT surface appears smooth.")?;
    } else {
        writeln!(w, "Outlier summary (|diff| > {OUTLIER_MULT}× piece mean):")?;
        for line in &all_outlier_lines {
            writeln!(w, "  {line}")?;
        }
    }
    w.flush()
}

// Convert (0..31) → (A1..D8).
fn sq_name(idx: usize) -> String {
    let file = (b'A' + (idx % FILES) as u8) as char;
    let rank = idx / FILES + 1;
    format!("{file}{rank}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sq_name_corners() {
        assert_eq!(sq_name(0), "A1");
        assert_eq!(sq_name(3), "D1");
        assert_eq!(sq_name(4), "A2");
        assert_eq!(sq_name(31), "D8");
    }
}
