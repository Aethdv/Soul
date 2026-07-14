//! Correction-history instrumentation: is a table dead, sparse, or noisy?
//!
//! Four numbers per table answer the question that key-shape guessing can't:
//! - hit rate (`hits/reads`): does a read find a trained value? Low = dead key.
//! - updates: how often the table is trained, its sampling density.
//! - mean |correction|: how large the value is when it hits. Tiny = noise.
//! - saturation: share of hits pinned at the clamp, railed rather than averaged.
//!
//! A table that reads often but hits rarely is sparse; one that hits but
//! corrects near zero is noise; one with few updates is starved; one that
//! saturates is railing instead of learning. Each is a distinct signature.
//!
//! Compiled only under the `corrstats` feature; zero cost in release builds.

use std::{
    io::IsTerminal,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use crate::{
    color,
    engine::{
        history::{CORRECTION_LIMIT, CORRECTION_SCALE, CORRECTION_WEIGHT_SCALE},
        search_params::SearchParams,
    },
};

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const HEADER: color::Rgb = (218, 165, 32);

const TABLES: usize = 3;
const NAMES: [&str; TABLES] = ["pawn", "minor", "major"];

#[rustfmt::skip]
const COLS: [(&str, usize, bool); 8] = [
    ("table",   6, false),
    ("reads",   8, true),
    ("hit%",    7, true),
    ("updates", 8, true),
    ("mean|c|", 9, true),
    ("wt",      4, true),
    ("eff cp",  7, true),
    ("sat%",    7, true),
];

#[derive(Clone, Copy)]
pub enum Table {
    Pawn = 0,
    Minor = 1,
    Major = 2,
}

struct Counters {
    reads: AtomicU64,
    hits: AtomicU64,
    /// Sum of |value| over hits, in raw fixed-point (`/CORRECTION_SCALE` → cp).
    abs_sum: AtomicU64,
    /// Hits whose |value| is pinned at `CORRECTION_LIMIT`, the EMA railed,
    /// so the entry is a clamped extreme, not a learned average.
    saturated: AtomicU64,
    updates: AtomicU64,
}

impl Counters {
    const fn new() -> Self {
        Self {
            reads: AtomicU64::new(0),
            hits: AtomicU64::new(0),
            abs_sum: AtomicU64::new(0),
            saturated: AtomicU64::new(0),
            updates: AtomicU64::new(0),
        }
    }
}

static STATS: [Counters; TABLES] = [Counters::new(), Counters::new(), Counters::new()];

/// Record one read of `table` returning raw fixed-point `value`.
/// A nonzero value is a hit; its magnitude feeds the mean.
#[inline]
pub fn record_read(table: Table, value: i32) {
    let c = &STATS[table as usize];
    c.reads.fetch_add(1, Relaxed);
    if value != 0 {
        c.hits.fetch_add(1, Relaxed);
        c.abs_sum.fetch_add(value.unsigned_abs() as u64, Relaxed);
        if value.abs() >= CORRECTION_LIMIT {
            c.saturated.fetch_add(1, Relaxed);
        }
    }
}

/// Record one training update of `table`.
#[inline]
pub fn record_update(table: Table) {
    STATS[table as usize].updates.fetch_add(1, Relaxed);
}

/// Print the per-table summary. Call once after a workload (e.g. bench).
///
/// `mean|c|` is the learned correction's raw magnitude; `eff` folds in the
/// blend weight, the centipawns the table actually moves the eval, the
/// number that ranks tables against each other. `hit%` and `sat%` carry the
/// only true verdicts; a cold hit rate means a dead key, high saturation
/// means entries pinned at the clamp. The magnitudes are descriptive; only
/// an SPRT says whether they're worth their slot.
pub fn report() {
    let weights = weights();
    let ansi = std::io::stdout().is_terminal();

    // Title and column labels in the section-header gold.
    let gold = if ansi { color::ansi_fg(HEADER) } else { String::new() };
    let (bold, reset) = if ansi { (BOLD, RESET) } else { ("", "") };

    println!("\n{gold}{bold}Correction History stats{reset}");
    let header: Vec<String> = COLS.iter().enumerate().map(|(i, &(label, ..))| cell(label, i, None, false)).collect();
    println!("  {gold}{bold}{}{reset}", header.join(" "));

    for i in 0..TABLES {
        let c = &STATS[i];
        let hit = hit_pct(i);
        let sat = sat_pct(i);

        let hit_rgb = color::advantage((hit / 35.0 - 1.0).clamp(-1.0, 1.0));
        let sat_rgb = color::advantage(1.0 - sat / 25.0);

        let row = [
            cell(NAMES[i], 0, None, ansi),
            cell(&human(c.reads.load(Relaxed)), 1, None, ansi),
            cell(&format!("{hit:.1}%"), 2, Some(hit_rgb), ansi),
            cell(&human(c.updates.load(Relaxed)), 3, None, ansi),
            cell(&format!("{:.2}cp", mean_cp(i)), 4, None, ansi),
            cell(&weights[i].to_string(), 5, None, ansi),
            cell(&format!("{:.2}", eff_cp(i, weights[i])), 6, None, ansi),
            cell(&format!("{sat:.1}%"), 7, Some(sat_rgb), ansi),
        ];
        println!("  {}", row.join(" "));
    }
}

fn hit_pct(i: usize) -> f64 {
    let (reads, hits) = (STATS[i].reads.load(Relaxed), STATS[i].hits.load(Relaxed));
    if reads > 0 { 100.0 * hits as f64 / reads as f64 } else { 0.0 }
}

fn mean_cp(i: usize) -> f64 {
    let (hits, abs_sum) = (STATS[i].hits.load(Relaxed), STATS[i].abs_sum.load(Relaxed));
    if hits > 0 { abs_sum as f64 / hits as f64 / CORRECTION_SCALE as f64 } else { 0.0 }
}

fn eff_cp(i: usize, weight: i32) -> f64 {
    mean_cp(i) * weight as f64 / f64::from(CORRECTION_WEIGHT_SCALE)
}

/// Share of hits pinned at `CORRECTION_LIMIT`, clamped extremes, not averages.
fn sat_pct(i: usize) -> f64 {
    let (hits, sat) = (STATS[i].hits.load(Relaxed), STATS[i].saturated.load(Relaxed));
    if hits > 0 { 100.0 * sat as f64 / hits as f64 } else { 0.0 }
}

/// Blend weight each table contributes at, over `CORRECTION_WEIGHT_SCALE`.
/// Pawn is unscaled, so it sits at full weight.
fn weights() -> [i32; TABLES] {
    let sp = SearchParams::default();
    [CORRECTION_WEIGHT_SCALE, sp.minor_corr_weight, sp.major_corr_weight]
}

/// Hooman counters.
fn human(n: u64) -> String {
    match n {
        1_000_000.. => format!("{:.2}M", n as f64 / 1e6),
        1_000.. => format!("{:.1}K", n as f64 / 1e3),
        _ => n.to_string(),
    }
}

/// Pad `text` to the given column's width and alignment, then wrap in `rgb`
/// if coloring is on. Padding is applied first so the visible width holds
/// regardless of the zero-width escape codes.
fn cell(text: &str, col: usize, rgb: Option<color::Rgb>, ansi: bool) -> String {
    let (_, w, right) = COLS[col];
    let padded = if right { format!("{text:>w$}") } else { format!("{text:<w$}") };

    match rgb {
        Some(c) if ansi => format!("{}{padded}{RESET}", color::ansi_fg(c)),
        _ => padded,
    }
}
