//! Runtime instrumentation and diagnostics for correction history tables.
//!
//! Five metrics per table:
//! - hit rate (`hit%`): reads that return a non-zero correction, `hits / reads`.
//! - updates: how often the table is trained, its sampling density.
//! - mean magnitude (`mean|c|`): the average correction on a hit.
//! - effective score (`eff cp`): that mean scaled by the table's blend weight, the
//!   centipawns it actually moves the eval, and the number that ranks tables.
//! - saturation (`sat%`): hits pinned at [`CORRECTION_LIMIT`].
//!
//! Compiled under the `corrstats` feature; zero cost in release builds.

use std::{
    io::IsTerminal,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use crate::{
    color::{self, BOLD, GOLD, RESET},
    core::util::human,
    engine::{
        history::{CORRECTION_LIMIT, CORRECTION_SCALE, CORRECTION_WEIGHT_SCALE},
        search_params::SearchParams,
    },
};

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
    /// Accumulated `|value|` across non-zero reads (fixed-point units).
    abs_sum: AtomicU64,
    /// Hits pinned at [`CORRECTION_LIMIT`], where the EMA railed
    /// and the entry is a clamped extreme rather than a learned average.
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

/// Records a read from `table` with the returned raw fixed-point `value`.
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

/// Records a training update to `table`.
#[inline]
pub fn record_update(table: Table) {
    STATS[table as usize].updates.fetch_add(1, Relaxed);
}

/// The per-table summary, printed once after a workload such as bench.
pub fn report() {
    let weights = weights();
    let ansi = std::io::stdout().is_terminal();
    let gold = if ansi { color::ansi_fg(GOLD) } else { String::new() };
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

fn sat_pct(i: usize) -> f64 {
    let (hits, sat) = (STATS[i].hits.load(Relaxed), STATS[i].saturated.load(Relaxed));
    if hits > 0 { 100.0 * sat as f64 / hits as f64 } else { 0.0 }
}

/// Blend weight each table contributes at, over [`CORRECTION_WEIGHT_SCALE`].
/// Pawn is unscaled, so it sits at full weight.
fn weights() -> [i32; TABLES] {
    let sp = SearchParams::default();
    [CORRECTION_WEIGHT_SCALE, sp.minor_corr_weight, sp.major_corr_weight]
}

/// Pads to the column's width and alignment first, then wraps in `rgb`,
/// so the visible width holds regardless of the zero-width escape codes.
fn cell(text: &str, col: usize, rgb: Option<color::Rgb>, ansi: bool) -> String {
    let (_, w, right) = COLS[col];
    let padded = if right { format!("{text:>w$}") } else { format!("{text:<w$}") };

    match rgb {
        Some(c) if ansi => format!("{}{padded}{RESET}", color::ansi_fg(c)),
        _ => padded,
    }
}
