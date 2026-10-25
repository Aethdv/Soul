//! Correction-history counters. Hits are reads returning a nonzero correction.

use std::{
    io::{self, IsTerminal},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

#[cfg(not(feature = "nocorr"))]
use crate::engine::history::CORRECTION_LIMIT;
use crate::{
    color::{self, BOLD, GOLD_PEN, RESET},
    core::util::{fmt_count, pct, ratio},
    engine::{history::CORRECTION_SCALE, search_params::SearchParams},
};

const REPORT_WEIGHT_SCALE: i32 = 256;
const NAMES: [&str; 3] = ["pawn", "minor", "major"];

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
pub(super) enum Table {
    Pawn = 0,
    Minor = 1,
    Major = 2,
}

struct Counters {
    reads: AtomicU64,
    hits: AtomicU64,
    abs_sum: AtomicU64,
    saturated_hits: AtomicU64,
    updates: AtomicU64,
}

static STATS: [Counters; NAMES.len()] = [const {
    Counters {
        reads: AtomicU64::new(0),
        hits: AtomicU64::new(0),
        abs_sum: AtomicU64::new(0),
        saturated_hits: AtomicU64::new(0),
        updates: AtomicU64::new(0),
    }
}; NAMES.len()];

/// `value` is the raw fixed-point correction returned by the table.
#[cfg(not(feature = "nocorr"))]
#[inline]
pub(super) fn record_read(table: Table, value: i32) {
    let c = &STATS[table as usize];
    c.reads.fetch_add(1, Relaxed);
    if value != 0 {
        c.hits.fetch_add(1, Relaxed);
        c.abs_sum.fetch_add(value.unsigned_abs() as u64, Relaxed);
        if value.abs() >= CORRECTION_LIMIT {
            c.saturated_hits.fetch_add(1, Relaxed);
        }
    }
}

#[inline]
pub(super) fn record_update(table: Table) { STATS[table as usize].updates.fetch_add(1, Relaxed); }

/// Reports cumulative counters after recording stops.
pub(crate) fn report() {
    let sp = SearchParams::default();
    let weights: [i32; NAMES.len()] = [REPORT_WEIGHT_SCALE, sp.minor_corr_weight, sp.major_corr_weight];
    let ansi = io::stdout().is_terminal();
    let gold = if ansi { GOLD_PEN } else { "" };
    let (bold, reset) = if ansi { (BOLD, RESET) } else { ("", "") };

    println!("\n{gold}{bold}Correction History stats{reset}");
    let header: Vec<String> = COLS.iter().enumerate().map(|(i, &(label, ..))| cell(label, i, None, false)).collect();
    println!("  {gold}{bold}{}{reset}", header.join(" "));

    for ((name, counters), weight) in NAMES.iter().zip(&STATS).zip(weights) {
        let reads = counters.reads.load(Relaxed);
        let hits = counters.hits.load(Relaxed);
        let updates = counters.updates.load(Relaxed);
        let abs_sum = counters.abs_sum.load(Relaxed);
        let saturated_hits = counters.saturated_hits.load(Relaxed);

        let hit_pct = pct(hits, reads);
        let saturation_pct = pct(saturated_hits, hits);
        let mean_cp = ratio(abs_sum as f64, hits as f64) / CORRECTION_SCALE as f64;
        let effective_cp = mean_cp * f64::from(weight) / f64::from(REPORT_WEIGHT_SCALE);
        let hit_rgb = color::advantage((hit_pct / 35.0 - 1.0).clamp(-1.0, 1.0));
        let sat_rgb = color::advantage(1.0 - saturation_pct / 25.0);

        let row = [
            cell(name, 0, None, ansi),
            cell(&fmt_count(reads), 1, None, ansi),
            cell(&format!("{hit_pct:.1}%"), 2, Some(hit_rgb), ansi),
            cell(&fmt_count(updates), 3, None, ansi),
            cell(&format!("{mean_cp:.2}cp"), 4, None, ansi),
            cell(&weight.to_string(), 5, None, ansi),
            cell(&format!("{effective_cp:.2}"), 6, None, ansi),
            cell(&format!("{saturation_pct:.1}%"), 7, Some(sat_rgb), ansi),
        ];
        println!("  {}", row.join(" "));
    }
}

/// ANSI escapes must not contribute to the padding width.
fn cell(text: &str, col: usize, rgb: Option<color::Rgb>, ansi: bool) -> String {
    let (_, w, right) = COLS[col];
    let padded = if right { format!("{text:>w$}") } else { format!("{text:<w$}") };

    match rgb {
        Some(c) if ansi => format!("{}{padded}{RESET}", color::ansi_fg(c)),
        _ => padded,
    }
}
