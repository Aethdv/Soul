//! Move-picker diagnostics and heuristic efficiency profiling.
//!
//! Tracks two primary metrics to evaluate move ordering performance:
//!
//! - Quiet Sort Utilization: Ratio of quiets sorted versus actually consumed.
//!   High utilization justifies eager `sort_unstable`; heavy mass at 0–1 quiets
//!   indicates a lazy partial-sort or linear scan would be cheaper. Read it against
//!   the selection bias: cut-nodes usually cut before quiets are generated at all,
//!   so the nodes that pay for the sort are the ones that consume it.
//!
//! - Beta-Cutoff Distribution: Move index (1-based) and heuristic tier that
//!   triggered a fail-high. The first-move cutoff percentage directly measures
//!   ordering quality.
//!
//! Gated by the `mvpstats` feature, and compiled out entirely when it is off.

use std::{
    io::{self, IsTerminal},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use crate::{
    color::{self, BOLD, GOLD, RESET},
    core::util::{human, pct},
};

/// Quiet consumption buckets: index = moves consumed; last bucket represents `N-1+`.
const QUIET_BUCKETS: usize = 12;

static QUIET_HIST: [AtomicU64; QUIET_BUCKETS] = [const { AtomicU64::new(0) }; QUIET_BUCKETS];
static QUIET_GENERATED: AtomicU64 = AtomicU64::new(0);
static QUIET_CONSUMED: AtomicU64 = AtomicU64::new(0);
static QUIET_NODES: AtomicU64 = AtomicU64::new(0);

/// Ordered to match [`CutoffKind`]'s discriminants, which index the counters.
const KIND_NAMES: [&str; 4] = ["hash", "capture", "killer", "quiet"];

/// Cutoff move index buckets: index = (1-based move index - 1); last bucket is `N-1+`.
const CUTOFF_BUCKETS: usize = 8;

/// Where the color scale reads neutral: half the sorted quiets consumed, and nine
/// cutoffs in ten landing on the first move.
const QUIET_USE_NEUTRAL: f64 = 50.0;
const FIRST_CUTOFF_NEUTRAL: f64 = 90.0;

static CUTOFF_HIST: [AtomicU64; CUTOFF_BUCKETS] = [const { AtomicU64::new(0) }; CUTOFF_BUCKETS];
static CUTOFF_KIND: [AtomicU64; KIND_NAMES.len()] = [const { AtomicU64::new(0) }; KIND_NAMES.len()];
static CUTOFF_NODES: AtomicU64 = AtomicU64::new(0);

/// Heuristic source that surfaced the cutoff move.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum CutoffKind {
    Hash = 0,
    Capture = 1,
    Killer = 2,
    Quiet = 3,
}

/// Records a quiet-stage search node: `generated` sorted vs. `consumed` yielded.
#[inline]
pub fn record_quiets(generated: u32, consumed: u32) {
    QUIET_HIST[(consumed as usize).min(QUIET_BUCKETS - 1)].fetch_add(1, Relaxed);
    QUIET_GENERATED.fetch_add(u64::from(generated), Relaxed);
    QUIET_CONSUMED.fetch_add(u64::from(consumed), Relaxed);
    QUIET_NODES.fetch_add(1, Relaxed);
}

/// Records a beta-cutoff occurring on the `index`-th searched move (1-based).
#[inline]
pub fn record_cutoff(index: u32, kind: CutoffKind) {
    CUTOFF_HIST[(index.max(1) as usize - 1).min(CUTOFF_BUCKETS - 1)].fetch_add(1, Relaxed);
    CUTOFF_KIND[kind as usize].fetch_add(1, Relaxed);
    CUTOFF_NODES.fetch_add(1, Relaxed);
}

/// Prints aggregated move picker statistics to stdout.
pub fn report() {
    report_quiets();
    report_cutoffs();
}

fn report_quiets() {
    let nodes = QUIET_NODES.load(Relaxed);
    if nodes == 0 {
        return;
    }

    let generated = QUIET_GENERATED.load(Relaxed);
    let consumed = QUIET_CONSUMED.load(Relaxed);
    let used = scored(pct(consumed, generated), QUIET_USE_NEUTRAL);

    header("MovePicker quiet stats");
    println!(
        "  sorting nodes {}   generated {}   consumed {}   ({used} used)",
        human(nodes),
        human(generated),
        human(consumed)
    );

    histogram(&QUIET_HIST, nodes, 0, "consumed");
}

fn report_cutoffs() {
    let nodes = CUTOFF_NODES.load(Relaxed);
    if nodes == 0 {
        return;
    }

    let rate = scored(pct(CUTOFF_HIST[0].load(Relaxed), nodes), FIRST_CUTOFF_NEUTRAL);

    header("MovePicker cutoff ordering");
    println!("  fail-high nodes {}   first-move cutoff {rate}", human(nodes));

    histogram(&CUTOFF_HIST, nodes, 1, "move");

    let (gold, _, reset) = style();
    println!("  {gold}by mechanism{reset}");

    for (name, count) in KIND_NAMES.iter().zip(&CUTOFF_KIND) {
        let count = count.load(Relaxed);
        println!("  {name:>8}  {:>11}  {:>6.1}%", human(count), pct(count, nodes));
    }
}

/// The bucket column carries `first`; the other three are the same table every time.
fn histogram(buckets: &[AtomicU64], nodes: u64, base: usize, first: &str) {
    let (gold, bold, reset) = style();
    println!("  {gold}{bold}{first:>8}  {:>11}  {:>7}  {:>7}{reset}", "nodes", "share", "cumul");

    let mut cumulative = 0u64;
    for (i, b) in buckets.iter().enumerate() {
        let count = b.load(Relaxed);
        cumulative += count;

        let n = base + i;
        let label = if i == buckets.len() - 1 { format!("{n}+") } else { n.to_string() };

        println!("  {label:>8}  {:>11}  {:>6.1}%  {:>6.1}%", human(count), pct(count, nodes), pct(cumulative, nodes));
    }
}

fn header(title: &str) {
    let (gold, bold, reset) = style();
    println!("\n{gold}{bold}{title}{reset}");
}

/// Paints a percentage against the point where the scale reads neutral: deep red at
/// zero, deep green at twice it.
fn scored(pct: f64, neutral: f64) -> String { colored(&format!("{pct:.1}%"), (pct / neutral - 1.0).clamp(-1.0, 1.0)) }

/// Gold, bold and reset, or three empty strings off a terminal. `BOLD` comes through
/// here so a piped report cannot keep an escape its reset was stripped from.
fn style() -> (String, &'static str, &'static str) {
    if io::stdout().is_terminal() { (color::ansi_fg(GOLD), BOLD, RESET) } else { (String::new(), "", "") }
}

fn colored(text: &str, advantage: f64) -> String {
    if io::stdout().is_terminal() {
        format!("{}{text}{RESET}", color::ansi_fg(color::advantage(advantage)))
    } else {
        text.to_string()
    }
}
