//! Move-ordering statistics.

use std::{
    io::{self, IsTerminal},
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use crate::{
    color::{self, BOLD, GOLD_PEN, RESET},
    core::util::{fmt_count, pct},
};

const CLASS_NAMES: [&str; 4] = ["hash", "capture", "killer", "quiet"];

const QUIET_BUCKETS: usize = 12;
const CUTOFF_BUCKETS: usize = 8;

const QUIET_USE_NEUTRAL: f64 = 50.0;
const FIRST_CUTOFF_NEUTRAL: f64 = 90.0;

static QUIET_HIST: [AtomicU64; QUIET_BUCKETS] = [const { AtomicU64::new(0) }; QUIET_BUCKETS];
static QUIET_GENERATED: AtomicU64 = AtomicU64::new(0);
static QUIET_CONSUMED: AtomicU64 = AtomicU64::new(0);
static QUIET_NODES: AtomicU64 = AtomicU64::new(0);

static CUTOFF_HIST: [AtomicU64; CUTOFF_BUCKETS] = [const { AtomicU64::new(0) }; CUTOFF_BUCKETS];
static CUTOFF_CLASS: [AtomicU64; CLASS_NAMES.len()] = [const { AtomicU64::new(0) }; CLASS_NAMES.len()];
static CUTOFF_NODES: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Copy)]
#[repr(usize)]
pub(super) enum CutoffClass {
    Hash = 0,
    Capture = 1,
    Killer = 2,
    Quiet = 3,
}

/// Records a quiet-stage search node: `generated` sorted vs. `consumed` yielded.
#[inline]
pub(super) fn record_quiets(generated: u32, consumed: u32) {
    QUIET_HIST[(consumed as usize).min(QUIET_BUCKETS - 1)].fetch_add(1, Relaxed);
    QUIET_GENERATED.fetch_add(u64::from(generated), Relaxed);
    QUIET_CONSUMED.fetch_add(u64::from(consumed), Relaxed);
    QUIET_NODES.fetch_add(1, Relaxed);
}

/// Records a beta-cutoff occurring on the `index`-th searched move (1-based).
#[inline]
pub(super) fn record_cutoff(index: u32, kind: CutoffClass) {
    CUTOFF_HIST[(index.max(1) as usize - 1).min(CUTOFF_BUCKETS - 1)].fetch_add(1, Relaxed);
    CUTOFF_CLASS[kind as usize].fetch_add(1, Relaxed);
    CUTOFF_NODES.fetch_add(1, Relaxed);
}

pub(crate) fn report() {
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
    let used = pct_colored(pct(consumed, generated), QUIET_USE_NEUTRAL);

    header("MovePicker quiet stats");
    println!(
        "  sorting nodes {}   generated {}   consumed {}   ({used} used)",
        fmt_count(nodes),
        fmt_count(generated),
        fmt_count(consumed)
    );

    histogram(&QUIET_HIST, nodes, 0, "consumed");
}

fn report_cutoffs() {
    let nodes = CUTOFF_NODES.load(Relaxed);
    if nodes == 0 {
        return;
    }

    let rate = pct_colored(pct(CUTOFF_HIST[0].load(Relaxed), nodes), FIRST_CUTOFF_NEUTRAL);

    header("MovePicker cutoff ordering");
    println!("  fail-high nodes {}   first-move cutoff {rate}", fmt_count(nodes));
    histogram(&CUTOFF_HIST, nodes, 1, "move");

    let (gold, _, reset) = style();
    println!("  {gold}by class{reset}");

    for (name, count) in CLASS_NAMES.iter().zip(&CUTOFF_CLASS) {
        let count = count.load(Relaxed);
        println!("  {name:>8}  {:>11}  {:>6.1}%", fmt_count(count), pct(count, nodes));
    }
}

fn histogram(buckets: &[AtomicU64], nodes: u64, base: usize, heading: &str) {
    let (gold, bold, reset) = style();
    println!("  {gold}{bold}{heading:>8}  {:>11}  {:>7}  {:>7}{reset}", "nodes", "share", "cumul");

    let mut cumulative = 0u64;
    for (i, b) in buckets.iter().enumerate() {
        let count = b.load(Relaxed);
        cumulative += count;

        let n = base + i;
        let label = if i == buckets.len() - 1 { format!("{n}+") } else { n.to_string() };

        println!("  {label:>8}  {:>11}  {:>6.1}%  {:>6.1}%", fmt_count(count), pct(count, nodes), pct(cumulative, nodes));
    }
}

fn header(title: &str) {
    let (gold, bold, reset) = style();
    println!("\n{gold}{bold}{title}{reset}");
}

fn pct_colored(pct: f64, neutral: f64) -> String {
    let text = format!("{pct:.1}%");
    if io::stdout().is_terminal() {
        let advantage = (pct / neutral - 1.0).clamp(-1.0, 1.0);
        format!("{}{text}{RESET}", color::ansi_fg(color::advantage(advantage)))
    } else {
        text
    }
}

fn style() -> (&'static str, &'static str, &'static str) {
    if io::stdout().is_terminal() { (GOLD_PEN, BOLD, RESET) } else { ("", "", "") }
}
