//! Move-picker instrumentation: does the ordering earn its keep?
//!
//! Two questions, two metric blocks. Adding a third is a static block, a
//! `record_*` hook, and one line in [`report`].
//!
//! - Quiet sort utilization. A node at the quiet stage sorts every quiet,
//!   then yields best-first until a cutoff. Generated-vs-consumed and the
//!   consumption histogram say whether that full sort is load-bearing or thrown
//!   away: back-loaded means justified, front-loaded (mass at 0–1) means a
//!   lazy max-scan would win. The catch is selection bias: cut-nodes usually
//!   cut before quiets are even generated, so the nodes that pay the sort are
//!   the ones that consume it.
//!
//! - Beta-cutoff ordering. At a fail-high node, which move number caused the
//!   cutoff and which heuristic surfaced it. The first-move cutoff rate is the
//!   single number that says the staging is good: the whole point of ordering
//!   is to make the cutoff happen on move one.
//!
//! Compiled only under the `mvpstats` feature; zero cost in release builds.

use std::{
    io,
    io::IsTerminal,
    sync::atomic::{AtomicU64, Ordering::Relaxed},
};

use crate::color;

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";
const HEADER: color::Rgb = (218, 165, 32);

/// Consumption buckets: index = quiets used, the last bucket is "11 or more".
const QUIET_BUCKETS: usize = 12;

static QUIET_HIST: [AtomicU64; QUIET_BUCKETS] = [const { AtomicU64::new(0) }; QUIET_BUCKETS];
static QUIET_GENERATED: AtomicU64 = AtomicU64::new(0);
static QUIET_CONSUMED: AtomicU64 = AtomicU64::new(0);
static QUIET_NODES: AtomicU64 = AtomicU64::new(0);

const KINDS: usize = 4;
const KIND_NAMES: [&str; KINDS] = ["hash", "capture", "killer", "quiet"];

/// Cutoff move-index buckets: 1-based move number, the last bucket is "8 or more".
const CUTOFF_BUCKETS: usize = 8;

static CUTOFF_HIST: [AtomicU64; CUTOFF_BUCKETS] = [const { AtomicU64::new(0) }; CUTOFF_BUCKETS];
static CUTOFF_KIND: [AtomicU64; KINDS] = [const { AtomicU64::new(0) }; KINDS];
static CUTOFF_NODES: AtomicU64 = AtomicU64::new(0);

/// Record one quiet-stage node; `generated` quiets sorted, `consumed` yielded
/// before the picker was dropped (a cutoff or natural exhaustion).
#[inline]
pub fn record_quiets(generated: u32, consumed: u32) {
    QUIET_HIST[(consumed as usize).min(QUIET_BUCKETS - 1)].fetch_add(1, Relaxed);
    QUIET_GENERATED.fetch_add(u64::from(generated), Relaxed);
    QUIET_CONSUMED.fetch_add(u64::from(consumed), Relaxed);
    QUIET_NODES.fetch_add(1, Relaxed);
}

/// Which mechanism surfaced the move that caused a beta cutoff.
#[derive(Clone, Copy)]
pub enum CutoffKind {
    Hash = 0,
    Capture = 1,
    Killer = 2,
    Quiet = 3,
}

/// The cutoff fired on the `index`-th searched move (1-based), surfaced by `kind`.
#[inline]
pub fn record_cutoff(index: u32, kind: CutoffKind) {
    CUTOFF_HIST[(index.max(1) as usize - 1).min(CUTOFF_BUCKETS - 1)].fetch_add(1, Relaxed);
    CUTOFF_KIND[kind as usize].fetch_add(1, Relaxed);
    CUTOFF_NODES.fetch_add(1, Relaxed);
}

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
    let used_pct = 100.0 * consumed as f64 / generated as f64;
    let used = colored(format!("{used_pct:.1}%"), (used_pct / 50.0 - 1.0).clamp(-1.0, 1.0));

    header("MovePicker quiet stats");

    println!(
        "  sorting nodes {}   generated {}   consumed {}   ({used} used)",
        human(nodes),
        human(generated),
        human(consumed)
    );

    columns(["consumed", "nodes", "share", "cumul"]);
    histogram(&QUIET_HIST, nodes, 0);
}

fn report_cutoffs() {
    let nodes = CUTOFF_NODES.load(Relaxed);

    if nodes == 0 {
        return;
    }

    let first = 100.0 * CUTOFF_HIST[0].load(Relaxed) as f64 / nodes as f64;
    let rate = colored(format!("{first:.1}%"), (first / 90.0 - 1.0).clamp(-1.0, 1.0));

    header("MovePicker cutoff ordering");
    println!("  fail-high nodes {}   first-move cutoff {rate}", human(nodes));
    columns(["move", "nodes", "share", "cumul"]);
    histogram(&CUTOFF_HIST, nodes, 1);

    let (gold, reset) = header_color();
    println!("  {gold}by mechanism{reset}");

    for (i, name) in KIND_NAMES.iter().enumerate() {
        let c = CUTOFF_KIND[i].load(Relaxed);
        println!("  {name:>8}  {:>11}  {:>6.1}%", human(c), 100.0 * c as f64 / nodes as f64);
    }
}

fn histogram(buckets: &[AtomicU64], nodes: u64, base: usize) {
    let mut cumulative = 0u64;

    for (i, b) in buckets.iter().enumerate() {
        let c = b.load(Relaxed);

        cumulative += c;

        let n = base + i;
        let label = if i == buckets.len() - 1 { format!("{n}+") } else { n.to_string() };

        println!(
            "  {label:>8}  {:>11}  {:>6.1}%  {:>6.1}%",
            human(c),
            100.0 * c as f64 / nodes as f64,
            100.0 * cumulative as f64 / nodes as f64
        );
    }
}

fn header(title: &str) {
    let (gold, reset) = header_color();
    println!("\n{gold}{BOLD}{title}{reset}");
}

fn columns(labels: [&str; 4]) {
    let (gold, reset) = header_color();
    println!("  {gold}{BOLD}{:>8}  {:>11}  {:>7}  {:>7}{reset}", labels[0], labels[1], labels[2], labels[3]);
}

fn header_color() -> (String, &'static str) {
    if io::stdout().is_terminal() { (color::ansi_fg(HEADER), RESET) } else { (String::new(), "") }
}

fn colored(text: String, advantage: f64) -> String {
    if io::stdout().is_terminal() {
        format!("{}{text}{RESET}", color::ansi_fg(color::advantage(advantage)))
    } else {
        text
    }
}

/// Hooman counts.
fn human(n: u64) -> String {
    match n {
        1_000_000.. => format!("{:.2}M", n as f64 / 1e6),
        1_000.. => format!("{:.1}K", n as f64 / 1e3),
        _ => n.to_string(),
    }
}
