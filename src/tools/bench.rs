//! Node counting and NPS benchmarking utility.
//!
//! Runs the search engine at fixed depths across a standard suite of positions
//! to measure raw search throughput and verify deterministic node counts.

use std::{
    io::{self, IsTerminal, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering::Relaxed},
    },
    thread,
    time::{Duration, Instant},
};

use crate::{
    color::{self, Rgb},
    core::{board::Position, defs::Protocol},
    engine::{
        history::History,
        search::{Limits, SearchConfig, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
};

const FENS: &str = include_str!("../data/bench.fens");

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

const SPINNER: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
const FPS: u64 = 30;

const PURPLE: Rgb = (180, 140, 255);

struct AnimationGuard {
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Drop for AnimationGuard {
    fn drop(&mut self) {
        self.running.store(false, Relaxed);

        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

pub fn run(depth: i32, hash_mb: usize) {
    let start = Instant::now();
    let stop_signal = Arc::new(AtomicBool::new(false));
    let fens: Vec<&str> = FENS.lines().collect();
    let total = fens.len();

    let done = Arc::new(AtomicUsize::new(0));
    let nodes = Arc::new(AtomicU64::new(0));
    let running = Arc::new(AtomicBool::new(true));

    let tty = io::stdout().is_terminal();

    if tty {
        println!("{}{BOLD}✦ Soul v{}{RESET}", color::ansi_fg(PURPLE), env!("CARGO_PKG_VERSION"));
    }

    let animator = tty.then(|| AnimationGuard {
        running: Arc::clone(&running),
        handle: Some(spawn_spinner(start, total, Arc::clone(&done), Arc::clone(&nodes), Arc::clone(&running))),
    });

    let limits = Limits { depth, silent: true, protocol: Protocol::Uci, ..Default::default() };

    // Clear between positions, never reallocate: a fresh TT per FEN means a
    // page-fault storm that, under many parallel bench processes, swamps the clock.
    let tt = Arc::new(TranspositionTable::new(hash_mb, 1));

    let mut search_time = Duration::ZERO;

    for fen in &fens {
        let board = Position::from_fen(fen);
        let history = vec![board.hash];

        tt.clear(1);

        let t0 = Instant::now();
        let cfg = SearchConfig::new(limits.clone(), Instant::now(), stop_signal.clone(), 0, SearchParams::default());

        let mut history_table = History::new();
        let mut searcher = Searcher::new(&cfg, &board, &history, tt.clone());

        searcher.iterative_deepening(&mut history_table);
        search_time += t0.elapsed();

        nodes.fetch_add(searcher.nodes, Relaxed);
        done.fetch_add(1, Relaxed);
    }

    if animator.is_some() {
        drop(animator);
        print!("\r\x1b[K");
        let _ = io::stdout().flush();
    }

    let total_nodes = nodes.load(Relaxed);
    let elapsed = search_time.as_secs_f64().max(0.000_001);
    let nps = (total_nodes as f64 / elapsed) as u64;
    println!("Hash {hash_mb} MB · {} pages", tt.page_kind());
    println!("Bench: {total_nodes} nodes {nps} nps · {elapsed:.1}s");

    #[cfg(feature = "mvpstats")]
    crate::engine::mvpstats::report();

    #[cfg(feature = "corrstats")]
    crate::engine::corrstats::report();
}

fn spawn_spinner(
    start: Instant,
    total: usize,
    done: Arc<AtomicUsize>,
    nodes: Arc<AtomicU64>,
    running: Arc<AtomicBool>,
) -> thread::JoinHandle<()> {
    thread::spawn(move || {
        let frame = Duration::from_nanos(1_000_000_000 / FPS);
        let mut phase = 0usize;

        let render_frame = |current_phase: usize| {
            let solved = done.load(Relaxed);
            let elapsed = start.elapsed().as_secs_f64().max(0.000_001);
            let nps = nodes.load(Relaxed) as f64 / elapsed;

            print!("\r\x1b[K{}", line(SPINNER[current_phase % SPINNER.len()], solved, total, nps, elapsed));
            let _ = io::stdout().flush();
        };

        while running.load(Relaxed) {
            render_frame(phase);
            phase += 1;
            thread::sleep(frame);
        }

        render_frame(phase);
    })
}

fn line(spin: char, solved: usize, total: usize, nps: f64, elapsed: f64) -> String {
    let purple = color::ansi_fg(PURPLE);
    format!("  {purple}{spin}{RESET}  {solved:>3}/{total}   {elapsed:>5.1}s   {nps:.0} nps")
}
