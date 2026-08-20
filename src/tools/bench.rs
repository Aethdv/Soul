//! Node counting and NPS benchmarking utility.
//!
//! Runs the search at fixed depth across a suite of positions.

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
    color::{self, BOLD, RESET, Rgb},
    core::{board::Position, defs::Protocol},
    engine::{
        history::History,
        search::{Limits, SearchConfig, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
};

const FENS: &str = include_str!("../data/bench.fens");

const THROBBER: [char; 8] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧'];
const FPS: u64 = 30;

const PURPLE: Rgb = (180, 140, 255);

/// The tty progress line.
struct Progress {
    running: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl Progress {
    fn spawn(start: Instant, total: usize, done: Arc<AtomicUsize>, nodes: Arc<AtomicU64>) -> Self {
        let running = Arc::new(AtomicBool::new(true));
        let drawing = Arc::clone(&running);

        let handle = thread::spawn(move || {
            let tick = Duration::from_nanos(1_000_000_000 / FPS);
            let purple = color::ansi_fg(PURPLE);

            let draw = |phase: usize| {
                let solved = done.load(Relaxed);
                let elapsed = start.elapsed().as_secs_f64().max(0.000_001);
                let nps = nodes.load(Relaxed) as f64 / elapsed;
                let glyph = THROBBER[phase % THROBBER.len()];
                print!("\r\x1b[K  {purple}{glyph}{RESET}  {solved:>3}/{total}   {elapsed:>5.1}s   {nps:.0} nps");
                let _ = io::stdout().flush();
            };

            let mut phase = 0usize;
            while drawing.load(Relaxed) {
                draw(phase);
                phase += 1;
                thread::sleep(tick);
            }
            draw(phase);
        });

        Self { running, handle: Some(handle) }
    }
}

impl Drop for Progress {
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
    let tty = io::stdout().is_terminal();
    if tty {
        println!("{}{BOLD}✦ Soul v{}{RESET}", color::ansi_fg(PURPLE), env!("CARGO_PKG_VERSION"));
    }

    let progress = tty.then(|| Progress::spawn(start, total, Arc::clone(&done), Arc::clone(&nodes)));
    let limits = Limits { depth, silent: true, protocol: Protocol::Uci, ..Default::default() };

    // Clear between positions, never reallocate.
    let tt = Arc::new(TranspositionTable::new(hash_mb, 1));

    let mut search_time = Duration::ZERO;

    for fen in &fens {
        let board = Position::from_fen(fen);
        let history = vec![board.hash];

        // SAFETY: bench runs each search inline on this thread, so none is in flight.
        unsafe { tt.clear(1) };

        let t0 = Instant::now();
        let cfg = SearchConfig::new(limits.clone(), Instant::now(), stop_signal.clone(), 0, SearchParams::default());

        let mut history_table = History::new();
        let mut searcher = Searcher::new(&cfg, &board, &history, tt.clone());
        searcher.iterative_deepening(&mut history_table);
        search_time += t0.elapsed();
        nodes.fetch_add(searcher.nodes, Relaxed);
        done.fetch_add(1, Relaxed);
    }

    drop(progress);

    if tty {
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
