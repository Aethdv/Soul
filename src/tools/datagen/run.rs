//! Orchestrator for self-play data generation.
//!
//! Spawns N persistent worker threads, each playing complete games
//! from book openings, filtering positions by search verification
//! and adjudication, then flushing whole games to disk in viriformat.
//!
//! The hot path lives in `WorkerState::play_game()`; this module is just
//! the conductor: load books, dispatch work, flush to disk, print stats.

use std::{
    fs::OpenOptions,
    io::{Write, stdout},
    num::NonZero,
    path::Path,
    process,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
        mpsc::channel,
    },
    thread::{available_parallelism, sleep, spawn},
    time::Duration,
};

use super::{
    config::DatagenConfig,
    stats::{GlobalStats, rss_kb},
    worker::{Game, WorkerState},
};
use crate::{
    cli::Help,
    color::{self, RESET},
    core::{
        board::STARTPOS,
        defs::MAX_DEPTH,
        util::{format_comma as format_num, format_duration, pct, ratio},
    },
    tools::dataset::{load_epd_fens, scan_viri_games, write_game},
};

const GREEN: &str = color::OK_PEN;
const YELLOW: &str = color::WARN_PEN;
const RED: &str = color::ALARM_PEN;

/// Dashboard refresh interval.
const DASHBOARD_INTERVAL: Duration = Duration::from_millis(100);

/// Number of lines the dashboard occupies.
const DASHBOARD_LINES: usize = 11;

pub fn run(args: &[&str], stop: &Arc<AtomicBool>) {
    let (parsed, resume) = parse_args(args);
    let book_fens = if parsed.startpos {
        vec![STARTPOS.to_string()]
    } else {
        let fens = load_books(&parsed.book_paths);
        if fens.is_empty() {
            eprintln!("{RED}Error: No opening positions loaded!{RESET}");
            return;
        }
        fens
    };

    println!("Total starting positions: {GREEN}{}{RESET}", book_fens.len());

    let mut config = resolve_config(parsed, resume);
    let start_count = if resume { load_existing_count(&config.output_path) } else { 0 };

    config.generated_count = start_count as u64;
    let _ = config.save();

    let book = Arc::new(book_fens);
    let target = config.target_count;
    let output_path = config.output_path.clone();
    let save_interval = config.save_interval;

    // Use half the available cores by default: datagen is I/O-light and
    // compute-heavy, but leaving cores free keeps the system responsive
    // during multi-hour runs.
    let all_cores = available_parallelism().map_or(1, NonZero::get);
    let num_threads = config.thread_count.unwrap_or_else(|| (all_cores / 2).max(1));
    let global = Arc::new(GlobalStats::new());

    print_banner(&config, num_threads, book.len(), start_count);

    // Graceful shutdown on Ctrl-C. Workers check this flag between games.
    let stop_for_handler = stop.clone();

    let _ = ctrlc::set_handler(move || {
        println!("\n{YELLOW}Interrupt received, stopping workers...{RESET}");
        stop_for_handler.store(true, Ordering::SeqCst);
    });

    let mut total_generated = start_count;
    let mut flushed_at = start_count;
    let mut pending: Vec<u8> = Vec::with_capacity(save_interval * 8);

    // Shared counter visible to the dashboard thread.
    let shared_generated = Arc::new(AtomicUsize::new(total_generated));
    let finished = Arc::new(AtomicBool::new(false));

    // Dashboard thread
    {
        let global_mon = global.clone();
        let gen_mon = shared_generated.clone();
        let stop_mon = stop.clone();
        let finish_mon = finished.clone();

        spawn(move || {
            let mut first_frame = true;

            while !stop_mon.load(Ordering::Relaxed) && !finish_mon.load(Ordering::Relaxed) {
                let snap = Snapshot::capture(&global_mon, &gen_mon, target);

                render_dashboard(&snap, &mut first_frame);
                sleep(DASHBOARD_INTERVAL);
            }
        });
    }

    // Each worker owns its TT and history table for the entire run.
    // One allocation per worker, no churn between games.
    let (tx, rx) = channel::<Game>();
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let mut worker = WorkerState::new(book.clone(), config.clone(), global.clone());
        let tx = tx.clone();
        let stop_w = stop.clone();
        let target_u = target;

        handles.push(spawn(move || {
            loop {
                if stop_w.load(Ordering::Relaxed) || worker.global.plies.load(Ordering::Relaxed) >= target_u {
                    break;
                }

                let game = worker.play_game();
                if !game.moves.is_empty() && tx.send(game).is_err() {
                    break;
                }
            }
        }));
    }

    drop(tx); // channel closes when all senders drop

    // Flush results to disk periodically
    for game in rx {
        total_generated += game.moves.len();
        shared_generated.store(total_generated, Ordering::Relaxed);
        config.generated_count = total_generated as u64;

        // The header's own eval field is the opening position's, which is the first
        // score recorded.
        let opening_eval = game.moves.first().map_or(0, |&(_, score)| score);
        write_game(&mut pending, &game.opening, game.result, opening_eval, &game.moves);

        if total_generated >= flushed_at + save_interval {
            flushed_at = total_generated;
            flush_to_disk(&output_path, &mut pending, &config);
        }

        if total_generated >= target as usize {
            stop.store(true, Ordering::SeqCst);
            break;
        }
    }

    // Wait for workers to finish their current game.
    for handle in handles {
        handle.join().unwrap();
    }

    // Don't lose the tail end of a long run.
    flush_to_disk(&output_path, &mut pending, &config);
    finished.store(true, Ordering::Relaxed);

    // Final report
    println!();
    let snap = Snapshot::capture(&global, &AtomicUsize::new(total_generated), target);
    print_final_report(&snap, &output_path);

    config.generated_count = snap.plies;
    let _ = config.save();
}

/// A consistent point-in-time capture of all atomic counters.
///
/// Both the live dashboard and the final report need the same metrics.
/// Rather than scattering `load(Relaxed)` calls everywhere and duplicating
/// all the derived calculations, we capture once and compute from the snapshot.
struct Snapshot {
    elapsed: f64,
    generated: u64,
    target: u64,
    games: u64,
    plies: u64,
    search_fail: u64,
    term_check: u64,
    term_stale: u64,
    term_d50: u64,
    term_drep: u64,
    term_dmat: u64,
    term_draw_adj: u64,
    term_resign: u64,
}

impl Snapshot {
    /// Reads every atomic counter with `Relaxed` ordering.
    /// Relaxed is fine here: we're displaying approximate progress,
    /// not synchronizing memory. Counters only ever increase.
    fn capture(global: &GlobalStats, generated: &AtomicUsize, target: u64) -> Self {
        Self {
            elapsed: global.start_time.elapsed().as_secs_f64(),
            generated: generated.load(Ordering::Relaxed) as u64,
            target,
            games: global.games.load(Ordering::Relaxed),
            plies: global.plies.load(Ordering::Relaxed),
            search_fail: global.search_fail.load(Ordering::Relaxed),
            term_check: global.term_check.load(Ordering::Relaxed),
            term_stale: global.term_stale.load(Ordering::Relaxed),
            term_d50: global.term_d50.load(Ordering::Relaxed),
            term_drep: global.term_drep.load(Ordering::Relaxed),
            term_dmat: global.term_dmat.load(Ordering::Relaxed),
            term_draw_adj: global.term_draw_adj.load(Ordering::Relaxed),
            term_resign: global.term_resign.load(Ordering::Relaxed),
        }
    }

    /// Positions recorded per second (current session only).
    fn rate(&self) -> f64 { ratio(self.plies as f64, self.elapsed) }

    /// Overall completion percentage.
    fn progress_pct(&self) -> f64 { pct(self.generated, self.target) }

    /// Average game length in plies, a useful sanity check.
    /// Typical selfplay games run 60-200 plies depending on resign threshold.
    fn avg_ply(&self) -> f64 { ratio(self.plies as f64, self.games as f64) }

    /// ETA in seconds based on current throughput.
    fn eta_secs(&self) -> f64 { ratio(self.target.saturating_sub(self.generated) as f64, self.rate()) }

    fn total_terminations(&self) -> u64 {
        self.term_check + self.term_stale + self.term_d50 + self.term_drep + self.term_dmat + self.term_draw_adj + self.term_resign
    }
}

/// Renders the live progress dashboard using ANSI cursor movement.
fn render_dashboard(snap: &Snapshot, first_frame: &mut bool) {
    let badge = if snap.avg_ply() > 60.0 {
        format!("{GREEN}[OK]{RESET}")
    } else if snap.avg_ply() > 30.0 {
        format!("{YELLOW}[LOW]{RESET}")
    } else {
        format!("{RED}[BAD]{RESET}")
    };

    if *first_frame {
        // Reserve vertical space so cursor-up doesn't overwrite previous output.
        for _ in 0..DASHBOARD_LINES {
            println!();
        }
        *first_frame = false;
    }

    // Cursor up N lines, then overwrite each with \r\x1b[K (clear to EOL).
    print!("\x1b[{DASHBOARD_LINES}A");

    println!(
        "\r\x1b[KPositions:       {}/{} ({:.1}%) {badge}",
        format_num(snap.generated),
        format_num(snap.target),
        snap.progress_pct(),
    );

    println!("\r\x1b[KRate:            {:.3} k/s", snap.rate() / 1000.0);

    println!("\r\x1b[KGames:           {}", format_num(snap.games));
    println!("\r\x1b[KAvg ply:         {:.1}", snap.avg_ply());
    println!("\r\x1b[KCheckmate:       {}", format_num(snap.term_check));
    println!("\r\x1b[KDraw (rules):    {}", format_num(snap.term_stale + snap.term_d50 + snap.term_drep + snap.term_dmat));
    println!("\r\x1b[KDraw (adj):      {}", format_num(snap.term_draw_adj));
    println!("\r\x1b[KResign (adj):    {}", format_num(snap.term_resign));
    println!("\r\x1b[KBest Move fails: {}", format_num(snap.search_fail));
    println!("\r\x1b[KRAM alloc:       {} MB", rss_kb() / 1024);
    println!("\r\x1b[KElapsed:         {}", format_duration((snap.elapsed * 1000.0) as u64));
    println!("\r\x1b[KETA:             {}", format_duration((snap.eta_secs() * 1000.0) as u64));

    let _ = stdout().flush();
}

/// Prints the final generation report: the definitive summary of a run.
fn print_final_report(snap: &Snapshot, output_path: &str) {
    let total_term = snap.total_terminations();

    println!(
        "{GREEN}[OK]{RESET} {} positions in {:.1}s ({:.3}k/s)",
        format_num(snap.plies),
        snap.elapsed,
        snap.rate() / 1000.0,
    );

    println!("{GREEN}[OK]{RESET} Saved to {output_path}");
    println!();
    println!("[FINAL STATS]");
    println!("  Total Games:      {}", format_num(snap.games));
    println!("  Total Positions:  {}", format_num(snap.plies));
    println!("  Avg Plies/Game:   {:.1}", snap.avg_ply());
    println!("  Search Failures:  {}", format_num(snap.search_fail));
    println!();
    println!("  Game Terminations:");

    // Table-driven to avoid six identical println! blocks.
    let terminations = [
        ("Checkmate", snap.term_check),
        ("Stalemate", snap.term_stale),
        ("Draw (50-move)", snap.term_d50),
        ("Draw (rep)", snap.term_drep),
        ("Draw (mat)", snap.term_dmat),
        ("Draw (adj)", snap.term_draw_adj),
        ("Resign", snap.term_resign),
    ];

    for (label, count) in terminations {
        println!("    {label:<15} {} ({:.1}%)", format_num(count), pct(count, total_term),);
    }
}

/// Loads opening positions from one or more EPD/FEN files.
/// Returns the merged set of starting positions for selfplay games.
fn load_books(paths: &[String]) -> Vec<String> {
    println!("Loading opening books...");
    let mut all = Vec::new();
    for path in paths {
        match load_epd_fens(path) {
            Ok(fens) => {
                println!("  Loaded {} positions from '{path}'", fens.len());
                all.extend(fens);
            },
            Err(e) => eprintln!("  {RED}Failed to load {path}: {e}{RESET}"),
        }
    }
    all
}

/// Resolves the generation config, optionally resuming a previous run by
/// loading saved state from disk and re-applying the CLI's target and books.
fn resolve_config(parsed: DatagenConfig, resume: bool) -> DatagenConfig {
    if resume && let Ok(mut cfg) = DatagenConfig::load() {
        cfg.target_count = parsed.target_count;
        cfg.book_paths = parsed.book_paths;
        return cfg;
    }
    parsed
}

/// Prints the launch banner: what we're about to do and how.
fn print_banner(config: &DatagenConfig, num_threads: usize, book_count: usize, start_count: usize) {
    println!("Starting with {num_threads} threads");
    println!("Target: {} positions", config.target_count);
    println!("Output: {}", config.output_path);

    if config.startpos {
        println!("Book: startpos");
    } else {
        println!("Book: {} ({book_count} openings)", config.book_paths.join(", "),);
    }

    match (config.soft_nodes, config.hard_nodes) {
        (None, None) => println!("Search: depth={}", config.depth),
        (soft, hard) => println!(
            "Search: depth={}, softnodes={}, hardnodes={}",
            config.depth,
            soft.map_or("-".into(), |n| n.to_string()),
            hard.map_or("-".into(), |n| n.to_string()),
        ),
    }

    println!("Resign: ±{}cp", config.resign_cp);

    if start_count > 0 {
        println!("Resume: {start_count} existing positions");
    }
    println!();
}

/// Flushes pending entries to disk and saves config state.
/// Called periodically during generation and once at the end.
fn flush_to_disk(output_path: &str, pending: &mut Vec<u8>, config: &DatagenConfig) {
    if pending.is_empty() {
        return;
    }

    let appended = OpenOptions::new()
        .create(true)
        .append(true)
        .open(output_path)
        .and_then(|mut f| f.write_all(pending));

    if let Err(e) = appended {
        eprintln!("{RED}[ERROR] Failed to save batch: {e}{RESET}");
    }
    pending.clear();
    let _ = config.save();
}

/// A resume appends to this file and counts up from what it already holds, so a count that
/// cannot be read ends the run rather than generating the whole target on top of it.
fn load_existing_count(path: &str) -> usize {
    if !Path::new(path).exists() {
        return 0;
    }

    scan_viri_games(path)
        .unwrap_or_else(|e| {
            eprintln!("{RED}Error: cannot resume {path}: {e}{RESET}");
            process::exit(1);
        })
        .plies as usize
}

fn print_help() {
    let h = Help::new(22);

    h.header("Usage:");
    println!("  soul datagen [OPTIONS]");
    println!();

    h.header("Options:");
    h.option_default("-n, --count", "<N>", "Target number of positions to generate", "8,000,000");
    h.option_default("-t, --threads", "<N>", "Number of threads", "auto");
    h.option_default("-o, --output", "<PATH>", "Output file path", "data.vf");
    h.option_default("-b, --book", "<PATH>", "Opening book path", "UHO_Lichess_4852_v1.epd");
    h.option_default("-d, --depth", "<N>", "Search depth (default 6; MAX when --soft/--nodes set without --depth)", "6");
    h.option("--soft", "<N>", "Soft node limit");
    h.option("--nodes", "<N>", "Hard node limit");
    h.option_default("--plies", "<N>", "Max game length", "300");
    h.option("--resume", "", "Resume from existing config/output");
    h.option_default("--save-interval", "<N>", "Save interval", "5000");
    h.option("--startpos", "", "Use standard start position instead of book files");
    h.option_default("--resign", "<CP>", "Resign threshold in centipawns", "800");
    h.header("Notes:");
    println!("  Every ply of every game is recorded. Which of them train is decided at load");
    println!("  time by the tuner's replay filter, not here.");
}

fn parse_args(args: &[&str]) -> (DatagenConfig, bool) {
    // Defaults come solely from DatagenConfig::default(); the CLI overrides
    // fields in place. depth stays a local Option so the node-limit resolution
    // can run after the loop; resume is a control flag, not config.
    let mut cfg = DatagenConfig::default();
    let mut depth: Option<i32> = None;
    let mut book_override = false;
    let mut resume = false;

    // Consume the next token and parse it into the field, leaving the field
    // unchanged on a missing or unparseable argument. take_opt writes None on
    // a parse failure, for the Option-typed fields.
    macro_rules! take {
        ($it:expr, $f:expr) => {
            if let Some(v) = $it.next() {
                $f = v.parse().unwrap_or($f);
            }
        };
    }

    macro_rules! take_opt {
        ($it:expr, $f:expr) => {
            if let Some(v) = $it.next() {
                $f = v.parse().ok();
            }
        };
    }

    let mut it = args.iter().copied();
    while let Some(flag) = it.next() {
        match flag {
            // Non-uniform arms: string values, accumulation, custom parsing.
            "-o" | "--output" => {
                if let Some(v) = it.next() {
                    cfg.output_path = v.to_string();
                }
            },
            "-b" | "--book" => {
                if let Some(v) = it.next() {
                    // First --book replaces the default; further ones accumulate.
                    if !book_override {
                        cfg.book_paths.clear();
                        book_override = true;
                    }
                    cfg.book_paths.push(v.to_string());
                }
            },
            "-n" | "--count" => {
                if let Some(v) = it.next() {
                    cfg.target_count = parse_suffix(v).unwrap_or(cfg.target_count);
                }
            },
            "-d" | "--depth" => {
                if let Some(v) = it.next() {
                    depth = v.parse().ok().or(depth);
                }
            },

            "--soft" => take_opt!(it, cfg.soft_nodes),
            "--nodes" => take_opt!(it, cfg.hard_nodes),
            "-t" | "--threads" => take_opt!(it, cfg.thread_count),

            "--resign" => take!(it, cfg.resign_cp),
            "--plies" => take!(it, cfg.max_plies),
            "--save-interval" => take!(it, cfg.save_interval),

            "--startpos" => cfg.startpos = true,
            "--resume" => resume = true,
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            },
            _ => {}, // Unknown flags silently ignored.
        }
    }

    // An explicit --depth wins; otherwise a node-limited run searches to max
    // depth and lets the node cap bound it; otherwise keep the default.
    if let Some(d) = depth {
        cfg.depth = d;
    } else if cfg.soft_nodes.is_some() || cfg.hard_nodes.is_some() {
        cfg.depth = MAX_DEPTH;
    }
    (cfg, resume)
}

/// Parses a number string with optional K/M/B suffix.
fn parse_suffix(s: &str) -> Option<u64> {
    let lower = s.to_lowercase();
    // Try stripping a magnitude suffix and multiplying.
    let suffixes: &[(&str, f64)] = &[("b", 1e9), ("m", 1e6), ("k", 1e3)];
    for &(suffix, multiplier) in suffixes {
        if let Some(stem) = lower.strip_suffix(suffix) {
            return stem.parse::<f64>().map(|n| (n * multiplier) as u64).ok();
        }
    }
    // No suffix: strip commas and parse directly.
    s.replace(',', "").parse().ok()
}
