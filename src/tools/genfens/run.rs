//! Orchestrator for self-play data generation ("genfens").
//!
//! Spawns N persistent worker threads, each playing complete games
//! from book openings, filtering positions by search verification
//! and quiet-move heuristics, then flushing results to disk in .soul.zst format.
//!
//! The hot path lives in `WorkerState::play_game()` — this module is just
//! the conductor: load books, dispatch work, flush to disk, print stats.

use std::{
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use super::{
    config::{GenfensArgs, GenfensConfig},
    stats::{GlobalStats, get_rss_kb},
    worker::WorkerState,
};
use crate::tools::dataset::{SoulEntry, append_encoded, load_encoded, parse_epd_str};

const GREEN: &str = "\x1b[92m";
const YELLOW: &str = "\x1b[93m";
const RED: &str = "\x1b[91m";
const RESET: &str = "\x1b[0m";

const BADGE_OK: &str = "\x1b[92m[OK]\x1b[0m";
const BADGE_LOW: &str = "\x1b[93m[LOW]\x1b[0m";
const BADGE_BAD: &str = "\x1b[91m[BAD]\x1b[0m";

/// Dashboard refresh interval.
/// 10 Hz is smooth enough without burning CPU on terminal writes.
const DASHBOARD_INTERVAL: std::time::Duration = std::time::Duration::from_millis(100);

/// Number of lines the dashboard occupies. We print this many newlines on
/// first render, then cursor-up by this amount on every subsequent frame.
const DASHBOARD_LINES: usize = 14;

pub fn run(args: &[&str], stop: &Arc<AtomicBool>) {
    let parsed = parse_args(args);

    let book_fens = if parsed.startpos {
        vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()]
    } else {
        let fens = load_books(&parsed.book_paths);
        if fens.is_empty() {
            eprintln!("{RED}Error: No opening positions loaded!{RESET}");
            return;
        }
        fens
    };
    println!("Total starting positions: {GREEN}{}{RESET}", book_fens.len(),);

    let mut config = resolve_config(&parsed);

    let start_count = if parsed.resume { load_existing_count(&config.output_path) } else { 0 };

    config.generated_count = start_count as u64;
    let _ = config.save();

    let book = Arc::new(book_fens);
    let target = config.target_count;
    let output_path = config.output_path.clone();
    let save_interval = config.save_interval;

    // Use half the available cores by default — datagen is I/O-light and
    // compute-heavy, but leaving cores free keeps the system responsive
    // during multi-hour runs.
    let all_cores = std::thread::available_parallelism().map_or(1, std::num::NonZero::get);
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
    let mut pending: Vec<SoulEntry> = Vec::with_capacity(save_interval * 2);

    // Shared counter visible to the dashboard thread.
    let shared_generated = Arc::new(AtomicUsize::new(total_generated));
    let finished = Arc::new(AtomicBool::new(false));

    // ── Dashboard thread ──
    {
        let global_mon = global.clone();
        let gen_mon = shared_generated.clone();
        let stop_mon = stop.clone();
        let finish_mon = finished.clone();

        std::thread::spawn(move || {
            let mut first_frame = true;
            while !stop_mon.load(Ordering::Relaxed) && !finish_mon.load(Ordering::Relaxed) {
                let snap = Snapshot::capture(&global_mon, &gen_mon, target);
                render_dashboard(&snap, &mut first_frame);
                std::thread::sleep(DASHBOARD_INTERVAL);
            }
        });
    }

    // ── Persistent worker threads ──
    // Each worker owns its TT and history table for the entire run.
    // One allocation per worker, no churn between games.
    let (tx, rx) = std::sync::mpsc::channel::<Vec<SoulEntry>>();
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let mut worker = WorkerState::new(book.clone(), config.clone(), global.clone());
        let tx = tx.clone();
        let stop_w = stop.clone();
        let target_u = target;
        handles.push(std::thread::spawn(move || {
            loop {
                if stop_w.load(Ordering::Relaxed) || worker.global.saved.load(Ordering::Relaxed) >= target_u {
                    break;
                }
                let entries = worker.play_game();
                if !entries.is_empty() && tx.send(entries).is_err() {
                    break;
                }
            }
        }));
    }
    drop(tx); // channel closes when all senders drop

    // ── Collection loop: flush results to disk periodically ──
    for entries in rx {
        total_generated += entries.len();
        shared_generated.store(total_generated, Ordering::Relaxed);
        config.generated_count = total_generated as u64;
        pending.extend(entries);

        if pending.len() >= save_interval {
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

    // Final flush — don't lose the tail end of a long run.
    flush_to_disk(&output_path, &mut pending, &config);
    finished.store(true, Ordering::Relaxed);

    // Final report
    println!();
    let snap = Snapshot::capture(&global, &AtomicUsize::new(total_generated), target);
    print_final_report(&snap, &output_path);

    config.generated_count = snap.saved;
    let _ = config.save();
}

// ──────── Display helpers ────────

/// Formats elapsed seconds into the most natural unit.
/// Datagen runs range from minutes to days, so we adapt the display.
fn format_eta(seconds: f64) -> String {
    let secs = seconds as u64;
    let (d, h, m, s) = (secs / 86_400, (secs % 86_400) / 3_600, (secs % 3_600) / 60, secs % 60);

    match d {
        1.. => format!("{d}d {h}h {m}m"),
        0 if h > 0 => format!("{h}h {m}m {s}s"),
        _ => format!("{m}m {s}s"),
    }
}

use crate::core::util::format_comma as format_num;

/// Safe division that returns 0.0 instead of NaN/Inf.
/// Percentage math shows up everywhere in datagen stats.
#[inline]
fn safe_div(num: f64, den: f64) -> f64 {
    if den > 0.0 { num / den } else { 0.0 }
}

/// Percentage with safe denominator.
#[inline]
fn pct(part: u64, whole: u64) -> f64 {
    safe_div(part as f64, whole as f64) * 100.0
}

// ──────── Stats snapshot ────────

/// A consistent point-in-time capture of all atomic counters.
///
/// Both the live dashboard and the final report need the same ~15 metrics.
/// Rather than scattering `load(Relaxed)` calls everywhere and duplicating
/// all the derived calculations, we capture once and compute from the snapshot.
struct Snapshot {
    elapsed: f64,
    generated: u64,
    target: u64,
    attempted: u64,
    saved: u64,
    passed_filters: u64,
    games: u64,
    plies: u64,
    filtered_quiet: u64,
    filtered_score: u64,
    filtered_ply: u64,
    filtered_pieces: u64,
    filtered_incorrect: u64,
    filtered_tactical: u64,
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
    /// Relaxed is fine here — we're displaying approximate progress,
    /// not synchronizing memory. Counters only ever increase.
    fn capture(global: &GlobalStats, generated: &AtomicUsize, target: u64) -> Self {
        Self {
            elapsed: global.start_time.elapsed().as_secs_f64(),
            generated: generated.load(Ordering::Relaxed) as u64,
            target,
            attempted: global.attempted.load(Ordering::Relaxed),
            saved: global.saved.load(Ordering::Relaxed),
            passed_filters: global.passed_filters.load(Ordering::Relaxed),
            games: global.games.load(Ordering::Relaxed),
            plies: global.plies.load(Ordering::Relaxed),
            filtered_quiet: global.filtered_quiet.load(Ordering::Relaxed),
            filtered_score: global.filtered_score.load(Ordering::Relaxed),
            filtered_ply: global.filtered_ply.load(Ordering::Relaxed),
            filtered_pieces: global.filtered_pieces.load(Ordering::Relaxed),
            filtered_incorrect: global.filtered_incorrect.load(Ordering::Relaxed),
            filtered_tactical: global.filtered_tactical.load(Ordering::Relaxed),
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

    /// Positions saved per second (current session only).
    fn rate(&self) -> f64 {
        safe_div(self.saved as f64, self.elapsed)
    }

    /// Overall completion percentage.
    fn progress_pct(&self) -> f64 {
        pct(self.generated, self.target)
    }

    /// What fraction of attempted positions survived all filters.
    fn pass_rate(&self) -> f64 {
        pct(self.passed_filters, self.attempted)
    }

    /// Average game length in plies — a useful sanity check.
    /// Typical selfplay games run 60-200 plies depending on resign threshold.
    fn avg_ply(&self) -> f64 {
        safe_div(self.plies as f64, self.games as f64)
    }

    /// ETA in seconds based on current throughput.
    fn eta_secs(&self) -> f64 {
        safe_div(self.target.saturating_sub(self.generated) as f64, self.rate())
    }

    fn total_filtered(&self) -> u64 {
        self.filtered_quiet
            + self.filtered_score
            + self.filtered_ply
            + self.filtered_pieces
            + self.filtered_incorrect
            + self.filtered_tactical
    }

    fn total_terminations(&self) -> u64 {
        self.term_check + self.term_stale + self.term_d50 + self.term_drep + self.term_dmat + self.term_draw_adj + self.term_resign
    }
}

// ──────── Dashboard ────────

/// Renders the live progress dashboard using ANSI cursor movement.
/// Overwrites the same 10 lines in-place for a clean, flicker-free display.
fn render_dashboard(snap: &Snapshot, first_frame: &mut bool) {
    // Quality badge reflects the underlying filter selectivity, not just how often
    // we skip positions for variety.
    let normalized = snap.pass_rate();

    let badge = if normalized > 30.0 {
        BADGE_OK
    } else if normalized > 10.0 {
        BADGE_LOW
    } else {
        BADGE_BAD
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
    println!("\r\x1b[KFiltered Quiet:  {}", format_num(snap.filtered_quiet));
    println!("\r\x1b[KFiltered Score:  {}", format_num(snap.filtered_score));
    println!("\r\x1b[KFiltered Ply:    {}", format_num(snap.filtered_ply));
    println!("\r\x1b[KFiltered Pieces: {}", format_num(snap.filtered_pieces));
    println!("\r\x1b[KFiltered Incorr: {}", format_num(snap.filtered_incorrect));
    println!("\r\x1b[KFiltered Tact:   {}", format_num(snap.filtered_tactical));
    println!("\r\x1b[KBest Move fails: {}", format_num(snap.search_fail));
    println!("\r\x1b[KRAM alloc:       {} MB", get_rss_kb() / 1024);
    println!("\r\x1b[KElapsed:         {}", format_eta(snap.elapsed));
    println!("\r\x1b[KETA:             {}", format_eta(snap.eta_secs()));

    let _ = std::io::stdout().flush();
}

/// Prints the final generation report — the definitive summary of a run.
fn print_final_report(snap: &Snapshot, output_path: &str) {
    let total_filt = snap.total_filtered();
    let total_term = snap.total_terminations();

    println!(
        "{GREEN}[OK]{RESET} {} positions in {:.1}s ({:.3}k/s)",
        format_num(snap.saved),
        snap.elapsed,
        snap.rate() / 1000.0,
    );
    println!("{GREEN}[OK]{RESET} Saved to {output_path}");
    println!();
    println!("[FINAL STATS]");
    println!("  Total Games:      {}", format_num(snap.games));
    println!("  Avg Plies/Game:   {:.1}", snap.avg_ply());
    println!();
    println!("  Positions:");
    println!("    Attempted:      {}", format_num(snap.attempted));
    println!("    Saved:          {} ({:.1}% pass rate)", format_num(snap.saved), snap.pass_rate(),);
    println!("    Filtered:       {}", format_num(total_filt));
    println!();
    println!("  Filter Breakdown:");
    println!("    Quiet filter:   {} ({:.1}%)", format_num(snap.filtered_quiet), pct(snap.filtered_quiet, total_filt),);
    println!("    Score filter:   {} ({:.1}%)", format_num(snap.filtered_score), pct(snap.filtered_score, total_filt),);
    println!("    Ply filter:     {} ({:.1}%)", format_num(snap.filtered_ply), pct(snap.filtered_ply, total_filt),);
    println!("    Pieces filter:  {} ({:.1}%)", format_num(snap.filtered_pieces), pct(snap.filtered_pieces, total_filt),);
    println!(
        "    Incorrect filt: {} ({:.1}%)",
        format_num(snap.filtered_incorrect),
        pct(snap.filtered_incorrect, total_filt),
    );
    println!("    Qsearch filt:   {} ({:.1}%)", format_num(snap.filtered_tactical), pct(snap.filtered_tactical, total_filt),);
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

    println!();
    println!("  Search Failures:  {}", format_num(snap.search_fail));
}

// ──────── Book loading ────────

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

/// Resolves the generation config from CLI args, optionally resuming
/// a previous run by loading saved state from disk.
fn resolve_config(args: &GenfensArgs) -> GenfensConfig {
    if args.resume
        && let Ok(mut cfg) = GenfensConfig::load()
    {
        cfg.target_count = args.target_count;
        cfg.book_paths.clone_from(&args.book_paths);
        return cfg;
    }
    GenfensConfig::from(args.clone())
}

/// Prints the launch banner — what we're about to do and how.
fn print_banner(config: &GenfensConfig, num_threads: usize, book_count: usize, start_count: usize) {
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

    println!("Resign: ±{}cp, Score filter: ±{}cp", config.resign_cp, config.score_filter,);

    if config.filter_quiet {
        println!("Filter: quiet positions only");
    }
    if config.min_ply > 0 {
        println!("Filter: min ply = {}", config.min_ply);
    }
    if config.min_pieces > 0 {
        println!("Filter: min pieces = {}", config.min_pieces);
    }
    if config.eval_contradiction_limit != i32::MAX {
        println!("Filter: eval contradiction limit = {} cp", config.eval_contradiction_limit);
    }
    if start_count > 0 {
        println!("Resume: {start_count} existing positions");
    }
    println!();
}

/// Flushes pending entries to disk and saves config state.
/// Called periodically during generation and once at the end.
fn flush_to_disk(output_path: &str, pending: &mut Vec<SoulEntry>, config: &GenfensConfig) {
    if pending.is_empty() {
        return;
    }
    if let Err(e) = append_encoded(output_path, pending) {
        eprintln!("{RED}[ERROR] Failed to save batch: {e}{RESET}");
    }
    pending.clear();
    let _ = config.save();
}

// ── Resume support ──

/// Reads the number of positions already present in an output file.
/// V3+ format stores the count in the header for 𝒪(1) lookup.
/// Older formats require a full deserialization pass.
fn load_existing_count(path: &str) -> usize {
    use std::{
        fs::File,
        io::{BufReader, Read},
        path::Path,
    };

    if !Path::new(path).exists() {
        return 0;
    }

    let Ok(file) = File::open(path) else { return 0 };
    let mut reader = BufReader::new(file);

    let mut magic = [0u8; 8];
    if reader.read_exact(&mut magic).is_err() {
        return 0;
    }

    if &magic == crate::tools::dataset::MAGIC_V5 || &magic == crate::tools::dataset::MAGIC_V6 {
        let mut buf = [0u8; 8];
        if reader.read_exact(&mut buf).is_ok() {
            return u64::from_le_bytes(buf) as usize;
        }
        return 0;
    }

    // Pre-V3: no header count — have to deserialize the whole file.
    // "Expensive", but only happens once on resume.
    load_encoded(path).map_or(0, |v| v.len())
}

fn print_help() {
    use crate::cli::Help;

    let h = Help::new(22);

    h.header("Usage:");
    println!("  soul genfens [OPTIONS]");
    println!();

    h.header("Options:");
    h.option_default("-n, --count", "<N>", "Target number of positions to generate", "8,000,000");
    h.option_default("-t, --threads", "<N>", "Number of threads", "auto");
    h.option_default("-o, --output", "<PATH>", "Output file path", "data.soul.zst");
    h.option_default("-b, --book", "<PATH>", "Opening book path", "UHO_Lichess_4852_v1.epd");
    h.option_default("-d, --depth", "<N>", "Search depth (default 6; MAX when --soft/--nodes set without --depth)", "6");
    h.option("--soft", "<N>", "Soft node limit");
    h.option("--nodes", "<N>", "Hard node limit");
    h.option_default("--plies", "<N>", "Max game length", "300");
    h.option_default("--buf", "<N>", "Buffer size per thread", "256");
    h.option("--resume", "", "Resume from existing config/output");
    h.option_default("--save-interval", "<N>", "Save interval", "5000");
    h.option_default("--random-plies", "<N>", "Random plies (half-moves) from book position (random-restart)", "6");
    h.option("--no-random-restart", "", "Disable random-restart; use full game mode");
    h.option("--startpos", "", "Use standard start position instead of book files");
    h.option_default("--sample", "<0-1>", "Randomly sample fraction of positions", "0.7");
    h.option("--all", "", "Disable quiet position filtering");
    h.option_default("--resign", "<CP>", "Resign threshold in centipawns", "800");
    h.option_default("--filter", "<CP>", "Max score for saved positions", "450");
    h.option_default("--qsearch", "<CP>", "Skip positions where |search - static| delta exceeds this threshold", "disabled");
    h.option_default("--min-ply", "<N>", "Skip positions before this ply", "0");
    h.option_default("--min-pieces", "<N>", "Skip positions with fewer pieces", "4");
    h.option_default(
        "--eval-contradiction-limit",
        "<CP>",
        "Skip positions where eval contradicts game outcome by more than this (centipawns)",
        "disabled",
    );
}

fn parse_args(args: &[&str]) -> GenfensArgs {
    let mut output_path = String::from("data.soul.zst");
    let mut book_paths = Vec::new();
    let mut target_count = 8_000_000;
    let mut depth: Option<i32> = None;
    let mut soft_nodes = None;
    let mut hard_nodes = None;
    let mut resign_cp = 800;
    let mut score_filter = 450;
    let mut max_plies = 300;
    let mut buffer_size = 256;
    let mut thread_count = None;
    let mut save_interval = 5000;
    let mut filter_quiet = true;
    let mut sample_rate = 0.7;
    let mut min_ply = 0usize;
    let mut min_pieces = 4u32;
    let mut eval_contradiction_limit = i32::MAX;
    let mut qsearch_filter = i32::MAX;
    let mut random_restart = true;
    let mut random_plies = 6usize;
    let mut startpos = false;
    let mut resume = false;

    let mut it = args.iter().copied();
    while let Some(flag) = it.next() {
        match flag {
            "-o" | "--output" => {
                if let Some(v) = it.next() {
                    output_path = v.to_string();
                }
            },
            "-b" | "--book" => {
                if let Some(v) = it.next() {
                    book_paths.push(v.to_string());
                }
            },
            "-n" | "--count" => {
                if let Some(v) = it.next() {
                    target_count = parse_suffix(v).unwrap_or(target_count);
                }
            },
            "-d" | "--depth" => {
                if let Some(v) = it.next() {
                    depth = v.parse().ok().or(depth);
                }
            },
            "--soft" => {
                if let Some(v) = it.next() {
                    soft_nodes = v.parse().ok();
                }
            },
            "--nodes" => {
                if let Some(v) = it.next() {
                    hard_nodes = v.parse().ok();
                }
            },
            "--resign" => {
                if let Some(v) = it.next() {
                    resign_cp = v.parse().unwrap_or(resign_cp);
                }
            },
            "--filter" => {
                if let Some(v) = it.next() {
                    score_filter = v.parse().unwrap_or(score_filter);
                }
            },
            "--plies" => {
                if let Some(v) = it.next() {
                    max_plies = v.parse().unwrap_or(max_plies);
                }
            },
            "--buf" => {
                if let Some(v) = it.next() {
                    buffer_size = v.parse().unwrap_or(buffer_size);
                }
            },
            "-t" | "--threads" => {
                if let Some(v) = it.next() {
                    thread_count = v.parse().ok();
                }
            },
            "--save-interval" => {
                if let Some(v) = it.next() {
                    save_interval = v.parse().unwrap_or(save_interval);
                }
            },
            "--all" => filter_quiet = false,
            "--sample" => {
                if let Some(v) = it.next() {
                    sample_rate = v.parse().unwrap_or(sample_rate);
                }
            },
            "--min-ply" => {
                if let Some(v) = it.next() {
                    min_ply = v.parse().unwrap_or(min_ply);
                }
            },
            "--min-pieces" => {
                if let Some(v) = it.next() {
                    min_pieces = v.parse().unwrap_or(min_pieces);
                }
            },
            "--eval-contradiction-limit" => {
                if let Some(v) = it.next() {
                    eval_contradiction_limit = v.parse().unwrap_or(eval_contradiction_limit);
                }
            },
            "--qsearch" => {
                if let Some(v) = it.next() {
                    qsearch_filter = v.parse().unwrap_or(qsearch_filter);
                }
            },
            "--random-plies" => {
                if let Some(v) = it.next() {
                    random_plies = v.parse().unwrap_or(random_plies);
                }
            },
            "--no-random-restart" => random_restart = false,
            "--startpos" => startpos = true,
            "--resume" => resume = true,
            "-h" | "--help" => {
                print_help();
                std::process::exit(0);
            },
            _ => {}, // Unknown flags silently ignored — engine convention.
        }
    }

    // Default book if none specified — UHO is the community standard
    // for balanced, non-degenerate openings.
    if book_paths.is_empty() {
        book_paths.push("UHO_Lichess_4852_v1.epd".to_string());
    }

    GenfensArgs {
        target_count,
        output_path,
        book_paths,
        depth,
        soft_nodes,
        hard_nodes,
        resign_cp,
        score_filter,
        max_plies,
        buffer_size,
        thread_count,
        save_interval,
        filter_quiet,
        sample_rate,
        min_ply,
        min_pieces,
        eval_contradiction_limit,
        qsearch_filter,
        random_restart,
        random_plies,
        startpos,
        resume,
    }
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

    // No suffix — strip commas and parse directly.
    s.replace(',', "").parse().ok()
}

/// Loads opening positions from an EPD file, falling back to raw FEN parsing.
/// Both formats are common in the chess datagen ecosystem.
fn load_epd_fens(path: &str) -> std::io::Result<Vec<String>> {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut fens = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some((board, _)) = parse_epd_str(&line) {
            // EPD parsed successfully — re-export as FEN to normalize formatting.
            fens.push(board.as_fen());
        } else if crate::core::board::Position::try_from_fen(&line).is_ok() {
            // Fallback: raw FEN line (no EPD operations field).
            fens.push(line);
        }
        // Lines that fail both parses are silently skipped,
        // they're comments, blank lines, or corrupted entries.
    }

    Ok(fens)
}
