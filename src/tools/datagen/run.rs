//! Self-play data generation.

use std::{
    fs::OpenOptions,
    io::{self, Write, stdout},
    num::NonZero,
    process,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::channel,
    },
    thread::{available_parallelism, sleep, spawn},
    time::Duration,
};

use super::{
    config::{CONFIG_FILENAME, DatagenConfig},
    stats::{RunStats, Snapshot, Termination, rss_kib},
    worker::{Game, SelfPlayWorker},
};
use crate::{
    cli::Help,
    color::{self, RESET},
    core::{
        board::STARTPOS,
        defs::MAX_DEPTH,
        util::{fmt_comma, fmt_duration, pct},
    },
    tools::dataset::{load_epd_fens, scan_viri_games, write_game},
};

const GREEN: &str = color::OK_PEN;
const YELLOW: &str = color::WARN_PEN;
const RED: &str = color::ALARM_PEN;

const DASHBOARD_INTERVAL: Duration = Duration::from_millis(100);
const DASHBOARD_LINES: usize = 12;

pub fn run(args: &[&str], stop: &Arc<AtomicBool>) -> bool {
    let (config, book_fens, existing_positions) = match prepare(args) {
        Ok(ready) => ready,
        Err(message) => {
            eprintln!("{RED}datagen: {message}{RESET}");
            return false;
        },
    };

    let book = Arc::new(book_fens);
    let target_positions = config.target_count;
    let output_path = config.output_path.clone();
    let save_interval = config.save_interval as u64;
    let num_threads = config
        .thread_count
        .unwrap_or_else(|| (available_parallelism().map_or(1, NonZero::get) / 2).max(1));
    let stats = Arc::new(RunStats::new(existing_positions, target_positions));

    print_banner(&config, num_threads, book.len(), existing_positions);

    let stop_for_handler = Arc::clone(stop);
    let _ = ctrlc::set_handler(move || {
        println!("\n{YELLOW}Interrupt received, stopping workers...{RESET}");
        stop_for_handler.store(true, Ordering::SeqCst);
    });

    let mut last_saved_positions = existing_positions;
    let mut saving_enabled = true;
    let mut pending: Vec<u8> = Vec::with_capacity(config.save_interval.saturating_mul(8));

    let dashboard = {
        let stats = Arc::clone(&stats);
        let stop = Arc::clone(stop);

        spawn(move || {
            let mut first_frame = true;

            while !stop.load(Ordering::Relaxed) {
                render_dashboard(&stats.snapshot(), &mut first_frame);
                sleep(DASHBOARD_INTERVAL);
            }
        })
    };

    let (tx, rx) = channel::<Game>();
    let mut handles = Vec::with_capacity(num_threads);

    for _ in 0..num_threads {
        let mut worker = SelfPlayWorker::new(Arc::clone(&book), &config);
        let tx = tx.clone();
        let stop = Arc::clone(stop);
        let stats = Arc::clone(&stats);

        handles.push(spawn(move || {
            while !stop.load(Ordering::Relaxed) && !stats.is_done() {
                let game = worker.play_game();
                if !game.moves.is_empty() && tx.send(game).is_err() {
                    break;
                }
            }
        }));
    }

    drop(tx);

    for game in rx {
        let total_positions = stats.record(game.moves.len() as u64, game.termination, game.search_failures);

        // The header's eval is the opening position's, the first recorded score.
        let opening_eval = game.moves.first().map_or(0, |&(_, score)| score);
        write_game(&mut pending, &game.opening, game.result, opening_eval, &game.moves);

        if saving_enabled && total_positions >= last_saved_positions.saturating_add(save_interval) {
            match append_pending(&output_path, &mut pending) {
                Ok(()) => last_saved_positions = total_positions,
                Err(e) => {
                    eprintln!("{RED}datagen: cannot save {output_path}: {e}{RESET}");
                    stop.store(true, Ordering::SeqCst);
                    saving_enabled = false;
                },
            }
        }

        if total_positions >= target_positions {
            stop.store(true, Ordering::SeqCst);
        }
    }

    for handle in handles {
        handle.join().unwrap();
    }

    stop.store(true, Ordering::Relaxed);
    let _ = dashboard.join();

    let saved = append_pending(&output_path, &mut pending)
        .inspect_err(|e| eprintln!("{RED}datagen: cannot save {output_path}: {e}{RESET}"))
        .is_ok();

    println!();
    print_final_report(&stats.snapshot(), &output_path, saved);
    saved
}

fn prepare(args: &[&str]) -> Result<(DatagenConfig, Vec<String>, u64), String> {
    let (config, resume) = resolve_config(args)?;
    if config.max_plies == 0 {
        return Err("--plies must be at least 1".into());
    }
    if config.thread_count == Some(0) {
        return Err("--threads must be at least 1".into());
    }

    let existing_positions = if resume { count_existing_positions(&config.output_path)? } else { 0 };

    let book_fens = if config.startpos {
        vec![STARTPOS.to_string()]
    } else {
        let fens = load_book_fens(&config.book_paths);
        if fens.is_empty() {
            return Err("no opening positions loaded".into());
        }
        fens
    };

    println!("Total starting positions: {GREEN}{}{RESET}", book_fens.len());
    config.save().map_err(|e| format!("cannot save {CONFIG_FILENAME}: {e}"))?;
    Ok((config, book_fens, existing_positions))
}

fn render_dashboard(snapshot: &Snapshot, first_frame: &mut bool) {
    let badge = if snapshot.games == 0 {
        String::new()
    } else if snapshot.avg_ply() > 60.0 {
        format!("{GREEN}[OK]{RESET}")
    } else if snapshot.avg_ply() > 30.0 {
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

    print!("\x1b[{DASHBOARD_LINES}A");

    println!(
        "\r\x1b[KPositions:       {}/{} ({:.1}%) {badge}",
        fmt_comma(snapshot.total_positions),
        fmt_comma(snapshot.target_count),
        snapshot.progress_pct(),
    );
    println!("\r\x1b[KRate:            {:.3} k/s", snapshot.rate() / 1000.0);
    println!("\r\x1b[KGames:           {}", fmt_comma(snapshot.games));
    println!("\r\x1b[KAvg ply:         {:.1}", snapshot.avg_ply());
    println!("\r\x1b[KCheckmate:       {}", fmt_comma(snapshot.count(Termination::Checkmate)));
    println!("\r\x1b[KDraw (rules):    {}", fmt_comma(snapshot.rule_draws()));
    println!("\r\x1b[KDraw (adj):      {}", fmt_comma(snapshot.count(Termination::AdjudicatedDraw)));
    println!("\r\x1b[KWin (adj):       {}", fmt_comma(snapshot.count(Termination::AdjudicatedWin)));
    println!("\r\x1b[KSearch failures: {}", fmt_comma(snapshot.search_failures));
    println!("\r\x1b[KRSS:             {} MiB", rss_kib() / 1024);
    println!("\r\x1b[KElapsed:         {}", fmt_duration((snapshot.elapsed * 1000.0) as u64));
    println!("\r\x1b[KETA:             {}", fmt_duration((snapshot.eta_secs() * 1000.0) as u64));

    let _ = stdout().flush();
}

fn print_final_report(snapshot: &Snapshot, output_path: &str, saved: bool) {
    println!(
        "{GREEN}[OK]{RESET} {} positions in {:.1}s ({:.3}k/s)",
        fmt_comma(snapshot.positions),
        snapshot.elapsed,
        snapshot.rate() / 1000.0,
    );

    if saved {
        println!("{GREEN}[OK]{RESET} Saved to {output_path}");
    } else {
        println!("{RED}[ERROR]{RESET} Not saved to {output_path}");
    }

    println!();
    println!("[FINAL STATS]");
    println!("  Games:            {}", fmt_comma(snapshot.games));
    println!("  Total Positions:  {}", fmt_comma(snapshot.total_positions));
    println!("  Avg Plies/Game:   {:.1}", snapshot.avg_ply());
    println!("  Search Failures:  {}", fmt_comma(snapshot.search_failures));
    println!();
    println!("  Game Terminations:");

    for t in Termination::ALL {
        let count = snapshot.count(t);
        println!("    {:<17} {} ({:.1}%)", t.label(), fmt_comma(count), pct(count, snapshot.games));
    }
}

fn load_book_fens(paths: &[String]) -> Vec<String> {
    println!("Loading opening books...");
    let mut all_fens = Vec::new();
    for path in paths {
        match load_epd_fens(path) {
            Ok(fens) => {
                println!("  Loaded {} positions from '{path}'", fens.len());
                all_fens.extend(fens);
            },
            Err(e) => eprintln!("  {RED}Failed to load {path}: {e}{RESET}"),
        }
    }
    all_fens
}

fn resolve_config(args: &[&str]) -> Result<(DatagenConfig, bool), String> {
    let mut config = DatagenConfig::default();
    let resume = apply_args(args, &mut config)?;
    if resume {
        config = DatagenConfig::load()
            .map_err(|e| format!("cannot read {CONFIG_FILENAME}: {e}"))?
            .ok_or_else(|| format!("no {CONFIG_FILENAME} to resume"))?;
        apply_args(args, &mut config)?;
    }
    Ok((config, resume))
}

fn print_banner(config: &DatagenConfig, num_threads: usize, opening_count: usize, existing_positions: u64) {
    println!("Starting with {num_threads} threads");
    println!("Target: {} positions", config.target_count);
    println!("Output: {}", config.output_path);

    if config.startpos {
        println!("Book: startpos");
    } else {
        println!("Book: {} ({opening_count} openings)", config.book_paths.join(", "));
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

    if !config.adjudicate {
        println!("Adjudication: off");
    }

    if existing_positions > 0 {
        println!("Resume: {existing_positions} existing positions");
    }
    println!();
}

fn append_pending(output_path: &str, pending: &mut Vec<u8>) -> io::Result<()> {
    let mut file = OpenOptions::new().create(true).append(true).open(output_path)?;
    if pending.is_empty() {
        return Ok(());
    }

    let original_len = file.metadata()?.len();
    if let Err(e) = file.write_all(pending) {
        let _ = file.set_len(original_len);
        return Err(e);
    }

    pending.clear();
    Ok(())
}

fn count_existing_positions(path: &str) -> Result<u64, String> {
    match scan_viri_games(path) {
        Ok(scan) => Ok(scan.plies),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Ok(0),
        Err(e) => Err(format!("cannot resume {path}: {e}")),
    }
}

fn print_help() {
    let h = Help::new(22);
    let defaults = DatagenConfig::default();
    let depth_help = "Search depth (MAX when --soft/--nodes set without --depth)";

    h.header("Usage:");
    println!("  soul datagen [OPTIONS]");
    println!();

    h.header("Options:");
    h.option_default("-n, --count", "<N>", "Target number of positions to generate", &fmt_comma(defaults.target_count));
    h.option_default("-t, --threads", "<N>", "Number of worker threads", "auto");
    h.option_default("-o, --output", "<PATH>", "Output file path", &defaults.output_path);
    h.option_default("-b, --book", "<PATH>", "Opening book path, repeatable", &defaults.book_paths.join(", "));
    h.option_default("-d, --depth", "<N>", depth_help, &defaults.depth.to_string());
    h.option("--soft", "<N>", "Soft node limit per move");
    h.option("--nodes", "<N>", "Hard node limit per move");
    h.option_default("--plies", "<N>", "Maximum plies per game", &defaults.max_plies.to_string());
    h.option("--no-adj", "", "Disable score adjudication; play to the ply cap");
    h.option("--adj", "", "Enable score adjudication (default)");
    h.option("--resume", "", &format!("Resume the run in {CONFIG_FILENAME}"));
    h.option_default("--save-interval", "<N>", "Positions between saves", &defaults.save_interval.to_string());
    h.option("--startpos", "", "Use the standard start position instead of book files");
    h.header("Notes:");
    println!("  Every ply of every game is recorded. Which of them train is decided at load");
    println!("  time by the tuner's replay filter, not here.");
}

fn apply_args(args: &[&str], config: &mut DatagenConfig) -> Result<bool, String> {
    let mut depth = None;
    let mut book_override = false;
    let mut resume = false;
    let mut it = args.iter().copied();

    while let Some(flag) = it.next() {
        match flag {
            "-o" | "--output" => config.output_path = next_value(&mut it, flag)?.to_string(),
            "-b" | "--book" => {
                if !book_override {
                    config.book_paths.clear();
                    book_override = true;
                }
                config.book_paths.push(next_value(&mut it, flag)?.to_string());
                config.startpos = false;
            },
            "-n" | "--count" => config.target_count = parse_count(next_value(&mut it, flag)?)?,
            "-d" | "--depth" => depth = Some(parse_next(&mut it, flag)?),
            "--soft" => config.soft_nodes = Some(parse_next(&mut it, flag)?),
            "--nodes" => config.hard_nodes = Some(parse_next(&mut it, flag)?),
            "-t" | "--threads" => config.thread_count = Some(parse_next(&mut it, flag)?),
            "--plies" => config.max_plies = parse_next(&mut it, flag)?,
            "--adj" => config.adjudicate = true,
            "--no-adj" => config.adjudicate = false,
            "--save-interval" => config.save_interval = parse_next(&mut it, flag)?,
            "--startpos" => config.startpos = true,
            "--resume" => resume = true,
            "-h" | "--help" => {
                print_help();
                process::exit(0);
            },
            _ => return Err(format!("unknown option: {flag}")),
        }
    }

    if let Some(d) = depth {
        config.depth = d;
    } else if config.soft_nodes.is_some() || config.hard_nodes.is_some() {
        config.depth = MAX_DEPTH;
    }

    Ok(resume)
}

fn next_value<'a>(it: &mut impl Iterator<Item = &'a str>, flag: &str) -> Result<&'a str, String> {
    it.next().ok_or_else(|| format!("{flag} needs a value"))
}

fn parse_next<'a, T: FromStr>(it: &mut impl Iterator<Item = &'a str>, flag: &str) -> Result<T, String> {
    let value = next_value(it, flag)?;
    value.parse().map_err(|_| format!("invalid value for {flag}: {value}"))
}

fn parse_count(value: &str) -> Result<u64, String> {
    let lower = value.to_lowercase();
    for (suffix, multiplier) in [("b", 1e9), ("m", 1e6), ("k", 1e3)] {
        if let Some(stem) = lower.strip_suffix(suffix) {
            let n: f64 = stem.parse().map_err(|_| format!("invalid count: {value}"))?;
            return (n >= 0.0)
                .then_some((n * multiplier) as u64)
                .ok_or_else(|| format!("invalid count: {value}"));
        }
    }
    value.replace(',', "").parse().map_err(|_| format!("invalid count: {value}"))
}
