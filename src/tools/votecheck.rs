//! Does thread voting pick a better move than the main thread alone?

use std::{
    collections::HashMap,
    fs,
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use crate::{
    color::{BOLD, RESET, Rgb, ansi_fg},
    core::{
        board::Position,
        defs::{INF, is_mate},
        moves::Move,
    },
    engine::{
        history::History,
        search::{Limits, SearchConfig, Searcher, ThreadResult},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    protocols::smp::table_and_pool,
};

const DIM: Rgb = (108, 112, 134);
const TEXT: Rgb = (205, 214, 244);
const GREEN: Rgb = (166, 227, 161);
const RED: Rgb = (243, 139, 168);

const FENS: &str = include_str!("../data/speedtest.fens");

#[derive(Clone, Copy)]
enum Weighting {
    ScoreDepth,
    Score,
    Depth,
}

impl Weighting {
    const ALL: [(Self, &'static str); 3] = [(Self::ScoreDepth, "score x depth"), (Self::Score, "score"), (Self::Depth, "depth")];

    fn of(self, r: &ThreadResult, min_score: i32) -> i32 {
        match self {
            Self::ScoreDepth => (r.score - min_score + 10) * r.depth,
            Self::Score => r.score - min_score + 14,
            Self::Depth => r.depth,
        }
    }
}

struct Probe {
    results: Vec<ThreadResult>,
    values: HashMap<u16, i32>,
}

pub fn run(args: &[&str]) {
    let positions = args.first().and_then(|s| s.parse::<usize>().ok()).unwrap_or(30);
    let threads = args.get(1).and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
    let movetime = args.get(2).and_then(|s| s.parse::<u64>().ok()).unwrap_or(200);
    let ref_depth = args.get(3).and_then(|s| s.parse::<i32>().ok()).unwrap_or(22);
    let owned = args.get(4).map(|path| fs::read_to_string(path).expect("read fen file"));
    let source = owned.as_deref().unwrap_or(FENS);
    let all = source.lines().filter(|l| !l.trim().is_empty());
    let fens: Vec<&str> = if positions > 0 { all.take(positions).collect() } else { all.collect() };

    if fens.is_empty() {
        eprintln!("votecheck: no positions to search");
        return;
    }
    let (tt, pool) = table_and_pool(64, threads);
    let ref_tt = Arc::new(TranspositionTable::new(64, 1));

    println!();
    println!(
        "  {}votecheck{}  {} positions, {threads} threads, {movetime}ms, reference depth {ref_depth}",
        BOLD,
        RESET,
        fens.len()
    );
    println!();

    let mut main_history = History::new();
    let mut spreads: Vec<i32> = Vec::new();
    let mut probed: Vec<Probe> = Vec::new();

    let start = Instant::now();

    for (i, fen) in fens.iter().enumerate() {
        print!("\r  {}searching {}/{}{}   ", ansi_fg(DIM), i + 1, fens.len(), RESET);
        io::stdout().flush().ok();

        let board = Position::from_fen(fen);
        let results = search_pool(&board, &tt, &pool, threads, movetime, &mut main_history);

        let voters: Vec<i32> = results.iter().filter(|r| r.score != -INF && !is_mate(r.score)).map(|r| r.score).collect();
        if !voters.is_empty() {
            spreads.push(voters.iter().max().unwrap() - voters.iter().min().unwrap());
        }

        let mut proposed: Vec<u16> = results.iter().map(|r| r.mv.inner()).collect();
        proposed.sort_unstable();
        proposed.dedup();

        let mut values = HashMap::new();
        if proposed.len() > 1 {
            for mv in proposed {
                values.insert(mv, settle(&board, Move::from_u16(mv), ref_depth, &ref_tt));
            }
        }
        probed.push(Probe { results, values });
    }

    println!("\r  {}{} positions in {:.1}s{}          ", ansi_fg(DIM), probed.len(), start.elapsed().as_secs_f64(), RESET);
    println!();

    spreads.sort_unstable();
    let split = probed.iter().filter(|p| !p.values.is_empty()).count();

    if spreads.is_empty() {
        eprintln!("votecheck: every position ended in a proven mate, nothing to compare");
        return;
    }

    println!(
        "  {}cross-thread score spread{}  p50 {}  p90 {}  max {}",
        ansi_fg(TEXT),
        RESET,
        spreads[spreads.len() / 2],
        spreads[spreads.len() * 9 / 10],
        spreads[spreads.len() - 1]
    );
    println!("  {}threads disagreed on the move in {split} of {} positions{}", ansi_fg(TEXT), probed.len(), RESET);
    println!();
    println!("  {}weighting        off-main   overruled   better   worse    net cp{}", ansi_fg(DIM), RESET);

    for (weighting, name) in Weighting::ALL {
        let (mut overruled, mut better, mut worse, mut net) = (0, 0, 0, 0i64);
        let mut off_main = 0;

        for p in &probed {
            let main_mv = p.results[0].mv.inner();
            let winner = pick(&p.results, weighting);
            off_main += usize::from(winner != 0);

            let picked = p.results[winner].mv.inner();
            if picked == main_mv || p.values.is_empty() {
                continue;
            }
            overruled += 1;
            let delta = p.values[&picked] - p.values[&main_mv];
            net += i64::from(delta);
            if delta > 0 {
                better += 1;
            } else if delta < 0 {
                worse += 1;
            }
        }

        let tint = if net > 0 { GREEN } else { RED };
        println!("  {name:<15} {off_main:>8}   {overruled:>9}   {better:>6}   {worse:>5}   {}{net:>+7}{}", ansi_fg(tint), RESET);
    }
    println!();
}

fn search_pool(
    board: &Position,
    tt: &Arc<TranspositionTable>,
    pool: &Arc<crate::protocols::smp::LazySmpPool>,
    threads: usize,
    movetime: u64,
    main_history: &mut History,
) -> Vec<ThreadResult> {
    let limits = Limits { movetime, silent: true, ..Default::default() };
    let mut cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, SearchParams::default());
    cfg.threads = threads;
    cfg.node_slots = SearchConfig::node_slots(threads);
    cfg.result_slots = SearchConfig::result_slots(threads);

    let trail = vec![board.hash];
    pool.launch(&cfg, *board, &trail);

    let mut ctx = Searcher::new(&cfg, board, &trail, tt.clone());
    ctx.iterative_deepening(main_history);

    cfg.stop.store(true, Ordering::Relaxed);
    pool.wait();
    cfg.stop.store(false, Ordering::Relaxed);

    cfg.result_slots.iter().map(|s| ThreadResult::unpack(s.load(Ordering::Acquire))).collect()
}

/// One move's value, from a deep search with the root narrowed to it alone.
fn settle(board: &Position, mv: Move, depth: i32, tt: &Arc<TranspositionTable>) -> i32 {
    // SAFETY: single-threaded.
    unsafe { tt.clear(1) };

    let limits = Limits { depth, silent: true, searchmoves: vec![mv], ..Default::default() };
    let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, SearchParams::default());
    let mut searcher = Searcher::new(&cfg, board, &[board.hash], tt.clone());
    searcher.iterative_deepening(&mut History::new());
    searcher.prev_score
}

/// The thread the tally lands on under one weighting.
fn pick(results: &[ThreadResult], weighting: Weighting) -> usize {
    let min_score = results.iter().filter(|r| r.score != -INF).map(|r| r.score).min().unwrap_or(0);
    let weight = |r: &ThreadResult| weighting.of(r, min_score);

    let mut votes: HashMap<u16, i32> = HashMap::new();
    for r in results {
        *votes.entry(r.mv.inner()).or_default() += weight(r);
    }

    let mut best = 0;
    for cur in 1..results.len() {
        let (incumbent, candidate) = (&results[best], &results[cur]);
        let take = votes[&candidate.mv.inner()] > votes[&incumbent.mv.inner()]
            || (votes[&candidate.mv.inner()] == votes[&incumbent.mv.inner()] && weight(candidate) > weight(incumbent));

        if take && candidate.score != -INF {
            best = cur;
        }
    }

    if results[best].mv == results[0].mv { 0 } else { best }
}
