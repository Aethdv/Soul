use std::{
    cell::RefCell,
    io::Error,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

use soul::{
    core::{
        board::Position as Board,
        defs::{Color, GameOutcome},
    },
    engine::{
        adjudication::check_adjudication,
        history::History,
        movegen::gen_legal_moves,
        search::{Limits, Protocol, SearchConfig, SearchDisplay, Searcher},
        search_params::SearchParams,
        tm::TimeManager,
        tt::TranspositionTable,
    },
};

use super::pentanomial::{GameResult, Pentanomial, pair_to_pentanomial};

/// Per-worker TT pool size.
const TT_POOL_MB: usize = 16;

/// Executes a full suite of candidate vs. baseline paired matches.
///
/// This is the primary evaluation loop for CMA-ES. It scatters the matches across
/// Rayon's thread pool, allowing the engine to gorge itself on all available CPU cores.
pub fn run_matches<F>(
    candidate_params: SearchParams,
    opponent_params: SearchParams,
    openings: &[String],
    tc: &str,
    on_pair_finish: F,
) -> (Pentanomial, u64, u64)
where
    F: Fn() + Sync + Send,
{
    use rayon::prelude::*;

    let limits = parse_tc(tc);

    let results: Vec<(Pentanomial, u64, u64)> = openings
        .par_iter()
        .map(|fen| play_match_pair(fen, &limits, candidate_params, opponent_params, &on_pair_finish))
        .collect();

    aggregate_results(results)
}

/// Facilitates direct validation matches between two specific parameter sets.
///
/// Shuffle once to get a random sample of `pairs` openings from the pool,
/// then share the index map across threads for zero-copy access.
#[must_use]
pub fn run_head_to_head(
    params_a: &SearchParams,
    params_b: &SearchParams,
    openings: &[String],
    pairs: usize,
    tc: &str,
) -> (f64, u64, u64) {
    use rayon::prelude::*;

    let limits = parse_tc(tc);

    // Shuffle indices once, share across threads
    let mut indices: Vec<usize> = (0..openings.len()).collect();
    fastrand::shuffle(&mut indices);

    let params_a = *params_a;
    let params_b = *params_b;

    let results: Vec<(Pentanomial, u64, u64)> = (0..pairs)
        .into_par_iter()
        .map(|i| {
            let fen = &openings[indices[i % indices.len()]];
            play_match_pair(fen, &limits, params_a, params_b, || {})
        })
        .collect();

    let (penta, nodes_a, nodes_b) = aggregate_results(results);
    (penta.mle_elo(), nodes_a, nodes_b)
}

/// Loads and validates opening FENs from disk.
///
/// # Errors
/// Returns an error if the I/O subsystem rejects the read request.
pub fn load_openings(path: &str) -> Result<Vec<String>, Error> {
    use std::{
        fs::File,
        io::{BufRead, BufReader},
    };

    if path == "startpos" {
        return Ok(vec!["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1".to_string()]);
    }

    let file = File::open(path)?;
    let reader = BufReader::new(file);

    let fens: Vec<String> = reader
        .lines()
        .map_while(Result::ok)
        .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
        .map(|line| {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 4 {
                if parts.len() >= 6 && parts[4].chars().all(|c| c.is_ascii_digit()) {
                    parts[..6].join(" ")
                } else {
                    format!("{} 0 1", parts[..4].join(" "))
                }
            } else {
                line.clone()
            }
        })
        .collect();

    Ok(fens)
}

fn aggregate_results(results: Vec<(Pentanomial, u64, u64)>) -> (Pentanomial, u64, u64) {
    let mut penta = Pentanomial::default();
    let mut total_candidate_nodes = 0u64;
    let mut total_baseline_nodes = 0u64;
    for (p, c_nodes, b_nodes) in results {
        penta.merge(&p);
        total_candidate_nodes += c_nodes;
        total_baseline_nodes += b_nodes;
    }
    (penta, total_candidate_nodes, total_baseline_nodes)
}

thread_local! {
    /// One pair of TTs per rayon worker, alive for the worker's lifetime.
    /// Cleared (not reallocated) between pairs so we pay 32 MB of malloc *once*
    /// per worker instead of per opening pair.
    static TT_POOL: RefCell<Option<(Arc<TranspositionTable>, Arc<TranspositionTable>)>> = const { RefCell::new(None) };
}

/// Hand out the worker-local TT pair, lazy-initialized and cleared on each call.
fn acquire_tts() -> (Arc<TranspositionTable>, Arc<TranspositionTable>) {
    TT_POOL.with(|cell| {
        let mut slot = cell.borrow_mut();
        let pair = slot
            .get_or_insert_with(|| (Arc::new(TranspositionTable::new(TT_POOL_MB)), Arc::new(TranspositionTable::new(TT_POOL_MB))));
        pair.0.clear();
        pair.1.clear();
        (Arc::clone(&pair.0), Arc::clone(&pair.1))
    })
}

fn play_match_pair<F: Fn()>(
    fen: &str,
    limits: &Limits,
    params_a: SearchParams,
    params_b: SearchParams,
    on_finish: F,
) -> (Pentanomial, u64, u64) {
    // Per-game stop flag, shared between both players of the pair.
    let stop = Arc::new(AtomicBool::new(false));

    let cfg_a = SearchConfig::new_full(limits.clone(), Instant::now(), stop.clone(), 0, SearchDisplay::SILENT, params_a);
    let cfg_b = SearchConfig::new_full(limits.clone(), Instant::now(), stop, 0, SearchDisplay::SILENT, params_b);

    // Thread local heap-allocated stacks
    let dummy_board = Board::default();
    let dummy_hist = vec![dummy_board.hash];
    let (tt_a, tt_b) = acquire_tts();
    let mut searcher_a = Box::new(Searcher::new(&cfg_a, &dummy_board, &dummy_hist, History::new(), tt_a));
    let mut searcher_b = Box::new(Searcher::new(&cfg_b, &dummy_board, &dummy_hist, History::new(), tt_b));

    // Round 1: A (White) vs B (Black)
    let (result_as_white, nodes_a_1, nodes_b_1) = play_game(fen, &cfg_a, &cfg_b, &mut searcher_a, &mut searcher_b);

    // ucinewgame-equivalent between the two games of the pair.
    searcher_a.clear_history();
    searcher_b.clear_history();

    // Round 2: B (White) vs A (Black)
    let (result_for_b, nodes_b_2, nodes_a_2) = play_game(fen, &cfg_b, &cfg_a, &mut searcher_b, &mut searcher_a);

    // Invert result (B's POV → A's POV)
    let result_as_black_perspective = match result_for_b {
        GameResult::Win => GameResult::Loss,
        GameResult::Loss => GameResult::Win,
        GameResult::Draw => GameResult::Draw,
    };

    on_finish();

    (
        pair_to_pentanomial(result_as_white, result_as_black_perspective),
        nodes_a_1 + nodes_a_2,
        nodes_b_1 + nodes_b_2,
    )
}

/// Dispatches a single game.
///
/// Automatically routes the match to either a fixed-depth/fixed-node simulator
/// (which is blindingly fast because it doesn't need to juggle system clocks)
/// or a fully simulated time-control match.
fn play_game<'a>(
    fen: &str,
    cfg_white: &'a SearchConfig,
    cfg_black: &'a SearchConfig,
    searcher_white: &mut Searcher<'a>,
    searcher_black: &mut Searcher<'a>,
) -> (GameResult, u64, u64) {
    let uses_clock =
        cfg_white.limits.wtime > 0 && cfg_white.limits.movetime == 0 && cfg_white.limits.depth == 0 && cfg_white.limits.nodes == 0;

    const MAX_GAME_PLIES: usize = 300;

    let mut board = Board::from_fen(fen);
    let mut history: Vec<u64> = vec![board.hash];

    let mut white_nodes = 0u64;
    let mut black_nodes = 0u64;
    let mut accumulator = board.get_initial_accumulator();

    let mut win_adj_counter = 0;
    let mut draw_adj_counter = 0;
    let mut last_score = 0;

    let mut white_time_ms = cfg_white.limits.wtime;
    let mut black_time_ms = cfg_black.limits.btime;
    let white_inc = cfg_white.limits.winc;
    let black_inc = cfg_black.limits.binc;

    for _ply in 0..MAX_GAME_PLIES {
        let moves = gen_legal_moves(&board);
        if moves.is_empty() {
            let in_check = board.checkers().is_not_empty();
            if in_check {
                return (if board.stm == Color::White { GameResult::Loss } else { GameResult::Win }, white_nodes, black_nodes);
            }
            return (GameResult::Draw, white_nodes, black_nodes);
        }
        if is_draw(&board, &history) {
            return (GameResult::Draw, white_nodes, black_nodes);
        }

        let move_start = Instant::now();

        let (searcher, cfg) =
            if board.stm == Color::White { (&mut *searcher_white, cfg_white) } else { (&mut *searcher_black, cfg_black) };

        searcher.reset(cfg, &board, &history);

        if uses_clock {
            let mut limits = cfg.limits.clone();
            // Just pass the current actual clocks — no branching needed.
            limits.wtime = white_time_ms;
            limits.btime = black_time_ms;
            let phase = i32::from(accumulator.to_array()[2]);
            searcher.tm =
                TimeManager::new(&limits, move_start, board.stm, cfg.overhead, phase, history.len() as u64, &cfg.search_params);
        }

        cfg.stop.store(false, Ordering::Release);
        searcher.iterative_deepening();

        let score = searcher.best_score().unwrap_or(0);
        let ply = (board.fullmove_number as usize - 1) * 2 + (board.stm as usize);
        if let Some(res) = check_adjudication(score, last_score, &mut win_adj_counter, &mut draw_adj_counter, board.stm, ply) {
            let result = match res {
                GameOutcome::WhiteWins => GameResult::Win,
                GameOutcome::BlackWins => GameResult::Loss,
                GameOutcome::Draw => GameResult::Draw,
            };
            return (result, white_nodes, black_nodes);
        }
        last_score = score;

        let nodes = searcher.nodes;
        if board.stm == Color::White {
            white_nodes += nodes;
        } else {
            black_nodes += nodes;
        }

        if uses_clock {
            let elapsed_ms = move_start.elapsed().as_millis() as u64;
            if board.stm == Color::White && elapsed_ms > white_time_ms {
                return (GameResult::Loss, white_nodes, black_nodes);
            } else if board.stm == Color::Black && elapsed_ms > black_time_ms {
                return (GameResult::Win, white_nodes, black_nodes);
            }

            if board.stm == Color::White {
                white_time_ms = white_time_ms - elapsed_ms + white_inc;
            } else {
                black_time_ms = black_time_ms - elapsed_ms + black_inc;
            }
        }

        let best_move = searcher.best_move().unwrap_or_else(|| *moves.iter().next().unwrap());

        board.make_move(best_move, &mut accumulator);
        history.push(board.hash);
    }
    (GameResult::Draw, white_nodes, black_nodes)
}

fn is_draw(pos: &Board, history: &[u64]) -> bool {
    pos.halfmove_clock >= 100 || pos.is_draw_by_material() || pos.is_threefold_repetition(history)
}

/// Parses a Time Control (TC) string into a Limits struct.
///
/// Supported formats:
/// - `movetime=100`: 100ms per move
/// - `depth=6`: search to depth 6
/// - `nodes=10000`: search 10,000 nodes
/// - `4+0.04`: 4 seconds base + 0.04s increment
/// - `5.0`: bare float as total time for the game (in seconds)
fn parse_tc(tc: &str) -> Limits {
    let mut limits = Limits { silent: true, protocol: Protocol::Uci, ..Default::default() };

    if let Some(val) = tc.strip_prefix("movetime=") {
        // Fixed time per move (in milliseconds)
        limits.movetime = val.parse().unwrap_or(100);
    } else if let Some(val) = tc.strip_prefix("depth=") {
        limits.depth = val.parse().unwrap_or(6);
    } else if let Some(val) = tc.strip_prefix("nodes=") {
        limits.nodes = val.parse().unwrap_or(10000);
    } else if tc.contains('+') {
        let parts: Vec<&str> = tc.split('+').collect();
        if parts.len() == 2 {
            let base_secs: f64 = parts[0].parse().unwrap_or(4.0);
            let inc_secs: f64 = parts[1].parse().unwrap_or(0.04);

            limits.wtime = (base_secs * 1000.0) as u64;
            limits.btime = (base_secs * 1000.0) as u64;
            limits.winc = (inc_secs * 1000.0) as u64;
            limits.binc = (inc_secs * 1000.0) as u64;
        }
    } else if let Ok(secs) = tc.parse::<f64>() {
        let ms = (secs * 1000.0) as u64;
        limits.wtime = ms;
        limits.btime = ms;
    } else {
        eprintln!("\x1b[33m[!] TC '{tc}' not recognized, defaulting to depth 6\x1b[0m");
        limits.depth = 6;
    }
    limits
}
