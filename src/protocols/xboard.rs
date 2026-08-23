//! XBoard / CECP protocol implementation.
//!
//! One `XBoardState` and a mode: Normal searches on the engine's turn, Force only
//! verifies the moves it is fed, Analyze searches until told to stop. Each search
//! runs on a thread spawned here, so the command loop keeps reading stdin, and
//! anything calling `stop_search` joins that thread before it proceeds.

use std::{
    io::{self, StdinLock, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        nonpoison::Mutex,
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use crate::{
    core::{
        board::{Position, STARTPOS},
        defs::{Color, Protocol},
        error::EngineError,
        zobrist::key_side,
    },
    engine::{
        history::History,
        search::{Limits, SearchConfig, SearchDisplay, Searcher, ThreadResult},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    protocols::{
        notation::parse_uci_move,
        smp::{self, LazySmpPool, table_and_pool},
    },
    weave::Vi16x8,
};

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Mode {
    Normal,  // thinks on its turn
    Force,   // only verifies moves, never searches
    Analyze, // searches infinitely
}

struct XBoardState {
    board: Position,
    accumulator: Vi16x8,
    history: Vec<u64>,
    persistent_history: Arc<Mutex<History>>,
    tt: Arc<TranspositionTable>,
    mode: Mode,
    stop_signal: Arc<AtomicBool>,
    search_thread: Option<JoinHandle<()>>,
    limits: Limits,
    hash_size: usize,
    overhead: u64,
    show_wdl: bool,
    engine_side: Option<Color>,
    is_frc: bool,
    nps: Option<u64>,
    threads: usize,
    smp_pool: Arc<LazySmpPool>,
}

pub fn main_loop(lines: &mut io::Lines<StdinLock>) {
    let mut state = XBoardState::new();

    while let Some(Ok(line)) = lines.next() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        let mut tokens = trimmed.split_whitespace();
        let Some(cmd) = tokens.next() else {
            continue;
        };

        if cmd == "quit" {
            state.stop_search();
            break;
        }

        handle_command(&mut state, cmd, &mut tokens);
    }
}

impl XBoardState {
    fn new() -> Self {
        let board = Position::from_fen(STARTPOS);
        let history = vec![board.hash];
        let (tt, smp_pool) = table_and_pool(16, 1);

        Self {
            accumulator: board.get_initial_accumulator(),
            board,
            history,
            persistent_history: Arc::new(Mutex::new(History::new())),
            tt,
            mode: Mode::Normal,
            stop_signal: Arc::new(AtomicBool::new(false)),
            search_thread: None,
            limits: Limits { protocol: Protocol::XBoard, ..Default::default() },
            hash_size: 16,
            overhead: 10,
            show_wdl: false,
            engine_side: None,
            is_frc: false,
            nps: None,
            threads: 1,
            smp_pool,
        }
    }

    fn stop_search(&mut self) {
        self.stop_signal.store(true, Ordering::Relaxed);

        if let Some(handle) = self.search_thread.take()
            && let Err(e) = handle.join()
        {
            eprintln!("Error: {}", EngineError::from_panic(e.as_ref()));
        }

        self.stop_signal.store(false, Ordering::Relaxed);
    }

    /// Reset to a fresh root position: rebuild the accumulator and reseed the
    /// repetition history with the new root hash.
    fn load_position(&mut self, board: Position) {
        self.board = board;
        self.board.is_frc = self.is_frc;
        self.accumulator = self.board.get_initial_accumulator();
        self.history.clear();
        self.history.push(self.board.hash);
    }

    fn start_search(&mut self) {
        self.stop_search();

        if self.mode == Mode::Force {
            return;
        }

        let mut limits = self.limits.clone();

        // Apply nps node budget if set alongside fixed move time.
        // The XBoard nps feature throttles output rate, but the pragmatic
        // approximation (used by most engines) is a total node cap per move.
        if let Some(nps) = self.nps
            && limits.movetime > 0
        {
            limits.nodes = nps * limits.movetime / 1000;
        }

        let board = self.board;
        let stop = self.stop_signal.clone();
        let history = self.history.clone();
        let overhead = self.overhead;
        let show_wdl = self.show_wdl;
        let shared_history = Arc::clone(&self.persistent_history);
        let mut history_table = shared_history.lock().clone();

        let tt = self.tt.clone();
        let threads = self.threads;
        let pool = self.smp_pool.clone();

        self.search_thread = Some(thread::spawn(move || {
            let display = SearchDisplay { show_wdl, ..SearchDisplay::DEFAULT };
            let mut cfg = SearchConfig::new_full(limits, Instant::now(), stop, overhead, display, SearchParams::default());
            cfg.threads = threads;
            cfg.node_slots = SearchConfig::node_slots(threads);
            cfg.result_slots = SearchConfig::result_slots(threads);

            // ── Lazy SMP
            // Persistent helpers fan out across the depth ladder alongside main;
            // the TT is the only shared surface. Main finishes, then we signal
            // the helpers and wait for them to park before clearing the flag.
            pool.launch(&cfg, board, &history);

            tt.bind_search_thread(0, cfg.threads);
            let mut ctx = Searcher::new(&cfg, &board, &history, tt);
            ctx.iterative_deepening(&mut history_table);

            cfg.stop.store(true, Ordering::Relaxed);
            smp::await_results(&cfg);

            // ── Thread Voting
            if !cfg.limits.silent && cfg.limits.perft.is_none() {
                let winner = smp::winner(&cfg);
                let result = ThreadResult::unpack(cfg.result_slots[winner].load(Ordering::Acquire));

                if winner != 0 {
                    ctx.report_voted(result.depth, result.score, result.mv);
                }

                println!("move {}", result.mv.to_uci(board.is_frc));
                let _ = io::stdout().flush();
            }

            pool.wait();
            cfg.stop.store(false, Ordering::Relaxed);
            *shared_history.lock() = history_table;
        }));
    }
}

fn handle_command<'a>(state: &mut XBoardState, cmd: &str, args: &mut impl Iterator<Item = &'a str>) {
    match cmd {
        "xboard" => { /* No-op, just protocol init */ },
        "protover" => {
            if let Some(ver) = args.next()
                && ver == "2"
            {
                print_features();
            }
        },

        "new" => {
            state.stop_search();
            state.load_position(Position::from_fen(STARTPOS));
            // SAFETY: stop_search joins the search thread, which returns only once
            // pool.wait() has parked the helpers, so nothing can probe the table.
            unsafe { state.tt.clear(state.threads) };
            state.mode = Mode::Normal;
            state.engine_side = Some(Color::Black);
            state.limits = Limits { protocol: Protocol::XBoard, ..Default::default() };
        },

        "force" => {
            state.stop_search();
            state.mode = Mode::Force;
        },

        "go" | "playother" => {
            state.stop_search();
            state.mode = Mode::Normal;
            state.engine_side = Some(state.board.stm);
            state.start_search();
        },

        "analyze" => {
            state.stop_search();
            state.mode = Mode::Analyze;
            state.limits.infinite = true;
            state.start_search();
        },

        "exit" => {
            state.stop_search();
            state.mode = Mode::Normal;
            state.limits.infinite = false;
        },

        "white" => {
            state.stop_search();
            state.engine_side = Some(Color::Black);
            if state.board.stm != Color::White {
                state.board.stm = Color::White;
                state.board.hash ^= key_side();
            }
        },

        "black" => {
            state.stop_search();
            state.engine_side = Some(Color::White);
            if state.board.stm != Color::Black {
                state.board.stm = Color::Black;
                state.board.hash ^= key_side();
            }
        },

        "level" => cmd_level(state, args),
        "st" => cmd_st(state, args),
        "sd" => cmd_sd(state, args),
        "nps" => cmd_nps(state, args),
        "time" => cmd_time(state, args),
        "otim" => cmd_otim(state, args),

        "ping" => {
            if let Some(n) = args.next() {
                println!("pong {n}");
            }
        },

        "setboard" => {
            let fen = args.collect::<Vec<_>>().join(" ");

            match Position::try_from_fen(&fen) {
                Ok(board) => {
                    state.stop_search();
                    state.load_position(board);
                    state.tt.begin_search();
                },
                Err(e) => eprintln!("Error (invalid fen): {e}"),
            }
        },

        "result" => {
            state.stop_search();
        },

        "option" => cmd_option(state, args),

        "memory" => {
            if let Some(arg) = args.next()
                && let Ok(mb) = arg.parse::<usize>()
            {
                state.hash_size = mb.clamp(1, 524288);
                (state.tt, state.smp_pool) = table_and_pool(state.hash_size, state.threads);
            }
        },

        "cores" => {
            if let Some(arg) = args.next()
                && let Ok(n) = arg.parse::<usize>()
            {
                let n = n.clamp(1, 1024);
                if n != state.threads {
                    state.threads = n;
                    if state.tt.spans_nodes() {
                        state.tt = Arc::new(TranspositionTable::new(state.hash_size, n));
                    }
                    state.smp_pool = LazySmpPool::new(n, state.tt.clone());
                }
            }
        },

        "usermove" => {
            if let Some(move_str) = args.next()
                && !cmd_move(state, move_str)
            {
                eprintln!("Illegal move: {move_str}");
            }
        },

        "accepted" | "rejected" => { /* Feature negotiation, no-op */ },
        _ => {
            if !cmd_move(state, cmd) {
                eprintln!("Error (unknown command): {cmd}");
            }
        },
    }
    io::stdout().flush().ok();
}

// XBoard protocol version 2. The stone age.
// We chisel our feature list into the GUI's cave wall.
// (Please put the pitchfork down, Lofty).
fn print_features() {
    let version = env!("CARGO_PKG_VERSION");
    println!("feature myname=\"Soul {version}\"");
    println!("feature ping=1");
    println!("feature setboard=1");
    println!("feature memory=1");
    println!("feature sigint=0");
    println!("feature sigterm=0");
    println!("feature colors=0");
    println!("feature debug=1");
    println!("feature smp=1");
    println!("feature nps=1");
    println!("feature variants=\"normal\"");
    println!("feature option=\"Hash -spin 16 1 524288\"");
    println!("feature option=\"Overhead -spin 10 0 2000\"");
    println!("feature option=\"ShowWDL -check 0\"");
    println!("feature done=1");
}

fn cmd_move(state: &mut XBoardState, move_str: &str) -> bool {
    let Ok(mv) = parse_uci_move(&state.board, move_str) else {
        return false;
    };

    state.stop_search();
    state.board.make_move(mv, &mut state.accumulator);
    state.history.push(state.board.hash);
    state.tt.begin_search();

    if state.board.is_threefold_repetition(&state.history) {
        println!("1/2-1/2 {{Draw by repetition}}");
    }

    let engine_turn = state.engine_side == Some(state.board.stm);
    if state.mode == Mode::Normal && engine_turn {
        state.start_search();
    }
    true
}

fn cmd_level<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    // level 40 5 0 (40 moves in 5 mins)

    let Some(moves_str) = args.next() else {
        return;
    };

    if let Ok(moves) = moves_str.parse::<u64>() {
        state.limits.movestogo = moves;
    }

    if let Some(base_str) = args.next() {
        // Can be 5 or 5:00
        let parts: Vec<&str> = base_str.split(':').collect();
        let mins = parts[0].parse::<u64>().unwrap_or(0);
        let secs = if parts.len() > 1 { parts[1].parse::<u64>().unwrap_or(0) } else { 0 };

        let total_ms = (mins * 60 + secs) * 1000;
        state.limits.wtime = total_ms;
        state.limits.btime = total_ms;
    }

    if let Some(inc) = args.next().and_then(|s| s.parse::<f64>().ok()) {
        let inc_ms = (inc * 1000.0) as u64;
        state.limits.winc = inc_ms;
        state.limits.binc = inc_ms;
    }
}

fn cmd_st<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(secs) = arg.parse::<u64>()
    {
        state.limits.movetime = secs * 1000;
    }
}

fn cmd_sd<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(depth) = arg.parse::<i32>()
    {
        state.limits.depth = depth;
    }
}

fn cmd_nps<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(nps) = arg.parse::<u64>()
    {
        state.nps = if nps > 0 { Some(nps) } else { None };
    }
}

/// `time` is the engine's own clock, in centiseconds. Which color that is comes
/// from `engine_side`, since the side to move only agrees with it on the engine's
/// turn, and falls back to the side to move before a color has been assigned.
fn cmd_time<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(cs) = arg.parse::<u64>()
    {
        let ms = cs * 10;

        if state.engine_side.unwrap_or(state.board.stm) == Color::White {
            state.limits.wtime = ms;
        } else {
            state.limits.btime = ms;
        }
    }
}

/// `otim` is the opponent's clock, so it lands on the color `cmd_time` doesn't.
fn cmd_otim<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(cs) = arg.parse::<u64>()
    {
        let ms = cs * 10;

        if state.engine_side.unwrap_or(state.board.stm) == Color::White {
            state.limits.btime = ms;
        } else {
            state.limits.wtime = ms;
        }
    }
}

fn cmd_option<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    let rest: Vec<&str> = args.collect();
    let full_arg = rest.join(" ");
    let parts: Vec<&str> = full_arg.split('=').collect();

    if parts.len() != 2 {
        return;
    }

    let name = parts[0].trim().to_lowercase();
    let value = parts[1].trim();

    match name.as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                state.hash_size = mb.clamp(1, 524288);
                (state.tt, state.smp_pool) = table_and_pool(state.hash_size, state.threads);
            }
        },
        "overhead" => {
            if let Ok(v) = value.parse::<u64>() {
                state.overhead = v.clamp(0, 2000);
            }
        },
        "showwdl" => {
            if let Ok(v) = value.parse::<u8>() {
                state.show_wdl = v != 0;
            }
        },
        _ => {},
    }
}
