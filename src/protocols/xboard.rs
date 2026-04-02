//! XBoard / CECP protocol implementation.
//!
//! Handles legacy I/O communication for compatibility with XBoard/WinBoard GUIs.

use std::{
    io::{self, StdinLock, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use crate::{
    core::{board::Position, defs::Protocol, error::EngineError},
    engine::{
        search::{Limits, Searcher},
        tt,
    },
};

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

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Mode {
    Normal,  // thinks on its turn
    Force,   // only verifies moves, never searches
    Analyze, // searches infinitely
}

struct XBoardState {
    board:              Position,
    accumulator:        crate::weave::Vi16x8,
    history:            Vec<u64>,
    persistent_history: Arc<parking_lot::Mutex<crate::engine::history::History>>,
    tt:                 Arc<tt::TranspositionTable>,
    mode:               Mode,
    stop_signal:        Arc<AtomicBool>,
    search_thread:      Option<JoinHandle<()>>,
    limits:             Limits,
    hash_size:          usize,
    overhead:           u64,
    show_wdl:           bool,
    engine_side:        Option<crate::core::defs::Color>,
    is_frc:             bool,
    nps:                Option<u64>,
}

impl XBoardState {
    fn new() -> Self {
        let board = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
        let history = vec![board.hash];

        Self {
            accumulator: board.get_initial_accumulator(),
            board,
            history,
            persistent_history: Arc::new(parking_lot::Mutex::new(crate::engine::history::History::new())),
            tt: Arc::new(tt::TranspositionTable::new(16)),
            mode: Mode::Normal,
            stop_signal: Arc::new(AtomicBool::new(false)),
            search_thread: None,
            limits: Limits {
                protocol: Protocol::XBoard,
                ..Default::default()
            },
            hash_size: 16,
            overhead: 10,
            show_wdl: false,
            engine_side: None,
            is_frc: false,
            nps: None,
        }
    }

    fn stop_search(&mut self) {
        self.stop_signal.store(true, Ordering::Release);
        if let Some(handle) = self.search_thread.take()
            && let Err(e) = handle.join()
        {
            eprintln!("Error: {}", EngineError::from_panic(e.as_ref()));
        }
        self.stop_signal.store(false, Ordering::Release);
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
        let history_arc = Arc::clone(&self.persistent_history);
        let persistent_history = *history_arc.lock();

        let tt = self.tt.clone();

        self.search_thread = Some(thread::spawn(move || {
            use crate::engine::{
                search::{SearchConfig, SearchDisplay},
                search_params::SearchParams,
            };
            let display = SearchDisplay {
                show_wdl,
                ..SearchDisplay::DEFAULT
            };
            let cfg = SearchConfig::new_full(
                limits,
                Instant::now(),
                stop,
                overhead,
                display,
                SearchParams::default(),
            );
            let mut ctx = Searcher::new(&cfg, &board, &history, persistent_history, tt);
            let final_history = ctx.iterative_deepening();
            *history_arc.lock() = final_history;
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
            state.board = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
            state.board.is_frc = state.is_frc;
            state.accumulator = state.board.get_initial_accumulator();
            state.history.clear();
            state.history.push(state.board.hash);
            state.tt.clear();
            state.mode = Mode::Normal;
            state.engine_side = Some(crate::core::defs::Color::Black);
            state.limits = Limits {
                protocol: Protocol::XBoard,
                ..Default::default()
            };
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
            state.engine_side = Some(crate::core::defs::Color::Black);
            if state.board.stm != crate::core::defs::Color::White {
                state.board.stm = crate::core::defs::Color::White;
                state.board.hash ^= crate::core::zobrist::key_side();
            }
        },
        "black" => {
            state.stop_search();
            state.engine_side = Some(crate::core::defs::Color::White);
            if state.board.stm != crate::core::defs::Color::Black {
                state.board.stm = crate::core::defs::Color::Black;
                state.board.hash ^= crate::core::zobrist::key_side();
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
            let rest: Vec<&str> = args.collect();
            let fen = rest.join(" ");
            state.stop_search();
            state.board = Position::from_fen(&fen);
            state.board.is_frc = state.is_frc;
            state.accumulator = state.board.get_initial_accumulator();
            state.history.clear();
            state.history.push(state.board.hash);
        },
        "result" => {
            state.stop_search();
        },
        "option" => cmd_option(state, args),
        "memory" => {
            if let Some(arg) = args.next()
                && let Ok(mb) = arg.parse::<usize>()
            {
                state.hash_size = mb;
                state.tt = Arc::new(tt::TranspositionTable::new(mb));
            }
        },
        "cores" => {
            // "Dummy" impl — TT not yet implemented, so thread count has no effect.
            if let Some(_arg) = args.next() {
                // state.threads = arg.parse() ...
            }
        },
        "usermove" => {
            if let Some(move_str) = args.next() {
                cmd_move(state, move_str);
            }
        },
        "accepted" | "rejected" => { /* Feature negotiation — no-op */ },
        _ => {
            cmd_move(state, cmd);
        },
    }
    io::stdout().flush().ok();
}

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
    println!("feature option=\"Hash -spin 16 1 65536\"");
    println!("feature option=\"Overhead -spin 10 0 1000\"");
    println!("feature option=\"ShowWDL -check 0\"");
    println!("feature done=1");
}

fn cmd_move(state: &mut XBoardState, move_str: &str) {
    let legal = crate::engine::movegen::gen_legal_moves(&state.board);

    if let Some(mv) = legal.iter().find(|mv| {
        mv.to_uci(state.board.is_frc) == move_str || mv.to_uci(state.board.is_frc) == move_str.to_lowercase()
    }) {
        state.stop_search();
        state.board.make_move(*mv, &mut state.accumulator);
        state.history.push(state.board.hash);

        if state.board.is_threefold_repetition(&state.history) {
            println!("1/2-1/2 {{Draw by repetition}}");
        }

        let engine_turn = state.engine_side == Some(state.board.stm);

        if state.mode == Mode::Normal && engine_turn {
            state.start_search();
        }
    } else {
        eprintln!("Error (unknown command): {move_str}");
    }
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
        let secs = if parts.len() > 1 {
            parts[1].parse::<u64>().unwrap_or(0)
        } else {
            0
        };

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

fn cmd_time<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    // Engine's time in centiseconds (1/100 sec)
    if let Some(arg) = args.next()
        && let Ok(cs) = arg.parse::<u64>()
    {
        let ms = cs * 10;
        if state.board.stm == crate::core::defs::Color::White {
            state.limits.wtime = ms;
        } else {
            state.limits.btime = ms;
        }
    }
}

fn cmd_otim<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    if let Some(arg) = args.next()
        && let Ok(cs) = arg.parse::<u64>()
    {
        let ms = cs * 10;
        if state.board.stm == crate::core::defs::Color::White {
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
                state.hash_size = mb;
                state.tt = Arc::new(tt::TranspositionTable::new(mb));
            }
        },
        "overhead" => {
            if let Ok(v) = value.parse() {
                state.overhead = v;
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
