//! XBoard / CECP protocol handling.

use std::{
    io::{self, BufRead, Write},
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Instant,
};

use crate::{
    core::{
        board::{Position, STARTPOS},
        defs::{Color, Protocol},
        zobrist::key_side,
    },
    engine::{
        history::History,
        search::{Limits, SearchConfig, SearchDisplay},
        search_params::SearchParams,
        tm::{DEFAULT_OVERHEAD_MS, MAX_OVERHEAD_MS},
    },
    protocols::{
        notation::parse_uci_move,
        smp::{DEFAULT_HASH_MB, MAX_HASH_MB, TableAndPool},
    },
    weave::I16x8,
};

#[derive(PartialEq, Eq, Debug, Clone, Copy)]
enum Mode {
    Normal,  // thinks on its turn
    Force,   // applies moves without searching
    Analyze, // searches infinitely
}

struct XBoardState {
    board: Position,
    accumulator: I16x8,
    trail: Vec<u64>,
    history: Option<Box<History>>,
    search: TableAndPool,
    mode: Mode,
    stop: Arc<AtomicBool>,
    search_thread: Option<JoinHandle<Box<History>>>,
    limits: Limits,
    overhead: u64,
    show_wdl: bool,
    engine_side: Option<Color>,
    frc_notation: bool,
    nps: Option<u64>,
}

pub fn main_loop(stdin: &mut impl BufRead) {
    let mut state = XBoardState::new();
    let mut raw = Vec::new();

    while let Some(line) = super::read_line(stdin, &mut raw) {
        let mut tokens = line.split_whitespace();
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
        let trail = vec![board.hash];

        Self {
            accumulator: board.psqt_accumulator(),
            board,
            trail,
            history: Some(Box::new(History::new())),
            search: TableAndPool::default(),
            mode: Mode::Normal,
            stop: Arc::new(AtomicBool::new(false)),
            search_thread: None,
            limits: Limits { protocol: Protocol::XBoard, ..Default::default() },
            overhead: DEFAULT_OVERHEAD_MS,
            show_wdl: false,
            engine_side: None,
            frc_notation: false,
            nps: None,
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);

        if let Some(handle) = self.search_thread.take() {
            match handle.join() {
                Ok(history) => self.history = Some(history),
                Err(err) => {
                    let message = err
                        .downcast_ref::<&str>()
                        .copied()
                        .or_else(|| err.downcast_ref::<String>().map(String::as_str))
                        .unwrap_or("unknown panic payload");
                    eprintln!("Error: Search thread panicked: {message}");
                },
            }
        }

        self.stop.store(false, Ordering::Relaxed);
    }

    fn load_position(&mut self, board: Position) {
        self.board = board;
        self.board.frc_notation = self.frc_notation;
        self.accumulator = self.board.psqt_accumulator();
        self.trail.clear();
        self.trail.push(self.board.hash);
    }

    fn start_search(&mut self) {
        self.stop_search();

        if self.mode == Mode::Force {
            return;
        }

        let mut limits = self.limits.clone();

        // nps turns the move time into a node limit.
        if let Some(nps) = self.nps
            && limits.movetime > 0
        {
            limits.nodes = nps * limits.movetime / 1000;
        }

        let board = self.board;
        let stop = self.stop.clone();
        let trail = self.trail.clone();
        let overhead = self.overhead;
        let show_wdl = self.show_wdl;
        let mut history = self.history.take().unwrap_or_else(|| Box::new(History::new()));
        let search = self.search.clone();

        self.search_thread = Some(thread::spawn(move || {
            let display = SearchDisplay { show_wdl, ..SearchDisplay::DEFAULT };
            let mut cfg = SearchConfig::new_full(limits, Instant::now(), stop, overhead, display, SearchParams::default());
            cfg.set_threads(search.threads());
            search.search(&cfg, board, &trail, &mut history);
            history
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
            if let Some(history) = &mut state.history {
                history.clear();
            }
            state.load_position(Position::from_fen(STARTPOS));
            // SAFETY: stop_search joins the search thread, which waits for all
            // helpers to finish before returning.
            unsafe { state.search.tt().clear(state.search.threads()) };
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
        "white" | "black" => {
            let stm = if cmd == "white" { Color::White } else { Color::Black };
            state.stop_search();
            state.engine_side = Some(!stm);
            if state.board.stm != stm {
                state.board.stm = stm;
                state.board.hash ^= key_side();

                // Repetition history must end at the current position's hash.
                if let Some(last) = state.trail.last_mut() {
                    *last = state.board.hash;
                }
            }
        },
        "level" => cmd_level(state, args),
        "st" => {
            if let Some(secs) = next_parsed::<u64>(args) {
                state.limits.movetime = secs * 1000;
            }
        },
        "sd" => {
            if let Some(depth) = next_parsed(args) {
                state.limits.depth = depth;
            }
        },
        "nps" => {
            if let Some(nps) = next_parsed::<u64>(args) {
                state.nps = (nps > 0).then_some(nps);
            }
        },
        "time" => cmd_clock(state, args, true),
        "otim" => cmd_clock(state, args, false),
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
                    state.search.tt().begin_search();
                },
                Err(e) => eprintln!("Error (invalid fen): {e}"),
            }
        },
        "result" => state.stop_search(),
        "?" => state.stop_search(),
        "option" => cmd_option(state, args),
        "memory" => {
            if let Some(mb) = next_parsed(args) {
                state.search.set_hash(mb);
            }
        },
        "cores" => {
            if let Some(n) = next_parsed(args) {
                state.search.set_threads(n);
            }
        },
        "usermove" => {
            if let Some(move_str) = args.next()
                && !cmd_move(state, move_str)
            {
                eprintln!("Illegal move: {move_str}");
            }
        },
        "accepted" | "rejected" | "hard" | "easy" => { /* No-op */ },
        _ => {
            if !cmd_move(state, cmd) {
                eprintln!("Error (unknown command): {cmd}");
            }
        },
    }
    io::stdout().flush().ok();
}

// XBoard protocol version 2. The stone age.
// We chisel our feature list into the controller's cave wall.
// (Please put the chainsaw down, Lofty).
fn print_features() {
    println!("feature myname=\"Soul v{}\"", crate::VERSION);
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
    println!("feature option=\"Hash -spin {DEFAULT_HASH_MB} 1 {MAX_HASH_MB}\"");
    println!("feature option=\"Overhead -spin {DEFAULT_OVERHEAD_MS} 0 {MAX_OVERHEAD_MS}\"");
    println!("feature option=\"ShowWDL -check 0\"");
    println!("feature done=1");
}

fn cmd_move(state: &mut XBoardState, move_str: &str) -> bool {
    let Ok(mv) = parse_uci_move(&state.board, move_str) else {
        return false;
    };

    state.stop_search();
    state.board.make_move(mv, &mut state.accumulator);
    state.trail.push(state.board.hash);
    state.search.tt().begin_search();

    if state.board.is_threefold_repetition(&state.trail) {
        println!("1/2-1/2 {{Draw by repetition}}");
    }

    let engine_turn = state.engine_side == Some(state.board.stm);
    if state.mode == Mode::Normal && engine_turn {
        state.start_search();
    }

    true
}

// level 40 5 0 (40 moves in 5 mins)
fn cmd_level<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    let Some(moves_str) = args.next() else { return };
    if let Ok(moves) = moves_str.parse::<u64>() {
        state.limits.movestogo = moves;
    }

    if let Some(base_str) = args.next() {
        let mut time = base_str.split(':').map(|s| s.parse::<u64>().unwrap_or(0));
        let mins = time.next().unwrap_or(0);
        let secs = time.next().unwrap_or(0);
        let total_ms = (mins * 60 + secs) * 1000;
        state.limits.wtime = total_ms;
        state.limits.btime = total_ms;
    }

    if let Some(inc) = next_parsed::<f64>(args) {
        let inc_ms = (inc * 1000.0) as u64;
        state.limits.winc = inc_ms;
        state.limits.binc = inc_ms;
    }
}

/// XBoard clock values are in centiseconds. `own_clock` selects the engine's or
/// opponent's clock. Until `engine_side` is known, assume the side to move is the engine.
fn cmd_clock<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>, own_clock: bool) {
    let Some(cs) = next_parsed::<u64>(args) else { return };
    let engine_is_white = state.engine_side.unwrap_or(state.board.stm) == Color::White;
    if engine_is_white == own_clock {
        state.limits.wtime = cs * 10;
    } else {
        state.limits.btime = cs * 10;
    }
}

fn next_parsed<'a, T: FromStr>(args: &mut impl Iterator<Item = &'a str>) -> Option<T> { args.next()?.parse().ok() }

fn cmd_option<'a>(state: &mut XBoardState, args: &mut impl Iterator<Item = &'a str>) {
    let option = args.collect::<Vec<_>>().join(" ");
    let mut parts = option.split('=');
    let (Some(name), Some(value), None) = (parts.next(), parts.next(), parts.next()) else { return };
    let name = name.trim().to_lowercase();
    let value = value.trim();

    match name.as_str() {
        "hash" => {
            if let Ok(mb) = value.parse() {
                state.search.set_hash(mb);
            }
        },
        "overhead" => {
            if let Ok(v) = value.parse::<u64>() {
                state.overhead = v.clamp(0, MAX_OVERHEAD_MS);
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
