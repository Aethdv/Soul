//! Universal Chess Interface (UCI) protocol implementation.
//!
//! Handles standard I/O communication, parsing GUI commands into internal engine state,
//! and formatting search results for the GUI.

use std::{
    io::{self, Write},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
    },
    thread,
    time::Instant,
};

use crate::{
    core::{
        board::{Position, STARTPOS},
        defs::{PieceType, Square},
        error::MoveError,
        moves::Move,
    },
    engine::{search::Limits, search_params::SearchParams},
    tools,
};

pub fn main_loop(initial_command: Option<String>) {
    let mut state = UciState::new();

    let rx = spawn_stdin_listener(state.stop.clone(), state.is_searching.clone());

    if let Some(cmd) = initial_command {
        process_command(&mut state, cmd.trim());
    }

    while let Ok(line) = rx.recv() {
        while let Ok(new_hist) = state.history_rx.try_recv() {
            state.persistent_history = new_hist;
        }
        if !process_command(&mut state, line.trim()) {
            break;
        }
    }
}

pub fn run_cli_go(args: &[String]) {
    let state = UciState::new();
    let board = state.board;
    let stop = state.stop.clone();
    let is_searching = state.is_searching.clone();

    // Use UciState defaults
    let overhead = state.overhead;
    let show_wdl = state.show_wdl;
    let go_pretty = state.go_pretty;
    let pretty_print = state.pretty_print;
    let show_currmove = state.show_currmove;

    let mut iter = args.iter().map(String::as_str).peekable();
    let limits = parse_go_limits(&board, &mut iter);
    let display = crate::engine::search::SearchDisplay {
        show_wdl,
        go_pretty,
        pretty_print,
        show_currmove,
        use_ansi: state.stdout_isatty.unwrap_or(true),
    };

    let cfg = crate::engine::search::SearchConfig::new_full(
        limits,
        Instant::now(),
        stop,
        overhead,
        display,
        crate::engine::search_params::SearchParams::default(),
    );

    let search_tx = state.search_tx.clone();
    is_searching.store(true, Ordering::Release);
    search_tx
        .send(SearchCommand::Go(
            Box::new(cfg),
            board,
            state.history.clone(),
            state.persistent_history,
            state.history_tx.clone(),
        ))
        .unwrap();

    while is_searching.load(Ordering::Acquire) {
        thread::yield_now();
    }
}

// ──────── Implementation ────────

use crate::engine::history;

#[allow(clippy::large_enum_variant)]
enum SearchCommand {
    Go(
        Box<crate::engine::search::SearchConfig>,
        Position,
        Vec<u64>,
        history::History,
        Sender<history::History>,
    ),
    Quit,
}

pub struct UciState {
    board:              Position,
    accumulator:        crate::weave::Vi16x8,
    history:            Vec<u64>,
    persistent_history: history::History,
    stop:               Arc<AtomicBool>,
    search_tx:          Sender<SearchCommand>,
    history_tx:         mpsc::Sender<history::History>,
    history_rx:         mpsc::Receiver<history::History>,
    is_searching:       Arc<AtomicBool>,
    hash_size:          usize,
    overhead:           u64,
    show_wdl:           bool,
    go_pretty:          bool,
    pretty_print:       bool,
    show_currmove:      bool,
    stdout_isatty:      Option<bool>,
    stderr_isatty:      Option<bool>,
    is_frc:             bool,
}

impl UciState {
    fn new() -> Self {
        let board = Position::from_fen(STARTPOS);
        let history = vec![board.hash];
        let stop = Arc::new(AtomicBool::new(false));
        let is_searching = Arc::new(AtomicBool::new(false));

        let (tx, rx) = mpsc::channel::<SearchCommand>();
        let (h_tx, h_rx) = mpsc::channel::<history::History>();

        let is_searching_worker = is_searching.clone();

        thread::spawn(move || {
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    SearchCommand::Go(cfg, board, history, history_table, result_tx) => {
                        let mut ctx =
                            crate::engine::search::Searcher::new(&cfg, &board, &history, history_table);
                        let final_history = ctx.iterative_deepening();
                        is_searching_worker.store(false, Ordering::Release);
                        let _ = result_tx.send(final_history);
                    },
                    SearchCommand::Quit => break,
                }
            }
        });

        Self {
            accumulator: board.get_initial_accumulator(),
            board,
            history,
            persistent_history: history::History::new(),
            stop,
            search_tx: tx,
            history_tx: h_tx,
            history_rx: h_rx,
            is_searching,
            hash_size: 16,
            overhead: 10,
            show_wdl: false,
            go_pretty: false,
            pretty_print: false,
            show_currmove: true,
            stdout_isatty: None,
            stderr_isatty: None,
            is_frc: false,
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Release);

        while self.is_searching.load(Ordering::Acquire) {
            thread::yield_now();
        }

        self.stop.store(false, Ordering::Release);
    }
}

/// Spawns a dedicated stdin listener thread that handles time critical commands
/// (`isready`, `stop`) immediately, even while the engine is searching.
/// Other commands are forwarded to the main loop.
///
/// This is required by UCI spec:
/// "the engine must always be able to process input from stdin, even while thinking."
fn spawn_stdin_listener(
    stop: Arc<AtomicBool>,
    is_searching: Arc<AtomicBool>,
) -> std::sync::mpsc::Receiver<String> {
    let (tx, rx) = std::sync::mpsc::channel();

    thread::spawn(move || {
        let stdin = io::stdin();
        loop {
            let mut line = String::new();
            match stdin.read_line(&mut line) {
                Ok(0) | Err(_) => break,
                Ok(_) => {},
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            match trimmed {
                "isready" => {
                    println!("readyok");
                    let _ = io::stdout().flush();
                },

                "quit" => {
                    stop.store(true, Ordering::Release);
                    let _ = tx.send(line);
                    break;
                },

                "stop" => {
                    stop.store(true, Ordering::Release);
                    if is_searching.load(Ordering::Acquire) {
                        let _ = tx.send(line);
                    }
                },
                _ => {
                    // Only forward commands when not searching.
                    // UCI spec says to silently ignore unexpected commands during search.
                    if !is_searching.load(Ordering::Acquire) && tx.send(line).is_err() {
                        break; // Main thread died
                    }
                },
            }
        }
    });
    rx
}

fn process_command(state: &mut UciState, input: &str) -> bool {
    if input.is_empty() {
        return true;
    }
    let mut tokens = input.split_whitespace().peekable();
    let cmd = tokens.next().unwrap();

    match cmd {
        "quit" => {
            state.stop_search();
            let _ = state.search_tx.send(SearchCommand::Quit);
            return false;
        },
        "uci" => {
            state.pretty_print = false;
            print_id();
            print_options();
            println!("uciok");
        },
        "isready" => println!("readyok"),
        "ucinewgame" => {
            state.stop_search();
            state.is_searching.store(false, Ordering::Release);
            state.history.clear();
            state.persistent_history.clear();
            state.board = Position::from_fen(STARTPOS);
            state.board.is_frc = state.is_frc;
            state.accumulator = state.board.get_initial_accumulator();
            state.history.push(state.board.hash);
        },
        "position" => cmd_position(state, &mut tokens),
        "go" => cmd_go(state, &mut tokens),
        "stop" => {
            state.stop_search();
            state.is_searching.store(false, Ordering::Release);
        },
        "setoption" => cmd_setoption(state, &mut tokens),
        "license" => print_license(),
        "isatty" => cmd_isatty(state, &mut tokens),
        "d" | "display" => state.board.pretty_print(),
        "fen" => println!("{}", state.board.as_fen()),
        "eval" => {
            let res = crate::engine::eval::detailed_eval(&state.board, &state.accumulator);
            println!("PSQT:     {:>5}", res.psqt);
            println!("Mobility: {:>5}", res.mobility);
            println!("Safety:   {:>5}", res.safety);
            println!("───────────────");
            println!("Total:    {:>5}", res.total);
        },
        "bench" => tools::bench::run(parse_val(&mut tokens)),
        "divide" => tools::perft::run(&state.board, parse_val(&mut tokens), true),
        "speedtest" => tools::speedtest::run(0),
        "genfens" => tools::genfens::run(&tokens.collect::<Vec<_>>(), &state.stop),
        "gopretty" => {
            state.go_pretty = !state.go_pretty;
            println!("GoPretty: {}", state.go_pretty);
        },
        "prettyprint" | "pp" => {
            state.pretty_print = !state.pretty_print;
            println!("PrettyPrint: {}", state.pretty_print);
        },
        "flip" => {
            state.board.stm = state.board.stm.opposite();
            state.board.hash ^= crate::core::zobrist::key_side();
            println!("Side to move: {:?}", state.board.stm);
        },
        "key" => println!("Zobrist: 0x{:016X}", state.board.hash),
        "help" => print_help(state.stdout_isatty.unwrap_or(true)),
        _ => {},
    }
    let _ = io::stdout().flush();
    true
}

fn print_id() {
    println!("id name Soul v{}", env!("CARGO_PKG_VERSION"));
    println!("id author Aethdv");
}

fn print_options() {
    println!("option name Hash type spin default 16 min 1 max 65536");
    println!("option name Threads type spin default 1 min 1 max 1");
    println!("option name Overhead type spin default 10 min 0 max 1000");
    println!("option name UCI_ShowWDL type check default false");
    println!("option name UCI_Chess960 type check default false");
    println!("option name UCI_ShowCurrMove type check default true");
}

fn cmd_position<'a, I>(state: &mut UciState, tokens: &mut std::iter::Peekable<I>)
where I: Iterator<Item = &'a str> {
    let Some(subcmd) = tokens.next() else {
        return;
    };

    if subcmd == "startpos" {
        state.board = Position::from_fen(STARTPOS);
        state.board.is_frc = state.is_frc;
        state.accumulator = state.board.get_initial_accumulator();
        state.history.clear();
        state.history.push(state.board.hash);

        if tokens.peek() == Some(&"moves") {
            tokens.next();
            process_moves(state, tokens);
        }
    } else if subcmd == "fen" {
        match Position::try_from_tokens(&mut *tokens) {
            Ok(board) => {
                state.board = board;
                state.board.is_frc = state.is_frc;
                state.accumulator = state.board.get_initial_accumulator();
                state.history.clear();
                state.history.push(state.board.hash);
            },
            Err(e) => {
                println!("info string warning: invalid fen: {e}");
                return;
            },
        }

        if tokens.peek() == Some(&"moves") {
            tokens.next();
            process_moves(state, tokens);
        }
    }
}

fn process_moves<'a, I>(state: &mut UciState, moves: &mut std::iter::Peekable<I>)
where I: Iterator<Item = &'a str> {
    use crate::engine::movegen::gen_legal_moves;

    for move_str in moves.by_ref() {
        let legal = gen_legal_moves(&state.board);
        let mv = legal
            .iter()
            .find(|mv| mv.to_uci(state.board.is_frc) == move_str);

        if let Some(&valid_move) = mv {
            state.board.make_move(valid_move, &mut state.accumulator);
            state.history.push(state.board.hash);
        } else {
            println!("info string warning: illegal move {move_str}");
            break;
        }
    }
}

fn cmd_go<'a, I>(state: &mut UciState, tokens: &mut std::iter::Peekable<I>)
where I: Iterator<Item = &'a str> {
    state.stop_search();

    let limits = parse_go_limits(&state.board, tokens);

    let board = state.board;
    let stop = state.stop.clone();
    let history = state.history.clone();
    let overhead = state.overhead;
    let show_wdl = state.show_wdl;
    let go_pretty = state.go_pretty;
    let pretty_print = state.pretty_print;
    let show_currmove = state.show_currmove;

    let start_time = Instant::now();

    use crate::engine::search::{SearchConfig, SearchDisplay};
    let display = SearchDisplay {
        show_wdl,
        go_pretty,
        pretty_print,
        show_currmove,
        use_ansi: state.stdout_isatty.unwrap_or(true),
    };

    let cfg = SearchConfig::new_full(limits, start_time, stop, overhead, display, SearchParams::default());

    state.is_searching.store(true, Ordering::Release);

    // persistent_history carries the move-ordering heuristic table across
    // positions within a game; it's reset by ucinewgame.
    state
        .search_tx
        .send(SearchCommand::Go(
            Box::new(cfg),
            board,
            history,
            state.persistent_history,
            state.history_tx.clone(),
        ))
        .unwrap();
}

pub fn parse_go_limits<'a, I>(board: &Position, tokens: &mut std::iter::Peekable<I>) -> Limits
where I: Iterator<Item = &'a str> {
    let mut limits = Limits::default();

    #[allow(clippy::while_let_on_iterator)]
    while let Some(token) = tokens.next() {
        match token {
            "wtime" => limits.wtime = parse_val(tokens),
            "btime" => limits.btime = parse_val(tokens),
            "winc" => limits.winc = parse_val(tokens),
            "binc" => limits.binc = parse_val(tokens),
            "movestogo" => limits.movestogo = parse_val(tokens),
            "depth" => limits.depth = parse_val(tokens),
            "softnodes" => limits.softnodes = parse_val(tokens),
            "hardnodes" | "nodes" => limits.nodes = parse_val(tokens),
            "movetime" => limits.movetime = parse_val(tokens),
            "infinite" => limits.infinite = true,
            "mate" => limits.mate = Some(parse_val(tokens)),
            "perft" => limits.perft = Some(parse_val(tokens)),
            "searchmoves" => {
                // Peek before consuming:
                // Tokens like depth must not be
                // swallowed when they fail to parse as a UCI move.
                while let Some(&mv_str) = tokens.peek() {
                    if let Ok(mv) = parse_uci_move(board, mv_str) {
                        limits.searchmoves.push(mv);
                        tokens.next();
                    } else {
                        break;
                    }
                }
            },
            _ => {},
        }
    }
    limits
}

fn cmd_setoption<'a, I>(state: &mut UciState, tokens: &mut std::iter::Peekable<I>)
where I: Iterator<Item = &'a str> {
    let mut name_parts = Vec::new();
    let mut value_parts = Vec::new();
    let mut reading_name = false;
    let mut reading_value = false;

    for token in tokens {
        if token == "name" {
            reading_name = true;
            reading_value = false;
            continue;
        }
        if token == "value" {
            reading_name = false;
            reading_value = true;
            continue;
        }

        if reading_name {
            name_parts.push(token);
        } else if reading_value {
            value_parts.push(token);
        }
    }

    let name = name_parts.join(" ").to_lowercase();
    let value = value_parts.join(" ");

    match name.as_str() {
        "hash" => {
            if let Ok(v) = value.parse() {
                state.hash_size = v;
                // TODO: we don't have a TT at all yet lol
                println!(
                    "info string note: Hash option accepted but transposition table is not yet implemented"
                );
            }
        },

        "overhead" => {
            if let Ok(v) = value.parse() {
                state.overhead = v;
            }
        },

        "uci_showwdl" => {
            state.show_wdl = parse_bool(&value);
            if state.show_wdl {
                println!(
                    "info string note: WDL output uses Stockfish coefficients on a different eval scale and will be inaccurate"
                );
            }
        },
        "uci_chess960" => state.is_frc = parse_bool(&value),
        "uci_showcurrmove" => {
            state.show_currmove = parse_bool(&value);
        },
        "prettyprint" => {
            state.pretty_print = parse_bool(&value);
        },
        "threads" => {
            // Dummy. not supported yet.
        },

        _ => {},
    }
}

fn parse_val<'a, T, I>(tokens: &mut std::iter::Peekable<I>) -> T
where
    T: std::str::FromStr + Default,
    I: Iterator<Item = &'a str>,
{
    if let Some(token) = tokens.peek()
        && let Ok(val) = token.parse()
    {
        tokens.next();
        return val;
    }
    T::default()
}

fn parse_bool(value: &str) -> bool {
    matches!(value.to_lowercase().as_str(), "true" | "t" | "yes" | "y" | "1")
}

fn parse_uci_move(board: &Position, uci: &str) -> Result<Move, MoveError> {
    use crate::engine::movegen::gen_legal_moves;

    if uci.len() < 4 {
        return Err(MoveError::InvalidFormat);
    }

    let from_sq = square_from_str(&uci[0..2])?;
    let to_sq = square_from_str(&uci[2..4])?;

    let promo = if uci.len() == 5 {
        Some(piece_from_char(uci.chars().nth(4).unwrap())?)
    } else {
        None
    };

    // Find matching legal move
    let legal = gen_legal_moves(board);
    legal
        .iter()
        .find(|&&mv| {
            if mv.from() == from_sq && mv.to() == to_sq && mv.promo() == promo {
                return true;
            }
            // ── Castling Normalization ──
            // Internal representation is King-onto-Rook (FRC), but incoming strings
            // may use standard King-to-destination notation (e.g., e1g1).
            // Delegate to to_uci to normalize both formats for reliable comparison.
            if mv.is_castling() && mv.to_uci(board.is_frc) == uci {
                return true;
            }
            // Fallback: If GUI sends standard castling (e1g1) but we are in FRC mode, still accept it.
            if mv.is_castling() && mv.from() == from_sq {
                let rank = from_sq.rank();
                let is_kingside = mv.to().file() > from_sq.file();
                let dest_file = if is_kingside { 6 } else { 2 }; // G or C
                if to_sq == crate::core::defs::Square::from_coords(dest_file, rank) {
                    return true;
                }
            }
            false
        })
        .copied()
        .ok_or(MoveError::NotFound)
}

fn square_from_str(s: &str) -> Result<Square, MoveError> {
    let file = s.as_bytes()[0].wrapping_sub(b'a');
    let rank = s.as_bytes()[1].wrapping_sub(b'1');

    if file > 7 || rank > 7 {
        return Err(MoveError::InvalidFormat);
    }

    Ok(Square::from_coords(file, rank))
}

fn piece_from_char(c: char) -> Result<PieceType, MoveError> {
    use crate::core::defs::PieceType;

    match c.to_ascii_lowercase() {
        'q' => Ok(PieceType::Queen),
        'r' => Ok(PieceType::Rook),
        'b' => Ok(PieceType::Bishop),
        'n' => Ok(PieceType::Knight),
        _ => Err(MoveError::InvalidFormat),
    }
}

pub fn print_help(use_ansi: bool) {
    use crate::cli::Help;

    let h = Help::new(22).with_ansi(use_ansi);

    h.header("UCI Protocol");
    h.command("uci", "Enter UCI mode, print engine info");
    h.command("isready", "Respond with 'readyok' when ready");
    h.command("ucinewgame", "Reset for a new game");
    h.command("position", "Set position (startpos/fen + moves)");
    h.command("go", "Start search");
    h.subcommand("depth", "<N>", "Search to depth N");
    h.subcommand("nodes", "<N>", "Search limit N nodes");
    h.subcommand("movetime", "<ms>", "Search for exact time");
    h.subcommand("infinite", "", "Search until stopped");
    h.subcommand("perft", "<N>", "Perft test to depth N");
    h.subcommand("mate", "<N>", "Search for mate in N moves");
    h.subcommand("searchmoves", "", "Restrict search to specific moves");

    h.command("stop", "Stop current search");
    h.command("setoption", "Set engine option");
    h.command("quit", "Exit engine");
    h.separator();

    h.header("Options");
    h.command_default("Hash", "Hash table size in MB (1-65536)", "16");
    h.command_default("Overhead", "Move overhead in ms (0-1000)", "10");
    h.command_default("UCI_ShowWDL", "Show win/draw/loss stats", "false");
    h.command_default("UCI_Chess960", "Enable Chess960/FRC mode", "false");
    h.command_default("UCI_ShowCurrMove", "Show current move being searched", "true");
    h.separator();

    h.header("Custom Commands");
    h.command("d/display", "Print board");
    h.command("fen", "Print current FEN");
    h.command("eval", "Print static evaluation");
    h.command("flip", "Switch side to move");
    h.command("key", "Print Zobrist hash");
    h.command_args("bench", "<N>", "Benchmark to depth N");
    h.command_args("divide", "<N>", "Perft divide test");
    h.command("speedtest", "Run performance test");
    h.command("dataset", "Manage datasets (inspect, info, encode)");
    h.command("genfens", "Run datagen");
    h.command("gopretty", "Toggle pretty-print mode for search output");
    h.command("prettyprint", "Toggle pretty-print mode (alias: pp)");
    h.separator();

    h.header("Info");
    h.command("help", "Show this help message");
    h.command("license", "Show license information");
}

pub fn print_license() {
    println!("{LICENSE_NOTICE}");
}

fn cmd_isatty<'a, I>(state: &mut UciState, tokens: &mut std::iter::Peekable<I>)
where I: Iterator<Item = &'a str> {
    let mut target = 0b11; // both stdout and stderr by default

    let arg1 = tokens.next();
    let arg2 = tokens.next();

    let mut bool_str = None;

    if let Some(tok) = arg1 {
        match tok.to_lowercase().as_str() {
            "stdout" => {
                target = 0b01;
                bool_str = arg2;
            },
            "stderr" => {
                target = 0b10;
                bool_str = arg2;
            },
            _val => {
                bool_str = Some(tok);
            },
        }
    }

    if let Some(tok) = bool_str {
        let val = matches!(tok.to_lowercase().as_str(), "true" | "t" | "yes" | "y" | "1");
        if target & 0b01 != 0 {
            state.stdout_isatty = Some(val);
        }
        if target & 0b10 != 0 {
            state.stderr_isatty = Some(val);
        }
    }
}

static LICENSE_NOTICE: &str = concat!(
    "Soul Chess Engine v",
    env!("CARGO_PKG_VERSION"),
    "\nCopyright (C) 2026 Aethdv\n\n",
    "This program is free software: you can redistribute it and/or modify\n",
    "it under the terms of the GNU Affero General Public License as published\n",
    "by the Free Software Foundation, either version 3 of the License, or\n",
    "(at your option) any later version.\n\n",
    "This program is distributed in the hope that it will be useful,\n",
    "but WITHOUT ANY WARRANTY; without even the implied warranty of\n",
    "MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the\n",
    "GNU Affero General Public License for more details.\n\n",
    "You should have received a copy of the GNU Affero General Public License\n",
    "along with this program. If not, see <https://www.gnu.org/licenses/>.\n\n",
    "Source code: <https://github.com/Aethdv/Soul>"
);
