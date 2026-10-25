//! Universal Chess Interface (UCI) protocol handling.

use std::{
    io::{self, Write},
    iter::Peekable,
    mem,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
        mpsc::{self, Sender},
    },
    thread,
    time::Instant,
};

use crate::{
    cli::Help,
    core::{
        board::{Position, STARTPOS},
        defs::{MAX_DEPTH, Protocol},
        zobrist::key_side,
    },
    engine::{
        eval::detailed_eval,
        history::History,
        search::{Limits, SearchConfig, SearchDisplay},
        search_params::{self, SearchParams},
        tm::{DEFAULT_OVERHEAD_MS, MAX_OVERHEAD_MS},
    },
    numa::NumaTopology,
    protocols::{
        notation::parse_uci_move,
        smp::{DEFAULT_HASH_MB, MAX_HASH_MB, MAX_THREADS, TableAndPool},
    },
    tools,
    weave::I16x8,
};

static LICENSE_NOTICE: &str = concat!(
    "Soul Chess Engine v",
    env!("SOUL_VERSION"),
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

#[allow(clippy::large_enum_variant)]
enum SearchCommand {
    Go { cfg: Box<SearchConfig>, board: Position, trail: Vec<u64>, history: History, search: TableAndPool },
    Quit,
}

struct UciState {
    board: Position,
    accumulator: I16x8,
    trail: Vec<u64>,
    persistent_history: History,
    search: TableAndPool,
    stop: Arc<AtomicBool>,
    search_tx: Sender<SearchCommand>,
    history_rx: mpsc::Receiver<History>,
    is_searching: Arc<AtomicBool>,
    overhead: u64,
    search_params: SearchParams,
    show_wdl: bool,
    go_pretty: bool,
    pretty_print: bool,
    show_currmove: bool,
    soft_nodes: bool,
    stdout_isatty: Option<bool>,
    frc_notation: bool,
}

impl UciState {
    fn new() -> Self {
        let board = Position::from_fen(STARTPOS);
        let trail = vec![board.hash];
        let stop = Arc::new(AtomicBool::new(false));
        let is_searching = Arc::new(AtomicBool::new(false));
        let (tx, rx) = mpsc::channel::<SearchCommand>();
        let (h_tx, h_rx) = mpsc::channel::<History>();
        let is_searching_worker = is_searching.clone();

        thread::spawn(move || {
            while let Ok(cmd) = rx.recv() {
                match cmd {
                    SearchCommand::Go { cfg, board, trail, mut history, search } => {
                        search.search(&cfg, board, &trail, &mut history);
                        let _ = h_tx.send(history);
                        is_searching_worker.store(false, Ordering::Relaxed);
                    },
                    SearchCommand::Quit => break,
                }
            }
        });

        Self {
            accumulator: board.psqt_accumulator(),
            board,
            trail,
            persistent_history: History::new(),
            search: TableAndPool::default(),
            stop,
            search_tx: tx,
            history_rx: h_rx,
            is_searching,
            overhead: DEFAULT_OVERHEAD_MS,
            search_params: SearchParams::default(),
            show_wdl: false,
            go_pretty: false,
            pretty_print: false,
            show_currmove: true,
            soft_nodes: false,
            stdout_isatty: None,
            frc_notation: false,
        }
    }

    fn absorb_history(&mut self) {
        while let Ok(table) = self.history_rx.try_recv() {
            self.persistent_history = table;
        }
    }

    fn stop_search(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        self.wait_for_search();
        self.stop.store(false, Ordering::Relaxed);
    }

    fn wait_for_search(&self) {
        while self.is_searching.load(Ordering::Relaxed) {
            thread::yield_now();
        }
    }

    fn load_position(&mut self, board: Position) {
        self.board = board;
        self.board.frc_notation = self.frc_notation;
        self.accumulator = self.board.psqt_accumulator();
        self.trail.clear();
        self.trail.push(self.board.hash);
    }

    fn search_display(&self) -> SearchDisplay {
        SearchDisplay {
            show_wdl: self.show_wdl,
            go_pretty: self.go_pretty,
            pretty_print: self.pretty_print,
            show_currmove: self.show_currmove,
            use_ansi: self.stdout_isatty.unwrap_or(true),
        }
    }
}

pub fn print_license() {
    println!("{LICENSE_NOTICE}");
}

pub fn main_loop(initial_command: Option<String>) {
    let mut state = UciState::new();
    let rx = spawn_stdin_listener(state.stop.clone());

    if let Some(cmd) = initial_command
        && !process_command(&mut state, cmd.trim())
    {
        return;
    }

    while let Ok(line) = rx.recv() {
        state.absorb_history();
        if !process_command(&mut state, line.trim()) {
            break;
        }
    }
}

pub fn run_commands(lines: &[String]) {
    let mut state = UciState::new();
    for line in lines {
        if !process_command(&mut state, line.trim()) {
            break;
        }

        // go returns while its search is still running, so the batch waits for it here.
        state.wait_for_search();
        state.absorb_history();
    }
}

pub fn run_cli_go(args: &[&str]) {
    let state = UciState::new();
    let mut iter = args.iter().copied().peekable();
    let limits = parse_go_limits(&state.board, &mut iter);
    dispatch_search(&state, limits);
    state.wait_for_search();
}

pub fn run_cli_pretty(depth: Option<i32>) {
    let mut state = UciState::new();
    state.show_wdl = true;
    state.go_pretty = true;
    dispatch_search(&state, Limits { depth: depth.unwrap_or(MAX_DEPTH), protocol: Protocol::Uci, ..Default::default() });
    state.wait_for_search();
}

fn dispatch_search(state: &UciState, limits: Limits) {
    let display = state.search_display();
    state.search.tt().set_age_factor(state.search_params.tt_age_factor);
    let mut cfg = SearchConfig::new_full(limits, Instant::now(), state.stop.clone(), state.overhead, display, state.search_params);
    cfg.set_threads(state.search.threads());
    state.is_searching.store(true, Ordering::Relaxed);

    state
        .search_tx
        .send(SearchCommand::Go {
            cfg: Box::new(cfg),
            board: state.board,
            trail: state.trail.clone(),
            history: state.persistent_history.clone(),
            search: state.search.clone(),
        })
        .unwrap();
}

fn parse_go_limits<'a, I>(board: &Position, tokens: &mut Peekable<I>) -> Limits
where I: Iterator<Item = &'a str> {
    let mut limits = Limits::default();

    // Match arms also consume from tokens.
    #[allow(clippy::while_let_on_iterator)]
    while let Some(token) = tokens.next() {
        match token {
            "wtime" => limits.wtime = parse_default(tokens, 0),
            "btime" => limits.btime = parse_default(tokens, 0),
            "winc" => limits.winc = parse_default(tokens, 0),
            "binc" => limits.binc = parse_default(tokens, 0),
            "movestogo" => limits.movestogo = parse_default(tokens, 0),
            "depth" => limits.depth = parse_default(tokens, 0),
            "softnodes" => limits.softnodes = parse_default(tokens, 0),
            "hardnodes" | "nodes" => limits.nodes = parse_default(tokens, 0),
            "movetime" => limits.movetime = parse_default(tokens, 0),
            "infinite" => limits.infinite = true,
            "mate" => limits.mate = Some(parse_default(tokens, 0)),
            "perft" => {
                limits.perft = Some(parse_default(tokens, 0));
                if matches!(tokens.peek(), Some(&t) if t.eq_ignore_ascii_case("bulk")) {
                    tokens.next();
                    limits.perft_bulk = true;
                }
            },
            "searchmoves" => {
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

pub fn print_help(use_ansi: bool) {
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
    h.subcommand("perft", "<N> [bulk]", "Perft test to depth N");
    h.subcommand("mate", "<N>", "Search for mate in N moves");
    h.subcommand("searchmoves", "", "Restrict search to specific moves");

    h.command("stop", "Stop current search");
    h.command("setoption", "Set engine option");
    h.command("quit", "Exit engine");
    h.separator();

    h.header("Options");
    h.command_default("Hash", &format!("Hash table size in MB (1-{MAX_HASH_MB})"), &DEFAULT_HASH_MB.to_string());
    h.command_default("Threads", &format!("Number of search threads (1-{MAX_THREADS})"), "1");
    h.command_default("Overhead", &format!("Move overhead in ms (0-{MAX_OVERHEAD_MS})"), &DEFAULT_OVERHEAD_MS.to_string());
    h.command_default("SoftNodes", "Make go nodes a per-iteration limit", "false");
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
    h.command_args("divide", "<N> [bulk]", "Perft divide test");
    #[cfg(feature = "rigs")]
    h.command("speedtest", "Run performance test");
    #[cfg(feature = "rigs")]
    h.command_args("atb", "<run|gen|export|validate|count|dump> ...", "Attack-table benchmark");
    h.command("spsa", "Print the tunable search params as an SPSA table; long tables in a terminal go to spsa.txt");
    h.command_args("spsa apply", "<REPORT> [--dry-run] [--params <PATH>]", "Write a finished tune's values into the defaults");
    #[cfg(feature = "datagen")]
    h.command("datagen", "Generate self-play training data");
    #[cfg(feature = "dataset")]
    h.command("dataset", "Manage datasets (inspect, info, scores)");
    h.command_args("genfens", "<N> seed <S> book <PATH|None>", "Print N opening FENs");
    h.command("gopretty", "Toggle pretty-print mode for search output");
    h.command("prettyprint", "Toggle pretty-print mode (alias: pp)");
    h.separator();

    h.header("Info");
    h.command("help", "Show this help message");
    h.command("license", "Show license information");
}

/// Reads commands while the main thread waits for search to finish.
/// Answers `isready` and signals `stop`/`quit` without waiting for search.
fn spawn_stdin_listener(stop: Arc<AtomicBool>) -> mpsc::Receiver<String> {
    let (tx, rx) = mpsc::channel();

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
                    continue;
                },
                "quit" => {
                    stop.store(true, Ordering::Relaxed);
                    let _ = tx.send(line);
                    break;
                },
                "stop" => {
                    stop.store(true, Ordering::Relaxed);
                },
                _ => {},
            }

            if tx.send(line).is_err() {
                break;
            }
        }
    });

    rx
}

fn process_command(state: &mut UciState, input: &str) -> bool {
    let mut tokens = input.split_whitespace().peekable();
    let Some(cmd) = tokens.next() else { return true };

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
            println!("info string numa: {}", NumaTopology::detect());
            println!("uciok");
        },

        "isready" => println!("readyok"),
        "ucinewgame" => {
            state.stop_search();
            state.absorb_history();
            state.persistent_history.clear();
            // SAFETY: stop_search waits for search() to return with all helpers parked.
            unsafe { state.search.tt().clear(state.search.threads()) };
            state.load_position(Position::from_fen(STARTPOS));
        },
        "position" => cmd_position(state, &mut tokens),
        "go" => cmd_go(state, &mut tokens),
        "stop" => state.stop_search(),

        "setoption" => cmd_setoption(state, &mut tokens),
        "license" => print_license(),
        "isatty" => cmd_isatty(state, &mut tokens),
        "d" | "display" => state.board.pretty_print(),
        "fen" => println!("{}", state.board.to_fen()),

        "eval" => {
            let res = detailed_eval(&state.board, &state.accumulator);
            println!("PSQT:     {:>5}", res.material_psqt);
            println!("Mobility: {:>5}", res.mobility);
            println!("Bonus:    {:>5}", res.bonus);
            println!("Safety:   {:>5}", res.safety);
            println!("───────────────");
            println!("Total:    {:>5}", res.total);
        },

        "bench" => tools::bench::run(parse_default(&mut tokens, tools::bench::DEFAULT_DEPTH), 16),
        "perft" | "divide" => {
            let depth = parse_default(&mut tokens, 5);
            let bulk = matches!(tokens.peek(), Some(&t) if t.eq_ignore_ascii_case("bulk"));
            tools::perft::run(&state.board, depth, cmd == "divide", bulk);
        },
        #[cfg(feature = "rigs")]
        "speedtest" => tools::speedtest::run(0),
        #[cfg(feature = "datagen")]
        "datagen" => {
            let _ = tools::datagen::run(&tokens.collect::<Vec<_>>(), &state.stop);
        },
        "genfens" => tools::genfens::run(&tokens.collect::<Vec<_>>()),

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
            state.board.hash ^= key_side();
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
    println!("id name Soul v{}", crate::VERSION);
    println!("id author Aethdv");
}

fn print_options() {
    println!("option name Hash type spin default {DEFAULT_HASH_MB} min 1 max {MAX_HASH_MB}");
    println!("option name Threads type spin default 1 min 1 max {MAX_THREADS}");
    println!("option name Overhead type spin default {DEFAULT_OVERHEAD_MS} min 0 max {MAX_OVERHEAD_MS}");
    println!("option name SoftNodes type check default false");
    println!("option name UCI_ShowWDL type check default false");
    println!("option name UCI_Chess960 type check default false");
    println!("option name UCI_ShowCurrMove type check default true");
    print!("{}", search_params::uci_options());
}

fn cmd_position<'a, I>(state: &mut UciState, tokens: &mut Peekable<I>)
where I: Iterator<Item = &'a str> {
    let board = match tokens.next() {
        Some("startpos") => Position::from_fen(STARTPOS),
        Some("fen") => match Position::try_from_tokens(&mut *tokens) {
            Ok(board) => board,
            Err(e) => {
                println!("info string warning: invalid fen: {e}");
                return;
            },
        },
        _ => return,
    };

    state.load_position(board);
    state.search.tt().begin_search();

    if tokens.next_if_eq(&"moves").is_some() {
        process_moves(state, tokens);
    }
}

fn process_moves<'a>(state: &mut UciState, moves: impl Iterator<Item = &'a str>) {
    for move_str in moves {
        match parse_uci_move(&state.board, move_str) {
            Ok(mv) => {
                state.board.make_move(mv, &mut state.accumulator);
                state.trail.push(state.board.hash);
            },
            Err(_) => {
                println!("info string warning: illegal move {move_str}");
                break;
            },
        }
    }
}

fn cmd_go<'a, I>(state: &mut UciState, tokens: &mut Peekable<I>)
where I: Iterator<Item = &'a str> {
    state.stop_search();
    let mut limits = parse_go_limits(&state.board, tokens);

    if state.soft_nodes && limits.nodes > 0 {
        limits.softnodes = mem::take(&mut limits.nodes);
    }

    dispatch_search(state, limits);
}

fn cmd_setoption<'a, I>(state: &mut UciState, tokens: &mut Peekable<I>)
where I: Iterator<Item = &'a str> {
    let mut name_parts = Vec::new();
    let mut value_parts = Vec::new();
    let mut parts: Option<&mut Vec<&str>> = None;

    for token in tokens {
        match token {
            "name" => parts = Some(&mut name_parts),
            "value" => parts = Some(&mut value_parts),
            _ => {
                if let Some(parts) = &mut parts {
                    parts.push(token);
                }
            },
        }
    }

    let name = name_parts.join(" ").to_lowercase();
    let value = value_parts.join(" ");

    match name.as_str() {
        "hash" => {
            if let Ok(mb) = value.parse::<usize>() {
                state.search.set_hash(mb);
            }
        },
        "overhead" => {
            if let Ok(v) = value.parse::<u64>() {
                state.overhead = v.clamp(0, MAX_OVERHEAD_MS);
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
        "softnodes" => state.soft_nodes = parse_bool(&value),
        "uci_chess960" => state.frc_notation = parse_bool(&value),
        "uci_showcurrmove" => {
            state.show_currmove = parse_bool(&value);
        },
        "prettyprint" => {
            state.pretty_print = parse_bool(&value);
        },
        "threads" => state.search.set_threads(value.parse().unwrap_or(1)),
        _ => {
            if let Ok(v) = value.parse::<i32>() {
                state.search_params.set(&name, v);
            }
        },
    }
}

/// Leaves an unparsable token untouched.
fn parse_default<'a, T, I>(tokens: &mut Peekable<I>, fallback: T) -> T
where
    T: FromStr,
    I: Iterator<Item = &'a str>,
{
    if let Some(token) = tokens.peek()
        && let Ok(val) = token.parse()
    {
        tokens.next();
        return val;
    }
    fallback
}

fn parse_bool(value: &str) -> bool { ["true", "t", "yes", "y", "1"].iter().any(|v| value.eq_ignore_ascii_case(v)) }

fn cmd_isatty<'a, I>(state: &mut UciState, tokens: &mut Peekable<I>)
where I: Iterator<Item = &'a str> {
    let (first, second) = (tokens.next(), tokens.next());
    let value = match first {
        Some(target) if target.eq_ignore_ascii_case("stdout") => second,
        _ => first,
    };

    if let Some(value) = value {
        state.stdout_isatty = Some(parse_bool(value));
    }
}
