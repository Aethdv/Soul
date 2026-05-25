//! Terminal UI formatting for search progress and board rendering.
//!
//! Provides ANSI-colored CLI output for interactive use, displaying Principal Variations (PV),
//! node counts, and static evaluation breakdown.

use std::{fmt::Write, io, io::Write as _};

use crate::{
    color::{self, Rgb},
    core::{
        board::Position,
        defs::{Color, MATE, MATE_BOUND, PieceType, Protocol, Square},
        moves::Move,
    },
    engine::{
        movegen::gen_legal_moves,
        search::{Line, PvSnapshot},
        wdl,
    },
};

const GOLD_DIM: Rgb = (218, 165, 32); // branding
const GOLD_BRIGHT: Rgb = (255, 215, 0); // branding
const STEEL: Rgb = (176, 196, 222); // header info
const SLATE: Rgb = (119, 136, 153); // header dim
const TEAL: Rgb = (72, 209, 204); // nps accent

const MATE_PRPL: Rgb = (151, 125, 191);

// WDL outcomes share the advantage palette's hues; win green, draw the level
// blue (`color::LEVEL`), loss red. Bar fill and percent ramp WDL_FLOOR→hue
// so intensity tracks probability; the empty track sits below the floor,
// so a filled cell always reads brighter than an unfilled one.
const WDL_EMPTY: Rgb = (58, 62, 72); // unfilled track
const WDL_FLOOR: Rgb = (112, 120, 134); // faintest fill / vanishing percent
const WIN_C: Rgb = (100, 200, 120);
const LOSE_C: Rgb = (224, 105, 100);

// PV moves split by temperature; White warm ivory (joins the gold identity),
// Black cool slate (joins the steel/slate telemetry), so a line alternates
// warm/cool as it alternates side.
const PV_WHITE: Rgb = (246, 238, 218);
const PV_BLACK: Rgb = (139, 154, 171);
const DIM: Rgb = (130, 130, 130); // timestamps, move numbers

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

pub struct SearchInfoData<'a> {
    pub depth: i32,
    pub sel_depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub nps: u64,
    pub time_ms: u128,
    pub hashfull: usize,
    pub pv: &'a Line,
    pub show_wdl: bool,
    pub material: u32,
    pub stm: usize,
    pub history: &'a [PvSnapshot],
    pub board: &'a Position,
    pub use_ansi: bool,
}

pub fn fmt_nodes(n: u64) -> String {
    match n {
        0..1_000 => format!("{n}"),
        1_000..1_000_000 => format!("{:.2}K", n as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.2}M", n as f64 / 1e6),
        _ => format!("{:.2}B", n as f64 / 1e9),
    }
}

pub fn fmt_time(ms: u64) -> String {
    match ms {
        0..1_000 => format!("{ms}ms"),
        1_000..60_000 => format!("{:.2}s", ms as f64 / 1e3),
        60_000..3_600_000 => {
            let m = ms / 60_000;
            let s = (ms % 60_000) as f64 / 1e3;
            format!("{m}m {s:.1}s")
        },
        _ => {
            let h = ms / 3_600_000;
            let m = (ms % 3_600_000) / 60_000;
            format!("{h}h {m}m")
        },
    }
}

pub fn fmt_nps(nps: u64) -> String {
    if nps < 1_000_000 {
        format!("{:.1} knps", nps as f64 / 1e3)
    } else {
        format!("{:.2} Mnps", nps as f64 / 1e6)
    }
}

pub fn fmt_score_uci(score: i32) -> String {
    if score.abs() > MATE_BOUND {
        let mate_in = if score > 0 { (MATE - score + 1) / 2 } else { -(MATE + score + 1) / 2 };
        format!("mate {mate_in}")
    } else {
        format!("cp {score}")
    }
}

pub fn print_search_info(protocol: Protocol, data: &SearchInfoData<'_>, pretty: bool) {
    match protocol {
        Protocol::Uci => print_uci(data, pretty),
        Protocol::XBoard => print_xboard(data),
    }
    let _ = io::stdout().flush();
}

/// Pretty TUI output for `GoPretty` mode; status strip, eval + WDL bars,
/// the principal variation in SAN, and a colored history of recent iterations.
pub fn print_pretty_search_info(data: &SearchInfoData<'_>) {
    let ansi = data.use_ansi;
    if ansi {
        print!("\x1b[H");
    }

    let reset = if ansi { RESET } else { "" };
    let bold = if ansi { BOLD } else { "" };
    let dim = tui_fg(DIM, ansi);
    let label = tui_fg(GOLD_DIM, ansi);

    // identity, depth/seldepth, clock, nodes, speed, TT fill.
    let t = data.time_ms.try_into().unwrap_or(u64::MAX);
    let dot = format!("{dim} · {reset}");

    #[rustfmt::skip]
    println!(
        "  {bold}{}✦ Soul{reset}{dot}{}{}/{}{reset}{dot}{}{}{reset}{dot}{}{}{reset}{dot}{}{}{reset}{dot}{}TT {}%{reset}\x1b[K",
        tui_fg(GOLD_BRIGHT, ansi),
        tui_fg(STEEL, ansi), data.depth, data.sel_depth,
        tui_fg(SLATE, ansi), fmt_time(t),
        tui_fg(SLATE, ansi), fmt_nodes(data.nodes),
        tui_fg(TEAL, ansi), fmt_nps(data.nps),
        tui_fg(SLATE, ansi), data.hashfull / 10,
    );

    print!("  ");
    for i in 0..48 {
        print!("{}━", tui_fg(color::mix(GOLD_BRIGHT, TEAL, f64::from(i) / 47.0), ansi));
    }
    println!("{reset}\x1b[K\n");

    // ── Eval + WDL ──
    // Labels share a 4-column gutter, so the eval value lines
    // up with the bars' left edge.
    let bar_width = 50;
    let (wf, df, lf) = wdl::wdl_model(data.score, data.material);
    println!("  {bold}{label}Eval{reset}    {bold}{}{reset}\x1b[K", fmt_score_pretty(data.score, ansi));
    println!("{}", wdl_row("Win", (wf * 100.0) as f32, WIN_C, bar_width, ansi));
    println!("{}", wdl_row("Draw", (df * 100.0) as f32, color::LEVEL, bar_width, ansi));
    println!("{}\n", wdl_row("Lose", (lf * 100.0) as f32, LOSE_C, bar_width, ansi));

    // ── Best PV ──
    // Numbered SAN, replayed from the root.
    println!("  {bold}{label}Best PV{reset}");
    print!("  {}", fmt_pv(data.board, &data.pv.moves[..data.pv.len.min(10)], ansi));
    if data.pv.len > 10 {
        print!("{dim}…{reset}");
    }
    println!("\x1b[K\n");

    // ── History ──
    // Eval trajectory as a sparkline, then the most recent
    // iterations: depth, clock, eval with a rise/fall arrow, and the line.
    println!("  {bold}{label}History{reset}  {}\x1b[K", eval_sparkline(data.history, ansi));
    let start = data.history.len().saturating_sub(6);
    for (i, snap) in data.history[start..].iter().enumerate() {
        let ts = fmt_time(snap.time_ms.try_into().unwrap_or(u64::MAX));
        let prev = (start + i).checked_sub(1).and_then(|p| data.history.get(p));
        let (arrow, arrow_c) = match prev {
            Some(p) if snap.score - p.score > 5 => ('▲', WIN_C),
            Some(p) if snap.score - p.score < -5 => ('▼', LOSE_C),
            _ => ('·', DIM),
        };
        print!(
            "  {dim}d{:>2}{reset}  {dim}{ts:>7}{reset}  {}{:>6}{reset} {}{arrow}{reset}  {}",
            snap.depth,
            tui_fg(score_color(snap.score), ansi),
            fmt_score_num(snap.score),
            tui_fg(arrow_c, ansi),
            fmt_pv(data.board, &snap.line.moves[..snap.line.len.min(8)], ansi),
        );
        if snap.line.len > 8 {
            print!("{dim}…{reset}");
        }
        println!("\x1b[K");
    }
    println!();

    if ansi {
        print!("\x1b[J"); // Clear to end of screen
    }
    let _ = io::stdout().flush();
}

/// Eval trajectory as a colored block sparkline — one cell per retained
/// iteration, height by win probability, hue by the advantage gradient.
fn eval_sparkline(history: &[PvSnapshot], enabled: bool) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
    let reset = if enabled { RESET } else { "" };
    let mut out = String::with_capacity(history.len() * 20);
    for snap in history {
        let level = ((sigmoid(snap.score) * 8.0) as usize).min(7);
        write!(out, "{}{}", tui_fg(score_color(snap.score), enabled), BLOCKS[level]).unwrap();
    }
    out.push_str(reset);
    out
}

#[inline]
fn tui_fg(c: Rgb, enabled: bool) -> String {
    if enabled { color::ansi_fg(c) } else { String::new() }
}

/// Maps centipawn score to [0, 1] win probability.
#[inline]
fn sigmoid(cp: i32) -> f64 {
    1.0 / (1.0 + (f64::from(-cp) / 150.0).exp())
}

/// Color for a centipawn score; purple for mate,
/// blue only at a dead-level `0.00`, advantage gradient otherwise.
fn score_color(score: i32) -> Rgb {
    if score.abs() > MATE_BOUND {
        MATE_PRPL
    } else if score == 0 {
        color::LEVEL
    } else {
        color::advantage((sigmoid(score) - 0.5) * 2.0)
    }
}

/// Bare score text; `M5` / `-M3` for mates, signed pawns (`+1.23`) otherwise.
fn fmt_score_num(score: i32) -> String {
    if score.abs() > MATE_BOUND {
        let moves = ((MATE - score.abs() + 1) / 2).max(1);
        if score > 0 { format!("M{moves}") } else { format!("-M{moves}") }
    } else {
        let pawns = f64::from(score) / 100.0;
        if score >= 0 { format!("+{pawns:.2}") } else { format!("{pawns:.2}") }
    }
}

fn fmt_score_pretty(score: i32, enabled: bool) -> String {
    let reset = if enabled { RESET } else { "" };
    format!("{}{}{}", tui_fg(score_color(score), enabled), fmt_score_num(score), reset)
}

fn fmt_score_colored(score: i32, enabled: bool) -> String {
    let reset = if enabled { RESET } else { "" };
    format!("{}{:>7}{}", tui_fg(score_color(score), enabled), fmt_score_num(score), reset)
}

/// One WDL row; a probability-keyed intensity bar plus a magnitude-lit percent.
/// Bar cells ramp `WDL_FLOOR`→`hue` by position, so length and brightness both
/// track `pct`; the percent ramps the same way by its own magnitude, settling
/// to the dim floor when the outcome is negligible.
fn wdl_row(label: &str, pct: f32, hue: Rgb, width: usize, enabled: bool) -> String {
    let reset = if enabled { RESET } else { "" };
    let bold = if enabled { BOLD } else { "" };
    let frac = f64::from(pct.clamp(0.0, 100.0)) / 100.0;
    let filled = (frac * width as f64) as usize;

    let mut bars = String::with_capacity(width * 20);
    for i in 0..filled {
        let t = i as f64 / width.saturating_sub(1).max(1) as f64;
        write!(bars, "{}#", tui_fg(color::mix(WDL_FLOOR, hue, t), enabled)).unwrap();
    }
    bars.push_str(&tui_fg(WDL_EMPTY, enabled));
    for _ in filled..width {
        bars.push('.');
    }

    let pct_fg = tui_fg(color::mix(WDL_FLOOR, hue, frac), enabled);
    format!("  {bold}{}{label:<4}{reset}    [{bars}{reset}] {pct_fg}{pct:>5.1}%{reset}", tui_fg(GOLD_DIM, enabled))
}

fn to_san(board: &mut Position, mv: Move, legal_moves: &[Move]) -> String {
    if mv.is_null() {
        return "0000".into();
    }

    let from = mv.from();
    let to = mv.to();
    let pt = board.piece_at(from);

    if pt == PieceType::King && mv.is_castling() {
        return if to > from { "O-O".into() } else { "O-O-O".into() };
    }

    let mut san = String::with_capacity(6);

    if pt == PieceType::Pawn {
        if board.piece_at(to) != PieceType::None || mv.is_en_passant() {
            san.push(sq_file(from));
        }
    } else {
        san.push(piece_ch(pt));

        let mut amb_file = false;
        let mut amb_rank = false;
        let mut needs = false;

        for &m in legal_moves {
            if m == mv || m.to() != to {
                continue;
            }
            if board.piece_at(m.from()) == pt {
                needs = true;
                if m.from().file() == from.file() {
                    amb_file = true;
                }
                if m.from().rank() == from.rank() {
                    amb_rank = true;
                }
            }
        }

        if needs {
            if !amb_file {
                san.push(sq_file(from));
            } else if !amb_rank {
                san.push(sq_rank(from));
            } else {
                san.push(sq_file(from));
                san.push(sq_rank(from));
            }
        }
    }

    if board.piece_at(to) != PieceType::None || mv.is_en_passant() {
        san.push('x');
    }

    san.push(sq_file(to));
    san.push(sq_rank(to));

    if let Some(ppt) = mv.promo() {
        san.push('=');
        san.push(piece_ch(ppt));
    }

    let mut acc = board.get_initial_accumulator();
    let undo = board.make_move(mv, &mut acc);
    if board.checkers().is_not_empty() {
        let responses = gen_legal_moves(board);
        san.push(if responses.is_empty() { '#' } else { '+' });
    }
    board.unmake_move(mv, &undo);
    san
}

#[inline]
fn sq_file(sq: Square) -> char {
    (b'a' + sq.file()) as char
}
#[inline]
fn sq_rank(sq: Square) -> char {
    (b'1' + sq.rank()) as char
}

const fn piece_ch(pt: PieceType) -> char {
    match pt {
        PieceType::Pawn => 'P',
        PieceType::Knight => 'N',
        PieceType::Bishop => 'B',
        PieceType::Rook => 'R',
        PieceType::Queen => 'Q',
        PieceType::King => 'K',
        PieceType::None => '?',
    }
}

fn print_uci(data: &SearchInfoData<'_>, pretty: bool) {
    let wdl_str = if data.show_wdl {
        let (wf, df, lf) = wdl::wdl_model(data.score, data.material);
        let w = (wf * 1000.0).round() as u32;
        let d = (df * 1000.0).round() as u32;
        let l = (lf * 1000.0).round() as u32;
        format!(" wdl {w} {d} {l}")
    } else {
        String::new()
    };

    if pretty {
        let t = data.time_ms.try_into().unwrap_or(u64::MAX);
        print!(
            "info depth {:>2} seldepth {:>2} score {}{} nodes {:>7} {:>11} time {:>9} hashfull {} pv",
            data.depth,
            data.sel_depth,
            fmt_score_colored(data.score, data.use_ansi),
            wdl_str,
            fmt_nodes(data.nodes),
            fmt_nps(data.nps),
            fmt_time(t),
            data.hashfull,
        );
    } else {
        print!(
            "info depth {} seldepth {} score {}{} nodes {} nps {} time {} hashfull {} pv",
            data.depth,
            data.sel_depth,
            fmt_score_uci(data.score),
            wdl_str,
            data.nodes,
            data.nps,
            data.time_ms,
            data.hashfull,
        );
    }

    let mut temp_board = if pretty { Some(*data.board) } else { None };
    let mut temp_acc = if pretty { Some(data.board.get_initial_accumulator()) } else { None };
    let white_first = data.stm == 0;

    for i in 0..data.pv.len {
        let mv = data.pv.moves[i];

        let s = match (temp_board.as_mut(), temp_acc.as_mut()) {
            (Some(b), Some(a)) => {
                let legal = gen_legal_moves(b);
                let san = to_san(b, mv, legal.as_slice());
                b.make_move(mv, a);
                san
            },
            _ => mv.to_uci(false),
        };

        if pretty {
            let is_white = (i % 2 == 0) == white_first;
            let color = if is_white { PV_WHITE } else { PV_BLACK };
            let reset = if data.use_ansi { RESET } else { "" };
            print!(" {}{}{}", tui_fg(color, data.use_ansi), s, reset);
        } else {
            print!(" {s}");
        }
    }
    println!();
}

fn print_xboard(data: &SearchInfoData<'_>) {
    let cs = (data.time_ms + 5) / 10;
    print!("{:>2} {:>5} {:>6} {:>10} ", data.depth, data.score, cs, data.nodes);
    for i in 0..data.pv.len {
        print!("{} ", data.pv.moves[i].to_uci(data.board.is_frc));
    }
    println!();
}

/// Render a PV as numbered SAN, replaying from `root` so each move can be
/// disambiguated and check-marked. Move numbers are dim, White's moves bright,
/// Black's muted; the count follows `root`'s side and fullmove so a line that
/// opens on Black reads `29… c5 30. Nf3`.
fn fmt_pv(root: &Position, moves: &[Move], enabled: bool) -> String {
    let reset = if enabled { RESET } else { "" };
    let mut board = *root;
    let mut acc = board.get_initial_accumulator();
    let mut num = board.fullmove_number;
    let mut white_to_move = board.stm == Color::White;

    let mut out = String::with_capacity(moves.len() * 12);
    for (i, &mv) in moves.iter().enumerate() {
        if white_to_move {
            write!(out, "{}{num}.{reset} ", tui_fg(DIM, enabled)).unwrap();
        } else if i == 0 {
            write!(out, "{}{num}…{reset} ", tui_fg(DIM, enabled)).unwrap();
        }

        let legal = gen_legal_moves(&board);
        let san = to_san(&mut board, mv, legal.as_slice());
        board.make_move(mv, &mut acc);

        let c = if white_to_move { PV_WHITE } else { PV_BLACK };
        write!(out, "{}{san}{reset} ", tui_fg(c, enabled)).unwrap();

        if !white_to_move {
            num += 1;
        }
        white_to_move = !white_to_move;
    }
    out
}
