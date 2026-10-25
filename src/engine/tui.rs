//! Terminal UI formatting for search output.

use std::{collections::VecDeque, fmt::Write, io, io::Write as _};

use crate::{
    color::{self, AMBER, BOLD, CLEAR_TO_EOL, GOLD, RESET, Rgb, SLATE, STEEL, TEAL},
    core::{
        board::Position,
        defs::{Color, MATE, PieceType, Protocol, Square, is_mate},
        moves::Move,
        util::{fmt_count, fmt_duration},
    },
    engine::{
        movegen::gen_legal_moves,
        search::{PvLine, PvSnapshot, SearchDisplay},
        wdl,
    },
    weave::I16x8,
};

const MATE_C: Rgb = color::MAUVE;

const WDL_TRACK: Rgb = color::TRACK;
const WDL_FLOOR: Rgb = color::FLOOR;

// Win green, loss red, dead draw blue.
const WIN_C: Rgb = color::JADE;
const LOSS_C: Rgb = color::CORAL;

const PV_WHITE: Rgb = color::IVORY;
const PV_BLACK: Rgb = color::HAZE;

const DIM: Rgb = color::GREY; // separators, timestamps, move numbers, unchanged scores

const RULE_CELLS: usize = 48;
const WDL_CELLS: usize = 50;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum ScoreBound {
    Exact,
    Lower,
    Upper,
}

pub(super) struct SearchInfoData<'a> {
    pub(super) bound: ScoreBound,
    pub(super) depth: i32,
    pub(super) sel_depth: i32,
    pub(super) score: i32,
    pub(super) nodes: u64,
    pub(super) nps: u64,
    pub(super) time_ms: u128,
    pub(super) hashfull: usize,
    pub(super) pv: &'a PvLine,
    pub(super) history: &'a VecDeque<PvSnapshot>,
    pub(super) board: &'a Position,
    pub(super) display: &'a SearchDisplay,
}

pub(super) fn print_search_info(protocol: Protocol, data: &SearchInfoData<'_>) {
    match protocol {
        Protocol::Uci => print_uci(data),
        Protocol::XBoard => print_xboard(data),
    }
    let _ = io::stdout().flush();
}

pub(super) fn print_pretty_search_info(data: &SearchInfoData<'_>) {
    let ansi = data.display.use_ansi;
    if ansi {
        print!("\x1b[H");
    }

    let reset = ansi_code(RESET, ansi);
    let bold = ansi_code(BOLD, ansi);
    let dim = tui_fg(DIM, ansi);
    let label = tui_fg(GOLD, ansi);
    let eol = ansi_code(CLEAR_TO_EOL, ansi);

    let brand = tui_fg(AMBER, ansi);
    let steel = tui_fg(STEEL, ansi);
    let slate = tui_fg(SLATE, ansi);
    let teal = tui_fg(TEAL, ansi);
    let dot = format!("{dim} · {reset}");

    let depth = data.depth;
    let sel = data.sel_depth;
    let time = fmt_duration(data.time_ms.try_into().unwrap_or(u64::MAX));
    let nodes = fmt_count(data.nodes);
    let nps = fmt_nps(data.nps);
    let hash = data.hashfull / 10;

    let mut status = String::with_capacity(256);
    write!(status, "  {bold}{brand}✦ Soul{reset}").unwrap();
    write!(status, "{dot}{steel}{depth}/{sel}{reset}").unwrap();
    write!(status, "{dot}{slate}{time}{reset}").unwrap();
    write!(status, "{dot}{slate}{nodes}{reset}").unwrap();
    write!(status, "{dot}{teal}{nps}{reset}").unwrap();
    write!(status, "{dot}{slate}TT {hash}%{reset}").unwrap();
    println!("{status}{eol}");

    print!("  ");

    for i in 0..RULE_CELLS {
        print!("{}━", tui_fg(color::mix(AMBER, TEAL, i as f64 / (RULE_CELLS - 1) as f64), ansi));
    }

    println!("{reset}{eol}\n");

    let (wf, df, lf) = wdl::wdl_model(data.score, data.board.material_count());
    let bound = bound_mark(data.bound, ansi);
    let score = fmt_score_colored(data.score, 0, ansi);

    println!("  {bold}{label}Eval{reset}    {bound} {bold}{score}{reset}{eol}");
    println!("{}", wdl_row("Win", (wf * 100.0) as f32, WIN_C, ansi));
    println!("{}", wdl_row("Draw", (df * 100.0) as f32, color::LEVEL, ansi));
    println!("{}\n", wdl_row("Loss", (lf * 100.0) as f32, LOSS_C, ansi));

    println!("  {bold}{label}Best PV{reset}");
    let pv = fmt_pv(data.board, data.pv, 10, ansi);
    println!("  {pv}{eol}\n");

    let sparkline = eval_sparkline(data.history, ansi);
    println!("  {bold}{label}History{reset}  {sparkline}{eol}");
    let start = data.history.len().saturating_sub(6);

    for (i, snap) in data.history.iter().skip(start).enumerate() {
        let depth = snap.depth;
        let ts = fmt_duration(snap.time_ms.try_into().unwrap_or(u64::MAX));
        let prev = (start + i).checked_sub(1).and_then(|p| data.history.get(p));

        let (arrow, arrow_color) = match prev {
            Some(p) if snap.score - p.score > 5 => ('▲', WIN_C),
            Some(p) if snap.score - p.score < -5 => ('▼', LOSS_C),
            _ => ('·', DIM),
        };

        let score_fg = tui_fg(score_color(snap.score), ansi);
        let score = fmt_score_num(snap.score);
        let arrow_fg = tui_fg(arrow_color, ansi);
        let pv = fmt_pv(data.board, &snap.line, 8, ansi);

        println!("  {dim}d{depth:>2}{reset}  {dim}{ts:>7}{reset}  {score_fg}{score:>6}{reset} {arrow_fg}{arrow}{reset}  {pv}{eol}");
    }

    println!();

    if ansi {
        print!("\x1b[J");
    }
    let _ = io::stdout().flush();
}

fn eval_sparkline(history: &VecDeque<PvSnapshot>, enabled: bool) -> String {
    const BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

    let reset = ansi_code(RESET, enabled);
    let mut out = String::with_capacity(history.len() * 20);

    for snap in history {
        let level = ((sigmoid(snap.score) * 8.0) as usize).min(7);
        write!(out, "{}{}", tui_fg(score_color(snap.score), enabled), BLOCKS[level]).unwrap();
    }
    out.push_str(reset);
    out
}

#[inline]
fn tui_fg(rgb: Rgb, enabled: bool) -> String { if enabled { color::ansi_fg(rgb) } else { String::new() } }

#[inline]
fn ansi_code(code: &'static str, enabled: bool) -> &'static str { if enabled { code } else { "" } }

fn fmt_nps(nps: u64) -> String {
    if nps < 1_000_000 {
        format!("{:.1} knps", nps as f64 / 1e3)
    } else {
        format!("{:.2} Mnps", nps as f64 / 1e6)
    }
}

fn fmt_score_uci(score: i32) -> String {
    if is_mate(score) {
        let moves = if score > 0 { mate_moves(score) } else { -mate_moves(score) };
        format!("mate {moves}")
    } else {
        format!("cp {score}")
    }
}

/// Bare score text; `M5` / `-M3` for mates, signed pawns (`+1.23`) otherwise.
fn fmt_score_num(score: i32) -> String {
    if is_mate(score) {
        let moves = mate_moves(score);
        if score > 0 { format!("M{moves}") } else { format!("-M{moves}") }
    } else {
        format!("{:+.2}", f64::from(score) / 100.0)
    }
}

fn mate_moves(score: i32) -> i32 { (MATE - score.abs() + 1) / 2 }

/// Logistic score mapping used by colors and the sparkline.
#[inline]
fn sigmoid(cp: i32) -> f64 { wdl::sigmoid(f64::from(cp), 1.0 / 150.0) }

fn score_color(score: i32) -> Rgb {
    if is_mate(score) {
        MATE_C
    } else if score == 0 {
        color::LEVEL
    } else {
        color::advantage((sigmoid(score) - 0.5) * 2.0)
    }
}

/// `≥` for a lower bound, `≤` for an upper bound, blank when the score is exact.
fn bound_mark(bound: ScoreBound, enabled: bool) -> String {
    let glyph = match bound {
        ScoreBound::Exact => return " ".into(),
        ScoreBound::Lower => '≥',
        ScoreBound::Upper => '≤',
    };
    format!("{}{glyph}{}", tui_fg(SLATE, enabled), ansi_code(RESET, enabled))
}

fn fmt_score_colored(score: i32, width: usize, enabled: bool) -> String {
    format!("{}{:>width$}{}", tui_fg(score_color(score), enabled), fmt_score_num(score), ansi_code(RESET, enabled))
}

fn wdl_row(label: &str, pct: f32, outcome: Rgb, enabled: bool) -> String {
    let mut bars = String::with_capacity(WDL_CELLS * 20);

    let reset = ansi_code(RESET, enabled);
    let bold = ansi_code(BOLD, enabled);
    let frac = f64::from(pct.clamp(0.0, 100.0)) / 100.0;
    let filled = (frac * WDL_CELLS as f64) as usize;
    for i in 0..filled {
        let t = i as f64 / (WDL_CELLS - 1) as f64;
        write!(bars, "{}#", tui_fg(color::mix(WDL_FLOOR, outcome, t), enabled)).unwrap();
    }

    bars.push_str(&tui_fg(WDL_TRACK, enabled));

    for _ in filled..WDL_CELLS {
        bars.push('.');
    }

    let pct_fg = tui_fg(color::mix(WDL_FLOOR, outcome, frac), enabled);
    format!("  {bold}{}{label:<4}{reset}    [{bars}{reset}] {pct_fg}{pct:>5.1}%{reset}", tui_fg(GOLD, enabled))
}

fn san_step(board: &mut Position, acc: &mut I16x8, mv: Move) -> String {
    if mv.is_null() {
        return "0000".into();
    }

    let from = mv.from();
    let to = mv.to();
    let mut san = String::with_capacity(6);

    if mv.is_castling() {
        san.push_str(if to.file() > from.file() { "O-O" } else { "O-O-O" });
    } else {
        let pt = board.piece_at(from);
        if pt == PieceType::Pawn {
            if mv.is_capture() {
                san.push(sq_file(from));
            }
        } else {
            san.push(pt.to_char(Color::White));

            let mut ambiguous = false;
            let mut shares_file = false;
            let mut shares_rank = false;

            let legal = gen_legal_moves(board);
            for &other in legal.iter() {
                if other == mv || other.to() != to || board.piece_at(other.from()) != pt {
                    continue;
                }

                ambiguous = true;
                shares_file |= other.from().file() == from.file();
                shares_rank |= other.from().rank() == from.rank();
            }

            if ambiguous {
                if !shares_file {
                    san.push(sq_file(from));
                } else if !shares_rank {
                    san.push(sq_rank(from));
                } else {
                    san.push(sq_file(from));
                    san.push(sq_rank(from));
                }
            }
        }

        if mv.is_capture() {
            san.push('x');
        }

        san.push(sq_file(to));
        san.push(sq_rank(to));

        if let Some(promo) = mv.promo() {
            san.push('=');
            san.push(promo.to_char(Color::White));
        }
    }

    board.make_move(mv, acc);

    if board.checkers().is_not_empty() {
        san.push(if gen_legal_moves(board).is_empty() { '#' } else { '+' });
    }
    san
}

#[inline]
fn sq_file(sq: Square) -> char { (b'a' + sq.file()) as char }

#[inline]
fn sq_rank(sq: Square) -> char { (b'1' + sq.rank()) as char }

fn print_uci(data: &SearchInfoData<'_>) {
    let ansi = data.display.use_ansi;
    let pretty = data.display.pretty_print;
    let wdl_str = if data.display.show_wdl {
        let (wf, _, lf) = wdl::wdl_model(data.score, data.board.material_count());
        let w = (wf * 1000.0).round() as u32;
        let l = (lf * 1000.0).round() as u32;
        let d = 1000u32.saturating_sub(w + l);
        format!(" wdl {w} {d} {l}")
    } else {
        String::new()
    };

    if pretty {
        let depth = data.depth;
        let sel = data.sel_depth;
        let mark = bound_mark(data.bound, ansi);
        let score = fmt_score_colored(data.score, 7, ansi);
        let nodes = fmt_count(data.nodes);
        let nps = fmt_nps(data.nps);
        let time = fmt_duration(data.time_ms.try_into().unwrap_or(u64::MAX));
        let hash = data.hashfull;

        print!(
            "info depth {depth:>2} seldepth {sel:>2} score {mark}{score}{wdl_str} nodes {nodes:>7} {nps:>11} time {time:>9} hashfull {hash} pv"
        );
    } else {
        let bound = match data.bound {
            ScoreBound::Exact => "",
            ScoreBound::Lower => " lowerbound",
            ScoreBound::Upper => " upperbound",
        };
        let depth = data.depth;
        let sel = data.sel_depth;
        let score = fmt_score_uci(data.score);
        let nodes = data.nodes;
        let nps = data.nps;
        let time = data.time_ms;
        let hash = data.hashfull;

        print!(
            "info depth {depth} seldepth {sel} score {score}{bound}{wdl_str} nodes {nodes} nps {nps} time {time} hashfull {hash} pv"
        );
    }

    let mut replay = pretty.then(|| (*data.board, data.board.psqt_accumulator()));

    let white_first = data.board.stm == Color::White;
    let reset = ansi_code(RESET, ansi);

    for (i, &mv) in data.pv.moves[..data.pv.len].iter().enumerate() {
        match replay.as_mut() {
            Some((board, acc)) => {
                let hue = if (i % 2 == 0) == white_first { PV_WHITE } else { PV_BLACK };
                print!(" {}{}{reset}", tui_fg(hue, ansi), san_step(board, acc, mv));
            },
            None => print!(" {}", mv.to_uci(data.board.frc_notation)),
        }
    }
    println!();
}

fn print_xboard(data: &SearchInfoData<'_>) {
    let cs = (data.time_ms + 5) / 10;
    print!("{:>2} {:>5} {:>6} {:>10} ", data.depth, data.score, cs, data.nodes);
    for mv in &data.pv.moves[..data.pv.len] {
        print!("{} ", mv.to_uci(data.board.frc_notation));
    }
    println!();
}

fn fmt_pv(root: &Position, line: &PvLine, max: usize, enabled: bool) -> String {
    let reset = ansi_code(RESET, enabled);
    let mut board = *root;
    let mut acc = board.psqt_accumulator();
    let mut fullmove = board.fullmove_number;
    let mut white_to_move = board.stm == Color::White;
    let mut out = String::with_capacity(max * 12);

    for (i, &mv) in line.moves[..line.len.min(max)].iter().enumerate() {
        if white_to_move {
            write!(out, "{}{fullmove}.{reset} ", tui_fg(DIM, enabled)).unwrap();
        } else if i == 0 {
            write!(out, "{}{fullmove}…{reset} ", tui_fg(DIM, enabled)).unwrap();
        }

        let san = san_step(&mut board, &mut acc, mv);
        let hue = if white_to_move { PV_WHITE } else { PV_BLACK };
        write!(out, "{}{san}{reset} ", tui_fg(hue, enabled)).unwrap();

        if !white_to_move {
            fullmove += 1;
        }
        white_to_move = !white_to_move;
    }

    if line.len > max {
        write!(out, "{}…{reset}", tui_fg(DIM, enabled)).unwrap();
    }
    out
}
