//! Terminal UI formatting for search output: the pretty status strip, eval
//! and WDL bars, the PV in SAN, and an iteration-history sparkline.

use std::{fmt::Write, io, io::Write as _};

use crate::{
    color::{self, BOLD, GOLD, RESET, Rgb},
    core::{
        board::Position,
        defs::{Color, MATE, PieceType, Protocol, Square, is_mate},
        moves::Move,
        util::{format_duration, human},
    },
    engine::{
        movegen::gen_legal_moves,
        search::{Line, PvSnapshot},
        wdl,
    },
    weave::Vi16x8,
};

const GOLD_BRIGHT: Rgb = color::AMBER; // branding
const STEEL: Rgb = color::STEEL; // header info
const SLATE: Rgb = color::SLATE; // header dim
const TEAL: Rgb = color::TEAL; // nps accent

const MATE_PRPL: Rgb = color::MAUVE;

// Win green, draw the level blue of the advantage palette, loss red.
const WDL_EMPTY: Rgb = color::TRACK;
const WDL_FLOOR: Rgb = color::FLOOR;
const WIN_C: Rgb = color::JADE;
const LOSE_C: Rgb = color::CORAL;

// One warm, one cool, so you can see whose move each one is.
const PV_WHITE: Rgb = color::IVORY;
const PV_BLACK: Rgb = color::HAZE;
const DIM: Rgb = color::GREY; // timestamps, move numbers

const RULE_CELLS: usize = 48;
const WDL_CELLS: usize = 50;

/// Whether a reported score is the iteration's answer or only a bound,
/// from a search that left its aspiration window.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum ScoreBound {
    Exact,
    Lower,
    Upper,
}

pub struct SearchInfoData<'a> {
    pub bound: ScoreBound,
    pub depth: i32,
    pub sel_depth: i32,
    pub score: i32,
    pub nodes: u64,
    pub nps: u64,
    pub time_ms: u128,
    pub hashfull: usize,
    pub pv: &'a Line,
    pub show_wdl: bool,
    pub history: &'a [PvSnapshot],
    pub board: &'a Position,
    pub use_ansi: bool,
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
///
/// Every frame redraws from the top of the screen, so each line clears its own
/// tail and the last one clears the rest. With no terminal there is nothing to
/// redraw and the escapes would print as garbage.
pub fn print_pretty_search_info(data: &SearchInfoData<'_>) {
    let ansi = data.use_ansi;
    if ansi {
        print!("\x1b[H");
    }

    let reset = ansi_code(RESET, ansi);
    let bold = ansi_code(BOLD, ansi);
    let dim = tui_fg(DIM, ansi);
    let label = tui_fg(GOLD, ansi);
    let eol = ansi_code("\x1b[K", ansi);

    let t = data.time_ms.try_into().unwrap_or(u64::MAX);
    let dot = format!("{dim} · {reset}");

    #[rustfmt::skip]
    println!(
        "  {bold}{}✦ Soul{reset}{dot}{}{}/{}{reset}{dot}{}{}{reset}{dot}{}{}{reset}{dot}{}{}{reset}{dot}{}TT {}%{reset}{eol}",
        tui_fg(GOLD_BRIGHT, ansi),
        tui_fg(STEEL, ansi), data.depth, data.sel_depth,
        tui_fg(SLATE, ansi), format_duration(t),
        tui_fg(SLATE, ansi), human(data.nodes),
        tui_fg(TEAL, ansi), fmt_nps(data.nps),
        tui_fg(SLATE, ansi), data.hashfull / 10,
    );

    print!("  ");

    for i in 0..RULE_CELLS {
        print!("{}━", tui_fg(color::mix(GOLD_BRIGHT, TEAL, i as f64 / (RULE_CELLS - 1) as f64), ansi));
    }

    println!("{reset}{eol}\n");

    let (wf, df, lf) = wdl::wdl_model(data.score, data.board.material_count());
    let bound = bound_mark(data.bound, ansi);

    println!("  {bold}{label}Eval{reset}    {bound} {bold}{}{reset}{eol}", fmt_score_colored(data.score, 0, ansi));
    println!("{}", wdl_row("Win", (wf * 100.0) as f32, WIN_C, ansi));
    println!("{}", wdl_row("Draw", (df * 100.0) as f32, color::LEVEL, ansi));
    println!("{}\n", wdl_row("Lose", (lf * 100.0) as f32, LOSE_C, ansi));

    println!("  {bold}{label}Best PV{reset}");
    println!("  {}{eol}\n", fmt_pv(data.board, data.pv, 10, ansi));

    println!("  {bold}{label}History{reset}  {}{eol}", eval_sparkline(data.history, ansi));
    let start = data.history.len().saturating_sub(6);

    for (i, snap) in data.history[start..].iter().enumerate() {
        let ts = format_duration(snap.time_ms.try_into().unwrap_or(u64::MAX));
        let prev = (start + i).checked_sub(1).and_then(|p| data.history.get(p));

        let (arrow, arrow_hue) = match prev {
            Some(p) if snap.score - p.score > 5 => ('▲', WIN_C),
            Some(p) if snap.score - p.score < -5 => ('▼', LOSE_C),
            _ => ('·', DIM),
        };

        println!(
            "  {dim}d{:>2}{reset}  {dim}{ts:>7}{reset}  {}{:>6}{reset} {}{arrow}{reset}  {}{eol}",
            snap.depth,
            tui_fg(score_color(snap.score), ansi),
            fmt_score_num(snap.score),
            tui_fg(arrow_hue, ansi),
            fmt_pv(data.board, &snap.line, 8, ansi),
        );
    }

    println!();

    if ansi {
        print!("\x1b[J");
    }
    let _ = io::stdout().flush();
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

/// Eval trajectory as a colored block sparkline: one cell per retained
/// iteration, height by win probability, hue by the advantage gradient.
fn eval_sparkline(history: &[PvSnapshot], enabled: bool) -> String {
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
fn tui_fg(c: Rgb, enabled: bool) -> String {
    if enabled { color::ansi_fg(c) } else { String::new() }
}

#[inline]
fn ansi_code(code: &'static str, enabled: bool) -> &'static str {
    if enabled { code } else { "" }
}

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
        let pawns = f64::from(score) / 100.0;
        if score >= 0 { format!("+{pawns:.2}") } else { format!("{pawns:.2}") }
    }
}

/// Moves, not plies, to the mate a score encodes.
fn mate_moves(score: i32) -> i32 {
    (MATE - score.abs() + 1) / 2
}

/// Maps centipawn score to [0, 1] win probability.
#[inline]
fn sigmoid(cp: i32) -> f64 {
    wdl::sigmoid(f64::from(cp), 1.0 / 150.0)
}

fn score_color(score: i32) -> Rgb {
    if is_mate(score) {
        MATE_PRPL
    } else if score == 0 {
        color::LEVEL
    } else {
        color::advantage((sigmoid(score) - 0.5) * 2.0)
    }
}

/// The aspiration bound as one colored column, a space when the score is exact,
/// so the score sits in the same place either way.
fn bound_mark(bound: ScoreBound, enabled: bool) -> String {
    let glyph = match bound {
        ScoreBound::Exact => return " ".into(),
        ScoreBound::Lower => '≥',
        ScoreBound::Upper => '≤',
    };
    format!("{}{glyph}{}", tui_fg(SLATE, enabled), ansi_code(RESET, enabled))
}

/// Colored bare score, right-aligned in `width` columns; 0 for its natural width.
fn fmt_score_colored(score: i32, width: usize, enabled: bool) -> String {
    format!("{}{:>width$}{}", tui_fg(score_color(score), enabled), fmt_score_num(score), ansi_code(RESET, enabled))
}

/// One WDL row: a bar whose length and brightness both track `pct`, and the
/// percent lit the same way. The empty track sits below `WDL_FLOOR`, so a filled
/// cell always reads brighter than an unfilled one.
fn wdl_row(label: &str, pct: f32, hue: Rgb, enabled: bool) -> String {
    let reset = ansi_code(RESET, enabled);
    let bold = ansi_code(BOLD, enabled);
    let frac = f64::from(pct.clamp(0.0, 100.0)) / 100.0;
    let filled = (frac * WDL_CELLS as f64) as usize;

    let mut bars = String::with_capacity(WDL_CELLS * 20);
    for i in 0..filled {
        let t = i as f64 / (WDL_CELLS - 1) as f64;
        write!(bars, "{}#", tui_fg(color::mix(WDL_FLOOR, hue, t), enabled)).unwrap();
    }

    bars.push_str(&tui_fg(WDL_EMPTY, enabled));

    for _ in filled..WDL_CELLS {
        bars.push('.');
    }

    let pct_fg = tui_fg(color::mix(WDL_FLOOR, hue, frac), enabled);
    format!("  {bold}{}{label:<4}{reset}    [{bars}{reset}] {pct_fg}{pct:>5.1}%{reset}", tui_fg(GOLD, enabled))
}

/// One step of a PV replay: `mv` in SAN, leaving the board and accumulator on the
/// position after it. The check and mate marks need the move played, and `acc`
/// advances with it because unmake does not restore the accumulator.
fn san_step(board: &mut Position, acc: &mut Vi16x8, mv: Move) -> String {
    if mv.is_null() {
        return "0000".into();
    }

    let from = mv.from();
    let to = mv.to();
    let mut san = String::with_capacity(6);

    if mv.is_castling() {
        // Castling is encoded king-takes-rook, so the rook's file says which side.
        san.push_str(if to.file() > from.file() { "O-O" } else { "O-O-O" });
    } else {
        let pt = board.piece_at(from);

        if pt == PieceType::Pawn {
            if mv.is_capture() {
                san.push(sq_file(from));
            }
        } else {
            san.push(piece_ch(pt));

            // Two pieces of the same type reaching the same square have to be
            // told apart, by file, or by rank when they share a file, or by both.
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
            san.push(piece_ch(promo));
        }
    }

    board.make_move(mv, acc);

    if board.checkers().is_not_empty() {
        san.push(if gen_legal_moves(board).is_empty() { '#' } else { '+' });
    }
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

fn print_uci(data: &SearchInfoData<'_>, pretty: bool) {
    let wdl_str = if data.show_wdl {
        let (wf, df, lf) = wdl::wdl_model(data.score, data.board.material_count());
        let w = (wf * 1000.0).round() as u32;
        let d = (df * 1000.0).round() as u32;
        let l = (lf * 1000.0).round() as u32;
        format!(" wdl {w} {d} {l}")
    } else {
        String::new()
    };

    if pretty {
        let t = data.time_ms.try_into().unwrap_or(u64::MAX);
        let mark = bound_mark(data.bound, data.use_ansi);

        print!(
            "info depth {:>2} seldepth {:>2} score {mark}{}{} nodes {:>7} {:>11} time {:>9} hashfull {} pv",
            data.depth,
            data.sel_depth,
            fmt_score_colored(data.score, 7, data.use_ansi),
            wdl_str,
            human(data.nodes),
            fmt_nps(data.nps),
            format_duration(t),
            data.hashfull,
        );
    } else {
        let bound = match data.bound {
            ScoreBound::Exact => "",
            ScoreBound::Lower => " lowerbound",
            ScoreBound::Upper => " upperbound",
        };

        print!(
            "info depth {} seldepth {} score {}{}{} nodes {} nps {} time {} hashfull {} pv",
            data.depth,
            data.sel_depth,
            fmt_score_uci(data.score),
            bound,
            wdl_str,
            data.nodes,
            data.nps,
            data.time_ms,
            data.hashfull,
        );
    }

    // SAN needs the position each move is played from, so pretty output replays the line.
    let mut replay = pretty.then(|| (*data.board, data.board.get_initial_accumulator()));
    let white_first = data.board.stm == Color::White;
    let reset = ansi_code(RESET, data.use_ansi);

    for (i, &mv) in data.pv.moves[..data.pv.len].iter().enumerate() {
        match replay.as_mut() {
            Some((board, acc)) => {
                let hue = if (i % 2 == 0) == white_first { PV_WHITE } else { PV_BLACK };
                print!(" {}{}{reset}", tui_fg(hue, data.use_ansi), san_step(board, acc, mv));
            },
            None => print!(" {}", mv.to_uci(data.board.is_frc)),
        }
    }
    println!();
}

fn print_xboard(data: &SearchInfoData<'_>) {
    let cs = (data.time_ms + 5) / 10;
    print!("{:>2} {:>5} {:>6} {:>10} ", data.depth, data.score, cs, data.nodes);
    for mv in &data.pv.moves[..data.pv.len] {
        print!("{} ", mv.to_uci(data.board.is_frc));
    }
    println!();
}

/// Render a PV as numbered SAN, replaying from `root` so each move can be
/// disambiguated and check-marked. The count follows `root`'s side and fullmove
/// number, so a line that opens on Black reads `29… c5 30. Nf3`.
fn fmt_pv(root: &Position, line: &Line, max: usize, enabled: bool) -> String {
    let reset = ansi_code(RESET, enabled);
    let mut board = *root;
    let mut acc = board.get_initial_accumulator();
    let mut num = board.fullmove_number;
    let mut white_to_move = board.stm == Color::White;
    let mut out = String::with_capacity(max * 12);

    for (i, &mv) in line.moves[..line.len.min(max)].iter().enumerate() {
        if white_to_move {
            write!(out, "{}{num}.{reset} ", tui_fg(DIM, enabled)).unwrap();
        } else if i == 0 {
            write!(out, "{}{num}…{reset} ", tui_fg(DIM, enabled)).unwrap();
        }

        let san = san_step(&mut board, &mut acc, mv);
        let hue = if white_to_move { PV_WHITE } else { PV_BLACK };
        write!(out, "{}{san}{reset} ", tui_fg(hue, enabled)).unwrap();

        if !white_to_move {
            num += 1;
        }
        white_to_move = !white_to_move;
    }

    if line.len > max {
        write!(out, "{}…{reset}", tui_fg(DIM, enabled)).unwrap();
    }
    out
}
