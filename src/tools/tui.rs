//! Terminal UI formatting for search progress and board rendering.
//!
//! Provides ANSI-colored CLI output for interactive use, displaying Principal Variations (PV),
//! node counts, and static evaluation breakdown.

use std::{fmt::Write, io::Write as _};

use crate::{
    core::{
        board::Position,
        defs::{MATE, MATE_BOUND, PieceType, Protocol, Square},
        moves::Move,
    },
    engine::{search::Line, wdl},
};

type Rgb = (u8, u8, u8);

const GOLD_DIM: Rgb = (218, 165, 32); // branding
const GOLD_BRIGHT: Rgb = (255, 215, 0); // branding
const STEEL: Rgb = (176, 196, 222); // header info
const SLATE: Rgb = (119, 136, 153); // header dim
const TEAL: Rgb = (72, 209, 204); // nps accent

const DRAW_BLUE: Rgb = (120, 170, 220); // oklch(0.74 0.08 230)
const SLIGHT_UP: Rgb = (200, 180, 100); // oklch(0.76 0.12  85) gold
const WARM_UP: Rgb = (210, 185, 80); // oklch(0.78 0.15  80) golden yellow
const WIN_GREEN: Rgb = (90, 190, 120); // oklch(0.75 0.16 145)
const WIN_DEEP: Rgb = (50, 185, 135); // oklch(0.72 0.17 160) teal-green
const SLIGHT_DOWN: Rgb = (225, 160, 140); // oklch(0.76 0.10  30) warm peach
const WARM_DOWN: Rgb = (230, 145, 100); // oklch(0.74 0.14  35) orange
const LOSE_RED: Rgb = (220, 110, 95); // oklch(0.68 0.16  20) coral
const LOSE_DEEP: Rgb = (195, 85, 80); // oklch(0.62 0.17  15) brick
const MATE_PRPL: Rgb = (151, 125, 191); // mate

const BAR_WIN_LO: Rgb = (144, 238, 144); // WDL bar start
const BAR_WIN_HI: Rgb = (60, 179, 113); // WDL bar end
const BAR_DRAW_LO: Rgb = (240, 230, 140);
const BAR_DRAW_HI: Rgb = (189, 183, 107);
const BAR_LOSE_LO: Rgb = (255, 127, 80);
const BAR_LOSE_HI: Rgb = (205, 92, 92);

const PV_WHITE: Rgb = (224, 255, 255);
const PV_BLACK: Rgb = (160, 160, 160);
const DIM: Rgb = (130, 130, 130); // timestamps

const RESET: &str = "\x1b[0m";
const BOLD: &str = "\x1b[1m";

pub struct SearchInfoData<'a> {
    pub depth:     i32,
    pub sel_depth: i32,
    pub score:     i32,
    pub nodes:     u64,
    pub nps:       u64,
    pub time_ms:   u128,
    pub hashfull:  usize,
    pub pv:        &'a Line,
    pub show_wdl:  bool,
    pub material:  u32,
    pub stm:       usize,
    pub history:   &'a [(u128, Line, i32)],
    pub board:     &'a Position,
    pub use_ansi:  bool,
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
        let mate_in = if score > 0 {
            (MATE - score + 1) / 2
        } else {
            -(MATE + score + 1) / 2
        };
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
    let _ = std::io::stdout().flush();
}

/// Pretty TUI output for `GoPretty` mode with WDL bars and history.
pub fn print_pretty_search_info(data: &SearchInfoData<'_>) {
    if data.use_ansi {
        print!("\x1b[H");
    }

    let (wf, df, lf) = wdl::wdl_model(data.score, data.material);
    let w = (wf * 1000.0).round() as u32;
    let d = (df * 1000.0).round() as u32;
    let l = (lf * 1000.0).round() as u32;
    let t = data.time_ms.try_into().unwrap_or(u64::MAX);

    let reset = if data.use_ansi { RESET } else { "" };
    let bold = if data.use_ansi { BOLD } else { "" };

    println!(
        "  {bold}{}✦ Soul{reset}  {}d{}/{}{reset}  {}{}  {}{}  {}{}\x1b[K",
        tui_fg(GOLD_BRIGHT, data.use_ansi),
        tui_fg(STEEL, data.use_ansi),
        data.depth,
        data.sel_depth,
        tui_fg(SLATE, data.use_ansi),
        fmt_time(t),
        tui_fg(SLATE, data.use_ansi),
        fmt_nodes(data.nodes),
        tui_fg(TEAL, data.use_ansi),
        fmt_nps(data.nps),
    );

    print!("  ");
    for i in 0..48 {
        print!("{}━", tui_fg(lerp(GOLD_BRIGHT, TEAL, i as f32 / 48.0), data.use_ansi));
    }
    println!("{reset}\x1b[K\n");

    // Eval + WDL bars
    let wdl_heat = (w as f32 - l as f32 + 1000.0) / 2000.0; // 0=losing, 1=winning
    let heat_color = if wdl_heat >= 0.5 {
        lerp(BAR_DRAW_LO, BAR_WIN_LO, (wdl_heat - 0.5) * 2.0)
    } else {
        lerp(BAR_LOSE_LO, BAR_DRAW_LO, wdl_heat * 2.0)
    };

    let score_str = fmt_score_pretty(data.score, data.use_ansi);
    let eval_color = if data.score.abs() > MATE_BOUND {
        String::new()
    } else {
        format!("{}{bold}", tui_fg(heat_color, data.use_ansi))
    };
    println!(
        "  {bold}{}Eval{reset} {}{}{reset}\x1b[K",
        tui_fg(GOLD_DIM, data.use_ansi),
        eval_color,
        score_str
    );

    let bar_width = 50;
    let (wp, dp, lp) = (w as f32 / 10.0, d as f32 / 10.0, l as f32 / 10.0);
    println!(
        "  {bold}{}Win {reset}    [{}] {}{wp:>5.1}%{reset}",
        tui_fg(GOLD_DIM, data.use_ansi),
        bar(bar_width, wp / 100.0, BAR_WIN_LO, BAR_WIN_HI, data.use_ansi),
        tui_fg(BAR_WIN_HI, data.use_ansi)
    );
    println!(
        "  {bold}{}Draw{reset}    [{}] {}{dp:>5.1}%{reset}",
        tui_fg(GOLD_DIM, data.use_ansi),
        bar(bar_width, dp / 100.0, BAR_DRAW_LO, BAR_DRAW_HI, data.use_ansi),
        tui_fg(BAR_DRAW_HI, data.use_ansi)
    );
    println!(
        "  {bold}{}Lose{reset}    [{}] {}{lp:>5.1}%{reset}\n",
        tui_fg(GOLD_DIM, data.use_ansi),
        bar(bar_width, lp / 100.0, BAR_LOSE_LO, BAR_LOSE_HI, data.use_ansi),
        tui_fg(BAR_LOSE_HI, data.use_ansi)
    );

    // Best PV
    let white_first = data.stm == 0;
    println!("  {bold}{}Best PV{reset}", tui_fg(GOLD_DIM, data.use_ansi));
    print!("  ");
    print_pv_line(
        &data.pv.moves[..data.pv.len.min(10)],
        white_first,
        data.board.is_frc,
        data.use_ansi,
    );
    if data.pv.len > 10 {
        print!("{}...{}", tui_fg(DIM, data.use_ansi), reset);
    }
    println!("\x1b[K\n");

    // History
    println!("  {bold}{}History{reset}", tui_fg(GOLD_DIM, data.use_ansi));
    let start = data.history.len().saturating_sub(6);
    for (time, pv, _) in &data.history[start..] {
        let ts = fmt_time((*time).try_into().unwrap_or(u64::MAX));
        print!("  {}{:>8} -> ", tui_fg(DIM, data.use_ansi), ts);
        print_pv_line(&pv.moves[..pv.len.min(8)], white_first, data.board.is_frc, data.use_ansi);
        if pv.len > 8 {
            print!("{}...{}", tui_fg(DIM, data.use_ansi), reset);
        }
        println!("\x1b[K");
    }
    println!();

    if data.use_ansi {
        print!("\x1b[J"); // Clear to end of screen
    }
    let _ = std::io::stdout().flush();
}

// ──────── Private Helpers ────────

#[inline]
fn tui_fg(c: Rgb, enabled: bool) -> String {
    if enabled {
        format!("\x1b[38;2;{};{};{}m", c.0, c.1, c.2)
    } else {
        String::new()
    }
}

/// Lerp between two colors. t=0 → a, t=1 → b.
#[inline]
fn lerp(a: Rgb, b: Rgb, t: f32) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let mix = |x: u8, y: u8| (f32::from(y) - f32::from(x)).mul_add(t, f32::from(x)) as u8;
    (mix(a.0, b.0), mix(a.1, b.1), mix(a.2, b.2))
}

/// Maps centipawn score to [0, 1] win probability.
#[inline]
fn sigmoid(cp: i32) -> f64 {
    1.0 / (1.0 + (f64::from(-cp) / 150.0).exp())
}

/// Maps win probability [0, 1] to a gradient color.
/// This is the One True Gradient™ used everywhere scores are colored.
fn eval_gradient(win_prob: f64) -> Rgb {
    // Dead draw zone
    if (win_prob - 0.5).abs() < 0.02 {
        return DRAW_BLUE;
    }

    if win_prob > 0.5 {
        // Winning: gray → warm gold → green → deep green
        let t = ((win_prob - 0.5) * 2.0).min(1.0) as f32;
        match () {
            _ if t < 0.3 => lerp(SLIGHT_UP, WARM_UP, t / 0.3),
            _ if t < 0.7 => lerp(WARM_UP, WIN_GREEN, (t - 0.3) / 0.4),
            _ => lerp(WIN_GREEN, WIN_DEEP, (t - 0.7) / 0.3),
        }
    } else {
        // Losing: gray → warm pink → red → deep red
        let t = ((0.5 - win_prob) * 2.0).min(1.0) as f32;
        match () {
            _ if t < 0.3 => lerp(SLIGHT_DOWN, WARM_DOWN, t / 0.3),
            _ if t < 0.7 => lerp(WARM_DOWN, LOSE_RED, (t - 0.3) / 0.4),
            _ => lerp(LOSE_RED, LOSE_DEEP, (t - 0.7) / 0.3),
        }
    }
}

fn fmt_score_pretty(score: i32, enabled: bool) -> String {
    let reset = if enabled { RESET } else { "" };
    if score > MATE_BOUND {
        let moves = (MATE - score + 1) / 2;
        format!("{}M{}{}", tui_fg(MATE_PRPL, enabled), moves.max(1), reset)
    } else if score < -MATE_BOUND {
        let moves = (MATE + score + 1) / 2;
        format!("{}-M{}{}", tui_fg(MATE_PRPL, enabled), moves.max(1), reset)
    } else {
        let pawns = f64::from(score) / 100.0;
        if score >= 0 {
            format!("+{pawns:.2}")
        } else {
            format!("{pawns:.2}")
        }
    }
}

fn fmt_score_white_pov(score: i32, stm: usize, enabled: bool) -> String {
    let ws = if stm == 0 { score } else { -score };
    let reset = if enabled { RESET } else { "" };

    if ws.abs() > MATE_BOUND {
        let plies = if ws > 0 { MATE - ws } else { MATE + ws };
        let moves = (plies + 1) / 2;
        let s = if ws > 0 {
            format!("M{}", moves.max(1))
        } else {
            format!("-M{}", moves.max(1))
        };
        return format!("{}{:>7}{}", tui_fg(MATE_PRPL, enabled), s, reset);
    }

    let color = eval_gradient(sigmoid(ws));
    let pawns = f64::from(ws) / 100.0;
    let s = if ws >= 0 {
        format!("+{pawns:.2}")
    } else {
        format!("{pawns:.2}")
    };
    format!("{}{:>7}{}", tui_fg(color, enabled), s, reset)
}

fn bar(width: usize, fill: f32, lo: Rgb, hi: Rgb, enabled: bool) -> String {
    let fill = fill.clamp(0.0, 1.0);
    let filled = (fill * width as f32) as usize;
    let reset = if enabled { RESET } else { "" };

    let mut out = String::with_capacity(width * 20);
    for i in 0..filled {
        let t = i as f32 / width.max(1) as f32;
        write!(out, "{}#", tui_fg(lerp(lo, hi, t), enabled)).unwrap();
    }
    out.push_str(&tui_fg(DIM, enabled));
    for _ in filled..width {
        out.push('.');
    }
    out.push_str(reset);
    out
}

fn to_san(board: &mut Position, mv: Move, legal_moves: &[Move]) -> String {
    if mv.is_null() {
        return "0000".into();
    }

    let from = mv.from();
    let to = mv.to();
    let pt = board.piece_at(from);

    if pt == PieceType::King && mv.is_castling() {
        return if to > from {
            "O-O".into()
        } else {
            "O-O-O".into()
        };
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
        use crate::engine::movegen::gen_legal_moves;
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
            fmt_score_white_pov(data.score, data.stm, data.use_ansi),
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
    let mut temp_acc = if pretty {
        Some(data.board.get_initial_accumulator())
    } else {
        None
    };
    let white_first = data.stm == 0;

    for i in 0..data.pv.len {
        let mv = data.pv.moves[i];

        let s = match (temp_board.as_mut(), temp_acc.as_mut()) {
            (Some(b), Some(a)) => {
                use crate::engine::movegen::gen_legal_moves;
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

/// Print a PV line with alternating White/Black colors.
fn print_pv_line(moves: &[Move], white_first: bool, is_frc: bool, enabled: bool) {
    let reset = if enabled { RESET } else { "" };
    for (i, m) in moves.iter().enumerate() {
        let is_white = (i % 2 == 0) == white_first;
        let c = if is_white { PV_WHITE } else { PV_BLACK };
        print!("{}{}{reset} ", tui_fg(c, enabled), m.to_uci(is_frc));
    }
}
