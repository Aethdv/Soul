//! Terminal pens, epoch progress, diagnostic tables, and the `eval_params.rs` paste block.
//!
//! Paste-block fidelity is guarded by `the_paste_block_reproduces_eval_params`.

use std::{
    cmp::Reverse,
    fmt::Write as _,
    fs::File,
    io::{self, BufWriter, Write},
};

use rayon::prelude::*;
use serde::Serialize;

pub(crate) use crate::engine::color::{CLEAR_TO_EOL, RESET, ansi_fg as fg};
use crate::{
    config::{EvalTuneConfig, KMode},
    engine::{BLOCKS, Group, PIECE_TABLES, TABLE_SQUARES, TOTAL_PHASE, Tunable, color, eval_record, pct},
    lion::GateTally,
    run::{Objective, artifact},
    training::{GROUP_NAMES, phase_of, piece_phase_weights, sigmoid},
};

pub(crate) const LAB: &str = color::TAUPE_PEN;
pub(crate) const VAL: &str = color::STEEL_PEN;
pub(crate) const COUNT: &str = color::STEEL_PEN;
pub(crate) const BRAND: &str = color::GOLD_PEN;
pub(crate) const DIM: &str = color::ASH_PEN;
pub(crate) const ALARM: &str = color::ALARM_PEN;
pub(crate) const MOVED: &str = color::JADE_PEN;

pub(crate) const PIECE_NAMES: [&str; PIECE_TABLES] = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];

const EIGHTH_BLOCKS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const SPARK_WINDOW: usize = 40;

// Offsets, widths, and order come from BLOCKS.
#[rustfmt::skip]
const ANNOTATIONS: &[(&str, &str)] = &[
    ("mobility_open",      "[mobility, battery, threats, xray threats]"),
    ("phase",              "[P, N, B, R, Q, K]"),
    ("king_safety",        "[Pawn Shield, Ortho Exp, Diag Exp]"),
    ("attacker",           "[0..5] attacked king-zone squares × weak"),
    ("xray",               "[Ortho King]"),
    ("king_danger",        "pressure curvature, over DANGER_SCALE; the floor at 0 holds the curvature where the data pulls negative"),
    ("tempo",              "[MG, EG], side-to-move initiative"),
    ("bishop_pair",        "[MG, EG]"),
    ("rook_open",          "[MG, EG]"),
    ("minor_behind_pawn",  "[MG, EG]"),
    ("doubled_pawn",       "[MG, EG]"),
    ("isolated_pawn",      "[MG, EG]"),
    ("backward_pawn",      "[MG, EG]"),
    ("phalanx_mg",         "by relative rank 2-7"),
    ("phalanx_eg",         "by relative rank 2-7"),
    ("defended_pawn_mg",   "by relative rank 2-7; rank 2 needs a defender on rank 1"),
    ("defended_pawn_eg",   "by relative rank 2-7; rank 2 needs a defender on rank 1"),
    ("passed_pawn_mg",     "by relative rank 2-7"),
    ("passed_pawn_eg",     "by relative rank 2-7"),
    ("enemy_king_dist_mg", "enemy king→passer dist, 7 clamps to 6"),
    ("enemy_king_dist_eg", "enemy king→passer dist, 7 clamps to 6"),
];

#[macro_export]
macro_rules! alarm {
    ($($arg:tt)*) => {
        eprintln!("{}[!] {}{}", $crate::report::ALARM, format_args!($($arg)*), $crate::report::RESET)
    };
}

pub(crate) struct BestSnapshots<'a> {
    pub(crate) best_val_params:   &'a [f64],
    /// `None` when the run had no holdout: nothing validated, so nothing ranks.
    pub(crate) best_val_loss:     Option<f64>,
    pub(crate) best_val_epoch:    usize,
    pub(crate) best_train_params: &'a [f64],
    pub(crate) best_train_loss:   f64,
    pub(crate) best_train_epoch:  usize,
    pub(crate) last_val:          Option<f64>,
    pub(crate) last_train:        Option<f64>,
}

/// Field order is the JSON record's. A run with no holdout leaves `val_loss` and `ref_loss`
/// at NaN, which serde writes as `null` and the status line shows as a dash.
#[derive(Serialize)]
pub(crate) struct EpochStats {
    pub(crate) epoch:         usize,
    pub(crate) train_loss:    f64,
    pub(crate) val_loss:      f64,
    pub(crate) ref_loss:      f64,
    pub(crate) lr:            f64,
    pub(crate) is_best:       bool,
    pub(crate) psqt_norm:     f64,
    pub(crate) mobility_norm: f64,
    pub(crate) overfit:       bool,
    pub(crate) drifted:       bool,
    pub(crate) moved:         usize,
    pub(crate) gauge:         f64,
    pub(crate) step_l1:       f64,
    pub(crate) clipped:       u64,
}

impl EpochStats {
    pub(crate) fn log_header() -> String {
        format!("{:>5}  {:>11}  {:>11}  {:>11}  {:>8}", "epoch", "L_train", "L_val", "L_ref", "LR")
    }

    pub(crate) fn log_row(&self) -> String {
        format!(
            "{:>5}  {:>11.6}  {:>11.6}  {:>11.6}  {:>8.4}{}",
            self.epoch,
            self.train_loss,
            self.val_loss,
            self.ref_loss,
            self.lr,
            if self.is_best { " *" } else { "" }
        )
    }

    /// Loss has no absolute scale, so the val cell is colored by its per-epoch trend (`prev_val` is NaN on epoch 1).
    pub(crate) fn status_line(&self, total_epochs: usize, prev_val: f64, elapsed: f32, mpos: f32) -> String {
        let Self { epoch, train_loss, val_loss, ref_loss, lr, moved, .. } = *self;
        let (arrow, trend) = if !val_loss.is_finite() || !prev_val.is_finite() || (val_loss - prev_val).abs() < 1e-7 {
            ('·', LAB.to_string())
        } else if val_loss < prev_val {
            ('▼', fg(color::advantage(0.7)))
        } else {
            ('▲', fg(color::advantage(-0.7)))
        };

        let (mark, epoch_c) = if self.is_best { ("✦ ", BRAND) } else { ("  ", DIM) };

        let warn = if self.overfit {
            format!("  {ALARM}⚠ overfit{RESET}")
        } else if self.drifted {
            format!("  {ALARM}⚠ drift{RESET}")
        } else {
            String::new()
        };

        let (val_cell, ref_cell) = if val_loss.is_finite() {
            (format!("{trend}{val_loss:.6}{RESET} {trend}{arrow}{RESET}"), format!("{DIM}{ref_loss:.6}{RESET}"))
        } else {
            (format!("{DIM}—{RESET}"), format!("{DIM}—{RESET}"))
        };

        format!(
            "{mark}{epoch_c}Epoch {epoch:>3}/{total_epochs}{RESET}  \
             {LAB}val{RESET} {val_cell}  {LAB}train{RESET} {DIM}{train_loss:.6}{RESET}  \
             {LAB}ref{RESET} {ref_cell}  {LAB}lr{RESET} {VAL}{lr:.4}{RESET}  {LAB}Δp{RESET} {DIM}{moved:>3}{RESET}  \
             {DIM}{elapsed:.2}s{RESET}  {DIM}{mpos:.1}M pos/s{RESET}{warn}{CLEAR_TO_EOL}"
        )
    }
}

pub(crate) fn fmt_loss(v: Option<f64>) -> String { v.map_or_else(|| "—".to_string(), |l| format!("{l:.6}")) }

pub(crate) fn print_results(
    params: &[Tunable],
    initial_values: &[f64],
    final_ema: &[f64],
    best: &BestSnapshots,
    final_epoch: usize,
) {
    let val_head = match best.best_val_loss {
        Some(l) => format!("Best L_val: {l:.6} (Epoch {})", best.best_val_epoch),
        None => "Best L_val: — (no holdout)".to_string(),
    };
    let train_head = format!("Best L_train: {:.6} (Epoch {})", best.best_train_loss, best.best_train_epoch);
    let final_head =
        format!("Final epoch {final_epoch}:  L_val {}  L_train {}", fmt_loss(best.last_val), fmt_loss(best.last_train));

    println!("\n{BRAND}{val_head}{RESET}\n{BRAND}{train_head}{RESET}\n\n{BRAND}{final_head}{RESET}");
    print_params(params, initial_values, final_ema);

    if let Ok(mut file) = File::create(artifact("evaltune_best.txt")) {
        let mut w = BufWriter::new(&mut file);
        writeln!(w, "{val_head}").ok();

        if best.best_val_loss.is_some() {
            write_params(&mut w, params, best.best_val_params, None);
        }

        writeln!(w, "\n{train_head}").ok();
        write_params(&mut w, params, best.best_train_params, None);
        writeln!(w, "\n{final_head}").ok();
        write_params(&mut w, params, final_ema, None);
    }
}

pub(crate) fn print_params(params: &[Tunable], initial: &[f64], values: &[f64]) {
    write_params(&mut io::stdout().lock(), params, values, Some(initial));
}

pub(crate) fn quantize(values: &[f64], out: &mut [i32]) {
    for (q, v) in out.iter_mut().zip(values) {
        *q = v.round() as i32;
    }
}

/// Tail EMA averages gauge-normalized iterates and lands fractionally under `1.0`,
/// so the warning threshold sits at `1%` rather than the gauge's `1e-6`.
pub(crate) fn off_scale_warning(shipped: f64) -> String {
    if (shipped - 1.0).abs() <= 0.01 {
        return String::new();
    }
    format!(
        "{ALARM}[!] Warning: the final-epoch parameters ship at {shipped:.3}× reference scale.\n\
         [!] `search_params` assumes centipawns; eval scale drift distorts all search pruning margins.{RESET}\n",
    )
}

pub(crate) fn clamped_k_warning(config: &EvalTuneConfig, k: f64) -> String {
    let margin = 0.01 * (config.k_max - config.k_min);
    if matches!(config.k_mode, KMode::Fixed { .. }) || (k - config.k_min >= margin && config.k_max - k >= margin) {
        return String::new();
    }
    format!(
        "{ALARM}[!] Warning: K = {k:.6} saturated against bracket boundary [{}, {}].\n\
         [!] Widen the bracket and rerun; this value reflects a clamp rather than an unconstrained optimum.{RESET}\n",
        config.k_min, config.k_max,
    )
}

/// Predicted against realized win rate on the validation split, by game phase and signed eval.
pub(crate) fn calibration_report(obj: &Objective, values: &[f64], k: f64) -> String {
    const BAND_WIDTH: usize = 4;
    const BANDS: usize = TOTAL_PHASE as usize / BAND_WIDTH;
    const EDGES: [f64; 2] = [-50.0, 50.0];
    const CELLS: usize = EDGES.len() + 1;

    #[derive(Clone, Copy, Default)]
    struct CalibrationCell {
        n:    u64,
        pred: f64,
        real: f64,
    }

    let phase_w = piece_phase_weights();
    let grid = obj
        .val
        .par_iter()
        .fold(
            || [CalibrationCell::default(); BANDS * CELLS],
            |mut acc, record| {
                if obj.passes_volatility_filter(record) {
                    let b = (phase_of(record, &phase_w) / BAND_WIDTH).min(BANDS - 1);
                    let eval = eval_record(record, values);
                    let c = &mut acc[b * CELLS + EDGES.iter().filter(|&&e| eval >= e).count()];
                    c.n += 1;
                    c.pred += sigmoid(eval, k);
                    c.real += f64::from(record.result) / 2.0;
                }
                acc
            },
        )
        .reduce(
            || [CalibrationCell::default(); BANDS * CELLS],
            |mut a, b| {
                for (x, y) in a.iter_mut().zip(b) {
                    x.n += y.n;
                    x.pred += y.pred;
                    x.real += y.real;
                }
                a
            },
        );

    let band_label = |b: usize| {
        let lo = b * BAND_WIDTH;
        format!("{lo}-{}", if b + 1 == BANDS { TOTAL_PHASE as usize } else { lo + BAND_WIDTH - 1 })
    };
    let rate = |sum: f64, n: u64| 100.0 * sum / n as f64;

    let mut out = String::new();
    let _ = writeln!(out, "\n{LAB}Calibration{RESET} {DIM}(best-val parameters, validation split at K = {k:.6}){RESET}");
    let _ = writeln!(out, "  {LAB}phase           n   predicted   realized  residual{RESET}");

    for b in 0..BANDS {
        let slice = &grid[b * CELLS..(b + 1) * CELLS];
        let n: u64 = slice.iter().map(|c| c.n).sum();

        if n == 0 {
            continue;
        }

        let p = rate(slice.iter().map(|c| c.pred).sum(), n);
        let r = rate(slice.iter().map(|c| c.real).sum(), n);
        let _ = writeln!(
            out,
            "  {:<7} {VAL}{n:>9}{RESET}      {VAL}{p:5.1}%{RESET}     {VAL}{r:5.1}%{RESET}     {VAL}{:+5.1}{RESET}",
            band_label(b),
            p - r
        );
    }

    let _ = writeln!(out, "\n{LAB}Residual by eval within phase{RESET} {DIM}(cell counts in parentheses){RESET}");
    let _ = writeln!(out, "  {LAB}{:<7} {:<13} {:<13} eval > +50{RESET}", "phase", "eval < -50", "-50..+50");
    for b in 0..BANDS {
        let row: [String; CELLS] = std::array::from_fn(|c| {
            let cell = grid[b * CELLS + c];
            if cell.n == 0 {
                "-".to_string()
            } else {
                format!("{:+.1} ({})", rate(cell.pred, cell.n) - rate(cell.real, cell.n), compact(cell.n))
            }
        });
        if row.iter().any(|cell| cell != "-") {
            let _ = writeln!(out, "  {:<7} {:<13} {:<13} {}", band_label(b), row[0], row[1], row[2]);
        }
    }
    out
}

/// `band` is where `m·g` gating and Liang's `c·g` cautious mask disagree.
pub(crate) fn gate_tally_report(groups: &[GateTally]) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "\n{LAB}Gate tally{RESET} {DIM}(share of parameter-updates){RESET}");
    let _ = writeln!(out, "  {LAB}group          φ     skip  canonical    band   c-only   waived     dead  no grad{RESET}");

    for (name, c) in GROUP_NAMES.iter().zip(groups) {
        let _ = writeln!(out, "  {name:<9} {}", gate_columns(c));
    }

    out
}

pub(crate) fn gate_columns(c: &GateTally) -> String {
    format!(
        "{VAL}{:6.4}{RESET}   {VAL}{:5.1}%{RESET}     {VAL}{:5.1}%{RESET}  {VAL}{:5.2}%{RESET}   \
         {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}",
        c.active_share(),
        c.percent(c.skipped),
        c.percent(c.canonical),
        c.percent(c.band),
        c.percent(c.canonical_only),
        c.percent(c.epsilon_waived),
        c.percent(c.dead),
        c.percent(c.absent),
    )
}

pub(crate) fn print_dataset_stats<T, F: Fn(&T) -> f64>(train: &[T], val: &[T], total: usize, result_fn: F) {
    println!("{LAB}Positions:{RESET}  {COUNT}{total}{RESET} ({} train / {} val)", train.len(), val.len());
    let (ww, bw, dr) = train.iter().fold((0, 0, 0), |(w, b, d), entry| {
        let r = result_fn(entry);
        if (r - 1.0).abs() < 1e-4 {
            (w + 1, b, d)
        } else if r.abs() < 1e-4 {
            (w, b + 1, d)
        } else {
            (w, b, d + 1)
        }
    });
    println!(
        "  {LAB}White wins:{RESET} {COUNT}{ww}{RESET}\n  {LAB}Black wins:{RESET} {COUNT}{bw}{RESET}\n  {LAB}Draws:{RESET}      {COUNT}{dr}{RESET}"
    );
    // A datagen run that never filled the result field looks identical to a set of drawn games.
    if ww + bw == 0 {
        alarm!(
            "Warning: no decisive outcomes detected. Training targets are constant 0.5; \
             pure outcome training will diverge or stall."
        );
    }
}

pub(crate) fn print_loss_history(val: &[f64], train: &[f64]) {
    if !val.is_empty() {
        println!("\n  {LAB}L_val{RESET}    {}", loss_sparkline(&val[val.len().saturating_sub(SPARK_WINDOW)..]));
    }
    println!("\n  {LAB}L_train{RESET}  {}", loss_sparkline(&train[train.len().saturating_sub(SPARK_WINDOW)..]));
}

/// Momentum keeps accumulating outward while only the value is clamped, so when the gradient
/// reverses the gate reads `m·g ≤ 0` and holds the step for roughly `1/(1−β₂)` updates more.
pub(crate) fn clip_report(params: &[Tunable], clipped: &[u64], updates: u64) -> String {
    let mut pinned: Vec<_> = params.iter().filter(|p| clipped[p.idx] > 0).collect();
    if pinned.is_empty() {
        return String::new();
    }

    pinned.sort_unstable_by_key(|p| Reverse(clipped[p.idx]));

    let width = pinned.iter().take(10).map(|p| p.name.len()).max().unwrap_or(20);
    let mut out = String::new();
    let _ = writeln!(out, "\n{LAB}Clip{RESET} {DIM}(share of updates truncated at the bound){RESET}");

    for p in pinned.into_iter().take(10) {
        let _ = writeln!(out, "  {:<width$}  {VAL}{:5.1}%{RESET}", p.name, pct(clipped[p.idx], updates));
    }

    out
}

pub(crate) fn sensitivity_report(params: &[Tunable], grad_ema: &[f64], fixed_mask: &[bool]) {
    let Ok(mut f) = File::create(artifact("sensitivity-report.txt")) else { return };
    let mut w = BufWriter::new(&mut f);
    writeln!(w, "Sensitivity Analysis\n").ok();

    let (mut active, mut frozen) = (Vec::new(), Vec::new());

    for p in params {
        let bucket = if p.is_fixed || fixed_mask[p.idx] { &mut frozen } else { &mut active };
        bucket.push((grad_ema[p.idx], p.name.as_str()));
    }

    active.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    frozen.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let max_width = |list: &[(f64, &str)]| list.iter().take(10).map(|r| r.1.len()).max().unwrap_or(20) + 1;
    let active_w = max_width(&active);
    let write_section = |w: &mut BufWriter<&mut File>, title: &str, width: usize, rows: &mut dyn Iterator<Item = &(f64, &str)>| {
        writeln!(w, "{title}").ok();
        for (i, (delta, name)) in rows.take(10).enumerate() {
            writeln!(w, "    {:>3}. {name:<width$} ΔL: {delta:.8}", i + 1).ok();
        }
    };

    write_section(&mut w, "  Most sensitive parameters:", active_w, &mut active.iter());
    write_section(&mut w, "\n  Least sensitive parameters:", active_w, &mut active.iter().rev());
    if !frozen.is_empty() {
        write_section(&mut w, "\n  Most sensitive frozen parameters:", max_width(&frozen), &mut frozen.iter());
    }
}

pub(crate) fn print_phase_balance(hist: &[u64], weights: &[f64], cap: f64, clamped: usize) {
    let max_pop = hist.iter().copied().max().unwrap_or(0);
    let min_pop = hist.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
    let imbalance = if min_pop > 0 { max_pop as f64 / min_pop as f64 } else { f64::INFINITY };
    let bars: String = hist
        .iter()
        .map(|&c| if c == 0 { ' ' } else { EIGHTH_BLOCKS[(((c as f64 / max_pop.max(1) as f64) * 7.0).round() as usize).min(7)] })
        .collect();
    let wmin = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let wmax = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    println!("{LAB}Phase balance:{RESET} {VAL}{bars}{RESET} {LAB}(phase 0..{}){RESET}", hist.len() - 1);
    println!(
        "  {LAB}imbalance{RESET} {VAL}{imbalance:.0}×{RESET} {LAB}vs cap{RESET} {VAL}{cap:.0}×{RESET}  \
         {LAB}weights{RESET} {VAL}{wmin:.2}–{wmax:.2}×{RESET}  {LAB}clamped{RESET} {VAL}{:.1}%{RESET}",
        pct(clamped as u64, weights.len() as u64)
    );
}

fn write_params<W: Write>(w: &mut W, params: &[Tunable], values: &[f64], initial: Option<&[f64]>) {
    let colored = initial.is_some();
    if colored {
        writeln!(w, "\n// --- Tuned Parameters (paste into eval_params.rs) ---").ok();
    }
    writeln!(w, "define_psqt_params! {{").ok();
    if colored {
        writeln!(w, "    // Files A-D (mirrored to E-H) × 8 ranks").ok();
    }

    for p_idx in 0..PIECE_TABLES {
        let off = p_idx * 2 * TABLE_SQUARES;
        let name = params[off]
            .name
            .strip_prefix("MG_")
            .unwrap_or(&params[off].name)
            .split('[')
            .next()
            .unwrap();
        writeln!(w, "    {name} = [").ok();
        for rank in 0..8 {
            write!(w, "        ").ok();
            for file in 0..4 {
                let sq = rank * 4 + file;
                let (pair, changed) = format_pair(off + sq, off + TABLE_SQUARES + sq, values, params, initial);
                let cell = if file < 3 { format!("{pair: <16}") } else { pair };
                write!(w, "{}", highlight(&cell, changed, initial)).ok();
            }
            writeln!(w).ok();
        }
        writeln!(w, "    ],").ok();
        if p_idx + 1 < PIECE_TABLES {
            writeln!(w).ok();
        }
    }

    let blocks_of = |g: Group| BLOCKS.iter().filter(|b| b.group == g).collect::<Vec<_>>();
    let simple = blocks_of(Group::Simple);

    if !simple.is_empty() {
        writeln!(w, "}}\n\ndefine_simple_params! {{").ok();
    }

    for block in simple {
        let half = block.len / 2;
        writeln!(w, "    {} = [", block.name).ok();
        for i in 0..half {
            let (pair, changed) = format_pair(block.offset + i, block.offset + half + i, values, params, initial);
            let label = PIECE_NAMES.get(i).map_or(String::new(), |n| format!(" // {n}"));
            writeln!(w, "         {}", highlight(&format!("{pair}{label}"), changed, initial)).ok();
        }
        writeln!(w, "    ],").ok();
    }
    writeln!(w, "}}").ok();

    let simd = blocks_of(Group::Simd);
    if !simd.is_empty() {
        writeln!(w, "\ndefine_simd_params! {{").ok();
        for block in simd {
            let half = block.len / 2;
            writeln!(w, "    {} {{", block.name).ok();
            write!(w, "        mg = [").ok();
            write_weight_array(w, block.offset, half, values, params, initial);
            writeln!(w, "],{}", annotation(block.name)).ok();
            write!(w, "        eg = [").ok();
            write_weight_array(w, block.offset + half, half, values, params, initial);
            writeln!(w, "],\n    }},").ok();
        }
        writeln!(w, "}}").ok();
    }

    let weights = blocks_of(Group::Weight);
    if !weights.is_empty() {
        writeln!(w, "\ndefine_weight_params! {{").ok();
        for (s, section) in weights.chunk_by(|a, b| a.section == b.section).enumerate() {
            let width = section.iter().map(|b| b.name.len()).max().unwrap_or(0);
            if s > 0 {
                writeln!(w).ok();
            }

            for (i, block) in section.iter().enumerate() {
                let end = if i + 1 == section.len() { ';' } else { ',' };
                write!(w, "    {:<width$} = [", block.name).ok();
                write_weight_array(w, block.offset, block.len, values, params, initial);
                writeln!(w, "]{end}{}", annotation(block.name)).ok();
            }
        }
        writeln!(w, "}}").ok();
    }
    if colored {
        writeln!(w, "// -------------------------------------\n").ok();
    }
}

fn format_pair(mg_idx: usize, eg_idx: usize, values: &[f64], params: &[Tunable], initial: Option<&[f64]>) -> (String, bool) {
    let (mg, eg) = (values[mg_idx].round() as i32, values[eg_idx].round() as i32);
    let pair = if params[mg_idx].is_fixed { format!("CS({mg:>3}, {eg:>4}),") } else { format!("S({mg:>4}, {eg:>4}),") };
    let changed = initial.is_some_and(|ini| mg != ini[mg_idx].round() as i32 || eg != ini[eg_idx].round() as i32);
    (pair, changed)
}

fn write_weight_array<W: Write>(
    w: &mut W,
    offset: usize,
    count: usize,
    values: &[f64],
    params: &[Tunable],
    initial: Option<&[f64]>,
) {
    for i in 0..count {
        let idx = offset + i;
        let val = values[idx].round() as i32;

        if i > 0 {
            write!(w, ", ").ok();
        }

        let s = format!("{}({val})", if params[idx].is_fixed { "CV" } else { "V" });
        let changed = initial.is_some_and(|ini| val != ini[idx].round() as i32);
        write!(w, "{}", highlight(&s, changed, initial)).ok();
    }
}

fn loss_sparkline(history: &[f64]) -> String {
    if history.is_empty() {
        return String::new();
    }

    let lo = history.iter().copied().fold(f64::MAX, f64::min);
    let hi = history.iter().copied().fold(f64::MIN, f64::max);
    let span = (hi - lo).max(1e-12);
    let mut out = String::with_capacity(history.len() * 20);

    for &v in history {
        let frac = (v - lo) / span;
        out.push_str(&fg(color::advantage(1.0 - 2.0 * frac)));
        out.push(EIGHTH_BLOCKS[(frac * 8.0).min(7.0) as usize]);
    }

    out.push_str(RESET);
    out
}

fn annotation(block: &str) -> String {
    ANNOTATIONS
        .iter()
        .find(|(n, _)| *n == block)
        .map_or_else(String::new, |(_, t)| format!(" // {t}"))
}

fn highlight(text: &str, changed: bool, initial: Option<&[f64]>) -> String {
    if initial.is_some() && changed { format!("{MOVED}{text}{RESET}") } else { text.to_string() }
}

fn compact(n: u64) -> String {
    match n {
        0..10_000 => n.to_string(),
        10_000..10_000_000 => format!("{}K", n / 1000),
        _ => format!("{}M", n / 1_000_000),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::{collect_parameters, default_values};

    #[test]
    fn the_paste_block_reproduces_eval_params() {
        let params = collect_parameters();
        let values = default_values(&params);
        let mut printed = Vec::new();
        write_params(&mut printed, &params, &values, Some(&values));
        let printed = String::from_utf8(printed).expect("the paste block is utf-8");
        let printed = printed
            .trim_start_matches('\n')
            .trim_start_matches("// --- Tuned Parameters (paste into eval_params.rs) ---\n")
            .trim_end()
            .trim_end_matches("// -------------------------------------")
            .trim_end();
        let source = include_str!("../../src/engine/eval_params.rs");
        let start = source.find("define_psqt_params! {").expect("missing PSQT macro block in eval_params.rs");
        let last = source
            .find("define_weight_params! {")
            .expect("missing weight macro block in eval_params.rs");
        let end = last + source[last..].find("\n}").expect("unclosed weight macro block") + 2;
        assert_eq!(printed, &source[start..end], "generated macro blocks deviated from eval_params.rs");
    }
}
