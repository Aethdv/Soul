//! Everything a run prints: the parameter tables it pastes back, the end-of-run
//! summary, and the diagnostics beside it, calibration, sensitivity, the gate
//! census and the two scale warnings.

use std::{
    fmt::Write as _,
    fs::{self, File},
    io::{self, BufWriter, Write},
};

use rayon::prelude::*;

use super::{
    engine::{BLOCKS, Block, Group, PIECE_TABLES, TABLE_SQUARES, TOTAL_PHASE, Tunable, color},
    groups::GROUP_NAMES,
    lion::GateCensus,
    loader,
    run::{TrainerContext, artifact},
    training::{phase_of, phase_weights, sigmoid},
};
use crate::{
    core::config::{EvalTuneConfig, KMode},
    evaltune::palette::{self, BRAND, COUNT, DIM, LAB, MOVED, RESET, VAL},
};

/// Eighth-blocks, shortest to tallest; every sparkline here indexes a height level into this.
const BARS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];

/// Trailing comments for the paste block, by block name.
/// Cosmetic: offsets, widths and order come from `BLOCKS`, so a missing entry
/// costs a comment and a stale one cannot move a number.
#[rustfmt::skip]
const ANNOTATIONS: &[(&str, &str)] = &[
    ("mobility_open",      "[mobility, battery, threats, xray threats]"),
    ("phase",              "[P, N, B, R, Q, K]"),
    ("king_safety",        "[Pawn Shield, Ortho Exp, Diag Exp]"),
    ("attacker",           "[0, 1, 2, 3, 4, 5] attackers × weak"),
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

pub struct BestEpochs<'a> {
    pub best_val_params: &'a [f64],
    /// `None` when the run had no holdout: nothing validated, so nothing ranks.
    pub best_val_loss: Option<f64>,
    pub best_val_epoch: usize,
    pub best_train_params: &'a [f64],
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    pub last_val: Option<f64>,
    pub last_train: Option<f64>,
}

/// `0.123456` for a loss, `—` for a run that had no holdout.
pub(super) fn fmt_loss(v: Option<f64>) -> String {
    match v {
        Some(l) => format!("{l:.6}"),
        None => "—".to_string(),
    }
}

/// Final epoch is printed; all three go to `evaltune_best.txt`.
pub fn print_results(all_params: &[Tunable], initial_values: &[f64], final_ema: &[f64], best: &BestEpochs, final_epoch: usize) {
    println!();

    match best.best_val_loss {
        Some(l) => println!("{BRAND}Best L_val: {:.6} (Epoch {}){RESET}", l, best.best_val_epoch),
        None => println!("{BRAND}Best L_val: — (no holdout){RESET}"),
    }

    println!("{BRAND}Best L_train: {:.6} (Epoch {}){RESET}", best.best_train_loss, best.best_train_epoch);

    println!();
    println!(
        "{BRAND}Final epoch {final_epoch}:  L_val {}  L_train {}{RESET}",
        fmt_loss(best.last_val),
        fmt_loss(best.last_train),
    );
    print_params(all_params, initial_values, final_ema);

    if let Ok(mut f) = File::create(artifact("evaltune_best.txt")) {
        let mut w = BufWriter::new(&mut f);

        // Without a holdout the val block is the train block: selection fell back to it.
        if let Some(l) = best.best_val_loss {
            writeln!(w, "Best L_val: {l:.6} (Epoch {})", best.best_val_epoch).ok();
            write_params(&mut w, all_params, best.best_val_params, None);
        } else {
            writeln!(w, "Best L_val: — (no holdout)").ok();
        }

        writeln!(w, "\nBest L_train: {:.6} (Epoch {})", best.best_train_loss, best.best_train_epoch).ok();
        write_params(&mut w, all_params, best.best_train_params, None);
        writeln!(w, "\nFinal epoch {final_epoch}:  L_val {}  L_train {}", fmt_loss(best.last_val), fmt_loss(best.last_train),).ok();
        write_params(&mut w, all_params, final_ema, None);
    }
}

/// Prints parameters to stdout with ANSI green highlighting for changed values.
pub fn print_params(params: &[Tunable], initial: &[f64], values: &[f64]) {
    let mut out = io::stdout().lock();
    write_params(&mut out, params, values, Some(initial));
}

/// Rounds parameters onto the integer grid the engine runs on.
///
/// `write_params` emits `round() as i32`, so two parameter vectors that agree here are the same
/// engine however far apart they sit in f64.
pub fn quantize(values: &[f64], out: &mut [i32]) {
    for (q, v) in out.iter_mut().zip(values) {
        *q = v.round() as i32;
    }
}

/// Pass `initial` to highlight changed values, `None` for plain output.
pub fn write_params<W: Write>(w: &mut W, params: &[Tunable], values: &[f64], initial: Option<&[f64]>) {
    let colored = initial.is_some();

    if colored {
        writeln!(w, "\n// --- Tuned Parameters (paste into eval_params.rs) ---").ok();
    }

    writeln!(w, "define_psqt_params! {{").ok();

    if colored {
        writeln!(w, "    // Files A-D (mirrored to E-H) × 8 ranks").ok();
    }

    for p_idx in 0..PIECE_TABLES {
        let psqt_offset = p_idx * 2 * TABLE_SQUARES;
        let name = params[psqt_offset]
            .name
            .strip_prefix("MG_")
            .unwrap_or(&params[psqt_offset].name)
            .split('[')
            .next()
            .unwrap();

        writeln!(w, "    {name} = [").ok();

        for row in 0..8 {
            write!(w, "        ").ok();

            for col in 0..4 {
                let sq_idx = row * 4 + col;
                let mg_idx = psqt_offset + sq_idx;
                let eg_idx = psqt_offset + TABLE_SQUARES + sq_idx;

                let mg_val = values[mg_idx].round() as i32;
                let eg_val = values[eg_idx].round() as i32;
                let fixed = params[mg_idx].is_fixed;

                let s = if fixed { format!("CS({mg_val:>3}, {eg_val:>4}),") } else { format!("S({mg_val:>4}, {eg_val:>4}),") };

                let changed =
                    initial.is_some_and(|ini| mg_val != ini[mg_idx].round() as i32 || eg_val != ini[eg_idx].round() as i32);

                let cell = if col < 3 { format!("{s: <16}") } else { s };
                write!(w, "{}", highlight(&cell, changed, initial)).ok();
            }

            writeln!(w).ok();
        }

        writeln!(w, "    ],").ok();

        if p_idx + 1 < PIECE_TABLES {
            writeln!(w).ok();
        }
    }

    let simple = present_blocks(Group::Simple);

    if !simple.is_empty() {
        writeln!(w, "}}\n\ndefine_simple_params! {{").ok();
    }

    for block in simple {
        let half = block.len / 2;
        let pieces = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];

        writeln!(w, "    {} = [", block.name).ok();

        for i in 0..half {
            let mg_idx = block.offset + i;
            let eg_idx = block.offset + half + i;

            let mg_val = values[mg_idx].round() as i32;
            let eg_val = values[eg_idx].round() as i32;
            let fixed = params[mg_idx].is_fixed;

            let tag = if fixed { "CS" } else { "S" };
            let label = pieces.get(i).map_or(String::new(), |name| format!(" // {name}"));
            let s = format!("{tag}({mg_val:>4}, {eg_val:>4}),{label}");

            let changed = initial.is_some_and(|ini| mg_val != ini[mg_idx].round() as i32 || eg_val != ini[eg_idx].round() as i32);
            writeln!(w, "         {}", highlight(&s, changed, initial)).ok();
        }

        writeln!(w, "    ],").ok();
    }

    writeln!(w, "}}").ok();

    let simd = present_blocks(Group::Simd);

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
            writeln!(w, "],").ok();
            writeln!(w, "    }},").ok();
        }

        writeln!(w, "}}").ok();
    }

    let weights = present_blocks(Group::Weight);

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

/// Writes a slice of weight parameters as a comma-separated list.
///
/// When `initial` is `Some`, changed values are highlighted with ANSI green.
pub fn write_weight_array<W: Write>(
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
        let fixed = params[idx].is_fixed;

        let tag = if fixed { "CV" } else { "V" };
        let s = format!("{tag}({val})");

        if i > 0 {
            write!(w, ", ").ok();
        }

        let changed = initial.is_some_and(|ini| val != ini[idx].round() as i32);
        write!(w, "{}", highlight(&s, changed, initial)).ok();
    }
}

/// The gauge line reports how hard the pull was; this warns when the final parameters ship off the reference scale.
///
/// A run can hold a statistic perfectly and still ship an eval off the scale
/// `search_params` was written against, which is worth −25 Elo and looks like
/// nothing in the loss. The tail EMA averages vectors that are individually on
/// the reference and lands fractionally under it, so the bar sits well clear of
/// that rather than at the gauge's own 1e-6.
pub fn off_scale_warning(shipped: f64) -> String {
    if (shipped - 1.0).abs() <= 0.01 {
        return String::new();
    }

    format!(
        "{}[!] Warning: the final-epoch parameters ship at {shipped:.3}× the reference scale.\n\
         [!] `search_params` reads centipawns; an eval off scale moves every margin with it.{RESET}\n",
        palette::ALARM,
    )
}

/// Warns when the run's K finished against `k_min` or `k_max`.
///
/// Both live modes clamp: the golden search never leaves its bracket and `on_batch` clamps the
/// learned K every batch. A K on a bound is therefore the bracket's answer rather than the data's,
/// and it is silent otherwise. The 32.8M set spent a run pinned to a `k_min` of 0.003 and settled
/// at 0.001350 once the floor moved.
pub fn clamped_k_warning(config: &EvalTuneConfig, k: f64) -> String {
    // Fixed K is the configured value by definition, bound or not.
    if matches!(config.k_mode, KMode::Fixed { .. }) {
        return String::new();
    }

    let margin = 0.01 * (config.k_max - config.k_min);

    if k - config.k_min >= margin && config.k_max - k >= margin {
        return String::new();
    }

    format!(
        "{}[!] Warning: K = {k:.6} finished against its bracket [{}, {}]. Widen it and rerun;\n\
         [!] this run reported a clamp rather than an optimum.{RESET}\n",
        palette::ALARM,
        config.k_min,
        config.k_max,
    )
}

/// Predicted against realized win rate on the validation split, by game phase.
///
/// One global K asserts that a centipawn buys the same win probability in a rook ending as it
/// does at full material. Whether it does is measurable rather than arguable, and this is the
/// measurement: a residual that walks monotonically with phase is a material-conditioned target
/// earning its keep, and noise around zero is that idea deflating before it costs a single game.
///
/// The second table splits each band by eval, because K is a slope and the first table reads an
/// offset. A band whose K is too flat under-predicts the winning side and over-predicts the
/// losing side, netting a mean residual of zero, so the split has to keep the sign: bucketed by
/// `|eval|` those two errors would cancel inside the bucket and hide exactly what is sought.
///
/// Realized rate comes from the label, so it means the same thing the loss means: on outcome data
/// it is the game result, and on score-target data it is whatever the blend made of it.
pub fn calibration_report(ctx: &TrainerContext, values: &[f64], k: f64) -> String {
    const BAND_WIDTH: usize = 4;
    const BANDS: usize = TOTAL_PHASE as usize / BAND_WIDTH;

    /// Centipawn cuts of the signed-eval buckets each phase band splits into.
    const EDGES: [f64; 2] = [-50.0, 50.0];
    const CELLS: usize = EDGES.len() + 1;

    let phase_w = phase_weights();

    let (counts, predicted, realized) = ctx
        .val
        .par_iter()
        .enumerate()
        .fold(
            || ([0u64; BANDS * CELLS], [0.0_f64; BANDS * CELLS], [0.0_f64; BANDS * CELLS]),
            |(mut counts, mut predicted, mut realized), (idx, entry)| {
                let record = &ctx.records[ctx.train_count + idx];

                if !ctx.passes_vol_filter(entry, record.static_eval) {
                    return (counts, predicted, realized);
                }

                // Full material divides into the top band rather than owning one of its own.
                let b = (phase_of(record, &phase_w) / BAND_WIDTH).min(BANDS - 1);
                let eval = loader::eval_record(record, values);
                let cell = b * CELLS + EDGES.iter().filter(|&&edge| eval >= edge).count();

                counts[cell] += 1;
                predicted[cell] += sigmoid(eval, k);
                realized[cell] += f64::from(entry.result) / 2.0;

                (counts, predicted, realized)
            },
        )
        .reduce(
            || ([0u64; BANDS * CELLS], [0.0_f64; BANDS * CELLS], [0.0_f64; BANDS * CELLS]),
            |(mut c1, mut p1, mut r1), (c2, p2, r2)| {
                for i in 0..BANDS * CELLS {
                    c1[i] += c2[i];
                    p1[i] += p2[i];
                    r1[i] += r2[i];
                }

                (c1, p1, r1)
            },
        );

    let band_label = |b: usize| {
        let lo = b * BAND_WIDTH;
        let hi = if b + 1 == BANDS { TOTAL_PHASE as usize } else { lo + BAND_WIDTH - 1 };

        format!("{lo}-{hi}")
    };

    let rate = |sum: f64, n: u64| 100.0 * sum / n as f64;

    let mut out = String::new();

    let _ = writeln!(out, "\n{LAB}Calibration{RESET} {DIM}(best-val parameters, validation split at K = {k:.6}){RESET}");
    let _ = writeln!(out, "  {LAB}phase           n   predicted   realized  residual{RESET}");

    for b in 0..BANDS {
        let cells = b * CELLS..(b + 1) * CELLS;
        let n: u64 = counts[cells.clone()].iter().sum();

        if n == 0 {
            continue;
        }

        let p = rate(predicted[cells.clone()].iter().sum(), n);
        let r = rate(realized[cells].iter().sum(), n);
        let band = band_label(b);

        let _ = writeln!(
            out,
            "  {band:<7} {VAL}{n:>9}{RESET}      {VAL}{p:5.1}%{RESET}     {VAL}{r:5.1}%{RESET}     {VAL}{:+5.1}{RESET}",
            p - r
        );
    }

    let _ = writeln!(out, "\n{LAB}Residual by eval within phase{RESET} {DIM}(cell counts in parentheses){RESET}");
    let _ = writeln!(out, "  {LAB}{:<7} {:<13} {:<13} eval > +50{RESET}", "phase", "eval < -50", "-50..+50");

    for b in 0..BANDS {
        let row: Vec<String> = (0..CELLS)
            .map(|c| {
                let i = b * CELLS + c;

                match counts[i] {
                    0 => "-".to_string(),
                    n => format!("{:+.1} ({})", rate(predicted[i], n) - rate(realized[i], n), compact(n)),
                }
            })
            .collect();

        if row.iter().all(|cell| cell == "-") {
            continue;
        }

        let _ = writeln!(out, "  {:<7} {:<13} {:<13} {}", band_label(b), row[0], row[1], row[2]);
    }

    out
}

/// Whole-run gate census, per parameter group.
///
/// `band` is the column the cautious-mask question turns on, since it is where our gate and
/// Liang's disagree; the rest of a retune's difference would be step length, not mask shape.
pub fn gate_census_report(groups: &[GateCensus]) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "\n{LAB}Gate census{RESET} {DIM}(share of parameter-updates){RESET}");
    let _ = writeln!(out, "  {LAB}group          φ     skip  canonical    band   c-only   waived     dead  no grad{RESET}");

    for (name, c) in GROUP_NAMES.iter().zip(groups) {
        let _ = writeln!(
            out,
            "  {name:<9} {VAL}{:6.4}{RESET}   {VAL}{:5.1}%{RESET}     {VAL}{:5.1}%{RESET}  {VAL}{:5.2}%{RESET}   \
             {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}   {VAL}{:5.1}%{RESET}",
            c.active_share(),
            c.percent(c.skipped),
            c.percent(c.canonical),
            c.percent(c.band),
            c.percent(c.canonical_only),
            c.percent(c.epsilon_waived),
            c.percent(c.dead),
            c.percent(c.absent),
        );
    }

    out
}

pub fn print_dataset_stats<T, F: Fn(&T) -> f64>(train: &[T], val: &[T], total: usize, result_fn: F) {
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

    println!("  {LAB}White wins:{RESET} {COUNT}{ww}{RESET}");
    println!("  {LAB}Black wins:{RESET} {COUNT}{bw}{RESET}");
    println!("  {LAB}Draws:{RESET}      {COUNT}{dr}{RESET}");

    // A datagen run that never filled the result field looks exactly like a set of drawn games.
    // The outcome target is then 0.5 everywhere and only a score-weighted blend can learn.
    if ww + bw == 0 {
        eprintln!(
            "{}[!] Warning: no decisive results. Every outcome target is 0.5, so a wdl_schedule\n\
             [!] near 0.0 trains on a constant.{RESET}",
            palette::ALARM,
        );
    }
}

/// Loss history as a sparkline: lower loss → shorter block.
pub fn loss_sparkline(history: &[f64]) -> String {
    if history.is_empty() {
        return String::new();
    }

    let lo = history.iter().copied().fold(f64::MAX, f64::min);
    let hi = history.iter().copied().fold(f64::MIN, f64::max);
    let span = (hi - lo).max(1e-12);

    let mut out = String::with_capacity(history.len() * 20);

    for &v in history {
        let frac = (v - lo) / span; // 0 = best (lowest), 1 = worst (highest)
        let level = (frac * 8.0).min(7.0) as usize;

        out.push_str(&palette::fg(color::advantage(1.0 - 2.0 * frac)));
        out.push(BARS[level]);
    }

    out.push_str(RESET);
    out
}

/// Which parameters the clip bound truncated, and how often.
///
/// Momentum keeps accumulating outward while only the value is clamped, so when the
/// gradient reverses the gate reads
/// `m·g ≤ 0` and holds the step for roughly `1/(1−β₂)` updates more. The clamp and the
/// gate are stickiest together at exactly the moment a parameter tries to leave the wall.
pub fn clip_report(params: &[Tunable], clipped: &[u64], updates: u64) -> String {
    let mut pinned: Vec<_> = params.iter().filter(|p| clipped[p.idx] > 0).collect();

    if pinned.is_empty() {
        return String::new();
    }

    pinned.sort_unstable_by_key(|p| std::cmp::Reverse(clipped[p.idx]));

    let width = pinned.iter().take(10).map(|p| p.name.len()).max().unwrap_or(20);

    let mut out = String::new();
    let _ = writeln!(out, "\n{LAB}Clip{RESET} {}(share of updates truncated at the bound){RESET}", palette::DIM);

    for p in pinned.iter().take(10) {
        let share = 100.0 * clipped[p.idx] as f64 / updates.max(1) as f64;
        let _ = writeln!(out, "  {:<width$}  {VAL}{share:5.1}%{RESET}", p.name);
    }

    out
}

pub fn sensitivity_report(params: &[Tunable], grad_ema: &[f64], fixed_mask: &[bool]) {
    let Ok(mut f) = fs::File::create(artifact("sensitivity-report.txt")) else { return };
    let mut w = io::BufWriter::new(&mut f);

    writeln!(w, "Sensitivity Analysis").ok();
    writeln!(w).ok();

    let mut sensitivities = Vec::new();
    let mut frozen = Vec::new();

    for p in params {
        let delta = grad_ema[p.idx];

        if p.is_fixed || fixed_mask[p.idx] {
            frozen.push((delta, p.idx, p.name.as_str()));
        } else {
            sensitivities.push((delta, p.idx, p.name.as_str()));
        }
    }

    sensitivities.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    frozen.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let max_width = |list: &[(f64, usize, &str)]| list.iter().take(10).map(|r| r.2.len()).max().unwrap_or(20) + 1;
    let active_width = max_width(&sensitivities);
    let frozen_width = max_width(&frozen);

    writeln!(w, "  Top Load-Bearing Parameters:").ok();

    for (i, (delta, _, name)) in sensitivities.iter().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    writeln!(w).ok();
    writeln!(w, "  Lowest-Impact Parameters:").ok();

    for (i, (delta, _, name)) in sensitivities.iter().rev().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    if !frozen.is_empty() {
        writeln!(w).ok();
        writeln!(w, "  Highest Sensitivity Auto-Frozen/Fixed Parameters:").ok();

        for (i, (delta, _, name)) in frozen.iter().take(10).enumerate() {
            writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = frozen_width).ok();
        }
    }
}

/// Set `phase_balance_cap` toward the printed imbalance to fully correct it,
/// or lower to spare the sparse buckets their variance.
pub fn report_phase_balance(hist: &[u64], weights: &[f64], cap: f64, clamped: usize) {
    let max_pop = hist.iter().copied().max().unwrap_or(0);
    let min_pop = hist.iter().copied().filter(|&c| c > 0).min().unwrap_or(0);
    let imbalance = if min_pop > 0 { max_pop as f64 / min_pop as f64 } else { f64::INFINITY };

    let bars: String = hist
        .iter()
        .map(|&c| if c == 0 { ' ' } else { BARS[(((c as f64 / max_pop.max(1) as f64) * 7.0).round() as usize).min(7)] })
        .collect();

    let wmin = weights.iter().copied().fold(f64::INFINITY, f64::min);
    let wmax = weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let clamp_pct = 100.0 * clamped as f64 / weights.len().max(1) as f64;

    println!("{LAB}Phase balance:{RESET} {VAL}{bars}{RESET} {LAB}(phase 0..{}){RESET}", hist.len() - 1);
    println!(
        "  {LAB}imbalance{RESET} {VAL}{imbalance:.0}×{RESET} {LAB}vs cap{RESET} {VAL}{cap:.0}×{RESET}  \
         {LAB}weights{RESET} {VAL}{wmin:.2}–{wmax:.2}×{RESET}  {LAB}clamped{RESET} {VAL}{clamp_pct:.1}%{RESET}"
    );
}

fn present_blocks(group: Group) -> Vec<&'static Block> {
    BLOCKS.iter().filter(|b| b.group == group).collect()
}

fn annotation(block: &str) -> String {
    match ANNOTATIONS.iter().find(|(name, _)| *name == block) {
        Some((_, text)) => format!(" // {text}"),
        None => String::new(),
    }
}

/// Green ANSI if `changed` and `initial` is `Some` (terminal context).
fn highlight(text: &str, changed: bool, initial: Option<&[f64]>) -> String {
    if initial.is_some() && changed { format!("{MOVED}{text}{RESET}") } else { text.to_string() }
}

/// Counts wide enough to crowd a table, shortened to three significant characters.
fn compact(n: u64) -> String {
    match n {
        0..10_000 => n.to_string(),
        10_000..10_000_000 => format!("{}k", n / 1000),
        _ => format!("{}M", n / 1_000_000),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        super::engine::{collect_parameters, default_values},
        *,
    };

    /// The output is text, so neither the compiler nor the oracle reads it.
    #[test]
    fn the_paste_block_reproduces_eval_params() {
        let params = collect_parameters();
        let values = default_values(&params);

        // The colored form is the one with the header comments; values as their own
        // baseline mark nothing changed, so no ANSI lands in it.
        let mut printed = Vec::new();
        write_params(&mut printed, &params, &values, Some(&values));
        let printed = String::from_utf8(printed).expect("the paste block is utf-8");
        let printed = printed
            .trim_start_matches('\n')
            .trim_start_matches("// --- Tuned Parameters (paste into eval_params.rs) ---\n")
            .trim_end()
            .trim_end_matches("// -------------------------------------")
            .trim_end();

        let source = include_str!("../../../src/engine/eval_params.rs");
        let start = source.find("define_psqt_params! {").expect("no psqt block in eval_params.rs");
        let last = source.find("define_weight_params! {").expect("no weight block in eval_params.rs");
        let end = last + source[last..].find("\n}").expect("the weight block never closes") + 2;

        assert_eq!(printed, &source[start..end], "the paste block no longer reproduces eval_params.rs");
    }
}
