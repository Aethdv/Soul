//! Tuner output formatting: final parameter tables, weight arrays, and
//! the training summary printed at the end of a run.

use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use soul::{
    color,
    engine::eval_params::{BLOCKS, Block, Group, Tunable},
};

use crate::evaltune::palette;

/// Trailing comments for the paste block, by block name. Cosmetic: offsets,
/// widths and order come from `BLOCKS`, so a missing entry costs a comment and
/// a stale one cannot move a number.
#[rustfmt::skip]
const ANNOTATIONS: &[(&str, &str)] = &[
    ("mobility_open",      "[mobility, battery, threats, xray threats]"),
    ("phase",              "[P, N, B, R, Q, K]"),
    ("attacker",           "[0, 1, 2, 3, 4, 5] attackers × weak"),
    ("king_safety",        "[Pawn Shield, Ortho Exp, Diag Exp]"),
    ("xray",               "[Ortho King]"),
    ("king_danger",        "pressure curvature, over DANGER_SCALE; floored at 0, the data pulls under"),
    ("bishop_pair",        "[MG, EG]"),
    ("rook_open",          "[MG, EG]"),
    ("passed_pawn_mg",     "by relative rank 1-6"),
    ("passed_pawn_eg",     "by relative rank 1-6"),
    ("enemy_king_dist_mg", "enemy king→passer dist, 7 clamps to 6"),
    ("enemy_king_dist_eg", "enemy king→passer dist, 7 clamps to 6"),
    ("doubled_pawn",       "[MG, EG]"),
    ("isolated_pawn",      "[MG, EG]"),
    ("phalanx_mg",         "by relative rank 2-7"),
    ("phalanx_eg",         "by relative rank 2-7"),
    ("defended_pawn_mg",   "by relative rank 2-7 (rank 2 unreachable)"),
    ("defended_pawn_eg",   "by relative rank 2-7 (rank 2 unreachable)"),
    ("backward_pawn",      "[MG, EG]"),
    ("tempo",              "[MG, EG], side-to-move initiative"),
    ("minor_behind_pawn",  "[MG, EG]"),
];

pub struct BestEpochs<'a> {
    pub best_val_params: &'a [f64],
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    pub best_train_params: &'a [f64],
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    pub last_val: f64,
    pub last_train: f64,
}

pub fn print_results(all_params: &[Tunable], initial_values: &[f64], final_ema: &[f64], best: &BestEpochs, final_epoch: usize) {
    let gold = color::ansi_fg((218, 165, 32));

    println!();
    println!("{gold}Best L_val: {:.6} (Epoch {}){}", best.best_val_loss, best.best_val_epoch, palette::RESET);
    print_params(all_params, initial_values, best.best_val_params);

    println!();
    println!("{gold}Best L_train: {:.6} (Epoch {}){}", best.best_train_loss, best.best_train_epoch, palette::RESET);
    print_params(all_params, initial_values, best.best_train_params);

    println!();
    println!(
        "{gold}Final epoch {final_epoch}:  L_val {:.6}  L_train {:.6}{}",
        best.last_val,
        best.last_train,
        palette::RESET
    );
    print_params(all_params, initial_values, final_ema);

    if let Ok(mut f) = File::create("evaltune_best.txt") {
        let mut w = BufWriter::new(&mut f);

        writeln!(w, "Best L_val: {:.6} (Epoch {})", best.best_val_loss, best.best_val_epoch).ok();
        write_params(&mut w, all_params, best.best_val_params, None);
        writeln!(w, "\nBest L_train: {:.6} (Epoch {})", best.best_train_loss, best.best_train_epoch).ok();
        write_params(&mut w, all_params, best.best_train_params, None);
        writeln!(w, "\nFinal epoch {final_epoch}:  L_val {:.6}  L_train {:.6}", best.last_val, best.last_train).ok();
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

    for p_idx in 0..6 {
        let psqt_offset = p_idx * 64;
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
                let eg_idx = psqt_offset + 32 + sq_idx;

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

        if p_idx < 5 {
            writeln!(w).ok();
        }
    }

    let simple = present_blocks(Group::Simple, params);

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

            if eg_idx >= params.len() {
                break;
            }

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

    let simd = present_blocks(Group::Simd, params);

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

    let weights = present_blocks(Group::Weight, params);

    if !weights.is_empty() {
        let width = weights.iter().map(|b| b.name.len()).max().unwrap_or(0);
        writeln!(w, "\ndefine_weight_params! {{").ok();

        for block in weights {
            write!(w, "    {:<width$} = [", block.name).ok();
            write_weight_array(w, block.offset, block.len, values, params, initial);
            writeln!(w, "],{}", annotation(block.name)).ok();
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

        if idx >= values.len() {
            break;
        }

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

fn present_blocks(group: Group, params: &[Tunable]) -> Vec<&'static Block> {
    BLOCKS.iter().filter(|b| b.group == group && params.len() > b.offset).collect()
}

fn annotation(block: &str) -> String {
    match ANNOTATIONS.iter().find(|(name, _)| *name == block) {
        Some((_, text)) => format!(" // {text}"),
        None => String::new(),
    }
}

/// Green ANSI if `changed` and `initial` is `Some` (terminal context).
fn highlight(text: &str, changed: bool, initial: Option<&[f64]>) -> String {
    if initial.is_some() && changed {
        format!("{}{text}{}", color::ansi_fg((100, 200, 120)), palette::RESET)
    } else {
        text.to_string()
    }
}

#[cfg(test)]
mod tests {
    use soul::engine::eval_params::collect_parameters;

    use super::*;

    /// The output is text, so neither the compiler nor the oracle reads it.
    #[test]
    fn the_paste_block_reproduces_eval_params() {
        let params = collect_parameters();
        let values: Vec<f64> = params.iter().map(|p| p.value).collect();

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
