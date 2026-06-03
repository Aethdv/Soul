use std::{
    fs,
    fs::File,
    io,
    io::{BufWriter, Write},
};

use soul::{color, core::psqt, engine::eval_params::Tunable};

use crate::evaltune::storage::Snapshot;

pub fn print_results(snapshots: &[Snapshot], all_params: &[Tunable], initial_values: &[f64], values: &[f64], final_epoch: usize) {
    let count = snapshots.len();
    if count == 0 {
        return;
    }

    let best_snap = snapshots.first().unwrap();
    let best_epoch = best_snap.epoch;
    println!();
    println!("Best Snapshot (Epoch {best_epoch}):");
    let mut best_values = vec![0.0; values.len()];

    for t in all_params {
        if let Some(&v) = best_snap.params.get(&t.name) {
            best_values[t.idx] = v;
        } else {
            best_values[t.idx] = values[t.idx];
        }
    }

    let best = best_snap.error;
    print_params(all_params, initial_values, &best_values);

    if let Ok(mut f) = File::create("top-snapshots.txt") {
        let mut w = BufWriter::new(&mut f);
        writeln!(w, "Top {count} snapshots (sorted by L_val):").ok();

        for (i, snap) in snapshots.iter().enumerate() {
            writeln!(w, "  {:>2}. Epoch {:>3} | L_val: {:.6}", i + 1, snap.epoch, snap.error).ok();
        }
    }

    if let Ok(log_file) = fs::OpenOptions::new().append(true).open("evaltune_log.txt") {
        let mut w = BufWriter::new(log_file);
        writeln!(w, "\n{0} Final EMA Parameters (Epoch {final_epoch}) {0}", "──").ok();
        write_params(&mut w, all_params, values, None);
        writeln!(w, "\n{0} Best Snapshot Parameters (Epoch {best_epoch}) {0}", "──").ok();
        write_params(&mut w, all_params, &best_values, None);
    }
    println!("{}Best L_val: {best:.6} (Epoch {best_epoch})\x1b[0m", color::ansi_fg((218, 165, 32)));
}

/// Prints parameters to stdout with ANSI green highlighting for changed values.
pub fn print_params(params: &[Tunable], initial: &[f64], values: &[f64]) {
    let mut out = io::stdout().lock();
    write_params(&mut out, params, values, Some(initial));
}

/// Writes all tuned parameters to any `impl Write` sink.
/// Pass `initial` to highlight changed values, `None` for plain log output.
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
                // Pad between columns, not after the last — trailing pad would
                // ride along on copy-paste.
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

    let mat = psqt::LAYOUT.material_offset;

    if params.len() > mat {
        writeln!(w, "}}\n\ndefine_simple_params! {{").ok();
        let pieces = ["Pawn", "Knight", "Bishop", "Rook", "Queen", "King"];
        writeln!(w, "    MATERIAL = [").ok();

        for (pt, name) in pieces.iter().enumerate() {
            let mg_idx = mat + pt;
            let eg_idx = mat + 6 + pt;

            if eg_idx >= params.len() {
                break;
            }

            let mg_val = values[mg_idx].round() as i32;
            let eg_val = values[eg_idx].round() as i32;
            let fixed = params[mg_idx].is_fixed;

            let tag = if fixed { "CS" } else { "S" };
            let s = format!("{tag}({mg_val:>4}, {eg_val:>4}), // {name}");

            let changed = initial.is_some_and(|ini| mg_val != ini[mg_idx].round() as i32 || eg_val != ini[eg_idx].round() as i32);
            writeln!(w, "         {}", highlight(&s, changed, initial)).ok();
        }
        writeln!(w, "    ],").ok();
    }

    writeln!(w, "}}").ok();

    if params.len() > psqt::LAYOUT.mobility_open_offset {
        writeln!(w, "\ndefine_simd_params! {{").ok();

        #[rustfmt::skip]
        let mobility_bands = [
            ("MG_MOBILITY_OPEN",   psqt::LAYOUT.mobility_open_offset,       " // [mobility, battery, threats, xray threats]"),
            ("EG_MOBILITY_OPEN",   psqt::LAYOUT.mobility_open_offset + 4,   ""),
            ("MG_MOBILITY_CLOSED", psqt::LAYOUT.mobility_closed_offset,     ""),
            ("EG_MOBILITY_CLOSED", psqt::LAYOUT.mobility_closed_offset + 4, ""),
        ];

        for (name, offset, comment) in &mobility_bands {
            writeln!(w, "    {name} = [").ok();
            write!(w, "        ").ok();
            write_weight_array(w, *offset, 4, values, params, initial);
            writeln!(w, "],{comment}").ok();
        }

        writeln!(w, "}}").ok();
    }

    if params.len() > psqt::LAYOUT.weight_offset {
        writeln!(w, "\ndefine_weight_params! {{").ok();

        let l = psqt::LAYOUT;

        #[rustfmt::skip]
        let bands: &[(&str, usize, usize, &str)] = &[
            ("PHASE_WEIGHTS",         l.weight_offset,             6, " // [P, N, B, R, Q, K]"),
            ("ATTACKER_WEIGHTS",      l.attacker_offset,           6, " // [0, 1, 2, 3, 4, 5] attackers × weak"),
            ("KING_SAFETY_WEIGHTS",   l.king_safety_offset,        3, " // [Pawn Shield, Ortho Exp, Diag Exp]"),
            ("XRAY_WEIGHTS",          l.xray_offset,               1, " // [Ortho King]"),
            ("BISHOP_PAIR_WEIGHTS",   l.bishop_pair_offset,        2, " // [MG, EG]"),
            ("ROOK_OPEN_WEIGHTS",     l.rook_open_offset,          2, " // [MG, EG]"),
            ("PASSED_PAWN_MG",        l.passed_pawn_mg_offset,     6, " // by relative rank 1-6"),
            ("PASSED_PAWN_EG",        l.passed_pawn_eg_offset,     6, " // by relative rank 1-6"),
            ("ENEMY_KING_DIST_MG",    l.enemy_king_dist_mg_offset, 6, " // enemy king→passer dist, 7 clamps to 6"),
            ("ENEMY_KING_DIST_EG",    l.enemy_king_dist_eg_offset, 6, " // enemy king→passer dist, 7 clamps to 6"),
            ("DOUBLED_PAWN_WEIGHTS",  l.doubled_pawn_offset,       2, " // [MG, EG]"),
            ("ISOLATED_PAWN_WEIGHTS", l.isolated_pawn_offset,      2, " // [MG, EG]"),
        ];

        for &(name, offset, count, comment) in bands {
            if params.len() > offset {
                write!(w, "    {name:<22}= [").ok();
                write_weight_array(w, offset, count, values, params, initial);
                writeln!(w, "],{comment}").ok();
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

/// Wraps text in the advantage-win green if `changed` is true and `initial` is `Some`.
fn highlight(s: &str, changed: bool, initial: Option<&[f64]>) -> String {
    if initial.is_some() && changed {
        format!("{}{s}\x1b[0m", color::ansi_fg((100, 200, 120)))
    } else {
        s.to_string()
    }
}
