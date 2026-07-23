use std::{
    fs::File,
    io::{self, BufWriter, Write},
};

use soul::{color, core::psqt, engine::eval_params::Tunable};

use crate::evaltune::palette;

/// The `define_weight_params!` paste block: `(name, offset, count, comment)`
/// per band, in layout order. A hand-maintained mirror of `LAYOUT`, like
/// `gradient.rs` and `register_terms!`.
#[rustfmt::skip]
const WEIGHT_BANDS: &[(&str, usize, usize, &str)] = {
    let l = psqt::LAYOUT;
    &[
        ("PHASE_WEIGHTS",             l.weight_offset,             6, " // [P, N, B, R, Q, K]"),
        ("ATTACKER_WEIGHTS",          l.attacker_offset,           6, " // [0, 1, 2, 3, 4, 5] attackers × weak"),
        ("KING_SAFETY_WEIGHTS",       l.king_safety_offset,        3, " // [Pawn Shield, Ortho Exp, Diag Exp]"),
        ("XRAY_WEIGHTS",              l.xray_offset,               1, " // [Ortho King]"),
        ("BISHOP_PAIR_WEIGHTS",       l.bishop_pair_offset,        2, " // [MG, EG]"),
        ("ROOK_OPEN_WEIGHTS",         l.rook_open_offset,          2, " // [MG, EG]"),
        ("PASSED_PAWN_MG",            l.passed_pawn_mg_offset,     6, " // by relative rank 1-6"),
        ("PASSED_PAWN_EG",            l.passed_pawn_eg_offset,     6, " // by relative rank 1-6"),
        ("ENEMY_KING_DIST_MG",        l.enemy_king_dist_mg_offset, 6, " // enemy king→passer dist, 7 clamps to 6"),
        ("ENEMY_KING_DIST_EG",        l.enemy_king_dist_eg_offset, 6, " // enemy king→passer dist, 7 clamps to 6"),
        ("DOUBLED_PAWN_WEIGHTS",      l.doubled_pawn_offset,       2, " // [MG, EG]"),
        ("ISOLATED_PAWN_WEIGHTS",     l.isolated_pawn_offset,      2, " // [MG, EG]"),
        ("PHALANX_MG",                l.phalanx_mg_offset,         6, " // by relative rank 2-7"),
        ("PHALANX_EG",                l.phalanx_eg_offset,         6, " // by relative rank 2-7"),
        ("DEFENDED_PAWN_MG",          l.defended_pawn_mg_offset,   6, " // by relative rank 2-7 (rank 2 unreachable)"),
        ("DEFENDED_PAWN_EG",          l.defended_pawn_eg_offset,   6, " // by relative rank 2-7 (rank 2 unreachable)"),
        ("BACKWARD_PAWN_WEIGHTS",     l.backward_pawn_offset,      2, " // [MG, EG]"),
        ("TEMPO_WEIGHTS",             l.tempo_offset,              2, " // [MG, EG], side-to-move initiative"),
        ("MINOR_BEHIND_PAWN_WEIGHTS", l.minor_behind_pawn_offset,  2, " // [MG, EG]"),
    ]
};

// The bands must tile weight_offset..total exactly, or a new LAYOUT field
// would silently vanish from the paste block until someone diffed it.
const _: () = {
    let mut expected = psqt::LAYOUT.weight_offset;
    let mut i = 0;

    while i < WEIGHT_BANDS.len() {
        assert!(WEIGHT_BANDS[i].1 == expected, "gap or overlap in WEIGHT_BANDS");
        expected = WEIGHT_BANDS[i].1 + WEIGHT_BANDS[i].2;

        i += 1;
    }

    assert!(expected == psqt::LAYOUT.total, "WEIGHT_BANDS stops short of LAYOUT's end");
};

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

        for &(name, offset, count, comment) in WEIGHT_BANDS {
            if params.len() > offset {
                write!(w, "    {name:<26}= [").ok();
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
        format!("{}{s}{}", color::ansi_fg((100, 200, 120)), palette::RESET)
    } else {
        s.to_string()
    }
}
