//! Search-specific tunable parameters.
//!
//! Each entry generates a struct field and a `PARAM_DEFS` entry in declaration
//! order, which is the order `./soul spsa` prints them in.
//!
//! Entry forms:
//!
//!   `T (name, default)`                 tune; auto-derive bounds and step
//!   `T (name, default, min)`            tune; explicit min, auto max and step
//!   `T (name, default, min, max)`       tune; explicit bounds, auto step
//!   `NT(name, default)`                 frozen; auto-derive still applies
//!   `NT(name, default, min, max)`       frozen; bounds kept for when it is not
//!
//! Auto-derive: `min = 0`, `max = default + default/2 + 10`, `step = max/20` (≥ 1).
//!
//! Search reads each value from its own [`SearchParams`]. A frozen entry stays out
//! of the SPSA table, and its bounds are the range it would be tuned over.

/// Static metadata for one tunable parameter.
#[derive(Debug)]
struct ParamDef {
    name: &'static str,
    min: f64,
    max: f64,
    step: f64,
    default: f64,
    frozen: bool,
}

// Default-derived upper bound; symmetric around 1.5× magnitude with a floor.
const fn auto_max(default: i32) -> i32 {
    let abs_d = default.abs();
    abs_d + 10 + abs_d / 2
}

// Default-derived step. Floor of 1 prevents zero-step on tiny defaults.
const fn auto_step(max: i32) -> i32 { (max / 20).max(1) }

/// The tunables as an SPSA table, one parameter a line:
/// `name, int, value, min, max, c_end, r_end`.
pub fn spsa_table() -> String {
    tunable_param_defs()
        .iter()
        .map(|p| format!("{}, int, {}, {}, {}, {}, {SPSA_R_END}\n", p.name, p.default, p.min, p.max, p.step))
        .collect()
}

macro_rules! search_params {
    ( pub struct $name:ident { $($body:tt)* } ) => {
        search_params!(@collect [$name] [] $($body)*);
    };

    // Final emit
    (@collect [$name:ident] [$($entries:tt)*]) => {
        search_params!(@emit $name [$($entries)*]);
    };

    // T(name, default): auto bounds + step
    (@collect [$name:ident] [$($entries:tt)*] T($field:ident, $def:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, 0, $crate::engine::search_params::auto_max($def),
             $crate::engine::search_params::auto_step($crate::engine::search_params::auto_max($def)), false)
        ] $($rest)*);
    };

    // T(name, default, min): auto max + step
    (@collect [$name:ident] [$($entries:tt)*] T($field:ident, $def:literal, $min:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $crate::engine::search_params::auto_max($def),
             $crate::engine::search_params::auto_step($crate::engine::search_params::auto_max($def) - $min), false)
        ] $($rest)*);
    };

    // T(name, default, min, max): auto step
    (@collect [$name:ident] [$($entries:tt)*] T($field:ident, $def:literal, $min:literal, $max:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $max, $crate::engine::search_params::auto_step($max - $min), false)
        ] $($rest)*);
    };

    // NT(name, default): frozen, auto bounds
    (@collect [$name:ident] [$($entries:tt)*] NT($field:ident, $def:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, 0, $crate::engine::search_params::auto_max($def),
             $crate::engine::search_params::auto_step($crate::engine::search_params::auto_max($def)), true)
        ] $($rest)*);
    };

    // NT(name, default, min, max)
    (@collect [$name:ident] [$($entries:tt)*] NT($field:ident, $def:literal, $min:literal, $max:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $max, $crate::engine::search_params::auto_step($max - $min), true)
        ] $($rest)*);
    };

    (@emit $name:ident [$( ($field:ident, $def:literal, $min:expr, $max:expr, $step:expr, $frozen:expr) )*]) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            $( pub $field: i32, )*
        }

        impl $name {
            /// The hand-picked defaults, usable in const context so a table built
            /// from these values can still be a `const`.
            pub const fn new() -> Self {
                Self { $( $field: $def, )* }
            }
        }

        impl Default for $name {
            fn default() -> Self {
                Self::new()
            }
        }

        static PARAM_DEFS: &[ParamDef] = &[
            $(
                ParamDef {
                    name: stringify!($field),
                    min: $min as f64,
                    max: $max as f64,
                    step: $step as f64,
                    default: $def as f64,
                    frozen: $frozen,
                }
            ),*
        ];

    };
}

/// Terminal learning rate for the SPSA table.
const SPSA_R_END: f64 = 0.002;

fn tunable_param_defs() -> Vec<&'static ParamDef> { PARAM_DEFS.iter().filter(|p| !p.frozen).collect() }

search_params! {
    pub struct SearchParams {
        //            default  min   max  step
        NT(asp_depth,       4,   1,    6),
        T (asp_initial,    15,   1,   32),
        T (asp_widen_div,   3,   1,   14),

        //                default   min  max  step
        NT(qs_lazy_margin,  160,  140, 300),
        NT(qs_lazy_divisor,  16,    1,  64),

        NT(vol_pawn,     5),
        NT(vol_knight,  10),
        NT(vol_bishop,   9),
        NT(vol_rook,     5),
        NT(vol_queen,   13),
        NT(vol_king,     0),

        NT(mtg_opening,      80),
        NT(mtg_endgame,      40),
        NT(tm_iter_scale,   200),
        NT(tm_single_root,    5),

        //  TM budget, values ·100 (tm_sd_ramp ·1000)
        NT(tm_hard_mult,       500),
        NT(tm_hard_clock_cap,   95),
        NT(tm_sd_base,          50),
        NT(tm_sd_ramp,           1),
        NT(tm_sd_cap,           80),
        NT(tm_soft_inc,         80),

        //                 default min  max  step
        NT(score_drop_depth,     5),
        T (score_swing_scale,  100, 10),

        NT(bm_stab_depth,    5),
        NT(bm_stab_base,   270),
        NT(bm_stab_scale,  220),
        NT(bm_stab_floor,   56),
        T (bm_inst_scale,  220,  0),

        //                default  min   max  step
        NT(mvvlva_ep,         100,  60,  150),
        NT(mvvlva_v_pawn,     100,  60,  150),
        NT(mvvlva_v_knight,   300, 200,  350),
        NT(mvvlva_v_bishop,   300, 200,  350),
        NT(mvvlva_v_rook,     500, 400,  600),
        NT(mvvlva_v_queen,    900, 700, 1100),
        NT(mvvlva_v_king,   10000),
        NT(mvvlva_a_pawn,      10,   0,   20),
        NT(mvvlva_a_knight,    30,  20,   60),
        NT(mvvlva_a_bishop,    30,  20,   60),
        NT(mvvlva_a_rook,      50,  40,  100),
        NT(mvvlva_a_queen,     90,  70,  150),
        NT(mvvlva_a_king,       0),


        T (see_value_pawn,     92),
        T (see_value_knight,  373),
        T (see_value_bishop,  372),
        T (see_value_rook,    568),
        T (see_value_queen,  1160),

        //                   default   min    max   step
        T (good_capture_margin,  200,    0,   300),
        T (bad_quiet_threshold, 3000,    0, 15000),
        T (bad_quiet_mul,       2500,  500,  5000),

        //             default  min  max  step
        T (qs_recapture_ply, 4,   2),

        //            default  min  max  step
        T (delta_margin,  200,  50),

        //                 default min  max  step
        T (see_capture_margin,  80,  1),
        T (see_quiet_margin,    60,  1),

        //              default  min  max  step
        NT(razoring_depth,    3),
        T (razoring_margin, 300,  50),

        //              default  min  max  step
        NT(rfp_depth,        12),
        T (rfp_margin,       40,  15),
        T (rfp_base_margin,  35),
        T (rfp_quad_margin,   3),

        //                default  min  max  step
        NT(probcut_depth_min,   5),
        T (probcut_margin,    200,  50),

        //                  default min   max  step
        T (nmp_base_r,            3,  1,    6),
        T (nmp_depth_divisor,     3,  1,   14),
        T (nmp_eval_divisor,    200,  1,  320),
        T (nmp_eval_max,          3,  1,    6),
        NT(nmp_ply_offset,        1),
        NT(nmp_verif_min_depth,  14),

        //                 default  min   max  step
        NT(singext_min_depth,    9),
        T (singext_margin,       2,   1),
        NT(singext_tt_depth,     3),

        //           default min  max  step
        NT(iir_depth,      4),
        T (iir_reduction,  1,  1,  3),

        //        default min  max  step
        NT(fp_depth,    6),
        T (fp_margin, 100, 25),

        //        default min  max  step
        NT(lmp_depth,   5),
        T (lmp_base,    2,  1,  6),

        //                 default   min  max  step
        NT(hist_prune_depth,     6),
        T (hist_prune_margin, 3000,  100),

        //                default   min  max  step
        T (lmr_base,          100,  10),
        T (lmr_divisor,       225,   1,  350),
        T (lmr_hist_div,        8,   1,   24),
        T (killer_lmr_bonus, 1024,  64),
        T (check_lmr_bonus,     1,   1,    3),
        T (threat_lmr_bonus, 1024,   0),
        T (fhc_lmr_malus,     512,   0),
        NT(lmr_retained,        1),

        NT(capt_hist_divisor,  32),
        NT(hist_bonus_mult,     4),
        NT(hist_bonus_cap,   1600),

        //                    default    min    max  step
        NT(quiet_hist_cap,      16384,  8192, 32000),
        NT(butterfly_hist_cap,  16384,  8192, 32000),
        NT(cont_hist_cap,       16384,  8192, 32000),
        NT(capt_hist_cap,       16384,  8192, 32000),

        //                default min  max  step
        T (minor_corr_weight, 128,  8),
        T (major_corr_weight, 128,  8),
    }
}

#[cfg(test)]
mod tests {
    use super::spsa_table;

    #[test]
    fn spsa_table_is_well_formed() {
        let table = spsa_table();
        let mut seen: Vec<&str> = Vec::new();

        for line in table.lines() {
            let fields: Vec<&str> = line.split(',').map(str::trim).collect();
            let [name, kind, value, min, max, c_end, r_end] = fields[..] else {
                panic!("expected seven fields: {line}");
            };

            assert!(!name.is_empty() && !name.contains(char::is_whitespace) && !name.contains('='), "bad name: {line}");
            assert!(!seen.contains(&name), "duplicate: {name}");
            seen.push(name);
            assert_eq!(kind, "int", "{line}");

            let num = |field: &str| field.parse::<f64>().expect("a number");
            let (value, min, max) = (num(value), num(min), num(max));
            assert!(min <= max, "{line}");
            assert!((min..=max).contains(&value), "{name} defaults outside its bounds: {line}");
            assert!(num(c_end) > 0.0, "{name} has a zero probe width: {line}");
            assert!(num(r_end) >= 0.0, "{line}");
        }
        assert!(!seen.is_empty());
    }
}
