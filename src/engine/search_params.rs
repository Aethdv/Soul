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
//! Search reads each value from its own [`SearchParams`]. A tunable entry is
//! advertised as a UCI spin option and settable by name; a frozen one is neither,
//! and its bounds are the range it would be tuned over.

/// Static metadata for one tunable parameter.
#[derive(Debug)]
pub(crate) struct ParamDef {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    step: f64,
    pub default: f64,
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

/// The tunables as UCI spin options, one a line.
pub fn uci_options() -> String {
    tunable_param_defs()
        .iter()
        .map(|p| format!("option name {} type spin default {} min {} max {}\n", p.name, p.default, p.min, p.max))
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

            pub fn get(&self, name: &str) -> Option<i32> {
                $(
                    if !$frozen && name == stringify!($field) {
                        return Some(self.$field);
                    }
                )*
                None
            }

            pub fn set(&mut self, name: &str, value: i32) -> bool {
                $(
                    if !$frozen && name == stringify!($field) {
                        self.$field = value.clamp($min as i32, $max as i32);
                        return true;
                    }
                )*
                false
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

pub(crate) fn tunable_param_defs() -> Vec<&'static ParamDef> { PARAM_DEFS.iter().filter(|p| !p.frozen).collect() }
pub(crate) fn frozen_param_defs() -> Vec<&'static ParamDef> { PARAM_DEFS.iter().filter(|p| p.frozen).collect() }

search_params! {
    pub struct SearchParams {
        //            default  min   max  step
        T (asp_depth,       5,   1,    6),
        T (asp_initial,    18,   1,   32),
        T (asp_widen_div,   4,   1,   14),

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
        NT(score_swing_scale,  100, 10, 160),

        NT(effort_depth,     5),
        NT(effort_base,    270),
        NT(effort_scale,   220),
        NT(effort_floor,    56),
        NT(bm_stab_base,   125),
        NT(bm_stab_scale,    4),
        NT(bm_stab_floor,   80),
        NT(bm_inst_scale,  220,  0, 340),
        NT(bm_inst_decay,   50,  0, 100),  // per-iteration decay, ·100

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

        //                  default  min   max  step
        T (see_value_pawn,      93,   50),
        T (see_value_knight,   381,  250),
        T (see_value_bishop,   431,  250),
        T (see_value_rook,     541,  400),
        T (see_value_queen,   1119,  800),

        //                  default  min   max  step
        T (good_capture_margin, 182,   0,  300),

        //             default  min  max  step
        T (qs_recapture_ply, 4,   2),

        //             default   min  max  step
        T (qs_see_margin,   4,  -100, 100),

        //            default  min  max  step
        T (delta_margin,  279,  50),

        //                 default min  max  step
        T (see_capture_margin,  50,  1),
        T (see_quiet_margin,    37,  1),

        //              default  min  max  step
        T (razoring_depth,    1,   1,    8),
        T (razoring_margin, 254,  50),

        //              default  min  max  step
        T (rfp_depth,         9,   4,   20),
        T (rfp_margin,       45,  15),
        T (rfp_base_margin,  32),
        T (rfp_quad_margin,   2),

        //                  default  min  max  step
        T (probcut_depth_min,    10,   3,   10),
        T (probcut_margin,      227,  50),
        T (probcut_reduction,     5,   1,    8),

        //                  default min   max  step
        T (nmp_base_r,            4,  1,    6),
        T (nmp_depth_divisor,     5,  1,   14),
        T (nmp_eval_divisor,    237,  1,  320),
        T (nmp_eval_max,          4,  1,    6),
        NT(nmp_ply_offset,        1),
        T (nmp_verif_min_depth,  16,  6,   24),

        //                  default  min   max  step
        T (singext_min_depth,     5,   4,   16),
        T (singext_margin,        1,   1),
        T (singext_tt_depth,      3,   1,    6),
        T (singext_depth_div,     4,   1,    4),

        //           default min  max  step
        T (iir_depth,      5,  2,  8),
        T (iir_reduction,  1,  1,  3),

        //        default min  max  step
        T (fp_depth,    6,  2, 12),
        T (fp_margin, 111, 25),

        //        default min  max  step
        T (lmp_depth,   5,  2, 10),
        T (lmp_base,    2,  1,  6),
        T (lmp_scale, 128, 25, 250),  // ·100

        //                 default   min  max  step
        T (hist_prune_depth,     8,    2,  12),
        T (hist_prune_margin, 2117,  100),

        //                   default   min   max  step
        T (lmr_min_depth,          1,    1,    6),
        T (lmr_base,              87,   10),
        T (lmr_divisor,          215,    1,  350),
        T (lmr_hist_div,           9,    1,   24),
        T (killer_lmr_bonus,    1197,   64),
        T (check_lmr_bonus,        1,    1,    3),
        T (threat_lmr_bonus,    1158,    0),
        T (fhc_lmr_malus,        604,    0),
        T (fhc_cutoff_min,         1,    0,    6),
        T (critical_lmr_bonus,   165,    0,  512),
        T (critical_lmr_cap,       3,    1,   24),
        NT(lmr_retained,           1),

        //                  default   min   max  step
        T (capt_hist_divisor,    39,    4,   64),
        T (hist_bonus_mult,      12,    1,   12),
        T (hist_bonus_cap,     2376,  400, 3200),

        //                    default    min    max  step
        T (quiet_hist_cap,       8265,  8192, 32000),
        T (butterfly_hist_cap,  10303,  8192, 32000),
        T (cont_hist_cap,       11324,  8192, 32000),
        T (capt_hist_cap,       12190,  8192, 32000),

        //              default min  max  step
        T (tt_age_factor,     2,   0,  16),

        //                default min  max  step
        T (minor_corr_weight, 114,  8),
        T (major_corr_weight, 144,  8),
        T (corr_weight_div,     1,  1,  16),
        T (corr_weight_max,    32,  4, 128),  // ·256
    }
}

#[cfg(test)]
mod tests {
    use super::{PARAM_DEFS, SearchParams, spsa_table, tunable_param_defs, uci_options};

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

    #[test]
    fn every_tunable_is_advertised_and_settable() {
        let options = uci_options();
        let advertised: Vec<&str> = options
            .lines()
            .map(|line| line.split_whitespace().nth(2).expect("option name <name>"))
            .collect();
        let tunable: Vec<&str> = tunable_param_defs().iter().map(|p| p.name).collect();
        assert_eq!(advertised, tunable);

        let mut sp = SearchParams::default();
        for def in tunable_param_defs() {
            assert!(sp.set(def.name, def.max as i32), "{} refused a write", def.name);
            assert_eq!(sp.get(def.name), Some(def.max as i32), "{} did not take it", def.name);
            assert!(sp.set(def.name, def.min as i32 - 1_000_000), "{} refused a write", def.name);
            assert_eq!(sp.get(def.name), Some(def.min as i32), "{} was not clamped", def.name);
        }
    }

    #[test]
    fn a_frozen_entry_is_neither_advertised_nor_settable() {
        let mut sp = SearchParams::default();
        for def in PARAM_DEFS.iter().filter(|p| p.frozen) {
            assert!(!sp.set(def.name, def.max as i32), "{} is frozen and took a write", def.name);
            assert_eq!(sp.get(def.name), None, "{} is frozen and read back", def.name);
        }
        assert!(!sp.set("no_such_param", 1));
    }
}
