//! Search-specific tunable parameters.
//!
//! Each entry generates a struct field and a `PARAM_DEFS` slice entry in
//! declaration order; the tuner shows and iterates params in the order they
//! appear in this file.
//!
//! Entry forms:
//!
//!   `T (name, default)`                      tune; auto-derive bounds + step
//!   `T (name, default, min)`                 tune; explicit min, auto max + step
//!   `T (name, default, min, max)`            tune; explicit bounds, auto step
//!   `T (name, default, min, max, step)`      tune; fully explicit
//!   `NT(name, default)`                      frozen; auto-derive still applies but tuner skips
//!
//! Auto-derive: `min = 0`, `max = default + default/2 + 10`, `step = max/20` (≥ 1).
//!
//! Search reads each value from its own [`SearchParams`]; the tuner reads
//! bounds and the tunable set from [`PARAM_DEFS`].

/// Static metadata for one tunable parameter. The live value rides in a
/// [`SearchParams`], one per searcher; this is only the bounds the tuner reasons over.
#[derive(Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
    /// Tuner skips frozen entries when assembling the search vector.
    pub frozen: bool,
}

impl ParamDef {
    /// Normalize a raw value to `[0, 1]`.
    #[inline]
    pub fn normalize(&self, raw: f64) -> f64 {
        let range = self.max - self.min;
        if range > 1e-9 { ((raw - self.min) / range).clamp(0.0, 1.0) } else { 0.5 }
    }

    /// Denormalize a `[0, 1]` value back to the parameter's range.
    /// Snap grid is anchored at `min` so both endpoints round-trip exactly,
    /// then clamped to `[min, max]` against any drift from the rounding.
    #[inline]
    pub fn denormalize(&self, normalized: f64) -> f64 {
        let val = normalized.mul_add(self.max - self.min, self.min);
        let snapped = if self.step > 1e-9 { self.min + ((val - self.min) / self.step).round() * self.step } else { val };
        snapped.clamp(self.min, self.max)
    }
}

/// Default-derived upper bound; symmetric around 1.5× magnitude with a floor.
pub const fn auto_max(default: i32) -> i32 {
    let abs_d = if default < 0 { -default } else { default };
    abs_d + 10 + abs_d / 2
}

/// Default-derived step. Floor of 1 prevents zero-step on tiny defaults.
pub const fn auto_step(max: i32) -> i32 {
    let s = max / 20;
    if s < 1 { 1 } else { s }
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

    // NT(name, default, min): auto max + step
    (@collect [$name:ident] [$($entries:tt)*] NT($field:ident, $def:literal, $min:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $crate::engine::search_params::auto_max($def),
             $crate::engine::search_params::auto_step($crate::engine::search_params::auto_max($def) - $min), true)
        ] $($rest)*);
    };

    // T(name, default, min, max): auto step
    (@collect [$name:ident] [$($entries:tt)*] T($field:ident, $def:literal, $min:literal, $max:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $max, $crate::engine::search_params::auto_step($max - $min), false)
        ] $($rest)*);
    };

    // T(name, default, min, max, step): fully explicit
    (@collect [$name:ident] [$($entries:tt)*] T($field:ident, $def:literal, $min:literal, $max:literal, $step:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $max, $step, false)
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

    // NT(name, default, min, max, step)
    (@collect [$name:ident] [$($entries:tt)*] NT($field:ident, $def:literal, $min:literal, $max:literal, $step:literal) , $($rest:tt)*) => {
        search_params!(@collect [$name] [$($entries)*
            ($field, $def, $min, $max, $step, true)
        ] $($rest)*);
    };

    (@emit $name:ident [$( ($field:ident, $def:literal, $min:expr, $max:expr, $step:expr, $frozen:expr) )*]) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $name {
            $( pub $field: i32, )*
        }

        impl Default for $name {
            fn default() -> Self {
                Self { $( $field: $def, )* }
            }
        }

        /// All registered params in source-declaration order, frozen + tunable mixed.
        /// Use [`tunable_param_defs`] when the tuner only needs the active set.
        pub static PARAM_DEFS: &[ParamDef] = &[
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

        impl $name {
            /// Build from a normalized tunable vector: frozen params keep their
            /// default, tunables take `values` in declaration order.
            pub fn from_normalized(values: &[f64]) -> Self {
                let defs = tunable_param_defs();
                let mut sp = Self::default();
                let mut i = 0;
                $(
                    if !$frozen {
                        sp.$field = defs[i].denormalize(values[i]).round() as i32;
                        i += 1;
                    }
                )*
                let _ = i;
                sp
            }

            /// Normalize this struct's tunable params back to `[0, 1]`, declaration order.
            pub fn to_normalized(&self) -> Vec<f64> {
                let defs = tunable_param_defs();
                let mut out = Vec::with_capacity(defs.len());
                let mut i = 0;
                $(
                    if !$frozen {
                        out.push(defs[i].normalize(self.$field as f64));
                        i += 1;
                    }
                )*
                let _ = i;
                out
            }
        }
    };
}

/// Active tunable params (frozen entries filtered out) in source-declaration order.
/// Tuner uses this as the canonical view; `PARAM_DEFS` is the unfiltered list.
pub fn tunable_param_defs() -> Vec<&'static ParamDef> {
    PARAM_DEFS.iter().filter(|p| !p.frozen).collect()
}

search_params! {
    pub struct SearchParams {
        //            default   min  max  step
        NT(asp_depth,       4,   1,    6),
        T (asp_initial,    15,   1,   32),
        T (asp_widen_div,   3,   1,   14),

        //                default   min  max  step
        T (lazy_eval_margin,  160,  140, 300),
        T (lazy_eval_divisor,  32,    1,  64),

        //         default   min  max  step
        NT(vol_pawn,     5),
        NT(vol_knight,  10),
        NT(vol_bishop,   9),
        NT(vol_rook,     5),
        NT(vol_queen,   13),
        NT(vol_king,     0),

        //              default   min  max  step
        NT(mtg_opening,      80),
        NT(mtg_endgame,      40),
        NT(tm_iter_scale,   200),
        NT(tm_single_root,    5),

        //                 default min  max  step
        NT(score_drop_depth,     5),
        T (score_swing_scale,  100, 10),

        //             default   min  max  step
        NT(bm_stab_depth,    5),
        NT(bm_stab_base,   270),
        NT(bm_stab_scale,  220),
        NT(bm_stab_floor,   56),

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

        //                 default  min   max  step
        T (good_capture_margin, 200,  0,  300),

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

        //                default   min  max  step
        NT(capt_hist_divisor,  32),
        NT(hist_bonus_mult,     4),
        NT(hist_bonus_cap,   1600),

        //                default min  max  step
        T (minor_corr_weight, 128,  8),
        T (major_corr_weight, 128,  8),
    }
}
