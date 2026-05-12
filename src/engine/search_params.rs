//! Search-specific tunable parameters.
//!
//! One line per parameter generates: a struct field, a `pub fn` accessor,
//! and a `linkme` distributed slice entry for CMA-ES iteration.
//!
//! Callers `use crate::engine::search_params::*;` for zero-prefix access
//! like `rfp_margin()`.

use std::sync::atomic::{AtomicI32, Ordering};

use linkme::distributed_slice;

/// Metadata + live value pointer for one tunable parameter.
#[derive(Debug)]
pub struct ParamDef {
    pub name: &'static str,
    pub value: &'static AtomicI32,
    pub min: f64,
    pub max: f64,
    pub step: f64,
    pub default: f64,
}

impl ParamDef {
    /// Normalize a raw value to `[0, 1]`.
    #[inline]
    pub fn normalize(&self, raw: f64) -> f64 {
        let range = self.max - self.min;
        if range > 1e-9 { ((raw - self.min) / range).clamp(0.0, 1.0) } else { 0.5 }
    }

    /// Denormalize a `[0, 1]` value back to the parameter's range.
    #[inline]
    pub fn denormalize(&self, normalized: f64) -> f64 {
        let val = normalized.mul_add(self.max - self.min, self.min);
        if self.step > 1e-9 { (val / self.step).round() * self.step } else { val }
    }

    #[inline]
    pub fn read(&self) -> i32 {
        self.value.load(Ordering::Relaxed)
    }

    #[inline]
    pub fn write(&self, v: i32) {
        self.value.store(v, Ordering::Relaxed);
    }
}

macro_rules! search_params {
    (
        pub struct $struct_name:ident {
            $(
                $(#[doc = $doc:expr])*
                pub $field:ident: $type:ty = $default:literal, min $min:literal, max $max:literal, step $step:literal
            ),* $(,)?
        }
    ) => {
        #[derive(Clone, Copy, Debug)]
        pub struct $struct_name {
            $(
                $(#[doc = $doc])*
                pub $field: $type,
            )*
        }

        impl Default for $struct_name {
            fn default() -> Self {
                Self { $($field: $default,)* }
            }
        }

        $(
            paste::paste! {
                #[allow(non_upper_case_globals)]
                static [<__VAL_ $field>]: AtomicI32 = AtomicI32::new($default as i32);

                $(#[doc = $doc])*
                #[inline(always)]
                pub fn $field() -> i32 {
                    [<__VAL_ $field>].load(Ordering::Relaxed)
                }

                #[allow(non_upper_case_globals)]
                #[distributed_slice(PARAM_DEFS)]
                static [<__REG_ $field>]: ParamDef = ParamDef {
                    name: stringify!($field),
                    value: &[<__VAL_ $field>],
                    min: $min as f64,
                    max: $max as f64,
                    step: $step as f64,
                    default: $default as f64,
                };
            }
        )*

        #[distributed_slice]
        pub static PARAM_DEFS: [ParamDef] = [..];

        pub fn flush_into(sp: &$struct_name) {
            $( paste::paste! { [<__VAL_ $field>].store(sp.$field as i32, Ordering::Relaxed); } )*
        }

        pub fn collect_as() -> $struct_name {
            let mut sp = $struct_name::default();
            $( paste::paste! { sp.$field = [<__VAL_ $field>].load(Ordering::Relaxed); } )*
            sp
        }

        impl $struct_name {
            pub fn from_normalized(values: &[f64]) -> Self {
                for (def, &norm) in PARAM_DEFS.iter().zip(values) {
                    def.write(def.denormalize(norm).round() as i32);
                }
                collect_as()
            }

            pub fn to_normalized() -> Vec<f64> {
                PARAM_DEFS.iter().map(|def| def.normalize(def.read() as f64)).collect()
            }
        }
    };
}

search_params! {
    pub struct SearchParams {
        pub lazy_eval_margin:      i32 = 160,   min 60,   max 200,   step 2,
        pub lazy_eval_divisor:     i32 = 32,    min 8,    max 128,   step 1,
        pub mtg_opening:           i32 = 80,    min 40,   max 300,   step 2,
        pub mtg_endgame:           i32 = 40,    min 10,   max 100,   step 2,
        pub vol_pawn:              i32 = 5,     min 0,    max 15,    step 1,
        pub vol_knight:            i32 = 10,    min 0,    max 15,    step 1,
        pub vol_bishop:            i32 = 9,     min 0,    max 10,    step 1,
        pub vol_rook:              i32 = 5,     min 0,    max 10,    step 1,
        pub vol_queen:             i32 = 13,    min 0,    max 15,    step 1,
        pub vol_king:              i32 = 0,     min 0,    max 5,     step 1,
        pub nmp_base_r:            i32 = 3,     min 2,    max 5,     step 1,
        pub nmp_depth_divisor:     i32 = 3,     min 2,    max 6,     step 1,
        pub nmp_eval_divisor:      i32 = 200,   min 50,   max 400,   step 10,
        pub nmp_eval_max:          i32 = 3,     min 1,    max 5,     step 1,
        pub nmp_verif_min_depth:   i32 = 14,    min 6,    max 20,    step 1,
        pub lmr_base:              i32 = 100,   min 50,   max 200,   step 5,
        pub lmr_divisor:           i32 = 225,   min 150,  max 350,   step 5,
        pub asp_initial:           i32 = 15,    min 8,    max 50,    step 1,
        pub rfp_margin:            i32 = 45,    min 30,   max 150,   step 5,
        pub rfp_depth:             i32 = 8,     min 3,    max 10,    step 1,
        pub lmp_base:              i32 = 2,     min 1,    max 6,     step 1,
        pub lmp_depth:             i32 = 5,     min 2,    max 8,     step 1,
        pub hist_prune_depth:      i32 = 6,     min 2,    max 10,    step 1,
        pub hist_prune_margin:     i32 = 3000,  min 500,  max 4000,  step 100,
        pub fp_margin:             i32 = 100,   min 50,   max 300,   step 5,
        pub fp_depth:              i32 = 6,     min 1,    max 8,     step 1,
        pub razoring_margin:       i32 = 300,   min 100,  max 600,   step 10,
        pub razoring_depth:        i32 = 3,     min 1,    max 5,     step 1,
        pub delta_margin:          i32 = 200,   min 50,   max 400,   step 10,
        pub qs_recapture_ply:      i32 = 4,     min 2,    max 10,    step 1,
        pub see_capture_margin:    i32 = 80,    min 20,   max 200,   step 2,
        pub see_quiet_margin:      i32 = 60,    min 15,   max 150,   step 2,
        pub mvvlva_ep:             i32 = 100,   min 50,   max 200,   step 10,
        pub mvvlva_v_pawn:         i32 = 100,   min 50,   max 200,   step 2,
        pub mvvlva_v_knight:       i32 = 300,   min 100,  max 600,   step 5,
        pub mvvlva_v_bishop:       i32 = 300,   min 100,  max 600,   step 5,
        pub mvvlva_v_rook:         i32 = 500,   min 200,  max 1000,  step 5,
        pub mvvlva_v_queen:        i32 = 900,   min 400,  max 1800,  step 10,
        pub mvvlva_v_king:         i32 = 10000, min 5000, max 20000, step 100,
        pub mvvlva_a_pawn:         i32 = 10,    min 0,    max 50,    step 1,
        pub mvvlva_a_knight:       i32 = 30,    min 0,    max 100,   step 1,
        pub mvvlva_a_bishop:       i32 = 30,    min 0,    max 100,   step 1,
        pub mvvlva_a_rook:         i32 = 50,    min 0,    max 200,   step 2,
        pub mvvlva_a_queen:        i32 = 90,    min 0,    max 300,   step 2,
        pub mvvlva_a_king:         i32 = 0,     min 0,    max 0,     step 0,
        pub capt_hist_divisor:     i32 = 32,    min 8,    max 256,   step 8,
        pub score_drop_depth:      i32 = 5,     min 3,    max 10,    step 1,
        pub score_factor_scale:    i32 = 100,   min 40,   max 200,   step 4,
        pub bm_stab_depth:         i32 = 5,     min 3,    max 10,    step 1,
        pub bm_stab_base:          i32 = 270,   min 150,  max 400,   step 5,
        pub bm_stab_scale:         i32 = 220,   min 100,  max 350,   step 5,
        pub bm_stab_floor:         i32 = 56,    min 30,   max 96,    step 2,
        pub np_corr_weight:        i32 = 128,   min 0,    max 512,   step 8,
    }
}
