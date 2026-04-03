//! Search-specific tunable parameters.
//!
//! Defines the heuristic thresholds and margins (e.g., MVV-LVA bias, lazy evaluation bounds)
//! used to control the search tree's pruning and ordering behavior.

/// Definition of a single tunable parameter.
#[derive(Debug, Clone, Copy)]
pub struct ParamDef {
    pub name:    &'static str,
    pub min:     f64,
    pub max:     f64,
    pub step:    f64,
    pub default: f64,
}

impl ParamDef {
    /// Normalize a raw value to the `[0, 1]` interval.
    #[inline]
    pub fn normalize(&self, raw: f64) -> f64 {
        let range = self.max - self.min;
        if range > 1e-9 {
            ((raw - self.min) / range).clamp(0.0, 1.0)
        } else {
            0.5 // Constant parameter: center of the normalized hypercube
        }
    }

    /// Denormalize a `[0, 1]` value back to the parameter's range.
    #[inline]
    pub fn denormalize(&self, normalized: f64) -> f64 {
        let val = normalized.mul_add(self.max - self.min, self.min);
        // Snap to step
        if self.step > 1e-9 {
            (val / self.step).round() * self.step
        } else {
            val
        }
    }
}

/// Macro to generate the `SearchParams` struct and metadata automatically.
/// This prevents desync between the struct definition and the tuning array.
macro_rules! search_params {
    (
        pub struct $struct_name:ident {
            $(
                $(#[doc = $doc:expr])*
                pub $field:ident: $type:ty {
                    default: $default:expr,
                    min:     $min:expr,
                    max:     $max:expr,
                    step:    $step:expr,
                }
            ),* $(,)?
        }
    ) => {
        /// Runtime search parameters.
        #[derive(Clone, Copy, Debug)]
        pub struct $struct_name {
            $(
                $(#[doc = $doc])*
                pub $field: $type,
            )*
        }

        impl Default for $struct_name {
            fn default() -> Self {
                Self {
                    $(
                        $field: $default,
                    )*
                }
            }
        }

        impl $struct_name {
            /// Create struct from a vector of normalized `[0, 1]` values.
            ///
            /// Excess parameters (missing from `values`) default to 0.5 (midpoint).
            /// In debug builds this panics to surface checkpoint version mismatches early.
            pub fn from_normalized(values: &[f64]) -> Self {
                let mut iter = values.iter();
                Self {
                    $(
                        $field: {
                            let norm = iter.next().copied().unwrap_or_else(|| {
                                debug_assert!(false, "Parameter count mismatch in from_normalized: missing value for {}", stringify!($field));
                                #[cfg(not(debug_assertions))]
                                eprintln!("Warning: Parameter count mismatch. Missing value for {}, defaulting to 0.5", stringify!($field));
                                0.5
                            });
                            let def = ParamDef {
                                name:    stringify!($field),
                                min:     $min     as f64,
                                max:     $max     as f64,
                                step:    $step    as f64,
                                default: $default as f64,
                            };
                            def.denormalize(norm).round() as $type
                        },
                    )*
                }
            }

            /// Export current values as a normalized `[0, 1]` vector.
            pub fn to_normalized(&self) -> Vec<f64> {
                let mut vec = Vec::new();
                $(
                    let def = ParamDef {
                        name:    stringify!($field),
                        min:     $min     as f64,
                        max:     $max     as f64,
                        step:    $step    as f64,
                        default: $default as f64,
                    };
                    vec.push(def.normalize(self.$field as f64));
                )*

                vec
            }
        }

        /// Global definition array for the tuner.
        pub const PARAM_DEFS: &[ParamDef] = &[
            $(
                ParamDef {
                    name:    stringify!($field),
                    min:     $min     as f64,
                    max:     $max     as f64,
                    step:    $step    as f64,
                    default: $default as f64,
                },
            )*
        ];
    };
}

search_params! {
    pub struct SearchParams {
        /// Base margin for lazy evaluation cutoff (cp)
        pub lazy_eval_margin: i32 {
            default: 160,
            min:      60,
            max:     200,
            step:      2,
        },
        /// Divisor for volatility scaling (unitless)
        pub lazy_eval_divisor: i32 {
            default:  32,
            min:       8,
            max:     128,
            step:      1,
        },

        /// Estimated moves-to-go in opening (phase=24) (ply)
        pub mtg_opening: i32 {
            default: 280,
            min:      40,
            max:     300,
            step:      2,
        },
        /// Estimated moves-to-go in endgame (phase=0) (ply)
        pub mtg_endgame: i32 {
            default: 100,
            min:      10,
            max:     150,
            step:      2,
        },

        /// Volatility contribution per pawn (cp/piece)
        pub vol_pawn: i32 {
            default :  5,
            min     :  0,
            max     :  15,
            step    :  1,
        },
        /// Volatility contribution per knight (cp/piece)
        pub vol_knight: i32 {
            default: 10,
            min:      0,
            max:     15,
            step:     1,
        },
        /// Volatility contribution per bishop (cp/piece)
        pub vol_bishop: i32 {
            default:  9,
            min:      0,
            max:     10,
            step:     1,
        },
        /// Volatility contribution per rook (cp/piece)
        pub vol_rook: i32 {
            default:  5,
            min:      0,
            max:     10,
            step:     1,
        },
        /// Volatility contribution per queen (cp/piece)
        pub vol_queen: i32 {
            default: 13,
            min:      0,
            max:     15,
            step:     1,
        },
        /// Flat lazy-eval margin addend scaled by king count — always 2 in any legal position
        /// (one king per side). Unlike other vol_* params, this contributes a constant
        /// `2 · vol_king` regardless of material balance.
        pub vol_king: i32 {
            default:  0,
            min:      0,
            max:      5,
            step:     1,
        },

        /// NMP base reduction (plies).
        pub nmp_base_r: i32 {
            default:   3,
            min:       2,
            max:       5,
            step:      1,
        },
        /// NMP depth divisor for scaling reduction.
        pub nmp_depth_divisor: i32 {
            default:   3,
            min:       2,
            max:       6,
            step:      1,
        },
        /// NMP eval-over-beta divisor (cp per ply of extra reduction).
        pub nmp_eval_divisor: i32 {
            default: 200,
            min:      50,
            max:     400,
            step:     10,
        },
        /// NMP max eval-based extra reduction (plies).
        pub nmp_eval_max: i32 {
            default:   3,
            min:       1,
            max:       5,
            step:      1,
        },

        /// LMR base constant (scaled by 100).
        pub lmr_base: i32 {
            default: 100,
            min:      50,
            max:     200,
            step:      5,
        },
        /// LMR divisor (scaled by 100).
        pub lmr_divisor: i32 {
            default: 225,
            min:     150,
            max:     350,
            step:      5,
        },

        /// Aspiration window initial delta (cp).
        pub asp_initial: i32 {
            default:  15,
            min:       8,
            max:      50,
            step:      1,
        },

        /// RFP per-depth margin (cp/ply).
        pub rfp_margin: i32 {
            default:  45,
            min:      30,
            max:     150,
            step:      5,
        },
        /// RFP maximum depth (plies).
        pub rfp_depth: i32 {
            default:   8,
            min:       3,
            max:      10,
            step:      1,
        },

        /// Futility pruning per-depth margin (cp/ply).
        pub fp_margin: i32 {
            default: 100,
            min:      50,
            max:     300,
            step:      5,
        },
        /// Futility pruning maximum depth (plies).
        pub fp_depth: i32 {
            default:   6,
            min:       1,
            max:       8,
            step:      1,
        },

        /// Razoring per-depth margin (cp/ply).
        pub razoring_margin: i32 {
            default: 300,
            min:     100,
            max:     600,
            step:     10,
        },
        /// Razoring maximum depth (plies).
        pub razoring_depth: i32 {
            default:   3,
            min:       1,
            max:       5,
            step:      1,
        },

        /// Delta pruning margin for qsearch (cp).
        /// If stand_pat + best_capturable + margin < alpha, prune.
        pub delta_margin: i32 {
            default: 200,
            min:      50,
            max:     400,
            step:     10,
        },

        // MVV-LVA capture ordering:
        //   score = V[victim] - A[attacker] (+ V[promo] for promotions)
        //
        // Captures are inherently lifted above quiets because MovePicker strictly
        // segregates them into Stage::GenCaptures and Stage::GenQuiets.
        // V[] and A[] are separate tables so CMA-ES can tune victim priority
        // independently of the attacker discount — classic MVV-LVA hardcodes these
        // as material values, but letting the tuner find the best ordering weights
        // is smarter :p
        /// Fixed score for en-passant (no piece on the target square to look up)
        pub mvvlva_ep: i32 {
            default:  100,
            min:       50,
            max:      200,
            step:     10,
        },
        /// `V[piece]` — victim reward: higher = try capturing this piece sooner
        pub mvvlva_v_pawn: i32 {
            default: 100,
            min:      50,
            max:     200,
            step:      2,
        },
        pub mvvlva_v_knight: i32 {
            default: 300,
            min:     100,
            max:     600,
            step:      5,
        },
        pub mvvlva_v_bishop: i32 {
            default: 300,
            min:     100,
            max:     600,
            step:      5,
        },
        pub mvvlva_v_rook: i32 {
            default:  500,
            min:      200,
            max:     1000,
            step:      5,
        },
        pub mvvlva_v_queen: i32 {
            default:  900,
            min:      400,
            max:     1800,
            step:     10,
        },
        /// Sentinel — king captures are illegal but the slot is indexed by PieceType
        pub mvvlva_v_king: i32 {
            default: 10000,
            min:      5000,
            max:     20000,
            step:    100,
        },
        /// `A[piece]` — attacker penalty: prefer cheap attackers (PxQ > QxQ)
        pub mvvlva_a_pawn: i32 {
            default:  10,
            min:       0,
            max:      50,
            step:      1,
        },
        pub mvvlva_a_knight: i32 {
            default:  30,
            min:       0,
            max:     100,
            step:      1,
        },
        pub mvvlva_a_bishop: i32 {
            default:  30,
            min:       0,
            max:     100,
            step:      1,
        },
        pub mvvlva_a_rook: i32 {
            default:  50,
            min:       0,
            max:     200,
            step:      2,
        },
        pub mvvlva_a_queen: i32 {
            default:  90,
            min:       0,
            max:     300,
            step:      2,
        },
        /// MVV-LVA Attacker King sentinel.
        /// King captures are illegal in all legal positions.
        /// The slot exists because `PieceType::King` is a valid index (5) and the
        /// MVVLVA attacker tables are indexed by `PieceType`.
        /// `step: 0` signals the macro that this value should never snap to a grid
        /// (denormalize returns the raw value unchanged when step < 1e-9).
        pub mvvlva_a_king: i32 {
            default:  0,
            min:      0,
            max:      0,
            step:     0,
        },
    }
}
