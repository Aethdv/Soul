//! Search-specific tunable parameters.
//!
//! Defines the heuristic thresholds and margins (e.g., MVV-LVA bias, lazy evaluation bounds)
//! used to control the search tree's pruning and ordering behavior.

/// Definition of a single tunable parameter.
#[derive(Debug, Clone, Copy)]
pub struct ParamDef {
    pub name: &'static str,
    pub min: f64,
    pub max: f64,
    pub step: f64,
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
        if self.step > 1e-9 { (val / self.step).round() * self.step } else { val }
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
            default:  80,
            min:      40,
            max:     300,
            step:      2,
        },
        /// Estimated moves-to-go in endgame (phase=0) (ply)
        pub mtg_endgame: i32 {
            default:  40,
            min:      10,
            max:     100,
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
        /// NMP verification minimum depth (plies).
        /// Below this, a null-move cutoff is trusted outright. At or above it,
        /// re-search the same position (no null move) at reduced depth — if
        /// that also fails high, the cutoff stands; otherwise fall through to
        /// the regular move loop. Catches zugzwangs NMP would otherwise miss.
        pub nmp_verif_min_depth: i32 {
            default:  14,
            min:       6,
            max:      20,
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

        /// LMP base move count before pruning kicks in.
        pub lmp_base: i32 {
            default:   2,
            min:       1,
            max:       6,
            step:      1,
        },
        /// LMP maximum depth (plies).
        pub lmp_depth: i32 {
            default:   5,
            min:       2,
            max:       8,
            step:      1,
        },

        /// History pruning maximum depth (plies).
        /// Beyond this depth, deeply-negative-history quiets still get full
        /// search — the heuristic stops being a reliable refuter once the
        /// tree has real verification budget.
        pub hist_prune_depth: i32 {
            default:   6,
            min:       2,
            max:      10,
            step:      1,
        },
        /// History pruning per-depth margin (hist units/ply).
        /// Prune quiet moves with `hist < -margin · depth`. Soul's combined
        /// hist_quiet sums main + butterfly + 3× cont tables, each soft-
        /// clamped to ±16384, so the sum reaches well past a single table's
        /// range in practice. Linear scaling keeps the threshold reachable
        /// across the full depth window.
        pub hist_prune_margin: i32 {
            default: 3000,
            min:      500,
            max:     4000,
            step:     100,
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

        /// QSearch recapture-only threshold (plies since QS entry).
        /// Past this ply, non-evasion nodes restrict captures to the square
        /// the opponent's last move landed on. Keeps forcing recapture
        /// chains intact; prunes off-square captures that cause branching
        /// to blow up in mutual-attack positions.
        pub qs_recapture_ply: i32 {
            default:   4,
            min:       2,
            max:      10,
            step:      1,
        },

        /// SEE pruning capture margin (cp/ply) — linear depth scaling.
        /// Prune noisy moves with `see < -margin · depth`. Captures have
        /// rigid material bounds, so the tolerance grows linearly with
        /// depth: deeper searches trust the tree to refute apparent
        /// material losses.
        pub see_capture_margin: i32 {
            default:  80,
            min:      20,
            max:     200,
            step:      2,
        },
        /// SEE pruning quiet margin (cp/ply²) — quadratic depth scaling.
        /// Prune quiet moves with `see < -margin · depth²`. Quiet moves
        /// are speculative at depth (tempi matter) so the tolerance grows
        /// quadratically — very lenient at high depth, strict at low.
        pub see_quiet_margin: i32 {
            default:  60,
            min:      15,
            max:     150,
            step:      2,
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

        /// Capture-history blend divisor for MovePicker capture ordering.
        /// Capture history is bounded ±16384; dividing by this factor scales
        /// the contribution to a range comparable to MVV-LVA. Higher values
        /// give MVV-LVA more weight; lower values give history more influence.
        pub capt_hist_divisor: i32 {
            default:  32,
            min:       8,
            max:     256,
            step:      8,
        },

        /// Score-swing minimum depth (plies).
        /// Below this, aspiration-window wobble dominates the signal, so an
        /// iteration-to-iteration score change is as likely to be noise as
        /// real instability.
        pub score_drop_depth: i32 {
            default:   5,
            min:       3,
            max:      10,
            step:      1,
        },
        /// Score-swing scale (cp per factor doubling).
        /// Maps the iteration-to-iteration score change to a multiplicative
        /// factor on the soft budget: `factor = 2 ^ (clamp(diff, ±scale) / scale)`.
        /// A drop of `scale` cp doubles the budget; a surge of `scale` cp halves it.
        pub score_factor_scale: i32 {
            default: 100,
            min:      40,
            max:     200,
            step:      4,
        },

        /// Best-move stability minimum depth (plies).
        /// Below this, per-move node accounting hasn't accumulated enough
        /// signal for the ratio to be meaningful — leave the soft budget alone.
        pub bm_stab_depth: i32 {
            default:   5,
            min:       3,
            max:      10,
            step:      1,
        },
        /// Best-move stability factor base term (×100 fixed-point).
        /// Intercept of `factor = base − scale · best_nodes/total_nodes`.
        /// Higher = more generous baseline when effort is scattered.
        pub bm_stab_base: i32 {
            default: 270,
            min:     150,
            max:     400,
            step:      5,
        },
        /// Best-move stability factor slope (×100 fixed-point).
        /// Coefficient on best/total ratio.
        /// Higher = faster shrink as search consolidates on one move.
        pub bm_stab_scale: i32 {
            default: 220,
            min:     100,
            max:     350,
            step:      5,
        },
        /// Best-move stability factor floor (×100 fixed-point).
        /// Lower bound so an overwhelmingly confirmed best move can't shrink
        /// the budget to near-zero and cut off mid-move.
        pub bm_stab_floor: i32 {
            default:  56,
            min:      30,
            max:      96,
            step:      2,
        },

        /// Non-pawn correction history weight (fixed-point, /256 scaling).
        /// Controls how strongly each non-pawn material configuration
        /// correction contributes to the static eval adjustment.
        /// 256 = full weight (same as pawn correction), 128 = half.
        pub np_corr_weight: i32 {
            default: 128,
            min:       0,
            max:     512,
            step:      8,
        },
    }
}
