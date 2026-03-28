//! Static evaluation of piece mobility, king safety, and spatial control.
//!
//! # Architecture
//!
//! Computes spatial characteristics in a single pass, sharing attack bitboards
//! across multiple evaluation features.

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_rook, line_bb},
            spatial::SpatialTensor,
        },
        defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
    },
    engine::eval_params::ATTACKER_WEIGHTS,
};

// ──────── Position Openness ────────

// Pawn rams (locked head-to-head) and total pawn count yield an openness scalar.
// In scoring, this blends between "closed" and "open" weight vectors.
// To avoid `f32` in the hot eval loop, we use 10-bit fixed-point precision.
//
//   openness = clamp(1 − 0.08 · rams − 0.02 · pawns, 0, 1)
//
/// Per-ram contribution toward closedness.
pub const RAM_SCALE: i32 = 82; // 0.08 · 1024
/// Per-pawn contribution toward closedness.
pub const PAWN_SCALE: i32 = 20; // 0.02 · 1024
/// Fixed-point precision (1.0 ≡ 1024).
pub const OPEN_UNITY: i32 = 1024;

pub struct Mobility;

impl Mobility {
    /// Attack maps, king-safety features, and scored metrics for both sides.
    /// The returned [`MobilityData`] is self-contained.
    #[inline]
    pub fn compute_all(
        pos: &Position,
        color: Color,
        tensor: &SpatialTensor,
        pinned_w: Bitboard,
        pinned_b: Bitboard,
    ) -> MobilityData {
        let ctx = EvalCtx::build(pos, color, tensor, pinned_w, pinned_b);

        // King safety — computed once per side, then both the raw features
        // and the derived score feed into the final metrics.
        let safety_us = SafetyMetrics::analyze(ctx.ksq_us, ctx.occ, ctx.atk_us, ctx.atk_them, ctx.pawn_us);
        let safety_them =
            SafetyMetrics::analyze(ctx.ksq_them, ctx.occ, ctx.atk_them, ctx.atk_us, ctx.pawn_them);

        let metrics_us =
            score_side(ctx.them, ctx.atk_us, ctx.area_us, ctx.area_them, ctx.pawn_atk_them, ctx.xray_us);
        let metrics_them =
            score_side(ctx.us, ctx.atk_them, ctx.area_them, ctx.area_us, ctx.pawn_atk_us, ctx.xray_them);

        MobilityData {
            metrics_us,
            metrics_them,
            safety_us,
            safety_them,
        }
    }

    /// Tapered, openness-interpolated score differential from `color`'s perspective.
    /// Uses SIMD vector operations via the `EvalMath::Vec4` trait for the final blend.
    #[inline(always)]
    pub fn evaluate_score_diff<T: crate::engine::autograd::traits::EvalMath<Scalar = T>>(
        metrics_us: &SideMetrics,
        metrics_them: &SideMetrics,
        openness: i32,
        phase: T,
        w_mg_o: T::Vec4,
        w_mg_c: T::Vec4,
        w_eg_o: T::Vec4,
        w_eg_c: T::Vec4,
    ) -> T {
        use crate::engine::autograd::traits::{EnvVec4, EnvVec8};

        let diff = T::Vec4::from_lanes(
            T::from_i32(metrics_us.mobility - metrics_them.mobility),
            T::from_i32(metrics_us.shadow_mobility - metrics_them.shadow_mobility),
            T::from_i32(metrics_us.threats - metrics_them.threats),
            T::from_i32(metrics_us.shadow_threats - metrics_them.shadow_threats),
        );

        let o = T::Vec4::splat(openness);
        let c = T::Vec4::splat(OPEN_UNITY - openness);
        let half = T::Vec4::splat(OPEN_UNITY / 2); // rounding bias

        // Interpolate between open/closed weight vectors, then taper MG↔EG.
        let w_mg = (w_mg_o * o + w_mg_c * c + half).srai::<10>();
        let w_eg = (w_eg_o * o + w_eg_c * c + half).srai::<10>();

        // Vec8 madd to compute (diff · w_mg) + (diff · w_eg) in parallel.
        // pack_i16 places [M0..M3] twice into diff_packed and [W_MG0..W_MG3] / [W_EG0..W_EG3] into w_packed.
        // The SSE _mm_madd_epi16 instruction computes adjacent pairwise sums:
        // [ (D0·W0 + D1·W1), (D2·W2 + D3·W3), ... ]
        // For i32, this directly maps to pmaddwd.
        // For Autograd, it constructs tracing nodes.
        let w_packed = w_mg.pack_i16(w_eg);
        let diff_packed = diff.pack_i16(diff);
        let madd = diff_packed.madd(w_packed);

        // Extract and sum the remaining pairs to complete the 4-element dot products.
        // madd.extract::<0>() + madd.extract::<1>() computes the full MG dot product.
        // madd.extract::<2>() + madd.extract::<3>() computes the full EG dot product.
        let mg_sum = madd.extract::<0>() + madd.extract::<1>();
        let eg_sum = madd.extract::<2>() + madd.extract::<3>();

        let t_total_phase = T::from_i32(TOTAL_PHASE);
        let t_eg_phase = t_total_phase - phase;

        (mg_sum * phase + eg_sum * t_eg_phase) / t_total_phase
    }

    /// Position openness in fixed-point [0, 1024].
    ///
    /// Note: always passes White as `us_pawns`. The result is position-symmetric —
    /// white-into-black rams count identically to black-into-white rams — so the
    /// direction is irrelevant. The function signature is kept general for the tuner.
    #[inline]
    pub fn compute_openness(pos: &Position) -> i32 {
        let white = pos.pieces(PieceType::Pawn, Color::White);
        let black = pos.pieces(PieceType::Pawn, Color::Black);
        compute_openness_raw(u64::from(white), u64::from(black))
    }
}

/// Position openness from raw pawn bitboards, in fixed-point [0, OPEN_UNITY].
///
/// `us_pawns` and `them_pawns` are the raw u64 bitboards for each side's pawns.
/// Rams are pawns directly facing the opponent (shifted one rank forward).
#[inline]
pub fn compute_openness_raw(us_pawns: u64, them_pawns: u64) -> i32 {
    let rams = (us_pawns << 8) & them_pawns;
    (OPEN_UNITY
        - RAM_SCALE * rams.count_ones() as i32
        - PAWN_SCALE * (us_pawns | them_pawns).count_ones() as i32)
        .clamp(0, OPEN_UNITY)
}

// ──────── Mobility Data ────────

/// Spatial metrics for one side: `[mobility, shadow_mobility, threats, shadow_threats]`.
///
/// - `mobility`: safe squares controlled, with contested squares counted twice.
/// - `shadow_mobility`: safe x-ray (battery) squares we control.
/// - `threats`: enemy pieces our direct attacks touch.
/// - `shadow_threats`: enemy pieces our x-rays touch.
#[derive(Clone, Copy, Default, Debug)]
pub struct SideMetrics {
    pub mobility:        i32,
    pub shadow_mobility: i32,
    pub threats:         i32,
    pub shadow_threats:  i32,
}

/// Complete mobility snapshot for one position, computed once and consumed by
/// both the engine (score diff) and the tuner (raw feature extraction).
#[derive(Clone, Default, Debug)]
pub struct MobilityData {
    pub metrics_us:   SideMetrics,
    pub metrics_them: SideMetrics,
    pub safety_us:    SafetyMetrics,
    pub safety_them:  SafetyMetrics,
}

// ──────── King Safety Features ────────

/// Raw king-zone features for one side.
///
/// Kept separate so the tuner can regress independent gradients against each component.
/// The engine folds them into a single score via [`Self::score()`].
#[derive(Clone, Copy, Default, Debug)]
pub struct SafetyMetrics {
    /// Enemy pieces hitting the king ring (capped to weight-table bounds).
    pub attackers:      usize,
    /// King-zone squares attacked by them but not defended by us.
    pub weak:           i32,
    /// Friendly pawns adjacent to the king.
    pub shield:         i32,
    /// Rook-reachable squares from the king (open lines = exposure).
    pub ortho_exposure: i32,
    /// Bishop-reachable squares from the king.
    pub diag_exposure:  i32,
}

impl SafetyMetrics {
    /// Collapses raw features into a single weighted score.
    /// Positive ⇒ well-sheltered king, negative ⇒ under fire.
    #[inline]
    pub fn score<T: crate::engine::autograd::traits::EvalMath<Scalar = T>>(
        &self,
        w_shield: T,
        w_ortho: T,
        w_diag: T,
        w_atk: T,
    ) -> T {
        let shelter = T::from_i32(self.shield) * w_shield;
        let exposure = T::from_i32(self.ortho_exposure) * w_ortho + T::from_i32(self.diag_exposure) * w_diag;

        // ──────── Non-linear Attacker Pressure ────────
        //
        // King safety isn't linear:
        // A second attacker is more than twice as dangerous as the first.
        // We model this as an accelerating danger curve where the w_atk
        // (indexed by the number of attackers) scales a base penalty,
        // which is then amplified by the number of weak squares.
        //
        // Semantic bridge: we divide by 10 to keep the internal resolution
        // high for the tuner (which sees 0.1 increments) while maintaining
        // integer performance in the search eval.
        //   pressure = (Weight[attackers] · weak_squares) / 10
        //
        // The division by 10 is a semantic bridge.
        // Our tuner (CMA-ES) works with high-precision floating point weights,
        // but the engine uses centipawn integers.
        // This scale factor allows the tuner to find subtle gradients
        // (e.g. a weight of 122 mapping to 12.2 cp) without losing resolution
        // in the engine's integer math.
        //
        // NOTE: For DualNode (autograd), .trunc() explicitly passes the gradient
        // through unmodified (Straight-Through Estimator). This assumes the local
        // gradient of the stair-step function is 1.0, which scales the effective
        // learning rate for w_atk relative to floating-point reality.
        let pressure = ((w_atk * T::from_i32(self.weak)) / T::from_i32(10)).trunc();

        shelter - exposure - pressure
    }

    /// Analyzes the king neighborhood
    /// and records how exposed or defended it is.
    #[inline(always)]
    fn analyze(
        ksq: Square,
        occ: Bitboard,
        atk_us: Bitboard,
        atk_them: Bitboard,
        our_pawns: Bitboard,
    ) -> Self {
        let zone = atk_king(ksq);
        Self {
            // Clamp to weight-table bounds — five-plus attackers all map to the maximum danger entry.
            attackers:      ((zone & atk_them).popcount() as usize).min(ATTACKER_WEIGHTS.len() - 1),
            weak:           (zone & atk_them & !atk_us).popcount() as i32,
            shield:         (zone & our_pawns).popcount() as i32,
            ortho_exposure: atk_rook(ksq, occ).popcount() as i32,
            diag_exposure:  atk_bishop(ksq, occ).popcount() as i32,
        }
    }
}

// ──────── Internal Evaluation Context ────────

/// Pre-computed attack maps for both sides. Built once per evaluation and
/// threaded through every sub-computation to avoid redundant slider work.
struct EvalCtx {
    /// Piece + pawn attacks, excluding king.
    /// Used for danger assessment — your own king's reach
    /// doesn't help assault the opponent's king zone.
    atk_us:   Bitboard,
    atk_them: Bitboard,

    /// Full control:
    /// piece + pawn + king attacks.
    /// Used for mobility and piece-protection calculations
    /// where the king's influence matters.
    area_us:   Bitboard,
    area_them: Bitboard,

    /// Pawn-only attack maps, cached to avoid recomputation.
    pawn_atk_us:   Bitboard,
    pawn_atk_them: Bitboard,

    ksq_us:   Square,
    ksq_them: Square,

    /// Pawn occupancy (used for shield evaluation, not attacks).
    pawn_us:   Bitboard,
    pawn_them: Bitboard,

    us:   Bitboard,
    them: Bitboard,
    occ:  Bitboard,

    /// Shadow/X-ray attack maps.
    xray_us:   Bitboard,
    xray_them: Bitboard,
}

impl EvalCtx {
    /// Builds symmetric attack maps for `color` ("us") vs its opponent
    /// ("them").
    ///
    /// Pinned pieces contribute mobility conservatively:
    /// - Knights: zero mobility. A knight can never legally move while
    ///   pinned (it can't stay on the pin ray), so its attacks are excluded entirely.
    /// - Sliders (bishops, rooks, queens): attacks restricted to the pin ray.
    ///   A rook pinned horizontally still threatens along that rank.
    ///   These are re-injected via `inject_pinned` after the tensor computation.
    ///
    /// This prevents the evaluation from crediting mobility that would leave the king in check.
    #[inline(always)]
    fn build(
        pos: &Position,
        color: Color,
        tensor: &SpatialTensor,
        pinned_w: Bitboard,
        pinned_b: Bitboard,
    ) -> Self {
        let opp = color.opposite();
        let us = pos.side_bb[color];
        let them = pos.side_bb[opp];
        let occ = pos.occupancy();

        let knights = pos.role_bb[PieceType::Knight];
        let kings = pos.role_bb[PieceType::King];

        let ksq_us = king_sq(kings & us);
        let ksq_them = king_sq(kings & them);

        let (pinned_us, pinned_them) = if color == Color::White {
            (pinned_w, pinned_b)
        } else {
            (pinned_b, pinned_w)
        };

        // Knights: pinned = zero mobility.
        let mut knight_atk_us = Bitboard(0);
        for sq in (knights & us) & !pinned_us {
            knight_atk_us |= atk_knight(sq);
        }
        let mut knight_atk_them = Bitboard(0);
        for sq in (knights & them) & !pinned_them {
            knight_atk_them |= atk_knight(sq);
        }

        // Sliders: Use SpatialTensor for direct attacks (pinned pieces are natively excluded).
        //
        // NOTE: Pinned pieces are deliberately excluded from `xray_us` and `xray_them` as well.
        // While a pinned piece could theoretically provide x-ray battery support along its pin ray,
        // this is a CPU-cycle tradeoff — sacrificing a rare edge case for pure engine speed.
        let (mut slider_atk_us, mut slider_atk_them, xray_us, xray_them) = if color == Color::White {
            (
                Bitboard(tensor.w_ortho_direct() | tensor.w_diag_direct()),
                Bitboard(tensor.b_ortho_direct() | tensor.b_diag_direct()),
                Bitboard(tensor.w_ortho_xray() | tensor.w_diag_xray()),
                Bitboard(tensor.b_ortho_xray() | tensor.b_diag_xray()),
            )
        } else {
            (
                Bitboard(tensor.b_ortho_direct() | tensor.b_diag_direct()),
                Bitboard(tensor.w_ortho_direct() | tensor.w_diag_direct()),
                Bitboard(tensor.b_ortho_xray() | tensor.b_diag_xray()),
                Bitboard(tensor.w_ortho_xray() | tensor.w_diag_xray()),
            )
        };

        // Inject the strictly legal (restricted) pin-rays for pinned sliders.
        let rq = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];
        let bq = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];

        let inject_pinned = |pinned: Bitboard, ksq: Square| {
            let mut atk = Bitboard(0);
            for sq in pinned & rq {
                atk |= atk_rook(sq, occ) & line_bb(ksq, sq);
            }
            for sq in pinned & bq {
                atk |= atk_bishop(sq, occ) & line_bb(ksq, sq);
            }
            atk
        };

        slider_atk_us |= inject_pinned(pinned_us, ksq_us);
        slider_atk_them |= inject_pinned(pinned_them, ksq_them);

        // Cache pawn attacks and combine with piece attacks.
        let pawn_atk_us = pos.pawn_attacks(color);
        let pawn_atk_them = pos.pawn_attacks(opp);

        let atk_us = slider_atk_us | knight_atk_us | pawn_atk_us;
        let atk_them = slider_atk_them | knight_atk_them | pawn_atk_them;

        Self {
            atk_us,
            atk_them,
            area_us: atk_us | atk_king(ksq_us),
            area_them: atk_them | atk_king(ksq_them),
            pawn_atk_us,
            pawn_atk_them,
            ksq_us,
            ksq_them,
            pawn_us: pos.pieces(PieceType::Pawn, color),
            pawn_them: pos.pieces(PieceType::Pawn, opp),
            us,
            them,
            occ,
            xray_us,
            xray_them,
        }
    }
}

/// Extracts the king square from a bitboard.
///
/// # Panics (debug) / UB (release)
/// `king_bb` must be non-empty. Callers are responsible for this invariant.
#[inline(always)]
fn king_sq(king_bb: Bitboard) -> Square {
    debug_assert!(king_bb.is_not_empty(), "king_sq called on empty bitboard");
    king_bb.lsb()
}

// ──────── Per-Side Scoring ────────

/// Computes a side's base metrics: `[mobility, shadow_mobility, threats, shadow_threats]`.
#[inline(always)]
fn score_side(
    them: Bitboard,
    atk_us: Bitboard,
    area_us: Bitboard,
    area_them: Bitboard,
    enemy_pawn_atk: Bitboard,
    xray_us: Bitboard,
) -> SideMetrics {
    // "Safe" squares are those we control that aren't under enemy pawn fire.
    let safe = area_us & !enemy_pawn_atk;

    // Mobility rewards exclusive territorial control more than shared space.
    // We count all safe squares, but double count those that they don't reach.
    let mobility = (safe & !area_them).popcount() as i32 + safe.popcount() as i32;

    // Shadow mobility (battery squares)
    let shadow_mobility = (xray_us & !enemy_pawn_atk).popcount() as i32;

    // Direct piece threats (excluding king)
    let threats = (atk_us & them).popcount() as i32;

    // Shadow threats (X-ray threats)
    let shadow_threats = (xray_us & them).popcount() as i32;

    SideMetrics {
        mobility,
        shadow_mobility,
        threats,
        shadow_threats,
    }
}
