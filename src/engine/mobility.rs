//! Static evaluation of piece mobility, king safety, and spatial control.
//!
//! Attack maps are the expensive part, so they're built once per position and
//! shared: mobility, king safety, threats, and x-ray batteries all read the same
//! bitboards instead of recomputing the slider attacks they each need.

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, line_bb},
            spatial::SpatialMaps,
            xorboard::XorBoard,
        },
        defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
    },
    engine::{
        autograd::traits::{EnvVec4, EnvVec8, EvalMath},
        combiner::Accumulators,
        eval::{EvalParams, SharedFeatures},
        eval_params::{ATTACKER, LAYOUT},
        term::{KingSafetyUpstream, LinearTerm, TaperPair, TermSource},
    },
    weave::Vf64x4,
};

// Pawn rams (locked head-to-head) and total pawn count yield an openness scalar,
// which blends between the "closed" and "open" weight vectors in scoring. The
// 10-bit fixed point keeps f32 out of the hot eval loop.
//
//   openness = clamp(1 − 0.08 · rams − 0.02 · pawns, 0, 1)
//
/// Per-ram contribution toward closedness.
pub const RAM_SCALE: i32 = 82; // 0.08 · 1024
/// Per-pawn contribution toward closedness.
pub const PAWN_SCALE: i32 = 20; // 0.02 · 1024
/// Fixed-point precision (1.0 ≡ 1024).
pub const OPEN_UNITY: i32 = 1024;
const INV_OPEN_UNITY: f64 = 1.0 / OPEN_UNITY as f64;

pub struct Mobility;
pub struct MobilityTerm;
pub struct KingSafetyTerm;

/// Complete mobility snapshot for one position, computed once and consumed by
/// both the engine (score diff) and the tuner (raw feature extraction).
///
/// `us` is White in every field; nothing here is side-to-move relative.
#[derive(Clone, Default, Debug)]
pub struct SpatialMetrics {
    pub metrics_us: SideMetrics,
    pub metrics_them: SideMetrics,
    pub safety_us: SafetyMetrics,
    pub safety_them: SafetyMetrics,
}

/// Spatial metrics for one side.
///
/// - `mobility`: squares off enemy pawn attacks, summed per piece rather than over the union.
/// - `shadow_mobility`: safe x-ray (battery) squares we control.
/// - `threats`: enemy pieces our direct attacks touch.
/// - `shadow_threats`: enemy pieces our x-rays touch.
#[derive(Clone, Copy, Default, Debug)]
pub struct SideMetrics {
    pub mobility: i32,
    pub shadow_mobility: i32,
    pub threats: i32,
    pub shadow_threats: i32,
}

/// Raw king-zone features for one side.
///
/// Kept separate so the tuner can regress independent gradients against each component.
/// The engine folds them into shelter and pressure, which the combiner collapses.
#[derive(Clone, Copy, Default, Debug)]
pub struct SafetyMetrics {
    /// Enemy pieces hitting the king ring, capped at `ATTACKER.len() - 1` on
    /// construction so every consumer indexes the weight table directly.
    pub attackers: usize,
    /// King-zone squares attacked by them but not defended by us.
    pub weak: i32,
    /// Friendly pawns adjacent to the king.
    pub shield: i32,
    /// Rook-reachable squares from the king (open lines = exposure).
    pub ortho_exposure: i32,
    /// Bishop-reachable squares from the king.
    pub diag_exposure: i32,
}

/// Extracted mobility features for generic dispatch.
pub struct MobilityInput {
    pub diff: Vf64x4,
    pub openness: i32,
}

/// Extracted king-safety features for generic dispatch. Per-side rather than
/// differenced, because the forward scores each side into its own bucket.
pub struct KingSafetyInput {
    pub us: SafetyMetrics,
    pub them: SafetyMetrics,
}

/// Position openness from raw pawn bitboards, in fixed-point [0, OPEN_UNITY].
///
/// `us_pawns` is White's, always: rams come from shifting it one rank north onto
/// `them_pawns`, which catches every locked pair exactly once. Swapped, the shift walks
/// Black's pawns backwards down the board and counts pairs that are not rams.
#[inline]
pub fn compute_openness_raw(us_pawns: u64, them_pawns: u64) -> i32 {
    let rams = (us_pawns << 8) & them_pawns;
    (OPEN_UNITY - RAM_SCALE * rams.count_ones() as i32 - PAWN_SCALE * (us_pawns | them_pawns).count_ones() as i32)
        .clamp(0, OPEN_UNITY)
}

impl SideMetrics {
    /// `self` minus `them`, in the lane order the weight vectors and the layout offsets
    /// both use. Nothing checks that order, so a swap here silently hands the shipped
    /// weights to the wrong metric.
    #[inline(always)]
    fn diff(&self, them: &Self) -> [i32; 4] {
        [
            self.mobility - them.mobility,
            self.shadow_mobility - them.shadow_mobility,
            self.threats - them.threats,
            self.shadow_threats - them.shadow_threats,
        ]
    }
}

impl SafetyMetrics {
    /// Shelter minus exposure. Positive → well-sheltered king.
    #[inline]
    pub fn shelter<T: EvalMath<Scalar = T>>(&self, w_shield: T, w_ortho: T, w_diag: T) -> T {
        let shelter = T::from_i32(self.shield) * w_shield;
        let exposure = T::from_i32(self.ortho_exposure) * w_ortho + T::from_i32(self.diag_exposure) * w_diag;
        shelter - exposure
    }

    /// What the attackers are worth against this king, before the combiner curves it.
    ///
    /// `w_atk` is indexed by attacker count, so escalation with the *number* of
    /// attackers lives in the weight table; `weak / 10` keeps resolution high for
    /// the tuner (0.1 cp increments) while staying integer in eval. DualNode
    /// passes gradient through `.trunc()` unmodified (straight-through).
    #[inline]
    pub fn pressure<T: EvalMath<Scalar = T>>(&self, w_atk: T) -> T { ((w_atk * T::from_i32(self.weak)) / T::from_i32(10)).trunc() }

    #[inline(always)]
    fn analyze(ksq: Square, occ: Bitboard, atk_us: Bitboard, atk_them: Bitboard, our_pawns: Bitboard) -> Self {
        let zone = atk_king(ksq);
        Self {
            // Five or more attackers all take the maximum danger entry.
            attackers: ((zone & atk_them).popcount() as usize).min(ATTACKER.len() - 1),
            weak: (zone & atk_them & !atk_us).popcount() as i32,
            shield: (zone & our_pawns).popcount() as i32,
            ortho_exposure: atk_rook(ksq, occ).popcount() as i32,
            diag_exposure: atk_bishop(ksq, occ).popcount() as i32,
        }
    }
}

impl Mobility {
    #[inline]
    pub fn compute_all(
        pos: &Position,
        maps: &SpatialMaps,
        pinned_w: Bitboard,
        pinned_b: Bitboard,
        rows: Option<&XorBoard>,
    ) -> SpatialMetrics {
        let ctx = EvalCtx::build(pos, maps, pinned_w, pinned_b);
        let mob_us = piece_mobility(pos, rows, Color::White, pinned_w, ctx.ksq_us, !ctx.pawn_atk_them);
        let mob_them = piece_mobility(pos, rows, Color::Black, pinned_b, ctx.ksq_them, !ctx.pawn_atk_us);
        let safety_us = SafetyMetrics::analyze(ctx.ksq_us, ctx.occ, ctx.atk_us, ctx.atk_them, ctx.pawn_us);
        let safety_them = SafetyMetrics::analyze(ctx.ksq_them, ctx.occ, ctx.atk_them, ctx.atk_us, ctx.pawn_them);
        let metrics_us = score_side(ctx.them, ctx.atk_us, mob_us, ctx.pawn_atk_them, ctx.xray_us);
        let metrics_them = score_side(ctx.us, ctx.atk_them, mob_them, ctx.pawn_atk_us, ctx.xray_them);
        SpatialMetrics { metrics_us, metrics_them, safety_us, safety_them }
    }

    /// Tapered, openness-interpolated score differential, `metrics_us` minus `metrics_them`.
    #[inline(always)]
    pub fn evaluate_score_diff<T: EvalMath<Scalar = T>>(
        metrics_us: &SideMetrics,
        metrics_them: &SideMetrics,
        openness: i32,
        phase: T,
        w_mg_o: T::Vec4,
        w_mg_c: T::Vec4,
        w_eg_o: T::Vec4,
        w_eg_c: T::Vec4,
    ) -> T {
        let d = metrics_us.diff(metrics_them);
        let diff = T::Vec4::from_lanes(T::from_i32(d[0]), T::from_i32(d[1]), T::from_i32(d[2]), T::from_i32(d[3]));

        let o = T::Vec4::splat(openness);
        let c = T::Vec4::splat(OPEN_UNITY - openness);
        let half = T::Vec4::splat(OPEN_UNITY / 2); // rounding bias
        // Interpolate the open/closed weight vectors by openness.
        let w_mg = (w_mg_o * o + w_mg_c * c + half).srai::<10>();
        let w_eg = (w_eg_o * o + w_eg_c * c + half).srai::<10>();
        // SIMD dot-product diff against the MG and EG weights.
        let w_packed = w_mg.pack_i16(w_eg);
        let diff_packed = diff.pack_i16(diff);
        let madd = diff_packed.madd(w_packed);

        let mg_sum = madd.extract::<0>() + madd.extract::<1>();
        let eg_sum = madd.extract::<2>() + madd.extract::<3>();

        let t_total_phase = T::from_i32(TOTAL_PHASE);
        let t_eg_phase = t_total_phase - phase;
        // Integer division truncates for the engine; the tuner's f64 has to be told.
        ((mg_sum * phase + eg_sum * t_eg_phase) / t_total_phase).trunc()
    }

    /// Position openness in fixed-point [0, 1024], White's pawns first as
    /// [`compute_openness_raw`] requires.
    #[inline]
    pub fn compute_openness(pos: &Position) -> i32 {
        let white = pos.pieces(PieceType::Pawn, Color::White);
        let black = pos.pieces(PieceType::Pawn, Color::Black);
        compute_openness_raw(u64::from(white), u64::from(black))
    }
}

impl LinearTerm for MobilityTerm {
    type Upstream = TaperPair;
    type Input = MobilityInput;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>) {
        acc.mobility = Mobility::evaluate_score_diff::<T>(
            &features.spatial.metrics_us, &features.spatial.metrics_them, features.openness, phase, params.mg_mob_open,
            params.mg_mob_closed, params.eg_mob_open, params.eg_mob_closed,
        );
    }

    /// Scalar mirror of the SIMD blend above: open/closed weights round to whole
    /// centipawns per lane, before the taper.
    #[inline(always)]
    fn apply_input(input: MobilityInput, values: &[f64], phase: f64, acc: &mut Accumulators<f64>) {
        let lo = LAYOUT.mobility_open_offset;
        let lc = LAYOUT.mobility_closed_offset;
        let unity = f64::from(OPEN_UNITY);
        let o = f64::from(input.openness);
        let c = unity - o;

        let mut diff = [0.0f64; 4];
        // SAFETY: `diff` is exactly the 4 lanes `storeu` writes.
        unsafe { input.diff.storeu(diff.as_mut_ptr()) };

        let blend = |open: usize, closed: usize| ((values[open] * o + values[closed] * c + unity / 2.0) / unity).floor();

        let mut mg_sum = 0.0;
        let mut eg_sum = 0.0;
        for (i, d) in diff.iter().enumerate() {
            mg_sum += d * blend(lo + i, lc + i);
            eg_sum += d * blend(lo + 4 + i, lc + 4 + i);
        }

        // Both halves sum before the taper, exactly as the madd above does.
        let total = f64::from(TOTAL_PHASE);
        acc.mobility = ((mg_sum * phase + eg_sum * (total - phase)) / total).trunc();
    }

    ///   `∂score/∂mg_mob_open[j]   = d_mg · diff[j] · (openness / 1024)`
    ///   `∂score/∂mg_mob_closed[j] = d_mg · diff[j] · (closedness / 1024)`
    ///
    /// The combiner pre-multiplies `d_mg = d · t_mg`, `d_eg = d · t_eg` so scatter skips the taper split.
    #[inline(always)]
    fn scatter(input: MobilityInput, upstream: TaperPair, grads: &mut [f64]) {
        let lo = LAYOUT.mobility_open_offset;
        let lc = LAYOUT.mobility_closed_offset;
        assert!(grads.len() >= lo.max(lc) + 8, "MobilityTerm::scatter: grads too short");

        let o_frac = f64::from(input.openness) * INV_OPEN_UNITY;
        let c_frac = 1.0 - o_frac;

        let mut do_scatter = |offset: usize, scale: f64| {
            // SAFETY: grads length verified by assert above and layout invariants.
            unsafe {
                let p = grads.as_mut_ptr().add(offset);
                (Vf64x4::loadu(p) + input.diff * Vf64x4::splat(scale)).storeu(p);
            }
        };

        do_scatter(lo, upstream.d_mg * o_frac);
        do_scatter(lo + 4, upstream.d_eg * o_frac);
        do_scatter(lc, upstream.d_mg * c_frac);
        do_scatter(lc + 4, upstream.d_eg * c_frac);
    }
}

impl TermSource<MobilityTerm> for SharedFeatures {
    type Input = MobilityInput;

    #[inline(always)]
    fn extract(&self) -> MobilityInput {
        let diff = self.spatial.metrics_us.diff(&self.spatial.metrics_them);
        MobilityInput { diff: Vf64x4::from(diff.map(f64::from)), openness: self.openness }
    }
}

impl LinearTerm for KingSafetyTerm {
    /// The combiner's MG taper is folded into every field, and the danger halves
    /// arrive already signed, which is why scatter adds both of them.
    type Upstream = KingSafetyUpstream;
    type Input = KingSafetyInput;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
        let (us, them) = (&features.spatial.safety_us, &features.spatial.safety_them);

        acc.safety_us = us.shelter(params.w_shield, params.w_ortho, params.w_diag);
        acc.safety_them = them.shelter(params.w_shield, params.w_ortho, params.w_diag);
        acc.danger_us = us.pressure(params.atk_weights[us.attackers]);
        acc.danger_them = them.pressure(params.atk_weights[them.attackers]);
    }

    #[inline(always)]
    fn apply_input(input: KingSafetyInput, values: &[f64], _phase: f64, acc: &mut Accumulators<f64>) {
        let ks = LAYOUT.king_safety_offset;
        let ao = LAYOUT.attacker_offset;
        acc.safety_us = input.us.shelter(values[ks], values[ks + 1], values[ks + 2]);
        acc.safety_them = input.them.shelter(values[ks], values[ks + 1], values[ks + 2]);
        acc.danger_us = input.us.pressure(values[ao + input.us.attackers]);
        acc.danger_them = input.them.pressure(values[ao + input.them.attackers]);
    }

    ///   `∂score/∂w_shield  =  upstream · (shield_us − shield_them)`
    ///   `∂score/∂w_ortho   = −upstream · (ortho_us  − ortho_them)`
    ///   `∂score/∂w_diag    = −upstream · (diag_us   − diag_them)`
    ///   `∂score/∂atk_weights[idx_side] = ±upstream · (weak_side / 10)`
    ///
    /// The combiner pre-multiplies the `MG · phase/24` taper into `upstream`.
    #[inline(always)]
    fn scatter(input: KingSafetyInput, upstream: KingSafetyUpstream, grads: &mut [f64]) {
        let ks = LAYOUT.king_safety_offset;
        let ao = LAYOUT.attacker_offset;
        grads[ks] += upstream.shelter * f64::from(input.us.shield - input.them.shield);
        grads[ks + 1] -= upstream.shelter * f64::from(input.us.ortho_exposure - input.them.ortho_exposure);
        grads[ks + 2] -= upstream.shelter * f64::from(input.us.diag_exposure - input.them.diag_exposure);

        grads[ao + input.us.attackers] += upstream.danger_us * (f64::from(input.us.weak) / 10.0);
        grads[ao + input.them.attackers] += upstream.danger_them * (f64::from(input.them.weak) / 10.0);
    }
}

impl TermSource<KingSafetyTerm> for SharedFeatures {
    type Input = KingSafetyInput;

    #[inline(always)]
    fn extract(&self) -> KingSafetyInput { KingSafetyInput { us: self.spatial.safety_us, them: self.spatial.safety_them } }
}

/// Pre-computed attack maps for both sides.
struct EvalCtx {
    us: Bitboard,
    them: Bitboard,
    occ: Bitboard,
    // Piece and pawn attacks, king excluded. A king's own reach never helps
    // assault the enemy king zone.
    atk_us: Bitboard,
    atk_them: Bitboard,
    pawn_atk_us: Bitboard,
    pawn_atk_them: Bitboard,
    ksq_us: Square,
    ksq_them: Square,
    // The pawns themselves, for the shield count.
    pawn_us: Bitboard,
    pawn_them: Bitboard,
    xray_us: Bitboard,
    xray_them: Bitboard,
}

impl EvalCtx {
    /// Builds both sides' attack maps: White is "us", Black is "them".
    ///
    /// A pinned piece contributes only what it could legally play:
    /// - Knights: nothing. A knight can never stay on the pin ray, so it drops out
    ///   of the attack map entirely.
    /// - Sliders (bishops, rooks, queens): the pin ray alone. A rook pinned
    ///   horizontally still threatens along that rank.
    ///
    /// Without this the threat and danger counts credit attacks that would leave the
    /// king in check.
    #[inline(always)]
    fn build(pos: &Position, maps: &SpatialMaps, pinned_w: Bitboard, pinned_b: Bitboard) -> Self {
        let us = pos.side_bb[Color::White];
        let them = pos.side_bb[Color::Black];
        let occ = pos.occupancy();

        let knights = pos.role_bb[PieceType::Knight];
        let kings = pos.role_bb[PieceType::King];

        let ksq_us = (kings & us).lsb();
        let ksq_them = (kings & them).lsb();

        let knight_attacks = |side: Bitboard, pinned: Bitboard| {
            let mut atk = Bitboard(0);
            for sq in (knights & side) & !pinned {
                atk |= atk_knight(sq);
            }
            atk
        };

        let knight_atk_us = knight_attacks(us, pinned_w);
        let knight_atk_them = knight_attacks(them, pinned_b);

        // SpatialMaps drops pinned pieces from the direct and the x-ray maps alike. Only
        // the direct maps get their pin rays back below: a pinned piece backing a battery
        // is rare enough that the second pass would cost more than it scores.
        let (mut slider_atk_us, mut slider_atk_them, xray_us, xray_them) = (
            Bitboard(maps.w_ortho_direct() | maps.w_diag_direct()),
            Bitboard(maps.b_ortho_direct() | maps.b_diag_direct()),
            Bitboard(maps.w_ortho_xray() | maps.w_diag_xray()),
            Bitboard(maps.b_ortho_xray() | maps.b_diag_xray()),
        );

        // A queen sits in both sets, and the pin ray masks off whichever half it cannot play.
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

        slider_atk_us |= inject_pinned(pinned_w, ksq_us);
        slider_atk_them |= inject_pinned(pinned_b, ksq_them);

        let pawn_atk_us = pos.pawn_attacks(Color::White);
        let pawn_atk_them = pos.pawn_attacks(Color::Black);

        let atk_us = slider_atk_us | knight_atk_us | pawn_atk_us;
        let atk_them = slider_atk_them | knight_atk_them | pawn_atk_them;

        Self {
            atk_us,
            atk_them,
            pawn_atk_us,
            pawn_atk_them,
            ksq_us,
            ksq_them,
            pawn_us: pos.pieces(PieceType::Pawn, Color::White),
            pawn_them: pos.pieces(PieceType::Pawn, Color::Black),
            us,
            them,
            occ,
            xray_us,
            xray_them,
        }
    }
}

/// What each piece of `color` reaches inside `area`, summed over the side.
///
/// A fill cannot produce this. ORing the sides together loses which piece got
/// where, so a square two of our pieces attack is worth one to the union and two
/// here. The rows keep the identity and answer in one vectorized pass; without
/// them the same sum is a probe per piece, which only an offline tuner can
/// afford.
///
/// The pin policy matches SpatialMaps: a pinned knight has nothing legal, a
/// pinned slider keeps its pin ray, a pinned pawn is left whole.
fn piece_mobility(pos: &Position, rows: Option<&XorBoard>, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
    if let Some(rows) = rows {
        return rows.mobility(color, pinned, ksq, area);
    }

    let occ = pos.occupancy();
    let mut total = 0;

    for square in pos.side_bb[color] & !pos.role_bb[PieceType::King] {
        let piece = pos.piece_at(square);
        let attacks = match piece {
            PieceType::Pawn => atk_pawn(square, color),
            PieceType::Knight => atk_knight(square),
            PieceType::Bishop => atk_bishop(square, occ),
            PieceType::Rook => atk_rook(square, occ),
            PieceType::Queen => atk_rook(square, occ) | atk_bishop(square, occ),
            _ => Bitboard(0),
        };

        let legal = match piece {
            _ if !pinned.check_bit(square) => attacks,
            PieceType::Knight => Bitboard(0),
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => attacks & line_bb(ksq, square),
            _ => attacks,
        };

        total += (legal & area).popcount() as i32;
    }
    total
}

#[inline(always)]
fn score_side(them: Bitboard, atk_us: Bitboard, mobility: i32, enemy_pawn_atk: Bitboard, xray_us: Bitboard) -> SideMetrics {
    // Enemy pieces our direct attacks touch (king excluded from attack map).
    let threats = (atk_us & them).popcount() as i32;
    let shadow_mobility = (xray_us & !enemy_pawn_atk).popcount() as i32;
    let shadow_threats = (xray_us & them).popcount() as i32;

    SideMetrics { mobility, shadow_mobility, threats, shadow_threats }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::{STARTPOS, xorboard::XorBoard};

    #[test]
    fn per_piece_mobility_matches_the_probes() {
        const FENS: [&str; 5] = [
            STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
            "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 1",
        ];

        for fen in FENS {
            let pos = Position::from_fen(fen);
            let rows = XorBoard::new(&pos);

            for color in [Color::White, Color::Black] {
                let pinned = pos.pinned_pieces(color);
                let ksq = pos.pieces(PieceType::King, color).lsb();
                let area = !pos.pawn_attacks(color.opposite());
                let from_rows = piece_mobility(&pos, Some(&rows), color, pinned, ksq, area);
                let from_probes = piece_mobility(&pos, None, color, pinned, ksq, area);
                assert_eq!(from_rows, from_probes, "{fen} {color:?}");
            }
        }
    }

    #[test]
    fn openness_counts_the_locked_pair() {
        let pos = Position::from_fen("4k3/8/8/4p3/4P3/8/8/4K3 w - - 0 1");
        assert_eq!(Mobility::compute_openness(&pos), OPEN_UNITY - RAM_SCALE - 2 * PAWN_SCALE);
    }
}
