//! Static evaluation of piece mobility, king safety, and spatial control.

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_queen, atk_rook, line_bb},
            spatial::SliderMaps,
            xorboard::XorBoard,
        },
        defs::{Bitboard, Color, PieceType, Square},
    },
    engine::{
        autograd::traits::{EvalMath, EvalVec4, EvalVec8},
        combiner::{Accumulators, taper},
        eval::{EvalParams, SharedFeatures},
        eval_params::{ATTACKER, LAYOUT},
        term::{KingSafetyUpstream, LinearTerm, TaperPair, TermSource},
    },
    weave::F64x4,
};

/// Per-ram contribution toward closedness.
const RAM_SCALE: i32 = 82; // 0.08 · 1024
/// Per-pawn contribution toward closedness.
const PAWN_SCALE: i32 = 20; // 0.02 · 1024
/// Fixed-point precision (1.0 ≡ 1024).
pub const OPEN_UNITY: i32 = 1024;
#[cfg_attr(not(feature = "dataset"), expect(dead_code, reason = "read by the tuner's scatter"))]
const INV_OPEN_UNITY: f64 = 1.0 / OPEN_UNITY as f64;

pub(crate) struct MobilityTerm;
pub(crate) struct KingSafetyTerm;

/// `us` is White in every field; nothing here is side-to-move relative.
#[derive(Clone, Default, Debug)]
pub(crate) struct SpatialMetrics {
    pub(crate) metrics_us: SideMetrics,
    pub(crate) metrics_them: SideMetrics,
    pub(crate) safety_us: SafetyMetrics,
    pub(crate) safety_them: SafetyMetrics,
}

#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct SideMetrics {
    /// Squares outside enemy pawn attacks, counted once per attacking piece.
    pub(crate) mobility: i32,
    /// X-ray squares outside enemy pawn attacks, counted once per square.
    pub(crate) shadow_mobility: i32,
    pub(crate) threats: i32,
    pub(crate) shadow_threats: i32,
}

/// Raw king-zone features for one side.
#[derive(Clone, Copy, Default, Debug)]
pub(crate) struct SafetyMetrics {
    /// King-zone squares they attack, capped at `ATTACKER.len() - 1`.
    pub(crate) attacked_zone: usize,
    /// King-zone squares attacked by them but not defended by us.
    pub(crate) weak: i32,
    /// Friendly pawns adjacent to the king.
    pub(crate) shield: i32,
    /// Rook-reachable squares from the king.
    pub(crate) ortho_exposure: i32,
    /// Bishop-reachable squares from the king.
    pub(crate) diag_exposure: i32,
}

#[cfg_attr(not(feature = "dataset"), expect(dead_code, reason = "read by the tuner's scatter"))]
pub(crate) struct MobilityInput {
    pub(crate) diff: F64x4,
    pub(crate) openness: i32,
}

#[cfg_attr(not(feature = "dataset"), expect(dead_code, reason = "read by the tuner's scatter"))]
pub(crate) struct KingSafetyInput {
    pub(crate) us: SafetyMetrics,
    pub(crate) them: SafetyMetrics,
}

/// Openness in fixed point, 0 to [`OPEN_UNITY`]. Pawn colors must be absolute;
/// the northward shift counts White pawns blocked by Black pawns.
#[inline]
pub(crate) fn compute_openness_raw(white_pawns: u64, black_pawns: u64) -> i32 {
    let rams = (white_pawns << 8) & black_pawns;
    (OPEN_UNITY - RAM_SCALE * rams.count_ones() as i32 - PAWN_SCALE * (white_pawns | black_pawns).count_ones() as i32)
        .clamp(0, OPEN_UNITY)
}

impl SideMetrics {
    /// Lane order must match the mobility weight vectors.
    #[inline(always)]
    pub(crate) fn diff(&self, them: &Self) -> [i32; 4] {
        [
            self.mobility - them.mobility,
            self.shadow_mobility - them.shadow_mobility,
            self.threats - them.threats,
            self.shadow_threats - them.shadow_threats,
        ]
    }
}

impl SafetyMetrics {
    #[inline]
    fn score<T: EvalMath<Scalar = T>>(&self, w_shield: T, w_ortho: T, w_diag: T) -> T {
        let shield = T::from_i32(self.shield) * w_shield;
        let exposure = T::from_i32(self.ortho_exposure) * w_ortho + T::from_i32(self.diag_exposure) * w_diag;
        shield - exposure
    }

    /// Attacker weights use 0.1 cp per weak square; truncation is straight-through for gradients.
    #[inline]
    fn pressure<T: EvalMath<Scalar = T>>(&self, w_atk: T) -> T { ((w_atk * T::from_i32(self.weak)) / T::from_i32(10)).trunc() }

    #[inline(always)]
    fn analyze(ksq: Square, occ: Bitboard, atk_us: Bitboard, atk_them: Bitboard, our_pawns: Bitboard) -> Self {
        let zone = atk_king(ksq);
        Self {
            // Five or more attacked squares all take the maximum danger entry.
            attacked_zone: ((zone & atk_them).popcount() as usize).min(ATTACKER.len() - 1),
            weak: (zone & atk_them & !atk_us).popcount() as i32,
            shield: (zone & our_pawns).popcount() as i32,
            ortho_exposure: atk_rook(ksq, occ).popcount() as i32,
            diag_exposure: atk_bishop(ksq, occ).popcount() as i32,
        }
    }
}

#[inline]
pub(crate) fn compute_all(
    pos: &Position,
    maps: &SliderMaps,
    pinned_w: Bitboard,
    pinned_b: Bitboard,
    rows: Option<&XorBoard>,
) -> SpatialMetrics {
    let attacks = EvalAttacks::build(pos, maps, pinned_w, pinned_b);
    let mob_us = piece_mobility(pos, rows, Color::White, pinned_w, attacks.ksq_us, !attacks.pawn_atk_them);
    let mob_them = piece_mobility(pos, rows, Color::Black, pinned_b, attacks.ksq_them, !attacks.pawn_atk_us);
    let safety_us = SafetyMetrics::analyze(
        attacks.ksq_us,
        pos.occ,
        attacks.atk_us,
        attacks.atk_them,
        pos.pieces(PieceType::Pawn, Color::White),
    );
    let safety_them = SafetyMetrics::analyze(
        attacks.ksq_them,
        pos.occ,
        attacks.atk_them,
        attacks.atk_us,
        pos.pieces(PieceType::Pawn, Color::Black),
    );
    let metrics_us = side_metrics(pos.side_bb[Color::Black], attacks.atk_us, mob_us, attacks.pawn_atk_them, attacks.xray_us);
    let metrics_them = side_metrics(pos.side_bb[Color::White], attacks.atk_them, mob_them, attacks.pawn_atk_us, attacks.xray_them);
    SpatialMetrics { metrics_us, metrics_them, safety_us, safety_them }
}

/// Tapered, openness-interpolated score differential, `metrics_us` minus `metrics_them`.
#[inline(always)]
fn evaluate_score_diff<T: EvalMath<Scalar = T>>(
    metrics_us: &SideMetrics,
    metrics_them: &SideMetrics,
    openness: i32,
    phase: T,
    w_mg_o: T::Vec4,
    w_mg_c: T::Vec4,
    w_eg_o: T::Vec4,
    w_eg_c: T::Vec4,
) -> T {
    let diff = T::from_i32_array(metrics_us.diff(metrics_them));

    let o = T::Vec4::splat(openness);
    let c = T::Vec4::splat(OPEN_UNITY - openness);
    let rounding_bias = T::Vec4::splat(OPEN_UNITY / 2);

    let w_mg = (w_mg_o * o + w_mg_c * c + rounding_bias).srai::<10>();
    let w_eg = (w_eg_o * o + w_eg_c * c + rounding_bias).srai::<10>();

    // madd sums adjacent products, leaving two partial sums per half: lanes 0 and 1 for MG, 2 and 3 for EG.
    let w_packed = w_mg.pack_i16(w_eg);
    let diff_packed = diff.pack_i16(diff);
    let madd = diff_packed.madd(w_packed);

    let mg_sum = madd.extract::<0>() + madd.extract::<1>();
    let eg_sum = madd.extract::<2>() + madd.extract::<3>();

    taper(mg_sum, eg_sum, phase)
}

#[inline]
pub(crate) fn compute_openness(pos: &Position) -> i32 {
    let white = pos.pieces(PieceType::Pawn, Color::White);
    let black = pos.pieces(PieceType::Pawn, Color::Black);
    compute_openness_raw(white.0, black.0)
}

impl LinearTerm for MobilityTerm {
    type Upstream = TaperPair;
    type Input = MobilityInput;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, phase: T, acc: &mut Accumulators<T>) {
        acc.mobility = evaluate_score_diff::<T>(
            &features.spatial.metrics_us, &features.spatial.metrics_them, features.openness, phase, params.mg_mob_open,
            params.mg_mob_closed, params.eg_mob_open, params.eg_mob_closed,
        );
    }

    // Round each blended weight before tapering, matching the SIMD arithmetic shift.
    #[inline(always)]
    fn apply_input(input: MobilityInput, values: &[f64], phase: f64, acc: &mut Accumulators<f64>) {
        let open_base = LAYOUT.mobility_open_offset;
        let closed_base = LAYOUT.mobility_closed_offset;
        let unity = f64::from(OPEN_UNITY);
        let open_raw = f64::from(input.openness);
        let closed_raw = unity - open_raw;

        let mut diff = [0.0f64; 4];
        // SAFETY: diff is exactly the 4 lanes storeu writes.
        unsafe { input.diff.storeu(diff.as_mut_ptr()) };

        let blend = |lane: usize| {
            ((values[open_base + lane] * open_raw + values[closed_base + lane] * closed_raw + unity / 2.0) / unity).floor()
        };

        let mut mg_sum = 0.0;
        let mut eg_sum = 0.0;
        for (i, d) in diff.iter().enumerate() {
            mg_sum += d * blend(i);
            eg_sum += d * blend(4 + i);
        }

        acc.mobility = taper(mg_sum, eg_sum, phase);
    }

    // Rounding in the weight blend uses a straight-through derivative.
    #[inline(always)]
    fn scatter(input: MobilityInput, upstream: TaperPair, grads: &mut [f64]) {
        let open_base = LAYOUT.mobility_open_offset;
        let closed_base = LAYOUT.mobility_closed_offset;
        assert!(grads.len() >= open_base.max(closed_base) + 8, "MobilityTerm::scatter: grads too short");

        let open_frac = f64::from(input.openness) * INV_OPEN_UNITY;
        let closed_frac = 1.0 - open_frac;

        let mut scatter_lanes = |offset: usize, scale: f64| {
            // SAFETY: grads length verified by assert above and layout invariants.
            unsafe {
                let p = grads.as_mut_ptr().add(offset);
                (F64x4::loadu(p) + input.diff * F64x4::splat(scale)).storeu(p);
            }
        };

        scatter_lanes(open_base, upstream.d_mg * open_frac);
        scatter_lanes(open_base + 4, upstream.d_eg * open_frac);
        scatter_lanes(closed_base, upstream.d_mg * closed_frac);
        scatter_lanes(closed_base + 4, upstream.d_eg * closed_frac);
    }
}

impl TermSource<MobilityTerm> for SharedFeatures {
    type Input = MobilityInput;

    #[inline(always)]
    fn extract(&self) -> MobilityInput {
        let diff = self.spatial.metrics_us.diff(&self.spatial.metrics_them);
        MobilityInput { diff: F64x4::from(diff.map(f64::from)), openness: self.openness }
    }
}

impl LinearTerm for KingSafetyTerm {
    /// The combiner's MG taper is folded into every field, and the pressure halves
    /// arrive already signed, which is why scatter adds both of them.
    type Upstream = KingSafetyUpstream;
    type Input = KingSafetyInput;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
        let (us, them) = (&features.spatial.safety_us, &features.spatial.safety_them);

        acc.safety_us = us.score(params.w_shield, params.w_ortho, params.w_diag);
        acc.safety_them = them.score(params.w_shield, params.w_ortho, params.w_diag);
        acc.pressure_us = us.pressure(params.atk_weights[us.attacked_zone]);
        acc.pressure_them = them.pressure(params.atk_weights[them.attacked_zone]);
    }

    #[inline(always)]
    fn apply_input(input: KingSafetyInput, values: &[f64], _phase: f64, acc: &mut Accumulators<f64>) {
        let ks = LAYOUT.king_safety_offset;
        let ao = LAYOUT.attacker_offset;
        acc.safety_us = input.us.score(values[ks], values[ks + 1], values[ks + 2]);
        acc.safety_them = input.them.score(values[ks], values[ks + 1], values[ks + 2]);
        acc.pressure_us = input.us.pressure(values[ao + input.us.attacked_zone]);
        acc.pressure_them = input.them.pressure(values[ao + input.them.attacked_zone]);
    }

    #[inline(always)]
    fn scatter(input: KingSafetyInput, upstream: KingSafetyUpstream, grads: &mut [f64]) {
        let ks = LAYOUT.king_safety_offset;
        let ao = LAYOUT.attacker_offset;
        grads[ks] += upstream.safety * f64::from(input.us.shield - input.them.shield);
        grads[ks + 1] -= upstream.safety * f64::from(input.us.ortho_exposure - input.them.ortho_exposure);
        grads[ks + 2] -= upstream.safety * f64::from(input.us.diag_exposure - input.them.diag_exposure);

        grads[ao + input.us.attacked_zone] += upstream.pressure_us * (f64::from(input.us.weak) / 10.0);
        grads[ao + input.them.attacked_zone] += upstream.pressure_them * (f64::from(input.them.weak) / 10.0);
    }
}

impl TermSource<KingSafetyTerm> for SharedFeatures {
    type Input = KingSafetyInput;

    #[inline(always)]
    fn extract(&self) -> KingSafetyInput { KingSafetyInput { us: self.spatial.safety_us, them: self.spatial.safety_them } }
}

struct EvalAttacks {
    // Piece and pawn attacks, king excluded.
    atk_us: Bitboard,
    atk_them: Bitboard,
    pawn_atk_us: Bitboard,
    pawn_atk_them: Bitboard,
    ksq_us: Square,
    ksq_them: Square,
    xray_us: Bitboard,
    xray_them: Bitboard,
}

impl EvalAttacks {
    /// White is "us", Black is "them". Pinned knights contribute no attacks;
    /// pinned sliders contribute only along the king's ray. Pawn attacks remain unfiltered.
    #[inline(always)]
    fn build(pos: &Position, maps: &SliderMaps, pinned_w: Bitboard, pinned_b: Bitboard) -> Self {
        let us = pos.side_bb[Color::White];
        let them = pos.side_bb[Color::Black];
        let occ = pos.occ;

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

        // SliderMaps excludes pinned sliders. Restore their direct attacks along
        // the pin ray; their x-rays stay excluded to avoid extra battery probes.
        let (mut slider_atk_us, mut slider_atk_them, xray_us, xray_them) = (
            maps.w_ortho_direct() | maps.w_diag_direct(),
            maps.b_ortho_direct() | maps.b_diag_direct(),
            maps.w_ortho_xray() | maps.w_diag_xray(),
            maps.b_ortho_xray() | maps.b_diag_xray(),
        );

        // A queen sits in both sets, and the pin ray masks off whichever half it cannot play.
        let rq = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];
        let bq = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];

        let pinned_attacks = |pinned: Bitboard, ksq: Square| {
            let mut atk = Bitboard(0);
            for sq in pinned & rq {
                atk |= atk_rook(sq, occ) & line_bb(ksq, sq);
            }
            for sq in pinned & bq {
                atk |= atk_bishop(sq, occ) & line_bb(ksq, sq);
            }
            atk
        };

        slider_atk_us |= pinned_attacks(pinned_w, ksq_us);
        slider_atk_them |= pinned_attacks(pinned_b, ksq_them);

        let pawn_atk_us = pos.pawn_attacks(Color::White);
        let pawn_atk_them = pos.pawn_attacks(Color::Black);

        let atk_us = slider_atk_us | knight_atk_us | pawn_atk_us;
        let atk_them = slider_atk_them | knight_atk_them | pawn_atk_them;

        Self { atk_us, atk_them, pawn_atk_us, pawn_atk_them, ksq_us, ksq_them, xray_us, xray_them }
    }
}

fn piece_mobility(pos: &Position, rows: Option<&XorBoard>, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
    if let Some(rows) = rows {
        return rows.mobility(color, pinned, ksq, area);
    }

    let mut total = 0;
    for_each_piece_mobility(pos, color, pinned, ksq, area, |_, count| total += count as i32);
    total
}

/// Visits each non-king piece with its mobility count inside `area`.
/// Pinned knights contribute zero; pinned sliders stay on the pin ray.
/// Pawn attacks remain unfiltered by pins.
fn for_each_piece_mobility(
    pos: &Position,
    color: Color,
    pinned: Bitboard,
    ksq: Square,
    area: Bitboard,
    mut visit: impl FnMut(PieceType, u32),
) {
    let occ = pos.occ;

    for square in pos.side_bb[color] & !pos.role_bb[PieceType::King] {
        let piece = pos.piece_at(square);
        let attacks = match piece {
            PieceType::Pawn => atk_pawn(square, color),
            PieceType::Knight => atk_knight(square),
            PieceType::Bishop => atk_bishop(square, occ),
            PieceType::Rook => atk_rook(square, occ),
            PieceType::Queen => atk_queen(square, occ),
            _ => Bitboard(0),
        };

        let attacks = match piece {
            _ if !pinned.check_bit(square) => attacks,
            PieceType::Knight => Bitboard(0),
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => attacks & line_bb(ksq, square),
            _ => attacks,
        };

        visit(piece, (attacks & area).popcount());
    }
}

/// Visits each non-king piece with the mobility count used by evaluation.
pub fn piece_mobility_counts(pos: &Position, color: Color, visit: impl FnMut(PieceType, u32)) {
    for_each_piece_mobility(
        pos,
        color,
        pos.pinned_pieces(color),
        pos.pieces(PieceType::King, color).lsb(),
        !pos.pawn_attacks(color.opposite()),
        visit,
    );
}

#[inline(always)]
fn side_metrics(them: Bitboard, atk_us: Bitboard, mobility: i32, enemy_pawn_atk: Bitboard, xray_us: Bitboard) -> SideMetrics {
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
            let rows = XorBoard::from_position(&pos);

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
        assert_eq!(compute_openness(&pos), OPEN_UNITY - RAM_SCALE - 2 * PAWN_SCALE);
    }
}
