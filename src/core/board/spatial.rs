//! Parallelized spatial influence mapping.
//!
//! Uses SIMD intrinsics to simultaneously compute orthogonal and diagonal attack rays
//! for multiple pieces, separating direct influence from X-ray (battery) paths.

use crate::{
    core::{
        board::Position,
        defs::{Color, PieceType},
    },
    weave::*,
};

/// Rook attacks (all 4 cardinal directions).
#[inline(always)]
pub fn atk_rook(generator: Vu64x4, empty: Vu64x4) -> Vu64x4 {
    let n = fill_north(generator, empty).shift_north();
    let s = fill_south(generator, empty).shift_south();
    let e = fill_east(generator, empty).shift_east();
    let w = fill_west(generator, empty).shift_west();
    (n | s | e | w) & !generator
}

/// Bishop attacks (all 4 diagonal directions).
#[inline(always)]
pub fn atk_bishop(generator: Vu64x4, empty: Vu64x4) -> Vu64x4 {
    let ne = fill_northeast(generator, empty).shift_ne();
    let nw = fill_northwest(generator, empty).shift_nw();
    let se = fill_southeast(generator, empty).shift_se();
    let sw = fill_southwest(generator, empty).shift_sw();
    (ne | nw | se | sw) & !generator
}

/// Knight attacks.
#[inline(always)]
pub fn atk_knight(knights: Vu64x4) -> Vu64x4 {
    let mask_a = Vu64x4::splat(FILE_A);
    let mask_h = Vu64x4::splat(FILE_H);
    let mask_ab = Vu64x4::splat(FILE_A | (FILE_A << 1));
    let mask_gh = Vu64x4::splat(FILE_H | (FILE_H >> 1));

    let west1 = mask_a.andnot(knights).shr::<1>();
    let east1 = mask_h.andnot(knights).shl::<1>();
    let west2 = mask_ab.andnot(knights).shr::<2>();
    let east2 = mask_gh.andnot(knights).shl::<2>();

    let h1 = west1 | east1;
    let h2 = west2 | east2;
    (h1.shl::<16>() | h1.shr::<16>()) | (h2.shl::<8>() | h2.shr::<8>())
}

/// King attacks: shift E/W, merge, then shift N/S.
#[inline(always)]
pub fn atk_king(kings: Vu64x4) -> Vu64x4 {
    let attacks = kings | kings.shift_east() | kings.shift_west();
    (attacks | attacks.shift_north() | attacks.shift_south()) & !kings
}

/// All eight spatial influence surfaces, computed in two vectorized passes.
///
/// Orthogonal fields (`ortho_direct`, `ortho_xray`):
/// - Lane 0: White (R+Q) orthogonal attack set
/// - Lane 1: Black (R+Q) orthogonal attack set
/// - Lane 2: Reserved (unused, currently 0)
/// - Lane 3: Reserved (unused, currently 0)
///
/// Diagonal fields (`diag_direct`, `diag_xray`):
/// - Same structure with (B+Q) generators.
///
/// All four lanes per field are computed simultaneously by a single set of
/// AVX2 `vpsllq`/`vpsrlq` instructions applied to the packed `Vu64x4`.
pub struct SpatialTensor {
    pub ortho_direct: Vu64x4,
    pub ortho_xray: Vu64x4,
    pub diag_direct: Vu64x4,
    pub diag_xray: Vu64x4,
}

impl SpatialTensor {
    #[inline]
    pub fn compute(pos: &Position, pinned_w: u64, pinned_b: u64) -> Self {
        let occ = pos.occ.0;
        let empty = Vu64x4::splat(!occ);
        let w_pcs = pos.side_bb[Color::White].0;
        let b_pcs = pos.side_bb[Color::Black].0;

        let w_rq = (pos.role_bb[PieceType::Rook] & pos.side_bb[Color::White]
            | pos.role_bb[PieceType::Queen] & pos.side_bb[Color::White])
            .0
            & !pinned_w;

        let b_rq = (pos.role_bb[PieceType::Rook] & pos.side_bb[Color::Black]
            | pos.role_bb[PieceType::Queen] & pos.side_bb[Color::Black])
            .0
            & !pinned_b;

        let w_bq = (pos.role_bb[PieceType::Bishop] & pos.side_bb[Color::White]
            | pos.role_bb[PieceType::Queen] & pos.side_bb[Color::White])
            .0
            & !pinned_w;

        let b_bq = (pos.role_bb[PieceType::Bishop] & pos.side_bb[Color::Black]
            | pos.role_bb[PieceType::Queen] & pos.side_bb[Color::Black])
            .0
            & !pinned_b;

        let gen_ortho = Vu64x4::from_lanes(w_rq, b_rq, 0, 0); // Lanes: [W_direct, B_direct, unused, unused]
        let us_pcs_ortho = Vu64x4::from_lanes(w_pcs, b_pcs, 0, 0);

        let (gen_n, gen_s, gen_e, gen_w) = (
            fill_north(gen_ortho, empty).shift_north(),
            fill_south(gen_ortho, empty).shift_south(),
            fill_east(gen_ortho, empty).shift_east(),
            fill_west(gen_ortho, empty).shift_west(),
        );

        let ortho_direct = gen_n | gen_s | gen_e | gen_w;

        // Computes orthogonal X-ray ("shadow") attacks passing through exactly one friendly piece.
        //
        // Friendly pieces hit during the first pass (`gen_* & us_pcs_ortho`) serve as
        // secondary generators. Refilling from these collision points continues the rays
        // past the first blocker until the next obstruction.
        let ortho_xray = {
            let (n, s, e, w) = (
                fill_north(gen_n & us_pcs_ortho, empty).shift_north(),
                fill_south(gen_s & us_pcs_ortho, empty).shift_south(),
                fill_east(gen_e & us_pcs_ortho, empty).shift_east(),
                fill_west(gen_w & us_pcs_ortho, empty).shift_west(),
            );

            (n | s | e | w) & !ortho_direct
        };

        let gen_diag = Vu64x4::from_lanes(w_bq, b_bq, 0, 0);
        let us_pcs_diag = Vu64x4::from_lanes(w_pcs, b_pcs, 0, 0);

        let (gen_ne, gen_nw, gen_se, gen_sw) = (
            fill_northeast(gen_diag, empty).shift_ne(),
            fill_northwest(gen_diag, empty).shift_nw(),
            fill_southeast(gen_diag, empty).shift_se(),
            fill_southwest(gen_diag, empty).shift_sw(),
        );

        let diag_direct = gen_ne | gen_nw | gen_se | gen_sw;

        let diag_xray = {
            let (ne, nw, se, sw) = (
                fill_northeast(gen_ne & us_pcs_diag, empty).shift_ne(),
                fill_northwest(gen_nw & us_pcs_diag, empty).shift_nw(),
                fill_southeast(gen_se & us_pcs_diag, empty).shift_se(),
                fill_southwest(gen_sw & us_pcs_diag, empty).shift_sw(),
            );

            (ne | nw | se | sw) & !diag_direct
        };

        Self { ortho_direct, ortho_xray, diag_direct, diag_xray }
    }
    #[inline]
    pub fn w_ortho_direct(&self) -> u64 {
        self.ortho_direct.extract::<0>()
    }
    #[inline]
    pub fn b_ortho_direct(&self) -> u64 {
        self.ortho_direct.extract::<1>()
    }

    #[inline]
    pub fn w_diag_direct(&self) -> u64 {
        self.diag_direct.extract::<0>()
    }
    #[inline]
    pub fn b_diag_direct(&self) -> u64 {
        self.diag_direct.extract::<1>()
    }

    /// Squares white controls ONLY via shadow (battery/X-ray),
    /// strictly through exactly 1 own piece blocker.
    #[inline]
    pub fn w_ortho_xray(&self) -> u64 {
        self.ortho_xray.extract::<0>()
    }
    #[inline]
    pub fn b_ortho_xray(&self) -> u64 {
        self.ortho_xray.extract::<1>()
    }
    #[inline]
    pub fn w_diag_xray(&self) -> u64 {
        self.diag_xray.extract::<0>()
    }
    #[inline]
    pub fn b_diag_xray(&self) -> u64 {
        self.diag_xray.extract::<1>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Position;

    #[test]
    fn test_phantom_xray_rook() {
        let pos = Position::from_fen("4k3/8/8/8/8/P7/8/R1N1K3 w - - 0 1");
        let st = SpatialTensor::compute(&pos, 0, 0);
        let xray = st.w_ortho_xray();
        let a4 = 1u64 << 24;
        let d1 = 1u64 << 3;
        let b3 = 1u64 << 17;
        assert!((xray & a4) != 0, "Should X-ray North from A3 to A4");
        assert!((xray & d1) != 0, "Should X-ray East from C1 to D1");
        assert!((xray & b3) == 0, "Phantom X-ray detected! A3 should not radiate East to B3");
    }

    #[test]
    fn test_phantom_xray_bishop() {
        let pos = Position::from_fen("4k3/8/8/3p4/4P3/P7/8/R4K1B w - - 0 1");
        let st = SpatialTensor::compute(&pos, 0, 0);
        let d5 = 1u64 << (4 * 8 + 3); // 35
        let f5 = 1u64 << (4 * 8 + 5); // 37
        let diag_xray = st.w_diag_xray();
        assert!((diag_xray & d5) != 0, "Should X-ray NW from E4 to D5");
        assert!((diag_xray & f5) == 0, "Phantom X-ray detected! E4 should not radiate NE (only NW)");
    }

    #[test]
    fn test_self_occlusion() {
        let pos = Position::from_fen("4k3/8/8/8/8/R7/8/R3K3 w - - 0 1");
        let st = SpatialTensor::compute(&pos, 0, 0);
        let a3 = 1u64 << 16;
        let w_ortho_direct = st.w_ortho_direct();
        assert!((w_ortho_direct & a3) != 0, "Rook at A1 should defend Rook at A3");
    }
}
