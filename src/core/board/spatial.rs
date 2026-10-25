//! SIMD attack sets and slider x-rays through one friendly blocker.

use crate::{
    core::{
        board::Position,
        defs::{Bitboard, Color, PieceType},
    },
    weave::*,
};

#[cfg(any(test, feature = "rigs", feature = "nostore"))]
#[inline(always)]
pub(super) fn atk_rooks(generators: U64x4, empty: U64x4) -> U64x4 {
    let n = fill_north(generators, empty).shift_north();
    let s = fill_south(generators, empty).shift_south();
    let e = fill_east(generators, empty).shift_east();
    let w = fill_west(generators, empty).shift_west();
    n | s | e | w
}

#[cfg(any(test, feature = "rigs", feature = "nostore"))]
#[inline(always)]
pub(super) fn atk_bishops(generators: U64x4, empty: U64x4) -> U64x4 {
    let ne = fill_northeast(generators, empty).shift_ne();
    let nw = fill_northwest(generators, empty).shift_nw();
    let se = fill_southeast(generators, empty).shift_se();
    let sw = fill_southwest(generators, empty).shift_sw();
    ne | nw | se | sw
}

#[cfg(any(test, feature = "rigs", feature = "nostore"))]
#[inline(always)]
pub(super) fn atk_knights(knights: U64x4) -> U64x4 {
    let mask_a = U64x4::splat(FILE_A);
    let mask_h = U64x4::splat(FILE_H);
    let mask_ab = U64x4::splat(FILE_A | (FILE_A << 1));
    let mask_gh = U64x4::splat(FILE_H | (FILE_H >> 1));

    let west1 = mask_a.andnot(knights).shr::<1>();
    let east1 = mask_h.andnot(knights).shl::<1>();
    let west2 = mask_ab.andnot(knights).shr::<2>();
    let east2 = mask_gh.andnot(knights).shl::<2>();

    let one_file = west1 | east1;
    let two_files = west2 | east2;
    (one_file.shl::<16>() | one_file.shr::<16>()) | (two_files.shl::<8>() | two_files.shr::<8>())
}

#[cfg(any(test, feature = "rigs", feature = "nostore"))]
#[inline(always)]
pub(super) fn atk_kings(kings: U64x4) -> U64x4 {
    let row = kings | kings.shift_east() | kings.shift_west();
    (row | row.shift_north() | row.shift_south()) & !kings
}

pub(crate) struct SliderMaps {
    ortho_direct: U64x4,
    ortho_xray: U64x4,
    diag_direct: U64x4,
    diag_xray: U64x4,
}

impl SliderMaps {
    #[inline]
    pub(crate) fn compute(pos: &Position, pinned_white: Bitboard, pinned_black: Bitboard) -> Self {
        let white = pos.side_bb[Color::White];
        let black = pos.side_bb[Color::Black];
        let rq = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];
        let bq = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];
        let unpinned_white = white & !pinned_white;
        let unpinned_black = black & !pinned_black;

        let empty = U64x4::splat(!pos.occ.0);
        let friendly = U64x4::from_lanes(white.0, black.0, 0, 0);
        let ortho_sliders = U64x4::from_lanes((rq & unpinned_white).0, (rq & unpinned_black).0, 0, 0);

        let (n, s, e, w) = (
            fill_north(ortho_sliders, empty).shift_north(),
            fill_south(ortho_sliders, empty).shift_south(),
            fill_east(ortho_sliders, empty).shift_east(),
            fill_west(ortho_sliders, empty).shift_west(),
        );

        let ortho_direct = n | s | e | w;

        let ortho_xray = {
            let n = fill_north(n & friendly, empty).shift_north();
            let s = fill_south(s & friendly, empty).shift_south();
            let e = fill_east(e & friendly, empty).shift_east();
            let w = fill_west(w & friendly, empty).shift_west();
            (n | s | e | w) & !ortho_direct
        };

        let diag_sliders = U64x4::from_lanes((bq & unpinned_white).0, (bq & unpinned_black).0, 0, 0);
        let (ne, nw, se, sw) = (
            fill_northeast(diag_sliders, empty).shift_ne(),
            fill_northwest(diag_sliders, empty).shift_nw(),
            fill_southeast(diag_sliders, empty).shift_se(),
            fill_southwest(diag_sliders, empty).shift_sw(),
        );

        let diag_direct = ne | nw | se | sw;

        let diag_xray = {
            let ne = fill_northeast(ne & friendly, empty).shift_ne();
            let nw = fill_northwest(nw & friendly, empty).shift_nw();
            let se = fill_southeast(se & friendly, empty).shift_se();
            let sw = fill_southwest(sw & friendly, empty).shift_sw();
            (ne | nw | se | sw) & !diag_direct
        };

        Self { ortho_direct, ortho_xray, diag_direct, diag_xray }
    }

    #[inline]
    pub(crate) fn w_ortho_direct(&self) -> Bitboard { Bitboard(self.ortho_direct.extract::<0>()) }
    #[inline]
    pub(crate) fn b_ortho_direct(&self) -> Bitboard { Bitboard(self.ortho_direct.extract::<1>()) }

    #[inline]
    pub(crate) fn w_diag_direct(&self) -> Bitboard { Bitboard(self.diag_direct.extract::<0>()) }
    #[inline]
    pub(crate) fn b_diag_direct(&self) -> Bitboard { Bitboard(self.diag_direct.extract::<1>()) }

    #[inline]
    pub(crate) fn w_ortho_xray(&self) -> Bitboard { Bitboard(self.ortho_xray.extract::<0>()) }
    #[inline]
    pub(crate) fn b_ortho_xray(&self) -> Bitboard { Bitboard(self.ortho_xray.extract::<1>()) }

    #[inline]
    pub(crate) fn w_diag_xray(&self) -> Bitboard { Bitboard(self.diag_xray.extract::<0>()) }
    #[inline]
    pub(crate) fn b_diag_xray(&self) -> Bitboard { Bitboard(self.diag_xray.extract::<1>()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::defs::Square;

    #[test]
    fn rook_xray_preserves_direction() {
        let pos = Position::from_fen("4k3/8/8/8/8/P7/8/R1N1K3 w - - 0 1");
        let maps = SliderMaps::compute(&pos, Bitboard::EMPTY, Bitboard::EMPTY);
        let xray = maps.w_ortho_xray();
        let a4 = Square(24);
        let d1 = Square(3);
        let b3 = Square(17);
        assert!(xray.check_bit(a4));
        assert!(xray.check_bit(d1));
        assert!(!xray.check_bit(b3));
    }

    #[test]
    fn bishop_xray_preserves_direction() {
        let pos = Position::from_fen("4k3/8/8/3p4/4P3/P7/8/R4K1B w - - 0 1");
        let maps = SliderMaps::compute(&pos, Bitboard::EMPTY, Bitboard::EMPTY);
        let d5 = Square(4 * 8 + 3);
        let f5 = Square(4 * 8 + 5);
        let diag_xray = maps.w_diag_xray();
        assert!(diag_xray.check_bit(d5));
        assert!(!diag_xray.check_bit(f5));
    }

    #[test]
    fn direct_attacks_include_friendly_slider() {
        let pos = Position::from_fen("4k3/8/8/8/8/R7/8/R3K3 w - - 0 1");
        let maps = SliderMaps::compute(&pos, Bitboard::EMPTY, Bitboard::EMPTY);
        let a3 = Square(16);
        let w_ortho_direct = maps.w_ortho_direct();
        assert!(w_ortho_direct.check_bit(a3));
    }
}
