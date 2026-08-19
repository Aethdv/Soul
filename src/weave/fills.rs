//! Kogge-Stone Parallel Prefix Fills (SIMD).
//!
//! This module is intentionally kept out of `MovePicker` and `movegen.rs`.
//! Move generation requires discrete `(from, to)` square pairs to encode `Move`
//! structs. Parallel prefix fills aggregate influence, destroying the `from`
//! square identity (i.e. you know a square is attacked, but not by which rook).
//! Disentangling a bulk fill back into discrete moves costs more cycles than
//! sequential `_pext_u64` (BMI2) lookups.
//!
//! Instead, this mathematically complete implementation powers the `SpatialTensor`
//! used during the Evaluation phase. By broadcasting a bitboard of all Rooks/Queens
//! into a `Vu64x4`, these functions calculate the aggregate board pressure, mobility,
//! and x-ray batteries of all sliders simultaneously in a single branchless sweep,
//! getting rid of piece-iteration loops entirely in the hot eval path.

use super::*;

/// Kogge-Stone occluded fill.
///
/// `generator`: sliding pieces,
/// `propagator`: empty squares.
/// Returns the fill including the generator squares.
///
/// Three doubling steps reach all seven distances, `shl` walking north and east,
/// `shr` south and west. A wrap mask, where given, is cleared from the propagator
/// before the walk.
macro_rules! kogge_fill {
    ($name:ident, $shift:ident, $s1:expr, $s2:expr, $s3:expr $(, $wrap:expr)?) => {
        #[inline(always)]
        pub fn $name(mut generator: Vu64x4, mut propagator: Vu64x4) -> Vu64x4 {
            $( propagator = Vu64x4::splat($wrap).andnot(propagator); )?
            generator |= propagator & generator.$shift::<$s1>();
            propagator &= propagator.$shift::<$s1>();
            generator |= propagator & generator.$shift::<$s2>();
            propagator &= propagator.$shift::<$s2>();
            generator |= propagator & generator.$shift::<$s3>();
            generator
        }
    };
}

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_H: u64 = 0x8080_8080_8080_8080;

// A file step off the h-file reappears on the a-file, so every eastward walk
// masks FILE_A and every westward one FILE_H. The vertical pair shifts straight
// off the end of the board and needs no mask.
kogge_fill!(fill_north, shl, 8, 16, 32);
kogge_fill!(fill_south, shr, 8, 16, 32);
kogge_fill!(fill_east, shl, 1, 2, 4, FILE_A);
kogge_fill!(fill_west, shr, 1, 2, 4, FILE_H);
kogge_fill!(fill_northeast, shl, 9, 18, 36, FILE_A);
kogge_fill!(fill_northwest, shl, 7, 14, 28, FILE_H);
kogge_fill!(fill_southeast, shr, 7, 14, 28, FILE_A);
kogge_fill!(fill_southwest, shr, 9, 18, 36, FILE_H);

impl Vu64x4 {
    #[inline(always)]
    pub fn shift_north(self) -> Self { self.shl::<8>() }

    #[inline(always)]
    pub fn shift_south(self) -> Self { self.shr::<8>() }

    #[inline(always)]
    pub fn shift_east(self) -> Self { Vu64x4::splat(FILE_H).andnot(self).shl::<1>() }

    #[inline(always)]
    pub fn shift_west(self) -> Self { Vu64x4::splat(FILE_A).andnot(self).shr::<1>() }

    #[inline(always)]
    pub fn shift_ne(self) -> Self { Vu64x4::splat(FILE_H).andnot(self).shl::<9>() }

    #[inline(always)]
    pub fn shift_nw(self) -> Self { Vu64x4::splat(FILE_A).andnot(self).shl::<7>() }

    #[inline(always)]
    pub fn shift_se(self) -> Self { Vu64x4::splat(FILE_H).andnot(self).shr::<7>() }

    #[inline(always)]
    pub fn shift_sw(self) -> Self { Vu64x4::splat(FILE_A).andnot(self).shr::<9>() }
}

/// Compute sliding attacks from a fill by shifting one step and removing the generators.
/// `fill_fn`: one of `fill_north`, etc.
/// `shift_fn`: matching single-step shift.
#[inline(always)]
pub fn attacks_from_fill<F, S>(generator: Vu64x4, pro: Vu64x4, fill_fn: F, shift_fn: S) -> Vu64x4
where
    F: Fn(Vu64x4, Vu64x4) -> Vu64x4,
    S: Fn(Vu64x4) -> Vu64x4,
{
    let filled = fill_fn(generator, pro);
    shift_fn(filled) & !generator
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount() {
        let v = Vu64x4::from_lanes(0, 1, 0xFF, u64::MAX);
        let pc = v.popcount();
        assert_eq!(pc.extract::<0>(), 0);
        assert_eq!(pc.extract::<1>(), 1);
        assert_eq!(pc.extract::<2>(), 8);
        assert_eq!(pc.extract::<3>(), 64);
    }

    #[test]
    fn test_fill_north() {
        // Single rook on A1, empty board
        let generator = Vu64x4::splat(1u64); // a1
        let propagator = Vu64x4::splat(!1u64); // everything except a1 is empty
        let fill = fill_north(generator, propagator);
        // Should fill the entire A file
        assert_eq!(fill.extract::<0>(), FILE_A);
    }

    #[test]
    fn test_shift_east_no_wrap() {
        // H1 = bit 7
        let bb = Vu64x4::splat(1u64 << 7);
        let shifted = bb.shift_east();
        // Should be zero: H file cannot shift east
        assert_eq!(shifted.extract::<0>(), 0);
    }

    #[test]
    fn test_bitwise_ops() {
        let a = Vu64x4::splat(0xFF00);
        let b = Vu64x4::splat(0x0FF0);
        let and = a & b;
        let or = a | b;
        let xor = a ^ b;
        assert_eq!(and.extract::<0>(), 0x0F00);
        assert_eq!(or.extract::<0>(), 0xFFF0);
        assert_eq!(xor.extract::<0>(), 0xF0F0);
    }
}
