// Typed SIMD wrappers — not CamelCase by design.
#![allow(non_camel_case_types)]

//! Kogge-Stone Parallel Prefix Fills (SIMD).
//!
//! Architectural Note:
//! This module is not currently used in `MovePicker` generation (which relies
//! on `_pext_u64` BMI2 lookups for sequential square-by-square generation).
//!
//! Instead, this mathematically complete implementation is preserved for the next
//! tuning cycle to provide Bulk SIMD Mobility and Zone Control generation during
//! the Evaluation phase. By broadcasting a bitboard of all Rooks/Queens into a `Vu64x4`,
//! these functions allow us to calculate the aggregate board pressure of all sliders
//! simultaneously in a single branchless sweep, eliminating piece-iteration loops
//! within the hot evaluation path.

use super::*;

// ──────── Kogge-Stone Parallel Prefix Fills — Pure ALU, 8 Directions ────────

/// Kogge-Stone occluded fill: left shift variant (North, East, NE, NW).
///
/// `generator`: sliding pieces,
/// `propagator`: empty squares.
/// Returns the fill including the generator squares.
macro_rules! kogge_fill_left {
    ($name:ident, $s1:expr, $s2:expr, $s3:expr) => {
        #[inline(always)]
        pub fn $name(mut generator: Vu64x4, mut propagator: Vu64x4) -> Vu64x4 {
            generator |= propagator & generator.shl::<$s1>();
            propagator &= propagator.shl::<$s1>();
            generator |= propagator & generator.shl::<$s2>();
            propagator &= propagator.shl::<$s2>();
            generator |= propagator & generator.shl::<$s3>();
            generator
        }
    };
}

/// Kogge-Stone occluded fill: left shift variant with file mask.
macro_rules! kogge_fill_left_masked {
    ($name:ident, $s1:expr, $s2:expr, $s3:expr, $mask:expr) => {
        #[inline(always)]
        pub fn $name(mut generator: Vu64x4, propagator: Vu64x4) -> Vu64x4 {
            let mut propagator = Vu64x4::splat($mask).andnot(propagator);
            generator |= propagator & generator.shl::<$s1>();
            propagator &= propagator.shl::<$s1>();
            generator |= propagator & generator.shl::<$s2>();
            propagator &= propagator.shl::<$s2>();
            generator |= propagator & generator.shl::<$s3>();
            generator
        }
    };
}

/// Kogge-Stone occluded fill: right shift variant (South, West, SE, SW).
macro_rules! kogge_fill_right {
    ($name:ident, $s1:expr, $s2:expr, $s3:expr) => {
        #[inline(always)]
        pub fn $name(mut generator: Vu64x4, mut propagator: Vu64x4) -> Vu64x4 {
            generator |= propagator & generator.shr::<$s1>();
            propagator &= propagator.shr::<$s1>();
            generator |= propagator & generator.shr::<$s2>();
            propagator &= propagator.shr::<$s2>();
            generator |= propagator & generator.shr::<$s3>();
            generator
        }
    };
}

/// Kogge-Stone occluded fill; right shift variant with file mask.
macro_rules! kogge_fill_right_masked {
    ($name:ident, $s1:expr, $s2:expr, $s3:expr, $mask:expr) => {
        #[inline(always)]
        pub fn $name(mut generator: Vu64x4, propagator: Vu64x4) -> Vu64x4 {
            let mut propagator = Vu64x4::splat($mask).andnot(propagator);
            generator |= propagator & generator.shr::<$s1>();
            propagator &= propagator.shr::<$s1>();
            generator |= propagator & generator.shr::<$s2>();
            propagator &= propagator.shr::<$s2>();
            generator |= propagator & generator.shr::<$s3>();
            generator
        }
    };
}

pub const FILE_A: u64 = 0x0101_0101_0101_0101;
pub const FILE_H: u64 = 0x8080_8080_8080_8080;

// North (+8): no file mask needed
kogge_fill_left!(fill_north, 8, 16, 32);
// South (-8): no file mask needed
kogge_fill_right!(fill_south, 8, 16, 32);
// East (+1): mask FILE_A on propagator (prevents H→A wrap on left shift)
kogge_fill_left_masked!(fill_east, 1, 2, 4, FILE_A);
// West (-1): mask FILE_H on propagator (prevents A→H wrap on right shift)
kogge_fill_right_masked!(fill_west, 1, 2, 4, FILE_H);
// NorthEast (+9): mask FILE_A
kogge_fill_left_masked!(fill_northeast, 9, 18, 36, FILE_A);
// NorthWest (+7): mask FILE_H
kogge_fill_left_masked!(fill_northwest, 7, 14, 28, FILE_H);
// SouthEast (-7): mask FILE_A
kogge_fill_right_masked!(fill_southeast, 7, 14, 28, FILE_A);
// SouthWest (-9): mask FILE_H
kogge_fill_right_masked!(fill_southwest, 9, 18, 36, FILE_H);

// ──────── Directional Single-Step Shifts (for deriving attacks from fills) ────────

impl Vu64x4 {
    #[inline(always)]
    pub fn shift_north(self) -> Self {
        self.shl::<8>()
    }

    #[inline(always)]
    pub fn shift_south(self) -> Self {
        self.shr::<8>()
    }

    #[inline(always)]
    pub fn shift_east(self) -> Self {
        Vu64x4::splat(FILE_H).andnot(self).shl::<1>()
    }

    #[inline(always)]
    pub fn shift_west(self) -> Self {
        Vu64x4::splat(FILE_A).andnot(self).shr::<1>()
    }

    #[inline(always)]
    pub fn shift_ne(self) -> Self {
        Vu64x4::splat(FILE_H).andnot(self).shl::<9>()
    }

    #[inline(always)]
    pub fn shift_nw(self) -> Self {
        Vu64x4::splat(FILE_A).andnot(self).shl::<7>()
    }

    #[inline(always)]
    pub fn shift_se(self) -> Self {
        Vu64x4::splat(FILE_H).andnot(self).shr::<7>()
    }

    #[inline(always)]
    pub fn shift_sw(self) -> Self {
        Vu64x4::splat(FILE_A).andnot(self).shr::<9>()
    }
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

// ──────── Tests ────────

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
        // Should be zero — H file cannot shift east
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
