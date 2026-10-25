//! Bitboards, squares and spatial primitives.

use std::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr, Sub, SubAssign},
};

use crate::core::defs::Direction;

pub(crate) const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
pub(crate) const FILE_B: Bitboard = FILE_A << 1;
pub(crate) const FILE_C: Bitboard = FILE_A << 2;
pub(crate) const FILE_D: Bitboard = FILE_A << 3;
pub(crate) const FILE_E: Bitboard = FILE_A << 4;
pub(crate) const FILE_F: Bitboard = FILE_A << 5;
pub(crate) const FILE_G: Bitboard = FILE_A << 6;
pub(crate) const FILE_H: Bitboard = FILE_A << 7;

pub(crate) const NOT_A: Bitboard = !FILE_A;
pub(crate) const NOT_H: Bitboard = !FILE_H;
pub(crate) const NOT_AB: Bitboard = !(FILE_A | FILE_B);
pub(crate) const NOT_GH: Bitboard = !(FILE_G | FILE_H);

pub(crate) const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
pub(crate) const RANK_2: Bitboard = RANK_1 << 8;
pub(crate) const RANK_3: Bitboard = RANK_1 << 16;
pub(crate) const RANK_4: Bitboard = RANK_1 << 24;
pub(crate) const RANK_5: Bitboard = RANK_1 << 32;
pub(crate) const RANK_6: Bitboard = RANK_1 << 40;
pub(crate) const RANK_7: Bitboard = RANK_1 << 48;
pub(crate) const RANK_8: Bitboard = RANK_1 << 56;

pub(crate) const FILE_MASKS: [Bitboard; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];
pub(crate) const RANK_MASKS: [Bitboard; 8] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8];

/// A set of squares, one bit each: bit 0 = A1, bit 7 = H1, bit 56 = A8, bit 63 = H8.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub(crate) struct Bitboard(pub(crate) u64);

/// `rank · 8 + file`, both counted from zero: A1 = 0, H1 = 7, H8 = 63.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, std::marker::ConstParamTy)]
#[repr(transparent)]
pub(crate) struct Square(pub(crate) u8);

const _: () = assert!(std::mem::size_of::<Square>() == 1);
const _: () = assert!(std::mem::size_of::<Option<Square>>() == 2);
const _: () = assert!(std::mem::size_of::<Bitboard>() == 8);

impl Bitboard {
    pub(crate) const EMPTY: Self = Self(0);

    #[inline(always)]
    #[must_use]
    pub(crate) const fn is_empty(self) -> bool { self.0 == 0 }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn is_not_empty(self) -> bool { self.0 != 0 }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn more_than_one(self) -> bool { self.0 & self.0.wrapping_sub(1) != 0 }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn popcount(self) -> u32 { self.0.count_ones() }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn lsb(self) -> Square {
        debug_assert!(self.is_not_empty(), "lsb() called on empty bitboard");
        Square(self.0.trailing_zeros() as u8)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn check_bit(self, sq: Square) -> bool { self.0 & (1u64 << sq.0) != 0 }

    #[inline(always)]
    pub(super) fn set_bit(&mut self, sq: Square) { self.0 |= 1u64 << sq.0; }

    #[inline(always)]
    pub(super) fn clear_bit(&mut self, sq: Square) { self.0 &= !(1u64 << sq.0); }

    #[inline(always)]
    pub(crate) fn pop_lsb(&mut self) -> Square {
        debug_assert!(self.is_not_empty());
        let sq = self.lsb();
        self.0 &= self.0.wrapping_sub(1);
        sq
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn iter(self) -> BitboardIter { BitboardIter(self) }

    /// Smears every set bit across its file; shifts by multiples of 8 cannot cross files.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn file_fill(self) -> Self { self.north_fill().south_fill() }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn north_fill(self) -> Self {
        let mut b = self.0;
        b |= b << 8;
        b |= b << 16;
        b |= b << 32;
        Bitboard(b)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn south_fill(self) -> Self {
        let mut b = self.0;
        b |= b >> 8;
        b |= b >> 16;
        b |= b >> 32;
        Bitboard(b)
    }

    #[inline(always)]
    pub(crate) const fn shift(self, dir: Direction) -> Self {
        match dir {
            Direction::North => Bitboard(self.0 << 8),
            Direction::South => Bitboard(self.0 >> 8),
            Direction::East => Bitboard((self.0 & NOT_H.0) << 1),
            Direction::West => Bitboard((self.0 & NOT_A.0) >> 1),
            Direction::NorthEast => Bitboard((self.0 & NOT_H.0) << 9),
            Direction::NorthWest => Bitboard((self.0 & NOT_A.0) << 7),
            Direction::SouthEast => Bitboard((self.0 & NOT_H.0) >> 7),
            Direction::SouthWest => Bitboard((self.0 & NOT_A.0) >> 9),
        }
    }
}

impl Square {
    #[inline(always)]
    #[must_use]
    pub(crate) const fn from_coords(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8, "file or rank out of range");
        Self((rank << 3) | file)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn file(self) -> u8 { self.0 & 7 }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn rank(self) -> u8 { self.0 >> 3 }

    /// The number of king moves between the two squares.
    #[inline(always)]
    #[must_use]
    pub(crate) const fn chebyshev_distance(self, other: Self) -> u8 {
        let file_d = self.file().abs_diff(other.file());
        let rank_d = self.rank().abs_diff(other.rank());
        file_d.max(rank_d)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn bitboard(self) -> Bitboard { Bitboard(1u64 << self.0) }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn offset_unchecked(self, off: i8) -> Self {
        let result = (self.0.cast_signed() + off).cast_unsigned();
        debug_assert!(result < 64, "Square::offset produced invalid square");
        Self(result)
    }

    #[inline(always)]
    #[must_use]
    pub(crate) const fn as_usize(self) -> usize { self.0 as usize }

    /// e1 → e8.
    #[cfg(any(test, feature = "dataset"))]
    #[inline(always)]
    #[must_use]
    pub(crate) const fn flip_rank(self) -> Self { Self(self.0 ^ 56) }

    /// Lowercase algebraic, `a1` through `h8`.
    pub(crate) fn from_algebraic(s: &str) -> Option<Self> {
        let &[file, rank] = s.as_bytes() else { return None };
        let (file, rank) = (file.wrapping_sub(b'a'), rank.wrapping_sub(b'1'));
        (file < 8 && rank < 8).then(|| Self::from_coords(file, rank))
    }
}

const impl From<Square> for u8 {
    #[inline(always)]
    fn from(sq: Square) -> Self { sq.0 }
}

const impl From<Square> for usize {
    #[inline(always)]
    fn from(sq: Square) -> Self { sq.0 as usize }
}

const impl BitXor<u8> for Square {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: u8) -> Self { Self(self.0 ^ rhs) }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub(crate) struct BitboardIter(Bitboard);

impl Iterator for BitboardIter {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> { if self.0.is_empty() { None } else { Some(self.0.pop_lsb()) } }

    #[inline(always)]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.0.popcount() as usize;
        (n, Some(n))
    }
}

impl ExactSizeIterator for BitboardIter {}
impl std::iter::FusedIterator for BitboardIter {}

impl IntoIterator for Bitboard {
    type Item = Square;
    type IntoIter = BitboardIter;

    #[inline(always)]
    fn into_iter(self) -> BitboardIter { self.iter() }
}

macro_rules! impl_bb_op {
    ($trait:ident, $func:ident, $assign_trait:ident, $assign_func:ident) => {
        const impl $trait for Bitboard {
            type Output = Self;

            #[inline(always)]
            fn $func(self, rhs: Self) -> Self { Self(self.0.$func(rhs.0)) }
        }

        const impl $trait<u64> for Bitboard {
            type Output = Self;
            #[inline(always)]
            fn $func(self, rhs: u64) -> Self { Self(self.0.$func(rhs)) }
        }

        impl $assign_trait for Bitboard {
            #[inline(always)]
            fn $assign_func(&mut self, rhs: Self) { self.0.$assign_func(rhs.0); }
        }

        impl $assign_trait<u64> for Bitboard {
            #[inline(always)]
            fn $assign_func(&mut self, rhs: u64) { self.0.$assign_func(rhs); }
        }
    };
}

impl_bb_op!(BitAnd, bitand, BitAndAssign, bitand_assign);
impl_bb_op!(BitOr, bitor, BitOrAssign, bitor_assign);
impl_bb_op!(BitXor, bitxor, BitXorAssign, bitxor_assign);
impl_bb_op!(Sub, sub, SubAssign, sub_assign);

const impl Not for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self { Self(!self.0) }
}

const impl Shl<u8> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u8) -> Self { Self(self.0 << rhs) }
}

const impl Shr<u8> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u8) -> Self { Self(self.0 >> rhs) }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "Bitboard({:#018x})", self.0) }
}

impl fmt::Display for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for rank in (0u8..8).rev() {
            for file in 0u8..8 {
                if file > 0 {
                    f.write_str(" ")?;
                }

                let sq = Square::from_coords(file, rank);
                f.write_str(if self.check_bit(sq) { "X" } else { "." })?;
            }

            if rank > 0 {
                writeln!(f)?;
            }
        }
        Ok(())
    }
}
