//! Bitboards, squares, and core spatial primitives.

use std::{
    fmt,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign, BitXor, BitXorAssign, Not, Shl, Shr, Sub, SubAssign},
};

use crate::core::defs::Direction;

pub const FILE_A: Bitboard = Bitboard(0x0101_0101_0101_0101);
pub const FILE_B: Bitboard = FILE_A << 1;
pub const FILE_C: Bitboard = FILE_A << 2;
pub const FILE_D: Bitboard = FILE_A << 3;
pub const FILE_E: Bitboard = FILE_A << 4;
pub const FILE_F: Bitboard = FILE_A << 5;
pub const FILE_G: Bitboard = FILE_A << 6;
pub const FILE_H: Bitboard = FILE_A << 7;

pub const NOT_A: Bitboard = !FILE_A;
pub const NOT_H: Bitboard = !FILE_H;
pub const NOT_AB: Bitboard = !(FILE_A | FILE_B);
pub const NOT_GH: Bitboard = !(FILE_G | FILE_H);

pub const RANK_1: Bitboard = Bitboard(0x0000_0000_0000_00FF);
pub const RANK_2: Bitboard = RANK_1 << 8;
pub const RANK_3: Bitboard = RANK_1 << 16;
pub const RANK_4: Bitboard = RANK_1 << 24;
pub const RANK_5: Bitboard = RANK_1 << 32;
pub const RANK_6: Bitboard = RANK_1 << 40;
pub const RANK_7: Bitboard = RANK_1 << 48;
pub const RANK_8: Bitboard = RANK_1 << 56;

pub const FILE_MASKS: [Bitboard; 8] = [FILE_A, FILE_B, FILE_C, FILE_D, FILE_E, FILE_F, FILE_G, FILE_H];
pub const RANK_MASKS: [Bitboard; 8] = [RANK_1, RANK_2, RANK_3, RANK_4, RANK_5, RANK_6, RANK_7, RANK_8];

/// A 64-bit bitboard representing a set of squares on a chess board.
///
/// Each bit maps to a square: bit 0 = A1, bit 7 = H1, bit 56 = A8, bit 63 = H8.
/// Supports all standard bitwise operations and iteration over set squares.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct Bitboard(pub u64);

/// A square on the chess board, encoded as an index in `0..64`.
///
/// Layout: `index = rank · 8 + file` where file ∈ [0, 7] (A–H) and rank ∈ [0, 7] (1–8).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default, Debug, std::marker::ConstParamTy)]
#[repr(transparent)]
pub struct Square(pub u8);

const _: () = assert!(std::mem::size_of::<Square>() == 1);
const _: () = assert!(std::mem::size_of::<Option<Square>>() == 2);
const _: () = assert!(std::mem::size_of::<Bitboard>() == 8);

impl Bitboard {
    /// Empty bitboard with no bits set.
    pub const EMPTY: Self = Self(0);

    /// Returns `true` if no bits are set.
    #[inline(always)]
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }

    /// Returns `true` if at least one bit is set.
    #[inline(always)]
    #[must_use]
    pub const fn is_not_empty(self) -> bool {
        self.0 != 0
    }

    /// Returns `true` if more than one bit is set.
    #[inline(always)]
    #[must_use]
    pub const fn more_than_one(self) -> bool {
        self.0 & self.0.wrapping_sub(1) != 0
    }

    /// Returns the number of set bits (population count).
    #[inline(always)]
    #[must_use]
    pub const fn popcount(self) -> u32 {
        self.0.count_ones()
    }

    /// Returns the least significant set bit as a [`Square`].
    ///
    /// # Panics
    /// In debug if the bitboard is empty.
    #[inline(always)]
    #[must_use]
    pub const fn lsb(self) -> Square {
        debug_assert!(self.is_not_empty(), "lsb() called on empty bitboard");
        Square(self.0.trailing_zeros() as u8)
    }

    /// Returns `true` if the bit for `sq` is set.
    #[inline(always)]
    #[must_use]
    pub const fn check_bit(self, sq: Square) -> bool {
        self.0 & (1u64 << sq.0) != 0
    }

    /// Sets the bit for `sq`.
    #[inline(always)]
    pub fn set_bit(&mut self, sq: Square) {
        self.0 |= 1u64 << sq.0;
    }

    /// Clears the bit for `sq`.
    #[inline(always)]
    pub fn clear_bit(&mut self, sq: Square) {
        self.0 &= !(1u64 << sq.0);
    }

    /// Removes and returns the least significant set bit as a [`Square`],
    /// or `None` if the bitboard is empty.
    #[inline(always)]
    pub fn pop_lsb(&mut self) -> Square {
        debug_assert!(self.is_not_empty());
        let sq = self.lsb();
        self.0 &= self.0.wrapping_sub(1);
        sq
    }

    #[inline(always)]
    #[must_use]
    pub const fn iter(self) -> BitboardIter {
        BitboardIter(self)
    }

    /// Smears every set bit across its entire file. Shifts are multiples of 8,
    /// which never cross a file boundary, so no file mask is needed.
    #[inline(always)]
    #[must_use]
    pub const fn file_fill(self) -> Self {
        let mut b = self.0;
        b |= b << 8;
        b |= b << 16;
        b |= b << 32;
        b |= b >> 8;
        b |= b >> 16;
        b |= b >> 32;
        Bitboard(b)
    }

    /// Smears every set bit northward (toward rank 8), inclusive.
    #[inline(always)]
    #[must_use]
    pub const fn north_fill(self) -> Self {
        let mut b = self.0;
        b |= b << 8;
        b |= b << 16;
        b |= b << 32;
        Bitboard(b)
    }

    /// Smears every set bit southward (toward rank 1), inclusive.
    #[inline(always)]
    #[must_use]
    pub const fn south_fill(self) -> Self {
        let mut b = self.0;
        b |= b >> 8;
        b |= b >> 16;
        b |= b >> 32;
        Bitboard(b)
    }

    #[inline(always)]
    pub const fn shift(self, dir: Direction) -> Self {
        match dir {
            Direction::North => Bitboard(self.0 << 8),
            Direction::South => Bitboard(self.0 >> 8),
            Direction::East => Bitboard((self.0 & !FILE_H.0) << 1),
            Direction::West => Bitboard((self.0 & !FILE_A.0) >> 1),
            Direction::NorthEast => Bitboard((self.0 & !FILE_H.0) << 9),
            Direction::NorthWest => Bitboard((self.0 & !FILE_A.0) << 7),
            Direction::SouthEast => Bitboard((self.0 & !FILE_H.0) >> 7),
            Direction::SouthWest => Bitboard((self.0 & !FILE_A.0) >> 9),
        }
    }
}

impl Square {
    /// Creates a [`Square`] from a raw index.
    ///
    /// # Panics
    /// if `sq >= 64`.
    #[inline(always)]
    #[must_use]
    pub const fn new(sq: u8) -> Self {
        debug_assert!(sq < 64, "Square index out of range");
        Self(sq)
    }

    /// Creates a [`Square`] from file and rank indices (both in `0..8`).
    ///
    /// # Panics
    /// In debug if either index is out of range.
    #[inline(always)]
    #[must_use]
    pub const fn from_coords(file: u8, rank: u8) -> Self {
        debug_assert!(file < 8 && rank < 8, "file or rank out of range");
        Self((rank << 3) | file)
    }

    /// Returns the file index (0 = A-file, 7 = H-file).
    #[inline(always)]
    #[must_use]
    pub const fn file(self) -> u8 {
        self.0 & 7
    }

    /// Returns the rank index (0 = 1st rank, 7 = 8th rank).
    #[inline(always)]
    #[must_use]
    pub const fn rank(self) -> u8 {
        self.0 >> 3
    }

    /// The number of king moves between the two squares.
    #[inline(always)]
    #[must_use]
    pub const fn chebyshev_distance(self, other: Self) -> u8 {
        let file_d = self.file().abs_diff(other.file());
        let rank_d = self.rank().abs_diff(other.rank());
        file_d.max(rank_d)
    }

    /// Returns a [`Bitboard`] with exactly this square's bit set.
    #[inline(always)]
    #[must_use]
    pub const fn bitboard(self) -> Bitboard {
        Bitboard(1u64 << self.0)
    }

    /// Returns a new square offset by `off` indices.
    ///
    /// # Panics
    /// In debug if the result falls outside `0..64`.
    #[inline(always)]
    #[must_use]
    pub const fn offset_unchecked(self, off: i8) -> Self {
        let result = (self.0.cast_signed() + off).cast_unsigned();
        debug_assert!(result < 64, "Square::offset produced invalid square");
        Self(result)
    }

    /// Returns this index as `usize`.
    #[inline(always)]
    #[must_use]
    pub const fn as_usize(self) -> usize {
        self.0 as usize
    }

    /// Mirrors vertically (rank → 7 − rank). E.g. e1 → e8.
    #[inline(always)]
    #[must_use]
    pub const fn flip_rank(self) -> Self {
        Self(self.0 ^ 56)
    }

    /// Mirrors horizontally (file → 7 − file). E.g. a4 → h4.
    #[inline(always)]
    #[must_use]
    pub const fn flip_file(self) -> Self {
        Self(self.0 ^ 7)
    }

    /// Returns the algebraic notation as an owned string (e.g. `"e4"`).
    #[must_use]
    pub fn to_algebraic(self) -> String {
        self.to_string()
    }
}

const impl From<u8> for Square {
    #[inline(always)]
    fn from(val: u8) -> Self {
        Self(val)
    }
}

const impl From<Square> for u8 {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.0
    }
}

const impl From<Square> for u16 {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.0 as u16
    }
}

const impl From<Square> for u64 {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.0 as u64
    }
}

const impl From<Square> for usize {
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.0 as usize
    }
}

const impl Shl<Square> for u64 {
    type Output = u64;
    #[inline(always)]
    fn shl(self, rhs: Square) -> u64 {
        self << rhs.0
    }
}

const impl Shr<Square> for u64 {
    type Output = u64;
    #[inline(always)]
    fn shr(self, rhs: Square) -> u64 {
        self >> rhs.0
    }
}

impl PartialEq<u8> for Square {
    #[inline(always)]
    fn eq(&self, other: &u8) -> bool {
        self.0 == *other
    }
}

impl PartialEq<Square> for u8 {
    #[inline(always)]
    fn eq(&self, other: &Square) -> bool {
        *self == other.0
    }
}

impl PartialOrd<u8> for Square {
    #[inline(always)]
    fn partial_cmp(&self, other: &u8) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(other)
    }
}

impl PartialOrd<Square> for u8 {
    #[inline(always)]
    fn partial_cmp(&self, other: &Square) -> Option<std::cmp::Ordering> {
        self.partial_cmp(&other.0)
    }
}

const impl BitXor<u8> for Square {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: u8) -> Self {
        Self(self.0 ^ rhs)
    }
}

const impl BitXor for Square {
    type Output = Self;
    #[inline(always)]
    fn bitxor(self, rhs: Self) -> Self {
        Self(self.0 ^ rhs.0)
    }
}

impl fmt::Display for Square {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let file = (b'a' + self.file()) as char;
        let rank = (b'1' + self.rank()) as char;
        write!(f, "{file}{rank}")
    }
}

/// Error returned when parsing a [`Square`] from algebraic notation fails.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseSquareError;

impl fmt::Display for ParseSquareError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("invalid square: expected format 'e4' (file a-h, rank 1-8)")
    }
}

impl std::error::Error for ParseSquareError {}

/// Iterator over the set bits of a [`Bitboard`],
/// yielding [`Square`]s from low to high.
#[derive(Copy, Clone, Debug, Default)]
pub struct BitboardIter(pub Bitboard);

impl std::str::FromStr for Square {
    type Err = ParseSquareError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 {
            return Err(ParseSquareError);
        }

        let file = bytes[0].wrapping_sub(b'a');
        let rank = bytes[1].wrapping_sub(b'1');

        if file >= 8 || rank >= 8 {
            return Err(ParseSquareError);
        }
        Ok(Self::from_coords(file, rank))
    }
}
impl Iterator for BitboardIter {
    type Item = Square;

    #[inline(always)]
    fn next(&mut self) -> Option<Square> {
        if self.0.is_empty() { None } else { Some(self.0.pop_lsb()) }
    }

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
    fn into_iter(self) -> BitboardIter {
        self.iter()
    }
}

const impl From<u64> for Bitboard {
    #[inline(always)]
    fn from(val: u64) -> Self {
        Self(val)
    }
}

const impl From<Bitboard> for u64 {
    #[inline(always)]
    fn from(bb: Bitboard) -> Self {
        bb.0
    }
}

const impl From<Square> for Bitboard {
    /// Converts a square into a bitboard with exactly that bit set.
    #[inline(always)]
    fn from(sq: Square) -> Self {
        sq.bitboard()
    }
}

macro_rules! impl_bb_op {
    ($trait:ident, $func:ident, $assign_trait:ident, $assign_func:ident) => {
        const impl $trait for Bitboard {
            type Output = Self;

            #[inline(always)]
            fn $func(self, rhs: Self) -> Self {
                Self(self.0.$func(rhs.0))
            }
        }

        const impl $trait<u64> for Bitboard {
            type Output = Self;
            #[inline(always)]
            fn $func(self, rhs: u64) -> Self {
                Self(self.0.$func(rhs))
            }
        }

        impl $assign_trait for Bitboard {
            #[inline(always)]
            fn $assign_func(&mut self, rhs: Self) {
                self.0.$assign_func(rhs.0);
            }
        }

        impl $assign_trait<u64> for Bitboard {
            #[inline(always)]
            fn $assign_func(&mut self, rhs: u64) {
                self.0.$assign_func(rhs);
            }
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
    fn not(self) -> Self {
        Self(!self.0)
    }
}

const impl Shl<u8> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: u8) -> Self {
        Self(self.0 << rhs)
    }
}

const impl Shr<u8> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: u8) -> Self {
        Self(self.0 >> rhs)
    }
}

const impl Shl<Square> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shl(self, rhs: Square) -> Self {
        Self(self.0 << rhs.0)
    }
}

const impl Shr<Square> for Bitboard {
    type Output = Self;

    #[inline(always)]
    fn shr(self, rhs: Square) -> Self {
        Self(self.0 >> rhs.0)
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bitboard({:#018x})", self.0)
    }
}

impl fmt::Display for Bitboard {
    /// Renders an 8×8 grid (`X` = set, `.` = empty) with rank 8 at the top.
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
