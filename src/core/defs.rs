//! Colors, pieces, directions, score bounds and mate-score arithmetic.

use std::ops::{Index, IndexMut, Not};

pub(crate) use crate::core::primitives::*;

// No legal position has more than 218 moves.
pub(crate) const MAX_MOVES: usize = 256;
pub(crate) const MAX_PLY: usize = 246;
pub(crate) const MAX_DEPTH: i32 = MAX_PLY as i32;

pub(crate) const INF: i32 = 32_000;
/// A mate n plies away scores MATE - n.
pub(crate) const MATE: i32 = 30_000;
/// Scores outside `±MATE_BOUND` are mate scores.
pub(crate) const MATE_BOUND: i32 = MATE - MAX_PLY as i32 - 1;

/// Opening phase at N=B=1, R=2, Q=4: 4 + 4 + 8 + 8 = 24.
pub const TOTAL_PHASE: i32 = 24;
pub(crate) const LANE_PHASE: usize = 2;

#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) enum Protocol {
    #[default]
    Uci,
    XBoard,
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, std::marker::ConstParamTy)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

impl Color {
    #[must_use]
    #[inline(always)]
    pub(crate) const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    #[inline(always)]
    pub(crate) const fn as_usize(self) -> usize { self as usize }

    #[inline(always)]
    pub(crate) const fn forward_dir(self) -> Direction {
        const DIRS: [Direction; 2] = [Direction::North, Direction::South];
        DIRS[self as usize]
    }

    #[inline(always)]
    pub(crate) const fn back_rank(self) -> u8 { (self as u8) * 7 }
}

const impl From<u8> for Color {
    #[inline(always)]
    fn from(val: u8) -> Self {
        // SAFETY: val & 1 is always 0 or 1, both valid Color variants.
        unsafe { std::mem::transmute(val & 1) }
    }
}

const impl From<Color> for usize {
    #[inline(always)]
    fn from(val: Color) -> Self { val as usize }
}

const impl Not for Color {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output { self.opposite() }
}

#[derive(Copy, Clone, PartialEq, Eq, Debug, std::marker::ConstParamTy)]
#[repr(u8)]
pub enum PieceType {
    Pawn = 0,
    Knight = 1,
    Bishop = 2,
    Rook = 3,
    Queen = 4,
    King = 5,
    None = 6,
}

impl PieceType {
    #[cfg(feature = "dataset")]
    pub(crate) const ALL: [Self; 6] = [Self::Pawn, Self::Knight, Self::Bishop, Self::Rook, Self::Queen, Self::King];

    #[inline(always)]
    pub(crate) const fn as_usize(self) -> usize { self as usize }

    #[inline(always)]
    pub const fn to_char(self, color: Color) -> char {
        const CHARS: [[char; 7]; 2] = [['P', 'N', 'B', 'R', 'Q', 'K', '?'], ['p', 'n', 'b', 'r', 'q', 'k', '?']];
        CHARS[color as usize][self as usize]
    }

    #[inline(always)]
    pub(crate) const fn from_char(ch: char) -> Self {
        match ch {
            'P' | 'p' => Self::Pawn,
            'N' | 'n' => Self::Knight,
            'B' | 'b' => Self::Bishop,
            'R' | 'r' => Self::Rook,
            'Q' | 'q' => Self::Queen,
            'K' | 'k' => Self::King,
            _ => Self::None,
        }
    }
}

const impl From<u8> for PieceType {
    #[inline(always)]
    fn from(val: u8) -> Self {
        match val {
            0 => Self::Pawn,
            1 => Self::Knight,
            2 => Self::Bishop,
            3 => Self::Rook,
            4 => Self::Queen,
            5 => Self::King,
            _ => Self::None,
        }
    }
}

const impl From<PieceType> for usize {
    #[inline(always)]
    fn from(val: PieceType) -> Self { val as usize }
}

/// Compass directions clockwise from North; opposites differ by 4.
///
/// ```text
///     NW(+7)   N(+8)   NE(+9)
///      W(−1)     ·      E(+1)
///     SW(−9)   S(−8)   SE(−7)
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub(crate) enum Direction {
    North = 0,
    NorthEast = 1,
    East = 2,
    SouthEast = 3,
    South = 4,
    SouthWest = 5,
    West = 6,
    NorthWest = 7,
}

impl Direction {
    #[must_use]
    #[inline(always)]
    pub(crate) const fn relative(self, color: Color) -> Self { Direction::from(self as u8 ^ ((color as u8) << 2)) }

    #[inline(always)]
    pub(crate) const fn delta(self) -> i8 {
        const DELTAS: [i8; 8] = [8, 9, 1, -7, -8, -9, -1, 7];
        DELTAS[self as usize]
    }
}

const impl From<u8> for Direction {
    #[inline(always)]
    fn from(val: u8) -> Self {
        // SAFETY: val & 7 is 0..=7, matching all eight Direction variants.
        unsafe { std::mem::transmute(val & 7) }
    }
}

#[cfg(any(test, feature = "datagen"))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum GameOutcome {
    WhiteWins,
    BlackWins,
    Draw,
}

#[cfg(any(test, feature = "datagen"))]
impl GameOutcome {
    /// White-relative. loss=0, draw=1, win=2.
    #[cfg(feature = "datagen")]
    #[inline(always)]
    pub(crate) const fn packed(self) -> u8 {
        match self {
            Self::WhiteWins => 2,
            Self::BlackWins => 0,
            Self::Draw => 1,
        }
    }

    #[inline(always)]
    pub(crate) const fn from_stm_score(score: i32, stm: Color) -> Self {
        if score > 0 {
            match stm {
                Color::White => Self::WhiteWins,
                Color::Black => Self::BlackWins,
            }
        } else if score < 0 {
            match stm {
                Color::White => Self::BlackWins,
                Color::Black => Self::WhiteWins,
            }
        } else {
            Self::Draw
        }
    }
}

/// Varies draws between -1 and +1 to avoid always preferring the first drawing line searched.
#[inline]
pub(crate) fn draw_score(nodes: u64) -> i32 { (nodes & 0x2) as i32 - 1 }

/// A forced mate the side to move delivers.
#[inline]
pub(crate) fn is_win(score: i32) -> bool { score > MATE_BOUND }

/// A forced mate delivered against the side to move.
#[inline]
pub(crate) fn is_loss(score: i32) -> bool { score < -MATE_BOUND }

#[inline]
pub(crate) fn is_mate(score: i32) -> bool { is_win(score) || is_loss(score) }

#[inline]
pub(crate) fn mate_in(ply: usize) -> i32 { MATE - ply as i32 }

#[inline]
pub(crate) fn mated_in(ply: usize) -> i32 { -MATE + ply as i32 }

/// Search mate scores are root-relative; TT scores are node-relative.
/// `score_from_tt` converts back.
#[inline(always)]
pub(crate) fn score_to_tt(score: i32, ply: usize) -> i32 {
    if is_win(score) {
        score + ply as i32
    } else if is_loss(score) {
        score - ply as i32
    } else {
        score
    }
}

#[inline(always)]
pub(crate) fn score_from_tt(score: i32, ply: usize) -> i32 {
    if is_win(score) {
        score - ply as i32
    } else if is_loss(score) {
        score + ply as i32
    } else {
        score
    }
}

/// Domain-typed array indexing, checked in debug and unchecked in release.
#[macro_export]
macro_rules! soul_index {
    ($type:ty, $size:expr) => {
        impl<T> Index<$type> for [T; $size] {
            type Output = T;

            #[inline(always)]
            fn index(&self, idx: $type) -> &Self::Output {
                let i = usize::from(idx);
                debug_assert!(i < $size, "Domain-typed index out of bounds");
                debug_index!(self, i)
            }
        }

        impl<T> IndexMut<$type> for [T; $size] {
            #[inline(always)]
            fn index_mut(&mut self, idx: $type) -> &mut Self::Output {
                let i = usize::from(idx);
                debug_assert!(i < $size, "Domain-typed index out of bounds");
                debug_index_mut!(self, i)
            }
        }
    };
}

soul_index!(Color, 2);
soul_index_checked!(PieceType, 6);
soul_index!(PieceType, 8);
soul_index!(PieceType, 12);
soul_index!(PieceType, 14);
soul_index!(Square, 64);

#[cfg(test)]
mod tests {
    use super::PieceType;

    #[test]
    #[should_panic(expected = "Domain-typed index out of bounds")]
    fn six_slot_piece_type_index_rejects_none() {
        let slots = [0u8; 6];
        let _ = slots[PieceType::None];
    }
}
