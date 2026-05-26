//! Domain types, constants, and fundamental definitions.
//!
//! Provides the core vocabulary of the engine: colors, piece types,
//! directions, castling rights, and bounds.

use std::ops::{Index, IndexMut, Not};

pub use crate::core::primitives::*;

/// Centipawn evaluation
pub type Score = i32;
/// Move-ordering heuristic value
pub type MoveScore = i16;
/// Search depth
pub type Depth = i32;
/// Half-move distance from the root
pub type Ply = usize;

/// Upper bound on legal moves in any position (theoretical legal max being 218).
pub const MAX_MOVES: usize = 256;
/// Absolute deepest ply the engine will ever explore.
pub const MAX_PLY: usize = 246;
pub const MAX_DEPTH: i32 = MAX_PLY as i32;
/// Stands in for +∞ in alpha-beta windows. Beats every evaluation but not mate.
pub const INF: i32 = 32_000;
/// Checkmate at the root. Actual mates are scored `MATE - ply_distance`.
pub const MATE: i32 = 30_000;
/// Any |score| above this threshold is a forced mate, not merely a large eval.
pub const MATE_BOUND: i32 = MATE - MAX_PLY as i32 - 1;
/// Total game phase material: N=1, B=1, R=2, Q=4 → 2·(1+1+2+4) = 24
pub const TOTAL_PHASE: i32 = 24;
/// Middlegame terms
pub const LANE_MG: usize = 0;
/// Endgame terms
pub const LANE_EG: usize = 1;
/// Tapered-eval phase counter
pub const LANE_PHASE: usize = 2;

/// |score| < this for N plies → Draw
pub const ADJ_DRAW_SCORE: i32 = 10;
/// |score| < ADJ_DRAW_SCORE for this many consecutive plies
pub const ADJ_DRAW_PLIES: usize = 24;
/// Only start draw adjudication after this many plies (half-moves)
pub const ADJ_DRAW_START_PLY: usize = 80;
/// |score| > this for N plies → Win
pub const ADJ_WIN_SCORE: i32 = 1200;
/// |score| > ADJ_WIN_SCORE for this many consecutive plies
pub const ADJ_WIN_PLIES: usize = 8;
/// |score| > this → Instant Resignation
pub const ADJ_RESIGN_SCORE: i32 = 3000;

const _: () = assert!(LANE_MG == 0 && LANE_EG == 1, "Tapered SIMD madd requires MG and EG at lanes 0 and 1 respectively.");

/// Which wire protocol the engine is currently speaking.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum Protocol {
    #[default]
    Uci,
    XBoard,
}

/// White = 0, Black = 1
///
/// | expression               | White | Black |
/// |--------------------------|-------|-------|
/// | `back_rank`              | 0     | 7     |
/// | `pawn_start_rank`        | 1     | 6     |
/// | `promotion_rank`         | 7     | 0     |
#[derive(Copy, Clone, PartialEq, Eq, Debug, std::marker::ConstParamTy)]
#[repr(u8)]
pub enum Color {
    White = 0,
    Black = 1,
}

const _: () = assert!(std::mem::size_of::<Color>() == 1);
const _: () = assert!(std::mem::size_of::<Option<Color>>() == 1); // used discriminants: 0..=1; niches: 2..=255

impl Color {
    #[must_use]
    #[inline(always)]
    pub const fn opposite(self) -> Self {
        match self {
            Self::White => Self::Black,
            Self::Black => Self::White,
        }
    }

    #[inline(always)]
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// The direction pawns march: North for White, South for Black.
    #[inline(always)]
    pub const fn forward_dir(self) -> Direction {
        const DIRS: [Direction; 2] = [Direction::North, Direction::South];
        DIRS[self as usize]
    }

    /// First rank from this side's perspective (0 or 7).
    #[inline(always)]
    pub const fn back_rank(self) -> u8 {
        (self as u8) * 7
    }

    /// Where this side's pawns stand in the starting position (1 or 6).
    #[inline(always)]
    pub const fn pawn_start_rank(self) -> u8 {
        1 + (self as u8) * 5
    }

    /// The rank where pawns promote (0 or 7).
    #[inline(always)]
    pub const fn promotion_rank(self) -> u8 {
        7 - (self as u8) * 7
    }
}

const impl From<u8> for Color {
    #[inline(always)]
    fn from(val: u8) -> Self {
        // SAFETY: val & 1 is always 0 or 1 — both valid Color variants.
        unsafe { std::mem::transmute(val & 1) }
    }
}

const impl From<Color> for usize {
    #[inline(always)]
    fn from(val: Color) -> Self {
        val as usize
    }
}

const impl Not for Color {
    type Output = Self;

    #[inline(always)]
    fn not(self) -> Self::Output {
        self.opposite()
    }
}

// The six chessmen (plus a sentinel)
/// Ordered Pawn..King by ascending material value for compact table lookups.
/// The `None` sentinel marks empty squares:
/// variant 7 is reserved as the niche so `Option<PieceType>`
/// still fits in a single byte.
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

const _: () = assert!(std::mem::size_of::<PieceType>() == 1);
const _: () = assert!(std::mem::size_of::<Option<PieceType>>() == 1); // niche = 7

const _: () = {
    assert!(PieceType::Knight as u8 == 1);
    assert!(PieceType::Bishop as u8 == 2);
    assert!(PieceType::Rook as u8 == 3);
    assert!(PieceType::Queen as u8 == 4);
};

impl PieceType {
    /// Every real piece type, excluding `None`.
    pub const ALL: [Self; 6] = [Self::Pawn, Self::Knight, Self::Bishop, Self::Rook, Self::Queen, Self::King];

    #[inline(always)]
    pub const fn is_none(self) -> bool {
        matches!(self, Self::None)
    }

    #[inline(always)]
    pub const fn is_some(self) -> bool {
        !self.is_none()
    }

    #[inline(always)]
    pub const fn as_usize(self) -> usize {
        self as usize
    }

    /// Construct from a raw index. Caller must guarantee `idx ∈ 0..=6`.
    #[inline(always)]
    pub const fn new(idx: u8) -> Self {
        match idx & 7 {
            0 => Self::Pawn,
            1 => Self::Knight,
            2 => Self::Bishop,
            3 => Self::Rook,
            4 => Self::Queen,
            5 => Self::King,
            6 | 7 => Self::None,
            _ => unsafe { core::hint::unreachable_unchecked() },
        }
    }

    /// FEN / SAN character — uppercase for White, lowercase for Black.
    #[inline(always)]
    pub const fn to_char(self, color: Color) -> char {
        const CHARS: [[char; 7]; 2] = [['P', 'N', 'B', 'R', 'Q', 'K', '?'], ['p', 'n', 'b', 'r', 'q', 'k', '?']];
        CHARS[color as usize][self as usize]
    }

    /// Parse a FEN piece letter.
    /// Case-insensitive: anything unrecognized → `None`.
    #[inline(always)]
    pub const fn from_char(ch: char) -> Self {
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
        // SAFETY: The engine guarantees that pieces are strictly within 0..=6.
        // We use an explicit match rather than modulo arithmetic to ensure
        // niche safety for Option<PieceType>
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
    fn from(val: PieceType) -> Self {
        val as usize
    }
}

/// The eight compass directions, numbered clockwise from North.
///
/// Square-index deltas (A1 = 0, H8 = 63, little-endian rank-file):
/// ```text
///     NW(+7)   N(+8)   NE(+9)
///      W(−1)     ·      E(+1)
///     SW(−9)   S(−8)   SE(−7)
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Direction {
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
    /// Mirror the direction for Black's viewpoint.
    /// XOR with 4 rotates the 3-bit compass index by exactly half a turn
    /// North→South, NorthEast→SouthWest, etc.
    #[must_use]
    #[inline(always)]
    pub const fn relative(self, color: Color) -> Self {
        Direction::from(self as u8 ^ ((color as u8) << 2))
    }

    /// Square-index delta for a single step in this direction.
    #[inline(always)]
    pub const fn delta(self) -> i8 {
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

/// Terminal result of a game, stored from White's point of view.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GameOutcome {
    WhiteWins,
    BlackWins,
    Draw,
}

impl GameOutcome {
    /// WDL float from White's perspective: 1.0 / 0.0 / 0.5.
    #[inline(always)]
    pub const fn wdl(self) -> f32 {
        match self {
            Self::WhiteWins => 1.0,
            Self::BlackWins => 0.0,
            Self::Draw => 0.5,
        }
    }

    /// WDL float from the perspective of the given side.
    #[inline(always)]
    pub const fn relative_to(self, stm: Color) -> f32 {
        match stm {
            Color::White => self.wdl(),
            Color::Black => 1.0 - self.wdl(),
        }
    }

    /// Converts a side-to-move–relative score into a game outcome.
    /// Positive score → STM is winning.
    #[inline(always)]
    pub const fn from_stm_score(score: i32, stm: Color) -> Self {
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

/// Deterministic ±8 cp jitter for draw scores, keyed by the node counter.
///
/// Returning exactly 0 for every draw makes the search indifferent between
/// drawing lines — it picks whichever comes first in move order and sticks.
/// In positions with multiple drawish paths, that can mean repeating the
/// same sequence when a slightly-less-drawn alternative exists. A small
/// node-keyed jitter breaks the tie without perturbing non-draw scores,
/// so the search naturally explores alternate draw routes where the
/// opponent has a chance to stray.
#[inline]
pub fn draw_score(nodes: u64) -> i32 {
    (nodes & 0x7) as i32 - 3
}

/// Let chess types serve as array indices directly:
/// `table[Color::White]`, `scores[PieceType::Knight]`, `board[Square::E4]`.
/// Bounds-checked in debug builds: compiles to bare pointer math in release.
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
soul_index!(PieceType, 6);
soul_index!(PieceType, 8);
soul_index!(PieceType, 12);
soul_index!(PieceType, 14);
soul_index!(Square, 64);
