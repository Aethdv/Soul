//! The 16-bit move encoding and the stack-allocated move list.

use std::{fmt, mem::MaybeUninit, ops::Deref};

use crate::core::{
    board::castling_targets,
    defs::{Color, MAX_MOVES, PieceType, Square},
};

/// Bits 0..=5: from square
/// Bits 6..=11: to square
/// Bits 12..=15: flag
///
/// | Flag       | Move                       |
/// |:----------:|----------------------------|
/// | `0000` (0) | Quiet                      |
/// | `0001` (1) | Capture                    |
/// | `0010` (2) | Knight promotion           |
/// | `0011` (3) | Knight promotion + capture |
/// | `0100` (4) | Double pawn push           |
/// | `0101` (5) | En passant                 |
/// | `0110` (6) | Bishop promotion           |
/// | `0111` (7) | Bishop promotion + capture |
/// | `1000` (8) | Reserved                   |
/// | `1001` (9) | Reserved                   |
/// | `1010` (10)| Rook promotion             |
/// | `1011` (11)| Rook promotion + capture   |
/// | `1100` (12)| Castling                   |
/// | `1101` (13)| Reserved                   |
/// | `1110` (14)| Queen promotion            |
/// | `1111` (15)| Queen promotion + capture  |
#[derive(Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub(crate) struct Move(u16);

// 0 is a valid null move, so Option<Move> has no niche.
const _: () = assert!(std::mem::size_of::<Move>() == 2);
const _: () = assert!(std::mem::size_of::<Option<Move>>() == 4);

impl Move {
    const C: u16 = 0b0001; // Capture
    const T: u16 = 0b0010; // Promotion target (N, B, R, Q)
    const D: u16 = 0b0100; // Diagonal attack capability (P, B, K, Q)
    const O: u16 = 0b1000; // Orthogonal attack capability (R, K, Q)

    pub(crate) const QUIET: u16 = 0;
    pub(crate) const CAPTURE: u16 = Self::C;

    pub(crate) const PROM_N: u16 = Self::T;
    pub(crate) const PROM_N_CAPTURE: u16 = Self::T | Self::C;

    pub(crate) const DOUBLE_PUSH: u16 = Self::D;
    pub(crate) const EP_CAPTURE: u16 = Self::D | Self::C;

    pub(crate) const PROM_B: u16 = Self::D | Self::T;
    pub(crate) const PROM_B_CAPTURE: u16 = Self::D | Self::T | Self::C;

    pub(crate) const PROM_R: u16 = Self::O | Self::T;
    pub(crate) const PROM_R_CAPTURE: u16 = Self::O | Self::T | Self::C;

    pub(crate) const CASTLE: u16 = Self::O | Self::D;

    pub(crate) const PROM_Q: u16 = Self::O | Self::D | Self::T;
    pub(crate) const PROM_Q_CAPTURE: u16 = Self::O | Self::D | Self::T | Self::C;

    const MASK_SQ: u16 = 0x3F;
    const MASK_FLAG: u16 = 0xF;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    #[inline(always)]
    pub(crate) const fn new(from: Square, to: Square, flag: u16) -> Self {
        debug_assert!(from.0 < 64, "Invalid from square (max 63)");
        debug_assert!(to.0 < 64, "Invalid to square (max 63)");
        debug_assert!(flag < 16, "Invalid flags (max 15)");

        let f = (from.0 as u16) & Self::MASK_SQ;
        let t = ((to.0 as u16) & Self::MASK_SQ) << Self::SHIFT_TO;
        let fl = (flag & Self::MASK_FLAG) << Self::SHIFT_FLAG;
        Self(f | t | fl)
    }

    #[inline(always)]
    pub(crate) const fn null() -> Self { Self(0) }

    #[inline(always)]
    pub(crate) const fn from_u16(raw: u16) -> Self { Self(raw) }

    #[inline(always)]
    pub(crate) const fn inner(self) -> u16 { self.0 }

    #[inline(always)]
    pub(crate) const fn is_null(self) -> bool { self.0 == 0 }

    #[inline(always)]
    pub(crate) const fn from(self) -> Square { Square((self.0 & Self::MASK_SQ) as u8) }

    #[inline(always)]
    pub(crate) const fn to(self) -> Square { Square(((self.0 >> Self::SHIFT_TO) & Self::MASK_SQ) as u8) }

    #[cfg(test)]
    #[inline(always)]
    pub(crate) const fn flag(self) -> u16 { (self.0 >> Self::SHIFT_FLAG) & Self::MASK_FLAG }

    /// True for ordinary quiet and capture moves (O = 0, D = 0, T = 0).
    #[inline(always)]
    pub(crate) const fn is_ordinary(self) -> bool { self.0 < 0x2000 }

    #[inline(always)]
    pub(crate) const fn is_capture(self) -> bool { (self.0 & 0x1000) != 0 }

    #[inline(always)]
    pub(crate) const fn is_en_passant(self) -> bool { (self.0 & 0xF000) == 0x5000 }

    /// Written king-onto-rook: `to` is the castling rook's square.
    #[inline(always)]
    pub(crate) const fn is_castling(self) -> bool { (self.0 & 0xF000) == 0xC000 }

    #[inline(always)]
    pub(crate) const fn is_promotion(self) -> bool { (self.0 & 0x2000) != 0 }

    #[inline(always)]
    pub(crate) const fn is_quiet(self) -> bool { !self.is_capture() }

    /// Castling is excluded because history keys on `to`, which is the rook square.
    #[inline(always)]
    pub(crate) const fn is_history_quiet(self) -> bool { self.is_quiet() && !self.is_castling() }

    /// Single bitmask testing adjacent bits T (0x2000) and C (0x1000).
    #[inline(always)]
    pub(crate) const fn is_tactical(self) -> bool { (self.0 & 0x3000) != 0 }

    #[inline(always)]
    pub(crate) const fn is_double_push(self) -> bool { (self.0 & 0xF000) == 0x4000 }

    #[inline(always)]
    const fn promo_piece(self) -> PieceType {
        assert!(self.is_promotion(), "promo_piece called on non-promotion move");
        PieceType::from(((self.0 >> 14) as u8) + 1)
    }

    #[inline(always)]
    pub(crate) const fn promo(self) -> Option<PieceType> { if self.is_promotion() { Some(self.promo_piece()) } else { None } }

    pub(crate) fn to_uci(self, frc_notation: bool) -> String {
        if self.is_null() {
            return "0000".to_string();
        }

        let from = self.from();
        let mut to = self.to();

        // Chess960 keeps the king-onto-rook form (e1h1).
        // Standard chess writes king-to-destination (e1g1).
        if self.is_castling() && !frc_notation {
            to = castling_targets(from, to).0;
        }

        let mut s = format!("{from}{to}");
        if let Some(pt) = self.promo() {
            s.push(pt.to_char(Color::Black));
        }
        s
    }
}

// PROM_N is the base T: its offset is zero by definition.
// B/R/Q are the nonzero offsets, so only they need an assertion.
const _: () = {
    assert!((Move::PROM_B >> 2) == PieceType::Bishop as u16 - PieceType::Knight as u16);
    assert!((Move::PROM_R >> 2) == PieceType::Rook as u16 - PieceType::Knight as u16);
    assert!((Move::PROM_Q >> 2) == PieceType::Queen as u16 - PieceType::Knight as u16);
};

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { write!(f, "{}", self.to_uci(false)) }
}

pub(crate) struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline(always)]
    pub(crate) const fn new() -> Self { Self { moves: [MaybeUninit::uninit(); MAX_MOVES], len: 0 } }

    #[inline(always)]
    pub(crate) fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MAX_MOVES, "MoveList capacity exceeded");
        // SAFETY: MAX_MOVES exceeds the 218 moves any legal position can offer,
        // so a generator filling one list cannot reach the end of it.
        unsafe { self.moves.get_unchecked_mut(self.len).write(mv) };
        self.len += 1;
    }
}

impl Default for MoveList {
    fn default() -> Self { Self::new() }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &[Move] {
        // SAFETY: push initializes every element below len.
        unsafe { self.moves[..self.len].assume_init_ref() }
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter { self.iter() }
}
