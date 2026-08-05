//! Move representation and stack-allocated move lists.
//!
//! Encodes a complete chess move into a compact 16-bit integer.

use std::{
    fmt,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Index, IndexMut},
};

use crate::core::defs::{MAX_MOVES, PieceType, Square};

/// Compact 16-bit move encoding: from square, to square, and a 4-bit flag.
///
/// - Bits 00..05: From square (0-63)
/// - Bits 06..11: To square   (0-63)
/// - Bits 12..15: Flag        (0-15)
///
/// | Binary | Dec |        Type         |             Notes              |
/// |--------|-----|---------------------|--------------------------------|
/// | 0000   | 0   | Quiet               | Nothing special                |
/// | 0001   | 1   | Capture             | Bit 0 = capture flag           |
/// | 0010   | 2   | Promo Knight        | Bit 1 = promotion flag         |
/// | 0011   | 3   | Promo Knight + Cap  | Bit 0 + Bit 1                  |
/// | 0100   | 4   | Double Pawn Push    | Sets En Passant square         |
/// | 0101   | 5   | En Passant          | Victim not on the to-square    |
/// | 0110   | 6   | Promo Bishop        |                                |
/// | 0111   | 7   | Promo Bishop + Cap  |                                |
/// | 1000   | 8   | (reserved)          |                                |
/// | 1001   | 9   | (reserved)          |                                |
/// | 1010   | 10  | Promo Rook          |                                |
/// | 1011   | 11  | Promo Rook + Cap    |                                |
/// | 1100   | 12  | Castle              | Standard or FRC Castling       |
/// | 1101   | 13  | (reserved)          |                                |
/// | 1110   | 14  | Promo Queen         |                                |
/// | 1111   | 15  | Promo Queen + Cap   |                                |
#[derive(Copy, Clone, PartialEq, Eq, Default)]
#[repr(transparent)]
pub struct Move(u16);

// No niche (0 is valid null move), so Option<Move> = 4 bytes.
const _: () = assert!(std::mem::size_of::<Move>() == 2);

impl Move {
    pub const QUIET: u16 = 0;
    pub const CAPTURE: u16 = 1;

    // Bit 1 of the flag nibble = promotion. MovePicker relies on this ordering:
    // Since PROM_Q (14) > PROM_R (10) > PROM_B (6) > PROM_N (2), and MovePicker
    // bitpacks the move's inner value into its sort key, Queen promotions natively
    // evaluate as larger u16 values and sort to the back of the array, which is popped first.
    // If you break this numeric hierarchy, quiet promotion ordering will silently regress!
    pub const PROM_N: u16 = 2;
    pub const PROM_N_CAPTURE: u16 = 3;

    // b2 = 1 (Special)
    pub const DOUBLE_PUSH: u16 = 4;
    pub const EP_CAPTURE: u16 = 5;

    pub const PROM_B: u16 = 6;
    pub const PROM_B_CAPTURE: u16 = 7;

    // b3 = 1 (Special High)
    pub const PROM_R: u16 = 10;
    pub const PROM_R_CAPTURE: u16 = 11;

    pub const CASTLE: u16 = 12;

    pub const PROM_Q: u16 = 14;
    pub const PROM_Q_CAPTURE: u16 = 15;

    // Compile-time guarantee that the promotion flag values preserve the
    // quality ordering that MovePicker's bitpacked sort implicitly depends on.
    const _PROMO_ORDER: () = {
        assert!(Self::PROM_Q > Self::PROM_R);
        assert!(Self::PROM_R > Self::PROM_B);
        assert!(Self::PROM_B > Self::PROM_N);
        assert!(Self::PROM_Q_CAPTURE > Self::PROM_R_CAPTURE);
        assert!(Self::PROM_R_CAPTURE > Self::PROM_B_CAPTURE);
        assert!(Self::PROM_B_CAPTURE > Self::PROM_N_CAPTURE);
    };

    const MASK_SQ: u16 = 0x3F;
    const MASK_FLAG: u16 = 0xF;
    const SHIFT_TO: u16 = 6;
    const SHIFT_FLAG: u16 = 12;

    #[inline(always)]
    pub fn new(from: Square, to: Square, flag: u16) -> Self {
        debug_assert!(from < 64, "Invalid from square: {from} (max 63)");
        debug_assert!(to < 64, "Invalid to square: {to} (max 63)");
        debug_assert!(flag < 16, "Invalid flags: {flag} (max 15)");

        let f = u16::from(from) & Self::MASK_SQ;
        let t = (u16::from(to) & Self::MASK_SQ) << Self::SHIFT_TO;
        let fl = (flag & Self::MASK_FLAG) << Self::SHIFT_FLAG;

        Self(f | t | fl)
    }

    /// Null move (no move).
    #[inline(always)]
    pub const fn null() -> Self {
        Self(0)
    }

    /// Construct from raw 16-bit encoding.
    #[inline(always)]
    pub const fn from_u16(raw: u16) -> Self {
        Self(raw)
    }

    /// Raw 16-bit encoding.
    #[inline(always)]
    pub const fn inner(self) -> u16 {
        self.0
    }

    /// Move is null (placeholder).
    #[inline(always)]
    pub const fn is_null(self) -> bool {
        self.0 == 0
    }

    /// Origin square.
    #[inline(always)]
    pub const fn from(self) -> Square {
        Square((self.0 & Self::MASK_SQ) as u8)
    }

    /// Destination square.
    #[inline(always)]
    pub const fn to(self) -> Square {
        Square(((self.0 >> Self::SHIFT_TO) & Self::MASK_SQ) as u8)
    }

    /// Move flag (type encoding).
    #[inline(always)]
    pub const fn flag(self) -> u16 {
        (self.0 >> Self::SHIFT_FLAG) & Self::MASK_FLAG
    }

    /// Move captures something.
    #[inline(always)]
    pub const fn is_capture(self) -> bool {
        (self.0 & 0x1000) != 0
    }

    /// Move is an en passant capture.
    #[inline(always)]
    pub const fn is_en_passant(self) -> bool {
        (self.0 & 0xF000) == 0x5000
    }

    /// Move is a castling (standard or Chess960).
    #[inline(always)]
    pub const fn is_castling(self) -> bool {
        (self.0 & 0xF000) == 0xC000
    }

    /// Move is a promotion (any piece).
    #[inline(always)]
    pub const fn is_promotion(self) -> bool {
        (self.0 & 0x2000) != 0
    }

    /// Move is non-capturing.
    #[inline(always)]
    pub const fn is_quiet(self) -> bool {
        !self.is_capture()
    }

    /// Non-capturing, non-castling: eligible for history heuristic.
    #[inline(always)]
    pub const fn is_history_quiet(self) -> bool {
        self.is_quiet() && !self.is_castling()
    }

    /// Capture or promotion.
    #[inline(always)]
    pub const fn is_tactical(self) -> bool {
        self.is_capture() || self.is_promotion()
    }

    /// Pawn double push (sets en passant square).
    #[inline(always)]
    pub const fn is_double_push(self) -> bool {
        (self.0 & 0xF000) == 0x4000
    }

    /// Promoted piece type, if any.
    #[inline(always)]
    pub fn promo(self) -> Option<PieceType> {
        if !self.is_promotion() {
            return None;
        }

        match self.flag() {
            Self::PROM_N | Self::PROM_N_CAPTURE => Some(PieceType::Knight),
            Self::PROM_B | Self::PROM_B_CAPTURE => Some(PieceType::Bishop),
            Self::PROM_R | Self::PROM_R_CAPTURE => Some(PieceType::Rook),
            _ => Some(PieceType::Queen),
        }
    }

    pub fn to_uci(self, is_frc: bool) -> String {
        if self.is_null() {
            return "0000".to_string();
        }

        let from = self.from();
        let mut to = self.to();

        // Standard Castling (e1g1) vs FRC (e1h1)
        if self.is_castling() && !is_frc {
            let rook_sq = to;
            let rank = from.rank();

            to = if rook_sq.file() > from.file() {
                Square::from_coords(6, rank) // G file
            } else {
                Square::from_coords(2, rank) // C file
            };
        }

        let mut s = String::with_capacity(5);

        let push_sq = |s: &mut String, sq: Square| {
            s.push((b'a' + sq.file()) as char);
            s.push((b'1' + sq.rank()) as char);
        };

        push_sq(&mut s, from);
        push_sq(&mut s, to);

        if let Some(pt) = self.promo() {
            let ch = match pt {
                PieceType::Knight => 'n',
                PieceType::Bishop => 'b',
                PieceType::Rook => 'r',
                PieceType::Queen => 'q',
                _ => ' ',
            };
            s.push(ch);
        }
        s
    }
}

impl fmt::Debug for Move {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_uci(false))
    }
}

/// Stack-allocated move list. No heap allocation on the hot path.
pub struct MoveList {
    moves: [MaybeUninit<Move>; MAX_MOVES],
    len: usize,
}

impl MoveList {
    #[inline(always)]
    pub const fn new() -> Self {
        Self { moves: [MaybeUninit::uninit(); MAX_MOVES], len: 0 }
    }

    #[inline(always)]
    pub fn push(&mut self, mv: Move) {
        debug_assert!(self.len < MAX_MOVES, "MoveList capacity exceeded");
        // SAFETY: Caller guarantees the list is not full; debug_assert above catches violations in debug builds.
        unsafe { self.moves.get_unchecked_mut(self.len).write(mv) };
        self.len += 1;
    }

    #[inline(always)]
    pub fn clear(&mut self) {
        self.len = 0;
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline(always)]
    pub fn iter(&self) -> std::slice::Iter<'_, Move> {
        self.as_slice().iter()
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[Move] {
        // SAFETY: The first len elements are guaranteed to be initialized.
        unsafe { self.moves[..self.len].assume_init_ref() }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [Move] {
        // SAFETY: The first len elements are guaranteed to be initialized.
        unsafe { self.moves[..self.len].assume_init_mut() }
    }

    #[inline(always)]
    pub fn get(&self, index: usize) -> Move {
        debug_assert!(index < self.len, "MoveList::get out of bounds: {index} >= {}", self.len);
        // SAFETY: The caller must ensure index < len.
        unsafe { self.moves[index].assume_init() }
    }
}

impl Default for MoveList {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for MoveList {
    type Output = Move;

    #[inline(always)]
    fn index(&self, index: usize) -> &Self::Output {
        debug_assert!(index < self.len);
        // SAFETY: index < len, and the first len elements are initialized.
        unsafe { self.moves.get_unchecked(index).assume_init_ref() }
    }
}

impl IndexMut<usize> for MoveList {
    #[inline(always)]
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        debug_assert!(index < self.len);
        // SAFETY: index < len, and the first len elements are initialized.
        unsafe { self.moves.get_unchecked_mut(index).assume_init_mut() }
    }
}

impl<'a> IntoIterator for &'a MoveList {
    type Item = &'a Move;
    type IntoIter = std::slice::Iter<'a, Move>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Deref for MoveList {
    type Target = [Move];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl DerefMut for MoveList {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.as_mut_slice()
    }
}
