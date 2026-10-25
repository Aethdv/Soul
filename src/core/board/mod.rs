//! Board representation and position state.

use std::iter;

#[cfg(any(test, feature = "rigs", feature = "nostore"))]
use crate::weave::U64x4;
use crate::{
    core::{
        defs::{Bitboard, Color, Direction, PieceType, Square},
        error::FenError,
        moves::Move,
        psqt, zobrist,
    },
    weave::I16x8,
};

pub(crate) mod attacks;
pub(crate) mod bitboard;
mod fen;
mod make;
pub(crate) mod spatial;
pub(crate) mod xorboard;

#[cfg(test)] mod tests;

pub(crate) use make::castling_targets;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub(crate) const WHITE_OO: u8 = 1;
pub(crate) const WHITE_OOO: u8 = 2;
pub(crate) const BLACK_OO: u8 = 4;
pub(crate) const BLACK_OOO: u8 = 8;

pub(crate) const ROOK_W_KS: usize = 0;
pub(crate) const ROOK_W_QS: usize = 1;
pub(crate) const ROOK_B_KS: usize = 2;
pub(crate) const ROOK_B_QS: usize = 3;

const ROOK_RIGHTS: [u8; 4] = [WHITE_OO, WHITE_OOO, BLACK_OO, BLACK_OOO];

const W_OO_EMPTY: Bitboard = Bitboard(0x60); // f1, g1
const W_OOO_EMPTY: Bitboard = Bitboard(0x0E); // b1, c1, d1
const B_OO_EMPTY: Bitboard = Bitboard(0x6000_0000_0000_0000); // f8, g8
const B_OOO_EMPTY: Bitboard = Bitboard(0x0E00_0000_0000_0000); // b8, c8, d8

const CASTLE_W_KS: [u8; 4] = [4, 7, 6, 5]; // e1, h1 → g1, f1
const CASTLE_W_QS: [u8; 4] = [4, 0, 2, 3]; // e1, a1 → c1, d1
const CASTLE_B_KS: [u8; 4] = [60, 63, 62, 61]; // e8, h8 → g8, f8
const CASTLE_B_QS: [u8; 4] = [60, 56, 58, 59]; // e8, a8 → c8, d8

const CASTLE_W_KS_CHECK: [u8; 3] = [4, 5, 6]; // e1, f1, g1
const CASTLE_W_QS_CHECK: [u8; 3] = [4, 3, 2]; // e1, d1, c1
const CASTLE_B_KS_CHECK: [u8; 3] = [60, 61, 62]; // e8, f8, g8
const CASTLE_B_QS_CHECK: [u8; 3] = [60, 59, 58]; // e8, d8, c8

const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<Position>() == 192, "Position must be exactly 192 bytes");
    assert!(align_of::<Position>() == 32, "Position must be 32-byte aligned");

    assert!(offset_of!(Position, side_bb) == 0);
    assert!(offset_of!(Position, role_bb) == 16);
    assert!(offset_of!(Position, occ) == 64);
    assert!(offset_of!(Position, hash) == 72);
    assert!(offset_of!(Position, pawn_key) == 80);
    assert!(offset_of!(Position, minor_key) == 88);
    assert!(offset_of!(Position, major_key) == 96);
    assert!(offset_of!(Position, pieces) == 104);
    assert!(offset_of!(Position, castling_rooks) == 168);
    assert!(offset_of!(Position, castling_rights) == 172);
    assert!(offset_of!(Position, stm) == 173);
    assert!(offset_of!(Position, halfmove_clock) == 174);
    assert!(offset_of!(Position, en_passant) == 175);
    assert!(offset_of!(Position, frc_notation) == 177);
    assert!(offset_of!(Position, fullmove_number) == 178);

    assert!(size_of::<StateInfo>() == 40, "StateInfo must be exactly 40 bytes");
    assert!(align_of::<StateInfo>() == 8);
};

#[rustfmt::skip]
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Position { // 180 B + 12 B tail padding
    pub(crate) side_bb: [Bitboard; 2],      // 16 B
    pub(crate) role_bb: [Bitboard; 6],      // 48 B
    pub(crate) occ: Bitboard,               // 8 B
    pub(crate) hash: u64,                   // 8 B
    pub(crate) pawn_key: u64,               // 8 B
    pub(crate) minor_key: u64,              // 8 B
    pub(crate) major_key: u64,              // 8 B
    pub(crate) pieces: [PieceType; 64],     // 64 B
    pub(crate) castling_rooks: [Square; 4], // 4 B
    pub(crate) castling_rights: u8,         // 1 B
    pub(crate) stm: Color,                  // 1 B
    pub(crate) halfmove_clock: u8,          // 1 B
    pub(crate) en_passant: Option<Square>,  // 2 B
    pub(crate) frc_notation: bool,          // 1 B
    pub(crate) fullmove_number: u16,        // 2 B
}

#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub(crate) struct StateInfo {
    pub(crate) hash: u64,
    pub(crate) pawn_key: u64,
    pub(crate) minor_key: u64,
    pub(crate) major_key: u64,
    pub(crate) en_passant: Option<Square>,
    pub(crate) castling_rights: u8,
    pub(crate) captured: PieceType,
    pub(crate) halfmove_clock: u8,
}

impl Position {
    pub(crate) fn empty() -> Self {
        let mut pos = Self {
            side_bb: [Bitboard(0); 2],
            role_bb: [Bitboard(0); 6],
            occ: Bitboard(0),
            hash: 0,
            pawn_key: 0,
            minor_key: 0,
            major_key: 0,
            pieces: [PieceType::None; 64],
            castling_rooks: [Square(0); 4],
            castling_rights: 0,
            stm: Color::White,
            halfmove_clock: 0,
            en_passant: None,
            frc_notation: false,
            fullmove_number: 1,
        };

        pos.hash = pos.calc_zobrist();
        pos
    }

    #[inline]
    pub fn stm(&self) -> Color { self.stm }

    /// Panics on invalid FEN.
    pub fn from_fen(fen: &str) -> Self { Self::try_from_fen(fen).unwrap_or_else(|e| panic!("Invalid FEN: {e}")) }
    pub(crate) fn try_from_fen(fen: &str) -> Result<Self, FenError> {
        Self::try_from_tokens(&mut fen.split_whitespace().peekable())
    }

    /// Leaves a trailing `moves` token unconsumed.
    pub(crate) fn try_from_tokens<'a, I>(parts: &mut iter::Peekable<I>) -> Result<Self, FenError>
    where I: Iterator<Item = &'a str> {
        fen::try_from_tokens(parts)
    }

    #[inline]
    pub(crate) fn make_move(&mut self, mv: Move, acc: &mut I16x8) -> StateInfo { make::make_move(self, mv, acc) }

    #[inline]
    pub(crate) fn unmake_move(&mut self, mv: Move, info: &StateInfo) { make::unmake_move(self, mv, info); }

    #[inline]
    pub(crate) fn make_null_move(&mut self) -> StateInfo {
        let info = StateInfo {
            hash: self.hash,
            pawn_key: self.pawn_key,
            minor_key: self.minor_key,
            major_key: self.major_key,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            captured: PieceType::None,
            halfmove_clock: self.halfmove_clock,
        };

        if let Some(ep) = self.en_passant.take() {
            self.hash ^= zobrist::key_ep(ep);
        }

        self.stm = self.stm.opposite();
        self.hash ^= zobrist::key_side();
        self.halfmove_clock = self.halfmove_clock.saturating_add(1);
        info
    }

    #[inline]
    pub(crate) fn unmake_null_move(&mut self, info: &StateInfo) {
        self.stm = self.stm.opposite();
        self.hash = info.hash;
        self.pawn_key = info.pawn_key;
        self.minor_key = info.minor_key;
        self.major_key = info.major_key;
        self.en_passant = info.en_passant;
        self.halfmove_clock = info.halfmove_clock;
    }

    #[inline]
    pub(crate) fn has_non_pawn_material(&self, side: Color) -> bool {
        let material =
            self.side_bb[side as usize] & !(self.role_bb[PieceType::Pawn as usize] | self.role_bb[PieceType::King as usize]);
        material.is_not_empty()
    }

    pub(crate) fn psqt_accumulator(&self) -> I16x8 {
        let mut acc = I16x8::splat(0);
        for sq in self.occ {
            acc += psqt::entry(self.piece_at(sq), sq, self.color_at(sq));
        }
        acc
    }

    pub(crate) fn is_threefold_repetition(&self, history: &[u64]) -> bool {
        // A pawn move or capture cuts off any possible repetition.
        let limit = history.len().saturating_sub(self.halfmove_clock as usize + 1);
        let mut count = 1;

        // Same side to move, excluding the current position.
        for &h in history[limit..].iter().rev().skip(2).step_by(2) {
            if h == self.hash {
                count += 1;
                if count >= 3 {
                    return true;
                }
            }
        }
        false
    }

    pub(crate) fn is_fifty_move_draw(&self) -> bool { self.halfmove_clock >= 100 }

    /// K vs K, K+N vs K, and K+B vs K only.
    pub(crate) fn is_draw_by_material(&self) -> bool {
        let pawns_or_majors = self.role_bb[PieceType::Pawn as usize]
            | self.role_bb[PieceType::Rook as usize]
            | self.role_bb[PieceType::Queen as usize];

        if pawns_or_majors.is_not_empty() {
            return false;
        }

        let minors = self.role_bb[PieceType::Knight as usize].popcount() + self.role_bb[PieceType::Bishop as usize].popcount();
        minors <= 1
    }

    /// `ksq` and `rsq` must hold `color`'s king and castling rook.
    #[inline]
    pub(crate) fn is_castle_legal(&self, color: Color, ksq: Square, rsq: Square) -> bool {
        let queenside = rsq.file() < ksq.file();
        let (right, layout, check_sqs, empty): (u8, &[u8; 4], &[u8], Bitboard) = match (color, queenside) {
            (Color::White, false) => (WHITE_OO, &CASTLE_W_KS, &CASTLE_W_KS_CHECK, W_OO_EMPTY),
            (Color::White, true) => (WHITE_OOO, &CASTLE_W_QS, &CASTLE_W_QS_CHECK, W_OOO_EMPTY),
            (Color::Black, false) => (BLACK_OO, &CASTLE_B_KS, &CASTLE_B_KS_CHECK, B_OO_EMPTY),
            (Color::Black, true) => (BLACK_OOO, &CASTLE_B_QS, &CASTLE_B_QS_CHECK, B_OOO_EMPTY),
        };

        self.castling_rights & right != 0 && self.is_castle_path_legal(ksq, rsq, layout, check_sqs, empty, color.opposite())
    }

    pub(crate) fn has_nonstandard_castling_rook(&self) -> bool {
        [(ROOK_W_KS, 7), (ROOK_W_QS, 0), (ROOK_B_KS, 63), (ROOK_B_QS, 56)]
            .into_iter()
            .any(|(slot, home)| self.castling_rights & ROOK_RIGHTS[slot] != 0 && self.castling_rooks[slot] != Square(home))
    }

    #[inline]
    fn is_castle_path_legal(
        &self,
        ksq: Square,
        rsq: Square,
        layout: &[u8; 4],
        std_king_path: &[u8],
        empty_mask: Bitboard,
        attacker: Color,
    ) -> bool {
        let &[std_king, std_rook, king_dst, rook_dst] = layout;

        if ksq.0 == std_king && rsq.0 == std_rook && (self.occ & empty_mask).is_empty() {
            for &sq in std_king_path {
                if self.is_attacked(Square(sq), attacker, Bitboard::EMPTY) {
                    return false;
                }
            }
            return true;
        }

        let king_dst = Square(king_dst);
        let rook_dst = Square(rook_dst);
        let king_path = bitboard::between_bb(ksq, king_dst) | ksq.bitboard() | king_dst.bitboard();
        let rook_path = bitboard::between_bb(rsq, rook_dst) | rsq.bitboard() | rook_dst.bitboard();
        let movers = ksq.bitboard() | rsq.bitboard();

        if (self.occ & !movers & (king_path | rook_path)).is_not_empty() {
            return false;
        }

        let king = ksq.bitboard();
        for sq in king_path {
            if self.is_attacked(sq, attacker, king) {
                return false;
            }
        }

        true
    }

    #[cfg(feature = "dataset")]
    #[inline]
    pub(crate) const fn castling_rook_slot(rook: Square, king: Square, color: Color) -> usize {
        match (color, rook.file() > king.file()) {
            (Color::White, true) => ROOK_W_KS,
            (Color::White, false) => ROOK_W_QS,
            (Color::Black, true) => ROOK_B_KS,
            (Color::Black, false) => ROOK_B_QS,
        }
    }

    #[inline]
    pub(crate) fn for_each_castle(&self, color: Color, mut emit: impl FnMut(Move)) {
        let king_bb = self.pieces(PieceType::King, color);
        if king_bb.is_empty() {
            return;
        }

        let ksq = king_bb.lsb();
        let slots = match color {
            Color::White => [ROOK_W_KS, ROOK_W_QS],
            Color::Black => [ROOK_B_KS, ROOK_B_QS],
        };

        for slot in slots {
            if self.castling_rights & ROOK_RIGHTS[slot] == 0 {
                continue;
            }

            let rsq = self.castling_rooks[slot];
            if self.is_castle_legal(color, ksq, rsq) {
                emit(Move::new(ksq, rsq, Move::CASTLE));
            }
        }
    }

    pub(crate) fn to_fen(self) -> String { self.to_string() }
    pub(crate) fn pretty_print(&self) { fen::pretty_print(self) }

    #[inline(always)]
    pub(crate) fn piece_at(&self, sq: Square) -> PieceType { self.pieces[sq.as_usize()] }

    #[inline(always)]
    pub(crate) fn expect_piece_at(&self, sq: Square) -> PieceType {
        debug_assert!(self.occ.check_bit(sq));
        let pt = self.piece_at(sq);
        debug_assert!(pt != PieceType::None);
        pt
    }

    /// The piece `mv` takes, read before the move is made. En passant takes a pawn
    /// that is not on `mv.to()`; castling and other quiet moves yield `PieceType::None`.
    #[inline(always)]
    pub(crate) fn captured_piece(&self, mv: Move) -> PieceType {
        if mv.is_castling() {
            PieceType::None
        } else if mv.is_en_passant() {
            PieceType::Pawn
        } else {
            self.piece_at(mv.to())
        }
    }

    #[inline(always)]
    pub(crate) const fn color_at(&self, sq: Square) -> Color {
        debug_assert!(self.occ.check_bit(sq));
        Color::from(self.side_bb[Color::Black.as_usize()].check_bit(sq) as u8)
    }

    #[inline(always)]
    pub(crate) fn pieces(&self, pt: PieceType, color: Color) -> Bitboard {
        debug_assert!(pt != PieceType::None);
        self.role_bb[pt.as_usize()] & self.side_bb[color.as_usize()]
    }

    /// P=1, N=B=3, R=5, Q=9 across both colors.
    #[inline]
    pub(crate) const fn material_count(&self) -> u32 {
        let p = self.role_bb[PieceType::Pawn as usize].popcount();
        let n = self.role_bb[PieceType::Knight as usize].popcount();
        let b = self.role_bb[PieceType::Bishop as usize].popcount();
        let r = self.role_bb[PieceType::Rook as usize].popcount();
        let q = self.role_bb[PieceType::Queen as usize].popcount();
        p + 3 * n + 3 * b + 5 * r + 9 * q
    }

    #[inline(always)]
    pub(crate) const fn piece_count(&self, pt: PieceType) -> i32 {
        if matches!(pt, PieceType::None) { 0 } else { self.role_bb[pt as usize].popcount() as i32 }
    }

    /// Excludes `mask_out` from slider occupancy; leaper attacks and piece sets are unchanged.
    #[inline(always)]
    pub(crate) fn is_attacked(&self, sq: Square, attacker: Color, mask_out: Bitboard) -> bool {
        attacks::is_attacked(self, sq, attacker, mask_out)
    }

    #[inline(always)]
    pub(crate) fn checkers(&self) -> Bitboard { attacks::checkers(self) }

    #[inline]
    pub(crate) fn pawn_attacks(&self, color: Color) -> Bitboard {
        let pawns = self.pieces(PieceType::Pawn, color);
        match color {
            Color::White => pawns.shift(Direction::NorthEast) | pawns.shift(Direction::NorthWest),
            Color::Black => pawns.shift(Direction::SouthEast) | pawns.shift(Direction::SouthWest),
        }
    }

    #[cfg(any(test, feature = "rigs", feature = "nostore"))]
    #[inline]
    pub(crate) fn threats(&self, color: Color) -> Bitboard {
        let us = self.side_bb[color];
        let empty = U64x4::splat(!self.occ.0);

        let knights = (self.role_bb[PieceType::Knight] & us).0;
        let bq = ((self.role_bb[PieceType::Bishop] | self.role_bb[PieceType::Queen]) & us).0;
        let rq = ((self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen]) & us).0;
        let king = (self.role_bb[PieceType::King] & us).0;

        let sliders = spatial::atk_rooks(U64x4::splat(rq), empty) | spatial::atk_bishops(U64x4::splat(bq), empty);
        let leapers = spatial::atk_knights(U64x4::splat(knights)) | spatial::atk_kings(U64x4::splat(king));
        Bitboard((sliders | leapers).extract::<0>()) | self.pawn_attacks(color)
    }

    /// Higher-value enemy pieces newly attacked from `to`, using the pre-move position.
    #[inline]
    pub(crate) fn new_threats(&self, pt: PieceType, from: Square, to: Square) -> Bitboard {
        let them = self.side_bb[self.stm.opposite()];
        let rq = (self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen]) & them;
        let q = self.role_bb[PieceType::Queen] & them;

        match pt {
            PieceType::Pawn => {
                let bigger = ((self.role_bb[PieceType::Knight] | self.role_bb[PieceType::Bishop]) & them) | rq;
                bitboard::atk_pawn(to, self.stm) & bigger
            },
            PieceType::Knight => bitboard::atk_knight(to) & rq,
            PieceType::Bishop => bitboard::atk_bishop(to, self.occ) & !bitboard::atk_bishop(from, self.occ) & rq,
            PieceType::Rook => bitboard::atk_rook(to, self.occ) & !bitboard::atk_rook(from, self.occ) & q,
            _ => Bitboard::EMPTY,
        }
    }

    #[inline]
    pub(crate) fn has_ep_attacker(&self, ep_sq: Square) -> bool { attacks::can_capture_ep(self, ep_sq, self.stm) }

    #[inline]
    pub(crate) fn pinned_pieces(&self, color: Color) -> Bitboard { attacks::pinned_pieces(self, color) }

    #[inline(always)]
    pub(crate) fn add_piece(&mut self, sq: Square, pt: PieceType, color: Color) { make::update_piece::<true>(self, sq, pt, color); }

    #[inline(always)]
    pub(crate) fn remove_piece(&mut self, sq: Square, pt: PieceType, color: Color) {
        make::update_piece::<false>(self, sq, pt, color);
    }

    pub(crate) fn recompute_hashes(&mut self) {
        self.hash = self.calc_zobrist();
        self.pawn_key = self.calc_pawn_hash();
        self.minor_key = self.calc_minor_hash();
        self.major_key = self.calc_major_hash();
    }

    pub(crate) fn calc_zobrist(&self) -> u64 {
        let mut key = self.calc_subset_hash(self.occ);

        if self.stm == Color::Black {
            key ^= zobrist::key_side();
        }

        if self.castling_rights != 0 {
            key ^= zobrist::key_castling(self.castling_rights);
        }

        if let Some(ep) = self.en_passant {
            key ^= zobrist::key_ep(ep);
        }
        key
    }

    pub(crate) fn calc_pawn_hash(&self) -> u64 { self.calc_subset_hash(self.role_bb[PieceType::Pawn]) }

    pub(crate) fn calc_minor_hash(&self) -> u64 {
        self.calc_subset_hash(self.role_bb[PieceType::Knight] | self.role_bb[PieceType::Bishop])
    }
    pub(crate) fn calc_major_hash(&self) -> u64 {
        self.calc_subset_hash(self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen])
    }

    fn calc_subset_hash(&self, pieces: Bitboard) -> u64 {
        let mut key = 0u64;
        for sq in pieces {
            key ^= zobrist::key_piece(self.piece_at(sq), self.color_at(sq), sq);
        }
        key
    }
}
