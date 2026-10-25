//! Attack, pin and en passant queries on a `Position`.

use crate::core::{
    board::{
        Position,
        bitboard::{BISHOP_RAYS, ROOK_RAYS, atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, between_bb},
    },
    defs::{Bitboard, Color, PieceType, Square},
};

#[inline(always)]
pub(super) fn is_attacked(pos: &Position, sq: Square, attacker: Color, mask_out: Bitboard) -> bool {
    let occ = pos.occ & !mask_out;
    let them = pos.side_bb[attacker];

    if (atk_pawn(sq, attacker.opposite()) & pos.role_bb[PieceType::Pawn] & them).is_not_empty() {
        return true;
    }

    if (atk_knight(sq) & pos.role_bb[PieceType::Knight] & them).is_not_empty() {
        return true;
    }

    if (atk_king(sq) & pos.role_bb[PieceType::King] & them).is_not_empty() {
        return true;
    }

    let rq = (pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen]) & them;
    if (atk_rook(sq, occ) & rq).is_not_empty() {
        return true;
    }

    let bq = (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen]) & them;
    (atk_bishop(sq, occ) & bq).is_not_empty()
}

#[inline(always)]
pub(super) fn checkers(pos: &Position) -> Bitboard {
    let king_bb = pos.pieces(PieceType::King, pos.stm);
    if king_bb.is_empty() {
        return Bitboard(0);
    }
    attackers_of(pos, king_bb.lsb(), pos.stm.opposite())
}

#[inline(always)]
fn attackers_of(pos: &Position, sq: Square, attacker: Color) -> Bitboard {
    all_attackers_to(pos, sq, pos.occ) & pos.side_bb[attacker]
}

/// All pieces attacking `sq` under `occ`.
/// Piece sets still come from `pos`, so callers must exclude captured pieces from the result.
#[inline(always)]
pub(crate) fn all_attackers_to(pos: &Position, sq: Square, occ: Bitboard) -> Bitboard {
    let pawns = pos.role_bb[PieceType::Pawn];
    let white_pawns = pawns & pos.side_bb[Color::White];
    let black_pawns = pawns & pos.side_bb[Color::Black];
    (atk_pawn(sq, Color::Black) & white_pawns)
        | (atk_pawn(sq, Color::White) & black_pawns)
        | (atk_knight(sq) & pos.role_bb[PieceType::Knight])
        | (atk_king(sq) & pos.role_bb[PieceType::King])
        | (atk_rook(sq, occ) & (pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen]))
        | (atk_bishop(sq, occ) & (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen]))
}

/// Whether a pawn of `color` can capture onto `ep_sq`.
/// `color` is explicit because `make_move` calls this after flipping the turn.
#[inline]
pub(super) fn can_capture_ep(pos: &Position, ep_sq: Square, color: Color) -> bool {
    let us = pos.side_bb[color];
    (atk_pawn(ep_sq, color.opposite()) & pos.role_bb[PieceType::Pawn] & us).is_not_empty()
}

#[inline]
pub(super) fn pinned_pieces(pos: &Position, color: Color) -> Bitboard {
    let opp = color.opposite();
    let us = pos.side_bb[color];
    let king_bb = pos.pieces(PieceType::King, color);

    if king_bb.is_empty() {
        return Bitboard(0);
    }

    let king_sq = king_bb.lsb();

    // Enemy sliders that would see the king on an empty board.
    let rq = pos.pieces(PieceType::Rook, opp) | pos.pieces(PieceType::Queen, opp);
    let bq = pos.pieces(PieceType::Bishop, opp) | pos.pieces(PieceType::Queen, opp);
    let snipers = (ROOK_RAYS[usize::from(king_sq)] & rq) | (BISHOP_RAYS[usize::from(king_sq)] & bq);

    let mut pinned = Bitboard(0);

    for sniper_sq in snipers {
        let between = between_bb(king_sq, sniper_sq) & pos.occ;
        if between.popcount() == 1 && (between & us).is_not_empty() {
            pinned |= between;
        }
    }

    pinned
}

/// Both colors' pinned pieces and king squares.
#[derive(Clone, Copy)]
pub(crate) struct Pins {
    pinned: [Bitboard; 2],
    king_sq: [Square; 2],
}

impl Pins {
    #[inline]
    pub(crate) fn new(pos: &Position) -> Self {
        debug_assert!(
            pos.pieces(PieceType::King, Color::White).is_not_empty() && pos.pieces(PieceType::King, Color::Black).is_not_empty(),
            "Pins requires both kings"
        );
        Self {
            pinned: [pinned_pieces(pos, Color::White), pinned_pieces(pos, Color::Black)],
            king_sq: [pos.pieces(PieceType::King, Color::White).lsb(), pos.pieces(PieceType::King, Color::Black).lsb()],
        }
    }

    #[inline]
    pub(crate) fn pinned(&self, color: Color) -> Bitboard { self.pinned[color] }

    #[inline]
    pub(crate) fn king(&self, color: Color) -> Square { self.king_sq[color] }
}
