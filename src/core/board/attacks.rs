//! Attack detection and square validation queries.

use crate::core::{
    board::{
        Position,
        bitboard::{PSEUDO_BISHOP_ATTACKS, PSEUDO_ROOK_ATTACKS, atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, between_bb},
    },
    defs::{Bitboard, Color, PieceType, Square},
};

/// Returns whether `sq` is attacked by any piece of `attacker`.
///
/// `VIRTUAL` removes `mask_out` from occupancy before the slider rays are cast, so a
/// king's departure square stops blocking the check it was hiding from.
#[inline(always)]
pub fn is_attacked<const VIRTUAL: bool>(pos: &Position, sq: Square, attacker: Color, mask_out: Bitboard) -> bool {
    let occ = if VIRTUAL { pos.occ & !mask_out } else { pos.occ };
    let them = pos.side_bb[attacker];

    // Non-slider attacks: evaluated first as they are independent of occupancy.
    if (atk_pawn(sq, attacker.opposite()) & pos.role_bb[PieceType::Pawn] & them).is_not_empty() {
        return true;
    }
    if (atk_knight(sq) & pos.role_bb[PieceType::Knight] & them).is_not_empty() {
        return true;
    }
    if (atk_king(sq) & pos.role_bb[PieceType::King] & them).is_not_empty() {
        return true;
    }

    // Slider attacks: evaluated against the (potentially masked) occupancy.
    let rq = (pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen]) & them;
    if (atk_rook(sq, occ) & rq).is_not_empty() {
        return true;
    }

    let bq = (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen]) & them;
    (atk_bishop(sq, occ) & bq).is_not_empty()
}

/// Returns a bitboard of every enemy piece currently giving check to the side-to-move's king.
#[inline(always)]
pub fn checkers(pos: &Position) -> Bitboard {
    let king_bb = pos.pieces(PieceType::King, pos.stm);
    if king_bb.is_empty() {
        return Bitboard(0);
    }
    attackers_of(pos, king_bb.lsb(), pos.stm.opposite())
}

/// Collects all pieces of `attacker`'s army that attack `sq`.
///
/// Unlike `is_attacked` (which short-circuits on the first hit),
/// this builds the full attacker set: needed for check evasion, SEE, and similar.
#[inline(always)]
pub fn attackers_of(pos: &Position, sq: Square, attacker: Color) -> Bitboard {
    let occ = pos.occ;
    let them = pos.side_bb[attacker];
    (atk_pawn(sq, attacker.opposite()) & pos.role_bb[PieceType::Pawn] & them)
        | (atk_knight(sq) & pos.role_bb[PieceType::Knight] & them)
        | (atk_king(sq) & pos.role_bb[PieceType::King] & them)
        | (atk_rook(sq, occ) & (pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen]) & them)
        | (atk_bishop(sq, occ) & (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen]) & them)
}

/// All pieces of both colors attacking `sq`, against an occupancy the caller supplies.
///
/// SEE needs that: each capture removes a blocker, and the attackers behind it only
/// appear against an occupancy that already has it gone. Pawns come off an inverse
/// lookup, since `atk_pawn(sq, opp)` names the squares a pawn of `color` strikes from.
#[inline(always)]
pub fn all_attackers_to(pos: &Position, sq: Square, occ: Bitboard) -> Bitboard {
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

/// Whether a pawn of `color` can capture onto `ep_sq`, pseudo-legally.
///
/// The side is a parameter and not `pos.stm`, because `make_move` asks after the
/// turn has already flipped.
#[inline]
pub fn can_capture_ep(pos: &Position, ep_sq: Square, color: Color) -> bool {
    let us = pos.side_bb[color];
    (atk_pawn(ep_sq, color.opposite()) & pos.role_bb[PieceType::Pawn] & us).is_not_empty()
}

/// Identifies friendly pieces absolutely pinned to the king along enemy slider rays.
///
/// Finds enemy sliders sharing an unobstructed line of sight with the king,
/// then flags rays containing exactly one intervening friendly blocker.
#[inline]
pub fn pinned_pieces(pos: &Position, color: Color) -> Bitboard {
    let opp = color.opposite();
    let us = pos.side_bb[color];
    let king_bb = pos.pieces(PieceType::King, color);
    if king_bb.is_empty() {
        return Bitboard(0);
    }
    let king_sq = king_bb.lsb();

    // Potential enemy sliders aligned along empty-board king rays ("snipers").
    let rq = pos.pieces(PieceType::Rook, opp) | pos.pieces(PieceType::Queen, opp);
    let bq = pos.pieces(PieceType::Bishop, opp) | pos.pieces(PieceType::Queen, opp);
    let snipers = (PSEUDO_ROOK_ATTACKS[usize::from(king_sq)] & rq) | (PSEUDO_BISHOP_ATTACKS[usize::from(king_sq)] & bq);

    let mut pinned = Bitboard(0);

    for sniper_sq in snipers {
        let between = between_bb(king_sq, sniper_sq) & pos.occ;
        // Exactly one intervening blocker, and it is friendly.
        if between.popcount() == 1 && (between & us).is_not_empty() {
            pinned |= between;
        }
    }
    pinned
}

/// Both colors' king-pinned pieces and their king squares.
///
/// A pin binds the same way whichever question is asked, so the position is scanned
/// once and legality and SEE read the one answer.
#[derive(Clone, Copy)]
pub struct Pins {
    pinned: [Bitboard; 2],
    king_sq: [Square; 2],
}

impl Pins {
    #[inline]
    pub fn new(pos: &Position) -> Self {
        debug_assert!(
            pos.pieces(PieceType::King, Color::White).is_not_empty() && pos.pieces(PieceType::King, Color::Black).is_not_empty(),
            "Pins reads both king squares, and a kingless position has none to give"
        );
        Self {
            pinned: [pinned_pieces(pos, Color::White), pinned_pieces(pos, Color::Black)],
            king_sq: [pos.pieces(PieceType::King, Color::White).lsb(), pos.pieces(PieceType::King, Color::Black).lsb()],
        }
    }

    /// Pieces of `color` pinned to their own king.
    #[inline]
    pub fn blockers(&self, color: Color) -> Bitboard {
        self.pinned[color]
    }

    /// `color`'s king square.
    #[inline]
    pub fn king(&self, color: Color) -> Square {
        self.king_sq[color]
    }
}
