//! Attack detection and square validation queries.

use crate::core::{
    board::{
        Position,
        bitboard::{
            PSEUDO_BISHOP_ATTACKS, PSEUDO_ROOK_ATTACKS, atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook,
            between_bb,
        },
    },
    defs::{Bitboard, Color, PieceType, Square},
};

/// Tests whether `sq` is attacked by any piece belonging to `attacker`.
///
/// The `VIRTUAL` const generic enables transparent-king mode:
/// `mask_out` is erased from occupancy before slider rays are cast.
/// This is essential when checking if the king's destination is safe,
/// slider rays must see through the king's departure square, not stop on it.
#[inline(always)]
pub fn is_attacked<const VIRTUAL: bool>(
    pos: &Position,
    sq: Square,
    attacker: Color,
    mask_out: Bitboard,
) -> bool {
    let occ = if VIRTUAL {
        pos.occ & !mask_out
    } else {
        pos.occ
    };
    let them = pos.side_bb[attacker];

    // Leapers first — cheapest to test, no occupancy dependency.
    if (atk_pawn(sq, attacker.opposite()) & pos.role_bb[PieceType::Pawn] & them).is_not_empty() {
        return true;
    }
    if (atk_knight(sq) & pos.role_bb[PieceType::Knight] & them).is_not_empty() {
        return true;
    }
    if (atk_king(sq) & pos.role_bb[PieceType::King] & them).is_not_empty() {
        return true;
    }

    // Sliders — rook-movers (R+Q), then bishop-movers (B+Q).
    // Pre-masking with the attacker's bitboard saves an AND in the hot intersection
    // by restricting the occupancy-generated attack set to only relevant targets early.
    let rq = (pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen]) & them;
    if (atk_rook(sq, occ) & rq).is_not_empty() {
        return true;
    }

    let bq = (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen]) & them;
    (atk_bishop(sq, occ) & bq).is_not_empty()
}

// ──────── Checker & Attacker Queries ────────

/// Returns a bitboard of every enemy piece
/// currently giving check to the side-to-move's king.
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
/// this builds the full attacker set — needed for check evasion, SEE, and similar.
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

/// All pieces of either color attacking `sq`, computed against an explicit
/// occupancy mask.
///
/// Unlike `attackers_of` (which takes `pos.occ` implicitly and filters by
/// one color), this accepts an arbitrary `occ` so the caller can simulate
/// mid-exchange board states — the primary use case is SEE, where the set
/// of revealed attackers changes as each capture removes a blocker.
///
/// Pawn attacks are symmetric: `atk_pawn(sq, color.opposite())` returns the
/// squares from which a pawn of `color` would attack `sq`.
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

// ──────── En Passant Legality ────────

/// Can any pawn of `color` legally reach `ep_sq`?
///
/// Uses the symmetry trick:
/// pawn attacks generated from the EP square with the opponent's perspective
/// land exactly on the squares where a friendly pawn could capture.
/// If any such pawn exists, en passant is pseudo-legal.
///
/// Takes `color` explicitly rather than reading `pos.stm` to avoid
/// temporal coupling — callers always know which side is capturing.
#[inline]
pub fn can_capture_ep(pos: &Position, ep_sq: Square, color: Color) -> bool {
    let us = pos.side_bb[color];
    (atk_pawn(ep_sq, color.opposite()) & pos.role_bb[PieceType::Pawn] & us).is_not_empty()
}

// ──────── Pins & X-Rays ────────

/// Identifies friendly pieces pinned to our king by enemy sliders.
///
/// A piece is a King-blocker if it is the sole occupant of the ray between
/// our king and an enemy slider aligned with that ray.
/// Moving it off the line would expose the king to check.
///
/// # Algorithm
/// 1. Cast rays from the king on an empty board to spot
///    enemy "snipers" — sliders that sit on a potential attack line.
/// 2. For each sniper, inspect the segment between it and the king.
/// 3. If exactly one piece occupies that segment and its ours → pinned.
#[inline]
pub fn pinned_pieces(pos: &Position, color: Color) -> Bitboard {
    let opp = color.opposite();
    let us = pos.side_bb[color];

    let king_bb = pos.pieces(PieceType::King, color);
    if king_bb.is_empty() {
        return Bitboard(0);
    }
    let king_sq = king_bb.lsb();

    // Enemy sliders that would threaten the king on a completely empty board.
    let rq = pos.pieces(PieceType::Rook, opp) | pos.pieces(PieceType::Queen, opp);
    let bq = pos.pieces(PieceType::Bishop, opp) | pos.pieces(PieceType::Queen, opp);
    let snipers =
        (PSEUDO_ROOK_ATTACKS[usize::from(king_sq)] & rq) | (PSEUDO_BISHOP_ATTACKS[usize::from(king_sq)] & bq);

    let mut pinned = Bitboard(0);

    for sniper_sq in snipers {
        let between = between_bb(king_sq, sniper_sq) & pos.occ;

        // One occupant on the ray, and it's ours — that piece is pinned.
        if between.popcount() == 1 && (between & us).is_not_empty() {
            pinned |= between;
        }
    }

    pinned
}
