//! Move application and incremental state updates.

use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, StateInfo, WHITE_OO, WHITE_OOO},
        defs::{Color, PieceType, Square},
        moves::Move,
        psqt, zobrist,
    },
    weave::Vi16x8,
};

/// Applies a move to the position, incrementally maintaining the Zobrist hash
/// and accumulator. Returns a [`StateInfo`] snapshot for perfect rollback.
#[inline(always)]
pub fn make_move(pos: &mut Position, mv: Move, acc: &mut Vi16x8) -> StateInfo {
    let stm = pos.stm;
    let opp = stm.opposite();
    let from = mv.from();
    let to = mv.to();
    let pt = pos.expect_piece_at(from);

    debug_assert!(pt != PieceType::None, "make_move: no piece on {from} (move: {})\n{}", mv.to_uci(pos.is_frc), pos.as_fen());

    let captured = if mv.is_castling() { PieceType::None } else { pos.piece_at(to) };

    let state = StateInfo {
        castling_rights: pos.castling_rights,
        hash: pos.hash,
        pawn_key: pos.pawn_key,
        minor_key: pos.minor_key,
        major_key: pos.major_key,
        captured,
        halfmove_clock: pos.halfmove_clock,
        en_passant: pos.en_passant,
    };

    let placed = if mv.is_promotion() { mv.promo().unwrap_or(PieceType::Queen) } else { pt };

    update_accumulator(pos, acc, mv, pt, captured, placed);

    pos.halfmove_clock =
        if pt == PieceType::Pawn || captured != PieceType::None { 0 } else { pos.halfmove_clock.saturating_add(1) };

    if captured != PieceType::None {
        pos.hash ^= zobrist::key_piece(captured, opp, to);
        toggle_corr_key(pos, captured, opp, to);
        pos.remove_piece(to, captured, opp);
    } else if mv.is_en_passant() {
        let victim_sq = to ^ 8;

        pos.hash ^= zobrist::key_piece(PieceType::Pawn, opp, victim_sq);
        toggle_corr_key(pos, PieceType::Pawn, opp, victim_sq);
        pos.remove_piece(victim_sq, PieceType::Pawn, opp);
    }

    if mv.is_castling() {
        apply_castling(pos, from, to, stm);
    } else {
        pos.remove_piece(from, pt, stm);
        pos.hash ^= zobrist::key_piece(pt, stm, from);
        toggle_corr_key(pos, pt, stm, from);

        pos.add_piece(to, placed, stm);
        pos.hash ^= zobrist::key_piece(placed, stm, to);
        toggle_corr_key(pos, placed, stm, to);
    }

    pos.stm = opp;

    if stm == Color::Black {
        pos.fullmove_number += 1;
    }

    pos.hash ^= zobrist::key_side();

    refresh_castling_rights(pos, pt, stm, from, to, state.castling_rights);
    refresh_en_passant(pos, mv, from, to, state.en_passant);
    state
}

/// Reverses a move, perfectly restoring the prior position.
///
/// The accumulator must be bulk-restored from the snapshot
/// by the caller, no incremental undo needed.
/// This keeps unmake fast and avoids tricky sign-flip bugs.
#[inline(always)]
pub fn unmake_move(pos: &mut Position, mv: Move, info: &StateInfo) {
    pos.stm = pos.stm.opposite();
    let stm = pos.stm;

    if stm == Color::Black {
        pos.fullmove_number -= 1;
    }

    let opp = stm.opposite();
    let from = mv.from();
    let to = mv.to();

    pos.castling_rights = info.castling_rights;
    pos.en_passant = info.en_passant;
    pos.halfmove_clock = info.halfmove_clock;
    pos.hash = info.hash;
    pos.pawn_key = info.pawn_key;
    pos.minor_key = info.minor_key;
    pos.major_key = info.major_key;

    if mv.is_castling() {
        revert_castling(pos, from, to);
        return;
    }

    let placed = pos.expect_piece_at(to);
    let original = if mv.is_promotion() { PieceType::Pawn } else { placed };

    pos.remove_piece(to, placed, stm);
    pos.add_piece(from, original, stm);

    if info.captured != PieceType::None {
        pos.add_piece(to, info.captured, opp);
    } else if mv.is_en_passant() {
        pos.add_piece(to ^ 8, PieceType::Pawn, opp);
    }
}

/// Incrementally updates the accumulator with PSQT vector deltas.
///
/// This is pure arithmetic over the position's read-only board state; nothing is mutated.
/// Castling is handled as a clean four-delta update rather than
/// the add-then-undo pattern, since we know the exact geometry up front.
#[inline]
pub fn update_accumulator(pos: &Position, acc: &mut Vi16x8, mv: Move, pt: PieceType, captured: PieceType, placed: PieceType) {
    let stm = pos.stm;
    let from = mv.from();
    let to = mv.to();

    if mv.is_castling() {
        let (king_to, rook_to) = castling_targets(from, to);
        *acc -= psqt::get_vec(PieceType::King, from, stm);
        *acc += psqt::get_vec(PieceType::King, king_to, stm);
        *acc -= psqt::get_vec(PieceType::Rook, to, stm);
        *acc += psqt::get_vec(PieceType::Rook, rook_to, stm);
        return;
    }

    let opp = stm.opposite();

    *acc -= psqt::get_vec(pt, from, stm);
    *acc += psqt::get_vec(placed, to, stm);

    if captured != PieceType::None {
        *acc -= psqt::get_vec(captured, to, opp);
    } else if mv.is_en_passant() {
        // The captured pawn is always exactly one rank from `to`:
        //   White captures up   → victim is on rank(to)-1 → to ^ 8 subtracts 8.
        //   Black captures down → victim is on rank(to)+1 → to ^ 8 adds 8.
        //
        // XOR-8 toggles bit 3, which equals ±8 in index space.
        // The direction is always correct because the captured pawn occupies
        // the rank that still has its bit-3 in the opposite state from `to`.
        *acc -= psqt::get_vec(PieceType::Pawn, to ^ 8, opp);
    }
}

/// Compile-time add/remove of a piece on the board.
/// The const generic eliminates the branch entirely: zero-cost abstraction.
#[inline(always)]
pub fn update_piece<const ADD: bool>(pos: &mut Position, sq: Square, pt: PieceType, color: Color) {
    if ADD {
        pos.role_bb[pt].set_bit(sq);
        pos.side_bb[color].set_bit(sq);
        pos.occ.set_bit(sq);
        pos.set_piece_type(sq, pt);
    } else {
        pos.role_bb[pt].clear_bit(sq);
        pos.side_bb[color].clear_bit(sq);
        pos.occ.clear_bit(sq);
        pos.set_piece_type(sq, PieceType::None);
    }
}

/// Routes one piece's Zobrist key into the correction key its type owns;
/// pawns to `pawn_key`, knight/bishop to `minor_key`, rook/queen to `major_key`,
/// king to neither. Mirrors the inline `hash` toggles in `make_move`;
/// XOR is self-inverse, so the same call serves a piece leaving its origin
/// and arriving on its destination.
#[inline(always)]
fn toggle_corr_key(pos: &mut Position, pt: PieceType, color: Color, sq: Square) {
    let key = zobrist::key_piece(pt, color, sq);

    match pt {
        PieceType::Pawn => pos.pawn_key ^= key,
        PieceType::Knight | PieceType::Bishop => pos.minor_key ^= key,
        PieceType::Rook | PieceType::Queen => pos.major_key ^= key,
        PieceType::King | PieceType::None => {},
    }
}

/// Undoes a castling move on the board.
/// Hash and accumulator are bulk-restored from the snapshot;
/// only bitboards and the mailbox need rewinding.
#[inline]
pub fn revert_castling(pos: &mut Position, king_from: Square, rook_from: Square) {
    let (king_to, rook_to) = castling_targets(king_from, rook_from);
    let stm = pos.stm;

    update_piece::<false>(pos, king_to, PieceType::King, stm);
    update_piece::<false>(pos, rook_to, PieceType::Rook, stm);
    update_piece::<true>(pos, king_from, PieceType::King, stm);
    update_piece::<true>(pos, rook_from, PieceType::Rook, stm);
}

/// Resolves the final landing squares for a castling move.
///
/// Our move encoding stores the rook's home square in `mv.to()`; this is FRC-safe
/// since the rook can live on any file. From the relative positions of king and rook
/// we derive:
///   O-O-O → king lands on c-file, rook on d-file
///   O-O   → king lands on g-file, rook on f-file
#[inline(always)]
fn castling_targets(king_sq: Square, rook_sq: Square) -> (Square, Square) {
    let queenside = rook_sq.file() < king_sq.file();
    let rank = king_sq.rank();
    (
        Square::from_coords(if queenside { 2 } else { 6 }, rank),
        Square::from_coords(if queenside { 3 } else { 5 }, rank),
    )
}

/// Executes castling on the board and patches the Zobrist hash.
///
/// Both pieces are lifted before either is placed; this is critical for
/// DFRC where a destination square can coincide with the other piece's origin.
/// The king's hash removal is deliberately handled here because `make_move` skips it.
#[inline]
fn apply_castling(pos: &mut Position, king_from: Square, rook_from: Square, stm: Color) {
    let (king_to, rook_to) = castling_targets(king_from, rook_from);

    // Lift both first (DFRC-safe: no source/destination aliasing).
    pos.remove_piece(king_from, PieceType::King, stm);
    pos.hash ^= zobrist::key_piece(PieceType::King, stm, king_from);
    pos.remove_piece(rook_from, PieceType::Rook, stm);
    pos.hash ^= zobrist::key_piece(PieceType::Rook, stm, rook_from);
    toggle_corr_key(pos, PieceType::Rook, stm, rook_from);

    // Land at final squares.
    pos.add_piece(king_to, PieceType::King, stm);
    pos.hash ^= zobrist::key_piece(PieceType::King, stm, king_to);
    pos.add_piece(rook_to, PieceType::Rook, stm);
    pos.hash ^= zobrist::key_piece(PieceType::Rook, stm, rook_to);
    toggle_corr_key(pos, PieceType::Rook, stm, rook_to);
}

/// Revokes castling rights touched by the move's origin and destination.
///
/// A single combined AND handles king/rook departures (from-square)
/// and rook captures (to-square). If anything changed, we XOR the old
/// and new rights into the hash in one shot.
#[inline]
fn refresh_castling_rights(pos: &mut Position, pt: PieceType, stm: Color, from: Square, to: Square, old: u8) {
    if old == 0 {
        return;
    }

    let mut rights = old;

    if pt == PieceType::King {
        rights &= if stm == Color::White { !(WHITE_OO | WHITE_OOO) } else { !(BLACK_OO | BLACK_OOO) };
    }

    if from == pos.castling_rooks[ROOK_W_KS] || to == pos.castling_rooks[ROOK_W_KS] {
        rights &= !WHITE_OO;
    }
    if from == pos.castling_rooks[ROOK_W_QS] || to == pos.castling_rooks[ROOK_W_QS] {
        rights &= !WHITE_OOO;
    }

    if from == pos.castling_rooks[ROOK_B_KS] || to == pos.castling_rooks[ROOK_B_KS] {
        rights &= !BLACK_OO;
    }
    if from == pos.castling_rooks[ROOK_B_QS] || to == pos.castling_rooks[ROOK_B_QS] {
        rights &= !BLACK_OOO;
    }

    if rights != old {
        pos.hash ^= zobrist::key_castling(old) ^ zobrist::key_castling(rights);
        pos.castling_rights = rights;
    }
}

/// Maintains the en passant square in the position and hash.
///
/// Only recorded after a double pawn push and only when an enemy pawn can
/// actually capture; this avoids polluting the transposition table with
/// positions that differ solely by a phantom EP square no one can use.
#[inline]
fn refresh_en_passant(pos: &mut Position, mv: Move, from: Square, to: Square, old_ep: Option<Square>) {
    if let Some(sq) = old_ep {
        pos.hash ^= zobrist::key_ep(sq);
    }

    pos.en_passant = None;

    if mv.is_double_push() {
        let ep_sq = Square(u8::midpoint(from.0, to.0));

        if pos.can_capture_ep(ep_sq, pos.stm) {
            pos.en_passant = Some(ep_sq);
            pos.hash ^= zobrist::key_ep(ep_sq);
        }
    }
}
