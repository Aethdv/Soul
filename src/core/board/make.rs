//! Move application and incremental state updates.

use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, StateInfo, WHITE_OO, WHITE_OOO},
        defs::{Color, PieceType, Square},
        moves::Move,
        psqt, zobrist,
    },
    weave::I16x8,
};

#[inline(always)]
pub(super) fn make_move(pos: &mut Position, mv: Move, acc: &mut I16x8) -> StateInfo {
    let stm = pos.stm;
    let opp = stm.opposite();
    let from = mv.from();
    let to = mv.to();
    let pt = pos.expect_piece_at(from);

    let captured = if mv.is_castling() { PieceType::None } else { pos.piece_at(to) };

    // Written out here and in make_null_move, and restored field by field in both unmakes.
    // A shared constructor and restorer cost 4.1M more instructions over a 4.4M-node bench.
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

    let placed = mv.promo().unwrap_or(pt);

    update_accumulator(pos, acc, mv, pt, captured, placed);

    pos.halfmove_clock =
        if pt == PieceType::Pawn || captured != PieceType::None { 0 } else { pos.halfmove_clock.saturating_add(1) };

    if captured != PieceType::None {
        toggle_piece_keys(pos, captured, opp, to);
        pos.remove_piece(to, captured, opp);
    } else if mv.is_en_passant() {
        // The captured pawn stands on to's file one rank back: flipping bit 3 steps
        // rank 6 to 5 for White and rank 3 to 4 for Black.
        let victim_sq = to ^ 8;
        toggle_piece_keys(pos, PieceType::Pawn, opp, victim_sq);
        pos.remove_piece(victim_sq, PieceType::Pawn, opp);
    }

    if mv.is_castling() {
        apply_castling(pos, from, to, stm);
    } else {
        pos.remove_piece(from, pt, stm);
        toggle_piece_keys(pos, pt, stm, from);

        pos.add_piece(to, placed, stm);
        toggle_piece_keys(pos, placed, stm, to);
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

/// The accumulator is not touched: the caller restores its own copy.
#[inline(always)]
pub(super) fn unmake_move(pos: &mut Position, mv: Move, info: &StateInfo) {
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

#[inline]
fn update_accumulator(pos: &Position, acc: &mut I16x8, mv: Move, pt: PieceType, captured: PieceType, placed: PieceType) {
    let stm = pos.stm;
    let from = mv.from();
    let to = mv.to();

    if mv.is_castling() {
        let (king_to, rook_to) = castling_targets(from, to);
        *acc -= psqt::entry(PieceType::King, from, stm);
        *acc += psqt::entry(PieceType::King, king_to, stm);
        *acc -= psqt::entry(PieceType::Rook, to, stm);
        *acc += psqt::entry(PieceType::Rook, rook_to, stm);
        return;
    }

    let opp = stm.opposite();

    *acc -= psqt::entry(pt, from, stm);
    *acc += psqt::entry(placed, to, stm);

    if captured != PieceType::None {
        *acc -= psqt::entry(captured, to, opp);
    } else if mv.is_en_passant() {
        *acc -= psqt::entry(PieceType::Pawn, to ^ 8, opp);
    }
}

#[inline(always)]
pub(super) fn update_piece<const ADD: bool>(pos: &mut Position, sq: Square, pt: PieceType, color: Color) {
    if ADD {
        pos.role_bb[pt].set_bit(sq);
        pos.side_bb[color].set_bit(sq);
        pos.occ.set_bit(sq);
        pos.pieces[sq.0 as usize] = pt;
    } else {
        pos.role_bb[pt].clear_bit(sq);
        pos.side_bb[color].clear_bit(sq);
        pos.occ.clear_bit(sq);
        pos.pieces[sq.0 as usize] = PieceType::None;
    }
}

#[inline(always)]
fn toggle_piece_keys(pos: &mut Position, pt: PieceType, color: Color, sq: Square) {
    let key = zobrist::key_piece(pt, color, sq);
    pos.hash ^= key;

    match pt {
        PieceType::Pawn => pos.pawn_key ^= key,
        PieceType::Knight | PieceType::Bishop => pos.minor_key ^= key,
        PieceType::Rook | PieceType::Queen => pos.major_key ^= key,
        PieceType::King | PieceType::None => {},
    }
}

/// Both destinations are cleared before either origin is restored because Chess960
/// castling squares can overlap.
#[inline]
fn revert_castling(pos: &mut Position, king_from: Square, rook_from: Square) {
    let (king_to, rook_to) = castling_targets(king_from, rook_from);
    let stm = pos.stm;

    update_piece::<false>(pos, king_to, PieceType::King, stm);
    update_piece::<false>(pos, rook_to, PieceType::Rook, stm);
    update_piece::<true>(pos, king_from, PieceType::King, stm);
    update_piece::<true>(pos, rook_from, PieceType::Rook, stm);
}

/// King and rook land on g/f for O-O and c/d for O-O-O, regardless of their starting files.
#[inline(always)]
pub(crate) fn castling_targets(king_sq: Square, rook_sq: Square) -> (Square, Square) {
    let queenside = rook_sq.file() < king_sq.file();
    let rank = king_sq.rank();
    (
        Square::from_coords(if queenside { 2 } else { 6 }, rank),
        Square::from_coords(if queenside { 3 } else { 5 }, rank),
    )
}

/// Both pieces are lifted before either is placed, because in Chess960 either destination
/// can be the other piece's origin.
#[inline]
fn apply_castling(pos: &mut Position, king_from: Square, rook_from: Square, stm: Color) {
    let (king_to, rook_to) = castling_targets(king_from, rook_from);

    pos.remove_piece(king_from, PieceType::King, stm);
    toggle_piece_keys(pos, PieceType::King, stm, king_from);
    pos.remove_piece(rook_from, PieceType::Rook, stm);
    toggle_piece_keys(pos, PieceType::Rook, stm, rook_from);

    pos.add_piece(king_to, PieceType::King, stm);
    toggle_piece_keys(pos, PieceType::King, stm, king_to);
    pos.add_piece(rook_to, PieceType::Rook, stm);
    toggle_piece_keys(pos, PieceType::Rook, stm, rook_to);
}

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

/// Recorded only when an enemy pawn can capture, so the same position never gets two hashes
/// for an en passant square nothing can use.
#[inline]
fn refresh_en_passant(pos: &mut Position, mv: Move, from: Square, to: Square, old_ep: Option<Square>) {
    if let Some(sq) = old_ep {
        pos.hash ^= zobrist::key_ep(sq);
    }

    pos.en_passant = None;

    if mv.is_double_push() {
        let ep_sq = Square(u8::midpoint(from.0, to.0));
        if pos.has_ep_attacker(ep_sq) {
            pos.en_passant = Some(ep_sq);
            pos.hash ^= zobrist::key_ep(ep_sq);
        }
    }
}
