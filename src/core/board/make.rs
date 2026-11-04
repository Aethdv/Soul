//! Move application and incremental state updates.

use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, StateInfo, WHITE_OO, WHITE_OOO},
        defs::{Color, Piece, PieceType, Square},
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

    if mv.is_castling() {
        apply_castling(pos, from, to, stm);
    } else if mv.is_en_passant() {
        // The captured pawn stands on to's file one rank back: flipping bit 3 steps
        // rank 6 to 5 for White and rank 3 to 4 for Black.
        let victim_sq = to ^ 8;
        let us_pawn = Piece::new(PieceType::Pawn, stm);
        let them_pawn = Piece::new(PieceType::Pawn, opp);
        let ep_delta =
            zobrist::key_piece(us_pawn, from) ^ zobrist::key_piece(us_pawn, to) ^ zobrist::key_piece(them_pawn, victim_sq);
        pos.hash ^= ep_delta;
        pos.pawn_key ^= ep_delta;

        pos.remove_piece(victim_sq, PieceType::Pawn, opp);
        pos.remove_piece(from, PieceType::Pawn, stm);
        pos.add_piece(to, PieceType::Pawn, stm);
    } else {
        if captured != PieceType::None {
            let victim = Piece::new(captured, opp);
            let victim_key = zobrist::key_piece(victim, to);
            pos.hash ^= victim_key;
            xor_corr_key(pos, victim, victim_key);
            pos.remove_piece(to, captured, opp);
        }

        pos.remove_piece(from, pt, stm);
        pos.add_piece(to, placed, stm);

        let moving_piece = Piece::new(pt, stm);
        let from_key = zobrist::key_piece(moving_piece, from);
        if !mv.is_promotion() {
            let delta_key = from_key ^ zobrist::key_piece(moving_piece, to);
            pos.hash ^= delta_key;
            xor_corr_key(pos, moving_piece, delta_key);
        } else {
            let promo_piece = Piece::new(placed, stm);
            let to_key = zobrist::key_piece(promo_piece, to);
            pos.hash ^= from_key ^ to_key;
            pos.pawn_key ^= from_key;
            if (mv.inner() & 0x8000) != 0 {
                pos.major_key ^= to_key;
            } else {
                pos.minor_key ^= to_key;
            }
        }
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
        *acc -= psqt::entry(Piece::new(PieceType::King, stm), from);
        *acc += psqt::entry(Piece::new(PieceType::King, stm), king_to);
        *acc -= psqt::entry(Piece::new(PieceType::Rook, stm), to);
        *acc += psqt::entry(Piece::new(PieceType::Rook, stm), rook_to);
        return;
    }

    let opp = stm.opposite();

    *acc -= psqt::entry(Piece::new(pt, stm), from);
    *acc += psqt::entry(Piece::new(placed, stm), to);

    if captured != PieceType::None {
        *acc -= psqt::entry(Piece::new(captured, opp), to);
    } else if mv.is_en_passant() {
        *acc -= psqt::entry(Piece::new(PieceType::Pawn, opp), to ^ 8);
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
fn xor_corr_key(pos: &mut Position, piece: Piece, delta_key: u64) {
    match piece.corr_group() {
        0 => pos.pawn_key ^= delta_key,
        1 => pos.minor_key ^= delta_key,
        2 => pos.major_key ^= delta_key,
        _ => {},
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

    let king = Piece::new(PieceType::King, stm);
    let rook = Piece::new(PieceType::Rook, stm);
    let king_delta = zobrist::key_piece(king, king_from) ^ zobrist::key_piece(king, king_to);
    let rook_delta = zobrist::key_piece(rook, rook_from) ^ zobrist::key_piece(rook, rook_to);
    pos.hash ^= king_delta ^ rook_delta;
    pos.major_key ^= rook_delta;

    pos.remove_piece(king_from, PieceType::King, stm);
    pos.remove_piece(rook_from, PieceType::Rook, stm);
    pos.add_piece(king_to, PieceType::King, stm);
    pos.add_piece(rook_to, PieceType::Rook, stm);
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
