//! Packing and unpacking for [`SoulEntry`]'s board fields.

use crate::{
    core::{
        board::Position,
        defs::{Bitboard, Color, PieceType, Square},
    },
    tools::dataset::SoulEntry,
};

const CASTLING_ROOK: usize = 6;

pub(super) fn pack_pieces(board: &Position) -> (u64, [u8; 16]) {
    assert!(board.occ.popcount() <= 32, "a packed board holds at most 32 pieces");

    let mut pieces = [0u8; 16];

    for (piece_idx, sq) in board.occ.into_iter().enumerate() {
        let color = board.color_at(sq);
        let raw_type = board.piece_at(sq).as_usize() & 0x07;
        let piece_type = if raw_type == PieceType::Rook.as_usize() && is_castling_rook(board, sq, color) {
            CASTLING_ROOK
        } else {
            raw_type
        };

        let color_bit = if color == Color::Black { 0x08 } else { 0x00 };
        pieces[piece_idx / 2] |= ((piece_type | color_bit) as u8) << ((piece_idx & 1) * 4);
    }

    (board.occ.0, pieces)
}

pub(super) fn unpack_pieces(occupancy: u64, pieces: &[u8; 16]) -> impl Iterator<Item = (Square, u8)> + '_ {
    assert!(occupancy.count_ones() <= 32, "a packed board holds at most 32 pieces");

    Bitboard(occupancy).into_iter().enumerate().map(move |(i, sq)| {
        let byte = pieces[i / 2];
        (sq, if i & 1 == 0 { byte & 0x0F } else { byte >> 4 })
    })
}

pub(super) fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> SoulEntry {
    assert!(result.is_finite() && (0.0..=1.0).contains(&result), "result must be a finite WDL fraction");

    let (occupancy, pieces) = pack_pieces(board);

    SoulEntry {
        occupancy,
        pieces,
        // One short of i16::MAX, so a clamped score cannot land on the NO_SCORE sentinel.
        score: search_score.map_or(SoulEntry::NO_SCORE, |s| s.clamp(i16::MIN as i32, i16::MAX as i32 - 1) as i16),
        result: (result * 2.0) as u8,
        stm_and_ep: (u8::from(board.stm == Color::Black) << 7) | (board.en_passant.map_or(64, |sq| sq.0) & 0x7F),
        castling_rights: board.castling_rights,
        _pad: [0u8; 3],
    }
}

pub(super) fn unpack_board(
    occupancy: u64,
    pieces: &[u8; 16],
    stm_and_ep: u8,
    halfmove: u8,
    fullmove: u16,
    expected_castling_rights: Option<u8>,
) -> Option<Position> {
    if occupancy.count_ones() > 32 {
        return None;
    }

    let stm = if stm_and_ep & 0x80 != 0 { Color::Black } else { Color::White };
    let ep = stm_and_ep & 0x7F;
    let en_passant = (ep < 64).then_some(Square(ep));
    if let Some(sq) = en_passant {
        let expected_rank = if stm == Color::White { 5 } else { 2 };
        if sq.rank() != expected_rank {
            return None;
        }
    }

    let mut pos = Position::empty();
    pos.stm = stm;
    pos.en_passant = en_passant;

    let mut kings = [None; 2];
    let mut unmoved_rooks = [(Color::White, Square(0)); 4];
    let mut rook_count = 0usize;

    for (sq, nibble) in unpack_pieces(occupancy, pieces) {
        let raw_type = nibble & 0x07;
        let color = if nibble & 0x08 != 0 { Color::Black } else { Color::White };
        let piece_type = match raw_type {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            6 => PieceType::Rook,
            _ => return None,
        };

        if piece_type == PieceType::Pawn && matches!(sq.rank(), 0 | 7) {
            return None;
        }

        pos.add_piece(sq, piece_type, color);
        if piece_type == PieceType::King {
            kings[color as usize] = Some(sq);
        }
        if raw_type == CASTLING_ROOK as u8 {
            if rook_count == unmoved_rooks.len() {
                return None;
            }
            unmoved_rooks[rook_count] = (color, sq);
            rook_count += 1;
        }
    }

    if pos.pieces(PieceType::King, Color::White).popcount() != 1 || pos.pieces(PieceType::King, Color::Black).popcount() != 1 {
        return None;
    }

    let their_king = (pos.role_bb[PieceType::King] & pos.side_bb[stm.opposite()]).lsb();
    if pos.is_attacked(their_king, stm, Bitboard::EMPTY) {
        return None;
    }

    let mut castling_rights = 0u8;
    for &(color, rook) in &unmoved_rooks[..rook_count] {
        let king = kings[color as usize]?;
        let slot = Position::castling_rook_slot(rook, king, color);
        let bit = 1u8 << slot;

        if castling_rights & bit != 0 {
            return None;
        }
        castling_rights |= bit;
        pos.castling_rooks[slot] = rook;
    }

    if expected_castling_rights.is_some_and(|rights| rights != castling_rights) {
        return None;
    }
    pos.castling_rights = castling_rights;
    pos.frc_notation = pos.has_nonstandard_castling_rook();
    pos.halfmove_clock = halfmove;
    pos.fullmove_number = fullmove.max(1);
    pos.recompute_hashes();
    Some(pos)
}

fn is_castling_rook(board: &Position, sq: Square, color: Color) -> bool {
    for slot in 0..4 {
        let bit = 1u8 << slot;
        let is_white_slot = slot < 2;
        if board.castling_rights & bit != 0 && board.castling_rooks[slot] == sq && is_white_slot == (color == Color::White) {
            return true;
        }
    }
    false
}

pub(crate) fn material_count(entry: &SoulEntry) -> u32 {
    const MATERIAL_WEIGHTS: [u32; 7] = [1, 3, 3, 5, 9, 0, 5];

    unpack_pieces(entry.occupancy, &entry.pieces)
        .map(|(_, nibble)| MATERIAL_WEIGHTS[usize::from(nibble & 0x07)])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::STARTPOS;

    #[test]
    fn unsearched_eval_encodes_as_sentinel() {
        let board = Position::from_fen(STARTPOS);
        assert_eq!(from_board(&board, 0.5, None).score, SoulEntry::NO_SCORE);
        assert_eq!(from_board(&board, 0.5, Some(0)).score, 0);
        assert_eq!(from_board(&board, 0.5, Some(-42)).score, -42);
        assert_eq!(from_board(&board, 0.5, Some(i32::MAX)).score, i16::MAX - 1);
        assert_eq!(from_board(&board, 0.5, Some(i32::MIN)).score, i16::MIN);
    }

    #[test]
    fn nibble_material_count_matches_board_evaluation() {
        for fen in [
            STARTPOS,
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
            "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1",
            "8/2k5/8/4q3/8/3N4/5PPP/6K1 b - - 0 1",
            "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        ] {
            let board = Position::from_fen(fen);
            assert_eq!(from_board(&board, 0.5, None).material_count(), board.material_count(), "{fen}");
        }
    }
}
