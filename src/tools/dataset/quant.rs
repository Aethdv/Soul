//! Position quantization and dataset serialization.
//!
//! Encodes board positions into 32-byte [`SoulEntry`] records using an occupancy
//! bitboard coupled with a 4-bit nibble array. Nibbles past the piece count are zero.
//!
//! Nibble layout, bits 0..=3:
//! - Bits 0..=2: Piece type (`0=Pawn, 1=Knight, 2=Bishop, 3=Rook, 4=Queen, 5=King, 6=Castling Rook`)
//! - Bit 3: Color (`0=White, 1=Black`)

use crate::{
    core::{
        board::Position,
        defs::{Bitboard, Color, PieceType, Square},
    },
    tools::dataset::SoulEntry,
};

/// FEN character mapping indexed by `[color_index][piece_type]`.
const PIECE_CHARS: [[u8; 6]; 2] = [*b"PNBRQK", *b"pnbrqk"];

/// Sentinel piece type for an unmoved rook retaining castling rights.
const CASTLING_ROOK: usize = 6;

/// Packs board occupancy and pieces into an occupancy bitmask and a 16-byte nibble array.
///
/// The i-th nibble (traversed LSB to MSB) corresponds to the i-th set bit in
/// `occupancy`. Rooks with active castling rights are assigned type 6 instead of 3,
/// allowing complete castling state recovery from piece placement alone.
pub fn pack_pieces(board: &Position) -> (u64, [u8; 16]) {
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

/// Each occupied square with its nibble, in occupancy order.
///
/// The array holds 32 nibbles, so a caller reading untrusted bytes checks the
/// occupancy's popcount before it gets here.
pub(super) fn packed_pieces(occupancy: u64, pieces: &[u8; 16]) -> impl Iterator<Item = (Square, u8)> + '_ {
    Bitboard(occupancy).into_iter().enumerate().map(move |(i, sq)| {
        let byte = pieces[i / 2];
        (sq, if i & 1 == 0 { byte & 0x0F } else { byte >> 4 })
    })
}

/// Encodes a position, outcome, and optional search evaluation into a `SoulEntry`.
///
/// The board goes in White-absolute while `result` and `search_score` arrive already
/// flipped, so the two halves of an entry sit in different perspectives. The tuner
/// flips the board when it extracts features.
pub fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> SoulEntry {
    let (occupancy, pieces) = pack_pieces(board);

    SoulEntry {
        occupancy,
        pieces,
        // One short of i16::MAX, so a clamped score cannot land on the NO_SCORE sentinel.
        score: search_score.map_or(SoulEntry::NO_SCORE, |s| s.clamp(i16::MIN as i32, i16::MAX as i32 - 1) as i16),
        result: (result * 2.0) as u8,
        stm_and_ep: (u8::from(board.stm == Color::Black) << 7) | (board.en_passant.map_or(64, |sq| sq.0) & 0x7F),
        castling: board.castling_rights,
        _pad: [0u8; 3],
    }
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

/// Reconstructs an absolute FEN string from a `SoulEntry`.
///
/// Supports both standard and Chess960 (Shredder-FEN) castling notation.
pub fn to_fen(entry: &SoulEntry) -> String {
    let mut board_chars = [b'.'; 64];
    let mut castling_rooks = [Square(0); 4];
    let mut castling_rights = 0u8;
    let mut king_sqs = [Square(0); 2];
    let mut pending_rooks = [(0usize, Square(0)); 4];
    let mut pending_count = 0usize;

    for (sq, nibble) in packed_pieces(entry.occupancy, &entry.pieces) {
        let raw_type = (nibble & 0x07) as usize;
        let color_idx = if (nibble & 0x08) != 0 { 1 } else { 0 };
        let piece_type = if raw_type == CASTLING_ROOK { 3 } else { raw_type };

        board_chars[usize::from(sq)] = PIECE_CHARS[color_idx][piece_type];

        if raw_type == PieceType::King.as_usize() {
            king_sqs[color_idx] = sq;
        }
        if raw_type == CASTLING_ROOK {
            pending_rooks[pending_count] = (color_idx, sq);
            pending_count += 1;
        }
    }

    for &(color_idx, sq) in &pending_rooks[..pending_count] {
        let color = if color_idx == 0 { Color::White } else { Color::Black };
        let (mask, slot) = Position::castling_side(sq, king_sqs[color_idx], color);

        castling_rooks[slot] = sq;
        castling_rights |= mask;
    }

    let mut fen = String::with_capacity(80);

    for rank in (0..8usize).rev() {
        let mut empty = 0u8;

        for file in 0..8usize {
            let ch = board_chars[rank * 8 + file];
            if ch == b'.' {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(ch as char);
            }
        }

        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
        if rank > 0 {
            fen.push('/');
        }
    }

    fen.push(' ');
    fen.push(if (entry.stm_and_ep & 0x80) == 0 { 'w' } else { 'b' });
    fen.push(' ');

    if castling_rights == 0 {
        fen.push('-');
    } else {
        let is_standard_layout = castling_rooks[0] == Square::from_coords(7, 0)
            && castling_rooks[1] == Square::from_coords(0, 0)
            && castling_rooks[2] == Square::from_coords(7, 7)
            && castling_rooks[3] == Square::from_coords(0, 7);

        if is_standard_layout {
            if castling_rights & 1 != 0 {
                fen.push('K');
            }
            if castling_rights & 2 != 0 {
                fen.push('Q');
            }
            if castling_rights & 4 != 0 {
                fen.push('k');
            }
            if castling_rights & 8 != 0 {
                fen.push('q');
            }
        } else {
            for (slot, &rook) in castling_rooks.iter().enumerate() {
                if castling_rights & (1u8 << slot) != 0 {
                    let base = if slot < 2 { b'A' } else { b'a' };
                    fen.push((base + rook.file()) as char);
                }
            }
        }
    }

    fen.push(' ');

    let ep = entry.stm_and_ep & 0x7F;
    if ep >= 64 {
        fen.push('-');
    } else {
        fen.push_str(&Square(ep).to_string());
    }

    fen.push_str(" 0 1");
    fen
}

/// Sums weighted material straight from the packed nibbles, on the `17..=78` scale
/// [`wdl_model`] was fitted to. Kings weigh nothing.
pub fn material_count(entry: &SoulEntry) -> u32 {
    // Indexed by piece type, so slot 6 repeats the rook's weight for a castling rook.
    const WEIGHTS: [u32; 7] = [1, 3, 3, 5, 9, 0, 5];

    packed_pieces(entry.occupancy, &entry.pieces)
        .map(|(_, nibble)| WEIGHTS[usize::from(nibble & 0x07)])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::STARTPOS;

    /// A zero would read as a dead-even eval, and the volatility filter measures
    /// every static eval against this field, so an unsearched dataset would come
    /// out cut down to its quiet positions.
    #[test]
    fn unsearched_eval_encodes_as_sentinel() {
        let board = Position::from_fen(STARTPOS);
        assert_eq!(from_board(&board, 0.5, None).score, SoulEntry::NO_SCORE);
        assert_eq!(from_board(&board, 0.5, Some(0)).score, 0);
        assert_eq!(from_board(&board, 0.5, Some(-42)).score, -42);
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
        assert_eq!(Position::from_fen(STARTPOS).material_count(), 78);
    }
}
