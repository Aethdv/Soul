//! Position quantization and dataset encoding.
//!
//! Packs board state into a 32-byte [`SoulEntry`] using occupancy-bitboard +
//! nibble-array encoding. The i-th nibble (LSB to MSB) in `pieces` describes
//! the piece on the i-th set bit of `occupancy`.
//!
//! Nibble layout: bits 0-2 = type (pawn=0..king=5, 6=castling rook), bit 3 = color (0=W,1=B).
//! Unused nibbles are zero.

use crate::{
    core::{
        board::Position,
        defs::{Color, PieceType, Square},
    },
    tools::dataset::SoulEntry,
};

/// Encode a board position into a [`SoulEntry`].
///
/// Board state is stored in raw (non-normalized) form;
/// occupancy bitboard + nibble-array of piece types/colors.
/// The tuner normalizes the perspective during feature extraction.
pub fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> SoulEntry {
    let mut pieces = [0u8; 16];
    let mut idx = 0usize;
    let mut occ = board.occ.0;

    while occ != 0 {
        let lsb_idx = occ.trailing_zeros() as u8;
        occ &= occ - 1;

        let sq = Square(lsb_idx);
        let piece = board.piece_at(sq);
        let color = board.color_at(sq);
        let pt_raw = piece.as_usize() & 0x07;

        let pt = if pt_raw == PieceType::Rook.as_usize() && is_castling_rook(board, sq, color) {
            CASTLING_ROOK
        } else {
            pt_raw
        };

        let color_bit = if color == Color::Black { 0x08 } else { 0x00 };
        let nibble = (pt | color_bit) as u8;

        pieces[idx / 2] |= nibble << ((idx & 1) * 4);
        idx += 1;
    }

    SoulEntry {
        occupancy: board.occ.0,
        pieces,
        score: search_score.map_or(SoulEntry::NO_SCORE, |s| s.clamp(i16::MIN as i32, i16::MAX as i32) as i16),
        result: (result * 2.0) as u8,
        stm_and_ep: (u8::from(board.stm == Color::Black) << 7) | (board.en_passant.map_or(64, |sq| sq.0) & 0x7F),
        castling: board.castling_rights,
        _pad: [0u8; 3],
    }
}

fn is_castling_rook(board: &Position, sq: Square, color: Color) -> bool {
    for slot in 0..4 {
        let bit = 1u8 << slot;

        if board.castling_rights & bit != 0 && board.castling_rooks[slot] == sq && (slot < 2) == (color == Color::White) {
            return true;
        }
    }

    false
}

/// FEN characters indexed by `[color_index][piece_type]`.
const PIECE_CHARS: [[u8; 6]; 2] = [*b"PNBRQK", *b"pnbrqk"];
const CASTLING_ROOK: usize = 6;

/// Reconstruct a FEN string from a [`SoulEntry`].
pub fn to_fen(entry: &SoulEntry) -> String {
    let mut board = [b'.'; 64];
    let mut castling_rooks = [Square(0); 4];
    let mut castling_bits = 0u8;
    let mut king_sq = [Square(0); 2];
    let mut pending_rooks = [(0usize, Square(0)); 4];
    let mut pending_count = 0usize;
    let mut occ = entry.occupancy;
    let mut idx = 0usize;

    while occ != 0 {
        let sq = Square(occ.trailing_zeros() as u8);
        occ &= occ - 1; // clear lowest set bit

        let nibble = next_nibble(&entry.pieces, &mut idx);
        let pt_raw = (nibble & 0x07) as usize;
        let color_idx = if (nibble & 0x08) != 0 { 1 } else { 0 };
        let pt = if pt_raw == CASTLING_ROOK { 3 } else { pt_raw };

        board[usize::from(sq)] = PIECE_CHARS[color_idx][pt];

        if pt_raw == 5 {
            king_sq[color_idx] = sq;
        }

        if pt_raw == CASTLING_ROOK {
            pending_rooks[pending_count] = (color_idx, sq);
            pending_count += 1;
        }
    }

    for &(color_idx, sq) in &pending_rooks[..pending_count] {
        let king_file = u8::from(king_sq[color_idx]) % 8;
        let rook_file = u8::from(sq) % 8;
        let is_kingside = rook_file > king_file;

        let slot = match (color_idx, is_kingside) {
            (0, true) => 0,
            (0, false) => 1,
            (_, true) => 2,
            (_, false) => 3,
        };

        castling_rooks[slot] = sq;
        castling_bits |= 1u8 << slot;
    }

    let mut fen = String::with_capacity(80);

    for rank in (0..8usize).rev() {
        let mut empty = 0u8;

        for file in 0..8usize {
            let ch = board[rank * 8 + file];

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

    if castling_bits == 0 {
        fen.push('-');
    } else {
        let standard = castling_rooks[0] == Square::from_coords(7, 0)
            && castling_rooks[1] == Square::from_coords(0, 0)
            && castling_rooks[2] == Square::from_coords(7, 7)
            && castling_rooks[3] == Square::from_coords(0, 7);

        if standard {
            if castling_bits & 1 != 0 {
                fen.push('K');
            }
            if castling_bits & 2 != 0 {
                fen.push('Q');
            }
            if castling_bits & 4 != 0 {
                fen.push('k');
            }
            if castling_bits & 8 != 0 {
                fen.push('q');
            }
        } else {
            for (slot, &rook) in castling_rooks.iter().enumerate() {
                if castling_bits & (1u8 << slot) != 0 {
                    let file = u8::from(rook) % 8;
                    let base = if slot < 2 { b'A' } else { b'a' };

                    fen.push((base + file) as char);
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

/// Weighted material over both sides, the scale `wdl_model` clamps into 17..=78.
///
/// The king weighs nothing and the castling-rook encoding weighs a rook, the two
/// cases [`to_fen`] also has to unpack.
pub fn material_count(entry: &SoulEntry) -> u32 {
    const WEIGHTS: [u32; 7] = [1, 3, 3, 5, 9, 0, 5];

    let mut total = 0;
    let mut idx = 0usize;
    let mut occ = entry.occupancy;

    while occ != 0 {
        occ &= occ - 1;
        total += WEIGHTS[usize::from(next_nibble(&entry.pieces, &mut idx) & 0x07)];
    }

    total
}

#[inline]
pub(super) fn next_nibble(pieces: &[u8; 16], idx: &mut usize) -> u8 {
    let i = *idx;
    *idx += 1;

    let byte = pieces[i / 2];
    if i & 1 == 0 { byte & 0x0F } else { byte >> 4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::STARTPOS;

    /// The volatility filter measures every static eval against this field, so a
    /// zero here quietly cuts an unsearched dataset down to its quiet positions.
    #[test]
    fn an_absent_search_score_encodes_as_the_sentinel() {
        let board = Position::from_fen(STARTPOS);

        assert_eq!(from_board(&board, 0.5, None).score, SoulEntry::NO_SCORE);
        assert_eq!(from_board(&board, 0.5, Some(0)).score, 0);
        assert_eq!(from_board(&board, 0.5, Some(-42)).score, -42);
    }

    /// Counting from the nibbles has to land where counting from the board does,
    /// including the castling rooks the packing stores under their own code.
    #[test]
    fn material_counted_from_the_nibbles_matches_the_board() {
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

        assert_eq!(Position::from_fen(STARTPOS).material_count(), 78, "the scale wdl_model was fitted on");
    }
}
