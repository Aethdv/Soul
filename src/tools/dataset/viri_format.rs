//! Viriformat parser.
//!
//! Each file consists of concatenated games. A game is:
//! - 32-byte PackedBoard header (position + result)
//! - Zero or more (Move, Score) pairs — 4 bytes each (2+2)
//! - Four-byte zero sentinel
//!
//! Scores in the (Move, Score) pairs are white-relative, converted to
//! STM-relative on the way into SoulEntry. The header's score field is a
//! stale placeholder — real evals live in the move pairs.

use std::io::{self, Read};

use super::SoulEntry;
use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
        defs::{Color, PieceType, Square},
        moves::Move,
    },
    weave::Vi16x8,
};

const PACKED_BOARD_SIZE: usize = 32;
const SENTINEL: [u8; 4] = [0, 0, 0, 0];

pub fn parse_viri_file(path: &str) -> io::Result<Vec<SoulEntry>> {
    let mut file = std::fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut entries = Vec::new();
    let mut pos = 0usize;

    while pos + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[pos..pos + PACKED_BOARD_SIZE];
        pos += PACKED_BOARD_SIZE;

        let Some((mut position, game_result)) = parse_packed_board(header) else {
            break;
        };

        loop {
            if pos + 4 > data.len() {
                break;
            }
            let candidate = &data[pos..pos + 4];
            if candidate == SENTINEL {
                pos += 4;
                break;
            }

            let viri_move = u16::from_le_bytes([candidate[0], candidate[1]]);
            let viri_score = i16::from_le_bytes([candidate[2], candidate[3]]);
            pos += 4;

            let Some(soul_move) = viri_to_soul_move(viri_move, &position) else {
                break;
            };

            entries.push(SoulEntry::from_board(
                &position,
                f64::from(stm_result(game_result, position.stm)) / 2.0,
                None,
                Some(relative_score(viri_score, position.stm)),
            ));

            let mut acc = Vi16x8::zero();
            position.make_move(soul_move, &mut acc);
        }
    }

    Ok(entries)
}

/// Convert a white-relative viri score to STM-relative `i32`.
fn relative_score(viri_score: i16, stm: Color) -> i32 {
    let s = i32::from(viri_score);
    if stm == Color::Black { -s } else { s }
}

/// Convert a white-relative viri result (0=black win, 1=draw, 2=white win)
/// to an STM-relative result (0=loss, 1=draw, 2=win).
fn stm_result(viri_result: u8, stm: Color) -> u8 {
    if stm == Color::Black { 2 - viri_result } else { viri_result }
}

fn parse_packed_board(data: &[u8]) -> Option<(Position, u8)> {
    if data.len() < PACKED_BOARD_SIZE {
        return None;
    }

    let occupancy = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let pieces = &data[8..24]; // [u4; 32] packed as [u8; 16]
    let stm_ep = data[24];
    let _halfmove = data[25];
    let _fullmove = u16::from_le_bytes([data[26], data[27]]);
    let _score = i16::from_le_bytes([data[28], data[29]]);
    let result = data[30];
    let _extra = data[31];

    if result > 2 {
        return None;
    }

    let stm = if stm_ep & 0x80 != 0 { Color::Black } else { Color::White };
    let ep = stm_ep & 0x7F;
    let en_passant = if ep < 64 { Some(Square(ep)) } else { None };

    let mut pos = Position::new();
    pos.stm = stm;
    pos.en_passant = en_passant;

    // Set castling-rook home squares for standard chess (FRC handled later).
    pos.castling_rooks[ROOK_W_KS] = Square(7); // h1
    pos.castling_rooks[ROOK_W_QS] = Square(0); // a1
    pos.castling_rooks[ROOK_B_KS] = Square(63); // h8
    pos.castling_rooks[ROOK_B_QS] = Square(56); // a8

    let mut white_king = None;
    let mut black_king = None;
    let mut unmoved_rooks: [Vec<u8>; 2] = [Vec::new(), Vec::new()];

    let mut occ = occupancy;
    let mut piece_idx: usize = 0;

    while occ != 0 {
        let sq_idx = occ.trailing_zeros() as usize;
        occ &= occ - 1;

        if piece_idx >= 32 {
            break;
        }
        let nibble = if piece_idx.is_multiple_of(2) { pieces[piece_idx / 2] & 0x0F } else { pieces[piece_idx / 2] >> 4 };
        piece_idx += 1;

        let viri_type = nibble & 0x07;
        let color = if nibble & 0x08 != 0 { Color::Black } else { Color::White };

        let pt = match viri_type {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            6 => PieceType::Rook,
            _ => continue,
        };

        let sq = Square(sq_idx as u8);
        pos.add_piece(sq, pt, color);

        if pt == PieceType::King {
            match color {
                Color::White => white_king = Some(sq),
                Color::Black => black_king = Some(sq),
            }
        }

        if viri_type == 6 {
            // Unmoved rook — record for castling-rights detection.
            unmoved_rooks[color as usize].push(sq_idx as u8);
        }
    }

    // Reconstruct castling rights from unmoved rooks relative to kings.
    let mut set_castling_rights = 0u8;
    if let Some(king_sq) = white_king {
        for &rook_idx in &unmoved_rooks[Color::White as usize] {
            let rook = Square(rook_idx);
            if rook.file() > king_sq.file() {
                set_castling_rights |= WHITE_OO;
                pos.castling_rooks[ROOK_W_KS] = rook;
            } else {
                set_castling_rights |= WHITE_OOO;
                pos.castling_rooks[ROOK_W_QS] = rook;
            }
        }
    }
    if let Some(king_sq) = black_king {
        for &rook_idx in &unmoved_rooks[Color::Black as usize] {
            let rook = Square(rook_idx);
            if rook.file() > king_sq.file() {
                set_castling_rights |= BLACK_OO;
                pos.castling_rooks[ROOK_B_KS] = rook;
            } else {
                set_castling_rights |= BLACK_OOO;
                pos.castling_rooks[ROOK_B_QS] = rook;
            }
        }
    }
    pos.castling_rights = set_castling_rights;

    // Mark as FRC if any castling rook is off its standard home square.
    pos.is_frc = pos.castling_rooks[ROOK_W_KS] != Square(7)
        || pos.castling_rooks[ROOK_W_QS] != Square(0)
        || pos.castling_rooks[ROOK_B_KS] != Square(63)
        || pos.castling_rooks[ROOK_B_QS] != Square(56);

    pos.hash = pos.calc_zobrist();

    Some((pos, result))
}

/// Convert a viriformat 16-bit move to a Soul `Move`.
///
/// Viri layout:
/// - bits  0..5  → from square (6 bits)
/// - bits  6..11 → to square (6 bits)
/// - bits 12..13 → promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
/// - bits 14..15 → move type (0=normal, 1=en-passant, 2=castling, 3=promotion)
///
/// Castling is king-takes-rook (same as Soul's internal encoding).
fn viri_to_soul_move(viri_move: u16, pos: &Position) -> Option<Move> {
    let from = Square((viri_move & 0x3F) as u8);
    let to = Square(((viri_move >> 6) & 0x3F) as u8);
    let promo_piece = (viri_move >> 12) & 0x3;
    let move_type = (viri_move >> 14) & 0x3;

    if from.0 >= 64 || to.0 >= 64 {
        return None;
    }

    let moving_piece = pos.piece_at(from);
    if moving_piece == PieceType::None {
        return None;
    }
    let capture = pos.piece_at(to) != PieceType::None;

    let flag: u16 = match move_type {
        0 => {
            if capture {
                Move::CAPTURE
            } else if moving_piece == PieceType::Pawn && from.file() == to.file() && from.rank().abs_diff(to.rank()) == 2 {
                Move::DOUBLE_PUSH
            } else {
                Move::QUIET
            }
        },
        1 => Move::EP_CAPTURE,
        2 => Move::CASTLE,
        3 => {
            // Promotion flag values embed the capture bit in bit 0.
            match promo_piece {
                0 => Move::PROM_N | if capture { 1 } else { 0 },
                1 => Move::PROM_B | if capture { 1 } else { 0 },
                2 => Move::PROM_R | if capture { 1 } else { 0 },
                3 => Move::PROM_Q | if capture { 1 } else { 0 },
                _ => return None,
            }
        },
        _ => return None,
    };

    Some(Move::new(from, to, flag))
}
