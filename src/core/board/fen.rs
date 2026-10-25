//! FEN (Forsyth-Edwards Notation) parsing and serialization.

use std::{
    fmt::{self, Display, Formatter, Write as _},
    iter::Peekable,
};

use crate::core::{
    board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
    defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
    error::FenError,
    primitives::{RANK_1, RANK_8},
};

const KINGSIDE_FILE: u8 = 7;
const QUEENSIDE_FILE: u8 = 0;
const SEARCH_LEFT: i8 = -1;
const SEARCH_RIGHT: i8 = 1;

/// `(right, rook slot, standard char, Shredder-FEN file base)`.
const CASTLING_FEN: [(u8, usize, char, u8); 4] = [
    (WHITE_OO, ROOK_W_KS, 'K', b'A'),
    (WHITE_OOO, ROOK_W_QS, 'Q', b'A'),
    (BLACK_OO, ROOK_B_KS, 'k', b'a'),
    (BLACK_OOO, ROOK_B_QS, 'q', b'a'),
];

pub(super) fn try_from_tokens<'a, I>(tokens: &mut Peekable<I>) -> Result<Position, FenError>
where I: Iterator<Item = &'a str> {
    let mut pos = Position::empty();

    parse_placement(&mut pos, tokens.next().ok_or(FenError::Empty)?)?;

    pos.stm = match tokens.next().ok_or(FenError::MissingStm)? {
        "w" => Color::White,
        "b" => Color::Black,
        s => return Err(FenError::InvalidStm { stm: s.to_string() }),
    };

    if let Some(castling) = tokens.next()
        && castling != "-"
    {
        parse_castling_rights(&mut pos, castling)?;
    }

    if let Some(ep) = tokens.next()
        && ep != "-"
    {
        parse_en_passant(&mut pos, ep)?;
    }

    if let Some(token) = tokens.next_if(|&t| t != "moves") {
        pos.halfmove_clock = token.parse().unwrap_or(0);
    }
    if let Some(token) = tokens.next_if(|&t| t != "moves") {
        pos.fullmove_number = token.parse().unwrap_or(1);
    }
    finish_position(pos)
}

/// FRC uses Shredder-FEN when castling rooks are off their standard squares.
impl Display for Position {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pos = self;
        for rank in (0..8u8).rev() {
            if rank < 7 {
                f.write_char('/')?;
            }

            let mut empty = 0u8;

            for file in 0..8u8 {
                let sq = Square::from_coords(file, rank);
                let pt = pos.piece_at(sq);

                if pt == PieceType::None {
                    empty += 1;
                } else {
                    if empty > 0 {
                        f.write_char((b'0' + empty) as char)?;
                        empty = 0;
                    }
                    f.write_char(pt.to_char(pos.color_at(sq)))?;
                }
            }

            if empty > 0 {
                f.write_char((b'0' + empty) as char)?;
            }
        }

        if pos.stm == Color::White {
            f.write_str(" w ")?;
        } else {
            f.write_str(" b ")?;
        };

        if pos.castling_rights == 0 {
            f.write_char('-')?;
        } else {
            let shredder = pos.frc_notation && pos.has_nonstandard_castling_rook();
            for &(bit, slot, std_ch, frc_base) in &CASTLING_FEN {
                if pos.castling_rights & bit != 0 {
                    if shredder {
                        f.write_char((frc_base + pos.castling_rooks[slot].file()) as char)?;
                    } else {
                        f.write_char(std_ch)?;
                    };
                }
            }
        }

        f.write_char(' ')?;

        match pos.en_passant {
            Some(sq) => write!(f, "{sq}")?,
            None => f.write_char('-')?,
        }
        write!(f, " {} {}", pos.halfmove_clock, pos.fullmove_number)
    }
}

pub(super) fn pretty_print(pos: &Position) {
    let ev = pos.psqt_accumulator().to_array();
    let [mg, eg, ph] = [ev[0], ev[1], ev[2]].map(i32::from);
    let phase = ph.min(TOTAL_PHASE);
    let raw = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    let eval = if pos.stm == Color::Black { -raw } else { raw };

    let castling: String = CASTLING_FEN
        .iter()
        .filter_map(|&(bit, _, ch, _)| (pos.castling_rights & bit != 0).then_some(ch))
        .collect();

    let ep = pos.en_passant.map_or("-".into(), |sq| sq.to_string());
    let king_bb = pos.pieces(PieceType::King, pos.stm);
    let in_check = !king_bb.is_empty() && pos.is_attacked(king_bb.lsb(), pos.stm.opposite(), Bitboard::EMPTY);

    let frc_suffix = if pos.frc_notation {
        let rooks: String = pos.castling_rooks.iter().map(|sq| format!(" {sq}")).collect();
        format!(" | Castling Rooks:{rooks}")
    } else {
        String::new()
    };

    let info = [
        format!(" Material + PSQT: {:+.2}{frc_suffix}", f64::from(eval) / 100.0),
        format!(" In Check: {in_check}"),
        format!(" Half Moves: {}", pos.halfmove_clock),
        format!(" En Passant: {ep}"),
        format!(" Side To Move: {:?}", pos.stm),
        format!(" Castle Rights: {}", if castling.is_empty() { "-".into() } else { castling }),
        format!(" Zobrist Hash: 0x{:016x}", pos.hash),
        format!(" FEN: {pos}"),
    ];

    println!("   +------------------------+");
    for rank in (0..8u8).rev() {
        let pieces: String = (0..8u8)
            .map(|file| {
                let sq = Square::from_coords(file, rank);
                let pt = pos.piece_at(sq);
                if pt == PieceType::None { " . ".into() } else { format!(" {} ", pt.to_char(pos.color_at(sq))) }
            })
            .collect();

        println!(" {} |{pieces}|{}", rank + 1, info[rank as usize]);
    }
    println!("   +------------------------+");
    println!("     a  b  c  d  e  f  g  h\n");
}

fn finish_position(mut pos: Position) -> Result<Position, FenError> {
    let kings = pos.role_bb[PieceType::King];
    if (kings & pos.side_bb[Color::White]).popcount() != 1 {
        return Err(FenError::MissingKing { color: "white" });
    }
    if (kings & pos.side_bb[Color::Black]).popcount() != 1 {
        return Err(FenError::MissingKing { color: "black" });
    }

    let illegal_pawns = pos.role_bb[PieceType::Pawn] & (RANK_1 | RANK_8);
    if illegal_pawns.is_not_empty() {
        return Err(FenError::IllegalPawnRank { square: illegal_pawns.lsb() });
    }

    let us = pos.stm;
    let them = us.opposite();
    let their_king_sq = (pos.role_bb[PieceType::King] & pos.side_bb[them]).lsb();

    if pos.is_attacked(their_king_sq, us, Bitboard::EMPTY) {
        return Err(FenError::IllegalCheck);
    }

    pos.recompute_hashes();
    Ok(pos)
}

fn parse_placement(pos: &mut Position, field: &str) -> Result<(), FenError> {
    let (mut rank, mut file): (i32, i32) = (7, 0);

    for ch in field.chars() {
        match ch {
            '/' => {
                if file != 8 {
                    return Err(FenError::InvalidRankWidth { rank: rank as u8, width: file as u8 });
                }

                rank -= 1;
                file = 0;

                if rank < 0 {
                    return Err(FenError::TooManyRanks { rank: 0, count: (8 - rank) as u8 });
                }
            },
            '1'..='8' => {
                file += (ch as u8 - b'0') as i32;
                if file > 8 {
                    return Err(FenError::FileOverflow { rank: rank as u8, file: file as u8 });
                }
            },
            _ => {
                let pt = PieceType::from_char(ch);
                if pt == PieceType::None {
                    return Err(FenError::InvalidPiece { ch, rank: rank as u8, file: file as u8 });
                }

                if rank < 0 || file >= 8 {
                    return Err(FenError::SquareOutOfBounds { square: rank * 8 + file });
                }

                let color = if ch.is_uppercase() { Color::White } else { Color::Black };
                pos.add_piece(Square::from_coords(file as u8, rank as u8), pt, color);
                file += 1;
            },
        }
    }

    if rank != 0 || file != 8 {
        Err(FenError::InvalidRankWidth { rank: rank as u8, width: file as u8 })
    } else {
        Ok(())
    }
}

/// FEN records every double-push target; Soul keeps it only when pseudo-legally capturable.
fn parse_en_passant(pos: &mut Position, token: &str) -> Result<(), FenError> {
    let rank = if pos.stm == Color::White { 5 } else { 2 };
    let sq = Square::from_algebraic(token)
        .filter(|sq| sq.rank() == rank)
        .ok_or_else(|| FenError::InvalidEnPassant { square: token.to_string() })?;
    let victim = sq ^ 8;

    if !pos.occ.check_bit(sq)
        && pos.piece_at(victim) == PieceType::Pawn
        && pos.side_bb[pos.stm.opposite()].check_bit(victim)
        && pos.has_ep_attacker(sq)
    {
        pos.en_passant = Some(sq);
    }
    Ok(())
}

/// `KQkq` finds the rook nearest the corresponding back-rank edge.
/// Shredder-FEN `A`-`H` / `a`-`h` names the rook's file directly.
fn parse_castling_rights(pos: &mut Position, token: &str) -> Result<(), FenError> {
    let wk_file = king_file(pos, Color::White);
    let bk_file = king_file(pos, Color::Black);

    for ch in token.chars() {
        let (color, bit, slot, sq) = match ch {
            'K' => {
                let king_file = wk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let sq = find_rook(pos, Color::White, king_file, KINGSIDE_FILE, SEARCH_LEFT)
                    .ok_or(FenError::InvalidCastlingRight { right: ch })?;
                (Color::White, WHITE_OO, ROOK_W_KS, sq)
            },
            'Q' => {
                let king_file = wk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let sq = find_rook(pos, Color::White, king_file, QUEENSIDE_FILE, SEARCH_RIGHT)
                    .ok_or(FenError::InvalidCastlingRight { right: ch })?;
                (Color::White, WHITE_OOO, ROOK_W_QS, sq)
            },
            'k' => {
                let king_file = bk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let sq = find_rook(pos, Color::Black, king_file, KINGSIDE_FILE, SEARCH_LEFT)
                    .ok_or(FenError::InvalidCastlingRight { right: ch })?;
                (Color::Black, BLACK_OO, ROOK_B_KS, sq)
            },
            'q' => {
                let king_file = bk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let sq = find_rook(pos, Color::Black, king_file, QUEENSIDE_FILE, SEARCH_RIGHT)
                    .ok_or(FenError::InvalidCastlingRight { right: ch })?;
                (Color::Black, BLACK_OOO, ROOK_B_QS, sq)
            },
            'A'..='H' => {
                let king_file = wk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let file = ch as u8 - b'A';
                let (bit, slot) = if file < king_file { (WHITE_OOO, ROOK_W_QS) } else { (WHITE_OO, ROOK_W_KS) };
                (Color::White, bit, slot, Square::from_coords(file, 0))
            },
            'a'..='h' => {
                let king_file = bk_file.ok_or(FenError::InvalidCastlingRight { right: ch })?;
                let file = ch as u8 - b'a';
                let (bit, slot) = if file < king_file { (BLACK_OOO, ROOK_B_QS) } else { (BLACK_OO, ROOK_B_KS) };
                (Color::Black, bit, slot, Square::from_coords(file, 7))
            },
            _ => return Err(FenError::InvalidCastlingRight { right: ch }),
        };

        if pos.castling_rights & bit != 0 || pos.piece_at(sq) != PieceType::Rook || !pos.side_bb[color].check_bit(sq) {
            return Err(FenError::InvalidCastlingRight { right: ch });
        }

        pos.castling_rights |= bit;
        pos.castling_rooks[slot] = sq;
    }
    Ok(())
}

fn king_file(pos: &Position, color: Color) -> Option<u8> {
    let kings = pos.role_bb[PieceType::King] & pos.side_bb[color];
    if kings.popcount() != 1 {
        return None;
    }

    let king = kings.lsb();
    (king.rank() == color.back_rank()).then(|| king.file())
}

fn find_rook(pos: &Position, color: Color, king_file: u8, start_file: u8, step: i8) -> Option<Square> {
    let rank = color.back_rank();
    let mut file = start_file as i8;

    while (step < 0 && file > king_file as i8) || (step > 0 && file < king_file as i8) {
        let sq = Square(rank * 8 + file as u8);
        if pos.piece_at(sq) == PieceType::Rook && pos.side_bb[color].check_bit(sq) {
            return Some(sq);
        }
        file += step;
    }
    None
}
