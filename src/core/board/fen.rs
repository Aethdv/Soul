//! FEN (Forsyth-Edwards Notation) parsing and serialization.

use std::{
    fmt::{self, Display, Formatter, Write as _},
    iter::Peekable,
};

use crate::core::{
    board::{
        BLACK_OO, BLACK_OOO, DEFAULT_KING_FILE, KINGSIDE_FILE, Position, QUEENSIDE_FILE, SEARCH_LEFT, SEARCH_RIGHT, WHITE_OO,
        WHITE_OOO,
    },
    defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
    error::FenError,
    primitives::{RANK_1, RANK_8},
};

/// `(bitmask, rook-slot index, standard char, Shredder-FEN base)`.
/// Adding the rook's file to the Shredder base yields the correct file letter
/// (`A`–`H` for white, `a`–`h` for black).
const CASTLING_FEN: [(u8, usize, char, u8); 4] =
    [(WHITE_OO, 0, 'K', b'A'), (WHITE_OOO, 1, 'Q', b'A'), (BLACK_OO, 2, 'k', b'a'), (BLACK_OOO, 3, 'q', b'a')];

/// Standard rook home squares.
/// Indexed by the same slot order as [`CASTLING_FEN`].
const STANDARD_ROOK_HOMES: [Square; 4] = [
    Square(7),  // h1: white O-O
    Square(0),  // a1: white O-O-O
    Square(63), // h8: black O-O
    Square(56), // a8: black O-O-O
];

/// Constructs a [`Position`] from an iterator over whitespace-split FEN tokens.
///
/// Expects up to six fields (piece placement, side-to-move, castling,
/// en passant, half-move clock, full-move number).
/// The last three gracefully default when absent or malformed.
pub fn try_from_tokens<'a, I>(tokens: &mut Peekable<I>) -> Result<Position, FenError>
where I: Iterator<Item = &'a str> {
    let mut pos = Position::new();

    parse_placement(&mut pos, tokens.next().ok_or(FenError::Empty)?)?;

    // 1. Side to move
    pos.stm = match tokens.next().ok_or(FenError::MissingStm)? {
        "w" => Color::White,
        "b" => Color::Black,
        s => return Err(FenError::InvalidStm { stm: s.to_string() }),
    };

    // 2. Castling availability
    if let Some(cr) = tokens.next()
        && cr != "-"
    {
        parse_castling_rights(&mut pos, cr);
    }

    // 3. En passant target
    if let Some(ep) = tokens.next()
        && ep != "-"
    {
        parse_en_passant(&mut pos, ep)?;
    }

    // 4. Clocks (lenient: malformed values silently default)
    if let Some(&h) = tokens.peek() {
        if h == "moves" {
            return finish_position(pos);
        }

        if let Some(token) = tokens.next() {
            pos.halfmove_clock = token.parse().unwrap_or(0);
        }
    }

    if let Some(&f) = tokens.peek() {
        if f == "moves" {
            return finish_position(pos);
        }

        if let Some(token) = tokens.next() {
            pos.fullmove_number = token.parse().unwrap_or(1);
        }
    }
    finish_position(pos)
}

/// A wrapper for streaming FEN serialization without intermediate allocations.
pub struct Fen<'a>(pub &'a Position);

impl Display for Fen<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let pos = self.0;

        // Piece placement (ranks 8 → 1)
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
                    f.write_char(pt.to_char(color_on(pos, sq)))?;
                }
            }

            if empty > 0 {
                f.write_char((b'0' + empty) as char)?;
            }
        }

        // Side to move
        if pos.stm == Color::White {
            f.write_str(" w ")?;
        } else {
            f.write_str(" b ")?;
        };

        // Castling availability
        if pos.castling_rights == 0 {
            f.write_char('-')?;
        } else {
            let shredder = pos.is_frc && !has_standard_rook_homes(pos);
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

        // En passant target
        f.write_char(' ')?;

        match pos.en_passant {
            Some(sq) => f.write_str(&sq.to_algebraic())?,
            None => f.write_char('-')?,
        }
        // Move clocks
        write!(f, " {} {}", pos.halfmove_clock, pos.fullmove_number)
    }
}

/// Produces the FEN string for the current position.
///
/// In FRC mode, switches to Shredder-FEN castling notation when the rooks
/// don't sit on their standard home files; otherwise `KQkq` would be ambiguous
/// and couldn't round-trip faithfully.
pub fn as_fen(pos: &Position) -> String {
    Fen(pos).to_string()
}

/// Prints a board diagram with evaluation and position metadata.
pub fn pretty_print(pos: &Position) {
    let ev = pos.get_initial_accumulator().to_array();
    let [mg, eg, ph] = [ev[0], ev[1], ev[2]].map(i32::from);
    let phase = ph.min(TOTAL_PHASE);
    let raw = (mg * phase + eg * (TOTAL_PHASE - phase)) / TOTAL_PHASE;
    let eval = if pos.stm == Color::Black { -raw } else { raw };

    let castling: String = CASTLING_FEN
        .iter()
        .filter_map(|&(bit, _, ch, _)| (pos.castling_rights & bit != 0).then_some(ch))
        .collect();

    let ep = pos.en_passant.map_or("-".into(), |sq| sq.to_algebraic());

    let king_bb = pos.pieces(PieceType::King, pos.stm);
    let in_check = !king_bb.is_empty() && pos.is_attacked::<false>(king_bb.lsb(), pos.stm.opposite(), Bitboard::EMPTY);

    let frc_suffix = if pos.is_frc {
        let rooks: String = pos.castling_rooks.iter().map(|sq| format!(" {}", sq.to_algebraic())).collect();
        format!(" | Castling Rooks:{rooks}")
    } else {
        String::new()
    };

    let info = [
        format!(" Static eval: {:+.2}{frc_suffix}", f64::from(eval) / 100.0),
        format!(" In Check: {in_check}"),
        format!(" Half Moves: {}", pos.halfmove_clock),
        format!(" En Passant: {ep}"),
        format!(" Side To Move: {:?}", pos.stm),
        format!(" Castle Rights: {}", if castling.is_empty() { "-".into() } else { castling }),
        format!(" Zobrist Hash: 0x{:016x}", pos.hash),
        format!(" FEN: {}", Fen(pos)),
    ];

    println!("   +------------------------+");
    for rank in (0..8u8).rev() {
        let pieces: String = (0..8u8)
            .map(|file| {
                let sq = Square::from_coords(file, rank);
                let pt = pos.piece_at(sq);
                if pt == PieceType::None { " . ".into() } else { format!(" {} ", pt.to_char(color_on(pos, sq))) }
            })
            .collect();

        println!(" {} |{pieces}|{}", rank + 1, info[rank as usize]);
    }
    println!("   +------------------------+");
    println!("     a  b  c  d  e  f  g  h\n");
}

fn finish_position(mut pos: Position) -> Result<Position, FenError> {
    // 1. King Existence Invariant
    let kings = pos.role_bb[PieceType::King];
    if (kings & pos.side_bb[Color::White]).popcount() != 1 {
        return Err(FenError::MissingKing { color: "white" });
    }
    if (kings & pos.side_bb[Color::Black]).popcount() != 1 {
        return Err(FenError::MissingKing { color: "black" });
    }

    // 2. Pawn Rank Invariant (Pawns cannot exist on 1st/8th ranks)
    let illegal_pawns = pos.role_bb[PieceType::Pawn] & (RANK_1 | RANK_8);
    if illegal_pawns.is_not_empty() {
        let sq = illegal_pawns.lsb();
        let color = if pos.side_bb[Color::White].check_bit(sq) { Color::White } else { Color::Black };

        return Err(FenError::InvalidPiece { ch: PieceType::Pawn.to_char(color), rank: sq.rank(), file: sq.file() });
    }

    // 3. Illegal Check Invariant (Side-not-to-move cannot be in check)
    let us = pos.stm;
    let them = us.opposite();
    let their_king_sq = (pos.role_bb[PieceType::King] & pos.side_bb[them]).lsb();

    // is_attacked::<false> is the most efficient way to check this.
    if pos.is_attacked::<false>(their_king_sq, us, Bitboard::EMPTY) {
        return Err(FenError::IllegalCheck);
    }

    // 4. Castling/Rook Consistency (Crucial for DFRC/Shredder-FEN)
    for slot in 0..4 {
        let bit = 1 << slot;
        if pos.castling_rights & bit != 0 {
            let rsq = pos.castling_rooks[slot];

            if pos.piece_at(rsq) != PieceType::Rook {
                return Err(FenError::InvalidCastlingRights { sq: rsq.to_algebraic() });
            }
        }
    }

    pos.hash = pos.calc_zobrist();
    pos.pawn_key = pos.calc_pawn_hash();
    pos.minor_key = pos.calc_minor_hash();
    pos.major_key = pos.calc_major_hash();
    Ok(pos)
}

/// Which side occupies `sq`?
/// Only meaningful when the square holds a piece.
#[inline]
fn color_on(pos: &Position, sq: Square) -> Color {
    if pos.side_bb[Color::White].check_bit(sq) { Color::White } else { Color::Black }
}

/// True when every active castling rook sits on its traditional home file (a or h).
/// In that case, standard `KQkq` notation is unambiguous even for FRC.
fn has_standard_rook_homes(pos: &Position) -> bool {
    CASTLING_FEN
        .iter()
        .zip(&STANDARD_ROOK_HOMES)
        .all(|(&(bit, slot, ..), &home)| pos.castling_rights & bit == 0 || pos.castling_rooks[slot] == home)
}

/// Parses the piece-placement field (`rnbqkbnr/pppppppp/...`).
///
/// Walks the string rank-by-rank from the 8th rank down.
/// Digits skip empty squares: letters place pieces. Slashes separate ranks.
fn parse_placement(pos: &mut Position, field: &str) -> Result<(), FenError> {
    let (mut rank, mut file): (i32, i32) = (7, 0);
    let mut rank_count: u8 = 1;

    for ch in field.chars() {
        match ch {
            '/' => {
                if file != 8 {
                    return Err(FenError::InvalidRankWidth { rank: rank as u8, width: file as u8 });
                }
                rank -= 1;
                file = 0;
                rank_count += 1;

                if rank < 0 {
                    return Err(FenError::TooManyRanks { rank: 0, count: rank_count });
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

/// Parses the en passant field and records the square only
/// if a friendly pawn can actually capture there.
///
/// Phantom EP squares, legal in the FEN spec but unreachable in the current
/// position, are silently discarded to prevent polluting the Zobrist Hash and
/// Transposition Table with distinctions that can never affect play.
fn parse_en_passant(pos: &mut Position, token: &str) -> Result<(), FenError> {
    let b = token.as_bytes();
    let rank = b.get(1).copied().unwrap_or(0);
    if b.len() != 2 || !(b'a'..=b'h').contains(&b[0]) || (rank != b'3' && rank != b'6') {
        return Err(FenError::InvalidEnPassant { square: token.to_string() });
    }

    let sq = Square((b[1] - b'1') * 8 + (b[0] - b'a'));
    if pos.can_capture_ep(sq, pos.stm) {
        pos.en_passant = Some(sq);
    }
    Ok(())
}

/// Interprets the castling-availability token, supporting two notations:
///
/// - Standard / X-FEN (`KQkq`): the rook is discovered by scanning the back
///   rank inward from the board edge.
/// - Shredder-FEN (`AHah`): each letter directly names the rook's file,
///   disambiguating FRC positions where rooks can sit on any square.
///
/// For Shredder notation, king-side vs queen-side is inferred by comparing
/// the rook's file to its king's file.
fn parse_castling_rights(pos: &mut Position, token: &str) {
    let wk_file = king_file(pos, Color::White);
    let bk_file = king_file(pos, Color::Black);

    for ch in token.chars() {
        match ch {
            // Standard: discover the rook by scanning from the board edge.
            'K' => assign_rook(pos, Color::White, WHITE_OO, 0, KINGSIDE_FILE, SEARCH_LEFT),
            'Q' => assign_rook(pos, Color::White, WHITE_OOO, 1, QUEENSIDE_FILE, SEARCH_RIGHT),
            'k' => assign_rook(pos, Color::Black, BLACK_OO, 2, KINGSIDE_FILE, SEARCH_LEFT),
            'q' => assign_rook(pos, Color::Black, BLACK_OOO, 3, QUEENSIDE_FILE, SEARCH_RIGHT),

            // Shredder: file letter directly identifies the rook.
            'A'..='H' => {
                let file = ch as u8 - b'A';
                let (bit, slot) = if file < wk_file { (WHITE_OOO, 1) } else { (WHITE_OO, 0) };
                pos.castling_rights |= bit;
                pos.castling_rooks[slot] = Square::from_coords(file, 0); // rank 0
            },
            'a'..='h' => {
                let file = ch as u8 - b'a';
                let (bit, slot) = if file < bk_file { (BLACK_OOO, 3) } else { (BLACK_OO, 2) };
                pos.castling_rights |= bit;
                pos.castling_rooks[slot] = Square::from_coords(file, 7); // rank 7
            },
            _ => {}, // Ignore unknown chars gracefully.
        }
    }
}

/// Scans the back rank for a rook and, if found, assigns it to the given
/// castling slot with the corresponding rights bit.
fn assign_rook(pos: &mut Position, color: Color, bit: u8, slot: usize, start: u8, step: i8) {
    if let Some(sq) = find_rook(pos, color, start, step) {
        pos.castling_rights |= bit;
        pos.castling_rooks[slot] = sq;
    }
}

/// Returns the file of `color`'s king, or the standard e-file if absent.
fn king_file(pos: &Position, color: Color) -> u8 {
    let bb = pos.role_bb[PieceType::King] & pos.side_bb[color];
    if bb.is_empty() { DEFAULT_KING_FILE } else { bb.lsb().file() }
}

/// Scans `color`'s back rank starting at `start_file`, stepping by `step`,
/// returning the first rook encountered.
///
/// Used to resolve `KQkq` notation in both standard and FRC positions; the
/// edge-inward scan guarantees we find the outermost rook on the correct side.
fn find_rook(pos: &Position, color: Color, start_file: u8, step: i8) -> Option<Square> {
    let rank = color.back_rank();
    let mut file = start_file as i8;
    while (0..8).contains(&file) {
        let sq = Square(rank * 8 + file as u8);

        if pos.piece_at(sq) == PieceType::Rook && pos.side_bb[color].check_bit(sq) {
            return Some(sq);
        }
        file += step;
    }
    None
}
