//! Shared coordinate move-string parsing for the UCI and XBoard protocols.

use crate::{
    core::{
        board::Position,
        defs::{PieceType, Square},
        error::MoveError,
        moves::Move,
    },
    engine::movegen::gen_legal_moves,
};

/// Resolve a coordinate move string (`e2e4`, `e7e8q`) to the legal move it names on
/// `board`. Castling is accepted in both king-to-file (`e1g1`) and king-onto-rook
/// (FRC `e1h1`) notation, so a GUI using either convention parses.
pub fn parse_uci_move(board: &Position, uci: &str) -> Result<Move, MoveError> {
    // Non-ASCII would put a multibyte boundary mid-slice below and panic; reject it.
    if !matches!(uci.len(), 4 | 5) || !uci.is_ascii() {
        return Err(MoveError::InvalidFormat);
    }

    let from_sq = square_from_str(&uci[0..2])?;
    let to_sq = square_from_str(&uci[2..4])?;

    let promo = if uci.len() == 5 { Some(piece_from_char(uci.chars().nth(4).unwrap())?) } else { None };
    let legal = gen_legal_moves(board);

    legal
        .iter()
        .find(|&&mv| {
            if mv.from() == from_sq && mv.to() == to_sq && mv.promo() == promo {
                return true;
            }

            // ── Castling Normalization
            // Castling is stored king-onto-rook, which the comparison above already
            // matches. A GUI sending the king-to-file form (e1g1) instead names a
            // square the king lands on, so it is resolved from the rook's side.
            if mv.is_castling() && promo.is_none() && mv.from() == from_sq {
                let rank = from_sq.rank();
                let is_kingside = mv.to().file() > from_sq.file();
                let dest_file = if is_kingside { 6 } else { 2 }; // G or C

                if to_sq == Square::from_coords(dest_file, rank) {
                    return true;
                }
            }
            false
        })
        .copied()
        .ok_or(MoveError::NotFound)
}

fn square_from_str(s: &str) -> Result<Square, MoveError> {
    let file = s.as_bytes()[0].wrapping_sub(b'a');
    let rank = s.as_bytes()[1].wrapping_sub(b'1');

    if file > 7 || rank > 7 {
        return Err(MoveError::InvalidFormat);
    }

    Ok(Square::from_coords(file, rank))
}

fn piece_from_char(c: char) -> Result<PieceType, MoveError> {
    match c.to_ascii_lowercase() {
        'q' => Ok(PieceType::Queen),
        'r' => Ok(PieceType::Rook),
        'b' => Ok(PieceType::Bishop),
        'n' => Ok(PieceType::Knight),
        _ => Err(MoveError::InvalidFormat),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CASTLING: &str = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1";

    #[test]
    fn castling_parses_from_either_notation() {
        let board = Position::from_fen(CASTLING);
        let kingside = parse_uci_move(&board, "e1h1").unwrap();
        let queenside = parse_uci_move(&board, "e1a1").unwrap();
        assert_eq!(parse_uci_move(&board, "e1g1").unwrap(), kingside);
        assert_eq!(parse_uci_move(&board, "e1c1").unwrap(), queenside);
        assert!(kingside.is_castling() && queenside.is_castling());
        assert!(parse_uci_move(&board, "e1g1q").is_err());
    }

    #[test]
    fn a_move_string_is_four_or_five_characters() {
        let board = Position::from_fen(CASTLING);
        assert!(parse_uci_move(&board, "e1g1xx").is_err());
        assert!(parse_uci_move(&board, "a1a").is_err());
    }
}
