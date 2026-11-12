//! Coordinate move parsing shared by UCI and XBoard.

use crate::{
    core::{
        board::{Position, castling_targets},
        defs::{PieceType, Square},
        error::MoveError,
        moves::Move,
    },
    engine::movegen::gen_legal_moves,
};

pub(super) fn parse_uci_move(board: &Position, uci: &str) -> Result<Move, MoveError> {
    let clean = uci.trim().trim_end_matches(['+', '#']);
    if clean.eq_ignore_ascii_case("o-o") || clean == "0-0" {
        return gen_legal_moves(board)
            .into_iter()
            .find(|mv| mv.is_castling() && mv.to().file() > mv.from().file())
            .copied()
            .ok_or(MoveError::NotFound);
    }
    if clean.eq_ignore_ascii_case("o-o-o") || clean == "0-0-0" {
        return gen_legal_moves(board)
            .into_iter()
            .find(|mv| mv.is_castling() && mv.to().file() < mv.from().file())
            .copied()
            .ok_or(MoveError::NotFound);
    }

    // Reject non-ASCII before byte slicing.
    if !matches!(uci.len(), 4 | 5) || !uci.is_ascii() {
        return Err(MoveError::InvalidFormat);
    }

    let from_sq = Square::from_algebraic(&uci[0..2]).ok_or(MoveError::InvalidFormat)?;
    let to_sq = Square::from_algebraic(&uci[2..4]).ok_or(MoveError::InvalidFormat)?;
    let promo = if uci.len() == 5 { Some(promotion_piece(uci.as_bytes()[4])?) } else { None };
    let legal = gen_legal_moves(board);

    legal
        .iter()
        .find(|&&mv| {
            if mv.from() == from_sq && mv.to() == to_sq && mv.promo() == promo {
                return true;
            }
            // Castling is stored king-onto-rook, so the king's destination square matches as well.
            mv.is_castling() && promo.is_none() && mv.from() == from_sq && castling_targets(from_sq, mv.to()).0 == to_sq
        })
        .copied()
        .ok_or(MoveError::NotFound)
}

fn promotion_piece(byte: u8) -> Result<PieceType, MoveError> {
    match byte.to_ascii_lowercase() {
        b'q' => Ok(PieceType::Queen),
        b'r' => Ok(PieceType::Rook),
        b'b' => Ok(PieceType::Bishop),
        b'n' => Ok(PieceType::Knight),
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
    fn rejects_malformed_move_strings() {
        let board = Position::from_fen(CASTLING);
        assert!(parse_uci_move(&board, "e1g1xx").is_err());
        assert!(parse_uci_move(&board, "a1a").is_err());
        assert!(parse_uci_move(&board, "aé1").is_err());
    }

    #[test]
    fn xboard_frc_castling() {
        let frc = Position::from_fen("4k3/8/8/8/8/8/8/6KR w H - 0 1");
        let mv = parse_uci_move(&frc, "O-O").unwrap();
        assert!(mv.is_castling());
        assert_eq!(mv.to_xboard(true), "O-O");
        assert_eq!(parse_uci_move(&Position::from_fen(CASTLING), "O-O-O").unwrap().to_xboard(true), "O-O-O");
    }
}
