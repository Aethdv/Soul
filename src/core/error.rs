//! The errors a FEN or a move string can raise.

use thiserror::Error;

use crate::core::primitives::Square;

#[derive(Debug, Error, Clone)]
pub(crate) enum FenError {
    #[error("Empty FEN string")]
    Empty,
    #[error("Too many ranks in FEN at rank {rank} (expected 8, got {count})")]
    TooManyRanks { rank: u8, count: u8 },
    #[error("File overflow at rank {rank}: column {file} exceeds 8")]
    FileOverflow { rank: u8, file: u8 },
    #[error("Rank {rank} has invalid width {width} (expected 8)")]
    InvalidRankWidth { rank: u8, width: u8 },
    #[error("Invalid piece character '{ch}' at rank {rank}, file {file}")]
    InvalidPiece { ch: char, rank: u8, file: u8 },
    #[error("Pawn on {square}: pawns cannot occupy rank 1 or 8")]
    IllegalPawnRank { square: Square },
    #[error("Square {square} out of bounds (0-63)")]
    SquareOutOfBounds { square: i32 },
    #[error("Missing side to move field")]
    MissingStm,
    #[error("Invalid side to move '{stm}' (expected 'w' or 'b')")]
    InvalidStm { stm: String },
    #[error("Invalid en passant square '{square}' for the side to move")]
    InvalidEnPassant { square: String },
    #[error("Missing {color} king in position")]
    MissingKing { color: &'static str },
    #[error("Opponent king is attacked (the side not to move is in check)")]
    IllegalCheck,
    #[error("Invalid castling right '{right}'")]
    InvalidCastlingRight { right: char },
}

#[derive(Debug, Error, Clone, Copy)]
pub(crate) enum MoveError {
    #[error("Move not found in legal list")]
    NotFound,
    #[error("Invalid move format")]
    InvalidFormat,
}
