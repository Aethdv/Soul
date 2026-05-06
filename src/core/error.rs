//! Unified error handling.
//!
//! Defines strongly typed error variants for FEN parsing, move application,
//! option handling, and engine runtime faults.

use thiserror::Error;

#[derive(Debug, Error, Clone)]
pub enum FenError {
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

    #[error("Square {square} out of bounds (0-63)")]
    SquareOutOfBounds { square: i32 },

    #[error("Missing side to move field")]
    MissingStm,

    #[error("Invalid side to move '{stm}' (expected 'w' or 'b')")]
    InvalidStm { stm: String },

    #[error("Invalid en passant square '{square}' (expected a3-h3, a6-h6, or -)")]
    InvalidEnPassant { square: String },

    #[error("Missing {color} king in position")]
    MissingKing { color: &'static str },

    #[error("Opponent king can be captured (illegal side-to-move in check)")]
    IllegalCheck,

    #[error("Castling rights specify a rook at {sq}, but none was found")]
    InvalidCastlingRights { sq: String },
}

/// Move parsing/application errors.
#[derive(Debug, Error, Clone, Copy)]
pub enum MoveError {
    #[error("Illegal move from {from} to {to}")]
    IllegalMove { from: u8, to: u8 },

    #[error("Move not found in legal list")]
    NotFound,

    #[error("Invalid move format")]
    InvalidFormat,
}

/// Engine runtime errors.
#[derive(Debug, Error)]
pub enum EngineError {
    #[error("Search thread panicked: {0}")]
    SearchPanic(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),

    #[error("Timeout after {ms}ms")]
    Timeout { ms: u64 },
}

/// Option parsing errors for setoption commands.
#[derive(Debug, Error)]
pub enum OptionError {
    #[error("Unknown option: '{name}'")]
    UnknownOption { name: String },

    #[error("Invalid value '{value}' for option '{name}': {reason}")]
    InvalidValue { name: String, value: String, reason: &'static str },

    #[error("Missing value for option '{0}'")]
    MissingValue(String),
}

/// Unified engine error type for Result returns.
#[derive(Debug, Error)]
pub enum SoulError {
    #[error(transparent)]
    Fen(#[from] FenError),

    #[error(transparent)]
    Move(#[from] MoveError),

    #[error(transparent)]
    Engine(#[from] EngineError),

    #[error(transparent)]
    Option(#[from] OptionError),
}

/// Convenient Result alias for engine operations.
pub type SoulResult<T> = Result<T, SoulError>;

// Helper to extract panic message from a dynamic trait object.
impl EngineError {
    /// Create `SearchPanic` from a thread join error.
    #[cold]
    pub fn from_panic(err: &(dyn std::any::Any + Send)) -> Self {
        let msg = if let Some(s) = err.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = err.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown panic payload".to_string()
        };
        Self::SearchPanic(msg)
    }
}
