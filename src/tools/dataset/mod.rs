//! Dataset representation and I/O.
//!
//! Provides the binary format for storing training positions, search evaluations,
//! and game outcomes for tuner consumption.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::core::{board::Position, defs::Color};

pub mod cli;
mod gradient;
mod io;
mod quant;
pub mod tape;
pub mod viri_format;

pub use gradient::{FeatureRecord, RecordEval, accumulate_record_grad, eval_record, eval_record_full};
pub use io::{EpdEntry, load_epd_fens, parse_epd_entry, parse_epd_str};
pub use viri_format::{GameScan, ReplayFilter, parse_viri_file, scan_viri_games, write_game};

/// Packed 32-byte training record storing board state, evaluation, and outcome.
///
/// Occupancy is stored as a 64-bit bitboard. Piece types and colors are stored in
/// a 16-byte array of 4-bit nibbles, where the i-th nibble corresponds to the
/// i-th set bit in `occupancy` (traversed from LSB to MSB).
#[derive(Clone, Copy, Immutable, IntoBytes, FromBytes)]
#[repr(C)]
pub struct SoulEntry {
    pub occupancy: u64,
    /// Up to 32 piece descriptors (4 bits each: bits 0..=2 piece type, bit 3 color).
    pub pieces: [u8; 16],
    /// Search evaluation in centipawns (side-to-move relative), or [`Self::NO_SCORE`].
    pub score: i16,
    /// Side-to-move outcome (`0 = Loss, 1 = Draw, 2 = Win`).
    pub result: u8,
    /// Bit 7: STM (`0 = White, 1 = Black`); Bits 0..=6: EP square index (`64` if none).
    pub stm_and_ep: u8,
    /// Castling rights bitmask (`KQkq`).
    pub castling: u8,
    /// Rounds the record out to 32 bytes.
    pub _pad: [u8; 3],
}

const _: () = assert!(size_of::<SoulEntry>() == 32);

impl Default for SoulEntry {
    fn default() -> Self {
        Self { occupancy: 0, pieces: [0u8; 16], score: 0, result: 0, stm_and_ep: 64, castling: 0, _pad: [0u8; 3] }
    }
}

impl SoulEntry {
    /// Sentinel evaluation assigned to positions without search labels (e.g., raw EPD records).
    pub const NO_SCORE: i16 = i16::MAX;

    pub fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> Self {
        quant::from_board(board, result, search_score)
    }

    #[inline]
    pub fn to_fen(&self) -> String {
        quant::to_fen(self)
    }

    /// Computes total weighted material (`P=1, N=3, B=3, R=5, Q=9`), matching [`Position::material_count`].
    #[inline]
    pub fn material_count(&self) -> u32 {
        quant::material_count(self)
    }

    /// Unpacks the entry into a `Position`, through a FEN.
    #[inline]
    pub fn to_board(&self) -> Position {
        Position::from_fen(&self.to_fen())
    }
}

/// Converts a WDL probability `[0.0, 1.0]` between White-relative and side-to-move relative perspective (self-inverse).
#[inline]
pub fn flip_wdl(wdl: f64, stm: Color) -> f64 {
    if stm == Color::Black { 1.0 - wdl } else { wdl }
}

/// Converts a packed outcome (`0=loss, 1=draw, 2=win`) between White-relative and side-to-move perspective (self-inverse).
#[inline]
pub const fn flip_result(result: u8, stm: Color) -> u8 {
    if matches!(stm, Color::Black) { 2 - result } else { result }
}

/// Converts a centipawn score between White-relative and side-to-move perspective (self-inverse).
#[inline]
pub const fn flip_score(score: i32, stm: Color) -> i32 {
    if matches!(stm, Color::Black) { -score } else { score }
}

#[cfg(test)]
mod tests {
    use super::{Color, flip_result, flip_wdl};

    #[test]
    fn perspective_flip_is_involutive() {
        for stm in [Color::White, Color::Black] {
            for result in 0..=2u8 {
                assert_eq!(flip_result(flip_result(result, stm), stm), result, "{result} under {stm:?}");
            }
            for wdl in [0.0, 0.25, 0.5, 1.0] {
                assert!((flip_wdl(flip_wdl(wdl, stm), stm) - wdl).abs() < f64::EPSILON, "{wdl} under {stm:?}");
            }
        }
    }

    #[test]
    fn white_perspective_is_identity() {
        assert_eq!(flip_result(2, Color::White), 2);
        assert_eq!(flip_result(2, Color::Black), 0);
        assert!((flip_wdl(1.0, Color::White) - 1.0).abs() < f64::EPSILON);
        assert!(flip_wdl(1.0, Color::Black).abs() < f64::EPSILON);
    }
}
