//! Dataset representation and I/O.
//!
//! Defines the binary format used for storing and loading self-play positions,
//! their evaluations, and game outcomes for the tuner.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::core::{board::Position, defs::Color};

pub mod cli;
mod gradient;
mod io;
mod quant;
pub mod tape;
pub mod viri_format;

pub use gradient::{FeatureRecord, RecordEval, accumulate_record_grad, eval_record, eval_record_full};
pub use io::{MAGIC_V6, append_encoded, count_encoded, load_encoded, load_epd_fens, parse_epd_entry, parse_epd_str, save_encoded};
pub use viri_format::{GameScan, ReplayFilter, parse_viri_file, scan_viri_games};

/// 32-byte entry, minimal ground truth: board state + labels.
///
/// Occupancy + nibble array: the i-th nibble (LSB-to-MSB) in `pieces` encodes
/// the piece on the i-th set bit of `occupancy`.
/// Nibble layout: bits 0-2 = type (pawn=0..king=5), bit 3 = color (0=White, 1=Black).
/// Unused nibbles (past popcount) are zero.
///
/// Fields are ordered from largest alignment requirement (`align 8`) down to
/// `align 1` for zero-pad under `repr(C)`.
#[derive(Clone, Copy, Immutable, IntoBytes, FromBytes)]
#[repr(C)]
pub struct SoulEntry {
    pub occupancy: u64,   //  8B - bitboard of occupied squares
    pub pieces: [u8; 16], // 16B - 32 × 4-bit piece nibbles
    pub score: i16,       //  2B - search eval label (centipawns, STM relative)
    pub result: u8,       //  1B - 0=loss, 1=draw, 2=win (from us perspective)
    pub stm_and_ep: u8,   //  1B - 7=STM (0=W,1=B), bits0-6=ep sq (64=none)
    pub castling: u8,     //  1B - standard FEN castling byte (KQkq)
    pub _pad: [u8; 3],    //  3B - to 32
}

const _: () = assert!(size_of::<SoulEntry>() == 32);

impl Default for SoulEntry {
    fn default() -> Self {
        Self { occupancy: 0, pieces: [0u8; 16], score: 0, result: 0, stm_and_ep: 64, castling: 0, _pad: [0u8; 3] }
    }
}

impl SoulEntry {
    /// `score` on an entry nothing ever searched, an EPD line being the usual one.
    /// Not zero, which is indistinguishable from a genuinely even position.
    pub const NO_SCORE: i16 = i16::MAX;

    /// Encode a board position into a training entry.
    pub fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> Self {
        quant::from_board(board, result, search_score)
    }

    /// Decode the packed entry back into a FEN string.
    #[inline]
    pub fn to_fen(&self) -> String {
        quant::to_fen(self)
    }

    /// Weighted 1/3/3/5/9 material, matching [`Position::material_count`].
    #[inline]
    pub fn material_count(&self) -> u32 {
        quant::material_count(self)
    }

    /// Decode back to a board, through a FEN. A nibble decoder would skip the
    /// string; no caller is hot enough to have wanted one.
    #[inline]
    pub fn to_board(&self) -> Position {
        Position::from_fen(&self.to_fen())
    }
}

/// Swaps a WDL between White-relative and side-to-move-relative.
///
/// An involution, so one function serves both directions and only the call site
/// says which one it meant.
#[inline]
pub fn flip_wdl(wdl: f64, stm: Color) -> f64 {
    if stm == Color::Black { 1.0 - wdl } else { wdl }
}

/// The same swap over a packed `0 = loss, 1 = draw, 2 = win` result.
#[inline]
pub const fn flip_result(result: u8, stm: Color) -> u8 {
    if matches!(stm, Color::Black) { 2 - result } else { result }
}

#[cfg(test)]
mod tests {
    use super::{Color, flip_result, flip_wdl};

    #[test]
    fn flipping_twice_returns_the_original() {
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
    fn white_is_the_identity() {
        assert_eq!(flip_result(2, Color::White), 2);
        assert_eq!(flip_result(2, Color::Black), 0);
        assert!((flip_wdl(1.0, Color::White) - 1.0).abs() < f64::EPSILON);
        assert!(flip_wdl(1.0, Color::Black).abs() < f64::EPSILON);
    }
}
