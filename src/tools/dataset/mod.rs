//! Dataset representation and I/O.
//!
//! Defines the binary format used for storing and loading self-play positions,
//! their evaluations, and game outcomes for the tuner.

use zerocopy::{FromBytes, Immutable, IntoBytes};

use crate::core::board::Position;

pub mod cli;
mod gradient;
mod io;
mod quant;
pub mod viri_format;

pub use gradient::{FeatureSlots, accumulate_gradient_cached, eval_soul_cached};
pub use io::{MAGIC_V5, MAGIC_V6, append_encoded, load_encoded, parse_epd_entry, parse_epd_str, save_encoded};
pub use viri_format::parse_viri_file;

/// 32-byte entry — minimal ground truth: board state + labels.
///
/// Occupancy + nibble array: the i-th nibble (LSB-to-MSB) in `pieces` encodes
/// the piece on the i-th set bit of `occupancy`.
/// Nibble layout: bits 0-2 = type (pawn=0..king=5), bit 3 = colour (0=White, 1=Black).
/// Unused nibbles (past popcount) are zero.
///
/// Fields are ordered from largest alignment requirement (`align 8`) down to
/// `align 1` for zero-pad under `repr(C)`.
#[derive(Clone, Copy, Immutable, IntoBytes, FromBytes)]
#[repr(C)]
pub struct SoulEntry {
    pub occupancy: u64,   //  8 B — bitboard of occupied squares
    pub pieces: [u8; 16], // 16 B — 32 × 4-bit piece nibbles
    pub score: i16,       //  2 B — search eval label (centipawns, STM relative)
    pub result: u8,       //  1 B — 0=loss, 1=draw, 2=win (from us perspective)
    pub stm_and_ep: u8,   //  1 B — 7=STM (0=W,1=B), bits0-6=ep sq (64=none)
    pub castling: u8,     //  1 B — standard FEN castling byte (KQkq)
    pub _pad: [u8; 3],    //  3 B — to 32
}

const _: () = assert!(size_of::<SoulEntry>() == 32);

impl Default for SoulEntry {
    fn default() -> Self {
        Self {
            occupancy: 0,
            pieces: [0u8; 16],
            score: 0,
            result: 0,
            stm_and_ep: 64, // 64 = en passant none, STM White
            castling: 0,
            _pad: [0u8; 3],
        }
    }
}

impl SoulEntry {
    /// Encode a board position into a training entry.
    ///
    /// `static_score` is accepted for call-site compatibility but not stored.
    /// V6 stores only the search eval label.
    pub fn from_board(board: &Position, result: f64, static_score: Option<i32>, search_score: Option<i32>) -> Self {
        quant::from_board(board, result, static_score, search_score)
    }

    /// Decode the packed entry back into a FEN string.
    #[inline]
    pub fn to_fen(&self) -> String {
        quant::to_fen(self)
    }
}
