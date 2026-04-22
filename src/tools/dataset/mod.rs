//! Dataset representation and I/O.
//!
//! Defines the binary format used for storing and loading self-play positions,
//! their evaluations, and game outcomes for the tuner.

use zerocopy::{FromBytes, FromZeros, Immutable, IntoBytes, Unaligned};

use crate::{
    core::{
        board::Position,
        defs::{Color, Square},
    },
    engine::mobility::SafetyMetrics,
};

pub mod cli;
mod gradient;
mod io;
mod quant;

pub use gradient::{accumulate_gradient, eval_soul};
pub use io::{MAGIC_V5, append_encoded, load_encoded, parse_epd_entry, parse_epd_str, save_encoded};
pub use quant::compute_openness_factors;

/// Packed piece encoding:
///
/// ```text
///   15        10  9    6  5       0
///   ┌─────────┬────────┬──────────┐
///   │ unused  │pt+color│  square  │
///   └─────────┴────────┴──────────┘
/// ```
///
/// 10 bits of payload in a `u16` — the upper 6 bits are dead weight we pay
/// for alignment, but it keeps the arithmetic trivial and the struct `Copy`.
#[derive(Clone, Copy, Default, Immutable, IntoBytes, FromBytes)]
#[repr(transparent)]
pub struct PackedPiece(pub u16);

impl PackedPiece {
    #[inline]
    pub fn new(pt: usize, color: Color, sq: Square) -> Self {
        // Bit 3 of the upper nibble encodes color: 0 = White, 1 = Black.
        let color_bit = if color == Color::White { 0u16 } else { 8 };
        Self(((pt as u16 | color_bit) << 6) | (u16::from(sq) & 0x3F))
    }

    #[inline]
    pub fn unpack(self) -> (usize, Color, Square) {
        let sq = Square((self.0 & 0x3F) as u8);
        let upper = (self.0 >> 6) as usize;
        (upper & 0x07, if upper & 0x08 != 0 { Color::Black } else { Color::White }, sq)
    }

    #[inline]
    pub fn set_square(&mut self, sq: Square) {
        self.0 = (self.0 & !0x3Fu16) | (u16::from(sq) & 0x3F);
    }
}

/// King-safety metrics, crammed into 4 bytes.
///
/// `exposure` packs two 4-bit values: `[ortho:4][diag:4]`.
#[derive(Clone, Copy, Default, Immutable, IntoBytes, FromBytes, Unaligned)]
#[repr(C)]
pub struct PackedSafety {
    pub attackers: u8,
    pub weak: i8,
    pub shield: i8,
    pub exposure: u8,
}

impl From<SafetyMetrics> for PackedSafety {
    #[inline]
    fn from(m: SafetyMetrics) -> Self {
        Self {
            attackers: m.attackers as u8,
            weak: m.weak as i8,
            shield: m.shield as i8,
            exposure: (m.ortho_exposure.clamp(0, 15) as u8) << 4 | (m.diag_exposure.clamp(0, 15) as u8),
        }
    }
}

impl PackedSafety {
    #[inline]
    pub fn to_metrics(self) -> SafetyMetrics {
        SafetyMetrics {
            attackers: self.attackers as usize,
            weak: self.weak as i32,
            shield: self.shield as i32,
            ortho_exposure: (self.exposure >> 4) as i32,
            diag_exposure: (self.exposure & 0x0F) as i32,
        }
    }
}

pub const STM_WHITE: u8 = 0;
pub const STM_BLACK: u8 = 1;

/// 96-byte entry — every field is STM-relative ("Us" = side to move).
///
/// Fields are ordered from largest alignment to smallest to naturally achieve
/// zero padding under `repr(C)`. This avoids the unaligned reference Undefined
/// Behavior of `repr(C, packed)` while still allowing direct blitting via zerocopy.
#[derive(Clone, Copy, Immutable, IntoBytes, FromBytes)]
#[repr(C)]
pub struct SoulEntry {
    pub result: f32,               // 1.0 = Us won, 0.5 = draw, 0.0 = Us lost
    pub pieces: [PackedPiece; 32], // piece list, squares normalised to Us perspective
    pub static_score: i16,         // raw static eval (centipawns, STM-relative)
    pub search_score: i16,         // search eval from last iteration
    /// STM-relative mobility [us_*, them_*]. Note: these are stored features
    /// and may go stale if the engine's mobility formula changes.
    pub mobility: [i8; 8],
    /// King safety metrics. Note: stored features, may go stale vs formula changes.
    pub safety_us: PackedSafety, // 4 bytes
    pub safety_them: PackedSafety, // 4 bytes
    pub piece_count: u8,           // total pieces on board (2–32)
    pub original_stm: u8,          // 0 = White, 1 = Black (see STM_WHITE/STM_BLACK)
    pub castling: u8,              // us_ks | us_qs | them_ks | them_qs
    /// Rank-flipped if Us is Black. 64 = none. Note: only stored if ep capture
    /// is actually legal, so round-tripping to FEN might lose phantom ep squares.
    pub ep_square: u8,
    pub xray_ortho: i8,
    pub _padding: [u8; 3], // explicit padding to satisfy zerocopy
}

const _: () = assert!(size_of::<SoulEntry>() == 96);

// All-zero bytes are a valid SoulEntry (f32 0.0 = 0x00000000, integers = 0).
// FromBytes implies FromZeros, so we get a correct, branchless default for free.
impl Default for SoulEntry {
    #[inline]
    fn default() -> Self {
        Self::new_zeroed()
    }
}

impl SoulEntry {
    /// Encode a board position into a training entry, normalised to STM perspective.
    pub fn from_board(board: &Position, result: f64, static_score: Option<i32>, search_score: Option<i32>) -> Self {
        quant::from_board(board, result, static_score, search_score)
    }

    #[inline]
    pub fn to_fen(&self) -> String {
        quant::to_fen(self)
    }
}
