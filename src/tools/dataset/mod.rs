//! Training records, dataset readers, and the tuner's cached gradient path.

use zerocopy::{Immutable, IntoBytes};

use crate::{
    core::{
        board::Position,
        defs::{Color, TOTAL_PHASE},
    },
    engine::eval_params::LAYOUT,
};

pub mod cli;
mod epd;
mod gradient;
mod packing;
mod tape;
mod viri_format;

#[cfg(feature = "datagen")]
pub(crate) use epd::load_epd_fens;
pub use epd::{EpdEntry, parse_epd_str};
pub use gradient::{FeatureRecord, RecordEval, accumulate_record_grad, eval_record, eval_record_full};
pub use tape::eval_f64;
#[cfg(feature = "datagen")]
pub(crate) use viri_format::write_game;
pub use viri_format::{GameScan, MAX_QUIET_ENDING, MIN_DECISIVE_ENDING, ReplayFilter, parse_viri_file, scan_viri_games};

#[derive(Clone, Copy, Immutable, IntoBytes)]
#[repr(C)]
pub struct SoulEntry {
    pub occupancy: u64,
    pub pieces: [u8; 16],
    pub score: i16,
    pub result: u8,
    pub stm_and_ep: u8,
    pub castling_rights: u8,
    _pad: [u8; 3],
}

const _: () = assert!(size_of::<SoulEntry>() == 32);

impl Default for SoulEntry {
    fn default() -> Self {
        Self {
            occupancy: 0,
            pieces: [0u8; 16],
            score: 0,
            result: 0,
            stm_and_ep: 64,
            castling_rights: 0,
            _pad: [0u8; 3],
        }
    }
}

impl SoulEntry {
    /// Sentinel for a record without a search label.
    pub const NO_SCORE: i16 = i16::MAX;

    #[inline]
    pub const fn stm(&self) -> Color { if self.stm_and_ep & 0x80 == 0 { Color::White } else { Color::Black } }

    /// Builds a record from `board`; both labels are side-to-move relative.
    pub fn from_board(board: &Position, result: f64, search_score: Option<i32>) -> Self {
        packing::from_board(board, result, search_score)
    }

    #[inline]
    pub(super) fn to_fen(self) -> String { self.to_board().to_string() }

    #[inline]
    pub(super) fn material_count(&self) -> u32 { packing::material_count(self) }

    /// Unpacks the stored board. Clocks are reset; callers that need them read the file header.
    #[inline]
    pub fn to_board(&self) -> Position {
        packing::unpack_board(self.occupancy, &self.pieces, self.stm_and_ep, 0, 1, Some(self.castling_rights))
            .expect("invalid packed SoulEntry")
    }
}

// Keep the phase whole: the combiner divides by TOTAL_PHASE once at the end.
fn compute_phase_f64(piece_counts: &[f64; 6], values: &[f64]) -> f64 {
    let weights = &values[LAYOUT.phase_offset..LAYOUT.phase_offset + 6];
    let mut phase_raw = 0.0;
    for (&count, &weight) in piece_counts.iter().zip(weights) {
        phase_raw += count * weight;
    }

    phase_raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc()
}

/// Flips a WDL probability between White-relative and side-to-move perspective.
#[inline]
pub fn flip_wdl(wdl: f64, stm: Color) -> f64 { if stm == Color::Black { 1.0 - wdl } else { wdl } }

#[inline]
pub(super) const fn flip_result(result: u8, stm: Color) -> u8 { if matches!(stm, Color::Black) { 2 - result } else { result } }

/// Flips a centipawn score between White-relative and side-to-move perspective.
#[inline]
pub const fn flip_score(score: i32, stm: Color) -> i32 { if matches!(stm, Color::Black) { -score } else { score } }

#[cfg(test)]
mod tests {
    use super::{Color, flip_result, flip_score, flip_wdl};

    #[test]
    fn perspective_flips_round_trip_and_white_is_identity() {
        for stm in [Color::White, Color::Black] {
            for result in 0..=2u8 {
                assert_eq!(flip_result(flip_result(result, stm), stm), result, "{result} under {stm:?}");
            }
            for wdl in [0.0, 0.25, 0.5, 1.0] {
                assert!((flip_wdl(flip_wdl(wdl, stm), stm) - wdl).abs() < f64::EPSILON, "{wdl} under {stm:?}");
            }
        }
        assert_eq!(flip_result(2, Color::White), 2);
        assert_eq!(flip_result(2, Color::Black), 0);
        assert!((flip_wdl(1.0, Color::White) - 1.0).abs() < f64::EPSILON);
        assert!(flip_wdl(1.0, Color::Black).abs() < f64::EPSILON);
        assert_eq!(flip_score(5, Color::White), 5);
        assert_eq!(flip_score(5, Color::Black), -5);
    }
}
