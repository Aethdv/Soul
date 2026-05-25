//! Search-tuner color palette.
//!
//! Warm neutrals carry most of the line's surface so the whole readout reads
//! warm; one cool accent (mint-teal) marks optimizer telemetry for contrast;
//! and the Elo values own the advantage gradient — the only place red/green
//! appears, so color there always means something. All gradient work defers to
//! [`soul::color`] so engine and tuner never drift.

use soul::color::{self, Rgb};

pub const LABEL: Rgb = (150, 140, 128); // field labels
pub const SEP: Rgb = (74, 68, 60); // `|` separators
pub const TIME: Rgb = (172, 156, 138); // incidental readout
pub const EPOCH: Rgb = (189, 148, 232); // row anchor
pub const TELEMETRY: Rgb = (122, 205, 196); // σ, η
pub const BUDGET: Rgb = (224, 178, 108); // pair count

/// Truecolor foreground escape for `c` (no reset).
#[must_use]
pub fn fg(c: Rgb) -> String {
    color::ansi_fg(c)
}

/// Elo → advantage-gradient escape over a ±100 Elo range; dead-level blue near 0.
/// Returns the color escape only, so callers append their own reset.
#[must_use]
pub fn elo_color(elo: f64) -> String {
    if elo.abs() < 0.05 { color::ansi_fg(color::LEVEL) } else { color::ansi_fg(color::advantage(elo / 100.0)) }
}
