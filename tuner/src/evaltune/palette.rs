//! Evaluation-tuner color palette.
//!
//! Same language as the rest of Soul's tooling; warm-neutral chrome, mint-teal
//! for configuration telemetry. Loss carries no absolute scale, so its color is
//! always relative — per-epoch trend on the live line, within-window rank on the
//! sparkline — both off the shared advantage gradient.

use soul::color::Rgb;

pub const LABEL: Rgb = (150, 140, 128); // field labels
pub const VALUE: Rgb = (122, 205, 196); // config / telemetry values
pub const COUNT: Rgb = (176, 196, 222); // dataset counts
pub const BRAND: Rgb = (218, 165, 32); // new-best marker
pub const DIM: Rgb = (118, 112, 104); // incidental (train/ref/time)

/// Truecolor foreground escape for `c`.
#[must_use]
pub fn fg(c: Rgb) -> String {
    soul::color::ansi_fg(c)
}
