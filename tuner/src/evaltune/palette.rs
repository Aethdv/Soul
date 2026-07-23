//! Evaluation-tuner color palette.
//!
//! Loss carries no absolute scale, so its color is always relative: per-epoch
//! trend on the live line, within-window rank on the sparkline, both off
//! the shared advantage gradient.

use soul::color::Rgb;

pub const LABEL: Rgb = (150, 140, 128); // field labels             (muted taupe)
pub const VALUE: Rgb = (122, 205, 196); // config / telemetry       (teal)
pub const COUNT: Rgb = (176, 196, 222); // dataset counts           (light steel blue)
pub const BRAND: Rgb = (218, 165, 32); // best-epoch marker         (goldenrod)
pub const DIM: Rgb = (118, 112, 104); // incidental (train/ref/lr)  (darker taupe)

/// ANSI reset.
pub const RESET: &str = "\x1b[0m";

/// Erase to end of line (non-destructive).
pub const CLEAR_LINE: &str = "\x1b[K";

/// Truecolor foreground escape for `c`.
#[must_use]
pub fn fg(c: Rgb) -> String {
    soul::color::ansi_fg(c)
}
