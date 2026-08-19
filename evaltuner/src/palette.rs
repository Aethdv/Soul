//! Evaluation-tuner color palette.
//!
//! Loss carries no absolute scale, so its color is always relative: per-epoch
//! trend on the live line, within-window rank on the sparkline, both off
//! the shared advantage gradient.
//!
//! The fixed pens are compile-time escapes, so a format string names one inline and
//! nothing rebuilds the same strings per report. [`fg`] stays for computed colors.

use crate::engine::{Rgb, color};

/// What each pen means in a report. The hues live in `color`, so the engine and the
/// tuner cannot drift apart.
/// Field labels.
pub const LAB: &str = color::TAUPE_PEN;
/// Config and telemetry values.
pub const VAL: &str = color::MINT_PEN;
/// Dataset counts.
pub const COUNT: &str = color::STEEL_PEN;
/// Best-epoch marker.
pub const BRAND: &str = color::GOLD_PEN;
/// Incidental figures: train, ref, lr.
pub const DIM: &str = color::ASH_PEN;
/// Warnings and refusals.
pub const ALARM: &str = color::ALARM_PEN;
/// Parameters the run changed.
pub const MOVED: &str = color::JADE_PEN;

pub use color::{CLEAR_LINE, RESET};

/// Truecolor foreground escape for a color decided at runtime.
#[must_use]
pub fn fg(c: Rgb) -> String { crate::engine::ansi_fg(c) }
