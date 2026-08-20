//! Evaluation-tuner color palette.
//!
//! The hues live in `src/color.rs`. The fixed pens are compile-time escapes,
//! so a format string names one inline and nothing rebuilds the same strings
//! per report. [`fg`] stays for computed colors.

use crate::engine::color;

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

pub use color::{CLEAR_LINE, RESET, ansi_fg as fg};

/// A warning line: the alarm pen, the `[!]` prefix and a reset.
#[macro_export]
macro_rules! alarm {
    ($($arg:tt)*) => {
        eprintln!("{}[!] {}{}", $crate::palette::ALARM, format_args!($($arg)*), $crate::palette::RESET)
    };
}
