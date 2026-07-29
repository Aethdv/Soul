//! Evaluation-tuner color palette.
//!
//! Loss carries no absolute scale, so its color is always relative: per-epoch
//! trend on the live line, within-window rank on the sparkline, both off
//! the shared advantage gradient.
//!
//! The fixed pens are escapes built from their channels at compile time, so a
//! format string names one inline and nothing rebuilds the same three strings
//! per report. [`fg`] stays for the colors that are computed.

use super::engine::Rgb;

/// Truecolor foreground escape for the channels given.
macro_rules! pens {
    ($( $name:ident = $r:literal, $g:literal, $b:literal; $note:literal )*) => {
        $(
            #[doc = $note]
            pub const $name: &str = concat!("\x1b[38;2;", $r, ";", $g, ";", $b, "m");
        )*
    };
}

#[rustfmt::skip]
pens! {
    LAB   = 150, 140, 128; "field labels (muted taupe)"
    VAL   = 122, 205, 196; "config and telemetry values (teal)"
    COUNT = 176, 196, 222; "dataset counts (light steel blue)"
    BRAND = 218, 165,  32; "best-epoch marker (goldenrod)"
    DIM   = 118, 112, 104; "incidental: train, ref, lr (darker taupe)"
    ALARM = 225,  89,  91; "warnings and refusals (signal red)"
    MOVED = 100, 200, 120; "parameters the run changed (green)"
}

/// ANSI reset.
pub const RESET: &str = "\x1b[0m";

/// Erase to end of line (non-destructive).
pub const CLEAR_LINE: &str = "\x1b[K";

/// Truecolor foreground escape for a color decided at runtime.
#[must_use]
pub fn fg(c: Rgb) -> String {
    super::engine::ansi_fg(c)
}
