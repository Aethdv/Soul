//! Perceptual color: sRGB → OkLab/OkLCH for gradients that never muddy.
//!
//! Two interpolation paths, both perceptual so no ramp dips through gray at its
//! midpoint; `advantage` sweeps authored OkLCH waypoints (hue-aware) for the
//! win/loss gradient, and `mix` blends any two sRGB colors through OkLab. The
//! engine's pretty-print and the search tuner both draw from here, so the
//! colors can't drift apart.

pub type Rgb = (u8, u8, u8);

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";

/// Dead-level neutral: the color of exactly zero advantage → `+0.00`.
/// Engine eval and tuner Elo both paint their zero state with it.
pub const LEVEL: Rgb = (120, 170, 220);

/// Branding gold: table headers and field labels.
pub const GOLD: Rgb = (218, 165, 32);

// Advantage-gradient waypoints as (L, C, H°). Authored in OkLCH, never flattened.
const WIN_GOLD: (f64, f64, f64) = (0.80, 0.13, 92.0);
const WIN_GREEN: (f64, f64, f64) = (0.76, 0.16, 145.0);
const WIN_DEEP: (f64, f64, f64) = (0.74, 0.155, 162.0);
const LOSS_PEACH: (f64, f64, f64) = (0.78, 0.11, 45.0);
const LOSS_ORANGE: (f64, f64, f64) = (0.72, 0.16, 35.0);
const LOSS_DEEP: (f64, f64, f64) = (0.64, 0.17, 22.0);

/// Canonical advantage gradient. `t` in `[-1, 1]`: −1 deep loss → +1 deep win.
/// Zero sits at the gold/peach seam; callers paint exact-level states themselves.
#[must_use]
pub fn advantage(t: f64) -> Rgb {
    let m = t.abs().min(1.0);
    let (lo, hi, seg) = match (t >= 0.0, m < 0.5) {
        (true, true) => (WIN_GOLD, WIN_GREEN, m / 0.5),
        (true, false) => (WIN_GREEN, WIN_DEEP, (m - 0.5) / 0.5),
        (false, true) => (LOSS_PEACH, LOSS_ORANGE, m / 0.5),
        (false, false) => (LOSS_ORANGE, LOSS_DEEP, (m - 0.5) / 0.5),
    };
    oklch_lerp(lo, hi, seg)
}

/// ANSI truecolor foreground escape for `c`.
#[must_use]
pub fn ansi_fg(c: Rgb) -> String {
    let mut s = String::with_capacity(20);
    let _ = write_ansi_fg(&mut s, c);
    s
}

/// Write an ANSI truecolor foreground escape for `c` into `w`.
pub fn write_ansi_fg(w: &mut impl core::fmt::Write, c: Rgb) -> core::fmt::Result {
    write!(w, "\x1b[38;2;{};{};{}m", c.0, c.1, c.2)
}

/// Drop the escapes from colored text bound for a file or a pipe.
///
/// Only what this module writes: `ESC [` params `m`, and `ESC [ K`. Anything
/// else keeps its text, escape included, since scanning on for a terminator
/// would swallow the prose up to the next one.
#[must_use]
pub fn strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(esc) = rest.find('\x1b') {
        out.push_str(&rest[..esc]);
        rest = &rest[esc..];

        let params = rest.strip_prefix("\x1b[").unwrap_or_default();
        let Some(end) = params.find(|c: char| !c.is_ascii_digit() && c != ';') else { break };

        // Narrower than the CSI grammar on purpose: over-accepting eats prose,
        // under-accepting leaves an escape in a log.
        if !matches!(params.as_bytes()[end], b'm' | b'K') {
            break;
        }

        rest = &params[end + 1..];
    }

    out.push_str(rest);
    out
}

/// Perceptual blend of two sRGB colors; interpolate in OkLab so the midpoint
/// stays bright and saturated instead of dipping through gray. `t` in `[0, 1]`.
#[must_use]
pub fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let (la, lb) = (srgb_to_oklab(a), srgb_to_oklab(b));
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: f64, y: f64| x + (y - x) * t;
    oklab_to_srgb(lerp(la.0, lb.0), lerp(la.1, lb.1), lerp(la.2, lb.2))
}

/// OkLCH → sRGB, gamut-clamped. `l` in `[0, 1]`, `c` chroma, `h` degrees.
fn oklch(l: f64, c: f64, h: f64) -> Rgb {
    let hr = h.to_radians();
    oklab_to_srgb(l, c * hr.cos(), c * hr.sin())
}

/// Interpolate two OkLCH waypoints and convert. Hue lerps linearly: the
/// gradient's waypoints are monotonic in hue, so there's no wrap to handle.
fn oklch_lerp(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> Rgb {
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: f64, y: f64| x + (y - x) * t;
    oklch(lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

/// OkLab → sRGB (Ottosson's matrices), gamut-clamped.
fn oklab_to_srgb(l: f64, a: f64, b: f64) -> Rgb {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548_0 * b).powi(3);

    let lr = 4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_;
    let lg = -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_;
    let lb = -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701_0 * s_;

    (encode(lr), encode(lg), encode(lb))
}

/// sRGB → OkLab.
fn srgb_to_oklab(c: Rgb) -> (f64, f64, f64) {
    let (r, g, b) = (decode(c.0), decode(c.1), decode(c.2));

    let l = 0.412_221_470_8 * r + 0.536_332_536_3 * g + 0.051_445_992_9 * b;
    let m = 0.211_903_498_2 * r + 0.680_699_545_1 * g + 0.107_396_956_6 * b;
    let s = 0.088_302_461_9 * r + 0.281_718_837_6 * g + 0.629_978_700_5 * b;
    let (l_, m_, s_) = (l.cbrt(), m.cbrt(), s.cbrt());

    (
        0.210_454_255_3 * l_ + 0.793_617_785_0 * m_ - 0.004_072_046_8 * s_,
        1.977_998_495_1 * l_ - 2.428_592_205_0 * m_ + 0.450_593_709_9 * s_,
        0.025_904_037_1 * l_ + 0.782_771_766_2 * m_ - 0.808_675_766_0 * s_,
    )
}

/// Linear-light channel → gamma-encoded sRGB byte.
fn encode(x: f64) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let s = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (s * 255.0).round() as u8
}

/// Gamma-encoded sRGB byte → linear-light channel.
fn decode(b: u8) -> f64 {
    let x = f64::from(b) / 255.0;
    if x <= 0.040_45 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_leaves_only_the_text() {
        let colored = format!("{}Gauge:{} 0.739x", ansi_fg((218, 165, 32)), "\x1b[0m");

        assert_eq!(strip(&colored), "Gauge: 0.739x");
        assert_eq!(strip("nothing to drop"), "nothing to drop");
        assert_eq!(strip(""), "");
    }

    /// An escape with no terminator would otherwise swallow the rest of the line,
    /// or the words up to whatever letter the prose reaches first.
    #[test]
    fn strip_keeps_text_after_an_unterminated_escape() {
        assert_eq!(strip("before\x1b[38;2;1;2;3after"), "before\x1b[38;2;1;2;3after");
        assert_eq!(strip("before\x1b[38;2;1;2;3 more"), "before\x1b[38;2;1;2;3 more");
    }
}
