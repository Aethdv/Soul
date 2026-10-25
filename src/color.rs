//! Shared terminal palette and perceptual color interpolation.
//!
//! Advantage ramps interpolate authored OkLCH waypoints; general blends use OkLab.

pub type Rgb = (u8, u8, u8);

pub const RESET: &str = "\x1b[0m";
pub const BOLD: &str = "\x1b[1m";
pub const CLEAR_TO_EOL: &str = "\x1b[K";

macro_rules! palette {
    ($( $name:ident / $pen:ident = $r:literal, $g:literal, $b:literal; )*) => {
        $(
            pub const $name: Rgb = ($r, $g, $b);
            pub const $pen: &str = concat!("\x1b[38;2;", $r, ";", $g, ";", $b, "m");
        )*
    };
}

#[rustfmt::skip]
palette! {
    LEVEL  / LEVEL_PEN  = 120, 170, 220;
    GOLD   / GOLD_PEN   = 218, 165,  32;
    AMBER  / AMBER_PEN  = 255, 215,   0;
    STEEL  / STEEL_PEN  = 176, 196, 222;
    SLATE  / SLATE_PEN  = 119, 136, 153;
    TEAL   / TEAL_PEN   =  72, 209, 204;
    JADE   / JADE_PEN   = 100, 200, 120;
    CORAL  / CORAL_PEN  = 224, 105, 100;
    MAUVE  / MAUVE_PEN  = 151, 125, 191;
    LILAC  / LILAC_PEN  = 180, 140, 255;
    IVORY  / IVORY_PEN  = 246, 238, 218;
    HAZE   / HAZE_PEN   = 139, 154, 171;
    TAUPE  / TAUPE_PEN  = 150, 140, 128;
    ASH    / ASH_PEN    = 118, 112, 104;
    GREY   / GREY_PEN   = 130, 130, 130;
    TRACK  / TRACK_PEN  =  58,  62,  72;
    FLOOR  / FLOOR_PEN  = 112, 120, 134;
}

// Status colors mirror the advantage ramp.
#[rustfmt::skip]
palette! {
    OK    / OK_PEN    =  36, 200, 140;
    WARN  / WARN_PEN  = 220, 187,  80;
    ALARM / ALARM_PEN = 225,  89,  91;
}

// Advantage-gradient waypoints as (L, C, H°).
const WIN_GOLD: (f64, f64, f64) = (0.80, 0.13, 92.0);
const WIN_GREEN: (f64, f64, f64) = (0.76, 0.16, 145.0);
const WIN_DEEP: (f64, f64, f64) = (0.74, 0.155, 162.0);
const LOSS_PEACH: (f64, f64, f64) = (0.78, 0.11, 45.0);
const LOSS_ORANGE: (f64, f64, f64) = (0.72, 0.16, 35.0);
const LOSS_DEEP: (f64, f64, f64) = (0.64, 0.17, 22.0);

/// Advantage gradient for `t` in `[-1, 1]`, from deep loss to deep win.
#[must_use]
pub fn advantage(t: f64) -> Rgb {
    let m = t.abs().min(1.0);
    let near = m < 0.5;

    let (lo, hi) = match (t >= 0.0, near) {
        (true, true) => (WIN_GOLD, WIN_GREEN),
        (true, false) => (WIN_GREEN, WIN_DEEP),
        (false, true) => (LOSS_PEACH, LOSS_ORANGE),
        (false, false) => (LOSS_ORANGE, LOSS_DEEP),
    };

    let frac = if near { m * 2.0 } else { m * 2.0 - 1.0 };
    oklch_lerp(lo, hi, frac)
}

/// ANSI truecolor foreground escape for `c`.
#[must_use]
pub fn ansi_fg((r, g, b): Rgb) -> String { format!("\x1b[38;2;{r};{g};{b}m") }

/// Removes ANSI escapes emitted by this module.
/// Stops at the first unknown or unterminated escape.
#[must_use]
pub fn strip(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;

    while let Some(esc) = rest.find('\x1b') {
        out.push_str(&rest[..esc]);
        rest = &rest[esc..];

        let Some(params) = rest.strip_prefix("\x1b[") else { break };
        let Some(end) = params.find(|c: char| !c.is_ascii_digit() && c != ';') else { break };
        if !matches!(params.as_bytes()[end], b'm' | b'K') {
            break;
        }

        rest = &params[end + 1..];
    }
    out.push_str(rest);
    out
}

/// Perceptually blends two sRGB colors through OkLab. `t` is clamped to `[0, 1]`.
#[must_use]
pub fn mix(a: Rgb, b: Rgb, t: f64) -> Rgb {
    let (la, lb) = (srgb_to_oklab(a), srgb_to_oklab(b));
    let t = t.clamp(0.0, 1.0);
    let lerp = |x: f64, y: f64| x + (y - x) * t;
    oklab_to_srgb(lerp(la.0, lb.0), lerp(la.1, lb.1), lerp(la.2, lb.2))
}

fn oklch_to_srgb(l: f64, c: f64, h: f64) -> Rgb {
    let hr = h.to_radians();
    oklab_to_srgb(l, c * hr.cos(), c * hr.sin())
}

// Waypoint hues never cross the wrap, so linear hue interpolation is sufficient.
fn oklch_lerp(a: (f64, f64, f64), b: (f64, f64, f64), t: f64) -> Rgb {
    let lerp = |x: f64, y: f64| x + (y - x) * t;
    oklch_to_srgb(lerp(a.0, b.0), lerp(a.1, b.1), lerp(a.2, b.2))
}

// Björn Ottosson's OkLab matrices.
fn oklab_to_srgb(l: f64, a: f64, b: f64) -> Rgb {
    let l_ = (l + 0.396_337_777_4 * a + 0.215_803_757_3 * b).powi(3);
    let m_ = (l - 0.105_561_345_8 * a - 0.063_854_172_8 * b).powi(3);
    let s_ = (l - 0.089_484_177_5 * a - 1.291_485_548_0 * b).powi(3);

    let lr = 4.076_741_662_1 * l_ - 3.307_711_591_3 * m_ + 0.230_969_929_2 * s_;
    let lg = -1.268_438_004_6 * l_ + 2.609_757_401_1 * m_ - 0.341_319_396_5 * s_;
    let lb = -0.004_196_086_3 * l_ - 0.703_418_614_7 * m_ + 1.707_614_701_0 * s_;

    (linear_to_srgb8(lr), linear_to_srgb8(lg), linear_to_srgb8(lb))
}

fn srgb_to_oklab(c: Rgb) -> (f64, f64, f64) {
    let (r, g, b) = (srgb8_to_linear(c.0), srgb8_to_linear(c.1), srgb8_to_linear(c.2));
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

fn linear_to_srgb8(x: f64) -> u8 {
    let x = x.clamp(0.0, 1.0);
    let x = if x <= 0.003_130_8 { 12.92 * x } else { 1.055 * x.powf(1.0 / 2.4) - 0.055 };
    (x * 255.0).round() as u8
}

fn srgb8_to_linear(b: u8) -> f64 {
    let x = f64::from(b) / 255.0;
    if x <= 0.040_45 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strips_known_ansi() {
        let colored = format!("{}Gauge:{RESET} 0.739x{CLEAR_TO_EOL}", ansi_fg(GOLD));
        assert_eq!(strip(&colored), "Gauge: 0.739x");
        assert_eq!(strip("nothing to drop"), "nothing to drop");
        assert_eq!(strip(""), "");
    }

    #[test]
    fn preserves_unknown_and_unterminated_ansi() {
        assert_eq!(strip("before\x1b[2Jafter"), "before\x1b[2Jafter");
        assert_eq!(strip("before\x1b[38;2;1;2;3"), "before\x1b[38;2;1;2;3");
    }

    #[test]
    fn status_colors_match_advantage_ramp() {
        assert_eq!(OK, advantage(1.0));
        assert_eq!(WARN, advantage(0.0));
        assert_eq!(ALARM, advantage(-1.0));
    }
}
