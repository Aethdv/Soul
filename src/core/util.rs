//! Low-level memory and concurrency utilities.
//!
//! - `Align64`: Forces 64-byte alignment on arbitrary types to prevent false sharing.

use std::ops::{Deref, DerefMut};

/// Cache-line aligned wrapper (64 bytes).
///
/// Prevents false sharing when data is accessed from multiple threads.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[repr(C, align(64))]
pub struct Align64<T>(pub T);

impl<T> Align64<T> {
    #[inline]
    pub const fn new(val: T) -> Self {
        Self(val)
    }
}

impl<T> Deref for Align64<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> DerefMut for Align64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// Division, 0.0 for an empty denominator instead of NaN or infinity.
#[inline]
#[must_use]
pub fn safe_div(num: f64, den: f64) -> f64 {
    if den > 0.0 { num / den } else { 0.0 }
}

/// `part` as a percentage of `whole`, 0.0 when nothing was counted.
#[inline]
#[must_use]
pub fn pct(part: u64, whole: u64) -> f64 {
    safe_div(part as f64, whole as f64) * 100.0
}

/// Formats a number with comma separators: 1234567 -> "1,234,567".
pub fn format_comma(n: u64) -> String {
    let raw = n.to_string();
    let len = raw.len();
    let comma_count = len.saturating_sub(1) / 3;

    let mut buf = String::with_capacity(len + comma_count);
    for (i, b) in raw.bytes().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            buf.push(',');
        }
        buf.push(b as char);
    }
    buf
}

/// Hooman counters: 1234 -> "1.23K", 1234567 -> "1.23M".
///
/// SI prefixes, so nodes read the way nps already does: G rather than B past a
/// billion.
pub fn human(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.2}K", n as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.2}M", n as f64 / 1e6),
        _ => format!("{:.2}G", n as f64 / 1e9),
    }
}

/// A duration in the largest two units that fit, milliseconds below a second.
///
/// One shape for a search clock and a datagen ETA: the first needs `26ms`, the
/// second needs `2h 5m 3s`.
pub fn format_duration(ms: u64) -> String {
    let (secs, mins) = (ms / 1_000 % 60, ms / 60_000 % 60);
    let (hours, days) = (ms / 3_600_000 % 24, ms / 86_400_000);

    match (days, hours, ms) {
        (0, 0, ..1_000) => format!("{ms}ms"),
        (0, 0, _) => format!("{mins}m {secs}s"),
        (0, ..) => format!("{hours}h {mins}m {secs}s"),
        _ => format!("{days}d {hours}h {mins}m"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn align64_size() {
        assert!(std::mem::align_of::<Align64<u8>>() == 64);
        assert!(std::mem::size_of::<Align64<u8>>() == 64);
    }
}
