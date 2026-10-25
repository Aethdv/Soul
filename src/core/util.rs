//! Cache-line alignment, bounded hash indexing and number formatting.

#[cfg(feature = "datagen")] use std::ops::{Deref, DerefMut};

#[cfg(feature = "datagen")]
#[derive(Debug, Clone, PartialEq, Eq, Default)]
#[repr(C, align(64))]
pub(crate) struct Align64<T>(pub(crate) T);

#[cfg(feature = "datagen")]
impl<T> Align64<T> {
    #[inline]
    pub(crate) const fn new(val: T) -> Self { Self(val) }
}

#[cfg(feature = "datagen")]
impl<T> Deref for Align64<T> {
    type Target = T;
    #[inline(always)]
    fn deref(&self) -> &Self::Target { &self.0 }
}

#[cfg(feature = "datagen")]
impl<T> DerefMut for Align64<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target { &mut self.0 }
}

#[inline(always)]
pub(crate) fn mulhi64(value: u64, bound: usize) -> usize { ((value as u128 * bound as u128) >> 64) as usize }

/// Returns zero unless `den` is positive.
#[inline]
#[must_use]
pub(crate) fn ratio(num: f64, den: f64) -> f64 { if den > 0.0 { num / den } else { 0.0 } }

#[inline]
#[must_use]
pub fn pct(part: u64, whole: u64) -> f64 { ratio(part as f64, whole as f64) * 100.0 }

/// 1234567 → "1,234,567".
pub fn fmt_comma(n: u64) -> String {
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

/// 1234 → "1.23K", 1234567 → "1.23M"; uses `G` for billions.
pub(crate) fn fmt_count(n: u64) -> String {
    match n {
        0..1_000 => n.to_string(),
        1_000..1_000_000 => format!("{:.2}K", n as f64 / 1e3),
        1_000_000..1_000_000_000 => format!("{:.2}M", n as f64 / 1e6),
        _ => format!("{:.2}G", n as f64 / 1e9),
    }
}

/// Uses milliseconds below one second, then the largest fitting units:
/// `26ms`, `2m 5s`, `2h 5m 3s`, `3d 2h 5m`.
pub(crate) fn fmt_duration(ms: u64) -> String {
    let (secs, mins) = (ms / 1_000 % 60, ms / 60_000 % 60);
    let (hours, days) = (ms / 3_600_000 % 24, ms / 86_400_000);

    match (days, hours, ms) {
        (0, 0, ..1_000) => format!("{ms}ms"),
        (0, 0, _) => format!("{mins}m {secs}s"),
        (0, ..) => format!("{hours}h {mins}m {secs}s"),
        _ => format!("{days}d {hours}h {mins}m"),
    }
}

#[cfg(all(test, feature = "datagen"))]
mod tests {
    use super::*;

    #[test]
    fn align64_size() {
        assert!(std::mem::align_of::<Align64<u8>>() == 64);
        assert!(std::mem::size_of::<Align64<u8>>() == 64);
    }
}
