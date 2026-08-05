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

/// Hooman counters: 1234 -> "1.2K", 1234567 -> "1.23M".
pub fn human(n: u64) -> String {
    match n {
        1_000_000.. => format!("{:.2}M", n as f64 / 1e6),
        1_000.. => format!("{:.1}K", n as f64 / 1e3),
        _ => n.to_string(),
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
