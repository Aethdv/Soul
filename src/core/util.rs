//! Low-level memory and concurrency utilities.
//!
//! - `BatchedAtomicCounter`: Buffers increments locally before flushing to an `AtomicU64`.
//! - `Align64`: Forces 64-byte alignment on arbitrary types to prevent false sharing.

use std::{
    ops::{Deref, DerefMut},
    sync::atomic::{AtomicU64, Ordering},
};

/// Batched atomic counter that reduces contention by buffering increments locally.
///
/// Flushes to the global counter every `GRANULARITY` increments,
/// trading precision for reduced atomic operation overhead under high contention.
#[derive(Debug)]
pub struct BatchedAtomicCounter<'a> {
    pending: u64,
    global: &'a AtomicU64,
    contributed: u64,
}

impl<'a> BatchedAtomicCounter<'a> {
    const GRANULARITY: u64 = 1024;

    #[inline]
    pub const fn new(global: &'a AtomicU64) -> Self {
        Self { pending: 0, global, contributed: 0 }
    }

    /// Increment the counter by 1.
    /// Flushes to global every `GRANULARITY` ops.
    #[inline]
    pub fn increment(&mut self) {
        self.pending += 1;
        if self.pending >= Self::GRANULARITY {
            self.global.fetch_add(self.pending, Ordering::Relaxed);
            self.contributed += self.pending;
            self.pending = 0;
        }
    }

    /// Read the global counter value (approximate if unflushed buffers exist).
    #[inline]
    pub fn get_global(&self) -> u64 {
        self.global.load(Ordering::Relaxed) + self.pending
    }

    /// Read this counter's local contributions (pending + flushed).
    #[inline]
    pub const fn get_local(&self) -> u64 {
        self.contributed + self.pending
    }

    /// Flush any buffered increments to the global counter.
    #[inline]
    pub fn flush(&mut self) {
        if self.pending > 0 {
            self.global.fetch_add(self.pending, Ordering::Relaxed);
            self.contributed += self.pending;
            self.pending = 0;
        }
    }

    /// Reset both local and global counters.
    #[inline]
    pub fn reset(&mut self) {
        self.pending = 0;
        self.global.store(0, Ordering::Relaxed);
        self.contributed = 0;
    }
}

impl Drop for BatchedAtomicCounter<'_> {
    /// Automatically flush remaining buffered increments when dropped.
    fn drop(&mut self) {
        if self.pending > 0 {
            self.global.fetch_add(self.pending, Ordering::Relaxed);
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_counter_basic() {
        let global = AtomicU64::new(0);
        let mut counter = BatchedAtomicCounter::new(&global);

        for _ in 0..1000 {
            counter.increment();
        }
        assert_eq!(counter.get_local(), 1000);
        assert_eq!(global.load(Ordering::Relaxed), 0); // 1000 < 1024, no flush triggered yet

        counter.increment(); // 1001 total
        counter.flush();
        assert_eq!(global.load(Ordering::Relaxed), 1001);
    }

    #[test]
    fn align64_size() {
        assert!(std::mem::align_of::<Align64<u8>>() == 64);
        assert!(std::mem::size_of::<Align64<u8>>() == 64);
    }
}
