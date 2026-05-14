//! Transposition table — size-bounded hash table for position memoization.
//!
//! # Notes
//!
//! Lockless single-threaded design: probe and store take `&self` and
//! mutate entries through raw pointers. The 64-bit full-key match prevents
//! hash collisions from corrupting position data. Replacement is depth-
//! preferred with exact-position upgrades; qsearch stores are conservative
//! to avoid evicting deeper negamax entries.
//!
//! Three-entry clusters (48 bytes per cluster): each hash index maps to
//! a 3-slot bucket, giving the replacement formula three candidates to
//! select from instead of one.

use std::{
    arch, mem, ptr,
    sync::atomic::{AtomicU8, Ordering},
};

use crate::core::{
    defs::{MATE, MAX_PLY},
    moves::Move,
};

/// TT entry bound flags.
pub const BOUND_NONE: u8 = 0;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2; // Beta cutoff (fail-high)
pub const BOUND_UPPER: u8 = 3; // Alpha cutoff (fail-low)

pub const SCORE_NONE: i32 = 32000;

const _: () = assert!(mem::size_of::<TtEntry>() == 16);

/// `quality = depth - gen_diff * AGE_FACTOR`.
/// Higher values evict stale entries faster.
/// Sirius: 2, Obsidian: 8
const AGE_FACTOR: i32 = 4;

const CLUSTER_SIZE: usize = 3;

/// Can we use this TT score as a cutoff given the current window?
#[inline(always)]
pub fn can_cutoff(bound: u8, score: i32, alpha: i32, beta: i32) -> bool {
    bound == BOUND_EXACT || (bound == BOUND_LOWER && score >= beta) || (bound == BOUND_UPPER && score <= alpha)
}

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TtEntry {
    /// Full 64-bit Zobrist key to mathematically prevent hash collisions
    pub key: u64,
    /// Best move found in this position
    pub mv: u16,
    pub score: i16,
    pub depth: u8,
    pub bound: u8,
    /// Generation at store time — prior-generation entries evict first.
    pub age: u8,
    pub pv: u8,
}

pub struct TranspositionTable {
    entries: Box<[TtEntry]>,
    /// Monotonically increments on every new search (position change).
    /// Wraps at 255; `wrapping_sub` handles roll-over correctly.
    pub generation: AtomicU8,
}

// SAFETY: The TT bypasses Rust's safety borrow rules via raw pointers during probe and store,
// which mutate fields through a &self shared reference. This is mathematically safe for single
// threaded execution.
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Allocates a new Transposition Table of the given size in MB.
    pub fn new(size_mb: usize) -> Self {
        let mut tt = Self { entries: Box::new([]), generation: AtomicU8::new(0) };
        tt.resize(size_mb);
        tt
    }

    pub fn resize(&mut self, size_mb: usize) {
        let bytes = size_mb.max(1) * 1024 * 1024;
        let count = (bytes / mem::size_of::<TtEntry>()).max(CLUSTER_SIZE);
        let clusters = count / CLUSTER_SIZE;
        self.entries = vec![TtEntry::default(); clusters * CLUSTER_SIZE].into_boxed_slice();
    }

    /// Returns TT occupancy in permille (0–1000).
    pub fn hashfull(&self) -> usize {
        let sample = self.entries.len().min(1000);
        self.entries[..sample].iter().filter(|e| e.bound != BOUND_NONE).count() * 1000 / sample.max(1)
    }

    /// Zero every entry and reset the generation counter.
    pub fn clear(&self) {
        if self.entries.is_empty() {
            return;
        }
        let ptr = self.entries.as_ptr() as *mut TtEntry;
        unsafe {
            ptr::write_bytes(ptr, 0, self.entries.len());
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Advance generation counter. Call on each new position —
    /// stale entries become easier to evict without being destroyed.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        unsafe {
            let ptr = self.entries.as_ptr().add(idx) as *const i8;
            #[cfg(target_arch = "x86_64")]
            arch::x86_64::_mm_prefetch::<{ arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64, ply: usize) -> Option<(Move, i32, i32, u8, bool)> {
        if self.entries.is_empty() {
            return None;
        }

        let idx = self.index(hash);

        for i in 0..CLUSTER_SIZE {
            let entry = unsafe { &*self.entries.as_ptr().add(idx + i) };
            // 64-bit exact match completely prevents hash-collision-induced memory corruption.
            if entry.key == hash && entry.bound != BOUND_NONE {
                let score = Self::score_from_tt(entry.score as i32, ply);
                let mv = Move::from_u16(entry.mv);

                return Some((mv, score, entry.depth as i32, entry.bound, entry.pv != 0));
            }
        }
        None
    }

    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: u8, pv: bool) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        let cur = self.generation.load(Ordering::Relaxed);
        let mut best_idx = idx;
        let mut best_quality = i32::MAX;

        for i in 0..CLUSTER_SIZE {
            let entry = unsafe { &*self.entries.as_ptr().add(idx + i) };

            if entry.bound == BOUND_NONE || entry.key == hash {
                best_idx = idx + i;
                break;
            }

            let gen_diff = cur.wrapping_sub(entry.age) as i32;
            let quality = entry.depth as i32 - gen_diff * AGE_FACTOR;
            if quality < best_quality {
                best_quality = quality;
                best_idx = idx + i;
            }
        }

        // SAFETY: Obtaining mutable reference to a slot via an immutable &self.
        // Standard high-performance lockless TT approach.
        let entry = unsafe { &mut *(self.entries.as_ptr().add(best_idx) as *mut TtEntry) };
        let is_exact_match = entry.key == hash;
        entry.key = hash;

        let mut store_mv = mv.inner();
        // Preserve an existing highly-valued move if we hit a beta-cutoff
        // and the new bound provided an empty (null) move.
        if mv.is_null() && is_exact_match {
            store_mv = entry.mv;
        }

        entry.mv = store_mv;
        entry.score = Self::score_to_tt(score, ply) as i16;
        entry.depth = depth as u8;
        entry.bound = bound;
        entry.age = cur;
        entry.pv = pv as u8;
    }

    /// Stores a qsearch result (depth = 0).
    ///
    /// Conservative replacement:
    /// only overwrites empty slots, existing depth-0 entries, or stale entries
    /// whose aged quality has dropped to zero — never evicts a fresh deep entry.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: u8, pv: bool) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        let cur = self.generation.load(Ordering::Relaxed);
        let mut best_idx: Option<usize> = None;
        let mut best_quality = i32::MAX;

        for i in 0..CLUSTER_SIZE {
            let entry = unsafe { &*self.entries.as_ptr().add(idx + i) };

            if entry.key == hash || entry.bound == BOUND_NONE || entry.depth == 0 {
                best_idx = Some(idx + i);
                break;
            }

            let gen_diff = cur.wrapping_sub(entry.age) as i32;
            let quality = entry.depth as i32 - gen_diff * AGE_FACTOR;
            if quality <= 0 && quality < best_quality {
                best_quality = quality;
                best_idx = Some(idx + i);
            }
        }

        if let Some(best) = best_idx {
            // SAFETY: Obtaining mutable reference to a slot via an immutable &self.
            // Standard high-performance lockless TT approach.
            let entry = unsafe { &mut *(self.entries.as_ptr().add(best) as *mut TtEntry) };
            entry.key = hash;
            entry.mv = mv.inner();
            entry.score = Self::score_to_tt(score, ply) as i16;
            entry.depth = 0;
            entry.bound = bound;
            entry.age = cur;
            entry.pv = pv as u8;
        }
    }

    /// Maps 64-bit hash to the first index of a 3-entry cluster.
    #[inline(always)]
    fn index(&self, hash: u64) -> usize {
        // High 64 bits of a 128-bit multiplication (mulhi64).
        // Uniformly maps a 64-bit hash into the range [0, clusters).
        let clusters = self.entries.len() / CLUSTER_SIZE;
        (((hash as u128) * (clusters as u128)) >> 64) as usize * CLUSTER_SIZE
    }

    /// Adjusts mate scores when storing into the TT.
    /// "Mate in 5 found at ply 10" is absolute (mate in 15 from root).
    #[inline(always)]
    fn score_to_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE - MAX_PLY as i32 {
            score + ply as i32
        } else if score <= -MATE + MAX_PLY as i32 {
            score - ply as i32
        } else {
            score
        }
    }

    /// Adjusts mate scores when retrieving from the TT.
    /// Absolute mates (mate in 15 from root) are converted back to relative (mate in 5 at ply 10).
    #[inline(always)]
    fn score_from_tt(score: i32, ply: usize) -> i32 {
        if score >= MATE - MAX_PLY as i32 {
            score - ply as i32
        } else if score <= -MATE + MAX_PLY as i32 {
            score + ply as i32
        } else {
            score
        }
    }
}
