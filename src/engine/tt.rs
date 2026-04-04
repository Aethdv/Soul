use crate::core::{
    defs::{MATE, MAX_PLY},
    moves::Move,
};

/// TT entry bound flags.
pub const BOUND_NONE: u8 = 0;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2; // Beta cutoff (fail-high)
pub const BOUND_UPPER: u8 = 3; // Alpha cutoff (fail-low)

/// Can we use this TT score as a cutoff given the current window?
#[inline(always)]
pub fn can_cutoff(bound: u8, score: i32, alpha: i32, beta: i32) -> bool {
    bound == BOUND_EXACT
        || (bound == BOUND_LOWER && score >= beta)
        || (bound == BOUND_UPPER && score <= alpha)
}

/// Uninitialized TT eval/score.
pub const SCORE_NONE: i32 = 32000;

#[derive(Clone, Copy, Default)]
#[repr(C)]
pub struct TtEntry {
    /// Full 64-bit Zobrist key to mathematically prevent hash collisions
    pub key:   u64,
    /// Best move found in this position
    pub mv:    u16,
    /// Search score
    pub score: i16,
    pub depth: u8,
    pub bound: u8,
    pub _pad:  u16,
}

const _: () = assert!(std::mem::size_of::<TtEntry>() == 16);

pub struct TranspositionTable {
    entries: Box<[TtEntry]>,
}

// SAFETY: The TT bypasses Rust's safety borrow rules via raw pointers during probe and store,
// which mutate fields through a &self shared reference. This is mathematically safe for single
// threaded execution.
unsafe impl Send for TranspositionTable {}
unsafe impl Sync for TranspositionTable {}

impl TranspositionTable {
    /// Allocates a new Transposition Table of the given size in MB.
    pub fn new(size_mb: usize) -> Self {
        let mut tt = Self {
            entries: Box::new([]),
        };
        tt.resize(size_mb);
        tt
    }

    /// Resizes the table to the specified size in MB.
    pub fn resize(&mut self, size_mb: usize) {
        let bytes = size_mb.max(1) * 1024 * 1024;
        let count = (bytes / std::mem::size_of::<TtEntry>()).max(1);
        self.entries = vec![TtEntry::default(); count].into_boxed_slice();
    }

    /// Returns TT occupancy in permille (0–1000) by sampling the first 1000 entries.
    pub fn hashfull(&self) -> usize {
        let sample = self.entries.len().min(1000);
        self.entries[..sample]
            .iter()
            .filter(|e| e.bound != BOUND_NONE)
            .count()
            * 1000
            / sample.max(1)
    }

    /// Clears the table cleanly. Safe because of boxed lifetime.
    pub fn clear(&self) {
        if self.entries.is_empty() {
            return;
        }
        let ptr = self.entries.as_ptr() as *mut TtEntry;
        unsafe {
            std::ptr::write_bytes(ptr, 0, self.entries.len());
        }
    }

    /// Prefetches the memory for the given Zobrist key into the CPU cache.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        unsafe {
            let ptr = self.entries.as_ptr().add(idx) as *const i8;
            #[cfg(target_arch = "x86_64")]
            core::arch::x86_64::_mm_prefetch::<{ core::arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    /// Retrieves an entry for the given position if it exists.
    #[inline(always)]
    pub fn probe(&self, hash: u64, ply: usize) -> Option<(Move, i32, i32, u8)> {
        if self.entries.is_empty() {
            return None;
        }

        let idx = self.index(hash);
        let entry = unsafe { &*self.entries.as_ptr().add(idx) };

        // 64-bit exact match completely prevents hash-collision-induced memory corruption.
        if entry.key == hash && entry.bound != BOUND_NONE {
            let score = Self::score_from_tt(entry.score as i32, ply);
            let mv = Move::from_u16(entry.mv);
            return Some((mv, score, entry.depth as i32, entry.bound));
        }

        None
    }

    /// Stores a new entry in the table.
    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: u8) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        // SAFETY: Obtaining mutable reference to a slot via an immutable &self.
        // Standard high-performance lockless TT approach.
        let entry = unsafe { &mut *(self.entries.as_ptr().add(idx) as *mut TtEntry) };

        // Fundamental Replacement Scheme: Depth-Preferred + Exact Position Upgrade
        // We overwrite if the new entry searched deeper, or if it's the exact same position
        // and we simply want to upgrade the bound or move.
        if entry.key != hash || depth >= entry.depth as i32 || entry.bound == BOUND_NONE {
            let is_exact_match = entry.key == hash;
            entry.key = hash;

            // Preserve an existing highly-valued move if we hit a beta-cutoff and
            // the new bound provided an empty (null) move.
            let mut store_mv = mv.inner();
            if mv.is_null() && is_exact_match {
                store_mv = entry.mv;
            }

            entry.mv = store_mv;
            entry.score = Self::score_to_tt(score, ply) as i16;
            entry.depth = depth as u8;
            entry.bound = bound;
        }
    }

    /// Stores a qsearch result (depth = 0). Conservative replacement policy:
    /// only overwrites empty slots or existing depth-0 entries so that deeper
    /// negamax entries are never evicted by high-volume qsearch stores.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: u8) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        // SAFETY: Obtaining mutable reference to a slot via an immutable &self.
        // Standard high-performance lockless TT approach.
        let entry = unsafe { &mut *(self.entries.as_ptr().add(idx) as *mut TtEntry) };

        // Only overwrite empty slots or existing depth-0 (qsearch) entries.
        // Depth-preferred negamax entries must not be evicted.
        if entry.bound == BOUND_NONE || entry.depth == 0 {
            entry.key = hash;
            entry.mv = mv.inner();
            entry.score = Self::score_to_tt(score, ply) as i16;
            entry.depth = 0;
            entry.bound = bound;
        }
    }

    /// Maps 64-bit hash to [0, count).
    #[inline(always)]
    fn index(&self, hash: u64) -> usize {
        // High 64 bits of a 128-bit multiplication (mulhi64).
        // Uniformly maps a 64-bit hash into the range [0, count).
        (((hash as u128) * (self.entries.len() as u128)) >> 64) as usize
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
