//! Transposition table — size-bounded hash table for position memoization.
//!
//! # Notes
//!
//! Lockless design: probe and store take `&self` and mutate entries
//! through per-word atomics. A 16-bit verification key guards
//! against hash collisions; the rare aliasing that slips through (~1 in 2¹⁶
//! within a probed cluster) can only return a stale score, never a corrupting
//! move, because every stored move is re-validated by `is_pseudo_legal` before
//! use. Replacement is depth-preferred with exact-position upgrades; qsearch
//! stores are conservative to avoid evicting deeper negamax entries.
//!
//! Three-entry clusters (36 bytes per cluster): each hash index maps to
//! a 3-slot bucket, giving the replacement formula three candidates to
//! select from instead of one.

use std::{
    arch, mem, ptr,
    sync::atomic::{AtomicU8, AtomicU32, Ordering},
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

const _: () = assert!(mem::size_of::<TtEntry>() == 12);

/// `quality = depth - gen_diff · AGE_FACTOR`.
/// Higher values evict stale entries faster.
/// Sirius: 2, Obsidian: 8
const AGE_FACTOR: i32 = 4;

const CLUSTER_SIZE: usize = 3;

/// Can we use this TT score as a cutoff given the current window?
#[inline(always)]
pub fn can_cutoff(bound: u8, score: i32, alpha: i32, beta: i32) -> bool {
    bound == BOUND_EXACT || (bound == BOUND_LOWER && score >= beta) || (bound == BOUND_UPPER && score <= alpha)
}

/// The slot's verification key: the low 16 bits of the Zobrist hash. The
/// cluster index already consumes the high bits, so these are the discriminating
/// bits left over within a bucket.
#[inline(always)]
const fn verification_key(hash: u64) -> u16 {
    hash as u16
}

/// A cluster slot, packed into three `AtomicU32`s for Lazy SMP.
/// An `AtomicU64` would force 8-byte alignment and pad the 12-byte slot to 16,
/// thinning the table.
///
/// The words are ordered by how often the cluster scan reads them, not by type.
/// Every scan touches only `a` (key+bound+depth); `b` adds age for replacement
/// quality; `c` is pure payload, read only after a key match. So a probing miss
/// costs one atomic load, not three — and atomic loads can't be elided, so the
/// layout is what keeps the hottest loop cheap.
///
/// A torn read across words mixes metadata onto a matching move;
/// `is_pseudo_legal` catches stale moves downstream. Worst case is a stale cutoff, never corruption.
#[repr(C)]
struct TtEntry {
    a: AtomicU32, // key(16)   | bound(8) | depth(8)
    b: AtomicU32, // mv(16)    | age(8)   | pv(8)
    c: AtomicU32, // score(16) | eval(16)
}

/// The decoded view of a [`TtEntry`], used by probe and store.
#[derive(Clone, Copy, Default)]
struct Decoded {
    /// 16-bit verification key (low bits of the Zobrist hash). Collisions that
    /// survive it are caught downstream by `is_pseudo_legal` on the stored move.
    key: u16,
    mv: u16,
    score: i16,
    /// Raw static eval at store time (`SCORE_NONE` when stored in check), so a
    /// hit can reuse it instead of recomputing the full evaluation.
    eval: i16,
    depth: u8,
    bound: u8,
    /// Generation at store time — prior-generation entries evict first.
    age: u8,
    pv: u8,
}

impl Default for TtEntry {
    fn default() -> Self {
        Self { a: AtomicU32::new(0), b: AtomicU32::new(0), c: AtomicU32::new(0) }
    }
}

impl TtEntry {
    /// Cheap scan read; word `a` decoded to (key, bound, depth) — every field
    /// the cluster scan compares. The payload stays packed until a key matches.
    #[inline(always)]
    fn meta(&self) -> (u16, u8, u8) {
        let a = self.a.load(Ordering::Acquire);
        (a as u16, (a >> 16) as u8, (a >> 24) as u8)
    }

    /// Store generation, for replacement-quality aging.
    #[inline(always)]
    fn age(&self) -> u8 {
        (self.b.load(Ordering::Relaxed) >> 16) as u8
    }

    #[inline(always)]
    fn load(&self) -> Decoded {
        let a = self.a.load(Ordering::Acquire);
        let b = self.b.load(Ordering::Relaxed);
        let c = self.c.load(Ordering::Relaxed);

        Decoded {
            key: a as u16,
            bound: (a >> 16) as u8,
            depth: (a >> 24) as u8,
            mv: b as u16,
            age: (b >> 16) as u8,
            pv: (b >> 24) as u8,
            score: c as u16 as i16,
            eval: (c >> 16) as u16 as i16,
        }
    }

    #[inline(always)]
    fn store(&self, d: Decoded) {
        let a = d.key as u32 | (d.bound as u32) << 16 | (d.depth as u32) << 24;
        let b = d.mv as u32 | (d.age as u32) << 16 | (d.pv as u32) << 24;
        let c = d.score as u16 as u32 | (d.eval as u16 as u32) << 16;

        self.c.store(c, Ordering::Relaxed);
        self.b.store(b, Ordering::Relaxed);
        // key published last — an Acquire load on `a` that sees the new key
        // is guaranteed to see the `b` and `c` stores that precede it.
        self.a.store(a, Ordering::Release);
    }
}

pub struct TranspositionTable {
    entries: Box<[TtEntry]>,
    /// Monotonically increments on every new search (position change).
    /// Wraps at 255; `wrapping_sub` handles roll-over correctly.
    pub generation: AtomicU8,
}

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
        let n = clusters * CLUSTER_SIZE;

        self.entries = (0..n).map(|_| TtEntry::default()).collect::<Vec<_>>().into_boxed_slice();
    }

    /// Returns TT occupancy in permille (0–1000).
    pub fn hashfull(&self) -> usize {
        let sample = self.entries.len().min(1000);

        self.entries[..sample].iter().filter(|e| e.meta().1 != BOUND_NONE).count() * 1000 / sample.max(1)
    }

    /// Zero every entry and reset the generation counter.
    pub fn clear(&self) {
        if self.entries.is_empty() {
            return;
        }

        let ptr = self.entries.as_ptr() as *mut TtEntry;

        // SAFETY: Only called by `ucinewgame` when the engine is idle.
        // Bypassing atomics for a bulk memset is safe with no concurrent searchers.
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
    pub fn probe(&self, hash: u64, ply: usize) -> Option<(Move, i32, i32, u8, bool, i32)> {
        if self.entries.is_empty() {
            return None;
        }

        let idx = self.index(hash);
        let key16 = verification_key(hash);

        for i in 0..CLUSTER_SIZE {
            let (key, bound, _) = self.entries[idx + i].meta();
            if key == key16 && bound != BOUND_NONE {
                let entry = self.entries[idx + i].load();
                // A concurrent store that published a new key after our meta()
                // but before load() can leave us with a mismatched key — the
                // Acquire on `a` in load() synchronizes with the store's Release,
                // and this re-check discards the torn hit.
                if entry.key != key16 {
                    continue;
                }
                let score = Self::score_from_tt(entry.score as i32, ply);
                let mv = Move::from_u16(entry.mv);

                return Some((mv, score, entry.depth as i32, entry.bound, entry.pv != 0, entry.eval as i32));
            }
        }
        None
    }

    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed);

        let mut replace_idx = idx;
        let mut worst_quality = i32::MAX;
        let mut is_exact_match = false;

        for i in 0..CLUSTER_SIZE {
            let (key, bound, depth) = self.entries[idx + i].meta();

            if bound == BOUND_NONE || key == key16 {
                replace_idx = idx + i;
                is_exact_match = key == key16;
                break;
            }

            let gen_diff = cur.wrapping_sub(self.entries[idx + i].age()) as i32;
            let quality = depth as i32 - gen_diff * AGE_FACTOR;

            if quality < worst_quality {
                worst_quality = quality;
                replace_idx = idx + i;
            }
        }

        let mut store_mv = mv.inner();
        let mut store_pv = pv as u8;
        // Preserve an existing highly-valued move if the new store provides
        // `Move::null()` and the hash matches. The pv flag rides with the
        // move it describes — losing it would erase the position's
        // PV-history context. Only this path needs the full slot, so decode it here.
        if mv.is_null() && is_exact_match {
            let existing = self.entries[replace_idx].load();
            store_mv = existing.mv;
            store_pv |= existing.pv;
        }

        self.entries[replace_idx].store(Decoded {
            key: key16,
            mv: store_mv,
            score: Self::score_to_tt(score, ply) as i16,
            eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            depth: depth as u8,
            bound,
            age: cur,
            pv: store_pv,
        });
    }

    /// Stores a qsearch result (depth = 0).
    ///
    /// Conservative replacement:
    /// only overwrites empty slots, existing depth-0 entries, or stale entries
    /// whose aged quality has dropped to zero — never evicts a fresh deep entry.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        if self.entries.is_empty() {
            return;
        }

        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed);
        let mut best_idx: Option<usize> = None;
        let mut best_quality = i32::MAX;
        let mut is_key_match = false;

        for i in 0..CLUSTER_SIZE {
            let (key, bound, depth) = self.entries[idx + i].meta();

            if key == key16 || bound == BOUND_NONE || depth == 0 {
                best_idx = Some(idx + i);
                is_key_match = key == key16;
                break;
            }

            let gen_diff = cur.wrapping_sub(self.entries[idx + i].age()) as i32;
            let quality = depth as i32 - gen_diff * AGE_FACTOR;

            if quality <= 0 && quality < best_quality {
                best_quality = quality;
                best_idx = Some(idx + i);
            }
        }

        if let Some(best) = best_idx {
            // Preserve any prior PV-history bit when overwriting the same
            // position — qs visits would otherwise erase the propagated flag
            // stamped by a previous negamax store. Only a key match reads it back.
            let store_pv = if is_key_match { (pv as u8) | self.entries[best].load().pv } else { pv as u8 };

            self.entries[best].store(Decoded {
                key: key16,
                mv: mv.inner(),
                score: Self::score_to_tt(score, ply) as i16,
                eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                depth: 0,
                bound,
                age: cur,
                pv: store_pv,
            });
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
