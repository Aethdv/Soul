//! Transposition table — the search's memory of positions it has already seen.
//!
//! Every node iterative deepening revisits, it hopes to find here: the score, the
//! best move, the depth that score was proven to. A hit can cut a whole subtree,
//! and that is what makes deepening cheap, since each iteration seeds the next.
//!
//! The table is lockless because Lazy SMP has every thread reading and writing it
//! at once. Probe and store take `&self` and touch entries through atomics, so no
//! thread waits on another. A 16-bit verification key catches most hash
//! collisions; the ~1-in-2¹⁶ that slips through can only hand back a wrong score,
//! never a wrong move, because every stored move is re-checked by `is_pseudo_legal`
//! before the search trusts it.
//!
//! Entries sit in three-slot clusters of 32 bytes, two to a 64-byte cache line,
//! so a probe is one line fetch that never straddles two. The three slots give
//! replacement a choice of victim: it favors deeper entries, and qsearch stores
//! tread lightly so a shallow result never evicts a deep one.

use std::{
    arch, mem, slice,
    sync::atomic::{AtomicU8, AtomicU16, Ordering},
    thread,
};

use crate::{
    core::{
        defs::{MATE, MAX_PLY},
        moves::Move,
    },
    hugepages::{HugePages, PageKind},
    numa::NumaTopology,
};

/// TT entry bound flags.
pub const BOUND_NONE: u8 = 0;
pub const BOUND_EXACT: u8 = 1;
pub const BOUND_LOWER: u8 = 2; // Beta cutoff (fail-high)
pub const BOUND_UPPER: u8 = 3; // Alpha cutoff (fail-low)

pub const SCORE_NONE: i32 = 32000;

const _: () = assert!(mem::size_of::<TtEntry>() == 10);
const _: () = assert!(mem::size_of::<Cluster>() == 32);
const _: () = assert!(mem::align_of::<Cluster>() == 32);

/// `quality = depth - gen_diff · AGE_FACTOR`.
/// Higher values evict stale entries faster.
const AGE_FACTOR: i32 = 4;
/// `age` is 5 bits; generation distance is read modulo 32.
const AGE_MASK: u8 = 0x1F;

const CLUSTER_SIZE: usize = 3;

/// Whether a stored score is a usable cutoff in the current window.
///
/// An exact score always is. A lower bound (the position failed high when stored)
/// only proves the truth is at least this high, so it cuts only once it clears
/// beta; an upper bound is the mirror, usable only at or below alpha.
#[inline(always)]
pub fn can_cutoff(bound: u8, score: i32, alpha: i32, beta: i32) -> bool {
    bound == BOUND_EXACT || (bound == BOUND_LOWER && score >= beta) || (bound == BOUND_UPPER && score <= alpha)
}

/// The slot's verification key: the low 16 bits of the Zobrist hash.
/// The cluster index is drawn from the high bits, so the low bits
/// are what's left to tell entries in the same bucket apart.
#[inline(always)]
const fn verification_key(hash: u64) -> u16 {
    hash as u16
}

/// One slot, five `AtomicU16` words. The atomics are what let Lazy SMP threads
/// share the table without locks; 16-bit words keep the entry at 10 bytes and
/// align 2, where a single `AtomicU32` would force align 4 and pad it back out,
/// thinning the table for nothing.
///
/// `key` is the commit point. A store writes the other four words first and
/// publishes `key` last, so any reader that sees a matching key is guaranteed the
/// payload that came with it. A probe that misses on `key` has touched one word,
/// which is the common case and the one worth keeping cheap.
///
/// Concurrency's price is the occasional torn read: a re-store landing fresh words
/// under a key you already matched. The cost is bounded. A stale move is caught by
/// `is_pseudo_legal`, and a stale score is only a bad cutoff, never corruption.
#[repr(C)]
struct TtEntry {
    key: AtomicU16,
    mv: AtomicU16,
    score: AtomicU16,
    eval: AtomicU16,
    /// depth(8) | bound(2) | pv(1) | age(5).
    packed: AtomicU16,
}

/// Sized and aligned to fit inside one 64-byte cache line.
#[repr(C, align(32))]
struct Cluster {
    slots: [TtEntry; CLUSTER_SIZE],
}

/// The decoded view of a [`TtEntry`], used by probe and store.
#[derive(Clone, Copy, Default)]
struct Decoded {
    /// 16-bit verification key (low bits of the Zobrist hash). Collisions that
    /// survive it are caught downstream by `is_pseudo_legal` on the stored move.
    key: u16,
    mv: u16,
    score: i16,
    /// Raw static eval at store time (`SCORE_NONE` when stored in check),
    /// so a hit can reuse it instead of recomputing the full evaluation.
    eval: i16,
    depth: u8,
    bound: u8,
    /// Generation at store time; older generations evict first.
    age: u8,
    pv: u8,
}

impl TtEntry {
    /// The cheap scan read: just the key and the (bound, depth) replacement
    /// weighs. Relaxed is enough, since these scans never read payload off this
    /// load; the paths that do go through `load()`, which carries its own ordering.
    #[inline(always)]
    fn meta(&self) -> (u16, u8, u8) {
        let key = self.key.load(Ordering::Relaxed);
        let packed = self.packed.load(Ordering::Relaxed);

        (key, ((packed >> 8) & 0x3) as u8, (packed & 0xFF) as u8)
    }

    /// The probe's per-slot read. One Acquire load of `key` gates it; the rest is
    /// pulled only on a match, so a miss costs a single atomic load. The Acquire
    /// pairs with the store's Release on `key`, the handshake that makes a matched
    /// key imply visible payload.
    #[inline(always)]
    fn probe_read(&self, key16: u16) -> Option<Decoded> {
        if self.key.load(Ordering::Acquire) != key16 {
            return None;
        }

        let packed = self.packed.load(Ordering::Relaxed);
        let bound = ((packed >> 8) & 0x3) as u8;

        if bound == BOUND_NONE {
            return None;
        }

        Some(Decoded {
            key: key16,
            mv: self.mv.load(Ordering::Relaxed),
            score: self.score.load(Ordering::Relaxed) as i16,
            eval: self.eval.load(Ordering::Relaxed) as i16,
            depth: (packed & 0xFF) as u8,
            bound,
            age: ((packed >> 11) & 0x1F) as u8,
            pv: ((packed >> 10) & 0x1) as u8,
        })
    }

    /// Store generation, for replacement-quality aging.
    #[inline(always)]
    fn age(&self) -> u8 {
        ((self.packed.load(Ordering::Relaxed) >> 11) & 0x1F) as u8
    }

    fn load(&self) -> Decoded {
        let key = self.key.load(Ordering::Acquire);
        let mv = self.mv.load(Ordering::Relaxed);
        let score = self.score.load(Ordering::Relaxed) as i16;
        let eval = self.eval.load(Ordering::Relaxed) as i16;
        let packed = self.packed.load(Ordering::Relaxed);

        Decoded {
            key,
            mv,
            score,
            eval,
            depth: (packed & 0xFF) as u8,
            bound: ((packed >> 8) & 0x3) as u8,
            age: ((packed >> 11) & 0x1F) as u8,
            pv: ((packed >> 10) & 0x1) as u8,
        }
    }

    #[inline(always)]
    fn store(&self, d: Decoded) {
        let packed =
            (d.depth as u16 & 0xFF) | ((d.bound as u16 & 0x3) << 8) | ((d.pv as u16 & 0x1) << 10) | ((d.age as u16 & 0x1F) << 11);

        self.mv.store(d.mv, Ordering::Relaxed);
        self.score.store(d.score as u16, Ordering::Relaxed);
        self.eval.store(d.eval as u16, Ordering::Relaxed);
        self.packed.store(packed, Ordering::Relaxed);
        // key goes last. A reader that Acquire-loads this new key is then
        // guaranteed the four payload writes above, released before it.
        self.key.store(d.key, Ordering::Release);
    }
}

pub struct TranspositionTable {
    clusters: HugePages<Cluster>,
    /// The machine's locality, detected once. Drives the first-touch placement
    /// so a multi-node box spreads the table across its memory controllers.
    numa: NumaTopology,
    /// Bumped once per search. Wraps at 255, and aging reads only its low 5 bits,
    /// so `wrapping_sub` measures generation distance modulo 32.
    pub generation: AtomicU8,
}

impl TranspositionTable {
    /// Allocates a new Transposition Table of the given size in MB.
    pub fn new(size_mb: usize) -> Self {
        let numa = NumaTopology::detect();
        let clusters = Self::alloc(size_mb, &numa);

        Self { clusters, numa, generation: AtomicU8::new(0) }
    }

    pub fn resize(&mut self, size_mb: usize) {
        self.clusters = Self::alloc(size_mb, &self.numa);
    }

    /// The page size backing the table, for the startup `info string`.
    pub fn page_kind(&self) -> PageKind {
        self.clusters.kind()
    }

    /// The cluster count follows from the page-rounded byte size, so a 1GB-page
    /// table holds a few more slots than a 4KB one of the same request.
    ///
    /// One memory domain takes the simple inline pre-fault. Two or more map the
    /// pages without faulting, then first-touch each slice on its own node so
    /// the table spreads across the controllers instead of pinning to one.
    fn alloc(size_mb: usize, numa: &NumaTopology) -> HugePages<Cluster> {
        let bytes = size_mb.max(1) * 1024 * 1024;

        // SAFETY (both paths): a zeroed Cluster is the empty state, every field
        // an AtomicU16(0); first_touch does the zeroing on the multi-node path.
        if numa.num_nodes() > 1 {
            let clusters = unsafe { HugePages::mapped(bytes) };
            first_touch(&clusters, numa);
            clusters
        } else {
            unsafe { HugePages::zeroed(bytes) }
        }
    }

    /// Returns TT occupancy in permille (0–1000).
    pub fn hashfull(&self) -> usize {
        let total = self.clusters.len() * CLUSTER_SIZE;
        let sample = total.min(1000);

        if sample == 0 {
            return 0;
        }

        self.clusters
            .iter()
            .flat_map(|c| c.slots.iter())
            .take(sample)
            .filter(|s| s.meta().1 != BOUND_NONE)
            .count()
            * 1000
            / sample
    }

    /// Reset to an empty table and reset the generation counter. Multi-node re-runs
    /// the distributed first-touch so the cleared pages stay spread across nodes.
    pub fn clear(&self) {
        if self.numa.num_nodes() > 1 {
            first_touch(&self.clusters, &self.numa);
        } else {
            self.clusters.clear();
        }

        self.generation.store(0, Ordering::Relaxed);
    }

    /// Advance the generation counter, once per new position. Entries from earlier
    /// generations don't vanish; they just age, growing easier to evict.
    pub fn new_search(&self) {
        self.generation.fetch_add(1, Ordering::Relaxed);
    }

    /// Pin the calling search thread to its NUMA-assigned L3 domain, when binding
    /// is worthwhile. Keeps the thread's compute on its cores and the per-thread
    /// state it allocates next in a warm L3. A no-op on one domain or one thread,
    /// so a single-threaded run never binds.
    pub fn bind_search_thread(&self, thread_id: usize, threads: usize) {
        if self.numa.should_bind(threads) {
            self.numa.bind_to_domain(self.numa.distribute(threads)[thread_id]);
        }
    }

    /// Pull a cluster's cache line toward the core ahead of the probe. A TT lookup
    /// is a near-random memory access, the kind that stalls the negamax loop on a
    /// miss; issued early, the fetch overlaps the work between here and the read.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let idx = self.index(hash);

        unsafe {
            let ptr = self.clusters.as_ptr().add(idx) as *const i8;
            #[cfg(target_arch = "x86_64")]
            arch::x86_64::_mm_prefetch::<{ arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64, ply: usize) -> Option<(Move, i32, i32, u8, bool, i32)> {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cluster = self.cluster(idx);

        for slot in &cluster.slots {
            if let Some(entry) = slot.probe_read(key16) {
                let score = Self::score_from_tt(entry.score as i32, ply);
                let mv = Move::from_u16(entry.mv);

                return Some((mv, score, entry.depth as i32, entry.bound, entry.pv != 0, entry.eval as i32));
            }
        }
        None
    }

    /// Insert or update this position. The cluster scan takes an empty slot, or the
    /// same position if it's already here, and otherwise evicts the lowest-quality
    /// entry, where quality weighs depth against generation age.
    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut replace = 0;
        let mut worst_quality = i32::MAX;
        let mut is_exact_match = false;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, bound, depth) = slot.meta();

            if bound == BOUND_NONE || key == key16 {
                replace = i;
                is_exact_match = key == key16;
                break;
            }

            let gen_diff = (cur.wrapping_sub(slot.age()) & AGE_MASK) as i32;
            let quality = depth as i32 - gen_diff * AGE_FACTOR;

            if quality < worst_quality {
                worst_quality = quality;
                replace = i;
            }
        }

        let mut store_mv = mv.inner();
        let mut store_pv = pv as u8;

        // Keep the existing move when the new store is a null move on the same
        // position. The pv flag travels with the move it describes; dropping it
        // would erase the position's PV history. Only this path needs the whole
        // slot, so it decodes here.
        if mv.is_null() && is_exact_match {
            let existing = cluster.slots[replace].load();
            store_mv = existing.mv;
            store_pv |= existing.pv;
        }

        cluster.slots[replace].store(Decoded {
            key: key16,
            mv: store_mv,
            score: Self::score_to_tt(score, ply) as i16,
            eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            // 8-bit field covers MAX_DEPTH (246); clamp just guards the cast.
            depth: depth.clamp(0, u8::MAX as i32) as u8,
            bound,
            age: cur,
            pv: store_pv,
        });
    }

    /// Stores a qsearch result (depth 0).
    ///
    /// Qsearch floods the table with shallow entries, so its replacement is timid;
    /// it takes only an empty slot, another depth-0 entry, or one whose quality has
    /// aged to nothing. A fresh deep negamax result is never evicted to make room.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut best_idx: Option<usize> = None;
        let mut best_quality = i32::MAX;
        let mut is_key_match = false;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, bound, depth) = slot.meta();

            if key == key16 || bound == BOUND_NONE || depth == 0 {
                best_idx = Some(i);
                is_key_match = key == key16;
                break;
            }

            let gen_diff = (cur.wrapping_sub(slot.age()) & AGE_MASK) as i32;
            let quality = depth as i32 - gen_diff * AGE_FACTOR;

            if quality <= 0 && quality < best_quality {
                best_quality = quality;
                best_idx = Some(i);
            }
        }

        if let Some(best) = best_idx {
            // Preserve a prior PV bit when overwriting the same position;
            // a qsearch visit would otherwise wipe the flag a previous
            // negamax store left here. Only a key match reads it back.
            let store_pv = if is_key_match { (pv as u8) | cluster.slots[best].load().pv } else { pv as u8 };

            cluster.slots[best].store(Decoded {
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

    /// The cache-line bucket for `idx`.
    #[inline(always)]
    fn cluster(&self, idx: usize) -> &Cluster {
        // SAFETY: idx is index()'s output, a mulhi64 in [0, clusters.len()).
        unsafe { self.clusters.get_unchecked(idx) }
    }

    /// Maps a 64-bit hash to a cluster index.
    #[inline(always)]
    fn index(&self, hash: u64) -> usize {
        // mulhi64: the top 64 bits of hash × len. Lands the hash uniformly
        // in [0, len), the way `hash % len` would, but with a multiply.
        let clusters = self.clusters.len();

        (((hash as u128) * (clusters as u128)) >> 64) as usize
    }

    /// Fold the current ply into a mate score before storing it.
    ///
    /// A mate score is distance-to-mate from this node ("mate in 5 at ply 10").
    /// The table is shared across the tree, so it must hold something node-independent:
    /// distance from the root ("mate in 15"). Store folds the ply in, `score_from_tt`
    /// takes it back out. Ordinary scores pass through untouched.
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

    /// Unfold a stored mate score back to a node-relative distance:
    /// the inverse of `score_to_tt`, subtracting the ply that store folded in.
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

/// First-touch the cluster region in parallel, each NUMA node's thread zeroing the
/// slice bound to it. First-touch decides a page's home node, so this is what spreads
/// the table across the memory controllers instead of leaving it all on one.
///
/// Only ever runs with the engine idle (allocation, or `ucinewgame`), so building
/// an exclusive `&mut` over the shared region is sound: no searcher is reading it,
/// and the chunks the threads take are disjoint.
fn first_touch(clusters: &HugePages<Cluster>, numa: &NumaTopology) {
    let nodes = numa.num_nodes();
    let len = clusters.len();

    // SAFETY: idle precondition (above); the region maps len clusters.
    let region = unsafe { slice::from_raw_parts_mut(clusters.as_ptr() as *mut Cluster, len) };

    thread::scope(|scope| {
        for (node, chunk) in region.chunks_mut(len.div_ceil(nodes)).enumerate() {
            scope.spawn(move || {
                numa.bind_to_node(node);
                // SAFETY: chunk is this thread's disjoint slice; a zeroed Cluster is
                // the empty state, so writing zeros first-touches it into a valid one.
                unsafe { chunk.as_mut_ptr().write_bytes(0, chunk.len()) };
            });
        }
    });
}
