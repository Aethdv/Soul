//! Transposition table: the search's memory of positions it has already seen.
//!
//! When iterative deepening revisits a node, it hopes to find it already here. A hit
//! can cut a whole subtree, and since each iteration seeds the next, deepening stays
//! cheap.
//!
//! The table is lockless because Lazy SMP has every thread reading and writing it
//! at once. Probe and store take `&self` and touch entries through atomics, so no
//! thread waits on another. A 16-bit verification key catches most hash
//! collisions; the roughly one in 2¹⁶ that slips through can hand back a wrong
//! score, or a move belonging to another position, which is why every stored move
//! is re-checked by `is_pseudo_legal` before the search plays it.
//!
//! Entries sit in three-slot clusters, so replacement has a choice of victim.

use std::{
    arch, mem, slice,
    sync::atomic::{AtomicU8, AtomicU16, Ordering},
    thread,
};

use crate::{
    core::{
        defs::{score_from_tt, score_to_tt},
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

/// No eval stored, which is what an in-check store leaves behind. Sits above MATE,
/// so it can never be mistaken for a score, and inside i16 so it survives the slot.
pub const SCORE_NONE: i32 = 32000;

const _: () = assert!(mem::size_of::<TtEntry>() == 10);
const _: () = assert!(mem::size_of::<Cluster>() == 32);
const _: () = assert!(mem::align_of::<Cluster>() == 32);

/// Higher values evict stale entries faster.
const AGE_FACTOR: i32 = 4;
/// `age` is 5 bits; generation distance is read modulo 32.
const AGE_MASK: u8 = 0x1F;

const CLUSTER_SIZE: usize = 3;

/// An exact score always is. A lower bound (the position failed high when stored)
/// only proves the truth is at least this high, so it cuts only once it clears
/// beta; an upper bound is the mirror, usable only at or below alpha.
#[inline(always)]
pub fn can_cutoff(bound: u8, score: i32, alpha: i32, beta: i32) -> bool {
    bound == BOUND_EXACT || (bound == BOUND_LOWER && score >= beta) || (bound == BOUND_UPPER && score <= alpha)
}

/// The static eval clamped into the range a searched score proves.
///
/// The score came from a search and the eval is a guess, so where the two disagree the
/// eval gives way. A lower bound puts a floor under the truth, so it can only lift an
/// eval sitting below it; an upper bound is the ceiling and can only pull one down. An
/// eval already inside the proven range stands, since the bound contradicts nothing.
#[inline(always)]
pub fn clamp_to_bound(bound: u8, score: i32, eval: i32) -> i32 {
    match bound {
        BOUND_EXACT => score,
        BOUND_LOWER => eval.max(score),
        BOUND_UPPER => eval.min(score),
        _ => eval,
    }
}

/// One slot, five `AtomicU16` words. The atomics let Lazy SMP threads share
/// the table without locks; 16-bit words keep the entry at 10 bytes and align 2,
/// where a single `AtomicU32` would force align 4 and pad it back out,
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

pub struct TranspositionTable {
    clusters: HugePages<Cluster>,
    /// Bumped once per search. Wraps at 255, and aging reads only its low 5 bits,
    /// so `wrapping_sub` measures generation distance modulo 32.
    pub generation: AtomicU8,
    /// The machine's locality, detected once. Drives the first-touch placement
    /// so a multi-node box spreads the table across its memory controllers.
    numa: NumaTopology,
}

/// Half a cache line, aligned so a cluster never straddles two.
#[repr(C, align(32))]
struct Cluster {
    slots: [TtEntry; CLUSTER_SIZE],
}

/// What a probe hands back, already unfolded into search units.
#[derive(Clone, Copy)]
pub struct TtHit {
    pub mv: Move,
    pub score: i32,
    pub depth: i32,
    pub bound: u8,
    pub pv: bool,
    /// Raw static eval at store time, `SCORE_NONE` when the position was in check.
    pub eval: i32,
}

#[derive(Clone, Copy, Default)]
struct Payload {
    key: u16,
    mv: u16,
    score: i16,
    eval: i16,
    depth: u8,
    bound: u8,
    age: u8,
    pv: u8,
}

/// The slot's verification key: the low 16 bits of the Zobrist hash.
/// The cluster index is drawn from the high bits, so the low bits
/// are what's left to tell entries in the same bucket apart.
#[inline(always)]
const fn verification_key(hash: u64) -> u16 { hash as u16 }

const fn pack(depth: u8, bound: u8, pv: u8, age: u8) -> u16 {
    depth as u16 | ((bound as u16 & 0x3) << 8) | ((pv as u16 & 0x1) << 10) | ((age as u16 & 0x1F) << 11)
}

const fn packed_depth(packed: u16) -> u8 { (packed & 0xFF) as u8 }

const fn packed_bound(packed: u16) -> u8 { ((packed >> 8) & 0x3) as u8 }

const fn packed_pv(packed: u16) -> u8 { ((packed >> 10) & 0x1) as u8 }

const fn packed_age(packed: u16) -> u8 { ((packed >> 11) & 0x1F) as u8 }

impl TtEntry {
    /// The cheap scan read: the key, and the packed word replacement weighs.
    /// Relaxed is enough, since a scan never reads payload off this load. The probe
    /// gates payload behind `probe_read`'s Acquire; the store path re-reads a slot it
    /// is about to overwrite, where a stale word costs a preserved move and no more.
    #[inline(always)]
    fn meta(&self) -> (u16, u16) { (self.key.load(Ordering::Relaxed), self.packed.load(Ordering::Relaxed)) }

    #[inline(always)]
    fn is_occupied(&self) -> bool { packed_bound(self.packed.load(Ordering::Relaxed)) != BOUND_NONE }

    /// The probe's per-slot read. One Acquire load of `key` gates it; the rest is
    /// pulled only on a match, so a miss costs a single atomic load. The Acquire
    /// pairs with the store's Release on `key`, the handshake that makes a matched
    /// key imply visible payload.
    #[inline(always)]
    fn probe_read(&self, key16: u16, ply: usize) -> Option<TtHit> {
        if self.key.load(Ordering::Acquire) != key16 {
            return None;
        }

        let packed = self.packed.load(Ordering::Relaxed);
        let bound = packed_bound(packed);
        if bound == BOUND_NONE {
            return None;
        }
        Some(TtHit {
            mv: Move::from_u16(self.mv.load(Ordering::Relaxed)),
            score: score_from_tt(i32::from(self.score.load(Ordering::Relaxed).cast_signed()), ply),
            depth: i32::from(packed_depth(packed)),
            bound,
            pv: packed_pv(packed) != 0,
            eval: i32::from(self.eval.load(Ordering::Relaxed).cast_signed()),
        })
    }

    #[inline(always)]
    fn store(&self, entry: Payload) {
        let packed = pack(entry.depth, entry.bound, entry.pv, entry.age);
        self.mv.store(entry.mv, Ordering::Relaxed);
        self.score.store(entry.score.cast_unsigned(), Ordering::Relaxed);
        self.eval.store(entry.eval.cast_unsigned(), Ordering::Relaxed);
        self.packed.store(packed, Ordering::Relaxed);
        // key goes last. A reader that Acquire-loads this new key is then
        // guaranteed the four payload writes above, released before it.
        self.key.store(entry.key, Ordering::Release);
    }
}

impl TranspositionTable {
    pub fn new(size_mb: usize, threads: usize) -> Self {
        let numa = NumaTopology::detect();
        let clusters = Self::alloc(size_mb, &numa, threads);
        Self { clusters, numa, generation: AtomicU8::new(0) }
    }

    pub fn resize(&mut self, size_mb: usize, threads: usize) { self.clusters = Self::alloc(size_mb, &self.numa, threads); }

    /// Whether this box spreads the TT across nodes at all, so a thread-count
    /// change can re-place it.
    pub fn distributes(&self) -> bool { self.numa.num_nodes() > 1 }

    pub fn page_kind(&self) -> PageKind { self.clusters.kind() }

    /// The cluster count follows from the page-rounded byte size, so a 1GB-page
    /// table holds a few more slots than a 4KB one of the same request.
    ///
    /// One memory domain takes the simple inline pre-fault. Two or more map the
    /// pages without faulting, then first-touch each slice on its own node so the
    /// table spreads across the controllers instead of pinning to one. A lone thread
    /// wants it pinned local, so that spread waits for the threads to share it.
    fn alloc(size_mb: usize, numa: &NumaTopology, threads: usize) -> HugePages<Cluster> {
        let bytes = size_mb.max(1) * 1024 * 1024;
        // SAFETY (both paths): Cluster is valid when zero-initialized (all fields
        // are AtomicU16(0), so every slot reads back as BOUND_NONE). Satisfies HugePages::mapped
        // and HugePages::zeroed's documented precondition. On the multi-node path,
        // first_touch does the actual zeroing before any reader reaches the data.
        if numa.should_distribute(threads) {
            let clusters = unsafe { HugePages::mapped(bytes) };
            // SAFETY: the mapping is fresh and unpublished, so nothing can read it.
            unsafe { first_touch(&clusters, numa) };
            clusters
        } else {
            unsafe { HugePages::zeroed(bytes) }
        }
    }

    /// Occupancy in permille (0..=1000), sampled over the first 1000 slots.
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
            .filter(|s| s.is_occupied())
            .count()
            * 1000
            / sample
    }

    /// Reset to an empty table and the generation counter, for the thread count the
    /// next search will use. Spreads the cleared pages across nodes only when it pays,
    /// more than one node and more than one thread; a lone thread keeps the table
    /// local. On a multi-node box this is also its first fault, since `alloc` left
    /// the mapping untouched.
    /// # Safety
    /// No search may be running. The clear writes the region non-atomically, so a
    /// probe landing in it is a data race however atomic the slots are.
    pub unsafe fn clear(&self, threads: usize) {
        // SAFETY: caller's contract, on both branches.
        unsafe {
            if self.numa.should_distribute(threads) {
                first_touch(&self.clusters, &self.numa);
            } else {
                self.clusters.clear();
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    /// Once per new position. Entries from earlier generations don't vanish; they just
    /// age, growing easier to evict.
    pub fn new_search(&self) { self.generation.fetch_add(1, Ordering::Relaxed); }

    /// Pin the calling search thread to its NUMA-assigned L3 domain, when binding
    /// is worthwhile. Keeps the thread's compute on its cores and the per-thread
    /// state it allocates next in a warm L3.
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
        // SAFETY: _mm_prefetch is a non-faulting hint instruction available on
        // all x86_64 targets; ptr is valid within the allocation.
        unsafe {
            let ptr = self.clusters.as_ptr().add(idx) as *const i8;
            #[cfg(target_arch = "x86_64")]
            arch::x86_64::_mm_prefetch::<{ arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64, ply: usize) -> Option<TtHit> {
        let key16 = verification_key(hash);
        self.cluster(self.index(hash)).slots.iter().find_map(|slot| slot.probe_read(key16, ply))
    }

    /// Insert or update this position.
    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut victim = 0;
        let mut worst_quality = i32::MAX;
        let mut is_key_match = false;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, packed) = slot.meta();
            if packed_bound(packed) == BOUND_NONE || key == key16 {
                victim = i;
                is_key_match = key == key16;
                break;
            }

            let quality = replacement_quality(packed, cur);
            if quality < worst_quality {
                worst_quality = quality;
                victim = i;
            }
        }

        let mut store_mv = mv.inner();
        let mut store_pv = pv as u8;

        // Keep the existing move when the new store is a null move on the same
        // position. The pv flag travels with the move it describes; dropping it
        // would erase the position's PV history.
        if mv.is_null() && is_key_match {
            let existing = &cluster.slots[victim];
            store_mv = existing.mv.load(Ordering::Relaxed);
            store_pv |= packed_pv(existing.packed.load(Ordering::Relaxed));
        }

        cluster.slots[victim].store(Payload {
            key: key16,
            mv: store_mv,
            score: score_to_tt(score, ply) as i16,
            eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            // 8-bit field covers MAX_DEPTH (246); clamp just guards the cast.
            depth: depth.clamp(0, u8::MAX as i32) as u8,
            bound,
            age: cur,
            pv: store_pv,
        });
    }

    /// Qsearch floods the table with shallow entries, so its replacement is timid;
    /// it takes this position's own slot, an empty one, another depth-0 entry, or one
    /// whose quality has aged to nothing. A fresh deep result is never evicted for it.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: u8, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut victim: Option<usize> = None;
        let mut worst_quality = i32::MAX;
        let mut is_key_match = false;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, packed) = slot.meta();
            if key == key16 || packed_bound(packed) == BOUND_NONE || packed_depth(packed) == 0 {
                victim = Some(i);
                is_key_match = key == key16;
                break;
            }

            let quality = replacement_quality(packed, cur);
            if quality <= 0 && quality < worst_quality {
                worst_quality = quality;
                victim = Some(i);
            }
        }

        if let Some(victim) = victim {
            // Preserve a prior PV bit when overwriting the same position;
            // a qsearch visit would otherwise wipe the flag a previous
            // negamax store left here. Only a key match reads it back.
            let store_pv = if is_key_match {
                pv as u8 | packed_pv(cluster.slots[victim].packed.load(Ordering::Relaxed))
            } else {
                pv as u8
            };
            cluster.slots[victim].store(Payload {
                key: key16,
                mv: mv.inner(),
                score: score_to_tt(score, ply) as i16,
                eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                depth: 0,
                bound,
                age: cur,
                pv: store_pv,
            });
        }
    }

    #[inline(always)]
    fn cluster(&self, idx: usize) -> &Cluster {
        // SAFETY: idx is index()'s output, a mulhi64 in [0, clusters.len()).
        unsafe { self.clusters.get_unchecked(idx) }
    }

    #[inline(always)]
    fn index(&self, hash: u64) -> usize {
        // mulhi64: the top 64 bits of hash · len. Lands the hash uniformly
        // in [0, len), the way hash % len would, but with a multiply.
        let clusters = self.clusters.len();
        (((hash as u128) * (clusters as u128)) >> 64) as usize
    }
}

/// How readily a slot gives way: its depth, discounted by how many generations
/// old it is. `AGE_FACTOR` sets the exchange rate between the two.
#[inline(always)]
fn replacement_quality(packed: u16, current_age: u8) -> i32 {
    let gen_diff = (current_age.wrapping_sub(packed_age(packed)) & AGE_MASK) as i32;
    packed_depth(packed) as i32 - gen_diff * AGE_FACTOR
}

/// First-touch the cluster region in parallel, each NUMA node's thread zeroing
/// the slice bound to it. First-touch decides a page's home node, so it spreads
/// the table across the memory controllers instead of leaving it all on one.
///
/// # Safety
/// No search may be running: this builds an exclusive `&mut` over the shared
/// region and writes it non-atomically.
unsafe fn first_touch(clusters: &HugePages<Cluster>, numa: &NumaTopology) {
    let nodes = numa.num_nodes();
    let len = clusters.len();
    // SAFETY: caller's contract; the region maps len clusters.
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
