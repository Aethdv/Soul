//! The transposition table caches search results by position hash.
//! A hit can improve move ordering or, when its bound is deep enough, skip a
//! subtree. Iterative deepening makes earlier results useful to later searches.
//!
//! Lazy SMP workers share the table without locks. Each entry has a 16-bit
//! verification key, so collisions can return another position's data; concurrent
//! writes can also mix fields. Before a stored move is used, `is_pseudo_legal`
//! checks it against the current position. If it fails, the entire entry is discarded.

use std::{
    arch, mem, slice,
    sync::atomic::{AtomicI32, AtomicU8, AtomicU16, Ordering},
    thread,
};

use crate::{
    core::{
        board::Position,
        defs::{score_from_tt, score_to_tt},
        moves::Move,
        util::mulhi64,
    },
    engine::{movegen::is_pseudo_legal, search_params::SearchParams},
    hugepages::{HugePages, PageKind},
    numa::NumaTopology,
};

/// No eval stored, which is what an in-check store leaves behind.
pub(super) const SCORE_NONE: i32 = 32000;

const CLUSTER_SIZE: usize = 3;
const AGE_MASK: u8 = 0x1F;

/// How much deeper a stored result must be to outrank a revisit of the same position.
const DEPTH_SLACK: i32 = 4;

const _: () = assert!(mem::size_of::<TtEntry>() == 10);
const _: () = assert!(mem::size_of::<Cluster>() == 32);
const _: () = assert!(mem::align_of::<Cluster>() == 32);

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub(super) enum Bound {
    #[default]
    None = 0,
    Exact = 1,
    /// Beta cutoff (fail-high).
    Lower = 2,
    /// Alpha cutoff (fail-low).
    Upper = 3,
}

impl Bound {
    /// Classifies the score against the window before search updates alpha.
    #[inline(always)]
    pub(super) fn from_score(score: i32, alpha: i32, beta: i32) -> Self {
        if score >= beta {
            Self::Lower
        } else if score > alpha {
            Self::Exact
        } else {
            Self::Upper
        }
    }

    #[inline(always)]
    const fn from_bits(bits: u16) -> Self {
        // SAFETY: masking with 0x3 restricts the discriminant to 0..=3, covering all four variants.
        unsafe { mem::transmute((bits & 0x3) as u8) }
    }
}

#[inline(always)]
pub(super) fn can_cutoff(bound: Bound, score: i32, alpha: i32, beta: i32) -> bool {
    match bound {
        Bound::Exact => true,
        Bound::Lower => score >= beta,
        Bound::Upper => score <= alpha,
        Bound::None => false,
    }
}

#[inline(always)]
pub(super) fn clamp_to_bound(bound: Bound, score: i32, eval: i32) -> i32 {
    match bound {
        Bound::Exact => score,
        Bound::Lower => eval.max(score),
        Bound::Upper => eval.min(score),
        Bound::None => eval,
    }
}

#[repr(C)]
struct TtEntry {
    key: AtomicU16,
    mv: AtomicU16,
    score: AtomicU16,
    eval: AtomicU16,
    /// depth(8) | bound(2) | pv(1) | age(5).
    packed: AtomicU16,
}

pub(crate) struct TranspositionTable {
    clusters: HugePages<Cluster>,
    /// Bumped per search; aging measures generation distance modulo 32.
    generation: AtomicU8,
    age_factor: AtomicI32,
    numa: NumaTopology,
}

#[repr(C, align(32))]
struct Cluster {
    slots: [TtEntry; CLUSTER_SIZE],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) enum TtMove {
    Null,
    Found(Move),
    /// A pseudo-illegal move means the verification key collided; discard.
    Collision,
}

impl TtMove {
    #[inline(always)]
    pub(super) fn get(self) -> Option<Move> {
        match self {
            Self::Found(mv) => Some(mv),
            Self::Null | Self::Collision => None,
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct TtData {
    raw_mv: Move,
    pub(super) pv: bool,
    pub(super) eval: i32,
    pub(super) score: i32,
    pub(super) bound: Bound,
    pub(super) depth: i32,
}

impl TtData {
    pub(super) const NONE: Self =
        Self { raw_mv: Move::null(), pv: false, eval: SCORE_NONE, score: SCORE_NONE, bound: Bound::None, depth: 0 };

    /// A pseudo-illegal move makes every field in the entry untrustworthy.
    #[inline(always)]
    pub(super) fn mv(&self, pos: &Position) -> TtMove {
        if self.raw_mv.is_null() {
            TtMove::Null
        } else if is_pseudo_legal(pos, self.raw_mv) {
            TtMove::Found(self.raw_mv)
        } else {
            TtMove::Collision
        }
    }
}

struct SlotWrite {
    key: u16,
    mv: u16,
    score: i16,
    eval: i16,
    depth: u8,
    bound: Bound,
    age: u8,
    pv: bool,
}

/// Low 16 bits; `index` draws the cluster from high bits.
#[inline(always)]
const fn verification_key(hash: u64) -> u16 { hash as u16 }

const fn pack(depth: u8, bound: Bound, pv: bool, age: u8) -> u16 {
    depth as u16 | ((bound as u16 & 0x3) << 8) | ((pv as u16) << 10) | ((age as u16 & 0x1F) << 11)
}

const fn packed_depth(packed: u16) -> u8 { (packed & 0xFF) as u8 }
const fn packed_bound(packed: u16) -> Bound { Bound::from_bits(packed >> 8) }
const fn packed_pv(packed: u16) -> bool { ((packed >> 10) & 0x1) != 0 }
const fn packed_age(packed: u16) -> u8 { ((packed >> 11) & 0x1F) as u8 }

impl TtEntry {
    #[inline(always)]
    fn scan_read(&self) -> (u16, u16) {
        // Replacement tolerates stale metadata; probes use the Acquire path.
        (self.key.load(Ordering::Relaxed), self.packed.load(Ordering::Relaxed))
    }

    #[inline(always)]
    fn is_occupied(&self) -> bool { packed_bound(self.packed.load(Ordering::Relaxed)) != Bound::None }

    /// The Acquire on `key` is the read half of the store's Release.
    #[inline(always)]
    fn probe_read(&self, key16: u16, ply: usize) -> Option<TtData> {
        if self.key.load(Ordering::Acquire) != key16 {
            return None;
        }

        let packed = self.packed.load(Ordering::Relaxed);
        let bound = packed_bound(packed);
        if bound == Bound::None {
            return None;
        }

        Some(TtData {
            raw_mv: Move::from_u16(self.mv.load(Ordering::Relaxed)),
            score: score_from_tt(i32::from(self.score.load(Ordering::Relaxed).cast_signed()), ply),
            depth: i32::from(packed_depth(packed)),
            bound,
            pv: packed_pv(packed),
            eval: i32::from(self.eval.load(Ordering::Relaxed).cast_signed()),
        })
    }

    #[inline(always)]
    fn store(&self, entry: SlotWrite) {
        let packed = pack(entry.depth, entry.bound, entry.pv, entry.age);
        self.mv.store(entry.mv, Ordering::Relaxed);
        self.score.store(entry.score.cast_unsigned(), Ordering::Relaxed);
        self.eval.store(entry.eval.cast_unsigned(), Ordering::Relaxed);
        self.packed.store(packed, Ordering::Relaxed);
        // Concurrent writers can still mix payload fields; Acquire does not guarantee a coherent snapshot.
        self.key.store(entry.key, Ordering::Release);
    }
}

impl TranspositionTable {
    pub(crate) fn new(size_mb: usize, threads: usize) -> Self {
        let numa = NumaTopology::detect();
        let clusters = Self::alloc(size_mb, &numa, threads);
        Self {
            clusters,
            numa,
            generation: AtomicU8::new(0),
            age_factor: AtomicI32::new(SearchParams::new().tt_age_factor),
        }
    }

    pub(crate) fn set_age_factor(&self, factor: i32) { self.age_factor.store(factor, Ordering::Relaxed); }
    pub(crate) fn spans_nodes(&self) -> bool { self.numa.num_nodes() > 1 }
    pub(crate) fn page_kind(&self) -> PageKind { self.clusters.kind() }

    /// Page rounding determines the cluster count, not the requested size.
    fn alloc(size_mb: usize, numa: &NumaTopology, threads: usize) -> HugePages<Cluster> {
        let bytes = size_mb.max(1) * 1024 * 1024;

        if numa.should_split_across_nodes(threads) {
            // SAFETY: zero-initialized clusters are valid: a zero bound marks each slot empty.
            // Page alignment satisfies Cluster's 32-byte alignment requirement.
            let clusters = unsafe { HugePages::mapped(bytes) };
            // SAFETY: The mapping is fresh and unpublished.
            unsafe { first_touch(&clusters, numa) };
            clusters
        } else {
            unsafe { HugePages::prefaulted(bytes) }
        }
    }

    /// Occupancy in permille, sampled from up to the first 1000 slots.
    pub(super) fn hashfull(&self) -> usize {
        let total = self.clusters.len() * CLUSTER_SIZE;
        let sample = total.min(1000);
        self.clusters
            .iter()
            .flat_map(|c| c.slots.iter())
            .take(sample)
            .filter(|s| s.is_occupied())
            .count()
            * 1000
            / sample
    }

    /// Clears the table and resets its generation. On NUMA systems, each slice
    /// is cleared by a thread bound to its node.
    ///
    /// # Safety
    /// No search may access the table during the clear; slots are zeroed non-atomically.
    pub(crate) unsafe fn clear(&self, threads: usize) {
        // SAFETY: The caller excludes concurrent access in both branches.
        unsafe {
            if self.numa.should_split_across_nodes(threads) {
                first_touch(&self.clusters, &self.numa);
            } else {
                self.clusters.clear();
            }
        }
        self.generation.store(0, Ordering::Relaxed);
    }

    pub(crate) fn begin_search(&self) { self.generation.fetch_add(1, Ordering::Relaxed); }

    pub(crate) fn bind_search_thread(&self, thread_id: usize, threads: usize) {
        if self.numa.should_bind_l3(threads) {
            self.numa.bind_to_l3_domain(self.numa.assign_l3_domains(threads)[thread_id]);
        }
    }

    /// TT probes have poor locality; prefetch early to overlap the cache miss with useful work.
    #[inline(always)]
    pub(super) fn prefetch(&self, hash: u64) {
        let idx = self.index(hash);
        // SAFETY: index bounds idx to the allocation; prefetch is non-faulting.
        unsafe {
            let ptr = self.clusters.as_ptr().add(idx) as *const i8;
            arch::x86_64::_mm_prefetch::<{ arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    #[inline(always)]
    pub(super) fn probe(&self, hash: u64, ply: usize) -> TtData {
        let key16 = verification_key(hash);
        self.cluster(self.index(hash))
            .slots
            .iter()
            .find_map(|slot| slot.probe_read(key16, ply))
            .unwrap_or(TtData::NONE)
    }

    #[inline(always)]
    pub(super) fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: Bound, pv: bool, eval: i32) {
        self.store_in::<false>(hash, ply, depth, score, mv, bound, pv, eval);
    }

    /// QSearch floods the table with shallow entries, so its replacement is timid:
    /// a fresh deep result is never evicted for one.
    #[inline(always)]
    pub(super) fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: Bound, pv: bool, eval: i32) {
        self.store_in::<true>(hash, ply, 0, score, mv, bound, pv, eval);
    }

    #[inline(always)]
    fn store_in<const QS: bool>(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: Bound, pv: bool, eval: i32) {
        let key16 = verification_key(hash);
        let cur_age = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let age_factor = self.age_factor.load(Ordering::Relaxed);
        let cluster = self.cluster(self.index(hash));

        let mut victim = (!QS).then_some(0);
        // QSearch evicts only at quality zero or below, so its bar sits one above it.
        let mut worst_quality = if QS { 1 } else { i32::MAX };
        let mut existing = None;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, packed) = slot.scan_read();
            if key == key16 || packed_bound(packed) == Bound::None || (QS && packed_depth(packed) == 0) {
                victim = Some(i);
                existing = (key == key16).then_some(packed);
                break;
            }

            let quality = replacement_quality(packed, cur_age, age_factor);
            if quality < worst_quality {
                worst_quality = quality;
                victim = Some(i);
            }
        }

        let Some(victim) = victim else { return };
        let slot = &cluster.slots[victim];

        if existing.is_some_and(|packed| stored_outranks(packed, depth, pv, bound, cur_age)) {
            if !mv.is_null() {
                slot.mv.store(mv.inner(), Ordering::Relaxed);
            }
            return;
        }

        let mut store_mv = mv.inner();
        let mut store_pv = pv;

        if existing.is_some() {
            if mv.is_null() && !QS {
                // No new move to order by; keep the stored one.
                store_mv = slot.mv.load(Ordering::Relaxed);
            }
            // The probe may have missed this entry; don't clear its PV bit.
            store_pv |= packed_pv(slot.packed.load(Ordering::Relaxed));
        }

        slot.store(SlotWrite {
            key: key16,
            mv: store_mv,
            score: score_to_tt(score, ply) as i16,
            eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            depth: depth.clamp(0, u8::MAX as i32) as u8,
            bound,
            age: cur_age,
            pv: store_pv,
        });
    }

    #[inline(always)]
    fn cluster(&self, idx: usize) -> &Cluster {
        // SAFETY: index maps every hash into 0..self.clusters.len().
        unsafe { self.clusters.get_unchecked(idx) }
    }

    #[inline(always)]
    fn index(&self, hash: u64) -> usize { mulhi64(hash, self.clusters.len()) }
}

/// Whether the stored entry outranks a revisit of the same position.
/// An exact result always replaces the stored entry, regardless of depth.
#[inline(always)]
fn stored_outranks(packed: u16, depth: i32, pv: bool, bound: Bound, cur_age: u8) -> bool {
    bound != Bound::Exact
        && packed_age(packed) == cur_age
        && i32::from(packed_depth(packed)) >= depth + 2 * i32::from(pv) + DEPTH_SLACK
}

/// Deeper entries score higher; each generation of age costs `age_factor`.
#[inline(always)]
fn replacement_quality(packed: u16, current_age: u8, age_factor: i32) -> i32 {
    let gen_diff = (current_age.wrapping_sub(packed_age(packed)) & AGE_MASK) as i32;
    packed_depth(packed) as i32 - gen_diff * age_factor
}

/// First-touch determines NUMA placement, so each thread zeroes its local slice.
///
/// # Safety
/// No search may run: this creates an exclusive mutable slice and writes it non-atomically.
unsafe fn first_touch(clusters: &HugePages<Cluster>, numa: &NumaTopology) {
    let nodes = numa.num_nodes();
    let len = clusters.len();
    // SAFETY: clusters provides storage for len aligned clusters.
    let region = unsafe { slice::from_raw_parts_mut(clusters.as_ptr() as *mut Cluster, len) };

    thread::scope(|scope| {
        for (node, chunk) in region.chunks_mut(len.div_ceil(nodes)).enumerate() {
            scope.spawn(move || {
                numa.bind_to_node(node);
                // SAFETY: Chunks are disjoint, and zeroed clusters represent empty slots.
                unsafe { chunk.as_mut_ptr().write_bytes(0, chunk.len()) };
            });
        }
    });
}
