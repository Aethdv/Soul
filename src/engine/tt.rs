//! Transposition table: the search's memory of positions it has already seen.
//!
//! When iterative deepening revisits a node, it hopes to find it already here. A hit
//! can cut a whole subtree, and since each iteration seeds the next, deepening stays
//! cheap.
//!
//! The table is lockless because Lazy SMP has every thread reading and writing it at
//! once. A 16-bit verification key catches most hash collisions; the roughly one in
//! 2¹⁶ that slips through can hand back a wrong score, or a move belonging to another
//! position, which is why every stored move is re-checked by `is_pseudo_legal` before
//! the search plays it.
//!
//! Entries sit in three-slot clusters, so replacement has a choice of victim.

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
    },
    engine::{movegen::is_pseudo_legal, search_params::SearchParams},
    hugepages::{HugePages, PageKind},
    numa::NumaTopology,
};

/// No eval stored, which is what an in-check store leaves behind.
/// Sits above MATE, so it can never be mistaken for a score.
pub const SCORE_NONE: i32 = 32000;

const CLUSTER_SIZE: usize = 3;

/// `age` is 5 bits; generation distance is read modulo 32.
const AGE_MASK: u8 = 0x1F;

const _: () = assert!(mem::size_of::<TtEntry>() == 10);
const _: () = assert!(mem::size_of::<Cluster>() == 32);
const _: () = assert!(mem::align_of::<Cluster>() == 32);

/// What a stored score proves. Discriminants are the on-slot encoding, two bits of `packed`.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
#[repr(u8)]
pub enum Bound {
    /// Empty slot.
    #[default]
    None = 0,
    Exact = 1,
    /// Beta cutoff (fail-high).
    Lower = 2,
    /// Alpha cutoff (fail-low).
    Upper = 3,
}

impl Bound {
    /// Total over the two bits, so a torn read decodes to a variant instead of nothing.
    #[inline(always)]
    const fn from_bits(bits: u16) -> Self {
        // SAFETY: the mask leaves 0..=3, and every one of those is a declared discriminant.
        unsafe { mem::transmute((bits & 0x3) as u8) }
    }
}

#[inline(always)]
pub fn can_cutoff(bound: Bound, score: i32, alpha: i32, beta: i32) -> bool {
    match bound {
        Bound::Exact => true,
        Bound::Lower => score >= beta,
        Bound::Upper => score <= alpha,
        Bound::None => false,
    }
}

/// The static eval clamped into the range a searched score proves.
#[inline(always)]
pub fn clamp_to_bound(bound: Bound, score: i32, eval: i32) -> i32 {
    match bound {
        Bound::Exact => score,
        Bound::Lower => eval.max(score),
        Bound::Upper => eval.min(score),
        Bound::None => eval,
    }
}

/// A stale move is caught by `is_pseudo_legal`,
/// and a stale score is only a bad cutoff.
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
    age_factor: AtomicI32,
    /// The machine's locality, detected once.
    numa: NumaTopology,
}

/// Half a cache line, aligned so a cluster never straddles two.
#[repr(C, align(32))]
struct Cluster {
    slots: [TtEntry; CLUSTER_SIZE],
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum TtMove {
    /// A fail-low store leaves a null move behind.
    Null,
    Found(Move),
    /// Garbage under a matching key, so the slot's score proves nothing either.
    Collision,
}

impl TtMove {
    #[inline(always)]
    pub fn get(self) -> Option<Move> {
        match self {
            Self::Found(mv) => Some(mv),
            Self::Null | Self::Collision => None,
        }
    }
}

/// What a probe hands back, already unfolded into search units.
#[derive(Clone, Copy)]
pub struct TtData {
    raw_mv: Move,
    pub pv: bool,
    /// Raw static eval at store time, `SCORE_NONE` when the position was in check.
    pub eval: i32,
    pub score: i32,
    pub bound: Bound,
    pub depth: i32,
}

impl TtData {
    /// No slot matched.
    pub const NONE: Self =
        Self { raw_mv: Move::null(), pv: false, eval: SCORE_NONE, score: SCORE_NONE, bound: Bound::None, depth: 0 };

    /// Bind once per node and after the cutoff test,
    /// since a probe that cuts never needs the move.
    #[inline(always)]
    pub fn mv(&self, pos: &Position) -> TtMove {
        if self.raw_mv.is_null() {
            TtMove::Null
        } else if is_pseudo_legal(pos, self.raw_mv) {
            TtMove::Found(self.raw_mv)
        } else {
            TtMove::Collision
        }
    }
}

#[derive(Clone, Copy, Default)]
struct SlotWrite {
    key: u16,
    mv: u16,
    score: i16,
    eval: i16,
    depth: u8,
    bound: Bound,
    age: u8,
    pv: u8,
}

/// The cluster index is drawn from the high bits.
#[inline(always)]
const fn verification_key(hash: u64) -> u16 { hash as u16 }

const fn pack(depth: u8, bound: Bound, pv: u8, age: u8) -> u16 {
    depth as u16 | ((bound as u16 & 0x3) << 8) | ((pv as u16 & 0x1) << 10) | ((age as u16 & 0x1F) << 11)
}

const fn packed_depth(packed: u16) -> u8 { (packed & 0xFF) as u8 }
const fn packed_bound(packed: u16) -> Bound { Bound::from_bits(packed >> 8) }
const fn packed_pv(packed: u16) -> u8 { ((packed >> 10) & 0x1) as u8 }
const fn packed_age(packed: u16) -> u8 { ((packed >> 11) & 0x1F) as u8 }

impl TtEntry {
    /// The replacement scan's read. Relaxed is enough, since a scan never reads payload
    /// off this load, and a stale word costs the store path a preserved move and no more.
    #[inline(always)]
    fn scan_read(&self) -> (u16, u16) { (self.key.load(Ordering::Relaxed), self.packed.load(Ordering::Relaxed)) }

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
            pv: packed_pv(packed) != 0,
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
        // key goes last. A reader that Acquire-loads this new key is then
        // guaranteed the four payload writes above, released before it.
        self.key.store(entry.key, Ordering::Release);
    }
}

impl TranspositionTable {
    pub fn new(size_mb: usize, threads: usize) -> Self {
        let numa = NumaTopology::detect();
        let clusters = Self::alloc(size_mb, &numa, threads);
        Self {
            clusters,
            numa,
            generation: AtomicU8::new(0),
            age_factor: AtomicI32::new(SearchParams::new().tt_age_factor),
        }
    }

    pub fn resize(&mut self, size_mb: usize, threads: usize) { self.clusters = Self::alloc(size_mb, &self.numa, threads); }

    pub fn set_age_factor(&self, factor: i32) { self.age_factor.store(factor, Ordering::Relaxed); }
    pub fn spans_nodes(&self) -> bool { self.numa.num_nodes() > 1 }
    pub fn page_kind(&self) -> PageKind { self.clusters.kind() }

    /// The cluster count follows from the page-rounded byte size, so a 1GB-page
    /// table holds a few more slots than a 4KB one of the same request.
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

    /// Empty the table and reset the generation counter, placing the cleared pages for
    /// the thread count the next search will use. On a multi-node box this is also the
    /// mapping's first fault, since `alloc` left it untouched.
    ///
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

    pub fn begin_search(&self) { self.generation.fetch_add(1, Ordering::Relaxed); }

    /// Keeps the thread's compute on its cores and the per-thread state it allocates
    /// next in a warm L3.
    pub fn bind_search_thread(&self, thread_id: usize, threads: usize) {
        if self.numa.should_bind(threads) {
            self.numa.bind_to_domain(self.numa.distribute(threads)[thread_id]);
        }
    }

    /// A TT lookup is a near-random memory access, the kind that stalls the negamax
    /// loop on a miss; issued early, the fetch overlaps the work up to the read.
    #[inline(always)]
    pub fn prefetch(&self, hash: u64) {
        let idx = self.index(hash);
        // SAFETY: _mm_prefetch is a non-faulting hint instruction available
        // on all x86_64 targets; ptr is valid within the allocation.
        unsafe {
            let ptr = self.clusters.as_ptr().add(idx) as *const i8;
            #[cfg(target_arch = "x86_64")]
            arch::x86_64::_mm_prefetch::<{ arch::x86_64::_MM_HINT_T0 }>(ptr);
        }
    }

    #[inline(always)]
    pub fn probe(&self, hash: u64, ply: usize) -> TtData {
        let key16 = verification_key(hash);
        self.cluster(self.index(hash))
            .slots
            .iter()
            .find_map(|slot| slot.probe_read(key16, ply))
            .unwrap_or(TtData::NONE)
    }

    #[inline(always)]
    pub fn store(&self, hash: u64, ply: usize, depth: i32, score: i32, mv: Move, bound: Bound, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur_age = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut victim = 0;
        let mut worst_quality = i32::MAX;
        let mut existing = None;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, packed) = slot.scan_read();
            if packed_bound(packed) == Bound::None || key == key16 {
                victim = i;
                existing = (key == key16).then_some(packed);
                break;
            }

            let quality = replacement_quality(packed, cur_age);
            if quality < worst_quality {
                worst_quality = quality;
                victim = i;
            }
        }

        if existing.is_some_and(|packed| stored_outranks(packed, depth, pv, bound, cur_age)) {
            if !mv.is_null() {
                cluster.slots[victim].mv.store(mv.inner(), Ordering::Relaxed);
            }
            return;
        }

        let mut store_mv = mv.inner();
        let mut store_pv = pv as u8;

        if mv.is_null() && existing.is_some() {
            let slot = &cluster.slots[victim];
            store_mv = slot.mv.load(Ordering::Relaxed);
            store_pv |= packed_pv(slot.packed.load(Ordering::Relaxed));
        }

        cluster.slots[victim].store(SlotWrite {
            key: key16,
            mv: store_mv,
            score: score_to_tt(score, ply) as i16,
            eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
            // 8-bit field covers MAX_DEPTH (246); clamp just guards the cast.
            depth: depth.clamp(0, u8::MAX as i32) as u8,
            bound,
            age: cur_age,
            pv: store_pv,
        });
    }

    /// Qsearch floods the table with shallow entries, so its replacement is timid:
    /// a fresh deep result is never evicted for one.
    #[inline(always)]
    pub fn store_qs(&self, hash: u64, ply: usize, score: i32, mv: Move, bound: Bound, pv: bool, eval: i32) {
        let idx = self.index(hash);
        let key16 = verification_key(hash);
        let cur_age = self.generation.load(Ordering::Relaxed) & AGE_MASK;
        let cluster = self.cluster(idx);

        let mut victim: Option<usize> = None;
        let mut worst_quality = i32::MAX;
        let mut existing = None;

        for (i, slot) in cluster.slots.iter().enumerate() {
            let (key, packed) = slot.scan_read();
            if key == key16 || packed_bound(packed) == Bound::None || packed_depth(packed) == 0 {
                victim = Some(i);
                existing = (key == key16).then_some(packed);
                break;
            }

            let quality = replacement_quality(packed, cur_age);
            if quality <= 0 && quality < worst_quality {
                worst_quality = quality;
                victim = Some(i);
            }
        }

        if let Some(victim) = victim {
            if existing.is_some_and(|packed| stored_outranks(packed, 0, pv, bound, cur_age)) {
                if !mv.is_null() {
                    cluster.slots[victim].mv.store(mv.inner(), Ordering::Relaxed);
                }
                return;
            }

            // A qsearch visit would otherwise wipe the flag a previous negamax
            // store left on this position.
            let store_pv = if existing.is_some() {
                pv as u8 | packed_pv(cluster.slots[victim].packed.load(Ordering::Relaxed))
            } else {
                pv as u8
            };
            cluster.slots[victim].store(SlotWrite {
                key: key16,
                mv: mv.inner(),
                score: score_to_tt(score, ply) as i16,
                eval: eval.clamp(i16::MIN as i32, i16::MAX as i32) as i16,
                depth: 0,
                bound,
                age: cur_age,
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

#[inline(always)]
fn flag_bonus(bound: Bound) -> i32 {
    match bound {
        Bound::Exact => 3,
        Bound::Lower => 2,
        Bound::Upper => 1,
        Bound::None => 0,
    }
}

/// Whether the stored entry outranks the new one.
#[inline(always)]
fn stored_outranks(packed: u16, depth: i32, pv: bool, bound: Bound, cur_age: u8) -> bool {
    if bound == Bound::Exact && packed_bound(packed) != Bound::Exact {
        return false;
    }
    let old_depth = i32::from(packed_depth(packed));
    let old_bonus = flag_bonus(packed_bound(packed));
    let new_bonus = flag_bonus(bound);
    let age_diff = (cur_age.wrapping_sub(packed_age(packed)) & AGE_MASK) as i32;
    let insert_priority = depth + new_bonus + (age_diff * age_diff) / 4 + i32::from(pv);
    let record_priority = old_depth + old_bonus;
    insert_priority * 3 < record_priority * 2
}

/// Replacement quality: depth plus flag and pv bonuses, discounted by age.
#[inline(always)]
fn replacement_quality(packed: u16, current_age: u8) -> i32 {
    let gen_diff = (current_age.wrapping_sub(packed_age(packed)) & AGE_MASK) as i32;
    let depth = i32::from(packed_depth(packed));
    let bonus = flag_bonus(packed_bound(packed)) + i32::from(packed_pv(packed));
    depth + bonus - (gen_diff * gen_diff) / 4
}

/// First-touch decides a page's home node, so zeroing each slice from a thread bound
/// to its own node spreads the table across the memory controllers.
///
/// # Safety
/// No search may be running: builds an exclusive `&mut` over the shared region
/// and writes it non-atomically.
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
