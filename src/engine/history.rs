//! History heuristics for move ordering, and eval correction alongside them.
//!
//! A move that caused a beta cutoff once tends to again, so the search records the
//! success and orders by it next time. This module holds the family of tables that
//! do it, each keyed on a different slice of context:
//!
//! - the main `[side][piece][to]` table, plus a `butterfly` table split by whether
//!   the from- and to-squares are under threat,
//! - continuation history, keyed on a recent move and the current reply (n-1, n-2,
//!   n-4 plies back),
//! - capture history for noisy moves, keyed on attacker, target, and victim.
//!
//! Correction history is the odd one out: it doesn't order moves at all. It nudges
//! the static eval toward what search actually found, keyed on pawn, minor, and
//! major structure. It learns the same way the others do, a table fed by search
//! outcomes, so it sits here despite the different job, and it's the heaviest
//! single Elo contributor in the file.

use crate::{
    core::defs::{Bitboard, Color, PieceType, Square},
    engine::search_params::SearchParams,
};

pub const CORRECTION_SIZE: usize = 16384;
pub const CORRECTION_SCALE: i32 = 256;
pub const CORRECTION_LIMIT: i32 = 256 * 32;
/// Denominator for the per-table blend weights: `weight / this` is a table's share.
pub const CORRECTION_WEIGHT_SCALE: i32 = 256;

const _: () = assert!(CORRECTION_SIZE.is_power_of_two());

/// Which `cont` table each continuation distance reads, in n-1, n-2, n-4 order.
const CONT_SLOTS: [usize; 3] = [0, 1, 1];

/// Per-table soft-gravity saturation caps, refreshed from `SearchParams` each
/// search. A table pulls its entries toward ±cap; the same value bounds the
/// entry and divides the gravity term, so the two move as one. Must stay in
/// `(0, i16::MAX]`: a zero cap divides by zero, a cap past `i16::MAX` overflows
/// the `as i16` store.
#[derive(Clone, Copy)]
pub struct HistoryCaps {
    pub quiet: i32,
    pub butterfly: i32,
    pub cont: i32,
    pub capt: i32,
}

/// Combined history tables for move ordering.
#[derive(Clone)]
pub struct History {
    /// `[side][piece][to_square]`: bounds ±cap
    table: [[[i16; 64]; 6]; 2],
    /// `[side][from_atk][to_atk][from · 64 + to]`: bounds ±cap
    butterfly: [[[[i16; 4096]; 2]; 2]; 2], // ~35 Elo
    /// `[ply_offset][side][prev_piece][prev_to][piece][to]`. Two tables, three distances:
    /// n-2 and n-4 pool into one on purpose.
    cont: [ContinuationHistory; 2], // n-1 (~13 Elo), n-2 (~3 Elo), n-4 (~3 Elo)
    /// `[side][pawn_hash & 0x3FFF]`
    correction: CorrectionHistory, // ~53 Elo
    /// `[side][minor_hash & 0x3FFF]`: knights + bishops, both colors
    minor_correction: CorrectionHistory, // minor + major: ~23 Elo
    /// `[side][major_hash & 0x3FFF]`: rooks + queens, both colors
    major_correction: CorrectionHistory,
    /// `[side][attacker][to][victim]`
    capt: CaptureHistory, // ~8 Elo
    /// Soft-gravity saturation caps, refreshed from `SearchParams` each search.
    pub caps: HistoryCaps,
}

#[derive(Clone, Copy)]
pub struct ContContext {
    pub pt: PieceType,
    pub to: Square,
}

#[derive(Clone)]
pub struct ContinuationHistory {
    data: Box<[i16]>,
}

/// Capture history: `[side][attacker][to][victim] -> i16`.
///
/// Tracks which captures historically caused beta cutoffs, indexed by the
/// attacker piece type, the destination square, and the victim piece type.
/// Plain captures and en passant participate; promotion-captures do NOT
/// (they bypass the normal MVV-LVA path in the picker and are already
/// strongly ordered by promotion piece, so we keep both sides of the table
/// consistent by skipping them entirely).
#[derive(Clone)]
pub struct CaptureHistory {
    data: Box<[i16]>,
}

/// Hash-keyed evaluator bias correction.
///
/// Observes the delta between static eval and search result, then applies
/// a weighted moving average so future evals of positions sharing the same
/// key are nudged toward the truth. Especially valuable for HCE,
/// where the evaluator has no mechanism to learn its own systematic errors.
///
/// The key is caller-supplied: any Zobrist slice that isolates a bias
/// worth tracking (pawn structure, non-pawn material, etc.).
/// A single table instance is tied to one key schema; `History` composes several
/// and blends their corrections at lookup time.
///
/// Layout: `[side][key & (N-1)]`. Entries are centipawn corrections
/// scaled by `CORRECTION_SCALE` for fixed-point precision, bounded by
/// `CORRECTION_LIMIT` so no single outlier can dominate.
#[derive(Clone)]
pub struct CorrectionHistory {
    data: Box<[i32]>,
}

impl From<&SearchParams> for HistoryCaps {
    fn from(sp: &SearchParams) -> Self {
        Self {
            quiet: sp.quiet_hist_cap,
            butterfly: sp.butterfly_hist_cap,
            cont: sp.cont_hist_cap,
            capt: sp.capt_hist_cap,
        }
    }
}

impl Default for HistoryCaps {
    fn default() -> Self {
        Self::from(&SearchParams::default())
    }
}

impl Default for ContContext {
    fn default() -> Self {
        Self { pt: PieceType::None, to: Square(0) }
    }
}

impl ContinuationHistory {
    pub fn new() -> Self {
        Self { data: vec![0; 2 * 6 * 64 * 6 * 64].into_boxed_slice() }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline(always)]
    pub fn get(&self, stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> i16 {
        self.data[Self::idx(stm, prev_pt, prev_to, pt, to)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> &mut i16 {
        let idx = Self::idx(stm, prev_pt, prev_to, pt, to);
        &mut self.data[idx]
    }

    #[inline(always)]
    fn idx(stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> usize {
        let mut i = stm as usize;
        i = i * 6 + prev_pt as usize;
        i = i * 64 + prev_to.0 as usize;
        i = i * 6 + pt as usize;
        i = i * 64 + to.0 as usize;
        i
    }
}

impl Default for ContinuationHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CaptureHistory {
    pub fn new() -> Self {
        Self { data: vec![0; 2 * 6 * 64 * 6].into_boxed_slice() }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline(always)]
    pub fn get(&self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> i16 {
        self.data[Self::idx(stm, attacker, to, victim)]
    }

    #[inline(always)]
    pub fn get_mut(&mut self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> &mut i16 {
        let idx = Self::idx(stm, attacker, to, victim);
        &mut self.data[idx]
    }

    #[inline(always)]
    fn idx(stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> usize {
        // Victim must be a real piece (Pawn..Queen, optionally King), never None.
        // Captures only target opponent non-king squares; en passant hardcodes Pawn.
        debug_assert!((attacker as usize) < 6, "attacker out of range");
        debug_assert!((victim as usize) < 6, "victim out of range");
        let mut i = stm as usize;
        i = i * 6 + attacker as usize;
        i = i * 64 + to.0 as usize;
        i = i * 6 + victim as usize;
        i
    }
}

impl Default for CaptureHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrectionHistory {
    pub fn new() -> Self {
        Self { data: vec![0; 2 * CORRECTION_SIZE].into_boxed_slice() }
    }

    pub fn clear(&mut self) {
        self.data.fill(0);
    }

    #[inline(always)]
    fn idx(stm: Color, pawn_hash: u64) -> usize {
        stm as usize * CORRECTION_SIZE + (pawn_hash as usize & (CORRECTION_SIZE - 1))
    }

    #[inline(always)]
    pub fn get(&self, stm: Color, pawn_hash: u64) -> i32 {
        self.data[Self::idx(stm, pawn_hash)]
    }

    /// Weighted moving average update.
    ///
    /// Weight ramps quadratically with depth, capped at 32/256;
    /// a deep search's verdict is far more trustworthy than a shallow one,
    /// so it overwrites a larger share of the entry, while the cap still
    /// keeps any single outlier from dominating.
    #[inline(always)]
    pub fn update(&mut self, stm: Color, pawn_hash: u64, raw_diff: i32, depth: i32) {
        let entry = &mut self.data[Self::idx(stm, pawn_hash)];
        let weight = ((1 + depth) * (1 + depth) / 4).min(32);
        let scaled = i64::from(raw_diff) * i64::from(CORRECTION_SCALE);
        // Re-clamped to ±CORRECTION_LIMIT, so the cast back can't truncate.
        let blended = (i64::from(*entry) * i64::from(256 - weight) + scaled * i64::from(weight)) / 256;
        *entry = blended.clamp(-i64::from(CORRECTION_LIMIT), i64::from(CORRECTION_LIMIT)) as i32;
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    /// Create a zeroed history table.
    pub fn new() -> Self {
        Self {
            table: [[[0; 64]; 6]; 2],
            butterfly: [[[[0; 4096]; 2]; 2]; 2],
            cont: [ContinuationHistory::new(), ContinuationHistory::new()],
            correction: CorrectionHistory::new(),
            minor_correction: CorrectionHistory::new(),
            major_correction: CorrectionHistory::new(),
            capt: CaptureHistory::new(),
            caps: HistoryCaps::default(),
        }
    }

    /// Clear all history scores.
    pub fn clear(&mut self) {
        self.table = [[[0; 64]; 6]; 2];
        self.butterfly = [[[[0; 4096]; 2]; 2]; 2];
        self.cont[0].clear();
        self.cont[1].clear();
        self.correction.clear();
        self.minor_correction.clear();
        self.major_correction.clear();
        self.capt.clear();
    }

    #[inline(always)]
    pub fn score_quiet(
        &self,
        stm: Color,
        pt: PieceType,
        from: Square,
        to: Square,
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
    ) -> i32 {
        let from_atk = threats.check_bit(from) as usize;
        let to_atk = threats.check_bit(to) as usize;
        let mut score =
            i32::from(self.table[stm][pt][to]) + i32::from(self.butterfly[stm][from_atk][to_atk][butterfly_idx(from, to)]);

        for (slot, ctx) in CONT_SLOTS.into_iter().zip([cont1, cont2, cont4]) {
            if ctx.pt != PieceType::None {
                score += i32::from(self.cont[slot].get(stm, ctx.pt, ctx.to, pt, to));
            }
        }
        score
    }

    /// Update a history entry with soft gravity.
    ///
    /// Each update pulls `*entry` toward its table's ±cap with strength proportional to `bonus.abs()`.
    /// Positive `bonus` drives it toward +cap (good move); negative drives it toward −cap (bad move).
    ///
    /// Unlike a plain accumulator, this naturally decays stale information:
    /// a move that caused a cutoff at depth 10 won't permanently dominate over
    /// the same move that failed at depth 8; large bonuses accelerate convergence
    /// both toward and away from extreme values.
    ///
    /// It also holds the table inside `i16` without hard clipping, which would flatten
    /// the ordering of everything sitting near the cap.
    #[inline(always)]
    pub fn update(
        &mut self,
        stm: Color,
        pt: PieceType,
        from: Square,
        to: Square,
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
        bonus: i32,
    ) {
        let from_atk = threats.check_bit(from) as usize;
        let to_atk = threats.check_bit(to) as usize;
        Self::update_entry(&mut self.table[stm][pt][to], bonus, self.caps.quiet);
        Self::update_entry(&mut self.butterfly[stm][from_atk][to_atk][butterfly_idx(from, to)], bonus, self.caps.butterfly);

        self.update_conthist(stm, pt, to, cont1, cont2, cont4, bonus);
    }

    #[inline(always)]
    pub fn update_conthist(
        &mut self,
        stm: Color,
        pt: PieceType,
        to: Square,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
        bonus: i32,
    ) {
        for (slot, ctx) in CONT_SLOTS.into_iter().zip([cont1, cont2, cont4]) {
            if ctx.pt != PieceType::None {
                Self::update_entry(self.cont[slot].get_mut(stm, ctx.pt, ctx.to, pt, to), bonus, self.caps.cont);
            }
        }
    }

    /// Single soft-gravity update step.
    #[inline(always)]
    fn update_entry(entry: &mut i16, bonus: i32, cap: i32) {
        debug_assert!(cap > 0, "a zero cap divides by zero in the gravity term");
        let e = i32::from(*entry);
        *entry = (e + bonus - e * bonus.abs() / cap).clamp(-cap, cap) as i16;
    }

    /// Blended correction. Pawn structure anchors; minor and major placement refine.
    ///
    /// The tables are estimates of one number: the gap between static eval and
    /// what search found, each keyed on a different read of the position. They
    /// correlate hard. A position that fools one usually fools the rest the same
    /// way, so summing counts that shared error once per table and the correction
    /// balloons the instant another joins. Average instead.
    ///
    /// Pawn stays whole; the placement tables fold into a weight-normalized pool,
    /// so the pool's size holds no matter how many feed it. The weights are ratios,
    /// not magnitudes: normalization cancels their scale and only the split survives.
    #[inline(always)]
    pub fn correction(
        &self,
        stm: Color,
        pawn_hash: u64,
        minor_hash: u64,
        major_hash: u64,
        minor_weight: i32,
        major_weight: i32,
    ) -> i32 {
        let pawn = self.correction.get(stm, pawn_hash);
        let minor = self.minor_correction.get(stm, minor_hash);
        let major = self.major_correction.get(stm, major_hash);

        #[cfg(feature = "corrstats")]
        {
            use crate::engine::corrstats::{Table, record_read};
            record_read(Table::Pawn, pawn);
            record_read(Table::Minor, minor);
            record_read(Table::Major, major);
        }

        let refine = (minor * minor_weight + major * major_weight) / (minor_weight + major_weight);
        pawn + refine
    }

    #[inline(always)]
    pub fn update_correction(&mut self, stm: Color, pawn_hash: u64, minor_hash: u64, major_hash: u64, diff: i32, depth: i32) {
        self.correction.update(stm, pawn_hash, diff, depth);
        self.minor_correction.update(stm, minor_hash, diff, depth);
        self.major_correction.update(stm, major_hash, diff, depth);

        #[cfg(feature = "corrstats")]
        {
            use crate::engine::corrstats::{Table, record_update};
            record_update(Table::Pawn);
            record_update(Table::Minor);
            record_update(Table::Major);
        }
    }

    #[inline(always)]
    pub fn score_capture(&self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> i32 {
        i32::from(self.capt.get(stm, attacker, to, victim))
    }

    #[inline(always)]
    pub fn update_capture(&mut self, stm: Color, attacker: PieceType, to: Square, victim: PieceType, bonus: i32) {
        Self::update_entry(self.capt.get_mut(stm, attacker, to, victim), bonus, self.caps.capt);
    }
}

impl Default for History {
    /// Returns a zero-cost sentinel.
    ///
    /// # Panics
    /// Indexing into `cont[-]`, `correction[-]`, or `capt` on a default
    /// `History` panics; the internal tables are empty boxes.
    /// Only use `Default::default()` as a placeholder for `mem::take`.
    fn default() -> Self {
        Self {
            table: [[[0; 64]; 6]; 2],
            butterfly: [[[[0; 4096]; 2]; 2]; 2],
            cont: [ContinuationHistory { data: Box::new([]) }, ContinuationHistory { data: Box::new([]) }],
            correction: CorrectionHistory { data: Box::new([]) },
            minor_correction: CorrectionHistory { data: Box::new([]) },
            major_correction: CorrectionHistory { data: Box::new([]) },
            capt: CaptureHistory { data: Box::new([]) },
            caps: HistoryCaps::default(),
        }
    }
}

#[inline(always)]
fn butterfly_idx(from: Square, to: Square) -> usize {
    from.0 as usize * 64 + to.0 as usize
}
