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

use crate::core::defs::{Bitboard, Color, PieceType, Square};

pub const CORRECTION_SIZE: usize = 16384;
pub const CORRECTION_SCALE: i32 = 256;
pub const CORRECTION_LIMIT: i32 = 256 * 32;
/// Denominator for the per-table blend weights: `weight / this` is a table's share.
pub const CORRECTION_WEIGHT_SCALE: i32 = 256;

const _: () = assert!(CORRECTION_SIZE.is_power_of_two());

/// Combined history tables for move ordering.
#[derive(Clone)]
pub struct History {
    /// `[side][piece][to_square]`: bounds `[-16384, 16384]`
    table: [[[i16; 64]; 6]; 2],
    /// `[side][from_atk][to_atk][from · 64 + to]`: bounds `[-16384, 16384]`
    butterfly: [[[[i16; 4096]; 2]; 2]; 2], // ~35 Elo
    /// `[ply_offset][side][prev_piece][prev_to][piece][to]`. Two slots, three distances:
    /// n-1 owns slot 0, while n-2 and n-4 deliberately share slot 1, pooling into one table.
    cont: [ContinuationHistory; 2], // n-1 (~13 Elo), n-2 (~3 Elo), n-4 (~3 Elo)
    /// `[side][pawn_hash & 0x3FFF]`
    correction: CorrectionHistory, // ~53 Elo
    /// `[side][minor_hash & 0x3FFF]`: knights + bishops, both colors
    minor_correction: CorrectionHistory, // minor + major split from lumped non-pawn (~18 Elo): net ~5 Elo
    /// `[side][major_hash & 0x3FFF]`: rooks + queens, both colors
    major_correction: CorrectionHistory,
    /// `[side][attacker][to][victim]`
    capt: CaptureHistory, // ~8 Elo
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
        let scaled = raw_diff * CORRECTION_SCALE;

        *entry = ((*entry * (256 - weight) + scaled * weight) / 256).clamp(-CORRECTION_LIMIT, CORRECTION_LIMIT);
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
        let mut score = i32::from(self.table[stm][pt][to])
            + i32::from(self.butterfly[stm][from_atk][to_atk][(from.0 as usize) * 64 + (to.0 as usize)]);

        if cont1.pt != PieceType::None {
            score += i32::from(self.cont[0].get(stm, cont1.pt, cont1.to, pt, to));
        }
        if cont2.pt != PieceType::None {
            score += i32::from(self.cont[1].get(stm, cont2.pt, cont2.to, pt, to));
        }
        if cont4.pt != PieceType::None {
            score += i32::from(self.cont[1].get(stm, cont4.pt, cont4.to, pt, to));
        }
        score
    }

    /// Update a history entry with soft gravity.
    ///
    /// Each update pulls `*entry` toward ±16384 with strength proportional to `bonus.abs()`.
    /// Positive `bonus` drives it toward +16384 (good move); negative drives it toward −16384 (bad move).
    ///
    /// Unlike a plain accumulator, this naturally decays stale information:
    /// a move that caused a cutoff at depth 10 won't permanently dominate over
    /// the same move that failed at depth 8; large bonuses accelerate convergence
    /// both toward and away from extreme values.
    ///
    /// This mechanism, sometimes called "history aging" or "soft clamping",
    /// ensures the table stays within `i16` bounds without hard-clipping
    /// destroying the relative ordering of values near the attractor limit.
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
        Self::update_entry(&mut self.table[stm][pt][to], bonus);
        Self::update_entry(&mut self.butterfly[stm][from_atk][to_atk][(from.0 as usize) * 64 + (to.0 as usize)], bonus);

        if cont1.pt != PieceType::None {
            Self::update_entry(self.cont[0].get_mut(stm, cont1.pt, cont1.to, pt, to), bonus);
        }
        if cont2.pt != PieceType::None {
            Self::update_entry(self.cont[1].get_mut(stm, cont2.pt, cont2.to, pt, to), bonus);
        }
        if cont4.pt != PieceType::None {
            Self::update_entry(self.cont[1].get_mut(stm, cont4.pt, cont4.to, pt, to), bonus);
        }
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
        if cont1.pt != PieceType::None {
            Self::update_entry(self.cont[0].get_mut(stm, cont1.pt, cont1.to, pt, to), bonus);
        }
        if cont2.pt != PieceType::None {
            Self::update_entry(self.cont[1].get_mut(stm, cont2.pt, cont2.to, pt, to), bonus);
        }
        if cont4.pt != PieceType::None {
            Self::update_entry(self.cont[1].get_mut(stm, cont4.pt, cont4.to, pt, to), bonus);
        }
    }

    /// Single soft-gravity update step. Extracted to keep updates DRY.
    #[inline(always)]
    fn update_entry(entry: &mut i16, bonus: i32) {
        let e = i32::from(*entry);
        *entry = (e + bonus - e * bonus.abs() / 16384).clamp(-16384, 16384) as i16;
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
        Self::update_entry(self.capt.get_mut(stm, attacker, to, victim), bonus);
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
        }
    }
}
