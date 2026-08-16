//! History heuristics for move ordering and static evaluation correction.
//!
//! A move that caused a beta cutoff once tends to again, so the search records the success
//! and orders by it next time. Four tables do that, each keyed on a different slice of
//! context:
//!
//! - main: `[side][piece][to]`, quiet moves.
//! - butterfly: `[side][from_atk][to_atk][from · 64 + to]`, the same quiets split by whether
//!   the from- and to-squares are under threat.
//! - continuation: a recent move at 1, 2 or 4 plies back paired with the current reply.
//! - capture: `[side][attacker][to][victim]`, for noisy moves.
//!
//! Correction history is the odd one out: it orders nothing. It tracks the error between
//! static eval and what search found, keyed on pawn, minor and major structure, and nudges
//! the eval toward the truth. It learns the way the others do, from search outcomes, which
//! is why it sits here, and it is the heaviest single Elo contributor in the file.

use crate::{
    core::defs::{Bitboard, Color, PieceType, Square},
    engine::search_params::SearchParams,
};

pub const CORRECTION_SIZE: usize = 16384;
pub const CORRECTION_SCALE: i32 = 256;
pub const CORRECTION_LIMIT: i32 = 256 * 32;
/// Denominator for correction table blend weights: `weight / CORRECTION_WEIGHT_SCALE`.
pub const CORRECTION_WEIGHT_SCALE: i32 = 256;

const _: () = assert!(CORRECTION_SIZE.is_power_of_two());

/// Continuation table indices for [n-1, n-2, n-4] plies back.
const CONT_SLOTS: [usize; 3] = [0, 1, 1];

/// Per-table soft-gravity saturation caps, refreshed from `SearchParams` each search.
///
/// The same value bounds an entry and divides the gravity term, so the two move together.
/// Must stay in `(0, i16::MAX]`: zero divides by zero, and past `i16::MAX` the `as i16`
/// store overflows.
#[derive(Clone, Copy)]
pub struct HistoryCaps {
    pub quiet: i32,
    pub butterfly: i32,
    pub cont: i32,
    pub capt: i32,
}

/// Combined move-ordering and evaluation-correction history tables.
#[derive(Clone)]
pub struct History {
    /// `[side][piece][to_square]` bounded to ±cap.
    table: [[[i16; 64]; 6]; 2],
    /// `[side][from_atk][to_atk][from · 64 + to]` bounded to ±cap (~35 Elo).
    butterfly: [[[[i16; 4096]; 2]; 2]; 2],
    /// Slot 0 is n-1 (~13 Elo); slot 1 is shared by n-2 (~3 Elo) and n-4 (~3 Elo) on purpose.
    cont: [ContinuationHistory; 2],
    /// Pawn structure correction: `[side][pawn_hash & 0x3FFF]` (~53 Elo).
    correction: CorrectionHistory,
    /// Minor placement (knights + bishops, both colors): `[side][minor_hash & 0x3FFF]`.
    /// Minor and major together ~23 Elo.
    minor_correction: CorrectionHistory,
    /// Major piece placement correction (rooks + queens, both colors): `[side][major_hash & 0x3FFF]`.
    major_correction: CorrectionHistory,
    /// Capture history: `[side][attacker][to][victim]` (~8 Elo).
    capt: CaptureHistory,
    /// Dynamic soft-gravity saturation caps synchronized with search parameters.
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

/// Capture history table tracking cutoff frequencies for tactical moves.
///
/// Indexed by `[side][attacker][to_square][victim]`.
/// Plain captures and en passant are recorded; promotion captures are not, because the
/// picker orders them by promotion piece outside the MVV-LVA path. Update and read skip
/// them alike, so the table stays consistent.
#[derive(Clone)]
pub struct CaptureHistory {
    data: Box<[i16]>,
}

/// Fixed-point exponential moving average (EMA) table for static evaluation correction.
///
/// Learns systematic evaluator bias by tracking `search_score - static_eval`. The key is
/// the caller's: any Zobrist slice that isolates a bias worth tracking. One instance serves
/// one key schema, and `History` composes several and blends them at lookup.
///
/// Worth most to a hand-crafted eval, which has no other way to learn its own mistakes.
///
/// Indexed by `[side][hash & (CORRECTION_SIZE - 1)]`. Scores are fixed-point centipawns
/// scaled by [`CORRECTION_SCALE`] and clamped to `±CORRECTION_LIMIT`.
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
    fn idx(stm: Color, hash: u64) -> usize {
        stm as usize * CORRECTION_SIZE + (hash as usize & (CORRECTION_SIZE - 1))
    }

    #[inline(always)]
    pub fn get(&self, stm: Color, hash: u64) -> i32 {
        self.data[Self::idx(stm, hash)]
    }

    /// Updates the moving average with a new search observation.
    ///
    /// The update weight scales quadratically with depth up to a limit of 32/256:
    /// `weight = min((1 + depth)^2 / 4, 32)`
    #[inline(always)]
    pub fn update(&mut self, stm: Color, hash: u64, raw_diff: i32, depth: i32) {
        let entry = &mut self.data[Self::idx(stm, hash)];
        let weight = ((1 + depth) * (1 + depth) / 4).min(32);
        let scaled = i64::from(raw_diff) * i64::from(CORRECTION_SCALE);
        let blended = (i64::from(*entry) * i64::from(256 - weight) + scaled * i64::from(weight)) / 256;
        // Re-clamped to ±CORRECTION_LIMIT, so the cast back cannot truncate.
        *entry = blended.clamp(-i64::from(CORRECTION_LIMIT), i64::from(CORRECTION_LIMIT)) as i32;
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl History {
    /// Creates an allocated, zero-initialized history table set.
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

    /// Resets all history scores and corrections to zero.
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

    /// Aggregates quiet move ordering scores from main, butterfly, and continuation tables.
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

    /// Updates main, butterfly and continuation entries with soft gravity: each pulls toward
    /// its table's ±cap with strength proportional to `bonus.abs()`, positive toward +cap and
    /// negative toward −cap. That decays stale information, so a cutoff at depth 10 cannot
    /// permanently outrank the same move failing at depth 8, and it holds the table inside
    /// `i16` without hard clipping, which would flatten the ordering of everything near the cap.
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

    /// Updates continuation history entries for 1, 2, and 4 plies back.
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

    /// Applies a soft-gravity update step:
    /// `entry = entry + bonus - entry * |bonus| / cap`
    ///
    /// Drives values toward `±cap` while decaying older entries without hard clipping.
    #[inline(always)]
    fn update_entry(entry: &mut i16, bonus: i32, cap: i32) {
        debug_assert!(cap > 0, "cap must be positive");
        let e = i32::from(*entry);
        *entry = (e + bonus - e * bonus.abs() / cap).clamp(-cap, cap) as i16;
    }

    /// Computes the blended static evaluation correction in fixed-point centipawns.
    ///
    /// Pawn correction is added directly, while minor and major piece corrections
    /// are combined via a normalized weighted average to avoid over-counting
    /// correlated structural errors.
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

    /// Updates all three evaluation correction tables (pawn, minor, major) with a search delta.
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
    /// Returns an unallocated sentinel `History` instance.
    ///
    /// # Panics
    /// Accessing table entries on a default instance will panic because heap buffers are empty.
    /// Use [`History::new()`] for active search instances.
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
