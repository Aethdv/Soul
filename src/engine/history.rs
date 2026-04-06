//! History heuristic for quiet move ordering.
//!
//! # Design
//!
//! Tracks which moves historically caused beta cutoffs in the search tree.
//! High-scoring moves are sorted to the front of the move list, maximizing
//! the probability of early cutoffs in subsequent searches.

use crate::core::defs::{Color, PieceType, Square};

#[derive(Clone)]
pub struct ContinuationHistory {
    data: Box<[i16]>,
}

impl ContinuationHistory {
    pub fn new() -> Self {
        Self {
            data: vec![0; 2 * 6 * 64 * 6 * 64].into_boxed_slice(),
        }
    }

    #[inline(always)]
    pub fn get(&self, stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> i16 {
        self.data[Self::idx(stm, prev_pt, prev_to, pt, to)]
    }

    #[inline(always)]
    pub fn get_mut(
        &mut self,
        stm: Color,
        prev_pt: PieceType,
        prev_to: Square,
        pt: PieceType,
        to: Square,
    ) -> &mut i16 {
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

/// Combined history tables for quiet move ordering.
/// Queries return the sum of all components, bounded to `[-32768, 32768]`.
#[derive(Clone)]
pub struct History {
    /// `[side][piece][to_square]` — bounds `[-16384, 16384]`
    table:     [[[i16; 64]; 6]; 2],
    /// `[side][from · 64 + to]` — bounds `[-16384, 16384]`
    butterfly: [[i16; 4096]; 2], // ~20 Elo
    /// `[side][prev_piece][prev_to][piece][to]`
    cont:      ContinuationHistory,
}

impl History {
    /// Create a zeroed history table.
    pub fn new() -> Self {
        Self {
            table:     [[[0; 64]; 6]; 2],
            butterfly: [[0; 4096]; 2],
            cont:      ContinuationHistory::new(),
        }
    }

    /// Clear all history scores.
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Retrieve the history score for a move.
    ///
    /// Used in `MovePicker` to prioritize moves that previously caused beta cutoffs.
    /// The scores are updated in `Searcher` using the soft-gravity mechanism.
    #[inline(always)]
    pub fn score_quiet(
        &self,
        stm: Color,
        pt: PieceType,
        from: Square,
        to: Square,
        prev_pt: PieceType,
        prev_to: Square,
    ) -> i32 {
        let mut score = i32::from(self.table[stm][pt][to])
            + i32::from(self.butterfly[stm][(from.0 as usize) * 64 + (to.0 as usize)]);

        if prev_pt != PieceType::None {
            score += i32::from(self.cont.get(stm, prev_pt, prev_to, pt, to));
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
    /// the same move that failed at depth 8 — large bonuses accelerate convergence
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
        prev_pt: PieceType,
        prev_to: Square,
        bonus: i32,
    ) {
        Self::update_entry(&mut self.table[stm][pt][to], bonus);
        Self::update_entry(&mut self.butterfly[stm][(from.0 as usize) * 64 + (to.0 as usize)], bonus);

        if prev_pt != PieceType::None {
            Self::update_entry(self.cont.get_mut(stm, prev_pt, prev_to, pt, to), bonus);
        }
    }

    /// Single soft-gravity update step. Extracted to keep updates DRY.
    #[inline(always)]
    fn update_entry(entry: &mut i16, bonus: i32) {
        let e = i32::from(*entry);
        *entry = (e + bonus - e * bonus.abs() / 16384).clamp(-16384, 16384) as i16;
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}
