//! History heuristic for quiet move ordering.
//!
//! # Design
//!
//! Tracks which moves historically caused beta cutoffs in the search tree.
//! High-scoring moves are sorted to the front of the move list, maximizing
//! the probability of early cutoffs in subsequent searches.

use crate::core::defs::{Color, PieceType, Square};

/// History table indexed by `[side][piece][to_square]`.
#[derive(Clone, Copy)]
pub struct History {
    /// Scores are soft-gravity bounded to `[-16384, 16384]` by `update()`.
    /// Stored as `i16` to halve cache footprint; widened to `i32` on reads.
    pub table: [[[i16; 64]; 6]; 2],
}

impl History {
    /// Create a zeroed history table.
    pub fn new() -> Self {
        Self {
            table: [[[0; 64]; 6]; 2],
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
    pub fn score_quiet(&self, stm: Color, pt: PieceType, to: Square) -> i32 {
        i32::from(self.table[stm][pt][to])
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
    pub fn update(&mut self, stm: Color, pt: PieceType, to: Square, bonus: i32) {
        let entry = &mut self.table[stm][pt][to];
        let e = i32::from(*entry);
        let new_val = (e + bonus - e * bonus.abs() / 16384).clamp(-16384, 16384);
        *entry = new_val as i16;
    }
}

impl Default for History {
    fn default() -> Self {
        Self::new()
    }
}
