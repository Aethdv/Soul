//! Global statistics tracking for the self-play workers.

use std::{sync::atomic::AtomicU64, time::Instant};

pub fn get_rss_kb() -> u64 {
    match std::fs::read_to_string("/proc/self/status") {
        Ok(content) => {
            for line in content.lines() {
                if line.starts_with("VmRSS:") {
                    return line.split_whitespace().nth(1).and_then(|val| val.parse().ok()).unwrap_or(0);
                }
            }
            0
        },
        Err(_) => 0,
    }
}

use crate::core::util::Align64;

/// Global statistics tracking for the self-play workers.
///
/// Contains aggressively padded atomic counters to prevent false sharing
/// across the N worker threads blasting random games during dataset generation.
pub struct GlobalStats {
    /// Total number of unique root FENs successfully generated.
    pub attempted: Align64<AtomicU64>,
    /// How many fully played out dataset games were actually written to disk.
    pub saved: Align64<AtomicU64>,
    /// How many positions passed all non-stochastic quality filters.
    pub passed_filters: Align64<AtomicU64>,
    /// Games discarded because the concluding move was a quiet move (prevents horizon effect).
    pub filtered_quiet: Align64<AtomicU64>,
    /// Positions filtered because |search_eval| exceeded the score window.
    pub filtered_score: Align64<AtomicU64>,
    /// Positions filtered: ply count too low.
    pub filtered_ply: Align64<AtomicU64>,
    /// Positions filtered: too few pieces remaining.
    pub filtered_pieces: Align64<AtomicU64>,
    /// Positions filtered: eval contradicted game outcome.
    pub filtered_incorrect: Align64<AtomicU64>,
    /// Positions filtered: |search - static| delta exceeded the qsearch threshold.
    pub filtered_tactical: Align64<AtomicU64>,
    /// Number of times the search aborted mid-game due to depth boundaries or extreme scores.
    pub search_fail: Align64<AtomicU64>,
    /// Total number of raw chess games completed.
    pub games: Align64<AtomicU64>,
    /// Cumulative half-moves played across all valid games.
    pub plies: Align64<AtomicU64>,
    /// Number of games terminated by checkmate.
    pub term_check: Align64<AtomicU64>,
    /// Number of games terminated by stalemate.
    pub term_stale: Align64<AtomicU64>,
    /// Number of games drawn via the 50-move rule horizon.
    pub term_d50: Align64<AtomicU64>,
    /// Number of games drawn via 3-fold repetition.
    pub term_drep: Align64<AtomicU64>,
    /// Number of games drawn due to insufficient mating material.
    pub term_dmat: Align64<AtomicU64>,
    /// Draw adjudicated by the engine's static evaluation plateau.
    pub term_draw_adj: Align64<AtomicU64>,
    /// Win/Loss adjudicated by overwhelming evaluation advantage persisting across plies.
    pub term_resign: Align64<AtomicU64>,
    /// The precise wallclock timestamp when the worker pool was constructed.
    pub start_time: Instant,
}

impl Default for GlobalStats {
    fn default() -> Self {
        Self::new()
    }
}

impl GlobalStats {
    pub fn new() -> Self {
        Self {
            attempted: Align64::new(AtomicU64::new(0)),
            saved: Align64::new(AtomicU64::new(0)),
            passed_filters: Align64::new(AtomicU64::new(0)),
            filtered_quiet: Align64::new(AtomicU64::new(0)),
            filtered_score: Align64::new(AtomicU64::new(0)),
            filtered_ply: Align64::new(AtomicU64::new(0)),
            filtered_pieces: Align64::new(AtomicU64::new(0)),
            filtered_incorrect: Align64::new(AtomicU64::new(0)),
            filtered_tactical: Align64::new(AtomicU64::new(0)),
            search_fail: Align64::new(AtomicU64::new(0)),
            games: Align64::new(AtomicU64::new(0)),
            plies: Align64::new(AtomicU64::new(0)),
            term_check: Align64::new(AtomicU64::new(0)),
            term_stale: Align64::new(AtomicU64::new(0)),
            term_d50: Align64::new(AtomicU64::new(0)),
            term_drep: Align64::new(AtomicU64::new(0)),
            term_dmat: Align64::new(AtomicU64::new(0)),
            term_draw_adj: Align64::new(AtomicU64::new(0)),
            term_resign: Align64::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }
}
