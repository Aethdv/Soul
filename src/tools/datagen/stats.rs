//! Global statistics tracking for the self-play workers.

use std::{sync::atomic::AtomicU64, time::Instant};

use crate::core::util::Align64;

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

/// Global statistics tracking for the self-play workers.
///
/// Contains atomic counters to prevent false sharing across the N worker
/// threads blasting random games during dataset generation.
pub struct GlobalStats {
    /// Number of times the search aborted mid-game due to depth boundaries or extreme scores.
    pub search_fail: Align64<AtomicU64>,
    /// Total number of chess games completed.
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
    /// Wallclock timestamp when the worker pool was constructed.
    pub start_time: Instant,
}

impl Default for GlobalStats {
    fn default() -> Self { Self::new() }
}

impl GlobalStats {
    pub fn new() -> Self {
        Self {
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
