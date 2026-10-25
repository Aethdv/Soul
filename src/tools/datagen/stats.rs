//! Counters shared by the self-play workers and the output writer.

use std::{sync::atomic::AtomicU64, time::Instant};

use crate::core::util::Align64;

pub(super) struct RunStats {
    pub(super) search_failures: Align64<AtomicU64>,
    pub(super) games: Align64<AtomicU64>,
    pub(super) positions: Align64<AtomicU64>,
    pub(super) checkmates: Align64<AtomicU64>,
    pub(super) stalemates: Align64<AtomicU64>,
    pub(super) fifty_move_draws: Align64<AtomicU64>,
    pub(super) repetition_draws: Align64<AtomicU64>,
    pub(super) material_draws: Align64<AtomicU64>,
    pub(super) ply_cap_draws: Align64<AtomicU64>,
    pub(super) adjudicated_draws: Align64<AtomicU64>,
    pub(super) adjudicated_wins: Align64<AtomicU64>,
    pub(super) start_time: Instant,
}

pub(super) fn rss_kib() -> u64 {
    let Ok(status) = std::fs::read_to_string("/proc/self/status") else { return 0 };
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kib| kib.parse().ok())
        .unwrap_or(0)
}

impl Default for RunStats {
    fn default() -> Self { Self::new() }
}

impl RunStats {
    pub(super) fn new() -> Self {
        Self {
            search_failures: Align64::new(AtomicU64::new(0)),
            games: Align64::new(AtomicU64::new(0)),
            positions: Align64::new(AtomicU64::new(0)),
            checkmates: Align64::new(AtomicU64::new(0)),
            stalemates: Align64::new(AtomicU64::new(0)),
            fifty_move_draws: Align64::new(AtomicU64::new(0)),
            repetition_draws: Align64::new(AtomicU64::new(0)),
            material_draws: Align64::new(AtomicU64::new(0)),
            ply_cap_draws: Align64::new(AtomicU64::new(0)),
            adjudicated_draws: Align64::new(AtomicU64::new(0)),
            adjudicated_wins: Align64::new(AtomicU64::new(0)),
            start_time: Instant::now(),
        }
    }
}
