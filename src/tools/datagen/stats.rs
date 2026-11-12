//! Session counters and termination tallies.

use std::{
    array, fs,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use crate::core::util::{pct, ratio};

#[derive(Clone, Copy)]
pub(super) enum Termination {
    Checkmate,
    Stalemate,
    FiftyMove,
    Repetition,
    Material,
    PlyCap,
    AdjudicatedDraw,
    AdjudicatedWin,
}

impl Termination {
    pub(super) const ALL: [Self; 8] = [
        Self::Checkmate,
        Self::Stalemate,
        Self::FiftyMove,
        Self::Repetition,
        Self::Material,
        Self::PlyCap,
        Self::AdjudicatedDraw,
        Self::AdjudicatedWin,
    ];

    pub(super) const fn label(self) -> &'static str {
        match self {
            Self::Checkmate => "Checkmates",
            Self::Stalemate => "Stalemates",
            Self::FiftyMove => "50-move draws",
            Self::Repetition => "Repetition draws",
            Self::Material => "Material draws",
            Self::PlyCap => "Ply-cap draws",
            Self::AdjudicatedDraw => "Adjudicated draws",
            Self::AdjudicatedWin => "Adjudicated wins",
        }
    }
}

pub(super) struct RunStats {
    start_time:         Instant,
    existing_positions: u64,
    target_count:       u64,
    games:              AtomicU64,
    positions:          AtomicU64,
    search_failures:    AtomicU64,
    terminations:       [AtomicU64; Termination::ALL.len()],
}

pub(super) struct Snapshot {
    pub(super) elapsed:         f64,
    pub(super) total_positions: u64,
    pub(super) target_count:    u64,
    pub(super) games:           u64,
    pub(super) positions:       u64,
    pub(super) search_failures: u64,
    terminations:               [u64; Termination::ALL.len()],
}

impl RunStats {
    pub(super) fn new(existing_positions: u64, target_count: u64) -> Self {
        Self {
            start_time: Instant::now(),
            existing_positions,
            target_count,
            games: AtomicU64::new(0),
            positions: AtomicU64::new(0),
            search_failures: AtomicU64::new(0),
            terminations: Default::default(),
        }
    }

    pub(super) fn record(&self, plies: u64, termination: Termination, failures: u64) -> u64 {
        self.games.fetch_add(1, Ordering::Relaxed);
        let prev = self.positions.fetch_add(plies, Ordering::Relaxed);
        self.search_failures.fetch_add(failures, Ordering::Relaxed);
        self.terminations[termination as usize].fetch_add(1, Ordering::Relaxed);
        self.existing_positions + prev + plies
    }

    pub(super) fn is_done(&self) -> bool { self.existing_positions + self.positions.load(Ordering::Relaxed) >= self.target_count }

    pub(super) fn snapshot(&self) -> Snapshot {
        let positions = self.positions.load(Ordering::Relaxed);
        Snapshot {
            elapsed: self.start_time.elapsed().as_secs_f64(),
            total_positions: self.existing_positions + positions,
            target_count: self.target_count,
            games: self.games.load(Ordering::Relaxed),
            positions,
            search_failures: self.search_failures.load(Ordering::Relaxed),
            terminations: array::from_fn(|i| self.terminations[i].load(Ordering::Relaxed)),
        }
    }
}

impl Snapshot {
    pub(super) fn count(&self, t: Termination) -> u64 { self.terminations[t as usize] }
    pub(super) fn rate(&self) -> f64 { ratio(self.positions as f64, self.elapsed) }
    pub(super) fn progress_pct(&self) -> f64 { pct(self.total_positions, self.target_count) }
    pub(super) fn avg_ply(&self) -> f64 { ratio(self.positions as f64, self.games as f64) }
    pub(super) fn eta_secs(&self) -> f64 { ratio(self.target_count.saturating_sub(self.total_positions) as f64, self.rate()) }

    pub(super) fn rule_draws(&self) -> u64 {
        [
            Termination::Stalemate,
            Termination::FiftyMove,
            Termination::Repetition,
            Termination::Material,
            Termination::PlyCap,
        ]
        .into_iter()
        .map(|t| self.count(t))
        .sum()
    }
}

pub(super) fn rss_kib() -> u64 {
    let Ok(status) = fs::read_to_string("/proc/self/status") else { return 0 };
    status
        .lines()
        .find(|line| line.starts_with("VmRSS:"))
        .and_then(|line| line.split_whitespace().nth(1))
        .and_then(|kib| kib.parse().ok())
        .unwrap_or(0)
}
