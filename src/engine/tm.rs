//! Time management and limits tracking.
//!
//! Monitors elapsed time against soft and hard constraints, calculating
//! phase-adjusted time budgets based on the time control.

use std::time::{Duration, Instant};

use crate::{
    core::defs::{Color, TOTAL_PHASE},
    engine::{search::Limits, search_params::SearchParams},
};

pub const MIN_MTG: u64 = 2;
pub const MAX_MTG: u64 = 100;
/// Fraction of remaining time used as the soft limit per move (5%).
/// At this budget we expect roughly 20 more moves before flagging.
pub const TIME_SOFT_CAP: f64 = 0.05;
/// Absolute hard limit as a fraction of remaining time (50%).
/// Never spend more than half the clock on a single move.
pub const TIME_HARD_CAP: f64 = 0.5;

pub struct TimeManager {
    start: Instant,
    hard:  Duration,
    soft:  Duration,
}

impl TimeManager {
    pub fn new(
        limits: &Limits,
        start: Instant,
        stm: Color,
        overhead: u64,
        phase: i32,
        params: &SearchParams,
    ) -> Self {
        let (time, inc, moves_to_go) = if stm == Color::White {
            (limits.wtime, limits.winc, limits.movestogo)
        } else {
            (limits.btime, limits.binc, limits.movestogo)
        };

        if limits.infinite {
            return Self {
                start,
                hard: Duration::MAX,
                soft: Duration::MAX,
            };
        }

        if limits.movetime > 0 {
            let limit_ms = limits.movetime.saturating_sub(overhead).max(1);
            let limit = Duration::from_millis(limit_ms);

            return Self {
                start,
                hard: limit,
                soft: limit,
            };
        };

        if time == 0 && inc == 0 {
            return Self {
                start,
                hard: Duration::MAX,
                soft: Duration::MAX,
            };
        }

        // Phase-based Moves To Go estimation.
        // The deeper into the game we go, the fewer moves remain.
        // Interpolating by phase prevents overspending time in the early game
        // while tightening the budget in the endgame where bounds are stricter.
        let mtg = if moves_to_go > 0 {
            moves_to_go.clamp(MIN_MTG, MAX_MTG)
        } else {
            let open = params.mtg_opening as i64;
            let end = params.mtg_endgame as i64;
            let p = (phase as i64).clamp(0, TOTAL_PHASE as i64);

            let mtg_raw = end + (open - end) * p / (TOTAL_PHASE as i64);
            mtg_raw.clamp(MIN_MTG as i64, MAX_MTG as i64) as u64
        };

        let hard_limit = (((time as f64 * TIME_HARD_CAP) as u64).saturating_add(inc)).min(time);
        let base_alloc = ((time as f64) / (mtg as f64)) as u64;
        let soft_limit = (base_alloc.min(((time as f64) * TIME_SOFT_CAP) as u64) + inc / 2).min(hard_limit);

        if limits.movestogo > 0 {
            const SUDDEN_DEATH_THRESHOLD: u64 = 20;
            const NORMALIZATION_FACTOR: u64 = 60;
            const MIN_TIME_LIMIT_MS: u64 = 10;

            let mtg_limit = limits.movestogo.min(40);
            // Interpolate between "blitz-style" and "sudden-death" regimes.
            // When moves remaining is low, we conservatively scale down the budget to avoid flagging.
            let adjusted_limit = (soft_limit * (mtg_limit + SUDDEN_DEATH_THRESHOLD) / NORMALIZATION_FACTOR)
                .max(MIN_TIME_LIMIT_MS);

            let soft_ms = adjusted_limit.saturating_sub(overhead).max(1);
            let hard_ms = hard_limit.saturating_sub(overhead).max(1);

            return Self {
                start,
                soft: Duration::from_millis(soft_ms),
                hard: Duration::from_millis(hard_ms),
            };
        }

        let soft_ms = soft_limit.saturating_sub(overhead).max(1);
        let hard_ms = hard_limit.saturating_sub(overhead).max(1);

        Self {
            start,
            soft: Duration::from_millis(soft_ms),
            hard: Duration::from_millis(hard_ms),
        }
    }

    #[inline]
    pub fn is_hard_limit_reached(&self) -> bool {
        self.start.elapsed() >= self.hard
    }

    #[inline]
    pub fn is_soft_limit_reached(&self) -> bool {
        self.start.elapsed() >= self.soft
    }

    #[inline]
    pub fn soft_limit(&self) -> Duration {
        self.soft
    }

    #[inline]
    pub fn hard_limit(&self) -> Duration {
        self.hard
    }

    #[inline]
    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }
}
