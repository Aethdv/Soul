//! Search time allocation and limit management.
//!
//! Allocates soft and hard time budgets derived from protocol limits and tracks
//! wall-clock consumption during iterative deepening.
//!
//! # Limits
//! - Soft: Target duration for deciding whether to start a new search iteration.
//!   Dynamically adjusted during search via move stability and score fluctuation factors.
//! - Hard: Upper bound where ongoing search operations must immediately abort to avoid
//!   timing out.
//!
//! # Allocation Precedence
//! 1. `infinite` or unclocked controls run unbounded (`Duration::MAX`).
//! 2. `movetime` allocates a fixed duration minus move overhead.
//! 3. Clocked controls allocate based on phase interpolation (sudden death) or
//!    moves-to-go (repeating time controls), incorporating increments.

use std::time::{Duration, Instant};

use crate::{
    core::defs::{Color, TOTAL_PHASE},
    engine::{search::Limits, search_params::SearchParams},
};

/// Tracks elapsed search time against soft and hard limits.
pub struct TimeManager {
    start: Instant,
    hard: Duration,
    soft: Duration,
    base_soft: Duration,
    bm_stab: f64,
    bm_inst: f64,
    score_factor: f64,
    fail_low: f64,
}

impl TimeManager {
    pub fn new(
        limits: &Limits,
        start: Instant,
        stm: Color,
        overhead: u64,
        phase: i32,
        game_ply: u64,
        params: &SearchParams,
    ) -> Self {
        let (soft, hard) = compute_budget(limits, stm, overhead, phase, game_ply, params);
        Self { start, soft, hard, base_soft: soft, bm_stab: 1.0, bm_inst: 1.0, score_factor: 1.0, fail_low: 1.0 }
    }

    #[inline]
    pub fn is_hard_limit_reached(&self) -> bool { self.start.elapsed() >= self.hard }

    #[inline]
    pub fn soft_limit(&self) -> Duration { self.soft }

    #[inline]
    pub fn hard_limit(&self) -> Duration { self.hard }

    #[inline]
    pub fn is_finite_budget(&self) -> bool { self.hard != Duration::MAX }

    #[inline]
    pub fn elapsed(&self) -> Duration { self.start.elapsed() }

    #[inline]
    pub fn set_bm_stab_factor(&mut self, factor: f64) {
        self.bm_stab = factor;
        self.recompute_soft();
    }

    #[inline]
    pub fn set_bm_inst_factor(&mut self, factor: f64) {
        self.bm_inst = factor;
        self.recompute_soft();
    }

    #[inline]
    pub fn set_score_factor(&mut self, factor: f64) {
        self.score_factor = factor;
        self.recompute_soft();
    }

    #[inline]
    pub fn set_fail_low_factor(&mut self, factor: f64) {
        self.fail_low = factor;
        self.recompute_soft();
    }

    #[inline]
    fn recompute_soft(&mut self) {
        let factor = self.bm_stab * self.bm_inst * self.score_factor * self.fail_low;
        let scaled = self.base_soft.as_millis() as f64 * factor;
        self.soft = Duration::from_millis(scaled as u64).min(self.hard);
    }
}

struct Clock {
    time: u64,
    inc: u64,
    movestogo: u64,
    game_ply: u64,
}

impl Clock {
    fn for_stm(limits: &Limits, stm: Color, game_ply: u64) -> Self {
        let (time, inc) = match stm {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };

        Self { time, inc, movestogo: limits.movestogo, game_ply }
    }

    #[inline]
    fn is_unclocked(&self) -> bool { self.time == 0 && self.inc == 0 }

    fn moves_to_go(&self, phase: i32, params: &SearchParams) -> f64 {
        if self.movestogo > 0 {
            return (self.movestogo as f64 - 0.5).max(1.0);
        }

        let open = params.mtg_opening as f64;
        let end = params.mtg_endgame as f64;
        let p = (phase as f64).clamp(0.0, TOTAL_PHASE as f64);
        (end + (open - end) * p / TOTAL_PHASE as f64).max(1.0)
    }

    fn hard_ms(&self, mtg: f64, params: &SearchParams) -> u64 {
        let hard = if self.movestogo > 0 {
            let mult = params.tm_hard_mult as f64 / 100.0;
            let clock_cap = params.tm_hard_clock_cap as f64 / 100.0;
            (self.time as f64 / mtg * mult).min(self.time as f64 * clock_cap) as u64
        } else {
            let cap = params.tm_sd_cap as f64 / 100.0;
            let frac = params.tm_sd_base as f64 / 100.0 + params.tm_sd_ramp as f64 / 1000.0 * self.game_ply as f64;
            let ceiling = (self.time as f64 * cap) as u64;
            let base = (self.time as f64 * frac) as u64;
            base.min(ceiling)
        };

        hard.saturating_add(self.inc).min(self.time)
    }

    fn soft_ms(&self, mtg: f64, params: &SearchParams) -> u64 {
        let base = (self.time as f64 / mtg) as u64;
        let inc_contrib = (self.inc as f64 * (params.tm_soft_inc as f64 / 100.0)) as u64;
        base + inc_contrib
    }
}

fn compute_budget(
    limits: &Limits,
    stm: Color,
    overhead: u64,
    phase: i32,
    game_ply: u64,
    params: &SearchParams,
) -> (Duration, Duration) {
    if limits.infinite {
        return (Duration::MAX, Duration::MAX);
    }

    if limits.movetime > 0 {
        let limit = with_overhead(limits.movetime, overhead);
        return (limit, limit);
    }

    let clock = Clock::for_stm(limits, stm, game_ply);
    if clock.is_unclocked() {
        return (Duration::MAX, Duration::MAX);
    }

    let mtg = clock.moves_to_go(phase, params);
    let soft_ms = clock.soft_ms(mtg, params);
    let hard_ms = clock.hard_ms(mtg, params);

    (with_overhead(soft_ms.min(hard_ms), overhead), with_overhead(hard_ms, overhead))
}

// The 1 ms floor keeps a zero-length window off the search, which would abort on entry.
#[inline]
fn with_overhead(ms: u64, overhead: u64) -> Duration { Duration::from_millis(ms.saturating_sub(overhead).max(1)) }
