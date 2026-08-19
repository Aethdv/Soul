//! Time management and limits tracking.
//!
//! Resolves a soft / hard millisecond budget from the protocol-supplied
//! time control, then watches the wall clock against those bounds during
//! search. The soft limit is what we aim to spend on this move; the
//! hard limit is what we refuse to exceed.
//!
//! Budgets are decided by the first matching rule, in this order:
//!
//! 1. `infinite`   - search until commanded to stop.
//! 2. `movetime`   - the wall the caller named, spent down to the overhead.
//! 3. unclocked    - no time and no increment; treat as infinite.
//! 4. clocked play - phase-blended budget for sudden death, or explicit
//!    remaining moves budget for classical time controls.

use std::time::{Duration, Instant};

use crate::{
    core::defs::{Color, TOTAL_PHASE},
    engine::{search::Limits, search_params::SearchParams},
};

/// Tracks elapsed time against precomputed soft / hard budgets.
///
/// Constructed once at the start of a search; queried each iteration to
/// decide when to stop iterating (`soft`) and when to bail mid-search
/// regardless of progress (`hard`).
///
/// Soft is composable: each dynamic signal owns one multiplicative factor.
/// `soft = base_soft · ∏ factors`, clamped to `hard`. Every setter
/// recomputes from `base_soft`, so factors never compound across
/// iterations; adding a new signal is one field plus one setter.
pub struct TimeManager {
    start: Instant,
    hard: Duration,
    soft: Duration,
    base_soft: Duration,
    bm_stab: f64,
    score_factor: f64,
}

impl TimeManager {
    /// `phase` feeds the moves-to-go interpolation; `overhead` is shaved off every
    /// finite budget to leave room for I/O and GUI lag, never below 1 ms.
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
        Self { start, soft, hard, base_soft: soft, bm_stab: 1.0, score_factor: 1.0 }
    }

    #[inline]
    pub fn is_hard_limit_reached(&self) -> bool { self.start.elapsed() >= self.hard }

    #[inline]
    pub fn soft_limit(&self) -> Duration { self.soft }

    #[inline]
    pub fn hard_limit(&self) -> Duration { self.hard }

    #[inline]
    pub fn elapsed(&self) -> Duration { self.start.elapsed() }

    #[inline]
    pub fn set_bm_stab_factor(&mut self, factor: f64) {
        self.bm_stab = factor;
        self.recompute_soft();
    }

    /// 1.0 is the no-op, and passing it is how a stretch from the previous
    /// iteration gets cleared.
    #[inline]
    pub fn set_score_factor(&mut self, factor: f64) {
        self.score_factor = factor;
        self.recompute_soft();
    }

    #[inline]
    fn recompute_soft(&mut self) {
        let scaled = self.base_soft.as_millis() as f64 * self.bm_stab * self.score_factor;
        self.soft = Duration::from_millis(scaled as u64).min(self.hard);
    }
}

/// The side-to-move's view of the time control: their remaining time,
/// their increment, and the CLI/GUI-supplied `movestogo` (0 if absent).
///
/// All the per-move budget arithmetic lives on this type so the public
/// constructor stays a flat list of decisions rather than a wall of math.
struct Clock {
    time: u64,
    inc: u64,
    movestogo: u64,
    game_ply: u64,
}

impl Clock {
    /// Pick the side-to-move's clock fields out of the protocol message.
    fn for_stm(limits: &Limits, stm: Color, game_ply: u64) -> Self {
        let (time, inc) = match stm {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };

        Self { time, inc, movestogo: limits.movestogo, game_ply }
    }

    /// True when the CLI/GUI sent neither time nor increment for this side:
    /// e.g. `go depth N` or analysis mode without a clock attached.
    #[inline]
    fn is_unclocked(&self) -> bool { self.time == 0 && self.inc == 0 }

    /// Estimated remaining moves in the game.
    ///
    /// If the CLI/GUI supplied `movestogo`, we trust it (front-loading slightly
    /// by subtracting 0.5). Otherwise, we interpolate between an opening estimate
    /// and an endgame estimate using the current game phase: more moves expected
    /// early, fewer late.
    fn moves_to_go(&self, phase: i32, params: &SearchParams) -> f64 {
        if self.movestogo > 0 {
            return (self.movestogo as f64 - 0.5).max(1.0);
        }

        let open = params.mtg_opening as f64;
        let end = params.mtg_endgame as f64;
        let p = (phase as f64).clamp(0.0, TOTAL_PHASE as f64);
        (end + (open - end) * p / TOTAL_PHASE as f64).max(1.0)
    }

    /// What a single move may spend before the search is forced to bail.
    ///
    /// Classical bursts to `tm_hard_mult`× the per-move share (`time / mtg`),
    /// capped at `tm_hard_clock_cap`% of the clock. Sudden death instead spends a
    /// fraction that ramps from `tm_sd_base`% with game ply (`tm_sd_ramp` per mille
    /// per ply) toward the `tm_sd_cap`% ceiling, so a longer game lets one move take
    /// a bigger slice. Either way the increment is added back, since it's regained,
    /// capped at the full remaining time.
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

        (hard.saturating_add(self.inc)).min(self.time)
    }

    /// The per-move share plus `tm_soft_inc`% of the increment.
    fn soft_ms(&self, mtg: f64, params: &SearchParams) -> u64 {
        let base = (self.time as f64 / mtg) as u64;
        let inc_contrib = (self.inc as f64 * (params.tm_soft_inc as f64 / 100.0)) as u64;
        base + inc_contrib
    }
}

/// Resolve the `(soft, hard)` budget pair for the side to move.
///
/// Walks the precedence ladder documented at the module level. The clocked path is
/// the only one that consults `phase` and `params`; everything above it short-circuits
/// before they're touched.
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

/// Subtract communication overhead from a millisecond budget, clamped to
/// at least 1 ms so we never produce a zero-length window the search would
/// abort on entry.
#[inline]
fn with_overhead(ms: u64, overhead: u64) -> Duration { Duration::from_millis(ms.saturating_sub(overhead).max(1)) }
