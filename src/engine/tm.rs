//! Time management and limits tracking.
//!
//! Resolves a soft / hard millisecond budget from the protocol-supplied
//! time control, then watches the wall clock against those bounds during
//! search. The soft limit is what we *aim* to spend on this move; the
//! hard limit is what we *refuse* to exceed.
//!
//! # Decision order
//!
//! Budgets are decided by the first matching rule, in this order:
//!
//! 1. `infinite`     — search until commanded to stop.
//! 2. `movetime`     — fixed wall clock per move; soft equals hard.
//! 3. unclocked      — no time and no increment; treat as infinite.
//! 4. clocked play   — phase-blended budget, with a tightening pass when
//!                     the CLI/GUI sent an explicit `movestogo`.

use std::time::{Duration, Instant};

use crate::{
    core::defs::{Color, TOTAL_PHASE},
    engine::{search::Limits, search_params::SearchParams},
};

/// Lower bound for moves-to-go.
pub const MIN_MTG: u64 = 2;
/// Upper bound for moves-to-go.
pub const MAX_MTG: u64 = 100;
/// Fraction of remaining time used as the soft limit per move.
pub const TIME_SOFT_CAP: f64 = 0.05;
/// Absolute hard limit as a fraction of remaining time.
pub const TIME_HARD_CAP: f64 = 0.5;

/// Tracks elapsed time against precomputed soft / hard budgets.
///
/// Constructed once at the start of a search; queried from the search loop
/// to decide when to stop iterating (`soft`) and when to bail mid-search
/// regardless of progress (`hard`).
pub struct TimeManager {
    start: Instant,
    hard:  Duration,
    soft:  Duration,
}

impl TimeManager {
    /// Resolve the budget for a single move and start the clock.
    ///
    /// `phase` is the current game phase (0 = endgame, `TOTAL_PHASE` = opening)
    /// and feeds the moves-to-go interpolation. `overhead` is shaved off both
    /// budgets to leave room for I/O and GUI lag — never enough to drop a
    /// budget below 1 ms.
    pub fn new(
        limits: &Limits,
        start: Instant,
        stm: Color,
        overhead: u64,
        phase: i32,
        params: &SearchParams,
    ) -> Self {
        let (soft, hard) = compute_budget(limits, stm, overhead, phase, params);
        Self { start, soft, hard }
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

// ──────── Private ────────

/// Resolve the `(soft, hard)` budget pair for the side to move.
///
/// Walks the precedence ladder documented at the module level. The clocked
/// path is the only one that consults `phase` and `params`; everything
/// above it short-circuits before they're touched.
fn compute_budget(
    limits: &Limits,
    stm: Color,
    overhead: u64,
    phase: i32,
    params: &SearchParams,
) -> (Duration, Duration) {
    if limits.infinite {
        return (Duration::MAX, Duration::MAX);
    }

    if limits.movetime > 0 {
        let limit = with_overhead(limits.movetime, overhead);
        return (limit, limit);
    }

    let clock = Clock::for_stm(limits, stm);

    if clock.is_unclocked() {
        return (Duration::MAX, Duration::MAX);
    }

    let mtg = clock.moves_to_go(phase, params);
    let hard_ms = clock.hard_ms();
    let soft_ms = clock.soft_ms(mtg, hard_ms);

    (with_overhead(soft_ms, overhead), with_overhead(hard_ms, overhead))
}

/// Subtract communication overhead from a millisecond budget, clamped to
/// at least 1 ms so we never produce a zero-length window the search would
/// abort on entry.
#[inline]
fn with_overhead(ms: u64, overhead: u64) -> Duration {
    Duration::from_millis(ms.saturating_sub(overhead).max(1))
}

/// The side-to-move's view of the time control: their remaining time,
/// their increment, and the CLI/GUI-supplied `movestogo` (0 if absent).
///
/// All the per-move budget arithmetic lives on this type so the public
/// constructor stays a flat list of decisions rather than a wall of math.
struct Clock {
    time:      u64,
    inc:       u64,
    movestogo: u64,
}

impl Clock {
    /// Pick the side-to-move's clock fields out of the protocol message.
    fn for_stm(limits: &Limits, stm: Color) -> Self {
        let (time, inc) = match stm {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };
        Self {
            time,
            inc,
            movestogo: limits.movestogo,
        }
    }

    /// True when the CLI/GUI sent neither time nor increment for this side —
    /// e.g. `go depth N` or analysis mode without a clock attached.
    #[inline]
    fn is_unclocked(&self) -> bool {
        self.time == 0 && self.inc == 0
    }

    /// Estimated remaining moves in the game.
    ///
    /// If the CLI/GUI supplied `movestogo`, trust it (clamped to a sane range).
    /// Otherwise interpolate between an opening estimate and an endgame
    /// estimate using the current game phase: more moves expected early,
    /// fewer late. Prevents overspending in the opening while tightening
    /// the budget where the time bound bites hardest.
    fn moves_to_go(&self, phase: i32, params: &SearchParams) -> u64 {
        if self.movestogo > 0 {
            return self.movestogo.clamp(MIN_MTG, MAX_MTG);
        }

        let open = params.mtg_opening as i64;
        let end = params.mtg_endgame as i64;
        let p = (phase as i64).clamp(0, TOTAL_PHASE as i64);
        let blend = end + (open - end) * p / TOTAL_PHASE as i64;

        blend.clamp(MIN_MTG as i64, MAX_MTG as i64) as u64
    }

    /// Hard cap: half the clock plus one increment, never more than the
    /// clock itself. The increment is added unconditionally because we'll
    /// receive it as soon as we move — it's effectively part of the budget.
    fn hard_ms(&self) -> u64 {
        ((self.time as f64 * TIME_HARD_CAP) as u64)
            .saturating_add(self.inc)
            .min(self.time)
    }

    /// Soft target: `time / mtg`, capped at `TIME_SOFT_CAP` of the clock,
    /// plus half the increment as a small bonus, never above `hard_ms`.
    /// In a classical (movestogo) control, also runs a tightening pass —
    /// see `classical_adjustment`.
    fn soft_ms(&self, mtg: u64, hard_ms: u64) -> u64 {
        let base = (self.time as f64 / mtg as f64) as u64;
        let cap = (self.time as f64 * TIME_SOFT_CAP) as u64;
        let raw = (base.min(cap) + self.inc / 2).min(hard_ms);

        if self.movestogo > 0 {
            self.classical_adjustment(raw)
        } else {
            raw
        }
    }

    /// Tighten the soft budget when the time control is `movestogo`-based.
    ///
    /// Interpolates between "blitz-style" pacing and "sudden-death" pacing.
    /// With the constants below, having 40+ moves to go scales the budget by 1.0;
    /// having only a couple left scales it down toward a third of that.
    fn classical_adjustment(&self, soft_ms: u64) -> u64 {
        const SUDDEN_DEATH_THRESHOLD: u64 = 20;
        const NORMALIZATION_FACTOR: u64 = 60;
        const MIN_TIME_LIMIT_MS: u64 = 10;

        let mtg_limit = self.movestogo.min(40);
        (soft_ms * (mtg_limit + SUDDEN_DEATH_THRESHOLD) / NORMALIZATION_FACTOR).max(MIN_TIME_LIMIT_MS)
    }
}
