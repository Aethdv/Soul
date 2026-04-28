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
//! 1. `infinite`   — search until commanded to stop.
//! 2. `movetime`   — fixed wall clock per move; soft equals hard.
//! 3. unclocked    — no time and no increment; treat as infinite.
//! 4. clocked play — phase-blended budget for sudden death, or explicit
//!    remaining moves budget for classical time controls.

use std::time::{Duration, Instant};

use crate::{
    core::defs::{Color, TOTAL_PHASE},
    engine::{search::Limits, search_params::SearchParams},
};

/// Absolute hard limit as a fraction of remaining time for sudden death.
pub const TIME_HARD_CAP: f64 = 0.5;

/// Tracks elapsed time against precomputed soft / hard budgets.
///
/// Constructed once at the start of a search; queried from the search loop
/// to decide when to stop iterating (`soft`) and when to bail mid-search
/// regardless of progress (`hard`).
pub struct TimeManager {
    start: Instant,
    hard: Duration,
    soft: Duration,
    /// Immutable baseline from the initial budget computation.
    /// Dynamic scalers are always applied relative to this,
    /// never to an already-scaled `soft` — so factors don't compound
    /// across iterations.
    base_soft: Duration,
}

impl TimeManager {
    /// Resolve the budget for a single move and start the clock.
    ///
    /// `phase` is the current game phase (0 = endgame, `TOTAL_PHASE` = opening)
    /// and feeds the moves-to-go interpolation. `overhead` is shaved off both
    /// budgets to leave room for I/O and GUI lag — never enough to drop a
    /// budget below 1 ms.
    pub fn new(limits: &Limits, start: Instant, stm: Color, overhead: u64, phase: i32, params: &SearchParams) -> Self {
        let (soft, hard) = compute_budget(limits, stm, overhead, phase, params);
        Self { start, soft, hard, base_soft: soft }
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

    /// Stretch the current soft budget by `percent / 100`, clamped to hard.
    ///
    /// Called when iterative deepening detects instability that warrants more
    /// time on this move — e.g. the root score just dropped sharply. The hard
    /// clamp is the invariant: soft may grow, but never past the clock cap.
    #[inline]
    pub fn extend_soft_limit(&mut self, percent: u32) {
        let current_ms = self.soft.as_millis() as u64;
        let extended = Duration::from_millis(current_ms.saturating_mul(percent as u64) / 100);
        self.soft = extended.min(self.hard);
    }

    /// Rescale the soft budget relative to the original baseline.
    ///
    ///   `soft = min(base_soft · percent / 100, hard)`
    ///
    /// Called once per ID iteration with a factor derived from per-root-move node effort:
    /// concentrated effort shrinks the budget, scattered effort stretches it.
    /// Anchoring on `base_soft` (not current `soft`) keeps the factor
    /// from compounding across iterations.
    #[inline]
    pub fn apply_stability_factor(&mut self, percent: u32) {
        let base_ms = self.base_soft.as_millis() as u64;
        let scaled = Duration::from_millis(base_ms.saturating_mul(percent as u64) / 100);
        self.soft = scaled.min(self.hard);
    }
}

// ──────── Private ────────

/// Resolve the `(soft, hard)` budget pair for the side to move.
///
/// Walks the precedence ladder documented at the module level. The clocked
/// path is the only one that consults `phase` and `params`; everything
/// above it short-circuits before they're touched.
fn compute_budget(limits: &Limits, stm: Color, overhead: u64, phase: i32, params: &SearchParams) -> (Duration, Duration) {
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
    let hard_ms = clock.hard_ms(mtg);
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
    time: u64,
    inc: u64,
    movestogo: u64,
}

impl Clock {
    /// Pick the side-to-move's clock fields out of the protocol message.
    fn for_stm(limits: &Limits, stm: Color) -> Self {
        let (time, inc) = match stm {
            Color::White => (limits.wtime, limits.winc),
            Color::Black => (limits.btime, limits.binc),
        };
        Self { time, inc, movestogo: limits.movestogo }
    }

    /// True when the CLI/GUI sent neither time nor increment for this side —
    /// e.g. `go depth N` or analysis mode without a clock attached.
    #[inline]
    fn is_unclocked(&self) -> bool {
        self.time == 0 && self.inc == 0
    }

    /// Estimated remaining moves in the game.
    ///
    /// If the CLI/GUI supplied `movestogo`, we trust it (front-loading slightly
    /// by subtracting 0.5). Otherwise, we interpolate between an opening estimate
    /// and an endgame estimate using the current game phase: more moves expected
    /// early, fewer late.
    fn moves_to_go(&self, phase: i32, params: &SearchParams) -> f64 {
        if self.movestogo > 0 {
            // Classical time control. Subtracting 0.5 front-loads time usage.
            return (self.movestogo as f64 - 0.5).max(1.0);
        }

        let open = params.mtg_opening as f64;
        let end = params.mtg_endgame as f64;
        let p = (phase as f64).clamp(0.0, TOTAL_PHASE as f64);

        (end + (open - end) * p / TOTAL_PHASE as f64).max(1.0)
    }

    /// Hard cap: the absolute maximum time we can spend on this move.
    fn hard_ms(&self, mtg: f64) -> u64 {
        let hard_base = if self.movestogo > 0 {
            // Classical: Bound by a multiple of the per-move budget,
            // but never exceed 95% of total remaining time to leave a sliver of safety.
            (self.time as f64 / mtg * 5.0).min(self.time as f64 * 0.95)
        } else {
            // Sudden death: hard cap at a fixed fraction (50%) of total remaining time.
            self.time as f64 * TIME_HARD_CAP
        };

        // We can safely add the increment because we'll get it back after this move.
        (hard_base as u64).saturating_add(self.inc).min(self.time)
    }

    /// Soft target: the optimum time we aim to spend on this move.
    fn soft_ms(&self, mtg: f64, hard_ms: u64) -> u64 {
        let base = (self.time as f64 / mtg) as u64;
        let inc_contrib = (self.inc as f64 * 0.8) as u64;

        (base + inc_contrib).min(hard_ms)
    }
}
