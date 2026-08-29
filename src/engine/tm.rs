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
    effort: f64,
    bm_inst: f64,
    score_factor: f64,
    bm_changes: f64,
}

pub struct Iteration {
    pub depth: i32,
    pub best_move_changed: bool,
    pub root_moves: usize,
    pub best_nodes: u64,
    pub total_nodes: u64,
    pub score: i32,
    pub prev_score: i32,
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
        Self { start, soft, hard, base_soft: soft, effort: 1.0, bm_inst: 1.0, score_factor: 1.0, bm_changes: 0.0 }
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
    pub fn scale_base_soft(&mut self, factor: f64) {
        self.base_soft = self.base_soft.mul_f64(factor);
        self.recompute_soft();
    }

    /// prev_depth_ms · 2 is a rough branching-factor proxy; each additional ply
    /// typically costs about twice the previous one.
    #[inline]
    pub fn should_stop(&self, elapsed_ms: u64, prev_depth_ms: u64, params: &SearchParams) -> bool {
        elapsed_ms >= self.soft.as_millis() as u64
            || elapsed_ms + prev_depth_ms * params.tm_iter_scale as u64 / 100 > self.hard.as_millis() as u64
    }

    /// Applies one completed iteration to the soft limit.
    pub fn update(&mut self, iter: &Iteration, params: &SearchParams) {
        // ── Node Effort TM (~20 Elo)
        // Scale the soft budget by the best move's share
        // of total search effort. A large share means the search keeps
        // confirming one move, so shrink the budget. A small share means
        // effort is scattered across candidates, so stretch it.
        //
        //   percent = clamp(floor, base − scale · best_nodes / total_nodes)
        //
        // Gated below effort_depth: early iterations haven't
        // accumulated enough node signal for the ratio to be meaningful.
        if iter.depth >= params.effort_depth && iter.root_moves > 1 {
            let effort_discount = params.effort_scale as u64 * iter.best_nodes / iter.total_nodes.max(1);
            let percent = (params.effort_base as u64)
                .saturating_sub(effort_discount)
                .max(params.effort_floor as u64);
            self.effort = percent as f64 / 100.0;
        }

        // ── Score Swing (~28 Elo)
        // Scale the soft budget by how far the score moved since last iteration.
        // A drop means a refutation surfaced, so double the budget to buy depth
        // and resolve it. A surge means we found something strong, so halve it
        // and bank the time.
        //
        //   factor = 2 ^ (clamp(prev − new, ±scale) / scale)
        //
        // Clamping pins the factor to [0.5, 2.0]; the exponent makes equal-size
        // gains and losses scale the budget by reciprocal amounts. Gated below
        // score_drop_depth: low-depth aspiration churn is noise, not signal.
        self.score_factor = if iter.depth >= params.score_drop_depth {
            let scale = params.score_swing_scale as f64;
            let diff = ((iter.prev_score - iter.score) as f64).clamp(-scale, scale);
            2.0_f64.powf(diff / scale)
        } else {
            1.0
        };

        // ── Best-Move Instability TM
        // Node effort and score swing both read a settled position as settled.
        // Neither sees the top two moves trading places under a steady score,
        // which is the position worth another iteration.
        //
        // Halving each iteration leaves the count reading recent churn rather
        // than everything the search ever reconsidered.
        if iter.best_move_changed {
            self.bm_changes += 1.0;
        }
        self.bm_inst = 1.0 + f64::from(params.bm_inst_scale) / 100.0 * self.bm_changes;
        self.bm_changes *= f64::from(params.bm_inst_decay) / 100.0;

        self.recompute_soft();
    }

    #[inline]
    fn recompute_soft(&mut self) {
        let factor = self.effort * self.bm_inst * self.score_factor;
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
    let future_reserve = if clock.movestogo > 0 { overhead.saturating_mul(clock.movestogo.saturating_add(1)) } else { 0 };
    let usable_time = clock.time.saturating_sub(future_reserve);
    let usable_clock = Clock { time: usable_time, ..clock };
    let soft_ms = usable_clock.soft_ms(mtg, params);
    let hard_ms = usable_clock.hard_ms(mtg, params);

    (with_overhead(soft_ms.min(hard_ms), overhead), with_overhead(hard_ms, overhead))
}

// The 1 ms floor keeps a zero-length window off the search, which would abort on entry.
#[inline]
fn with_overhead(ms: u64, overhead: u64) -> Duration { Duration::from_millis(ms.saturating_sub(overhead).max(1)) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_forced_move_discount_outlives_the_iteration_factors() {
        let params = SearchParams::default();
        let limits = Limits { wtime: 8000, btime: 8000, winc: 80, binc: 80, ..Default::default() };
        let mut tm = TimeManager::new(&limits, Instant::now(), Color::White, 10, TOTAL_PHASE, 0, &params);
        let undiscounted = tm.soft_limit();

        tm.scale_base_soft(f64::from(params.tm_single_root) / 100.0);
        let discounted = tm.soft_limit();
        assert!(discounted < undiscounted, "{discounted:?} is not a discount on {undiscounted:?}");

        let settled = Iteration {
            depth: 12,
            best_move_changed: false,
            root_moves: 4,
            best_nodes: 950,
            total_nodes: 1000,
            score: 0,
            prev_score: 0,
        };
        tm.update(&settled, &params);
        assert!(tm.soft_limit() <= discounted, "the iteration factors restored {:?}", tm.soft_limit());
    }
}
