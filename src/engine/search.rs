//! Negamax alpha-beta search
//!
//! Iterative deepening is the outer loop: depth 1, then 2, then 3, each pass
//! seeding the next's move ordering. Every interior node runs negamax alpha-beta
//! with PVS on top, a full window for the first move and zero-width scouts for
//! the rest, and the leaves fall through to quiescence so the horizon never falls
//! in the middle of an exchange.
//!
//! Lazy SMP is a parallel search algorithm that never divides the work. Each
//! thread runs its own iterative-deepening search on the same root position,
//! sharing only the transposition table, a stop flag and the node counters.
//! Their trees drift apart on their own: thread A writes an entry at the edge of
//! what it can reach, thread B probes it mid-search and redirects its tree along a
//! branch A already explored. The TT is the coordination surface.
//!
//! Each thread holds a [`Searcher`] and a [`Worker`]:
//! - [`Searcher`]: the root position and its move list, the time manager, the PV,
//!   and the search-wide counters, all of it read by the iteration and the protocol.
//! - [`Worker`]: the board, the SIMD accumulator, the ply stack, and the caches
//!   maintained beside them through make and unmake. Built once per search and
//!   dropped with it; its history tables are borrowed, since they outlive it.
//!
//! [`NodeType`] monomorphizes negamax three ways, root, PV and non-PV.

use std::{
    cmp::Reverse,
    collections::VecDeque,
    hint::{likely, unlikely},
    io,
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicU64, Ordering},
    },
    time::Instant,
};

#[cfg(not(feature = "nostore"))]
use crate::core::board::xorboard::Undo as XbUndo;
pub use crate::core::defs::Protocol;
use crate::{
    core::{
        board::{Position, attacks::Pins, xorboard::XorBoard},
        defs::{
            Bitboard, Color, INF, MATE_BOUND, MAX_DEPTH, MAX_PLY, PieceType, Square, draw_score, is_loss, is_mate, is_win, mate_in,
            mated_in,
        },
        moves::Move,
    },
    engine::{
        eval::{EvalParams, PawnCache, SharedFeatures, evaluate_generic, evaluate_psqt, extract_phase, lazy_eval_margin},
        history::{self, ContContext, History, HistoryCaps},
        movegen::{gen_legal_moves, is_legal},
        movepicker::MovePicker,
        search_params::*,
        see::see_ge,
        tm::TimeManager,
        tt,
        tt::{TranspositionTable, TtData, TtMove},
        tui,
    },
    tools::perft::perft,
    weave::{Vi16x8, Vu64x4},
};

const NODE_CHECK_INTERVAL: u64 = 2048;
/// Minimum node count before printing `currmove` UCI output.
const CURRMOVE_NODE_THRESHOLD: u64 = 100_000_000;
const PRINT_UPDATE_MS: u128 = 25;
/// Matches MAX_MOVES, so the list never truncates. Only the prefix searched before
/// a cutoff takes the malus, so less would serve.
const MAX_TRACKED_QUIETS: usize = 256;
/// Captures are far fewer per position than quiets. 64 covers all realistic
/// legal capture counts with headroom.
const MAX_TRACKED_CAPTURES: usize = 64;

/// One ply in LMR units. Reductions accumulate in fractions of a ply, so an
/// adjustment like the fail-high malus can move one by half a ply instead of
/// rounding to nothing or to a whole one.
const LMR_SCALE: i32 = 1024;

/// Chess search has three distinct contexts:
/// root (first ply, owns the move list), PV, and non-PV
/// (zero-window scouts that just need a yes/no answer).
///
/// By encoding these as types rather than runtime flags, the compiler
/// monomorphizes negamax into three variants, each with the other two
/// contexts' branches compiled out.
pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
    type Next: NodeType;
}

pub struct RootNode;
pub struct PvNode;
pub struct NonPvNode;

impl NodeType for RootNode {
    const PV: bool = true;
    const ROOT: bool = true;
    type Next = PvNode;
}

impl NodeType for PvNode {
    const PV: bool = true;
    const ROOT: bool = false;
    // PvNode::Next = PvNode because re-searches after fail-highs should stay on the PV.
    // The transition to NonPvNode happens inside pvs(), which uses NonPvNode for scouts
    // explicitly, regardless of the enclosing node type.
    type Next = PvNode;
}

impl NodeType for NonPvNode {
    const PV: bool = false;
    const ROOT: bool = false;
    type Next = NonPvNode;
}

pub struct Searcher<'cfg> {
    pub cfg: &'cfg SearchConfig,
    pub tm: TimeManager,
    pub root_moves: Vec<RootMove>,
    pub zobrist_trail: Vec<u64>,
    pub root_pos: Position,
    pub prev_pv: Line,
    pub prev_score: i32,
    pub nodes: u64,
    pub sel_depth: i32,
    pub iter_depth: i32,
    pub last_print: u128,
    pub pv_history: VecDeque<PvSnapshot>,
    pub tt: Arc<TranspositionTable>,
}

#[repr(align(32))]
pub struct Worker<'h> {
    pub pos: Position,
    pub accumulator: Vi16x8,
    pub stack: Box<[Stack; MAX_PLY + 2]>,
    /// Per-piece attack rows, carried through make and unmake beside the board.
    #[cfg(not(feature = "nostore"))]
    pub xorboard: XorBoard,
    /// One undo record per ply, boxed for the same reason the ply stack is.
    #[cfg(not(feature = "nostore"))]
    pub xb_undo: Box<[XbUndo; MAX_PLY + 2]>,
    pub history: &'h mut History,
    /// Per-search pawn-structure cache, keyed on the incremental `pawn_key`.
    pub pawn_cache: PawnCache,
    /// Search-constant eval weights, built once instead of per leaf.
    pub eval_params: EvalParams<i32>,
    /// Set while a null-move verification search is on the stack. Without it the
    /// verification would try its own null move and verify that in turn, each pass
    /// re-searching the same position a little shallower.
    pub is_nmp_verif: bool,
}

#[derive(Clone, Default, Debug)]
pub struct Limits {
    pub wtime: u64,
    pub btime: u64,
    pub winc: u64,
    pub binc: u64,
    pub movestogo: u64,
    pub movetime: u64,
    pub depth: i32,
    pub nodes: u64,
    pub softnodes: u64,
    pub infinite: bool,
    pub silent: bool,
    pub protocol: Protocol,
    pub mate: Option<i32>,
    pub perft: Option<u8>,
    pub searchmoves: Vec<Move>,
}

/// Display configuration for search output.
#[derive(Clone, Copy, Default)]
pub struct SearchDisplay {
    pub show_wdl: bool,
    pub go_pretty: bool,
    pub pretty_print: bool,
    pub show_currmove: bool,
    pub use_ansi: bool,
}

#[derive(Clone)]
pub struct SearchConfig {
    pub limits: Limits,
    pub start_time: Instant,
    pub stop: Arc<AtomicBool>,
    pub display: SearchDisplay,
    pub search_params: SearchParams,
    pub overhead: u64,
    pub threads: usize,
    pub thread_id: usize,
    /// Per-thread node counters, one slot per thread.
    /// Each thread writes only its own slot, inside `check_signals`
    /// (every 2048 nodes), with a Relaxed store.
    pub node_slots: Arc<[AtomicU64]>,
    pub mvvlva_v: [i32; 8], // victim values, indexed by PieceType
    pub mvvlva_a: [i32; 8], // attacker penalties, indexed by PieceType
    /// `ln(i) · LMR_SCALE` lookup, indexed by depth or move count.
    /// Reduction composes as `base + table[d] · table[m] / div`, all in LMR_SCALE units.
    pub lmr_table: Box<[i16; MAX_PLY + 1]>,
}

/// A legal move at ply 0 paired with its best known score.
/// After each iteration these are sorted by score so the strongest move
/// is searched first next time, the single most important factor for
/// alpha-beta efficiency.
pub struct RootMove {
    pub mv: Move,
    pub score: i32,
    pub pv: Box<Line>,
    /// Cumulative subtree nodes across all ID iterations. The best move's
    /// share of total nodes feeds the stability factor in time management.
    pub nodes: u64,
}

/// The PV is our predicted best play for both sides.
/// When a new best move is found at any ply, we compose the line:
/// this move first, then the child's continuation, bubbling the full
/// sequence from leaves to root, one ply at a time.
#[derive(Clone, Copy)]
pub struct Line {
    pub moves: [Move; MAX_PLY],
    pub len: usize,
}

/// One completed iterative-deepening pass, retained for the pretty TUI's
/// history and eval sparkline.
#[derive(Clone, Copy)]
pub struct PvSnapshot {
    pub depth: i32,
    pub time_ms: u128,
    pub score: i32,
    pub line: Line,
}

/// Per-ply scratch data.
#[derive(Clone, Copy)]
pub struct Stack {
    pub pv: Line,
    pub quiet_moves: [Move; MAX_TRACKED_QUIETS],
    pub quiet_count: usize,
    pub capture_moves: [Move; MAX_TRACKED_CAPTURES],
    pub capture_count: usize,
    pub killers: [Move; 2],
    pub moved_pt: PieceType,
    pub moved_to: Square,
    pub static_eval: i32,
    pub is_null: bool,
    /// Beta cutoffs among this node's children.
    pub cutoff_count: i32,
    /// The move a singular verification skips. Null in a normal search;
    /// set to the TT move while we test whether it stands alone.
    pub excluded: Move,
}

/// Search was cut short: time, node limit, or external stop signal.
#[derive(Debug)]
pub struct SearchAborted;

impl SearchDisplay {
    pub const SILENT: Self = Self { show_wdl: false, go_pretty: false, pretty_print: false, show_currmove: false, use_ansi: false };
    pub const DEFAULT: Self = Self { show_currmove: true, use_ansi: true, ..Self::SILENT };
}

impl SearchConfig {
    pub fn new(limits: Limits, start_time: Instant, stop: Arc<AtomicBool>, overhead: u64, search_params: SearchParams) -> Self {
        Self::new_full(limits, start_time, stop, overhead, SearchDisplay::DEFAULT, search_params)
    }

    pub fn new_full(
        limits: Limits,
        start_time: Instant,
        stop: Arc<AtomicBool>,
        overhead: u64,
        display: SearchDisplay,
        search_params: SearchParams,
    ) -> Self {
        let (mvvlva_v, mvvlva_a) = Self::build_mvvlva(&search_params);
        let lmr_table = Self::build_lmr_table();

        Self {
            limits,
            start_time,
            stop,
            overhead,
            display,
            search_params,
            threads: 1,
            thread_id: 0,
            node_slots: Self::node_slots(1),
            mvvlva_v,
            mvvlva_a,
            lmr_table,
        }
    }

    pub fn node_slots(threads: usize) -> Arc<[AtomicU64]> { (0..threads).map(|_| AtomicU64::new(0)).collect() }

    /// Composed LMR reduction in `LMR_SCALE` units.
    #[inline(always)]
    pub fn lmr(&self, depth: i32, move_count: usize) -> i32 {
        let sp = &self.search_params;
        let base = sp.lmr_base * LMR_SCALE / 100;
        let div = sp.lmr_divisor * LMR_SCALE / 100;
        let log_depth = i32::from(self.lmr_table[depth as usize]);
        let log_moves = i32::from(self.lmr_table[move_count]);
        base + log_depth * log_moves / div
    }

    /// MVV-LVA lookup table from tunable parameters.
    ///
    /// Most Valuable Victim - Least Valuable Attacker:
    /// The simplest capture ordering that works.
    /// Prefer taking queens with pawns over taking pawns with queens.
    fn build_mvvlva(sp: &SearchParams) -> ([i32; 8], [i32; 8]) {
        let mut v = [0; 8];
        let mut a = [0; 8];

        macro_rules! map {
            ($pt:ident, $v:expr, $a:expr) => {
                v[PieceType::$pt] = $v;
                a[PieceType::$pt] = $a;
            };
        }
        map!(Pawn, sp.mvvlva_v_pawn, sp.mvvlva_a_pawn);
        map!(Knight, sp.mvvlva_v_knight, sp.mvvlva_a_knight);
        map!(Bishop, sp.mvvlva_v_bishop, sp.mvvlva_a_bishop);
        map!(Rook, sp.mvvlva_v_rook, sp.mvvlva_a_rook);
        map!(Queen, sp.mvvlva_v_queen, sp.mvvlva_a_queen);
        map!(King, sp.mvvlva_v_king, sp.mvvlva_a_king);
        (v, a)
    }

    /// `ln(i) · LMR_SCALE` lookup for LMR reduction composition.
    ///
    ///   `R(d, m) = base + ln(d) · ln(m) / divisor`
    ///
    /// Factored into a 1D log table reused on both axes; deeper searches
    /// tolerate larger reductions, and later moves deserve them. Looking
    /// up two values and multiplying beats a 2D table that spills L1.
    fn build_lmr_table() -> Box<[i16; MAX_PLY + 1]> {
        let scale = LMR_SCALE as f64;
        let mut lut = Box::new([0i16; MAX_PLY + 1]);
        for i in 1..=MAX_PLY {
            lut[i] = ((i as f64).ln() * scale).round() as i16;
        }
        lut
    }
}

impl Line {
    pub const fn new() -> Self { Self { moves: [Move::null(); MAX_PLY], len: 0 } }

    /// Prepend `mv` to `tail`, forming a complete PV line.
    pub fn compose(&mut self, mv: Move, tail: &Line) {
        let n = tail.len.min(MAX_PLY - 1);
        self.moves[0] = mv;
        self.moves[1..=n].copy_from_slice(&tail.moves[..n]);
        self.len = 1 + n;
    }

    /// Retrieve a move from the PV line if the index is within bounds.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<Move> { if idx < self.len { Some(self.moves[idx]) } else { None } }
}

impl Default for Line {
    fn default() -> Self { Self::new() }
}

impl RootMove {
    #[inline]
    pub fn new(mv: Move) -> Self { Self { mv, score: -INF, pv: Box::new(Line::new()), nodes: 0 } }
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            pv: Line::new(),
            quiet_moves: [Move::null(); MAX_TRACKED_QUIETS],
            quiet_count: 0,
            capture_moves: [Move::null(); MAX_TRACKED_CAPTURES],
            capture_count: 0,
            killers: [Move::null(); 2],
            moved_pt: PieceType::None,
            moved_to: Square(0),
            static_eval: tt::SCORE_NONE,
            is_null: false,
            cutoff_count: 0,
            excluded: Move::null(),
        }
    }
}

impl<'cfg> Searcher<'cfg> {
    /// Search depth 1, then 2, then 3, ...
    ///
    /// The repeated shallow work pays for itself: each iteration's move ordering
    /// feeds the next, and a deep search with good ordering costs far less than a
    /// blind one. It also makes the search stoppable at any moment, since the last
    /// completed iteration always has a move ready.
    #[inline]
    pub fn iterative_deepening(&mut self, history: &mut History) {
        let sp = &self.cfg.search_params;
        history.caps = HistoryCaps::from(sp);
        self.nodes = 0;

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            print!("\x1b[2J\x1b[H");
            let _ = io::stdout().flush();
        }

        if let Some(perft_depth) = self.cfg.limits.perft {
            let mut board = self.root_pos;
            let mut acc = board.get_initial_accumulator();
            println!("Nodes searched: {}", perft(&mut board, perft_depth, &mut acc));
            return;
        }

        if self.root_moves.is_empty() {
            if !self.cfg.limits.silent {
                println!("bestmove 0000");
            }
            return;
        }

        if !self.cfg.limits.searchmoves.is_empty() {
            self.root_moves.retain(|rm| self.cfg.limits.searchmoves.contains(&rm.mv));

            if self.root_moves.is_empty() {
                eprintln!("info string error: no legal moves match searchmoves");
                if !self.cfg.limits.silent {
                    println!("bestmove 0000");
                }
                return;
            }
        }

        let depth_limit = match (self.cfg.limits.mate, self.cfg.limits.depth) {
            (Some(mate_d), _) => (mate_d * 2).min(MAX_DEPTH),
            (_, d) if d > 0 => d,
            _ => MAX_DEPTH,
        };

        // The root position is fixed for the whole search, so its PSQT
        // accumulator is invariant; compute it once and restore from it on
        // every aspiration re-search rather than rescanning the board.
        let root_acc = self.root_pos.get_initial_accumulator();

        // The worker is built once here, not inside the loop: a fresh one per
        // iteration would memset 287 KB of ply stack every pass, for nothing.
        let mut worker = Worker {
            pos: self.root_pos,
            accumulator: root_acc,
            stack: boxed_array(Stack::default()),
            #[cfg(not(feature = "nostore"))]
            xorboard: XorBoard::new(&self.root_pos),
            #[cfg(not(feature = "nostore"))]
            xb_undo: boxed_array(XbUndo::default()),
            history,
            pawn_cache: PawnCache::new(),
            eval_params: EvalParams::<i32>::from_const(),
            is_nmp_verif: false,
        };

        let mut last_iter_elapsed = 0;
        let mut bm_changes = 0.0;

        // ── Singular Bailout
        // Only one legal move. Slash the budget to 5% so we exit the depth
        // loop almost instantly, banking the saved time.
        if self.root_moves.len() == 1 {
            self.tm.set_bm_stab_factor(sp.tm_single_root as f64 / 100.0);
        }

        // ── Lazy SMP
        // Helpers search the full depth ladder alongside main, and the only
        // asymmetry is soft time management: main alone decides when to stop
        // starting iterations. Helpers still stop on the global flag and the
        // shared hard cap through check_signals, so they feed the TT for as
        // long as main is still running.
        for depth in 1..=depth_limit {
            self.iter_depth = depth;
            let elapsed = self.tm.elapsed().as_millis() as u64;
            let prev_depth_time = elapsed.saturating_sub(last_iter_elapsed);
            last_iter_elapsed = elapsed;

            // A fixed movetime has no later move to bank unspent budget for,
            // so check_signals' hard wall is its only stop.
            let clocked = self.cfg.limits.movetime == 0;

            // Between iterations:
            // Bail if soft limits say we probably
            // can't finish the next depth in time.
            //
            // prev_depth_time · 2 is a rough branching-factor proxy;
            // each additional ply typically costs about twice the previous one,
            // so if we can't afford that estimate we stop before starting it.
            //
            // Helpers skip time management as their job is to fill the TT,
            // not decide when to stop. Main calls the shots.
            if self.cfg.thread_id == 0
                && depth > 1
                && ((clocked
                    && (elapsed >= self.tm.soft_limit().as_millis() as u64
                        || elapsed + (prev_depth_time * sp.tm_iter_scale as u64 / 100) > self.tm.hard_limit().as_millis() as u64))
                    || (self.cfg.limits.softnodes > 0 && self.nodes >= self.cfg.limits.softnodes))
            {
                break;
            }

            // ── Proven Mate Bailout
            if self.cfg.thread_id == 0
                && self.tm.is_finite_budget()
                && (self.prev_score >= mate_in(3) || self.prev_score == mated_in(2))
            {
                break;
            }

            if self.check_signals() {
                break;
            }

            // A root move composes its line only when it becomes the new best, so
            // one that never does keeps whatever it found at an earlier depth.
            // Clearing here means a line still standing afterwards was built by
            // this iteration.
            for rm in &mut self.root_moves {
                rm.pv.len = 0;
            }

            // ── Aspiration Windows (~42 Elo)
            // A score rarely lurches between iterations, so we bracket the last
            // one in a narrow window instead of searching (-INF, INF). Tighter
            // bounds prune far more, and most iterations land inside for free.
            // When the score escapes, the bound that broke says which way we
            // were wrong: a fail-low drops alpha, a fail-high lifts beta. Each
            // retry widens the window so a score that truly moved doesn't thrash.
            let mut delta = sp.asp_initial;
            let mut alpha = if depth >= sp.asp_depth { (self.prev_score - delta).max(-INF) } else { -INF };
            let mut beta = if depth >= sp.asp_depth { (self.prev_score + delta).min(INF) } else { INF };
            let mut aborted = false;

            loop {
                worker.pos = self.root_pos;
                worker.accumulator = root_acc;
                #[cfg(not(feature = "nostore"))]
                {
                    worker.xorboard = XorBoard::new(&worker.pos);
                }
                // The node's own best score, not root_moves[0]: the list is still
                // in last iteration's order, so a fail-high on any other move would
                // read as a score inside the window and end the iteration on a bound.
                let Ok(score) = worker.negamax::<RootNode>(self, depth, alpha, beta, 0) else {
                    aborted = true;
                    break;
                };

                if score <= alpha {
                    self.print_bound(depth, score, tui::ScoreBound::Upper);
                    alpha = (score - delta).max(-INF);
                } else if score >= beta {
                    self.print_bound(depth, score, tui::ScoreBound::Lower);
                    // The move that broke the window goes first in the retry, where
                    // being first earns the full window: one search settles its score
                    // and its line instead of a scout plus a re-search.
                    //
                    // The cutoff broke the move loop, so moves before the culprit
                    // scored under beta this iteration and moves after it still hold
                    // the previous depth's score. The culprit is the first that can carry
                    // this score, and rotating keeps the rest in the order the last
                    // completed iteration left them, where a sort would rank this depth's
                    // scores against the previous one's.
                    if let Some(i) = self.root_moves.iter().position(|rm| rm.score == score) {
                        self.root_moves[..=i].rotate_right(1);
                    }
                    beta = (score + delta).min(INF);
                } else {
                    break;
                }
                delta += delta / sp.asp_widen_div;
            }

            if aborted {
                // Don't sort: this iteration scored only the moves it reached, so ranking
                // them now would rank a half-finished depth against a complete one. The
                // order the last completed iteration left still stands.
                break;
            }

            // Best move floats to the front, feeding the next iteration's ordering.
            self.root_moves.sort_by_key(|m| Reverse(m.score));

            let new_score = self.root_moves[0].score;

            // ── Best-Move Stability TM (~20 Elo)
            // Scale the soft budget by the best move's share
            // of total search effort. A large share means the search keeps
            // confirming one move, so shrink the budget. A small share means
            // effort is scattered across candidates, so stretch it.
            //
            //   percent = clamp(floor, base − scale · best_nodes / total_nodes)
            //
            // Gated below bm_stab_depth: early iterations haven't
            // accumulated enough node signal for the ratio to be meaningful.
            if depth >= sp.bm_stab_depth {
                let best_nodes = self.root_moves[0].nodes;
                let total_nodes = self.nodes.max(1);
                let effort_discount = sp.bm_stab_scale as u64 * best_nodes / total_nodes;
                let percent = (sp.bm_stab_base as u64).saturating_sub(effort_discount).max(sp.bm_stab_floor as u64);
                self.tm.set_bm_stab_factor(percent as f64 / 100.0);
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
            let score_factor = if depth >= sp.score_drop_depth {
                let scale = sp.score_swing_scale as f64;
                let diff = ((self.prev_score - new_score) as f64).clamp(-scale, scale);
                2.0_f64.powf(diff / scale)
            } else {
                1.0
            };

            self.tm.set_score_factor(score_factor);

            // ── Best-Move Instability TM
            // Node effort and score swing both read a settled position as settled.
            // Neither sees the top two moves trading places under a steady score,
            // which is the position worth another iteration.
            //
            // Halving each iteration leaves the count reading recent churn rather
            // than everything the search ever reconsidered.
            if self.prev_pv.get(0).is_some_and(|prev| prev != self.root_moves[0].mv) {
                bm_changes += 1.0;
            }

            let instability = 1.0 + f64::from(sp.bm_inst_scale) / 100.0 * bm_changes / self.cfg.threads as f64;
            self.tm.set_bm_inst_factor(instability);
            bm_changes *= 0.5;

            self.prev_pv = *self.root_moves[0].pv;
            self.prev_score = self.root_moves[0].score;
            self.print_info(depth, self.prev_score, &self.prev_pv);
            let elapsed = self.tm.elapsed().as_millis().max(1);
            self.pv_history
                .push_back(PvSnapshot { depth, time_ms: elapsed, score: self.prev_score, line: self.prev_pv });

            // Bounded history: the TUI only needs the most recent points for the sparkline.
            if self.pv_history.len() > 30 {
                self.pv_history.pop_front();
            }
        }

        if !self.cfg.limits.silent {
            let best = self.prev_pv.get(0).unwrap_or(self.root_moves[0].mv);

            match self.cfg.limits.protocol {
                Protocol::Uci => println!("bestmove {}", best.to_uci(self.root_pos.is_frc)),
                Protocol::XBoard => println!("move {}", best.to_uci(self.root_pos.is_frc)),
            }
            let _ = io::stdout().flush();
        }
    }

    #[inline]
    pub fn new(cfg: &'cfg SearchConfig, pos: &Position, history: &[u64], tt: Arc<TranspositionTable>) -> Self {
        let tm = Self::build_tm(cfg, pos, history);
        let root_moves = gen_legal_moves(pos).iter().map(|&mv| RootMove::new(mv)).collect();
        let mut trail = Vec::with_capacity(1024);
        Self::fill_trail(&mut trail, pos, history);
        Self {
            cfg,
            tm,
            root_moves,
            zobrist_trail: trail,
            root_pos: *pos,
            prev_pv: Line::new(),
            pv_history: VecDeque::new(),
            prev_score: -INF,
            nodes: 0,
            sel_depth: 0,
            iter_depth: 0,
            last_print: 0,
            tt,
        }
    }

    /// Per-move reset. History lives at the caller; clear it between games via `History::clear`.
    #[inline]
    pub fn reset(&mut self, cfg: &'cfg SearchConfig, pos: &Position, history: &[u64]) {
        Self::fill_trail(&mut self.zobrist_trail, pos, history);
        self.cfg = cfg;
        self.tm = Self::build_tm(cfg, pos, history);
        self.root_pos = *pos;
        self.root_moves = gen_legal_moves(pos).iter().map(|&mv| RootMove::new(mv)).collect();
        self.nodes = 0;
        self.sel_depth = 0;
        self.iter_depth = 0;
        self.last_print = 0;
        self.pv_history.clear();
        self.prev_score = -INF;
        self.prev_pv = Line::new();
    }

    #[inline]
    pub fn best_move(&self) -> Option<Move> { self.root_moves.first().map(|rm| rm.mv) }

    #[inline]
    pub fn best_score(&self) -> Option<i32> { self.root_moves.first().map(|rm| rm.score) }

    /// Rebuild the repetition trail from game history, trimmed to the 50-move
    /// horizon; positions older than the last capture or pawn push can never
    /// repeat. The root hash is always appended last so root repetitions
    /// surface at ply 2, 4, and so on.
    fn fill_trail(trail: &mut Vec<u64>, pos: &Position, history: &[u64]) {
        trail.clear();
        let keep = history.len().min(pos.halfmove_clock as usize);
        if keep > 0 {
            trail.extend_from_slice(&history[history.len() - keep..]);
        }
        trail.push(pos.hash);
    }

    /// Time manager for this root position, derived from the live config
    /// and the accumulator's phase lane.
    fn build_tm(cfg: &SearchConfig, pos: &Position, history: &[u64]) -> TimeManager {
        let phase = i32::from(pos.get_initial_accumulator().to_array()[2]);
        TimeManager::new(&cfg.limits, cfg.start_time, pos.stm, cfg.overhead, phase, history.len() as u64, &cfg.search_params)
    }

    /// Periodic signal check: stop flag, hard time limit, node limit.
    /// Also drives realtime TUI updates, piggybacking on the same interval.
    #[inline]
    fn check_signals(&mut self) -> bool {
        // Publish the local node count into this thread's slot so the
        // display and node-limit paths see the aggregate.
        self.cfg.node_slots[self.cfg.thread_id].store(self.nodes, Ordering::Relaxed);

        if self.cfg.stop.load(Ordering::Relaxed)
            || self.tm.is_hard_limit_reached()
            || (self.cfg.limits.nodes > 0 && self.node_count() >= self.cfg.limits.nodes)
        {
            self.cfg.stop.store(true, Ordering::Relaxed);
            // The flag is stored either way, so the pool stops on time; this thread
            // carries its first iteration to the end, so bestmove names a move the
            // search looked at rather than movegen's first.
            return self.iter_depth > 1;
        }

        if self.cfg.display.go_pretty {
            let now = self.tm.elapsed().as_millis();
            if now - self.last_print > PRINT_UPDATE_MS {
                self.last_print = now;
                self.print_realtime();
            }
        }
        false
    }

    /// Sums per-thread node counters. Each thread publishes its local count
    /// into its own slot every 2048 nodes, so the aggregate may trail by
    /// up to `NODE_CHECK_INTERVAL · threads`, invisible at display intervals.
    #[inline(always)]
    fn node_count(&self) -> u64 { self.cfg.node_slots.iter().map(|s| s.load(Ordering::Relaxed)).sum() }

    /// Assemble the display snapshot shared by depth-complete and realtime reporting.
    #[cold]
    fn search_info_data<'a>(&'a self, depth: i32, score: i32, pv: &'a Line, history: &'a [PvSnapshot]) -> tui::SearchInfoData<'a> {
        let ms = self.tm.elapsed().as_millis().max(1);
        let total = self.node_count();
        let nps = (u128::from(total) * 1000) / ms;
        tui::SearchInfoData {
            bound: tui::ScoreBound::Exact,
            depth,
            score,
            pv,
            sel_depth: self.sel_depth,
            nodes: total,
            nps: u64::try_from(nps).unwrap_or(u64::MAX),
            time_ms: ms,
            hashfull: self.tt.hashfull(),
            history,
            board: &self.root_pos,
            display: &self.cfg.display,
        }
    }

    #[cold]
    fn print_line(&self, depth: i32, score: i32, pv: &Line, bound: tui::ScoreBound) {
        let history: Vec<_> = self.pv_history.iter().copied().collect();
        let mut data = self.search_info_data(depth, score, pv, &history);
        data.bound = bound;

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            tui::print_pretty_search_info(&data);
        } else {
            tui::print_search_info(self.cfg.limits.protocol, &data);
        }
    }

    #[cold]
    fn print_info(&self, depth: i32, score: i32, pv: &Line) {
        if self.cfg.limits.silent {
            return;
        }
        self.print_line(depth, score, pv, tui::ScoreBound::Exact);
    }

    /// Reports a score that left its aspiration window, before the re-search
    /// settles it.
    ///
    /// Skipped for mate scores, where a bound reads as a mate claim the search
    /// has not made.
    #[cold]
    fn print_bound(&self, depth: i32, score: i32, bound: tui::ScoreBound) {
        // A bound is a claim about the position, not about the move that broke the
        // window, so it reports the line the engine currently believes. There is
        // none to report until an iteration has completed one.
        if self.cfg.limits.silent || is_mate(score) || self.prev_pv.len == 0 {
            return;
        }
        self.print_line(depth, score, &self.prev_pv, bound);
    }

    /// UCI `currmove`: tells the GUI which root move is being searched.
    #[cold]
    #[inline]
    fn print_currmove(&self, depth: i32, mv: Move, move_number: usize) {
        if self.cfg.limits.silent
            || self.cfg.display.go_pretty
            || self.cfg.limits.protocol != Protocol::Uci
            || !self.cfg.display.show_currmove
            || self.nodes < CURRMOVE_NODE_THRESHOLD
        {
            return;
        }
        println!("info depth {depth} currmove {} currmovenumber {move_number}", mv.to_uci(self.root_pos.is_frc));
    }

    /// Redraws the panel mid-iteration: counters live, score and line from the
    /// last completed depth. `root_moves` is still in the previous iteration's
    /// order while this one overwrites its scores in place, so its head can pair
    /// a fresh score with a line from an older depth.
    #[cold]
    fn print_realtime(&self) {
        if self.cfg.limits.silent || self.prev_pv.len == 0 {
            return;
        }

        let history_vec: Vec<_> = self.pv_history.iter().copied().collect();
        let data = self.search_info_data(self.iter_depth, self.prev_score, &self.prev_pv, &history_vec);
        tui::print_pretty_search_info(&data);
    }
}

struct MoveResult {
    move_count: usize,
    best_eval: i32,
    alpha: i32,
    best_move: Move,
}

/// A heap `[T; N]` built through a `Vec`, so the array never lands on the stack.
fn boxed_array<T: Clone, const N: usize>(value: T) -> Box<[T; N]> {
    vec![value; N].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!())
}

/// Piece-to-square contexts 1, 2, and 4 plies back, for cont-hist lookup.
/// Empty when the ply predates the search root.
fn cont_contexts(stack: &[Stack], ply: usize) -> (ContContext, ContContext, ContContext) {
    let at = |back: usize| {
        if ply >= back {
            ContContext { pt: stack[ply - back].moved_pt, to: stack[ply - back].moved_to }
        } else {
            ContContext::default()
        }
    };

    (at(1), at(2), at(4))
}

impl Worker<'_> {
    /// Static evaluation with the pawn structure pulled from the cache.
    #[inline]
    fn evaluate(&mut self) -> i32 {
        let phase = extract_phase(&self.accumulator);
        let pawn = self.pawn_cache.probe(&self.pos);
        let features = SharedFeatures::with_pawn(&self.pos, &pawn, self.xb_store());
        evaluate_generic::<i32>(&self.pos, &self.accumulator, phase, &self.eval_params, Some(&features))
    }

    /// Raw eval shifted by the correction history tables, clamped to non-mate.
    #[inline]
    fn corrected_eval(&self, raw_eval: i32, sp: &SearchParams) -> i32 {
        let correction = self.history.correction(
            self.pos.stm, self.pos.pawn_key, self.pos.minor_key, self.pos.major_key, sp.minor_corr_weight, sp.major_corr_weight,
        ) / history::CORRECTION_SCALE;

        (raw_eval + correction).clamp(-MATE_BOUND, MATE_BOUND)
    }

    /// Negamax with alpha-beta pruning.
    ///
    /// Since chess is zero-sum, we maximize the evaluation from the current side's
    /// perspective at every node, negating the score as it returns up the tree.
    ///
    /// PVS layered on top:
    /// After the presumed best move gets a full-window search,
    /// all others are probed with a zero-width "scout" window (alpha, alpha+1).
    /// Most confirm they're worse, and a fail-high costs a re-search rarely
    /// enough to stay a net win.
    fn negamax<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
    ) -> Result<i32, SearchAborted> {
        self.stack[ply].pv.len = 0;
        let sp = &searcher.cfg.search_params;

        if searcher.nodes.is_multiple_of(NODE_CHECK_INTERVAL) && searcher.check_signals() {
            return Err(SearchAborted);
        }

        searcher.nodes += 1;

        if self.is_draw(ply, &searcher.zobrist_trail) {
            return Ok(draw_score(searcher.nodes));
        }

        if ply as i32 > searcher.sel_depth {
            searcher.sel_depth = ply as i32;
        }

        if depth <= 0 {
            return self.qsearch::<N>(searcher, alpha, beta, ply, None, 0);
        }

        if ply >= MAX_PLY {
            return Ok(self.evaluate());
        }

        // The slot LMR reads at ply + 1 was zeroed here by the parent.
        self.stack[ply + 2].cutoff_count = 0;

        // ── Mate Distance Pruning
        // Clamp [alpha, beta] to the theoretical score limits for this ply:
        // the worst achievable score is mated_in(ply), and the best is mate_in(ply + 1).
        // If the window collapses (a >= b), every score this subtree can reach already
        // satisfies the bound, so return a without searching a move.
        //
        // Skipped at the root, where a cutoff would return before root_moves is scored.
        let (alpha, beta) = if N::ROOT {
            (alpha, beta)
        } else {
            let a = alpha.max(mated_in(ply));
            let b = beta.min(mate_in(ply + 1));
            if a >= b {
                return Ok(a);
            }
            (a, b)
        };

        let alpha_orig = alpha;
        let excluded = self.stack[ply].excluded;

        // ── TT Probe (~128 Elo)
        // Have we seen this position before? Move orders transpose, so often enough we
        // have, and an entry stored deep enough settles the window without a search.
        //
        // No probe during a singular verification: the entry here is the excluded move
        // itself, and its score is the very cutoff the verification exists to test.
        let tt_probe = if excluded.is_null() { searcher.tt.probe(self.pos.hash, ply) } else { TtData::NONE };
        let tt_move = tt_probe.mv(&self.pos);

        // A collision invalidates the whole slot, not just the move it stored, so the
        // entry reads as a miss to the cutoff, the static eval and the pv bit alike.
        let tt_probe = if tt_move == TtMove::Collision { TtData::NONE } else { tt_probe };

        #[rustfmt::skip]
        if !N::PV
            && tt_probe.depth >= depth
            && tt::can_cutoff(tt_probe.bound, tt_probe.score, alpha, beta)
        {
            return Ok(tt_probe.score);
        }

        let checkers = self.xb_checkers();
        let in_check = checkers.is_not_empty();

        // ── Check Extension (~11 Elo)
        // Being in check is forcing, so don't let the horizon cut us off
        // mid-tactic. Extend by one ply so the reply is always searched.
        let depth = if in_check { depth + 1 } else { depth };

        // ── Static Eval
        // Our best guess at how good this position is without searching deeper.
        // Meaningless when in check (we're forced to respond, not evaluate).
        // A TT hit already carries this position's raw eval, so reuse it and skip
        // the full evaluation; the stored sentinel (an in-check store) falls through.
        let raw_static_eval = if in_check {
            tt::SCORE_NONE
        } else if tt_probe.eval != tt::SCORE_NONE {
            tt_probe.eval
        } else {
            self.evaluate()
        };

        // ── Correction History
        // The evaluator has systematic biases tied to structural features
        // it can't see directly. Correction tables observe (search - eval)
        // deltas keyed by such features: pawn structure, minor and major piece
        // placement. They then nudge future evals of positions sharing those
        // keys toward the truth.
        let static_eval = if in_check { tt::SCORE_NONE } else { self.corrected_eval(raw_static_eval, sp) };

        // A node in check has no eval of its own to publish, so it republishes its
        // grandparent's and a descendant's two-ply hop still lands on the last eval
        // there was, however long the check sequence ran. The local stays NONE: in-frame
        // logic needs "no eval here" to mean exactly that.
        self.stack[ply].static_eval = if in_check && ply >= 2 { self.stack[ply - 2].static_eval } else { static_eval };

        // ── Improving Flag
        // Has our position strengthened since our last turn?
        // Inheritance above makes one hop reach the last eval; no eval two
        // plies back means the root edge, which counts as improving.
        let improving = !in_check
            && (ply >= 2)
                .then(|| self.stack[ply - 2].static_eval)
                .filter(|&e| e != tt::SCORE_NONE)
                .is_none_or(|past| static_eval > past);

        // ── TT-Clamped Eval
        // A mate score is a distance, not a valuation, so it never stands in for one.
        let tt_clamped_eval = if is_mate(tt_probe.score) {
            static_eval
        } else {
            tt::clamp_to_bound(tt_probe.bound, tt_probe.score, static_eval)
        };

        // ── Reverse Futility Pruning (~52 Elo)
        // Position is already so good that even after subtracting a generous
        // margin, we're still above beta. The opponent wouldn't have let us
        // get here, so cut the node without searching it.
        //
        // The eval is an unsearched guess and beta is the least this node
        // proved, so the score handed back splits the difference.
        #[rustfmt::skip]
        if !in_check
            && !N::PV
            && excluded.is_null()
            && depth <= sp.rfp_depth
            && !is_mate(tt_clamped_eval)
        {
            let margin = sp.rfp_base_margin
                + sp.rfp_margin * (depth - improving as i32)
                + sp.rfp_quad_margin * depth * depth;

            if tt_clamped_eval - margin >= beta {
                return Ok((tt_clamped_eval + beta) / 2);
            }
        }

        // ── Razoring (~17 Elo)
        // Position is so far below alpha that a full-depth search is unlikely
        // to recover. Drop straight into qsearch to confirm.
        if !in_check
            && !N::PV
            && excluded.is_null()
            && depth <= sp.razoring_depth
            && static_eval + sp.razoring_margin * depth < alpha
        {
            let score = self.qsearch::<N>(searcher, alpha, beta, ply, None, 0)?;
            if score <= alpha {
                return Ok(score);
            }
        }

        // ── Null Move Pruning (~85 Elo)
        // If our position is so good that we can pass the turn (do nothing)
        // and still beat beta after a reduced search, the opponent would
        // never allow this line. Skip it. The "null move" is the pass.
        if !in_check
            && !N::PV
            && excluded.is_null()
            && !self.stack[ply].is_null
            && !self.is_nmp_verif
            && tt_clamped_eval >= beta
            && self.pos.has_non_pawn_material(self.pos.stm)
        {
            let eval_r = ((tt_clamped_eval - beta) / sp.nmp_eval_divisor).min(sp.nmp_eval_max);
            let r = sp.nmp_base_r + depth / sp.nmp_depth_divisor + eval_r;
            let null_depth = (depth - r - sp.nmp_ply_offset).max(0);

            self.stack[ply].moved_pt = PieceType::None;
            self.stack[ply + 1].is_null = true;

            let undo = self.pos.make_null_move();
            searcher.zobrist_trail.push(self.pos.hash);

            let score = self.negamax::<NonPvNode>(searcher, null_depth, -beta, -beta + 1, ply + 1);

            searcher.zobrist_trail.pop();
            self.pos.unmake_null_move(&undo);
            self.stack[ply + 1].is_null = false;

            let score = -score?;
            if score >= beta && !is_loss(score) {
                let null_score = if is_win(score) { beta } else { score };

                // ── Verification Search
                // At or below nmp_verif_min_depth the cutoff is taken on trust, where verifying
                // would cost more than the rare zugzwang it catches. Above it, re-search
                // the position at null_depth with NMP suppressed: a second fail-high
                // confirms the cutoff, and a fail-low means passing hid a zugzwang or a
                // tactic, so the move loop runs after all.
                if depth <= sp.nmp_verif_min_depth {
                    return Ok(null_score);
                }

                self.is_nmp_verif = true;
                let verif = self.negamax::<NonPvNode>(searcher, null_depth, beta - 1, beta, ply);
                self.is_nmp_verif = false;
                if verif? >= beta {
                    return Ok(null_score);
                }
            }
        }

        // One pin scan for the whole node: legality and every SEE read it.
        let pins = Pins::new(&self.pos);

        // ── ProbCut (~4 Elo)
        // A capture that clears a raised beta (beta + margin) under a shallow search
        // would almost surely clear plain beta at full depth, so the node is a
        // near-certain cutoff. Prove it cheaply instead of searching every move:
        // qsearch as the filter, a reduced search only to confirm a pass.
        if !N::PV && !in_check && excluded.is_null() && depth >= sp.probcut_depth_min && !is_mate(beta) {
            let probcut_beta = beta + sp.probcut_margin;
            let probcut_depth = (depth - 4).max(1);

            let stm = self.pos.stm;
            let opp = stm.opposite();
            let ksq = pins.king(stm);
            let pinned = pins.blockers(stm);

            let mut picker = MovePicker::new_qsearch(None, searcher.cfg, pins, false);

            self.xb_enter(ply);

            while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
                if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                    continue;
                }

                // The capture must win enough material to plausibly reach the
                // raised beta from here; a smaller swing can't clear it.
                if !see_ge(&self.pos, mv, probcut_beta - static_eval, &pins) {
                    continue;
                }

                let saved_acc = self.accumulator;
                let undo = self.pos.make_move(mv, &mut self.accumulator);
                self.xb_make(mv);

                searcher.tt.prefetch(self.pos.hash);
                searcher.zobrist_trail.push(self.pos.hash);

                let qscore = self.qsearch::<NonPvNode>(searcher, -probcut_beta, -probcut_beta + 1, ply + 1, Some(mv.to()), 0);
                let value = match qscore {
                    Ok(v) if -v >= probcut_beta => self
                        .negamax::<NonPvNode>(searcher, probcut_depth, -probcut_beta, -probcut_beta + 1, ply + 1)
                        .map(|x| -x),

                    Ok(v) => Ok(-v),
                    Err(e) => Err(e),
                };

                searcher.zobrist_trail.pop();
                self.pos.unmake_move(mv, &undo);
                self.xb_unmake(mv, ply);
                self.accumulator = saved_acc;

                let value = value?;
                if value >= probcut_beta {
                    searcher
                        .tt
                        .store(self.pos.hash, ply, probcut_depth, value, mv, tt::Bound::Lower, tt_probe.pv, raw_static_eval);

                    return Ok(value);
                }
            }
        }

        // ── Internal Iterative Reduction (~14 Elo)
        // No TT move means we are searching blind, and blind ordering does not
        // deserve full depth. The entry this search stores hands the next iteration
        // the move it was missing.
        let depth = if depth >= sp.iir_depth && tt_move.get().is_none() { depth - sp.iir_reduction } else { depth };

        // ──────── Move loop ────────

        let mut res = MoveResult { move_count: 0, best_eval: -INF, alpha, best_move: Move::null() };

        if N::ROOT {
            self.xb_enter(ply);

            for i in 0..searcher.root_moves.len() {
                let mv = searcher.root_moves[i].mv;

                // ── Root LMR (~21 Elo)
                // Root moves are pre-sorted by the previous iteration's scores,
                // so late moves in the list are already our worst guesses.
                // Scout them at reduced depth; a fail-high triggers a full re-search.
                let reduction = if depth >= 2 && i >= 1 && mv.is_quiet() && !in_check {
                    searcher.cfg.lmr(depth, i + 1) / LMR_SCALE
                } else {
                    0
                };

                // Per-root-move node accounting for best-move stability TM.
                // Aspiration re-searches accumulate into the same slot;
                // all that work belongs to this move.
                let nodes_before = searcher.nodes;
                self.search_move::<N>(searcher, mv, depth, &mut res, beta, ply, Some(i), reduction, 0)?;
                searcher.root_moves[i].nodes += searcher.nodes - nodes_before;

                if res.alpha >= beta {
                    break;
                }
            }
        } else {
            let stm = self.pos.stm;
            let opp = stm.opposite();
            // Wider than the setwise fill, which ends in & !generator and so
            // drops the squares holding that side's own rooks and queens. Our
            // pieces can never stand there, so the two maps are interchangeable
            // here and the node count does not move.
            let threats = self.xb_threats(opp);
            let ksq = pins.king(stm);
            let pinned = pins.blockers(stm);

            self.stack[ply].quiet_count = 0;
            self.stack[ply].capture_count = 0;

            // Continuation history contexts: the piece-to-square pair of
            // the move played 1, 2, and 4 plies ago from this node.
            // These index into separate cont-hist tables and let the move
            // picker incorporate recent positional context into quiet ordering.
            let (cont1, cont2, cont4) = cont_contexts(&self.stack[..], ply);

            let mut picker =
                MovePicker::new(tt_move.get(), searcher.cfg, pins, self.stack[ply].killers, threats, cont1, cont2, cont4);

            self.xb_enter(ply);

            while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
                if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                    continue;
                }

                // The verification weighs every move except the one it set aside.
                if mv == excluded {
                    continue;
                }

                let mut appended_quiet = false;

                if mv.is_history_quiet() && self.stack[ply].quiet_count < MAX_TRACKED_QUIETS {
                    let count = self.stack[ply].quiet_count;
                    self.stack[ply].quiet_moves[count] = mv;
                    self.stack[ply].quiet_count += 1;
                    appended_quiet = true;
                }

                let mut appended_capture = false;

                if mv.is_capture() && !mv.is_promotion() && self.stack[ply].capture_count < MAX_TRACKED_CAPTURES {
                    let count = self.stack[ply].capture_count;
                    self.stack[ply].capture_moves[count] = mv;
                    self.stack[ply].capture_count += 1;
                    appended_capture = true;
                }

                // ── Futility Pruning (~10 Elo)
                // At shallow depth, if static eval is already so far below alpha
                // that a quiet move is unlikely to raise it, skip the move.
                if !in_check
                    && mv.is_quiet()
                    && !N::PV
                    && res.move_count >= 1
                    && depth <= sp.fp_depth
                    && static_eval + sp.fp_margin * depth <= res.alpha
                {
                    continue;
                }

                // ── Late Move Pruning (~14 Elo)
                // At shallow depth, quiet moves beyond a fixed count threshold
                // are unlikely to be the best move, so skip them entirely.
                if !in_check
                    && mv.is_quiet()
                    && !N::PV
                    && depth <= sp.lmp_depth
                    && res.move_count as i32 >= sp.lmp_base + depth * depth
                {
                    continue;
                }

                // ── History Pruning (~7 Elo)
                // A quiet with deeply negative history has been punished
                // repeatedly by the gravity update. Trust that signal at
                // shallow depth and skip the full search.
                //
                // Gate uses !is_tactical() rather than is_quiet() so
                // non-capture promotions are excluded; is_quiet only
                // filters captures, and pruning a queen promotion on
                // global history would be absurd.
                //
                // Killers are exempt; a killer is a per-ply refutation whose
                // global history is often deeply negative, terrible in most
                // positions and saving in this one.
                if !in_check
                    && !N::PV
                    && !mv.is_tactical()
                    && res.move_count >= 1
                    && mv != self.stack[ply].killers[0]
                    && mv != self.stack[ply].killers[1]
                    && depth <= sp.hist_prune_depth
                {
                    let pt = self.pos.expect_piece_at(mv.from());
                    let hist = self
                        .history
                        .score_quiet(self.pos.stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4);

                    if hist < -sp.hist_prune_margin * depth {
                        continue;
                    }
                }

                // ── SEE Pruning (~20 Elo)
                // Skip moves whose destination-square exchange clearly loses material.
                //
                // Captures scale linearly: SEE is an accurate verdict on a capture, since
                // the material swings on that square and nowhere else, so the tolerance
                // only has to loosen gently with depth.
                //
                // Quiets scale quadratically: there SEE is a crude proxy, because a quiet
                // move's value usually lives elsewhere in the tree (threats, structure,
                // follow-ups several plies out). Deeper searches find that for themselves,
                // so the margin loosens fast and only the shallow "moving into a trap"
                // cases get cut.
                if !in_check && !N::PV && res.move_count >= 1 {
                    let margin =
                        if mv.is_capture() { -sp.see_capture_margin * depth } else { -sp.see_quiet_margin * depth * depth };

                    if !see_ge(&self.pos, mv, margin, &pins) {
                        continue;
                    }
                }

                // ── Late Move Reductions (~90 Elo)
                // Moves late in the list are unlikely to beat alpha.
                // Search them at reduced depth; re-search fully on surprise.
                let reduction = if depth >= 2 && res.move_count >= 1 && mv.is_quiet() && !in_check {
                    let mut r = searcher.cfg.lmr(depth, res.move_count + 1);
                    let pt = self.pos.expect_piece_at(mv.from());
                    let hist = self
                        .history
                        .score_quiet(self.pos.stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4);

                    if mv == self.stack[ply].killers[0] || mv == self.stack[ply].killers[1] {
                        r -= sp.killer_lmr_bonus;
                    }

                    // A quiet that newly attacks a bigger piece is forcing the way a check is:
                    // answer it or lose the material.
                    r -= self.pos.new_threats(pt, mv.from(), mv.to()).popcount() as i32 * sp.threat_lmr_bonus;
                    // Fail-highs are piling up at this depth; the late quiets here are
                    // unlikely to be the move, so reduce them harder.
                    r += sp.fhc_lmr_malus * (self.stack[ply + 1].cutoff_count > 2) as i32;

                    r += (!improving as i32) * (384 - depth * 12).max(96);

                    let max_r = (depth - sp.lmr_retained).max(0) * LMR_SCALE;
                    (r - hist / sp.lmr_hist_div).clamp(0, max_r) / LMR_SCALE
                } else {
                    0
                };

                let mut extension = 0i32;

                // ── Singular Extensions (~7 Elo)
                // The TT move already came back strong from a deep search, and the question
                // here is whether it stands alone. Re-search every other move in a window
                // pinned just under its score; if they all fall short, nothing rivals it.
                // A position held up by a single move is sharp, and depth is worth most
                // where the score turns on one line, so that move gets the extra ply.
                if !N::ROOT
                    && !N::PV
                    && excluded.is_null()
                    && Some(mv) == tt_move.get()
                    && depth >= sp.singext_min_depth
                    && tt_probe.depth >= depth - sp.singext_tt_depth
                    && tt_probe.bound != tt::Bound::Upper
                    && !is_mate(tt_probe.score)
                {
                    let sing_beta = (tt_probe.score - depth * sp.singext_margin).max(-MATE_BOUND);
                    let sing_depth = (depth - 1) / 2;

                    // The verification recurses at this same ply and stomps the quiet
                    // and capture lists this node is still building for its own history
                    // update, so snapshot them and restore once it returns.
                    let saved = self.stack[ply];

                    self.stack[ply].excluded = mv;
                    let sing_score = self.negamax::<NonPvNode>(searcher, sing_depth, sing_beta - 1, sing_beta, ply);
                    self.stack[ply].excluded = Move::null();
                    self.stack[ply].quiet_moves = saved.quiet_moves;
                    self.stack[ply].quiet_count = saved.quiet_count;
                    self.stack[ply].capture_moves = saved.capture_moves;
                    self.stack[ply].capture_count = saved.capture_count;

                    let sing_score = sing_score?;
                    if sing_score < sing_beta {
                        extension = 1;
                    } else if sing_score >= beta {
                        // ── Multicut (~15 Elo)
                        // The TT bound already reads the TT move as a fail-high, and with it
                        // excluded the verification cleared beta anyway: a second move beats it
                        // too, so this is a cut node, not a singular one, and the bound stands.
                        // Mate scores return plain beta instead, since a verification missing the
                        // best move cannot be trusted on the distance.
                        return Ok(if is_mate(sing_score) { beta } else { sing_score });
                    }
                }

                // Context for child node's cont-hist lookup
                self.stack[ply].moved_pt = self.pos.expect_piece_at(mv.from());
                self.stack[ply].moved_to = mv.to();

                self.search_move::<N>(searcher, mv, depth, &mut res, beta, ply, None, reduction, extension)?;

                if likely(res.alpha >= beta) {
                    self.stack[ply].cutoff_count += 1;

                    #[cfg(feature = "mvpstats")]
                    {
                        use crate::engine::mvpstats::{CutoffKind, record_cutoff};

                        let kind = if Some(mv) == tt_move.get() {
                            CutoffKind::Hash
                        } else if mv.is_capture() {
                            CutoffKind::Capture
                        } else if mv == self.stack[ply].killers[0] || mv == self.stack[ply].killers[1] {
                            CutoffKind::Killer
                        } else {
                            CutoffKind::Quiet
                        };

                        record_cutoff(res.move_count as u32, kind);
                    }

                    // ── History Gravity Heuristic (~95 Elo)
                    // When a move causes a beta-cutoff, it's presumably a strong response.
                    // We reward it so it surfaces earlier in future sibling nodes,
                    // and we punish the preceding moves that failed to refute the branch.
                    //
                    // Bonus is quadratic with depth to prioritize deep heuristics,
                    // then scaled and capped (hist_bonus_mult, hist_bonus_cap) so
                    // no single deep search permanently dominates the table's ±16384 attractor.
                    let bonus = (depth.pow(2) * sp.hist_bonus_mult).min(sp.hist_bonus_cap);

                    if mv.is_history_quiet() {
                        let pt = self.pos.expect_piece_at(mv.from());

                        self.history.update(stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4, bonus);

                        // ── Killer Moves (~35 Elo)
                        // Maintain a 2-slot pseudo-Least-Recently-Used cache for tracking quiet cutoffs.
                        // If the move isn't already the primary killer, shift the old primary to slot 1
                        // and promote the new move to slot 0. If it was slot 1, this natively swaps them.
                        if mv != self.stack[ply].killers[0] {
                            self.stack[ply].killers[1] = self.stack[ply].killers[0];
                            self.stack[ply].killers[0] = mv;
                        }
                    } else if mv.is_capture() && !mv.is_promotion() {
                        // ── Capture History Update
                        // Promotion-captures are excluded: the picker scores them outside the
                        // MVV-LVA and capture-history blend (see add_promo_caps), so nothing
                        // would ever read the entry.
                        //
                        // self.pos is the parent position here, since search_move already unmade
                        // the move: the victim sits back on the destination square, or it is en
                        // passant and a pawn by definition.
                        let attacker = self.pos.expect_piece_at(mv.from());
                        let victim = if mv.is_en_passant() { PieceType::Pawn } else { self.pos.piece_at(mv.to()) };
                        self.history.update_capture(stm, attacker, mv.to(), victim, bonus);
                    }

                    // ── Quiet Malus (~25 Elo)
                    // A bonus alone can only lift entries, so a move that cut once
                    // and keeps failing never comes back down.
                    let quiet_limit = self.stack[ply].quiet_count - appended_quiet as usize;
                    for i in 0..quiet_limit {
                        let qm = self.stack[ply].quiet_moves[i];
                        let q_pt = self.pos.expect_piece_at(qm.from());
                        self.history.update(stm, q_pt, qm.from(), qm.to(), threats, cont1, cont2, cont4, -bonus);
                    }

                    // ── Capture Malus
                    // MVV-LVA ranks by material alone, so the table is the only place
                    // a rich capture that keeps failing here can be marked down.
                    let capture_limit = self.stack[ply].capture_count - appended_capture as usize;
                    for i in 0..capture_limit {
                        let cm = self.stack[ply].capture_moves[i];
                        let attacker = self.pos.expect_piece_at(cm.from());
                        let victim = if cm.is_en_passant() { PieceType::Pawn } else { self.pos.piece_at(cm.to()) };
                        self.history.update_capture(stm, attacker, cm.to(), victim, -bonus);
                    }
                    break;
                }
            }
        }

        // No legal moves: checkmate (in check) or stalemate (not).
        // Excluding the only legal move empties the list too, so fail low:
        // the verification reads the TT move as singular, not the board as lost.
        if unlikely(res.move_count == 0) {
            return if !excluded.is_null() {
                Ok(alpha)
            } else if in_check {
                Ok(mated_in(ply))
            } else {
                Ok(0)
            };
        }

        let bound = if res.best_eval >= beta {
            tt::Bound::Lower
        } else if res.best_eval > alpha_orig {
            tt::Bound::Exact
        } else {
            tt::Bound::Upper
        };

        // ── TT store
        // The verification searched this position with a move missing, so its
        // result is a lie about the real node. Keep it out of the table.
        if excluded.is_null() {
            searcher.tt.store(
                self.pos.hash,
                ply,
                depth,
                res.best_eval,
                res.best_move,
                bound,
                N::PV || tt_probe.pv,
                raw_static_eval,
            );
        }

        // ── Correction History Update
        // A capture or promotion settles the node on tactics, which say nothing
        // about evaluator bias.
        // Skip when the bound direction contradicts the diff: a fail-high with
        // best_eval <= static_eval, or a fail-low with best_eval >= static_eval,
        // carries no useful structural signal.
        if !in_check
            && excluded.is_null()
            && !res.best_move.is_null()
            && !res.best_move.is_tactical()
            && res.best_eval.abs() < MATE_BOUND
            && !((bound == tt::Bound::Lower && res.best_eval <= static_eval)
                || (bound == tt::Bound::Upper && res.best_eval >= static_eval))
        {
            let diff = res.best_eval - raw_static_eval;

            self.history
                .update_correction(self.pos.stm, self.pos.pawn_key, self.pos.minor_key, self.pos.major_key, diff, depth);
        }
        Ok(res.best_eval)
    }

    /// The rows a ply's moves all return to. Taken once, before the moves,
    /// because a child's make would otherwise overwrite what its parent's
    /// unmake has to replay.
    #[inline(always)]
    fn xb_enter(&mut self, ply: usize) {
        #[cfg(not(feature = "nostore"))]
        self.xorboard.snapshot(&mut self.xb_undo[ply]);
        #[cfg(feature = "nostore")]
        let _ = ply;
    }

    #[cfg(not(feature = "nostore"))]
    #[inline(always)]
    fn xb_store(&self) -> Option<&XorBoard> { Some(&self.xorboard) }

    #[cfg(feature = "nostore")]
    #[inline(always)]
    fn xb_store(&self) -> Option<&XorBoard> { None }

    /// Brings the rows up to date for a move the board has already made.
    #[inline(always)]
    fn xb_make(&mut self, mv: Move) {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.make(&self.pos, mv);
            debug_assert!(self.xorboard.agrees_with(&self.pos), "xorboard drift after {mv:?}");
        }
        #[cfg(feature = "nostore")]
        let _ = mv;
    }

    #[inline(always)]
    fn xb_unmake(&mut self, mv: Move, ply: usize) {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.unmake(mv, &self.xb_undo[ply]);
        }
        #[cfg(feature = "nostore")]
        let _ = (mv, ply);
    }

    #[inline(always)]
    fn xb_checkers(&self) -> Bitboard {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.checkers(&self.pos)
        }
        #[cfg(feature = "nostore")]
        {
            self.pos.checkers()
        }
    }

    #[cfg(not(feature = "nostore"))]
    #[inline(always)]
    fn xb_rows(&self) -> &XorBoard { &self.xorboard }

    #[cfg(feature = "nostore")]
    #[inline(always)]
    fn xb_rows(&self) -> core::marker::PhantomData<&()> { core::marker::PhantomData }

    #[inline(always)]
    fn xb_threats(&self, color: Color) -> Bitboard {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.danger(color)
        }
        #[cfg(feature = "nostore")]
        {
            self.pos.threats(color)
        }
    }

    /// Make a move, search it, unmake it. The innermost loop body every
    /// move in the tree passes through exactly once.
    ///
    /// `mv` must be legal: at the root the move list was generated legal, and in
    /// the interior the move loop calls `is_legal` before every invocation.
    fn search_move<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        mv: Move,
        depth: i32,
        res: &mut MoveResult,
        beta: i32,
        ply: usize,
        root_idx: Option<usize>,
        mut reduction: i32,
        extension: i32,
    ) -> Result<(), SearchAborted> {
        let sp = &searcher.cfg.search_params;
        let saved_acc = self.accumulator;
        let undo = self.pos.make_move(mv, &mut self.accumulator);
        self.xb_make(mv);

        searcher.tt.prefetch(self.pos.hash);

        // ── Gives-Check LMR Adjustment (~4 Elo)
        // A move that delivers check is forcing: the opponent must respond.
        // Reduce it less; a reduction here drops the horizon inside
        // the forced sequence, which is the worst place to stop.
        if self.xb_checkers().is_not_empty() {
            reduction = (reduction - sp.check_lmr_bonus).max(0);
        }

        res.move_count += 1;
        searcher.zobrist_trail.push(self.pos.hash);

        if N::ROOT {
            searcher.print_currmove(depth, mv, res.move_count);
        }

        let eval = self.pvs::<N>(searcher, depth, res.alpha, beta, ply, res.move_count == 1, reduction, extension, mv);

        searcher.zobrist_trail.pop();
        self.pos.unmake_move(mv, &undo);
        self.xb_unmake(mv, ply);
        self.accumulator = saved_acc;

        let eval = eval?;

        if N::ROOT
            && let Some(i) = root_idx
        {
            searcher.root_moves[i].score = eval;
        }

        if eval > res.best_eval {
            res.best_eval = eval;
            res.best_move = mv;

            if N::ROOT
                && let Some(i) = root_idx
            {
                searcher.root_moves[i].pv.compose(mv, &self.stack[ply + 1].pv);
            }

            if eval > res.alpha {
                res.alpha = eval;
                let (current_stack, next_stack) = self.stack.split_at_mut(ply + 1);
                current_stack[ply].pv.compose(mv, &next_stack[0].pv);
            }
        }
        Ok(())
    }

    // ── Principal Variation Search (~14 Elo)
    /// Full window for the first move,
    /// zero-width scout for the rest,
    /// full window again on surprise fail-high.
    #[inline]
    fn pvs<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        is_first: bool,
        reduction: i32,
        extension: i32,
        mv: Move,
    ) -> Result<i32, SearchAborted> {
        let sp = &searcher.cfg.search_params;

        // Only the singular move arrives with an extension; the rest pass 0.
        let search_depth = depth - 1 + extension;

        if is_first {
            // No bound yet; search wide open.
            return Ok(-self.negamax::<N::Next>(searcher, search_depth, -beta, -alpha, ply + 1)?);
        }

        // ── LMR Scout
        // Late quiet moves get a shallower scout. If the reduced search
        // still beats alpha, the move earned a full-depth re-search.
        let reduced_depth = search_depth - reduction;
        let mut score = -self.negamax::<NonPvNode>(searcher, reduced_depth, -alpha - 1, -alpha, ply + 1)?;

        // Re-search at full depth if the reduced scout found something.
        if score > alpha && reduction > 0 {
            score = -self.negamax::<NonPvNode>(searcher, search_depth, -alpha - 1, -alpha, ply + 1)?;

            // ── Post-LMR Continuation History (~8 Elo)
            // The reduced scout beat alpha; the full-depth re-search settles it.
            // A fail-low means the reduction over-promised; a fail-high means a cutoff.
            // Punish or reward continuation history accordingly, ordering only.
            if mv.is_history_quiet() && (score <= alpha || score >= beta) {
                let stm = self.pos.stm.opposite();
                let pt = self.stack[ply].moved_pt;
                let to = self.stack[ply].moved_to;

                let (cont1, cont2, cont4) = cont_contexts(&self.stack[..], ply);

                let bonus = (depth.pow(2) * sp.hist_bonus_mult).min(sp.hist_bonus_cap);
                let signed = if score <= alpha { -bonus } else { bonus };

                self.history.update_conthist(stm, pt, to, cont1, cont2, cont4, signed);
            }
        }

        if score > alpha && score < beta {
            // Genuine improvement; search with full window on the PV.
            score = -self.negamax::<N::Next>(searcher, search_depth, -beta, -alpha, ply + 1)?;
        }
        Ok(score)
    }

    // ── Quiescence Search (~655 Elo)
    /// Evaluates positions only after all "noisy" (tactical/forcing) moves are resolved,
    /// preventing the horizon effect where a search stops right before a massive blunder.
    fn qsearch<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        mut alpha: i32,
        beta: i32,
        ply: usize,
        prev_to: Option<Square>,
        qs_ply: i32,
    ) -> Result<i32, SearchAborted> {
        self.stack[ply].pv.len = 0;
        let sp = &searcher.cfg.search_params;

        if searcher.nodes.is_multiple_of(NODE_CHECK_INTERVAL) && searcher.check_signals() {
            return Err(SearchAborted);
        }

        searcher.nodes += 1;

        if self.is_draw(ply, &searcher.zobrist_trail) {
            return Ok(draw_score(searcher.nodes));
        }

        if ply as i32 > searcher.sel_depth {
            searcher.sel_depth = ply as i32;
        }

        if ply >= MAX_PLY {
            return Ok(self.evaluate());
        }

        let alpha_orig = alpha;

        // ── QSearch TT Probe (~22 Elo)
        // Read TT entries stored by negamax or prior qsearch visits.
        // Cutoffs gated on non-PV nodes: PV nodes need the full capture
        // sequence for accurate PV reporting.
        //
        // Quiescence TT Move (~9 Elo)
        let qs_tt = searcher.tt.probe(self.pos.hash, ply);
        let qs_tt_move = qs_tt.mv(&self.pos);
        let qs_tt = if qs_tt_move == TtMove::Collision { TtData::NONE } else { qs_tt };

        if !N::PV && tt::can_cutoff(qs_tt.bound, qs_tt.score, alpha, beta) {
            return Ok(qs_tt.score);
        }

        let checkers = self.xb_checkers();
        let in_check = checkers.is_not_empty();
        let stm = self.pos.stm;
        let opp = stm.opposite();

        // ── QSearch Evaluations & Evasions
        // In check, static eval is meaningless; stand-pat drops to -INF
        // and the picker generates all evasions instead of just captures.
        // A TT hit already carries this position's raw eval; the stored
        // sentinel (an in-check store) falls through.
        let raw_eval = if in_check {
            tt::SCORE_NONE
        } else if qs_tt.eval != tt::SCORE_NONE {
            qs_tt.eval
        } else {
            // ── Lazy Eval
            let phase = extract_phase(&self.accumulator);
            let lazy = self.corrected_eval(evaluate_psqt(&self.pos, &self.accumulator, phase), sp);
            let lazy_floor = lazy - lazy_eval_margin(&self.pos, phase, sp);
            if lazy_floor >= beta {
                return Ok((lazy + beta) / 2);
            }
            self.evaluate()
        };

        let mut best_eval = if in_check {
            -INF
        } else {
            let eval = self.corrected_eval(raw_eval, sp);

            let stand_pat = if is_mate(qs_tt.score) { eval } else { tt::clamp_to_bound(qs_tt.bound, qs_tt.score, eval) };

            if stand_pat >= beta {
                return Ok((stand_pat + beta) / 2);
            }

            alpha = alpha.max(stand_pat);

            // ── Delta Pruning (~20 Elo)
            // Stand-pat already failed to beat alpha.
            // Even if we capture the opponent's most valuable piece, can we reach alpha?
            // If not, no capture in this position can raise us high enough, so bail early.
            let best_capturable = [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight, PieceType::Pawn]
                .into_iter()
                .find(|&pt| self.pos.pieces(pt, opp).is_not_empty())
                .map_or(0, |pt| searcher.cfg.mvvlva_v[pt]);

            if stand_pat + best_capturable + sp.delta_margin < alpha {
                return Ok(alpha);
            }
            stand_pat
        };

        let mut moves_made = 0;
        let mut best_move = Move::null();

        let pins = Pins::new(&self.pos);
        let ksq = pins.king(stm);
        let pinned = pins.blockers(stm);

        let mut picker = MovePicker::new_qsearch(qs_tt_move.get(), searcher.cfg, pins, in_check);

        let recapture_only = !in_check && qs_ply >= sp.qs_recapture_ply;

        self.xb_enter(ply);

        while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
            if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            // ── Recapture-only Deep QS (~1 Elo)
            // Past qs_recapture_ply, captures only matter if they continue
            // the forcing exchange on the square the opponent just moved to.
            // Speculative off-square captures are what explode in mutual-
            // attack soup (e.g. 30-bishop pathologicals). Tactics survive
            // the cut because combinations this deep are recapture chains.
            if recapture_only
                && let Some(prev) = prev_to
                && mv.to() != prev
            {
                continue;
            }

            // ── QSearch SEE Pruning (~65 Elo)
            // Skip captures whose destination-square trade loses material
            // for us. Disabled in check because evasions are forced and
            // the only legal reply is often a losing defensive capture.
            if !in_check && !see_ge(&self.pos, mv, 0, &pins) {
                continue;
            }

            let saved_acc = self.accumulator;
            let undo = self.pos.make_move(mv, &mut self.accumulator);
            self.xb_make(mv);

            moves_made += 1;
            searcher.zobrist_trail.push(self.pos.hash);

            let score = self.qsearch::<N>(searcher, -beta, -alpha, ply + 1, Some(mv.to()), qs_ply + 1);

            searcher.zobrist_trail.pop();
            self.pos.unmake_move(mv, &undo);
            self.xb_unmake(mv, ply);
            self.accumulator = saved_acc;

            let score = -score?;
            if score > best_eval {
                best_eval = score;
                best_move = mv;

                if score > alpha {
                    // The last plies of a line are searched here; without
                    // the compose the PV stops at the negamax boundary and
                    // a mate line is reported without its mating move.
                    let (current_stack, next_stack) = self.stack.split_at_mut(ply + 1);
                    current_stack[ply].pv.compose(mv, &next_stack[0].pv);

                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
        }

        if in_check && moves_made == 0 {
            return Ok(mated_in(ply));
        }

        // ── QSearch TT Store
        let bound = if best_eval >= beta {
            tt::Bound::Lower
        } else if best_eval > alpha_orig {
            tt::Bound::Exact
        } else {
            tt::Bound::Upper
        };

        searcher
            .tt
            .store_qs(self.pos.hash, ply, best_eval, best_move, bound, N::PV || qs_tt.pv, raw_eval);

        Ok(best_eval)
    }

    /// Fifty move rule, insufficient material, or repetition → immediate draw.
    ///
    /// Checked before move generation, so a checkmate delivered on exactly the
    /// 100th half-move is scored as a draw, the standard compromise.
    #[inline]
    fn is_draw(&self, ply: usize, trail: &[u64]) -> bool {
        self.pos.is_fifty_move_draw() || self.pos.is_draw_by_material() || (ply > 0 && self.is_repetition(self.pos.hash, trail))
    }

    /// Scans the history for a previous occurrence of the current hash.
    ///
    /// Any second occurrence is treated as a draw.
    /// While technically 3-fold is the rule, engines score the second
    /// to avoid searching infinite cycles and to prevent the match runners
    /// from flagging PVs that repeat.
    ///
    /// This uses a raw scan without side-to-move skipping because
    /// Zobrist already includes the side-to-move key. Contrast with
    /// `Position::is_threefold_repetition` which is optimized for
    /// adjudication contexts.
    #[inline]
    fn is_repetition(&self, key: u64, trail: &[u64]) -> bool {
        if trail.len() < 2 {
            return false;
        }

        let window = self.pos.halfmove_clock as usize;
        let start = trail.len().saturating_sub(window + 1);

        // search_move pushes the current hash before descending, so it sits at
        // len - 1 and saturating_sub(2) stops the scan one short of it.
        let end = trail.len().saturating_sub(2);
        if start > end {
            return false;
        }

        let needle = Vu64x4::splat(key);
        let slice = &trail[start..=end];
        let chunks = slice.rchunks_exact(4);
        let remainder = chunks.remainder();

        // We vector-scan the full slice, including opponent-ply hashes.
        // Zobrist already encodes side-to-move, so cross-ply hashes can never match.
        // The false-check cost is zero, and contiguous SIMD loads are cheaper than strided.
        //
        // SAFETY: Vu64x4::load uses _mm256_loadu_si256 (unaligned), which is correct
        // because Vec<u64> guarantees only 8-byte alignment, not the 32-byte alignment
        // that an aligned load would require. The chunk pointer is valid for 32 bytes
        // by Vec layout and rchunks_exact(4).
        for chunk in chunks {
            let vec = unsafe { Vu64x4::load(chunk.as_ptr()) };
            if vec.cmp_eq(needle).any() {
                return true;
            }
        }
        remainder.contains(&key)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::STARTPOS;

    /// Bench can't catch this: one param vector, nothing to differentiate. The tuner
    /// ran blind for the engine's life because every searcher read one global, so
    /// distinct candidates searched identically and scored noise.
    #[test]
    fn params_reach_search() {
        let run = |params| {
            let board = Position::from_fen(STARTPOS);
            let limits = Limits { depth: 8, silent: true, protocol: Protocol::Uci, ..Default::default() };
            let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, params);
            let mut s = Searcher::new(&cfg, &board, &[board.hash], Arc::new(TranspositionTable::new(16, 1)));
            s.iterative_deepening(&mut History::new());
            s.nodes
        };

        let tweaked = SearchParams { lmr_base: 0, lmr_divisor: 350, rfp_margin: 400, ..SearchParams::default() };
        assert_ne!(run(SearchParams::default()), run(tweaked), "params didn't reach search");
    }

    /// A `stop` landing before the first iteration ends: `bestmove` still has to
    /// name a move the search looked at, not the first one movegen listed.
    #[test]
    fn a_stop_before_the_first_iteration_still_leaves_a_line() {
        let board = Position::from_fen(STARTPOS);
        let limits = Limits { depth: 32, silent: true, protocol: Protocol::Uci, ..Default::default() };
        let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(true)), 0, SearchParams::default());
        let mut searcher = Searcher::new(&cfg, &board, &[board.hash], Arc::new(TranspositionTable::new(16, 1)));
        searcher.iterative_deepening(&mut History::new());
        assert!(searcher.prev_pv.len > 0, "the first iteration was abandoned");
        assert_eq!(searcher.best_move(), searcher.prev_pv.get(0), "the move played is not the one searched");
    }
}
