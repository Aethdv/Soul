//! Negamax alpha-beta search
//!
//! Iterative deepening is the outer loop: depth 1, then 2, then 3, each pass
//! seeding the next's move ordering. Every interior node runs negamax alpha-beta
//! with PVS on top, a full window for the first move and zero-width scouts for
//! the rest, and the leaves fall through to quiescence so the horizon never
//! lands mid-capture.
//!
//! # Architecture
//!
//! Lazy SMP: threads search the same root in parallel, sharing only the TT and
//! a stop flag. No explicit work distribution: each thread runs its own
//! iterative-deepening loop, and the TT is the coordination surface. Diversity
//! is emergent from threads cross-probing each other's entries at different depths.
//!
//! State splits into two entities per thread:
//! - `Searcher`: Owns global engine state (time management, history table, root moves).
//! - `Worker`: Owns thread-local mutability (the board, SIMD accumulator, ply stack).
//!
//! NodeType specialization: `RootNode`, `PvNode`, and `NonPvNode` are zero-cost
//! marker types, so negamax monomorphizes into three variants with the dead
//! branches compiled out of the hot path.

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

pub use crate::core::defs::Protocol;
use crate::{
    core::{
        board::Position,
        defs::{INF, MATE, MATE_BOUND, MAX_DEPTH, MAX_PLY, PieceType, Square, draw_score},
        moves::Move,
    },
    engine::{
        eval::{EvalParams, PawnCache, SharedFeatures, evaluate_generic, extract_phase},
        history::{self, ContContext, History},
        movegen::{gen_legal_moves, is_legal, is_pseudo_legal},
        movepicker::MovePicker,
        search_params::*,
        see::see_ge,
        tm::TimeManager,
        tt,
        tt::TranspositionTable,
    },
    tools::{perft::perft, tui},
    weave::{Vi16x8, Vu64x4},
};

pub const NODE_CHECK_INTERVAL: u64 = 2048;
/// Minimum node count before printing `currmove` UCI output.
/// Suppresses per-move noise during fast games while still
/// reporting progress during long analysis.
pub const CURRMOVE_NODE_THRESHOLD: u64 = 100_000_000;
pub const PRINT_UPDATE_MS: u128 = 25;

/// MAX_TRACKED_QUIETS matches MAX_MOVES today, but only because the legal quiet
/// count is bounded by the total pseudo-legal count. If MAX_MOVES grows, this
/// can shrink independently; we only care that quiet penalties cover all
/// quiets searched before a cutoff, not all possible quiets in a position.
pub const MAX_TRACKED_QUIETS: usize = 256;
/// Captures are far fewer per position than quiets. 64 covers all realistic
/// legal capture counts with headroom.
pub const MAX_TRACKED_CAPTURES: usize = 64;

/// One ply expressed in LMR table units. Sub-ply granularity lets each
/// per-move adjustment dose uniformly across the table, regardless of
/// the cell's absolute reduction.
pub const LMR_SCALE: i32 = 1024;

/// Chess search has three distinct contexts:
/// root (first ply, owns the move list), PV, and non-PV
/// (zero-window scouts that just need a yes/no answer).
///
/// By encoding these as types rather than runtime flags, the compiler
/// monomorphizes negamax into three tight variants with dead branches
/// eliminated entirely. Zero-cost polymorphism.
pub trait NodeType {
    const PV: bool;
    const ROOT: bool;
    type Next: NodeType;
}

pub struct RootNode;
pub struct PvNode;
pub struct NonPvNode;

pub struct MoveResult {
    pub move_count: usize,
    pub best_eval: i32,
    pub alpha: i32,
    pub best_move: Move,
}

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

// ──────── Searcher & Worker ────────
//
// Searcher owns the global search state: root moves, node counter,
// time management, zobrist trail. One per thread.
//
// Worker owns the mutable board that changes as we descend the tree:
// position, accumulator, per-ply stack.
//
// Each thread gets its own Searcher+Worker pair. Threads share only
// the TT, the stop flag, and the node counter; everything else is
// thread-private.

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
    pub stack: Box<[Stack; MAX_PLY + 1]>,
    pub history: &'h mut History,
    /// Per-search pawn-structure cache, keyed on the incremental `pawn_key`.
    pub pawn_cache: PawnCache,
    /// Search-constant eval weights, built once instead of per leaf.
    pub eval_params: EvalParams<i32>,
    /// Set while a null-move verification search is on the stack.
    /// Suppresses nested NMP, which would otherwise recurse forever on
    /// the same position at ever-shrinking depth.
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
    /// (every 2048 nodes), with a Relaxed store: no lock prefix,
    /// no cross-core contention. The display and node-limit paths sum
    /// all slots. Slightly stale, invisible at display intervals.
    pub node_slots: Arc<[AtomicU64]>,
    pub mvvlva_v: [i32; 8], // victim values, indexed by PieceType
    pub mvvlva_a: [i32; 8], // attacker penalties, indexed by PieceType
    /// `ln(i) · LMR_SCALE` lookup, indexed by depth or move count.
    /// Reduction composes as `base + table[d] · table[m] / div`, all in milli-plies.
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

// ── Principal Variation Line ──
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
}

/// Search was cut short: time, node limit, or external stop signal.
#[derive(Debug)]
pub struct SearchAborted;

impl SearchDisplay {
    pub const SILENT: Self = Self { show_wdl: false, go_pretty: false, pretty_print: false, show_currmove: false, use_ansi: false };
    pub const DEFAULT: Self = Self { show_currmove: true, use_ansi: true, ..Self::SILENT };
}

impl SearchConfig {
    /// Creates a configuration with default display settings.
    pub fn new(limits: Limits, start_time: Instant, stop: Arc<AtomicBool>, overhead: u64, search_params: SearchParams) -> Self {
        Self::new_full(limits, start_time, stop, overhead, SearchDisplay::DEFAULT, search_params)
    }

    /// Full constructor for fine-grained control over all parameters.
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

    /// `threads` zeroed node counters, one slot per thread.
    pub fn node_slots(threads: usize) -> Arc<[AtomicU64]> {
        (0..threads).map(|_| AtomicU64::new(0)).collect()
    }

    /// Composed LMR reduction in `LMR_SCALE` units.
    #[inline(always)]
    pub fn lmr(&self, depth: i32, m: usize) -> i32 {
        let base = lmr_base() * LMR_SCALE / 100;
        let div = lmr_divisor() * LMR_SCALE / 100;
        let d = self.lmr_table[depth as usize] as i32;
        let m = self.lmr_table[m] as i32;

        base + d * m / div
    }

    /// MVV-LVA lookup table from tunable parameters.
    ///
    /// Most Valuable Victim – Least Valuable Attacker:
    /// The simplest capture ordering that works.
    /// Prefer taking queens with pawns over taking pawns with queens.
    fn build_mvvlva(sp: &SearchParams) -> ([i32; 8], [i32; 8]) {
        let mut v = [0; 8];
        let mut a = [0; 8];

        macro_rules! map {
            ($pt:ident, $v:expr, $a:expr) => {
                v[PieceType::$pt.as_usize()] = $v;
                a[PieceType::$pt.as_usize()] = $a;
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
    pub const fn new() -> Self {
        Self { moves: [Move::null(); MAX_PLY], len: 0 }
    }

    /// Prepend `mv` to `tail`, forming a complete PV line.
    pub fn compose(&mut self, mv: Move, tail: &Line) {
        let n = tail.len.min(MAX_PLY - 1);

        self.moves[0] = mv;
        self.moves[1..=n].copy_from_slice(&tail.moves[..n]);
        self.len = 1 + n;
    }

    /// Retrieve a move from the PV line if the index is within bounds.
    #[inline(always)]
    pub fn get(&self, idx: usize) -> Option<Move> {
        if idx < self.len { Some(self.moves[idx]) } else { None }
    }
}

impl Default for Line {
    fn default() -> Self {
        Self::new()
    }
}

impl RootMove {
    #[inline]
    pub fn new(mv: Move) -> Self {
        Self { mv, score: -INF, pv: Box::new(Line::new()), nodes: 0 }
    }
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
        }
    }
}

impl<'cfg> Searcher<'cfg> {
    /// Search depth 1, then 2, then 3, ...
    ///
    /// Seems wasteful. Why redo shallow work? Two reasons:
    ///   1. Each iteration's move ordering feeds the next. A deep search with
    ///      good ordering is faster than a blind deep search.
    ///   2. Natural anytime algorithm: always have a best move ready from the
    ///      last completed iteration.
    #[inline]
    pub fn iterative_deepening(&mut self, history: &mut History) {
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
        // iteration would memset 123KB of ply stack every pass, for nothing.
        let mut worker = Worker {
            pos: self.root_pos,
            accumulator: root_acc,
            stack: vec![Stack::default(); MAX_PLY + 1]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            history,
            pawn_cache: PawnCache::new(),
            eval_params: EvalParams::<i32>::from_const(),
            is_nmp_verif: false,
        };

        let mut last_iter_elapsed = 0;

        // ── Singular Bailout ──
        // Only one legal move. Slash the budget to 5% so we exit the depth
        // loop almost instantly, banking the saved time.
        if self.root_moves.len() == 1 {
            self.tm.set_bm_stab_factor(tm_single_root() as f64 / 100.0);
        }

        // ── Lazy SMP ──
        // Every thread searches every depth, sharing nothing but the TT and
        // the stop flag. Diversity is emergent: thread A writes an entry at
        // the edge of what it can reach, thread B probes it mid-search and
        // redirects its tree along a branch A already explored. The TT is
        // the coordination surface.
        //
        // Helpers search the full depth ladder alongside main. The only
        // asymmetry is soft time management: main alone decides when to stop
        // starting iterations. Helpers still honor the global flag and the
        // shared hard cap (same start_time, same limits) via check_signals;
        // they just don't gate their own iterations on the clock; they feed
        // the TT for as long as the main thread is still running.

        for depth in 1..=depth_limit {
            self.iter_depth = depth;

            let elapsed = self.tm.elapsed().as_millis() as u64;
            let prev_depth_time = elapsed.saturating_sub(last_iter_elapsed);

            last_iter_elapsed = elapsed;

            // Between iterations:
            // Bail if soft limits say we probably
            // can't finish the next depth in time.
            //
            // prev_depth_time * 2 is a rough branching-factor proxy;
            // each additional ply typically costs about twice the previous one,
            // so if we can't afford that estimate we stop before starting it.
            //
            // Helpers skip time management: their job is to fill the TT, not
            // decide when to stop. Main calls the shots.
            if self.cfg.thread_id == 0
                && depth > 1
                && (elapsed >= self.tm.soft_limit().as_millis() as u64
                    || elapsed + (prev_depth_time * tm_iter_scale() as u64 / 100) > self.tm.hard_limit().as_millis() as u64
                    || (self.cfg.limits.softnodes > 0 && self.nodes >= self.cfg.limits.softnodes))
            {
                break;
            }

            if self.check_signals() {
                break;
            }

            // ── Aspiration Windows (~42 Elo) ──
            // A score rarely lurches between iterations, so we bracket the last
            // one in a narrow window instead of searching (-INF, INF). Tighter
            // bounds prune far more, and most iterations land inside for free.
            // When the score escapes, the bound that broke says which way we
            // were wrong: a fail-low drops alpha, a fail-high lifts beta. Each
            // retry widens the window so a score that truly moved doesn't thrash.
            let mut delta = asp_initial();
            let mut alpha = if depth >= asp_depth() { (self.prev_score - delta).max(-INF) } else { -INF };
            let mut beta = if depth >= asp_depth() { (self.prev_score + delta).min(INF) } else { INF };

            let mut aborted = false;

            loop {
                worker.pos = self.root_pos;
                worker.accumulator = root_acc;

                if worker.negamax::<RootNode>(self, depth, alpha, beta, 0, None).is_err() {
                    aborted = true;
                    break;
                }

                let score = self.root_moves[0].score;

                if score <= alpha {
                    alpha = (score - delta).max(-INF);
                } else if score >= beta {
                    beta = (score + delta).min(INF);
                } else {
                    break;
                }

                delta += delta / asp_widen_div();
            }

            if aborted {
                // On abort, we deliberately don't sort. The array is already correctly sorted
                // from the last fully completed iteration context. Sorting a partially aborted
                // iteration would corrupt the strict root ordering with cross-depth horizon nodes.
                break;
            }

            if self.is_stopped() {
                break;
            }

            // best move floats to front, feeding the next iteration's ordering.
            self.root_moves.sort_by_key(|m| Reverse(m.score));

            if self.is_stopped() {
                break;
            }

            let new_score = self.root_moves[0].score;

            // ── Best-Move Stability TM (~20 Elo) ──
            // Scale the soft budget by the best move's share
            // of total search effort. A large share means the search keeps
            // confirming one move, so shrink the budget. A small share means
            // effort is scattered across candidates, so stretch it.
            //
            //   percent = clamp(floor, base − scale · best_nodes / total_nodes)
            //
            // Gated below bm_stab_depth: early iterations haven't
            // accumulated enough node signal for the ratio to be meaningful.
            if depth >= bm_stab_depth() {
                let best_nodes = self.root_moves[0].nodes;
                let total_nodes = self.nodes.max(1);
                let effort_discount = bm_stab_scale() as u64 * best_nodes / total_nodes;
                let percent = (bm_stab_base() as u64).saturating_sub(effort_discount).max(bm_stab_floor() as u64);

                self.tm.set_bm_stab_factor(percent as f64 / 100.0);
            }

            // ── Score Swing (~28 Elo) ──
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
            let score_factor = if depth >= score_drop_depth() {
                let scale = score_swing_scale() as f64;
                let diff = ((self.prev_score - new_score) as f64).clamp(-scale, scale);
                2.0_f64.powf(diff / scale)
            } else {
                1.0
            };

            self.tm.set_score_factor(score_factor);

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
            let mut best = self.prev_pv.get(0).unwrap_or(self.root_moves[0].mv);

            // Guard: if the PV move is somehow illegal for the root position,
            // fall back to root_moves[0], which was generated legally.
            if !is_pseudo_legal(&self.root_pos, best) {
                best = self.root_moves[0].mv;
            }

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

    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        self.root_moves.first().map(|rm| rm.mv)
    }

    #[inline]
    pub fn best_score(&self) -> Option<i32> {
        self.root_moves.first().map(|rm| rm.score)
    }

    /// Periodic Heartbeat:
    /// Check stop flag, hard time limit, node limit.
    /// Also drives realtime TUI updates, piggybacking on the same interval.
    #[inline]
    fn check_signals(&mut self) -> bool {
        // Publish the local node count into this thread's slot so the
        // display and node-limit paths see the aggregate. Relaxed store
        // (plain mov on x86); this slot has exactly one writer.
        self.cfg.node_slots[self.cfg.thread_id].store(self.nodes, Ordering::Relaxed);

        if self.cfg.stop.load(Ordering::Acquire)
            || self.tm.is_hard_limit_reached()
            || (self.cfg.limits.nodes > 0 && self.node_count() >= self.cfg.limits.nodes)
        {
            self.cfg.stop.store(true, Ordering::Release);
            return true;
        }

        if self.cfg.display.go_pretty && self.nodes.is_multiple_of(NODE_CHECK_INTERVAL) {
            let now = self.tm.elapsed().as_millis();

            if now - self.last_print > PRINT_UPDATE_MS {
                self.last_print = now;
                self.print_realtime();
            }
        }
        false
    }

    fn is_stopped(&self) -> bool {
        self.cfg.stop.load(Ordering::Acquire)
    }

    /// Sums per-thread node counters. Each thread publishes its local count
    /// into its own slot every 2048 nodes, so the aggregate may trail by
    /// up to `NODE_CHECK_INTERVAL · threads`, invisible at display intervals.
    #[inline(always)]
    fn node_count(&self) -> u64 {
        self.cfg.node_slots.iter().map(|s| s.load(Ordering::Relaxed)).sum()
    }

    /// Assemble the display snapshot shared by depth-complete and realtime reporting.
    /// `pv` and `history` are borrowed by the returned struct,
    /// so the caller owns them for the duration of the print.
    #[cold]
    fn search_info_data<'a>(&'a self, depth: i32, score: i32, pv: &'a Line, history: &'a [PvSnapshot]) -> tui::SearchInfoData<'a> {
        let ms = self.tm.elapsed().as_millis().max(1);
        let total = self.node_count();
        let nps = (u128::from(total) * 1000) / ms;

        tui::SearchInfoData {
            depth,
            score,
            pv,
            sel_depth: self.sel_depth,
            nodes: total,
            nps: u64::try_from(nps).unwrap_or(u64::MAX),
            time_ms: ms,
            hashfull: self.tt.hashfull(),
            show_wdl: self.cfg.display.show_wdl,
            material: self.root_pos.material_count(),
            stm: self.root_pos.stm.as_usize(),
            history,
            board: &self.root_pos,
            use_ansi: self.cfg.display.use_ansi,
        }
    }

    #[cold]
    fn print_info(&self, depth: i32, score: i32, pv: &Line) {
        if self.cfg.limits.silent {
            return;
        }

        let history_vec: Vec<_> = self.pv_history.iter().copied().collect();
        let data = self.search_info_data(depth, score, pv, &history_vec);

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            tui::print_pretty_search_info(&data);
        } else {
            tui::print_search_info(self.cfg.limits.protocol, &data, self.cfg.display.pretty_print);
        }
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

    #[cold]
    fn print_realtime(&mut self) {
        if self.cfg.limits.silent {
            return;
        }
        let history_vec: Vec<_> = self.pv_history.iter().copied().collect();
        let best = &self.root_moves[0];
        let data = self.search_info_data(self.iter_depth, best.score, &best.pv, &history_vec);

        tui::print_pretty_search_info(&data);
    }
}

impl Worker<'_> {
    /// Static evaluation with the pawn structure pulled from the cache.
    #[inline]
    fn evaluate(&mut self) -> i32 {
        let phase = extract_phase(&self.accumulator);
        let pawn = self.pawn_cache.probe(&self.pos);
        let features = SharedFeatures::with_pawn(&self.pos, &pawn);

        evaluate_generic::<i32>(&self.pos, &self.accumulator, phase, &self.eval_params, Some(&features))
    }

    /// Negamax with alpha-beta pruning. Since chess is zero-sum, we maximize the
    /// score from the current side's perspective at every node, negating the score
    /// as it returns up the tree.
    ///
    /// PVS layered on top:
    /// After the presumed best move gets a full-window search,
    /// all others are probed with a zero-width "scout" window (alpha, alpha+1).
    /// Most confirm they're worse. The rare fail-high triggers a re-search,
    /// and that's rare enough to be a net win.
    fn negamax<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        pv_move: Option<Move>,
    ) -> Result<i32, SearchAborted> {
        self.stack[ply].pv.len = 0;

        if (searcher.nodes & (NODE_CHECK_INTERVAL - 1)) == 0 && searcher.check_signals() {
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

        // ── Mate Distance Pruning ──
        // Scores are bounded; no line from here can find mate faster than
        // MATE - ply plies, and we can't be mated before -MATE + ply.
        // Tighten the search window to those limits. If the tightened
        // window collapses (a >= b), every achievable score in this
        // subtree already satisfies the bound; return a, the tightest
        // value provable without searching a single move.
        //
        // Skipped at the root so iterative_deepening always has a scored
        // move list to sort; the cutoff path bypasses root_moves updates.
        let (alpha, beta) = if N::ROOT {
            (alpha, beta)
        } else {
            let a = alpha.max(-MATE + ply as i32);
            let b = beta.min(MATE - ply as i32 - 1);
            if a >= b {
                return Ok(a);
            }
            (a, b)
        };

        let alpha_orig = alpha;

        // ── TT Probe (~128 Elo) ──
        // Have we seen this position before?
        // If a previous search already explored it to sufficient depth,
        // we can reuse its result and skip the entire subtree.
        // This is what makes iterative deepening fast: earlier iterations populate the table for later ones.
        let (tt_move, tt_pv, tt_eval) =
            if let Some((mv, score, depth_stored, bound, pv, eval)) = searcher.tt.probe(self.pos.hash, ply) {
                // Hash collisions can inject moves from unrelated positions.
                // Full pseudo-legality check rejects garbage before it reaches
                // the move picker or triggers a cutoff with a bogus score.
                let valid = mv.is_null() || is_pseudo_legal(&self.pos, mv);

                #[rustfmt::skip]
                if valid
                    && !N::PV
                    && depth_stored >= depth
                    && tt::can_cutoff(bound, score, alpha, beta)
                {
                    return Ok(score);
                }

                let tt_move = if valid && !mv.is_null() { Some(mv) } else { None };
                (tt_move, pv, Some(eval))
            } else {
                (None, false, None)
            };

        // ── TT Move Ordering (~56 Elo) ──
        // Even when the TT score didn't produce a cutoff, the move it stored
        // is still our best guess at what's good here. Searching it first makes
        // beta cutoffs happen earlier, which lets alpha-beta prune far more of the tree.
        let pv_move = pv_move.filter(|&mv| is_pseudo_legal(&self.pos, mv));
        let hash_move = tt_move.or(pv_move);

        let checkers = self.pos.checkers();
        let in_check = checkers.is_not_empty();

        // ── Check Extension (~11 Elo) ──
        // Being in check is forcing, so don't let the horizon cut us off
        // mid-tactic. Extend by one ply so the reply is always searched.
        let depth = if in_check { depth + 1 } else { depth };

        // ── Static Eval ──
        // Our best guess at how good this position is without searching deeper.
        // Used as a baseline for pruning decisions: if the position looks
        // overwhelmingly good or hopeless, we can take shortcuts.
        // Meaningless when in check (we're forced to respond, not evaluate).
        // A TT hit already carries this position's raw eval, so reuse it and skip
        // the full evaluation; the stored sentinel (an in-check store) falls through.
        let raw_static_eval = if in_check {
            tt::SCORE_NONE
        } else if let Some(eval) = tt_eval.filter(|&e| e != tt::SCORE_NONE) {
            eval
        } else {
            self.evaluate()
        };

        // ── Correction History ──
        // The evaluator has systematic biases tied to structural features
        // it can't see directly. Correction tables observe (search - eval)
        // deltas keyed by such features: pawn structure, non-pawn material,
        // and whatever else we add. They then nudge future evals of positions
        // sharing those keys toward the truth.
        // raw_static_eval stays untouched so the update at the end of this
        // frame learns from the unshifted baseline, not the already-corrected
        // value we use for pruning.
        let pawn_hash = if in_check { 0 } else { self.pos.pawn_key };
        let minor_hash = if in_check { 0 } else { self.pos.minor_key };
        let major_hash = if in_check { 0 } else { self.pos.major_key };
        let static_eval = if in_check {
            tt::SCORE_NONE
        } else {
            let correction =
                self.history
                    .correction(self.pos.stm, pawn_hash, minor_hash, major_hash, minor_corr_weight(), major_corr_weight())
                    / history::CORRECTION_SCALE;

            (raw_static_eval + correction).clamp(-MATE_BOUND, MATE_BOUND)
        };

        self.stack[ply].static_eval = static_eval;

        // ── Improving Flag ──
        // Has our position strengthened since our last turn?
        // Step further back if the closer reference was in check:
        // no static_eval was stored there.
        let _improving = if !in_check && ply >= 2 && self.stack[ply - 2].static_eval != tt::SCORE_NONE {
            static_eval > self.stack[ply - 2].static_eval
        } else if !in_check && ply >= 4 && self.stack[ply - 4].static_eval != tt::SCORE_NONE {
            static_eval > self.stack[ply - 4].static_eval
        } else {
            false
        };

        // ── Reverse Futility Pruning (~52 Elo) ──
        // Position is already so good that even after subtracting a generous
        // margin, we're still above beta. The opponent wouldn't have let us
        // get here, so just return the eval and move on.
        #[rustfmt::skip]
        if !in_check
            && !N::PV
            && depth <= rfp_depth()
            && static_eval - rfp_margin() * depth >= beta
        {
            return Ok(static_eval);
        }

        // ── Razoring (~17 Elo) ──
        // Position is so far below alpha that a full-depth search is unlikely
        // to recover. Drop straight into qsearch to confirm.
        #[rustfmt::skip]
        if !in_check
            && !N::PV
            && depth <= razoring_depth()
            && static_eval + razoring_margin() * depth < alpha
        {
            let score = self.qsearch::<N>(searcher, alpha, beta, ply, None, 0)?;

            if score <= alpha {
                return Ok(score);
            }
        }

        // ── Null Move Pruning (~85 Elo) ──
        // If our position is so good that we can pass the turn (do nothing)
        // and still beat beta after a reduced search, the opponent would
        // never allow this line. Skip it. The "null move" is the pass.
        if !in_check
            && !N::PV
            && !self.stack[ply].is_null
            && !self.is_nmp_verif
            && static_eval >= beta
            && self.pos.has_non_pawn_material(self.pos.stm)
        {
            let eval_r = ((static_eval - beta) / nmp_eval_divisor()).min(nmp_eval_max());
            let r = nmp_base_r() + depth / nmp_depth_divisor() + eval_r;

            self.stack[ply].moved_pt = PieceType::None;
            self.stack[ply + 1].is_null = true;

            let undo = self.pos.make_null_move();
            searcher.zobrist_trail.push(self.pos.hash);

            let score =
                match self.negamax::<NonPvNode>(searcher, (depth - r - nmp_ply_offset()).max(0), -beta, -beta + 1, ply + 1, None) {
                    Ok(v) => -v,
                    Err(e) => {
                        searcher.zobrist_trail.pop();
                        self.pos.unmake_null_move(&undo);
                        self.stack[ply + 1].is_null = false;
                        return Err(e);
                    },
                };

            searcher.zobrist_trail.pop();
            self.pos.unmake_null_move(&undo);
            self.stack[ply + 1].is_null = false;

            if score >= beta {
                let null_score = if score > MATE_BOUND { beta } else { score };

                // ── Verification Search ──
                // At or below nmp_verif_min_depth, trust the cutoff outright.
                // Cheap nodes are almost never zugzwangs. Above the threshold,
                // re-search the same position without a null move at reduced depth.
                // If that also fails high, the cutoff is real; if it fails low we
                // were about to prune a zugzwang or a tactic the null search
                // couldn't see, so fall through to the regular move loop.
                if depth <= nmp_verif_min_depth() {
                    return Ok(null_score);
                }

                self.is_nmp_verif = true;
                let verif = self.negamax::<NonPvNode>(searcher, (depth - r - nmp_ply_offset()).max(0), beta - 1, beta, ply, None);
                self.is_nmp_verif = false;

                if verif? >= beta {
                    return Ok(null_score);
                }
            }
        }

        // ── Internal Iterative Reduction (~14 Elo) ──
        // No TT move means we're searching blind: our first guesses are just
        // that, guesses. Reduce by one ply to acknowledge the uncertainty
        // and avoid investing full depth into an unguided search.
        // The next iteration will have a TT move and do it properly.
        let depth = if depth >= iir_depth() && tt_move.is_none() { depth - iir_reduction() } else { depth };

        // ──────── Move loop ────────

        let mut res = MoveResult { move_count: 0, best_eval: -INF, alpha, best_move: Move::null() };

        if N::ROOT {
            // Iterate the pre-sorted root move list.
            for i in 0..searcher.root_moves.len() {
                let mv = searcher.root_moves[i].mv;

                // ── Root LMR (~21 Elo) ──
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
                self.search_move::<N>(searcher, mv, depth, &mut res, beta, ply, Some(i), Some(mv) == pv_move, reduction)?;
                searcher.root_moves[i].nodes += searcher.nodes - nodes_before;

                if res.alpha >= beta {
                    break;
                }
            }
        } else {
            let stm = self.pos.stm;
            let opp = stm.opposite();
            let threats = self.pos.threats(opp);
            let ksq = self.pos.pieces(PieceType::King, stm).lsb();
            let pinned = self.pos.king_blockers();

            // Track searched quiets and captures to penalize them if a later move causes a cutoff.
            self.stack[ply].quiet_count = 0;
            self.stack[ply].capture_count = 0;

            // Continuation history contexts: the piece-to-square pair of
            // the move played 1, 2, and 4 plies ago from this node.
            // These index into separate cont-hist tables and let the move
            // picker incorporate recent positional context into quiet ordering.
            let cont1 = if ply > 0 {
                ContContext { pt: self.stack[ply - 1].moved_pt, to: self.stack[ply - 1].moved_to }
            } else {
                ContContext::default()
            };

            let cont2 = if ply > 1 {
                ContContext { pt: self.stack[ply - 2].moved_pt, to: self.stack[ply - 2].moved_to }
            } else {
                ContContext::default()
            };

            let cont4 = if ply > 3 {
                ContContext { pt: self.stack[ply - 4].moved_pt, to: self.stack[ply - 4].moved_to }
            } else {
                ContContext::default()
            };

            // Interior: staged move generation via MovePicker.
            let mut picker = MovePicker::new(hash_move, searcher.cfg, self.stack[ply].killers, threats, cont1, cont2, cont4);
            while let Some(mv) = picker.next(&self.pos, self.history) {
                if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
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

                // ── Futility Pruning (~10 Elo) ──
                // At shallow depth, if static eval is already so far below alpha
                // that a quiet move is unlikely to raise it, skip the move.
                if !in_check
                    && mv.is_quiet()
                    && !N::PV
                    && res.move_count >= 1
                    && depth <= fp_depth()
                    && static_eval + fp_margin() * depth <= res.alpha
                {
                    continue;
                }

                // ── Late Move Pruning (~14 Elo) ──
                // At shallow depth, quiet moves beyond a fixed count threshold
                // are unlikely to be the best move, so skip them entirely.
                if !in_check
                    && mv.is_quiet()
                    && !N::PV
                    && depth <= lmp_depth()
                    && res.move_count as i32 >= lmp_base() + depth * depth
                {
                    continue;
                }

                // ── History Pruning (~7 Elo) ──
                // A quiet with deeply negative history has been punished
                // repeatedly by the gravity update. Trust that signal at
                // shallow depth and skip the full search.
                //
                // Gate uses !is_tactical() rather than is_quiet() so
                // non-capture promotions are excluded; is_quiet only
                // filters captures, and pruning a queen promotion on
                // global history would be absurd.
                //
                // Killers are exempt; a killer is a per-ply tactical
                // refutation whose global history is often deeply negative
                // (terrible in most positions, saving in this one).
                // The picker's ordering already says "this move works here",
                // pruning it on global stats would nuke the exact move
                // MovePicker promoted.
                if !in_check
                    && !N::PV
                    && !mv.is_tactical()
                    && res.move_count >= 1
                    && mv != self.stack[ply].killers[0]
                    && mv != self.stack[ply].killers[1]
                    && depth <= hist_prune_depth()
                {
                    let pt = self.pos.expect_piece_at(mv.from());
                    let hist = self
                        .history
                        .score_quiet(self.pos.stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4);

                    if hist < -hist_prune_margin() * depth {
                        continue;
                    }
                }

                // ── SEE Pruning (~20 Elo) ──
                // Skip moves whose destination-square exchange clearly
                // loses material.
                //
                // Captures scale linearly: SEE is an accurate verdict on
                // a capture (the value is realized right there at the
                // destination square) so the tolerance grows modestly
                // with depth — we just give deeper searches some slack
                // in case the tree refutes an apparent loss.
                //
                // Quiets scale quadratically: for a quiet move, SEE is a
                // crude proxy — the move's real value usually lives
                // elsewhere in the tree (threats, structure, follow-ups
                // several plies out). Deeper searches will find it
                // themselves, so we loosen aggressively with depth and
                // only prune "this is obviously moving into a trap" cases
                // at shallow depth.
                if !in_check && !N::PV && res.move_count >= 1 {
                    let margin = if mv.is_capture() { -see_capture_margin() * depth } else { -see_quiet_margin() * depth * depth };

                    if !see_ge(&self.pos, mv, margin) {
                        continue;
                    }
                }

                // ── Late Move Reductions (~90 Elo) ──
                // Moves late in the list are unlikely to beat alpha.
                // Search them at reduced depth; re-search fully on surprise.
                let reduction = if depth >= 2 && res.move_count >= 1 && mv.is_quiet() && !in_check {
                    let mut r = searcher.cfg.lmr(depth, res.move_count + 1);
                    let pt = self.pos.expect_piece_at(mv.from());
                    let hist = self
                        .history
                        .score_quiet(self.pos.stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4);

                    if mv == self.stack[ply].killers[0] || mv == self.stack[ply].killers[1] {
                        r -= killer_lmr_bonus();
                    }

                    // A quiet that newly attacks a bigger piece is tactically live,
                    // like a check; reduce it less so the threat gets a real search.
                    r -= self.pos.new_threats(pt, mv.from(), mv.to()).popcount() as i32 * threat_lmr_bonus();

                    let max_r = (depth - lmr_retained()) * LMR_SCALE;

                    (r - hist / lmr_hist_div()).clamp(0, max_r) / LMR_SCALE
                } else {
                    0
                };

                // Context for child node's cont-hist lookup
                self.stack[ply].moved_pt = self.pos.expect_piece_at(mv.from());
                self.stack[ply].moved_to = mv.to();

                self.search_move::<N>(searcher, mv, depth, &mut res, beta, ply, None, Some(mv) == pv_move, reduction)?;

                if likely(res.alpha >= beta) {
                    #[cfg(feature = "mvpstats")]
                    {
                        use crate::engine::mvpstats::{CutoffKind, record_cutoff};
                        let kind = if Some(mv) == hash_move {
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

                    // ── History Gravity Heuristic (~95 Elo) ──
                    // When a move causes a beta-cutoff, it's presumably a strong response.
                    // We reward it so it surfaces earlier in future sibling nodes,
                    // and we punish the preceding moves that failed to refute the branch.
                    //
                    // Bonus is quadratic with depth to prioritize deep heuristics,
                    // scaled 4x and capped at 1600 to push entries toward the
                    // nominal ±16384 attractor without a single deep search
                    // permanently dominating the history table.
                    let bonus = (depth.pow(2) * hist_bonus_mult()).min(hist_bonus_cap());

                    // Only reward if the cutoff itself was caused by a quiet move.
                    // Captures and structural moves (castling) are handled differently.
                    if mv.is_history_quiet() {
                        let pt = self.pos.expect_piece_at(mv.from());

                        self.history.update(stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4, bonus);

                        // ── Killer Moves (~35 Elo) ──
                        // Maintain a 2-slot pseudo-Least-Recently-Used cache for tracking quiet cutoffs.
                        // If the move isn't already the primary killer, shift the old primary to slot 1
                        // and promote the new move to slot 0. If it was slot 1, this natively swaps them.
                        if mv != self.stack[ply].killers[0] {
                            self.stack[ply].killers[1] = self.stack[ply].killers[0];
                            self.stack[ply].killers[0] = mv;
                        }
                    } else if mv.is_capture() && !mv.is_promotion() {
                        // ── Capture History Update ──
                        // Promotion-captures are deliberately excluded: they bypass
                        // the normal MVV-LVA + capture-history blend in the picker
                        // (see add_promo_caps), so updating their entries here would
                        // train a table that nothing reads.
                        //
                        // self.pos is the parent position: search_move has already
                        // unmade the move, so the captured piece is back on to
                        // (or it's en passant, where the victim is a pawn by definition).
                        let attacker = self.pos.expect_piece_at(mv.from());
                        let victim = if mv.is_en_passant() { PieceType::Pawn } else { self.pos.piece_at(mv.to()) };
                        self.history.update_capture(stm, attacker, mv.to(), victim, bonus);
                    }

                    // ── Asymmetric Penalty (~25 Elo) ──
                    // When a move causes a beta-cutoff, all moves searched before it
                    // at this ply are "losers": they failed to refute the branch.
                    // We drive their history scores down so they surface later in
                    // future sibling nodes.
                    //
                    // Quiets: penalize all preceding quiets. If the cutoff was a
                    // capture, quiet_count is 0 (captures precede quiets in the
                    // picker), so the loop is a no-op.
                    let penalty_limit =
                        if appended_quiet { self.stack[ply].quiet_count.saturating_sub(1) } else { self.stack[ply].quiet_count };

                    for i in 0..penalty_limit {
                        let qm = self.stack[ply].quiet_moves[i];
                        let q_pt = self.pos.expect_piece_at(qm.from());

                        // Over time, this "anti-history" pushes bad moves deeper into the list.
                        self.history.update(stm, q_pt, qm.from(), qm.to(), threats, cont1, cont2, cont4, -bonus);
                    }

                    // Captures: penalize all preceding captures that were searched
                    // and failed to cut. Without this, capture history only drifts
                    // positive: it can reward good captures but never push bad
                    // ones down.
                    let cap_penalty_limit = if appended_capture {
                        self.stack[ply].capture_count.saturating_sub(1)
                    } else {
                        self.stack[ply].capture_count
                    };

                    for i in 0..cap_penalty_limit {
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
        if unlikely(res.move_count == 0) {
            return if in_check { Ok(-MATE + ply as i32) } else { Ok(0) };
        }

        let bound = if res.best_eval >= beta {
            tt::BOUND_LOWER
        } else if res.best_eval > alpha_orig {
            tt::BOUND_EXACT
        } else {
            tt::BOUND_UPPER
        };

        // ── TT store ──
        searcher
            .tt
            .store(self.pos.hash, ply, depth, res.best_eval, res.best_move, bound, N::PV || tt_pv, raw_static_eval);

        // ── Correction History Update ──
        // Only learn from positions resolved by quiet moves: tactical
        // resolutions (captures/promotions) reflect tactics, not evaluator bias.
        // Skip when the bound direction contradicts the diff: a fail-high with
        // best_eval <= static_eval, or a fail-low with best_eval >= static_eval,
        // carries no useful structural signal.
        if !in_check
            && !res.best_move.is_null()
            && !res.best_move.is_tactical()
            && res.best_eval.abs() < MATE_BOUND
            && !((bound == tt::BOUND_LOWER && res.best_eval <= static_eval)
                || (bound == tt::BOUND_UPPER && res.best_eval >= static_eval))
        {
            let diff = res.best_eval - raw_static_eval;

            self.history
                .update_correction(self.pos.stm, pawn_hash, minor_hash, major_hash, diff, depth);
        }

        Ok(res.best_eval)
    }

    /// Make a move, search it, unmake it. The "heartbeat" of alpha-beta.
    ///
    /// # Safety
    /// The move `mv` MUST be legal. Root legality is guaranteed by the root
    /// move list generated at search start; interior legality is verified by
    /// the explicit `is_legal` call in the move loop before each invocation.
    fn search_move<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        mv: Move,
        depth: i32,
        res: &mut MoveResult,
        beta: i32,
        ply: usize,
        root_idx: Option<usize>,
        is_pv_move: bool,
        mut reduction: i32,
    ) -> Result<(), SearchAborted> {
        let saved_acc = self.accumulator;
        let undo = self.pos.make_move(mv, &mut self.accumulator);

        searcher.tt.prefetch(self.pos.hash);

        // ── Gives-Check LMR Adjustment (~4 Elo) ──
        // A move that delivers check is forcing: the opponent has no choice
        // but to respond. Don't reduce it as aggressively; give it a bit more
        // depth so the resulting tactics are properly resolved.
        if self.pos.checkers().is_not_empty() {
            reduction = (reduction - check_lmr_bonus()).max(0);
        }

        res.move_count += 1;
        searcher.zobrist_trail.push(self.pos.hash);

        if N::ROOT {
            searcher.print_currmove(depth, mv, res.move_count);
        }

        let eval = match self.pvs::<N>(searcher, depth, res.alpha, beta, ply, res.move_count == 1, is_pv_move, reduction, mv) {
            Ok(v) => v,
            Err(e) => {
                searcher.zobrist_trail.pop();
                self.pos.unmake_move(mv, &undo);
                self.accumulator = saved_acc;
                return Err(e);
            },
        };

        searcher.zobrist_trail.pop();
        self.pos.unmake_move(mv, &undo);
        self.accumulator = saved_acc;

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
                let child_pv = self.stack[ply + 1].pv; // Root copy is fine, happens rarely
                searcher.root_moves[i].pv.compose(mv, &child_pv);
            }

            if eval > res.alpha {
                res.alpha = eval;
                // Borrow checker requires a disjoint split; we need a mutable
                // reference to stack[ply].pv and a shared one to stack[ply+1].pv
                // simultaneously. Rust can't prove ply and ply+1 don't alias
                // through two separate borrows of self.stack, so split_at_mut
                // gives us two non-overlapping slices as proof.
                let (current_stack, next_stack) = self.stack.split_at_mut(ply + 1);
                let child_pv = &next_stack[0].pv;
                let child_len = child_pv.len.min(MAX_PLY - 1);
                let current_pv = &mut current_stack[ply].pv;

                current_pv.moves[0] = mv;
                current_pv.moves[1..=child_len].copy_from_slice(&child_pv.moves[..child_len]);
                current_pv.len = child_len + 1;
            }
        }

        Ok(())
    }

    // ── Principal Variation Search (~14 Elo) ──
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
        is_pv_move: bool,
        reduction: i32,
        mv: Move,
    ) -> Result<i32, SearchAborted> {
        // Retrieve the expected PV move for the next ply (ply + 1 is the child node's level).
        // If we are on the PV line and we just played the PV move, we expect the child to also have a PV move.
        let next_pv = if (N::ROOT || N::PV) && is_pv_move { searcher.prev_pv.get(ply + 1) } else { None };

        if is_first {
            // No bound yet; search wide open.
            return Ok(-self.negamax::<N::Next>(searcher, depth - 1, -beta, -alpha, ply + 1, next_pv)?);
        }

        // ── LMR Scout ──
        // Late quiet moves get a shallower scout. If the reduced search
        // still beats alpha, the move earned a full-depth re-search.
        let reduced_depth = depth - 1 - reduction;
        let mut score = -self.negamax::<NonPvNode>(searcher, reduced_depth, -alpha - 1, -alpha, ply + 1, None)?;

        // Re-search at full depth if the reduced scout found something.
        if score > alpha && reduction > 0 {
            score = -self.negamax::<NonPvNode>(searcher, depth - 1, -alpha - 1, -alpha, ply + 1, None)?;

            // ── Post-LMR Continuation History (~8 Elo) ──
            // The reduced scout beat alpha; the full-depth re-search settles it.
            // A fail-low means the reduction over-promised; a fail-high means a cutoff.
            // Punish or reward continuation history accordingly, ordering only.
            if mv.is_history_quiet() && (score <= alpha || score >= beta) {
                let stm = self.pos.stm.opposite();
                let pt = self.stack[ply].moved_pt;
                let to = self.stack[ply].moved_to;

                let cont1 = if ply > 0 {
                    ContContext { pt: self.stack[ply - 1].moved_pt, to: self.stack[ply - 1].moved_to }
                } else {
                    ContContext::default()
                };

                let cont2 = if ply > 1 {
                    ContContext { pt: self.stack[ply - 2].moved_pt, to: self.stack[ply - 2].moved_to }
                } else {
                    ContContext::default()
                };

                let cont4 = if ply > 3 {
                    ContContext { pt: self.stack[ply - 4].moved_pt, to: self.stack[ply - 4].moved_to }
                } else {
                    ContContext::default()
                };

                let bonus = (depth.pow(2) * hist_bonus_mult()).min(hist_bonus_cap());
                let signed = if score <= alpha { -bonus } else { bonus };

                self.history.update_conthist(stm, pt, to, cont1, cont2, cont4, signed);
            }
        }

        if score > alpha && score < beta {
            // Genuine improvement; search with full window on the PV.
            score = -self.negamax::<N::Next>(searcher, depth - 1, -beta, -alpha, ply + 1, next_pv)?;
        }

        Ok(score)
    }

    // ── Quiescence Search (~655 Elo) ──
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

        if (searcher.nodes & (NODE_CHECK_INTERVAL - 1)) == 0 && searcher.check_signals() {
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

        // ── QSearch TT Probe (~22 Elo) ──
        // Read TT entries stored by negamax or prior qsearch visits.
        // Cutoffs gated on non-PV nodes: PV nodes need the full capture
        // sequence for accurate PV reporting.
        //
        // Quiescence TT Move (~9 Elo)
        let (qs_tt_move, qs_tt_pv, qs_tt_eval) =
            if let Some((mv, score, _depth, bound, pv, eval)) = searcher.tt.probe(self.pos.hash, ply) {
                if !N::PV && tt::can_cutoff(bound, score, alpha, beta) {
                    return Ok(score);
                }

                let mv = if !mv.is_null() && is_pseudo_legal(&self.pos, mv) { Some(mv) } else { None };
                (mv, pv, Some(eval))
            } else {
                (None, false, None)
            };

        let checkers = self.pos.checkers();
        let in_check = checkers.is_not_empty();
        let stm = self.pos.stm;
        let opp = stm.opposite();

        // ── QSearch Evaluations & Evasions ──
        // If we are in check, the position is forced and static evaluation is meaningless.
        // We drop the stand-pat evaluation (-INF) and the MovePicker will generate
        // all legal evasions instead of just captures/queen promotions.
        // A TT hit already carries this position's raw eval; reuse it over a fresh
        // evaluation. SCORE_NONE when in check, which is also the value stored below.
        let raw_eval = if in_check {
            tt::SCORE_NONE
        } else if let Some(eval) = qs_tt_eval.filter(|&e| e != tt::SCORE_NONE) {
            eval
        } else {
            self.evaluate()
        };

        let mut best_eval = if in_check {
            -INF
        } else {
            let pawn_hash = self.pos.pawn_key;
            let minor_hash = self.pos.minor_key;
            let major_hash = self.pos.major_key;
            let correction =
                self.history
                    .correction(self.pos.stm, pawn_hash, minor_hash, major_hash, minor_corr_weight(), major_corr_weight())
                    / history::CORRECTION_SCALE;

            let eval = (raw_eval + correction).clamp(-MATE_BOUND, MATE_BOUND);

            if eval >= beta {
                // Static eval is blind to quiet replies;
                // the stand-pat verdict overstates certainty.
                // Blend toward beta to deflate the overconfidence.
                return Ok((eval + beta) / 2);
            }

            // ── Delta Pruning (~20 Elo) ──
            // Stand-pat already failed to beat alpha.
            // Even if we capture the most valuable piece on the board, can we reach alpha?
            // If not, no capture in this position can raise us high enough, so bail early.
            //
            // best_capturable is just the highest MVV-LVA value among opponent pieces
            // still on the board; delta_margin covers promotion/positional upside.
            let best_capturable = [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight, PieceType::Pawn]
                .into_iter()
                .find(|&pt| self.pos.pieces(pt, opp).is_not_empty())
                .map_or(0, |pt| searcher.cfg.mvvlva_v[pt as usize]);

            if eval + best_capturable + delta_margin() < alpha {
                return Ok(alpha);
            }

            alpha = alpha.max(eval);
            eval
        };

        let mut moves_made = 0;
        let mut best_move = Move::null();

        let ksq = self.pos.pieces(PieceType::King, stm).lsb();
        let pinned = self.pos.king_blockers();

        let mut picker = MovePicker::new_qsearch(qs_tt_move, searcher.cfg, in_check);

        let recapture_only = !in_check && qs_ply >= qs_recapture_ply();

        while let Some(mv) = picker.next(&self.pos, self.history) {
            if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            // ── Recapture-only Deep QS (~1 Elo) ──
            // Past qs_recapture_ply, captures only matter if they continue
            // the forcing exchange on the square the opponent just moved to.
            // Speculative off-square captures are what explodes in mutual-
            // attack soup (e.g. 30-bishop pathologicals). Tactics are
            // preserved because real combinations are recapture chains.
            if recapture_only
                && let Some(prev) = prev_to
                && mv.to() != prev
            {
                continue;
            }

            // ── QSearch SEE Pruning (~65 Elo) ──
            // Skip captures whose destination-square trade loses material
            // for us. Disabled in check because evasions are forced and
            // the only legal reply is often a losing defensive capture.
            if !in_check && !see_ge(&self.pos, mv, 0) {
                continue;
            }

            let saved_acc = self.accumulator;
            let undo = self.pos.make_move(mv, &mut self.accumulator);

            moves_made += 1;
            searcher.zobrist_trail.push(self.pos.hash);

            let score = match self.qsearch::<N>(searcher, -beta, -alpha, ply + 1, Some(mv.to()), qs_ply + 1) {
                Ok(v) => -v,
                Err(e) => {
                    searcher.zobrist_trail.pop();
                    self.pos.unmake_move(mv, &undo);
                    self.accumulator = saved_acc;
                    return Err(e);
                },
            };

            searcher.zobrist_trail.pop();
            self.pos.unmake_move(mv, &undo);
            self.accumulator = saved_acc;

            if score > best_eval {
                best_eval = score;
                best_move = mv;

                if score > alpha {
                    if score >= beta {
                        break;
                    }
                    alpha = score;
                }
            }
        }

        if in_check && moves_made == 0 {
            return Ok(-MATE + ply as i32);
        }

        // ── QSearch TT Store ──
        let bound = if best_eval >= beta {
            tt::BOUND_LOWER
        } else if best_eval > alpha_orig {
            tt::BOUND_EXACT
        } else {
            tt::BOUND_UPPER
        };

        searcher
            .tt
            .store_qs(self.pos.hash, ply, best_eval, best_move, bound, N::PV || qs_tt_pv, raw_eval);

        Ok(best_eval)
    }

    /// Fifty move rule, insufficient material, or repetition → immediate draw.
    ///
    /// Checked before move generation, so a theoretical checkmate on
    /// exactly the 100th half-move is scored as a draw. This is the
    /// standard compromise; every top engine accepts this edge case.
    #[inline]
    pub fn is_draw(&self, ply: usize, history: &[u64]) -> bool {
        self.pos.is_fifty_move_draw() || self.pos.is_draw_by_material() || (ply > 0 && self.is_repetition(self.pos.hash, history))
    }

    /// Scans the history for a previous occurrence of the current hash.
    ///
    /// Any second occurrence is treated as a draw.
    /// While technically 3-fold is the rule,
    /// engines score the second to avoid searching infinite
    /// cycles and to prevent the GUI from flagging PVs that repeat.
    ///
    /// This uses a raw scan without side-to-move skipping because
    /// Zobrist already includes the side-to-move key. Contrast with
    /// `Position::is_threefold_repetition` which is optimized for
    /// adjudication contexts.
    #[inline]
    fn is_repetition(&self, key: u64, history: &[u64]) -> bool {
        if history.len() < 2 {
            return false;
        }

        let window = self.pos.halfmove_clock as usize;
        let start = history.len().saturating_sub(window + 1);

        // We exclude the current position using saturating_sub(2).
        // This works because the caller (search_move) has already pushed the current
        // hash into the zobrist trail before descending, placing it at len - 1.
        let end = history.len().saturating_sub(2);

        if start > end {
            return false;
        }

        let needle = Vu64x4::splat(key);
        let slice = &history[start..=end];
        let chunks = slice.rchunks_exact(4);
        let remainder = chunks.remainder();

        // We vector-scan the full slice, including opponent-ply hashes.
        // Zobrist already encodes side-to-move, so cross-ply hashes can never match.
        // The STM bit differs for every other entry, making false positives impossible.
        // The false-check cost is zero, and contiguous SIMD loads are cheaper than strided.
        //
        // Vu64x4::load uses _mm256_loadu_si256 internally (unaligned), which is correct
        // because Vec<u64> guarantees only 8-byte alignment, not the 32-byte alignment
        // that an aligned load would require.
        for chunk in chunks {
            let vec = unsafe { Vu64x4::load(chunk.as_ptr()) };

            if vec.cmp_eq(needle).any() {
                return true;
            }
        }
        remainder.contains(&key)
    }
}
