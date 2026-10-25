//! Negamax alpha-beta search.

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
use crate::core::board::xorboard::Snapshot as XbSnapshot;
#[cfg(not(feature = "nocorr"))]
use crate::engine::history::CORRECTION_SCALE;
use crate::{
    core::{
        board::{Position, attacks::Pins, xorboard::XorBoard},
        defs::{
            Bitboard, Color, INF, LANE_PHASE, MATE_BOUND, MAX_DEPTH, MAX_PLY, PieceType, Protocol, Square, draw_score, is_loss,
            is_mate, is_win, mate_in, mated_in,
        },
        moves::Move,
    },
    engine::{
        eval::{EvalParams, PawnCache, SharedFeatures, evaluate_generic, evaluate_psqt, extract_phase, lazy_eval_margin},
        history::{ContContext, History, HistoryParams},
        movegen::{gen_legal_moves, is_legal},
        movepicker::MovePicker,
        search_params::*,
        see::{self, see_ge},
        tm::{Iteration, TimeManager},
        tt,
        tt::{TranspositionTable, TtData, TtMove},
        tui,
    },
    weave::{I16x8, U64x4},
};

const NODE_CHECK_INTERVAL: u64 = 2048;
const CURRMOVE_NODE_THRESHOLD: u64 = 100_000_000;
const PRINT_UPDATE_MS: u128 = 25;

const MAX_TRACKED_QUIETS: usize = 256;
const MAX_TRACKED_CAPTURES: usize = 64;

const LMR_SCALE: i32 = 1024;

pub(crate) struct Searcher<'cfg> {
    cfg: &'cfg SearchConfig,
    tm: TimeManager,
    root_moves: Vec<RootMove>,
    zobrist_trail: Vec<u64>,
    root_pos: Position,
    prev_pv: PvLine,
    prev_score: i32,
    pub(crate) nodes: u64,
    sel_depth: i32,
    iter_depth: i32,
    last_print: u128,
    pv_history: VecDeque<PvSnapshot>,
    tt: Arc<TranspositionTable>,
}

#[repr(align(32))]
struct Worker<'h> {
    pos: Position,
    accumulator: I16x8,
    stack: Box<[Stack; MAX_PLY + 2]>,
    #[cfg(not(feature = "nostore"))]
    xorboard: XorBoard,
    #[cfg(not(feature = "nostore"))]
    xb_snapshots: Box<[XbSnapshot; MAX_PLY + 2]>,
    history: &'h mut History,
    /// Keyed by `pawn_key`.
    pawn_cache: PawnCache,
    eval_params: EvalParams<i32>,
    /// Prevents nested null-move verification.
    is_nmp_verif: bool,
}

#[derive(Clone, Default, Debug)]
pub(crate) struct Limits {
    pub(crate) wtime: u64,
    pub(crate) btime: u64,
    pub(crate) winc: u64,
    pub(crate) binc: u64,
    pub(crate) movestogo: u64,
    pub(crate) movetime: u64,
    pub(crate) depth: i32,
    pub(crate) nodes: u64,
    pub(crate) softnodes: u64,
    pub(crate) infinite: bool,
    pub(crate) silent: bool,
    pub(crate) protocol: Protocol,
    pub(crate) mate: Option<i32>,
    pub(crate) perft: Option<u8>,
    pub(crate) perft_bulk: bool,
    pub(crate) searchmoves: Vec<Move>,
}

#[derive(Clone, Copy, Default)]
pub(crate) struct SearchDisplay {
    pub(crate) show_wdl: bool,
    pub(crate) go_pretty: bool,
    pub(crate) pretty_print: bool,
    pub(crate) show_currmove: bool,
    pub(crate) use_ansi: bool,
}

#[derive(Clone)]
pub(crate) struct SearchConfig {
    pub(crate) limits: Limits,
    pub(crate) start_time: Instant,
    pub(crate) stop: Arc<AtomicBool>,
    pub(crate) display: SearchDisplay,
    pub(crate) search_params: SearchParams,
    pub(crate) overhead: u64,
    pub(crate) threads: usize,
    pub(crate) thread_id: usize,
    /// Each thread updates only its own slot.
    node_slots: Arc<[AtomicU64]>,
    pub(crate) mvvlva_victim: [i32; 8],
    pub(crate) mvvlva_attacker: [i32; 8],
    pub(crate) see_values: [i32; 8],
    lmr_table: Box<[i16; MAX_PLY + 1]>,
}

#[derive(Clone, Copy)]
pub(super) struct PvLine {
    pub(super) moves: [Move; MAX_PLY],
    pub(super) len: usize,
}

#[derive(Clone, Copy)]
pub(super) struct PvSnapshot {
    pub(super) depth: i32,
    pub(super) time_ms: u128,
    pub(super) score: i32,
    pub(super) line: PvLine,
}

#[derive(Clone, Copy)]
struct Stack {
    pv: PvLine,
    quiet_moves: [Move; MAX_TRACKED_QUIETS],
    quiet_count: usize,
    capture_moves: [Move; MAX_TRACKED_CAPTURES],
    capture_count: usize,
    killers: [Move; 2],
    moved_pt: PieceType,
    moved_to: Square,
    static_eval: i32,
    is_null: bool,
    /// Cutoffs at this ply since the grandparent was entered, used by the parent's LMR.
    cutoff_count: i32,
    /// TT move excluded during singular verification.
    excluded: Move,
}

trait NodeType {
    const PV: bool;
    const ROOT: bool;
    type Next: NodeType;
}

struct RootNode;
struct PvNode;
struct NonPvNode;

impl NodeType for RootNode {
    const PV: bool = true;
    const ROOT: bool = true;
    type Next = PvNode;
}

impl NodeType for PvNode {
    const PV: bool = true;
    const ROOT: bool = false;
    type Next = PvNode;
}

impl NodeType for NonPvNode {
    const PV: bool = false;
    const ROOT: bool = false;
    type Next = NonPvNode;
}

struct RootMove {
    mv: Move,
    score: i32,
    pv: Box<PvLine>,
    /// Cumulative nodes searched under this move across iterations.
    nodes: u64,
}

struct MoveResult {
    move_count: usize,
    best_eval: i32,
    alpha: i32,
    best_move: Move,
}

/// Time, the node limit or a stop command cut the search short.
#[derive(Debug)]
struct SearchAborted;

impl SearchDisplay {
    pub(crate) const SILENT: Self =
        Self { show_wdl: false, go_pretty: false, pretty_print: false, show_currmove: false, use_ansi: false };
    pub(crate) const DEFAULT: Self = Self { show_currmove: true, use_ansi: true, ..Self::SILENT };
}

impl SearchConfig {
    pub(crate) fn new(
        limits: Limits,
        start_time: Instant,
        stop: Arc<AtomicBool>,
        overhead: u64,
        search_params: SearchParams,
    ) -> Self {
        Self::new_full(limits, start_time, stop, overhead, SearchDisplay::DEFAULT, search_params)
    }

    pub(crate) fn new_full(
        limits: Limits,
        start_time: Instant,
        stop: Arc<AtomicBool>,
        overhead: u64,
        display: SearchDisplay,
        search_params: SearchParams,
    ) -> Self {
        let (mvvlva_victim, mvvlva_attacker) = Self::build_mvvlva(&search_params);
        let see_values = see::see_values(&search_params);
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
            mvvlva_victim,
            mvvlva_attacker,
            see_values,
            lmr_table,
        }
    }

    pub(crate) fn set_threads(&mut self, threads: usize) {
        self.threads = threads;
        self.node_slots = Self::node_slots(threads);
    }

    /// Reduction in `LMR_SCALE` units.
    #[inline(always)]
    fn lmr(&self, depth: i32, move_count: usize) -> i32 {
        let sp = &self.search_params;
        let base = sp.lmr_base * LMR_SCALE / 100;
        let div = sp.lmr_divisor * LMR_SCALE / 100;
        let log_depth = i32::from(self.lmr_table[depth as usize]);
        let log_moves = i32::from(self.lmr_table[move_count]);

        base + log_depth * log_moves / div
    }

    fn node_slots(threads: usize) -> Arc<[AtomicU64]> { (0..threads).map(|_| AtomicU64::new(0)).collect() }

    /// Most Valuable Victim, Least Valuable Attacker.
    fn build_mvvlva(sp: &SearchParams) -> ([i32; 8], [i32; 8]) {
        let mut victim = [0; 8];
        let mut attacker = [0; 8];

        macro_rules! map {
            ($pt:ident, $v:expr, $a:expr) => {
                victim[PieceType::$pt] = $v;
                attacker[PieceType::$pt] = $a;
            };
        }
        map!(Pawn, sp.mvvlva_v_pawn, sp.mvvlva_a_pawn);
        map!(Knight, sp.mvvlva_v_knight, sp.mvvlva_a_knight);
        map!(Bishop, sp.mvvlva_v_bishop, sp.mvvlva_a_bishop);
        map!(Rook, sp.mvvlva_v_rook, sp.mvvlva_a_rook);
        map!(Queen, sp.mvvlva_v_queen, sp.mvvlva_a_queen);
        map!(King, sp.mvvlva_v_king, sp.mvvlva_a_king);
        (victim, attacker)
    }

    /// `R(d, m) = base + ln(d) · ln(m) / divisor`.
    fn build_lmr_table() -> Box<[i16; MAX_PLY + 1]> {
        let scale = LMR_SCALE as f64;
        let mut lut = Box::new([0i16; MAX_PLY + 1]);
        for i in 1..=MAX_PLY {
            lut[i] = ((i as f64).ln() * scale).round() as i16;
        }
        lut
    }
}

impl PvLine {
    pub(super) const fn new() -> Self { Self { moves: [Move::null(); MAX_PLY], len: 0 } }

    fn compose(&mut self, mv: Move, tail: &PvLine) {
        let n = tail.len.min(MAX_PLY - 1);
        self.moves[0] = mv;
        self.moves[1..=n].copy_from_slice(&tail.moves[..n]);
        self.len = 1 + n;
    }

    #[inline(always)]
    pub(super) fn get(&self, idx: usize) -> Option<Move> { if idx < self.len { Some(self.moves[idx]) } else { None } }
}

impl Default for PvLine {
    fn default() -> Self { Self::new() }
}

impl RootMove {
    #[inline]
    fn new(mv: Move) -> Self { Self { mv, score: -INF, pv: Box::new(PvLine::new()), nodes: 0 } }
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            pv: PvLine::new(),
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
    /// The last completed iteration supplies the move if a later one is interrupted.
    #[inline]
    pub(crate) fn iterative_deepening(&mut self, history: &mut History) {
        let sp = &self.cfg.search_params;
        history.params = HistoryParams::from(sp);
        self.nodes = 0;

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            print!("\x1b[2J\x1b[H");
            let _ = io::stdout().flush();
        }

        if let Some(perft_depth) = self.cfg.limits.perft {
            let mut board = self.root_pos;
            let mut acc = board.psqt_accumulator();
            let bulk = self.cfg.limits.perft_bulk;
            let nodes = crate::tools::perft::perft(&mut board, perft_depth, &mut acc, bulk);
            println!("Nodes searched: {nodes}");
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

        let root_acc = self.root_pos.psqt_accumulator();

        // Keep the worker across iterations; rebuilding it would zero 287 KB
        // of ply stack and discard the pawn cache.
        let mut worker = Worker {
            pos: self.root_pos,
            accumulator: root_acc,
            stack: boxed_array(Stack::default()),
            #[cfg(not(feature = "nostore"))]
            xorboard: XorBoard::from_position(&self.root_pos),
            #[cfg(not(feature = "nostore"))]
            xb_snapshots: boxed_array(XbSnapshot::default()),
            history,
            pawn_cache: PawnCache::new(),
            eval_params: EvalParams::<i32>::from_const(),
            is_nmp_verif: false,
        };

        let mut last_iter_elapsed = 0;

        // ── Forced Root Move
        // With one legal move the score cannot change the choice; keep only
        // tm_single_root percent of the soft budget for the next move.
        if self.root_moves.len() == 1 {
            self.tm.scale_base_soft(f64::from(sp.tm_single_root) / 100.0);
        }

        // ── Lazy SMP
        // Each thread searches independently, but they share the TT.
        // Different search paths let them benefit from each other's discoveries.
        for depth in 1..=depth_limit {
            self.iter_depth = depth;
            let elapsed = self.tm.elapsed().as_millis() as u64;
            let prev_depth_time = elapsed.saturating_sub(last_iter_elapsed);
            last_iter_elapsed = elapsed;

            // Fixed movetime is for this move alone; there's no later move to save it for.
            let clocked = self.cfg.limits.movetime == 0;
            if self.cfg.thread_id == 0
                && depth > 1
                && ((clocked && self.tm.should_stop(elapsed, prev_depth_time, sp))
                    || (self.cfg.limits.softnodes > 0 && self.nodes >= self.cfg.limits.softnodes))
            {
                break;
            }

            // ── Proven Mate Bailout
            // We've found a forced mate, so there's no point searching further.
            if self.cfg.thread_id == 0
                && self.tm.is_finite_budget()
                && (self.prev_score >= mate_in(3) || self.prev_score == mated_in(2))
            {
                break;
            }

            if self.check_signals() {
                break;
            }

            // A move that never becomes best keeps its previous PV unless we clear it here.
            for rm in &mut self.root_moves {
                rm.pv.len = 0;
            }

            // ── Aspiration Windows (~42 Elo)
            // Scores usually don't change much between iterations,
            // so a narrow window around the previous score gets us more cutoffs.
            let mut delta = sp.asp_initial;
            let mut alpha = if depth >= sp.asp_depth { (self.prev_score - delta).max(-INF) } else { -INF };
            let mut beta = if depth >= sp.asp_depth { (self.prev_score + delta).min(INF) } else { INF };
            let mut aborted = false;

            // ── Aspiration Fail-High Reduction (~10 Elo)
            // Repeated fail-highs get expensive, so spend less depth on each retry.
            let mut asp_reduction = 0;

            loop {
                worker.pos = self.root_pos;
                worker.accumulator = root_acc;
                #[cfg(not(feature = "nostore"))]
                {
                    worker.xorboard = XorBoard::from_position(&worker.pos);
                }

                let retry_depth = (depth - asp_reduction).max(1);

                // root_moves is still in last iteration's order,
                // so root_moves[0] can miss a fail-high on another move.
                let Ok(score) = worker.negamax::<RootNode>(self, retry_depth, alpha, beta, 0, 0, false) else {
                    aborted = true;
                    break;
                };

                if score <= alpha {
                    self.print_bound(depth, score, tui::ScoreBound::Upper);
                    asp_reduction = 0;
                    beta = lerp(beta, alpha, sp.asp_narrow);
                    alpha = (score - delta).max(-INF);
                } else if score >= beta {
                    self.print_bound(depth, score, tui::ScoreBound::Lower);
                    // Earlier moves scored below beta, so the first match is the cutoff move.
                    // Rotate it to the front; sorting would mix scores from different depths.
                    if let Some(i) = self.root_moves.iter().position(|rm| rm.score == score) {
                        self.root_moves[..=i].rotate_right(1);
                    }
                    asp_reduction = (asp_reduction + 1).min(sp.asp_reduction_max);
                    beta = (score + delta).min(INF);
                } else {
                    break;
                }

                delta += delta / sp.asp_widen_div;
            }

            if aborted {
                break; // don't sort an incomplete iteration; its scores come from different depths.
            }

            self.root_moves.sort_by_key(|m| Reverse(m.score));

            let new_score = self.root_moves[0].score;

            self.tm.update(
                &Iteration {
                    depth,
                    best_move_changed: self.prev_pv.get(0).is_some_and(|prev| prev != self.root_moves[0].mv),
                    root_moves: self.root_moves.len(),
                    best_nodes: self.root_moves[0].nodes,
                    total_nodes: self.nodes,
                    score: new_score,
                    prev_score: self.prev_score,
                },
                sp,
            );

            self.prev_pv = *self.root_moves[0].pv;
            self.prev_score = self.root_moves[0].score;
            self.print_info(depth, self.prev_score, &self.prev_pv);

            let elapsed = self.tm.elapsed().as_millis().max(1);
            self.pv_history
                .push_back(PvSnapshot { depth, time_ms: elapsed, score: self.prev_score, line: self.prev_pv });

            if self.pv_history.len() > 30 {
                self.pv_history.pop_front();
            }
        }

        if !self.cfg.limits.silent {
            let best = self.prev_pv.get(0).unwrap_or(self.root_moves[0].mv);
            match self.cfg.limits.protocol {
                Protocol::Uci => println!("bestmove {}", best.to_uci(self.root_pos.frc_notation)),
                Protocol::XBoard => println!("move {}", best.to_uci(self.root_pos.frc_notation)),
            }
            let _ = io::stdout().flush();
        }
    }

    #[inline]
    pub(crate) fn new(cfg: &'cfg SearchConfig, pos: &Position, history: &[u64], tt: Arc<TranspositionTable>) -> Self {
        let tm = Self::build_tm(cfg, pos);
        let root_moves = gen_legal_moves(pos).iter().map(|&mv| RootMove::new(mv)).collect();
        let mut trail = Vec::with_capacity(1024);
        Self::fill_trail(&mut trail, pos, history);
        Self {
            cfg,
            tm,
            root_moves,
            zobrist_trail: trail,
            root_pos: *pos,
            prev_pv: PvLine::new(),
            pv_history: VecDeque::new(),
            prev_score: -INF,
            nodes: 0,
            sel_depth: 0,
            iter_depth: 0,
            last_print: 0,
            tt,
        }
    }

    #[cfg(any(test, feature = "datagen"))]
    #[inline]
    pub(crate) fn best_move(&self) -> Option<Move> { self.root_moves.first().map(|rm| rm.mv) }

    #[cfg(feature = "datagen")]
    #[inline]
    pub(crate) fn best_score(&self) -> Option<i32> { self.root_moves.first().map(|rm| rm.score) }

    /// `history` must end with the root hash.
    fn fill_trail(trail: &mut Vec<u64>, pos: &Position, history: &[u64]) {
        trail.clear();
        let keep = history.len().min(pos.halfmove_clock as usize + 1);
        trail.extend_from_slice(&history[history.len() - keep..]);
    }

    fn build_tm(cfg: &SearchConfig, pos: &Position) -> TimeManager {
        let phase = i32::from(pos.psqt_accumulator().to_array()[LANE_PHASE]);
        let game_ply = u64::from(pos.fullmove_number.saturating_sub(1)) * 2 + u64::from(pos.stm == Color::Black);
        TimeManager::new(&cfg.limits, cfg.start_time, pos.stm, cfg.overhead, phase, game_ply, &cfg.search_params)
    }

    #[inline]
    fn check_signals(&mut self) -> bool {
        self.cfg.node_slots[self.cfg.thread_id].store(self.nodes, Ordering::Relaxed);

        if self.cfg.stop.load(Ordering::Relaxed)
            || self.tm.is_hard_limit_reached()
            || (self.cfg.limits.nodes > 0 && self.node_count() >= self.cfg.limits.nodes)
        {
            self.cfg.stop.store(true, Ordering::Relaxed);
            // Signal the pool to stop, but finish depth 1 so bestmove comes from search.
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

    /// Each thread updates its slot every `NODE_CHECK_INTERVAL` nodes.
    #[inline(always)]
    fn node_count(&self) -> u64 { self.cfg.node_slots.iter().map(|s| s.load(Ordering::Relaxed)).sum() }

    #[cold]
    fn search_info_data<'a>(&'a self, depth: i32, score: i32, pv: &'a PvLine) -> tui::SearchInfoData<'a> {
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
            history: &self.pv_history,
            board: &self.root_pos,
            display: &self.cfg.display,
        }
    }

    #[cold]
    fn print_line(&self, depth: i32, score: i32, pv: &PvLine, bound: tui::ScoreBound) {
        let mut data = self.search_info_data(depth, score, pv);
        data.bound = bound;

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            tui::print_pretty_search_info(&data);
        } else {
            tui::print_search_info(self.cfg.limits.protocol, &data);
        }
    }

    #[cold]
    fn print_info(&self, depth: i32, score: i32, pv: &PvLine) {
        if self.cfg.limits.silent {
            return;
        }
        self.print_line(depth, score, pv, tui::ScoreBound::Exact);
    }

    #[cold]
    fn print_bound(&self, depth: i32, score: i32, bound: tui::ScoreBound) {
        // The current iteration is unfinished, therefore report the previous PV.
        if self.cfg.limits.silent || is_mate(score) || self.prev_pv.len == 0 {
            return;
        }
        self.print_line(depth, score, &self.prev_pv, bound);
    }

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
        println!("info depth {depth} currmove {} currmovenumber {move_number}", mv.to_uci(self.root_pos.frc_notation));
    }

    #[cold]
    fn print_realtime(&self) {
        if self.cfg.limits.silent || self.prev_pv.len == 0 {
            return;
        }

        let data = self.search_info_data(self.iter_depth, self.prev_score, &self.prev_pv);
        tui::print_pretty_search_info(&data);
    }
}

/// Builds the array through a `Vec` to avoid a stack-sized temporary.
fn boxed_array<T: Clone, const N: usize>(value: T) -> Box<[T; N]> {
    vec![value; N].into_boxed_slice().try_into().unwrap_or_else(|_| unreachable!())
}

fn lerp(from: i32, to: i32, pct: i32) -> i32 { (from * (100 - pct) + to * pct) / 100 }

/// Continuation-history contexts from 1, 2, and 4 plies back.
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
    #[inline]
    fn evaluate(&mut self) -> i32 {
        let phase = extract_phase(&self.accumulator);
        let pawn = self.pawn_cache.probe(&self.pos);
        let features = SharedFeatures::with_pawn(&self.pos, &pawn, self.xb_store());
        evaluate_generic::<i32>(&self.pos, &self.accumulator, phase, &self.eval_params, &features)
    }

    #[cfg(feature = "nocorr")]
    #[inline]
    fn corrected_eval(&self, raw_eval: i32) -> i32 { raw_eval.clamp(-MATE_BOUND, MATE_BOUND) }

    #[cfg(not(feature = "nocorr"))]
    #[inline]
    fn corrected_eval(&self, raw_eval: i32) -> i32 {
        let correction = self
            .history
            .correction(self.pos.stm, self.pos.pawn_key, self.pos.minor_key, self.pos.major_key)
            / CORRECTION_SCALE;

        (raw_eval + correction).clamp(-MATE_BOUND, MATE_BOUND)
    }

    /// At MAX_PLY, checked positions score as draws because there is no room to search an evasion.
    fn max_ply_score(&mut self, nodes: u64) -> i32 {
        if self.xb_checkers().is_not_empty() {
            return draw_score(nodes);
        }
        let raw_eval = self.evaluate();
        self.corrected_eval(raw_eval)
    }

    /// We do negamax instead of minimax because the zero-sum symmetry folds both
    /// players into one recursion. Scores are side-to-move relative, negated at
    /// every return, and `NodeType` stays three-way instead of six.
    ///
    /// Returns `Err` once `check_signals` calls for a stop.
    fn negamax<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        last_critical_ply: usize,
        cut_node: bool,
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
            return Ok(self.max_ply_score(searcher.nodes));
        }

        // Grandchildren share this counter, so reset it once for the whole subtree.
        self.stack[ply + 2].cutoff_count = 0;

        // ── Mate Distance Pruning
        // We know how close a mate can be at this ply, so don't search beyond it.
        // Root still needs every move scored.
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
        // Have we seen this position before? Different move orders often reach the
        // same position, and a deep enough entry can save us another search.
        //
        // Singular verification excludes the TT move, so the stored score can't tell
        // us whether that move stands alone.
        let tt_probe = if excluded.is_null() { searcher.tt.probe(self.pos.hash, ply) } else { TtData::NONE };
        let tt_move = tt_probe.mv(&self.pos);

        // A collision invalidates the whole entry, including its score and PV flag.
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
        // A check can lead straight into a tactic, so search the evasion too.
        let depth = if in_check { depth + 1 } else { depth };

        // ── Static Eval
        // A score for the position without searching moves, used by pruning.
        let raw_static_eval = if in_check {
            tt::SCORE_NONE
        } else if tt_probe.eval != tt::SCORE_NONE {
            tt_probe.eval
        } else {
            self.evaluate()
        };

        // ── Correction History
        // The same pawn or piece structure can make static eval consistently too high or low.
        // Correction history learns that bias from the gap between static eval and search score.
        let static_eval = if in_check { tt::SCORE_NONE } else { self.corrected_eval(raw_static_eval) };

        // Positions in check have no static eval, but we still need the previous
        // one to judge whether that side's next position is improving.
        self.stack[ply].static_eval = if in_check && ply >= 2 { self.stack[ply - 2].static_eval } else { static_eval };

        // ── Improving Flag
        // Has our position strengthened since our last turn?
        // Without an earlier eval to compare against, count it as improving.
        let improving = !in_check
            && (ply >= 2)
                .then(|| self.stack[ply - 2].static_eval)
                .filter(|&e| e != tt::SCORE_NONE)
                .is_none_or(|past| static_eval > past);

        // ── TT-Clamped Eval
        // A mate score is a distance, not an evaluation, so we don't use it to clamp static eval.
        let tt_clamped_eval = if is_mate(tt_probe.score) {
            static_eval
        } else {
            tt::clamp_to_bound(tt_probe.bound, tt_probe.score, static_eval)
        };

        // ── Reverse Futility Pruning (~52 Elo)
        // If eval clears beta by the margin, prune without searching.
        // Blend the result toward beta to temper the unsearched estimate.
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
                return Ok(lerp(tt_clamped_eval, beta, sp.rfp_blend));
            }
        }

        // ── Razoring (~17 Elo)
        // If eval is far below alpha, let qsearch see if tactics can
        // save the position before committing to a full-depth search.
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
        // Give the opponent a free move. If we still beat beta,
        // assume a legal move can do the same and skip the full search.
        // With only pawns left, passing can be better than moving. That's zugzwang.
        if !in_check
            && !N::PV
            && excluded.is_null()
            && !self.stack[ply].is_null
            && !self.is_nmp_verif
            && depth >= sp.nmp_min_depth
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

            let score = self.negamax::<NonPvNode>(searcher, null_depth, -beta, -beta + 1, ply + 1, ply + 1, !cut_node);

            searcher.zobrist_trail.pop();
            self.pos.unmake_null_move(&undo);
            self.stack[ply + 1].is_null = false;

            let score = -score?;
            if score >= beta && !is_loss(score) {
                let null_score = if is_win(score) { beta } else { score };

                // ── NMP Verification Search (~1 Elo)
                // A deep null-move cutoff might be a zugzwang or a false cutoff.
                // Verify it with null move pruning disabled; shallow nodes skip the cost.
                if depth <= sp.nmp_verif_min_depth {
                    return Ok(null_score);
                }

                self.is_nmp_verif = true;
                let verif = self.negamax::<NonPvNode>(searcher, null_depth, beta - 1, beta, ply, last_critical_ply, false);
                self.is_nmp_verif = false;

                if verif? >= beta {
                    return Ok(null_score);
                }
            }
        }

        // We scan pins once per node and reuse them for legality and SEE.
        let pins = Pins::new(&self.pos);

        // ── ProbCut (~4 Elo)
        // If a shallow search clears beta comfortably, the full-depth search likely will too.
        // SEE and qsearch filter captures before the reduced-depth search.
        if !N::PV && !in_check && excluded.is_null() && depth >= sp.probcut_depth_min && !is_mate(beta) {
            let probcut_beta = beta + sp.probcut_margin;
            let probcut_depth = (depth - sp.probcut_reduction).max(1);

            let stm = self.pos.stm;
            let opp = stm.opposite();
            let ksq = pins.king(stm);
            let pinned = pins.pinned(stm);

            let mut picker = MovePicker::new_qsearch(None, searcher.cfg, pins, false);

            self.xb_snapshot(ply);

            while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
                if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                    continue;
                }

                if !see_ge(&self.pos, mv, probcut_beta - static_eval, &pins, &searcher.cfg.see_values) {
                    continue;
                }

                let saved_acc = self.accumulator;
                let undo = self.pos.make_move(mv, &mut self.accumulator);
                self.xb_update(mv);

                searcher.tt.prefetch(self.pos.hash);
                searcher.zobrist_trail.push(self.pos.hash);

                let qscore = self.qsearch::<NonPvNode>(searcher, -probcut_beta, -probcut_beta + 1, ply + 1, Some(mv.to()), 0);
                let value = match qscore {
                    Ok(v) if -v >= probcut_beta => self
                        .negamax::<NonPvNode>(
                            searcher,
                            probcut_depth,
                            -probcut_beta,
                            -probcut_beta + 1,
                            ply + 1,
                            last_critical_ply,
                            !cut_node,
                        )
                        .map(|x| -x),

                    Ok(v) => Ok(-v),
                    Err(e) => Err(e),
                };

                searcher.zobrist_trail.pop();
                self.pos.unmake_move(mv, &undo);
                self.xb_restore(mv, ply);
                self.accumulator = saved_acc;

                let value = value?;
                if value >= probcut_beta {
                    searcher.tt.store(
                        self.pos.hash,
                        ply,
                        probcut_depth + 1,
                        value,
                        mv,
                        tt::Bound::Lower,
                        tt_probe.pv,
                        raw_static_eval,
                    );

                    return Ok(lerp(value, beta, sp.probcut_blend));
                }
            }
        }

        // ── Internal Iterative Reduction (~14 Elo)
        // Without a TT move to guide ordering, search a little shallower
        // to find one for later visits.
        let depth = if depth >= sp.iir_depth && tt_move.get().is_none() { depth - sp.iir_reduction } else { depth };

        // ──────── Move loop ────────

        let mut res = MoveResult { move_count: 0, best_eval: -INF, alpha, best_move: Move::null() };

        if N::ROOT {
            self.xb_snapshot(ply);

            for i in 0..searcher.root_moves.len() {
                let mv = searcher.root_moves[i].mv;

                // ── Root LMR (~21 Elo)
                // Root moves are ordered by the previous iteration, so later moves
                // are our worst guesses. Scout them at reduced depth; if one beats
                // alpha, search it again at full depth.
                let reduction = if depth >= sp.lmr_min_depth && i >= 1 && mv.is_quiet() && !in_check {
                    searcher.cfg.lmr(depth, i + 1) / LMR_SCALE
                } else {
                    0
                };

                // Count this move's nodes across aspiration retries for time management.
                let nodes_before = searcher.nodes;
                self.search_move::<N>(
                    searcher,
                    mv,
                    depth,
                    &mut res,
                    beta,
                    ply,
                    last_critical_ply,
                    Some(i),
                    reduction,
                    0,
                    cut_node,
                )?;

                searcher.root_moves[i].nodes += searcher.nodes - nodes_before;

                if res.alpha >= beta {
                    break;
                }
            }
        } else {
            let stm = self.pos.stm;
            let opp = stm.opposite();
            let threats = self.xb_threats(opp);
            let ksq = pins.king(stm);
            let pinned = pins.pinned(stm);

            self.stack[ply].quiet_count = 0;
            self.stack[ply].capture_count = 0;

            let (cont1, cont2, cont4) = cont_contexts(&self.stack[..], ply);

            let mut picker =
                MovePicker::new(tt_move.get(), searcher.cfg, pins, self.stack[ply].killers, threats, cont1, cont2, cont4);

            self.xb_snapshot(ply);

            let lmp_scale = (sp.lmp_scale + if improving { sp.lmp_imp } else { -sp.lmp_imp }).max(0);

            while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
                if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                    continue;
                }

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
                // At shallow depth, skip quiet moves whose eval is too low to raise alpha.
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
                // At shallow depth, later quiet moves are less likely to be the best.
                if !in_check
                    && mv.is_quiet()
                    && !N::PV
                    && depth <= sp.lmp_depth
                    && res.move_count as i32 >= sp.lmp_base + lmp_scale * depth * depth / 100
                {
                    continue;
                }

                // ── History Pruning (~7 Elo)
                // Deeply negative history makes a quiet move a poor candidate at shallow depth.
                // Promotions and killers are exempt. A killer's history can be poor in general
                // and still be saving in this position.
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
                // SEE is reliable for captures, but only a rough guide for quiets,
                // whose value may lie in threats and follow-ups elsewhere.
                if !in_check && !N::PV && res.move_count >= 1 {
                    let margin =
                        if mv.is_capture() { -sp.see_capture_margin * depth } else { -sp.see_quiet_margin * depth * depth };

                    if !see_ge(&self.pos, mv, margin, &pins, &searcher.cfg.see_values) {
                        continue;
                    }
                }

                // ── Late Move Reductions (~90 Elo)
                // Later quiets are less likely to beat alpha, so scout them at reduced depth.
                // A reduced scout that beats alpha gets a full-depth re-search.
                let reduction = if depth >= sp.lmr_min_depth && res.move_count >= 1 && mv.is_quiet() && !in_check {
                    let mut r = searcher.cfg.lmr(depth, res.move_count + 1);
                    let pt = self.pos.expect_piece_at(mv.from());
                    let hist = self
                        .history
                        .score_quiet(self.pos.stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4);

                    // Killers have cut off at this ply before; new attacks on more valuable pieces are forcing.
                    if mv == self.stack[ply].killers[0] || mv == self.stack[ply].killers[1] {
                        r -= sp.lmr_killer_bonus;
                    }

                    // A quiet that newly attacks more valuable pieces is more forcing, so reduce it less.
                    r -= self.pos.new_threats(pt, mv.from(), mv.to()).popcount() as i32 * sp.lmr_threat_bonus;

                    // Frequent cutoffs in earlier moves' child searches make later quiets less promising.
                    r += sp.lmr_fhc_malus * (self.stack[ply + 1].cutoff_count > sp.lmr_fhc_cutoff) as i32;

                    // The farther we get from the last reduced scout, the less we reduce.
                    r -= sp.lmr_critical_bonus * (ply as i32 - last_critical_ply as i32).min(sp.lmr_critical_cap);

                    if cut_node {
                        // At a cut node, later moves are less likely to find the cutoff, so reduce them more.
                        r += sp.lmr_cutnode_malus;

                        // A deep TT hit that didn't cut off still argues for a smaller reduction.
                        r -= sp.lmr_tt_cutnode_bonus * (tt_probe.depth >= depth) as i32;
                    }

                    let max_r = (depth - sp.lmr_retained).max(0) * LMR_SCALE;
                    // History gives promising moves more depth and poor ones less.
                    (r - hist / sp.lmr_hist_div).clamp(0, max_r) / LMR_SCALE
                } else {
                    0
                };

                let mut extension = 0i32;

                // ── Singular Extensions (~7 Elo)
                // A strong TT move may be holding the position together. Search the
                // alternatives against a bound below its score; if none reaches it, extend.
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
                    let sing_depth = (depth - 1) / sp.singext_depth_div;

                    // Verification runs at this ply and overwrites the move lists
                    // this node still needs for its history update.
                    let saved = self.stack[ply];

                    self.stack[ply].excluded = mv;
                    let sing_score = self.negamax::<NonPvNode>(searcher, sing_depth, sing_beta - 1, sing_beta, ply, ply, cut_node);
                    self.stack[ply].excluded = Move::null();

                    self.stack[ply].quiet_moves = saved.quiet_moves;
                    self.stack[ply].quiet_count = saved.quiet_count;
                    self.stack[ply].capture_moves = saved.capture_moves;
                    self.stack[ply].capture_count = saved.capture_count;

                    let sing_score = sing_score?;
                    if sing_score < sing_beta {
                        extension = 1;
                        if sing_score < sing_beta - sp.singext_double_margin {
                            extension = 2;
                        }
                    } else if sing_score >= beta {
                        // ── Multicut (~15 Elo)
                        // The search still clears beta without the TT move, so this is a multicut.
                        // Mate distances are unreliable with that move excluded, therefore return
                        // beta for mates and the score itself otherwise.
                        return Ok(if is_mate(sing_score) { beta } else { sing_score });
                    }
                }

                self.stack[ply].moved_pt = self.pos.expect_piece_at(mv.from());
                self.stack[ply].moved_to = mv.to();

                self.search_move::<N>(
                    searcher, mv, depth, &mut res, beta, ply, last_critical_ply, None, reduction, extension, cut_node,
                )?;

                if likely(res.alpha >= beta) {
                    self.stack[ply].cutoff_count += 1;

                    #[cfg(feature = "mvpstats")]
                    {
                        use crate::engine::mvpstats::{CutoffClass, record_cutoff};

                        let kind = if Some(mv) == tt_move.get() {
                            CutoffClass::Hash
                        } else if mv.is_capture() {
                            CutoffClass::Capture
                        } else if mv == self.stack[ply].killers[0] || mv == self.stack[ply].killers[1] {
                            CutoffClass::Killer
                        } else {
                            CutoffClass::Quiet
                        };

                        record_cutoff(res.move_count as u32, kind);
                    }

                    // ── History Gravity Heuristic (~95 Elo)
                    // Reward the cutoff move and penalize earlier quiets that failed.
                    // Deeper cutoffs get larger bonuses, capped to limit any one update.
                    let bonus = (depth.pow(2) * sp.hist_bonus_mult).min(sp.hist_bonus_cap);

                    if mv.is_history_quiet() {
                        let pt = self.pos.expect_piece_at(mv.from());

                        self.history
                            .update_quiet(stm, pt, mv.from(), mv.to(), threats, cont1, cont2, cont4, bonus);

                        // ── Killer Moves (~35 Elo)
                        // The two most recent quiet moves to cause a beta cutoff at this ply.
                        if mv != self.stack[ply].killers[0] {
                            self.stack[ply].killers[1] = self.stack[ply].killers[0];
                            self.stack[ply].killers[0] = mv;
                        }

                        // ── Quiet Malus (~25 Elo)
                        // Cutoffs reward one move, so penalize earlier quiets too.
                        // Otherwise, history only accumulates bonuses.
                        let quiet_limit = self.stack[ply].quiet_count - appended_quiet as usize;
                        for i in 0..quiet_limit {
                            let qm = self.stack[ply].quiet_moves[i];
                            let q_pt = self.pos.expect_piece_at(qm.from());
                            self.history
                                .update_quiet(stm, q_pt, qm.from(), qm.to(), threats, cont1, cont2, cont4, -bonus);
                        }
                    } else if mv.is_capture() && !mv.is_promotion() {
                        // ── Capture History Update
                        // Promotion captures use material-only ordering, without capture history.
                        self.update_capture_history(mv, bonus);
                    }

                    // ── Capture Malus
                    // MVV-LVA favors material-rich captures; capture history learns which ones keep failing.
                    let capture_limit = self.stack[ply].capture_count - appended_capture as usize;
                    for i in 0..capture_limit {
                        let cm = self.stack[ply].capture_moves[i];
                        self.update_capture_history(cm, -bonus);
                    }
                    break;
                }
            }
        }

        // No legal moves means mate or stalemate. In singular verification,
        // the TT move is excluded; if it was the only legal move, fail low.
        if unlikely(res.move_count == 0) {
            return if !excluded.is_null() {
                Ok(alpha)
            } else if in_check {
                Ok(mated_in(ply))
            } else {
                Ok(0)
            };
        }

        let bound = tt::Bound::from_score(res.best_eval, alpha_orig, beta);

        // ── TT store
        // Singular verification excludes the TT move, so its bound can't represent the full position.
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
        // A tactical best move tells us little about static-eval bias.
        // A fail-low can expose an overestimate; a fail-high, an underestimate.
        // Compare against raw eval so the correction doesn't move its own target.
        if !in_check
            && excluded.is_null()
            && !res.best_move.is_null()
            && !res.best_move.is_tactical()
            && res.best_eval.abs() < MATE_BOUND
            && !((bound == tt::Bound::Lower && res.best_eval <= raw_static_eval)
                || (bound == tt::Bound::Upper && res.best_eval >= raw_static_eval))
        {
            let diff = res.best_eval - raw_static_eval;
            self.history
                .update_correction(self.pos.stm, self.pos.pawn_key, self.pos.minor_key, self.pos.major_key, diff, depth);
        }
        Ok(res.best_eval)
    }

    /// Reads the attacker and victim from the pre-move position, so `mv` must be unmade first.
    #[inline(always)]
    fn update_capture_history(&mut self, mv: Move, bonus: i32) {
        let attacker = self.pos.expect_piece_at(mv.from());
        self.history
            .update_capture(self.pos.stm, attacker, mv.to(), self.pos.captured_piece(mv), bonus);
    }

    /// One snapshot serves every sibling move at this ply.
    #[inline(always)]
    fn xb_snapshot(&mut self, ply: usize) {
        #[cfg(not(feature = "nostore"))]
        self.xorboard.snapshot(&mut self.xb_snapshots[ply]);
        #[cfg(feature = "nostore")]
        let _ = ply;
    }

    #[cfg(not(feature = "nostore"))]
    #[inline(always)]
    fn xb_store(&self) -> Option<&XorBoard> { Some(&self.xorboard) }

    #[cfg(feature = "nostore")]
    #[inline(always)]
    fn xb_store(&self) -> Option<&XorBoard> { None }

    /// Updates the XorBoard to match the post-move position.
    #[inline(always)]
    fn xb_update(&mut self, mv: Move) {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.update(&self.pos, mv);
            debug_assert!(self.xorboard.tracks(&self.pos), "xorboard drift after {mv:?}");
        }
        #[cfg(feature = "nostore")]
        let _ = mv;
    }

    #[inline(always)]
    fn xb_restore(&mut self, mv: Move, ply: usize) {
        #[cfg(not(feature = "nostore"))]
        {
            self.xorboard.restore(mv, &self.xb_snapshots[ply]);
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
            self.xorboard.threats(color)
        }
        #[cfg(feature = "nostore")]
        {
            self.pos.threats(color)
        }
    }

    /// `mv` must be legal. Root moves come from `gen_legal_moves`;
    /// interior moves pass `is_legal` before reaching this function.
    fn search_move<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        mv: Move,
        depth: i32,
        res: &mut MoveResult,
        beta: i32,
        ply: usize,
        last_critical_ply: usize,
        root_idx: Option<usize>,
        mut reduction: i32,
        extension: i32,
        cut_node: bool,
    ) -> Result<(), SearchAborted> {
        let sp = &searcher.cfg.search_params;
        let saved_acc = self.accumulator;
        let undo = self.pos.make_move(mv, &mut self.accumulator);
        self.xb_update(mv);

        searcher.tt.prefetch(self.pos.hash);

        // ── Gives-Check LMR Adjustment (~4 Elo)
        // A checking move forces a reply, so reduce it less.
        if self.xb_checkers().is_not_empty() {
            reduction = (reduction - sp.lmr_check_bonus).max(0);
        }

        res.move_count += 1;
        searcher.zobrist_trail.push(self.pos.hash);

        if N::ROOT {
            searcher.print_currmove(depth, mv, res.move_count);
        }

        let eval = self.pvs::<N>(
            searcher,
            depth,
            res.alpha,
            beta,
            ply,
            last_critical_ply,
            res.move_count == 1,
            reduction,
            extension,
            mv,
            cut_node,
        );

        searcher.zobrist_trail.pop();
        self.pos.unmake_move(mv, &undo);
        self.xb_restore(mv, ply);
        self.accumulator = saved_acc;

        let eval = eval?;

        if N::ROOT
            && let Some(i) = root_idx
        {
            // A fail-low gives only an upper bound, not a useful score for ordering.
            // Stable sorting keeps these moves in their previous iteration's order.
            if res.move_count == 1 || eval > res.alpha {
                searcher.root_moves[i].score = eval;
                searcher.root_moves[i].pv.compose(mv, &self.stack[ply + 1].pv);
            } else {
                searcher.root_moves[i].score = -INF;
            }
        }

        if eval > res.best_eval {
            res.best_eval = eval;
            res.best_move = mv;

            if eval > res.alpha {
                res.alpha = eval;
                let (current_stack, next_stack) = self.stack.split_at_mut(ply + 1);
                current_stack[ply].pv.compose(mv, &next_stack[0].pv);
            }
        }
        Ok(())
    }

    // ── Principal Variation Search (~14 Elo)
    /// Search the first move with the full window; scout later moves at zero width.
    /// Re-search reduced fail-highs at full depth, and improvements below beta
    /// with the full window.
    #[inline]
    fn pvs<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        depth: i32,
        alpha: i32,
        beta: i32,
        ply: usize,
        last_critical_ply: usize,
        is_first: bool,
        reduction: i32,
        extension: i32,
        mv: Move,
        cut_node: bool,
    ) -> Result<i32, SearchAborted> {
        let sp = &searcher.cfg.search_params;

        // Singular extensions can stack across plies, so cap depth to keep LMR lookups in bounds.
        let search_depth = (depth - 1 + extension).min(MAX_DEPTH);

        if is_first {
            return Ok(-self.negamax::<N::Next>(
                searcher,
                search_depth,
                -beta,
                -alpha,
                ply + 1,
                last_critical_ply,
                !N::PV && !cut_node,
            )?);
        }

        let scout_cut = reduction > 0 || !cut_node;

        // ── LMR Scout
        // Scout later moves at zero width, reducing late quiets when possible.
        // A reduced scout that beats alpha gets a full-depth re-search.
        let reduced_depth = search_depth - reduction;
        let mut score = -self.negamax::<NonPvNode>(searcher, reduced_depth, -alpha - 1, -alpha, ply + 1, ply + 1, scout_cut)?;

        if score > alpha && reduction > 0 {
            score =
                -self.negamax::<NonPvNode>(searcher, search_depth, -alpha - 1, -alpha, ply + 1, last_critical_ply, !cut_node)?;

            // ── Post-LMR Continuation History (~8 Elo)
            // Reward quiets that still cut at full depth; penalize those that fall back below alpha.
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
            // An improvement below beta needs a full-window search to build the PV.
            score = -self.negamax::<N::Next>(searcher, search_depth, -beta, -alpha, ply + 1, last_critical_ply, false)?;
        }

        Ok(score)
    }

    // ── Quiescence Search (~655 Elo)
    /// Searches captures and quiet queen promotions beyond the horizon to avoid
    /// scoring a position mid-tactic. In check, searches every evasion.
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
            return Ok(self.max_ply_score(searcher.nodes));
        }

        let alpha_orig = alpha;

        // ── QSearch TT Probe (~22 Elo)
        // PV nodes must search the line to rebuild the principal variation.
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
        // In check there is no stand-pat; the picker must generate every legal evasion.
        let raw_eval = if in_check {
            tt::SCORE_NONE
        } else if qs_tt.eval != tt::SCORE_NONE {
            qs_tt.eval
        } else {
            let phase = extract_phase(&self.accumulator);
            let lazy = self.corrected_eval(evaluate_psqt(&self.pos, &self.accumulator, phase));
            let lazy_floor = lazy - lazy_eval_margin(&self.pos, phase, sp); // ~1 Elo
            if lazy_floor >= beta {
                return Ok(lerp(lazy, beta, sp.qs_lazy_blend));
            }
            self.evaluate()
        };

        let mut best_eval = if in_check {
            -INF
        } else {
            let eval = self.corrected_eval(raw_eval);
            let stand_pat = if is_mate(qs_tt.score) { eval } else { tt::clamp_to_bound(qs_tt.bound, qs_tt.score, eval) };
            if stand_pat >= beta {
                return Ok(lerp(stand_pat, beta, sp.qs_standpat_blend));
            }

            alpha = alpha.max(stand_pat);

            // ── Delta Pruning (~20 Elo)
            // If even the most valuable capture plus a margin can't bridge the gap to alpha, prune.
            let best_capturable = [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight, PieceType::Pawn]
                .into_iter()
                .find(|&pt| self.pos.pieces(pt, opp).is_not_empty())
                .map_or(0, |pt| searcher.cfg.mvvlva_victim[pt]);

            if stand_pat + best_capturable + sp.delta_margin < alpha {
                return Ok(alpha);
            }
            stand_pat
        };

        let mut moves_made = 0;
        let mut best_move = Move::null();

        let pins = Pins::new(&self.pos);
        let ksq = pins.king(stm);
        let pinned = pins.pinned(stm);

        // ── Quiescence TT Move (~9 Elo)
        let mut picker = MovePicker::new_qsearch(qs_tt_move.get(), searcher.cfg, pins, in_check);

        let recapture_only = !in_check && qs_ply >= sp.qs_recapture_ply;

        self.xb_snapshot(ply);

        while let Some(mv) = picker.next(&self.pos, self.xb_rows(), self.history) {
            if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            // ── Recapture-only Deep QS (~1 Elo)
            // Past qs_recapture_ply, captures must continue the exchange on the
            // square the opponent just moved to. Off-square captures cause mutual-
            // attack soup (e.g. 30-bishop positions); forcing tactics this deep
            // are recapture chains.
            if recapture_only
                && let Some(prev) = prev_to
                && mv.to() != prev
            {
                continue;
            }

            // ── QSearch SEE Pruning (~65 Elo)
            // Prune moves whose destination-square exchange loses too much material.
            // In check, even a losing capture may be the only legal evasion.
            if !in_check && !see_ge(&self.pos, mv, sp.qs_see_margin, &pins, &searcher.cfg.see_values) {
                continue;
            }

            let saved_acc = self.accumulator;
            let undo = self.pos.make_move(mv, &mut self.accumulator);
            self.xb_update(mv);

            moves_made += 1;
            searcher.zobrist_trail.push(self.pos.hash);

            let score = self.qsearch::<N>(searcher, -beta, -alpha, ply + 1, Some(mv.to()), qs_ply + 1);

            searcher.zobrist_trail.pop();
            self.pos.unmake_move(mv, &undo);
            self.xb_restore(mv, ply);
            self.accumulator = saved_acc;

            let score = -score?;
            if score > best_eval {
                best_eval = score;
                best_move = mv;

                if score > alpha {
                    // Keep the PV going through qsearch instead of stopping at the search horizon.
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
        let bound = tt::Bound::from_score(best_eval, alpha_orig, beta);
        searcher
            .tt
            .store_qs(self.pos.hash, ply, best_eval, best_move, bound, N::PV || qs_tt.pv, raw_eval);

        Ok(best_eval)
    }

    /// Draws are checked before mate detection, so mate on the 100th half-move scores as a draw.
    #[inline]
    fn is_draw(&self, ply: usize, trail: &[u64]) -> bool {
        self.pos.is_fifty_move_draw() || self.pos.is_draw_by_material() || (ply > 0 && self.is_repetition(self.pos.hash, trail))
    }

    /// A second occurrence ends search cycles without waiting for threefold.
    #[inline]
    fn is_repetition(&self, key: u64, trail: &[u64]) -> bool {
        if trail.len() < 2 {
            return false;
        }

        let window = self.pos.halfmove_clock as usize;
        let start = trail.len().saturating_sub(window + 1);
        let end = trail.len() - 2;

        if start > end {
            return false;
        }

        let needle = U64x4::splat(key);
        let slice = &trail[start..=end];
        let chunks = slice.rchunks_exact(4);
        let remainder = chunks.remainder();

        // Side to move is hashed, so scanning every ply still finds only same-side repetitions.
        for chunk in chunks {
            // SAFETY: Each chunk is four u64s (32 bytes); load permits unaligned reads.
            let vec = unsafe { U64x4::load(chunk.as_ptr()) };
            if vec.cmp_eq(needle).any() {
                return true;
            }
        }

        remainder.contains(&key)
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use crate::core::{board::STARTPOS, defs::TOTAL_PHASE};

    const REACH_FENS: [&str; 2] = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    ];
    const REACH_DEPTHS: [i32; 2] = [10, 14];
    const REACH_HASH_MB: usize = 1;

    #[derive(PartialEq, Eq)]
    struct SearchOutcome {
        nodes: u64,
        score: Option<i32>,
        best_move: Option<Move>,
    }

    #[derive(PartialEq, Eq)]
    struct TmObservation {
        hard: Duration,
        stops: bool,
        soft_after: [Duration; 3],
    }

    fn search_outcomes(params: SearchParams, depth: i32) -> Vec<SearchOutcome> {
        let limits = Limits { depth, silent: true, protocol: Protocol::Uci, ..Default::default() };
        let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, params);
        let tt = Arc::new(TranspositionTable::new(REACH_HASH_MB, 1));
        tt.set_age_factor(params.tt_age_factor);
        let mut history = History::new();
        let mut out = Vec::new();

        // One generation apart, or tt_age_factor has no distance to discount.
        for fen in REACH_FENS {
            let board = Position::from_fen(fen);
            for _ in 0..2 {
                tt.begin_search();
                let mut s = Searcher::new(&cfg, &board, &[board.hash], tt.clone());
                s.iterative_deepening(&mut history);
                let best = s.best_move();
                out.push(SearchOutcome { nodes: s.nodes, score: best.map(|_| s.prev_score), best_move: best });
            }
        }
        out
    }

    fn tm_observations(params: &SearchParams) -> Vec<TmObservation> {
        let mut out = Vec::new();

        for (movestogo, forced) in [(0, false), (30, true), (2, false)] {
            let limits = Limits { wtime: 60_000, btime: 60_000, winc: 600, binc: 600, movestogo, ..Default::default() };
            let mut tm = TimeManager::new(&limits, Instant::now(), Color::White, 10, TOTAL_PHASE / 2, 40, params);

            if forced {
                tm.scale_base_soft(f64::from(params.tm_single_root) / 100.0);
            }

            let hard = tm.hard_limit();
            let (soft_ms, hard_ms) = (tm.soft_limit().as_millis() as u64, tm.hard_limit().as_millis() as u64);
            let stops = tm.should_stop(soft_ms / 2, hard_ms / 3, params);

            let mut prev_score = 0;
            let mut soft_after = [Duration::ZERO; 3];

            for (i, (depth, best_move_changed, best_nodes, score)) in
                [(4, true, 100, 20), (8, false, 700, 15), (12, false, 950, -60)].into_iter().enumerate()
            {
                let iter = Iteration { depth, best_move_changed, root_moves: 4, best_nodes, total_nodes: 1000, score, prev_score };
                tm.update(&iter, params);
                soft_after[i] = tm.soft_limit();
                prev_score = score;
            }

            out.push(TmObservation { hard, stops, soft_after });
        }

        out
    }

    #[test]
    fn every_tunable_reaches_search() {
        let baselines = REACH_DEPTHS.map(|d| search_outcomes(SearchParams::default(), d));
        let tm_baseline = tm_observations(&SearchParams::default());

        for def in crate::engine::search_params::tunable_param_defs() {
            #[cfg(feature = "nocorr")]
            if def.name.contains("corr") {
                continue;
            }

            let moved = [def.min, def.max].into_iter().any(|v| {
                let mut params = SearchParams::default();
                assert!(params.set(def.name, v));
                tm_observations(&params) != tm_baseline
                    || REACH_DEPTHS
                        .iter()
                        .zip(&baselines)
                        .any(|(&depth, baseline)| search_outcomes(params, depth) != *baseline)
            });
            assert!(moved, "{} does not reach the search at either bound", def.name);
        }
    }

    #[test]
    fn the_game_ply_tracks_the_position_not_the_move_list() {
        let limits = Limits { wtime: 60_000, btime: 60_000, silent: true, protocol: Protocol::Uci, ..Default::default() };
        let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, SearchParams::default());
        let tt = Arc::new(TranspositionTable::new(1, 1));
        let opening = Position::from_fen(STARTPOS);
        let late = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 60");
        let early_hard = Searcher::new(&cfg, &opening, &[opening.hash], tt.clone()).tm.hard_limit();
        let late_hard = Searcher::new(&cfg, &late, &[late.hash], tt).tm.hard_limit();
        assert!(late_hard > early_hard, "the ramp did not open: {early_hard:?} to {late_hard:?}");
    }

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

    #[test]
    fn trail_reaches_the_last_irreversible_move() {
        let limits = Limits { depth: 32, silent: true, protocol: Protocol::Uci, ..Default::default() };
        let cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, SearchParams::default());
        let tt = Arc::new(TranspositionTable::new(1, 1));

        let mut board = Position::from_fen(STARTPOS);
        let mut acc = board.psqt_accumulator();
        let mut history = vec![board.hash];
        for uci in ["e2e4", "e7e5", "g1f3", "b8c6", "f3g1"] {
            let mv = gen_legal_moves(&board)
                .iter()
                .copied()
                .find(|mv| mv.to_uci(board.frc_notation) == uci)
                .expect("test moves must be legal");
            board.make_move(mv, &mut acc);
            history.push(board.hash);
        }

        let after_e5 = history[2];
        let searcher = Searcher::new(&cfg, &board, &history, tt);
        assert_eq!(&searcher.zobrist_trail, &history[2..], "trail does not hold the positions since 1...e5");

        let mut board_history = History::new();
        let worker = Worker {
            pos: searcher.root_pos,
            accumulator: searcher.root_pos.psqt_accumulator(),
            stack: boxed_array(Stack::default()),
            #[cfg(not(feature = "nostore"))]
            xorboard: XorBoard::from_position(&searcher.root_pos),
            #[cfg(not(feature = "nostore"))]
            xb_snapshots: boxed_array(XbSnapshot::default()),
            history: &mut board_history,
            pawn_cache: PawnCache::new(),
            eval_params: EvalParams::<i32>::from_const(),
            is_nmp_verif: false,
        };
        assert!(worker.is_repetition(after_e5, &searcher.zobrist_trail), "position after 1...e5 did not repeat");
    }
}
