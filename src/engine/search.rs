//! Negamax alpha-beta search
//!
//! # Architecture
//!
//! The search is single-threaded, but separates state into two entities
//! to provide a clean seam for future parallelism:
//! - `Searcher`: Owns global engine state (time management, history table, root moves).
//! - `Worker`: Owns thread-local mutability (the board, SIMD accumulator, ply stack).
//!
//! NodeType Specialization: Uses zero-cost traits (`RootNode`, `PvNode`, `NonPvNode`)
//! to eliminate runtime branches in the hot path.

use std::{
    collections::VecDeque,
    hint::{likely, unlikely},
    io::Write,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::Instant,
};

pub use crate::core::defs::Protocol;
use crate::{
    core::{
        board::Position,
        defs::{INF, MATE, MATE_BOUND, MAX_DEPTH, MAX_PLY, PieceType},
        moves::Move,
    },
    engine::{
        eval::evaluate,
        history,
        movegen::{gen_legal_moves, is_legal, is_pseudo_legal},
        movepicker::MovePicker,
        search_params::SearchParams,
        tm::TimeManager,
        tt,
    },
    tools::tui,
    weave::Vu64x4,
};

pub const NODE_CHECK_INTERVAL: u64 = 2048;
/// Minimum node count before printing `currmove` UCI output.
/// Suppresses per-move noise during fast games while still
/// reporting progress during long analysis.
pub const CURRMOVE_NODE_THRESHOLD: u64 = 100_000_000;
pub const PRINT_UPDATE_MS: u128 = 25;

/// ── Node type specialization ──
///
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

/// Move execution result: encapsulates value, PV, and move stats.
pub struct MoveResult {
    pub move_count: usize,
    pub best_eval:  i32,
    pub alpha:      i32,
    pub best_move:  Move,
}

// ──────── Searcher & Worker ────────
//
// Searcher owns the global search state — root moves, node counter,
// time management, zobrist history. One per go command.
//
// Worker owns the mutable board that changes as we descend the tree:
// position, accumulator, per-ply stack.
//
// This separation is the natural seam for future parallelism:
// one Searcher, N Workers, each exploring a different subtree.

pub struct Searcher<'cfg> {
    pub cfg:           &'cfg SearchConfig,
    pub tm:            TimeManager,
    pub root_moves:    Vec<RootMove>,
    pub history:       Vec<u64>,
    pub root_pos:      Position,
    pub prev_pv:       Line,
    pub prev_score:    i32,
    pub nodes:         u64,
    pub sel_depth:     i32,
    pub iter_depth:    i32,
    pub last_print:    u128,
    pub pv_history:    VecDeque<(u128, Line, i32)>,
    pub history_table: history::History,
    pub tt:            Arc<tt::TranspositionTable>,
}

#[repr(align(32))]
pub struct Worker {
    pub pos:         Position,
    pub accumulator: crate::weave::Vi16x8,
    pub stack:       Box<[Stack; MAX_PLY + 1]>,
    pub history:     history::History,
}

impl<'cfg> Searcher<'cfg> {
    /// ── Iterative deepening ──
    /// Search depth 1, then 2, then 3, ...
    ///
    /// Seems wasteful — why redo shallow work? Two reasons:
    ///   1. Each iteration's move ordering feeds the next. A deep search with
    ///      good ordering is faster than a blind deep search.
    ///   2. Natural anytime algorithm: always have a best move ready from the
    ///      last completed iteration.
    #[inline]
    pub fn iterative_deepening(&mut self) -> history::History {
        self.nodes = 0;

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            print!("\x1b[2J\x1b[H");
            let _ = std::io::stdout().flush();
        }

        if let Some(perft_depth) = self.cfg.limits.perft {
            use crate::tools::perft::perft;
            let mut board = self.root_pos;
            let mut acc = board.get_initial_accumulator();
            println!("Nodes searched: {}", perft(&mut board, perft_depth, &mut acc));
            return self.history_table;
        }

        if self.root_moves.is_empty() {
            if !self.cfg.limits.silent {
                println!("bestmove 0000");
            }
            return self.history_table;
        }

        if !self.cfg.limits.searchmoves.is_empty() {
            self.root_moves
                .retain(|rm| self.cfg.limits.searchmoves.contains(&rm.mv));
            if self.root_moves.is_empty() {
                eprintln!("info string error: no legal moves match searchmoves");
                if !self.cfg.limits.silent {
                    println!("bestmove 0000");
                }
                return self.history_table;
            }
        }

        let depth_limit = match (self.cfg.limits.mate, self.cfg.limits.depth) {
            (Some(mate_d), _) => (mate_d * 2).min(MAX_DEPTH),
            (_, d) if d > 0 => d,
            _ => MAX_DEPTH,
        };

        // fresh worker initialized ONCE — prevents 123KB stack memset per iteration.
        let mut worker = Worker {
            pos:         self.root_pos,
            accumulator: self.root_pos.get_initial_accumulator(),
            stack:       vec![Stack::default(); MAX_PLY + 1]
                .into_boxed_slice()
                .try_into()
                .unwrap_or_else(|_| unreachable!()),
            history:     self.history_table,
        };

        let mut last_iter_elapsed = 0;

        for depth in 1..=depth_limit {
            self.iter_depth = depth;

            let elapsed = self.tm.elapsed().as_millis() as u64;
            let prev_depth_time = elapsed.saturating_sub(last_iter_elapsed);
            last_iter_elapsed = elapsed;

            // Between iterations:
            // Bail if soft limits say we probably
            // can't finish the next depth in time.
            if depth > 1
                && (elapsed >= self.tm.soft_limit().as_millis() as u64
                    || elapsed + (prev_depth_time * 2) > self.tm.hard_limit().as_millis() as u64
                    || (self.cfg.limits.softnodes > 0 && self.nodes >= self.cfg.limits.softnodes))
            {
                break;
            }

            if self.check_signals() {
                break;
            }

            // ── Aspiration windows (~42 Elo) ──
            let mut delta = self.cfg.search_params.asp_initial;
            let mut alpha = if depth >= 4 {
                (self.prev_score - delta).max(-INF)
            } else {
                -INF
            };
            let mut beta = if depth >= 4 {
                (self.prev_score + delta).min(INF)
            } else {
                INF
            };

            let mut aborted = false;
            loop {
                worker.pos = self.root_pos;
                worker.accumulator = self.root_pos.get_initial_accumulator();

                if worker
                    .negamax::<RootNode>(self, depth, alpha, beta, 0, None)
                    .is_err()
                {
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

                delta += delta / 3;
            }

            if aborted {
                // On abort, we explicitly DO NOT sort. The array is already correctly sorted
                // from the last fully completed iteration context. Sorting a partially aborted
                // iteration would corrupt the strict root ordering with cross-depth horizon nodes.
                break;
            }

            if self.is_stopped() {
                break;
            }

            // best move floats to front — feeds next iteration's ordering.
            self.root_moves.sort_by_key(|m| std::cmp::Reverse(m.score));

            if self.is_stopped() {
                break;
            }

            self.prev_pv = *self.root_moves[0].pv;
            self.prev_score = self.root_moves[0].score;
            self.print_info(depth, self.prev_score, &self.prev_pv);

            let elapsed = self.tm.elapsed().as_millis().max(1);
            self.pv_history
                .push_back((elapsed, self.prev_pv, self.prev_score));

            // Bounded history: the TUI only needs the most recent points for the sparkline.
            if self.pv_history.len() > 30 {
                self.pv_history.pop_front();
            }
        }

        if !self.cfg.limits.silent {
            let mut best = self.prev_pv.get(0).unwrap_or(self.root_moves[0].mv);

            // Guard: if the PV move is somehow not legal for the root position
            // fall back to root_moves[0] which was generated legally.
            if !is_pseudo_legal(&self.root_pos, best) {
                best = self.root_moves[0].mv;
            }

            match self.cfg.limits.protocol {
                Protocol::Uci => println!("bestmove {}", best.to_uci(self.root_pos.is_frc)),
                Protocol::XBoard => println!("move {}", best.to_uci(self.root_pos.is_frc)),
            }
            let _ = std::io::stdout().flush();
        }

        worker.history
    }

    #[inline]
    pub fn new(
        cfg: &'cfg SearchConfig,
        pos: &Position,
        history: &[u64],
        history_table: history::History,
        tt: Arc<tt::TranspositionTable>,
    ) -> Self {
        let phase = i32::from(pos.get_initial_accumulator().to_array()[2]);
        let tm =
            TimeManager::new(&cfg.limits, cfg.start_time, pos.stm, cfg.overhead, phase, &cfg.search_params);

        let root_moves = gen_legal_moves(pos)
            .iter()
            .map(|&mv| RootMove::new(mv))
            .collect();

        // Game history limited by the 50-move rule horizon.
        // Positions older than the last capture or pawn push can never repeat.
        let mut hist_vec = Vec::with_capacity(1024);
        let keep = history.len().min(pos.halfmove_clock as usize);
        if keep > 0 {
            let start = history.len() - keep;
            hist_vec.extend_from_slice(&history[start..]);
        }

        // Always include the current root position in history.
        // This ensures repetitions of the root are detected at ply 2, 4, etc.
        hist_vec.push(pos.hash);

        Self {
            cfg,
            tm,
            root_moves,
            history: hist_vec,
            root_pos: *pos,
            prev_pv: Line::new(),
            pv_history: VecDeque::new(),
            prev_score: -INF,
            nodes: 0,
            sel_depth: 0,
            iter_depth: 0,
            last_print: 0,
            history_table,
            tt,
        }
    }

    #[inline]
    pub fn reset(
        &mut self,
        cfg: &'cfg SearchConfig,
        pos: &Position,
        history: &[u64],
        history_table: history::History,
    ) {
        let phase = i32::from(pos.get_initial_accumulator().to_array()[2]);

        self.history.clear();
        let keep = history.len().min(pos.halfmove_clock as usize);
        if keep > 0 {
            let start = history.len() - keep;
            self.history.extend_from_slice(&history[start..]);
        }

        // Always include the current root position in history.
        self.history.push(pos.hash);

        self.cfg = cfg;
        self.tm =
            TimeManager::new(&cfg.limits, cfg.start_time, pos.stm, cfg.overhead, phase, &cfg.search_params);
        self.root_pos = *pos;
        self.root_moves = gen_legal_moves(pos)
            .iter()
            .map(|&mv| RootMove::new(mv))
            .collect();
        self.nodes = 0;
        self.sel_depth = 0;
        self.iter_depth = 0;
        self.last_print = 0;
        self.pv_history.clear();
        self.history_table = history_table;
    }

    #[inline]
    pub fn best_move(&self) -> Option<Move> {
        self.root_moves.first().map(|rm| rm.mv)
    }

    #[inline]
    pub fn best_score(&self) -> Option<i32> {
        self.root_moves.first().map(|rm| rm.score)
    }

    /// Periodic heartbeat:
    /// Check stop flag, hard time limit, node limit.
    /// Also drives realtime TUI updates — piggybacks on the same interval.
    #[inline]
    fn check_signals(&mut self) -> bool {
        if self.cfg.stop.load(Ordering::Acquire)
            || self.tm.is_hard_limit_reached()
            || (self.cfg.limits.nodes > 0 && self.nodes >= self.cfg.limits.nodes)
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

    #[cold]
    fn print_info(&self, depth: i32, score: i32, pv: &Line) {
        if self.cfg.limits.silent {
            return;
        }

        let ms = self.tm.elapsed().as_millis().max(1);
        let nps = (u128::from(self.nodes) * 1000) / ms;

        let history_vec: Vec<_> = self.pv_history.iter().copied().collect();
        let data = tui::SearchInfoData {
            depth,
            score,
            pv,
            sel_depth: self.sel_depth,
            nodes: self.nodes,
            nps: u64::try_from(nps).unwrap_or(u64::MAX),
            time_ms: ms,
            hashfull: self.tt.hashfull(),
            show_wdl: self.cfg.display.show_wdl,
            material: self.root_pos.material_count(),
            stm: self.root_pos.stm.as_usize(),
            history: &history_vec,
            board: &self.root_pos,
            use_ansi: self.cfg.display.use_ansi,
        };

        if self.cfg.display.go_pretty && self.cfg.limits.protocol == Protocol::Uci {
            tui::print_pretty_search_info(&data);
        } else {
            tui::print_search_info(self.cfg.limits.protocol, &data, self.cfg.display.pretty_print);
        }
    }

    /// UCI `currmove` — tells the GUI which root move is being searched.
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
        println!(
            "info depth {depth} currmove {} currmovenumber {move_number}",
            mv.to_uci(self.root_pos.is_frc)
        );
    }

    #[cold]
    fn print_realtime(&mut self) {
        if self.cfg.limits.silent {
            return;
        }
        let ms = self.tm.elapsed().as_millis().max(1);
        let best = &self.root_moves[0];
        let nps = (u128::from(self.nodes) * 1000) / ms;

        let history_vec: Vec<_> = self.pv_history.iter().copied().collect();
        let data = tui::SearchInfoData {
            depth:     self.iter_depth,
            sel_depth: self.sel_depth,
            score:     best.score,
            nodes:     self.nodes,
            nps:       u64::try_from(nps).unwrap_or(u64::MAX),
            time_ms:   ms,
            hashfull:  self.tt.hashfull(),
            pv:        &best.pv,
            show_wdl:  self.cfg.display.show_wdl,
            material:  self.root_pos.material_count(),
            stm:       self.root_pos.stm.as_usize(),
            history:   &history_vec,
            board:     &self.root_pos,
            use_ansi:  self.cfg.display.use_ansi,
        };

        tui::print_pretty_search_info(&data);
    }
}

impl Worker {
    /// Negamax with alpha-beta pruning. Since chess is zero-sum, we maximize the
    /// score from the current side's perspective at every node, negating the score
    /// as it returns up the tree.
    ///
    /// PVS layered on top:
    /// after the presumed best move gets a full-window search,
    /// all others are probed with a zero-width "scout" window (alpha, alpha+1).
    /// Most confirm they're worse. The rare fail-high triggers a re-search,
    /// but that's rare enough to be a net win.
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

        if self.is_draw(ply, &searcher.history) {
            return Ok(0);
        }

        if ply as i32 > searcher.sel_depth {
            searcher.sel_depth = ply as i32;
        }

        if depth <= 0 {
            return self.qsearch::<N>(searcher, alpha, beta, ply);
        }
        if ply >= MAX_PLY {
            return Ok(evaluate(&self.pos, &self.accumulator));
        }

        let alpha_orig = alpha;

        // ── TT probe (~128 Elo) ──
        let tt_move = if let Some((mv, score, depth_stored, bound)) = searcher.tt.probe(self.pos.hash, ply) {
            // Hash collisions can inject moves from unrelated positions.
            // Full pseudo-legality check rejects garbage before it reaches
            // the move picker or triggers a cutoff with a bogus score.
            let valid = mv.is_null() || is_pseudo_legal(&self.pos, mv);

            if valid && !N::PV && depth_stored >= depth && tt::can_cutoff(bound, score, alpha, beta) {
                return Ok(score);
            }
            if valid && !mv.is_null() {
                Some(mv)
            } else {
                None
            }
        } else {
            None
        };

        // ── TT move ordering (~56 Elo) ──
        let pv_move = pv_move.filter(|&mv| is_pseudo_legal(&self.pos, mv));
        let hash_move = tt_move.or(pv_move);

        let checkers = self.pos.checkers();
        let in_check = checkers.is_not_empty();

        // ── Check extension (~11 Elo) ──
        // Being in check is forcing — don't let the horizon cut us off
        // mid-tactic. Extend by one ply so the reply is always searched.
        let depth = if in_check { depth + 1 } else { depth };

        // ── Static eval ──
        let static_eval = if in_check {
            tt::SCORE_NONE
        } else {
            evaluate(&self.pos, &self.accumulator)
        };
        self.stack[ply].static_eval = static_eval;

        // ── Reverse Futility Pruning (~67 Elo) ──
        if !in_check
            && !N::PV
            && depth <= searcher.cfg.search_params.rfp_depth
            && static_eval - searcher.cfg.search_params.rfp_margin * depth >= beta
        {
            return Ok(static_eval);
        }

        // ── Razoring (~17 Elo) ──
        // Position is so far below alpha that a full-depth search is unlikely
        // to recover. Drop straight into qsearch to confirm.
        if !in_check
            && !N::PV
            && depth <= searcher.cfg.search_params.razoring_depth
            && static_eval + searcher.cfg.search_params.razoring_margin * depth < alpha
        {
            let score = self.qsearch::<N>(searcher, alpha, beta, ply)?;
            if score <= alpha {
                return Ok(score);
            }
        }

        // ── Null Move Pruning (~85 Elo) ──
        if !in_check
            && !N::PV
            && !self.stack[ply].is_null
            && static_eval >= beta
            && self.pos.has_non_pawn_material(self.pos.stm)
        {
            let sp = &searcher.cfg.search_params;
            let eval_r = ((static_eval - beta) / sp.nmp_eval_divisor).min(sp.nmp_eval_max);
            let r = sp.nmp_base_r + depth / sp.nmp_depth_divisor + eval_r;

            self.stack[ply + 1].is_null = true;
            let undo = self.pos.make_null_move();
            searcher.history.push(self.pos.hash);
            let score = match self.negamax::<NonPvNode>(
                searcher,
                (depth - r - 1).max(0),
                -beta,
                -beta + 1,
                ply + 1,
                None,
            ) {
                Ok(v) => -v,
                Err(e) => {
                    searcher.history.pop();
                    self.pos.unmake_null_move(&undo);
                    self.stack[ply + 1].is_null = false;
                    return Err(e);
                },
            };
            searcher.history.pop();
            self.pos.unmake_null_move(&undo);
            self.stack[ply + 1].is_null = false;

            if score >= beta {
                return Ok(if score > MATE_BOUND { beta } else { score });
            }
        }

        // ──────── Move loop ────────

        let mut res = MoveResult {
            move_count: 0,
            best_eval: -INF,
            alpha,
            best_move: Move::null(),
        };

        if N::ROOT {
            // Iterate the pre-sorted root move list.
            for i in 0..searcher.root_moves.len() {
                let mv = searcher.root_moves[i].mv;
                self.search_move::<N>(
                    searcher,
                    mv,
                    depth,
                    &mut res,
                    beta,
                    ply,
                    Some(i),
                    Some(mv) == pv_move,
                    0,
                )?;
                if likely(res.alpha >= beta) {
                    break;
                }
            }
        } else {
            let stm = self.pos.stm;
            let opp = stm.opposite();
            let ksq = self.pos.pieces(PieceType::King, stm).lsb();
            let pinned = self.pos.king_blockers();

            // Track searched quiets to penalize them if a later move causes a cutoff.
            // We reset the pre-allocated counter in the ply stack.
            self.stack[ply].quiet_count = 0;

            // Interior: staged move generation via MovePicker.
            let mut picker = MovePicker::new(hash_move, searcher.cfg);
            while let Some(mv) = picker.next(&self.pos, &self.history) {
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

                // ── Futility Pruning ──
                // At shallow depth, if static eval is already so far below alpha
                // that a quiet move is unlikely to raise it, skip the move.
                if !in_check
                    && mv.is_quiet()
                    && res.move_count >= 1
                    && depth <= searcher.cfg.search_params.fp_depth
                    && static_eval + searcher.cfg.search_params.fp_margin * depth <= res.alpha
                {
                    continue;
                }

                // ── Late Move Reductions (~90 Elo) ──
                // Moves late in the list are unlikely to beat alpha.
                // Search them at reduced depth; re-search fully on surprise.
                let reduction = if depth >= 2 && res.move_count >= 1 && mv.is_quiet() && !in_check {
                    let r = searcher.cfg.lmr_table[depth as usize][res.move_count + 1] as i32;
                    r.clamp(0, depth - 1)
                } else {
                    0
                };

                self.search_move::<N>(
                    searcher,
                    mv,
                    depth,
                    &mut res,
                    beta,
                    ply,
                    None,
                    Some(mv) == pv_move,
                    reduction,
                )?;

                if likely(res.alpha >= beta) {
                    // ── History Gravity Heuristic ──
                    // When a move causes a beta-cutoff, its presumably a strong response.
                    // We reward it so it surfaces earlier in future sibling nodes,
                    // and we punish the preceding moves that failed to refute the branch.
                    //
                    // Bonus is quadratic with depth to prioritize deep heuristics,
                    // capped at 400 to prevent a single deep search
                    // from permanently dominating the history table.
                    let bonus = depth.pow(2).min(400);

                    // Only reward if the cutoff itself was caused by a quiet move.
                    // Captures and structural moves (castling) are handled differently.
                    if mv.is_history_quiet() {
                        let pt = self.pos.expect_piece_at(mv.from());
                        let to = mv.to();
                        self.history.update(stm, pt, to, bonus);
                    }

                    // ── Asymmetric Penalty ──
                    // If a quiet move caused the cutoff, we penalize all PRECEDING
                    // quiets searched at this ply (the losers).
                    //
                    // If the cutoff was a capture, quiet_count is 0 because
                    // captures precede quiets in our picker.
                    let penalty_limit = if appended_quiet {
                        self.stack[ply].quiet_count.saturating_sub(1)
                    } else {
                        self.stack[ply].quiet_count
                    };

                    for i in 0..penalty_limit {
                        let qm = self.stack[ply].quiet_moves[i];
                        let q_pt = self.pos.expect_piece_at(qm.from());
                        let q_to = qm.to();

                        // Over time, this "anti-history" pushes bad moves deeper into the list.
                        self.history.update(stm, q_pt, q_to, -bonus);
                    }

                    break;
                }
            }
        }

        // No legal moves: checkmate (in check) or stalemate (not).
        if unlikely(res.move_count == 0) {
            return if in_check {
                Ok(-MATE + ply as i32)
            } else {
                Ok(0)
            };
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
            .store(self.pos.hash, ply, depth, res.best_eval, res.best_move, bound);

        Ok(res.best_eval)
    }

    /// Make a move, search it, unmake it. The "heartbeat" of alpha-beta.
    ///
    /// # Safety
    /// The move `mv` MUST be legal. Root legality is filtered during move list
    /// generation; interior legality is filtered by the MovePicker loop.
    ///
    /// Returns `Ok(())` on success, or `Err` if the search was aborted mid-flight.
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
        reduction: i32,
    ) -> Result<(), SearchAborted> {
        let saved_acc = self.accumulator;
        let undo = self.pos.make_move(mv, &mut self.accumulator);
        searcher.tt.prefetch(self.pos.hash);

        res.move_count += 1;
        searcher.history.push(self.pos.hash);

        if N::ROOT {
            searcher.print_currmove(depth, mv, res.move_count);
        }

        let eval = match self.pvs::<N>(
            searcher,
            depth,
            res.alpha,
            beta,
            ply,
            res.move_count == 1,
            is_pv_move,
            reduction,
        ) {
            Ok(v) => v,
            Err(e) => {
                searcher.history.pop();
                self.pos.unmake_move(mv, &undo);
                self.accumulator = saved_acc;
                return Err(e);
            },
        };

        searcher.history.pop();
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
                // Disjoint borrow of the stack for the hot path (copies PV into current ply)
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

    /// ── Principal Variation Search ──
    /// Full window for the first move,
    /// zero-width scout for the rest,
    /// Re-search on surprise fail-high.
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
    ) -> Result<i32, SearchAborted> {
        // Retrieve the expected PV move for the NEXT ply (ply + 1 is the child node's level).
        // If we are on the PV line AND we just played the PV move, we expect the child to also have a PV move.
        let next_pv = if (N::ROOT || N::PV) && is_pv_move {
            searcher.prev_pv.get(ply + 1)
        } else {
            None
        };

        if is_first {
            // No bound yet — search wide open.
            return Ok(-self.negamax::<N::Next>(searcher, depth - 1, -beta, -alpha, ply + 1, next_pv)?);
        }

        // ── LMR scout ──
        // Late quiet moves get a shallower scout. If the reduced search
        // still beats alpha, the move earned a full-depth re-search.
        let reduced_depth = depth - 1 - reduction;
        let mut score =
            -self.negamax::<NonPvNode>(searcher, reduced_depth, -alpha - 1, -alpha, ply + 1, None)?;

        // Re-search at full depth if the reduced scout found something.
        if score > alpha && reduction > 0 {
            score = -self.negamax::<NonPvNode>(searcher, depth - 1, -alpha - 1, -alpha, ply + 1, None)?;
        }

        if score > alpha && score < beta {
            // Genuine improvement — search with full window on the PV.
            score = -self.negamax::<N::Next>(searcher, depth - 1, -beta, -alpha, ply + 1, next_pv)?;
        }

        Ok(score)
    }

    /// ── Quiescence Search (~655 Elo) ──
    /// Evaluates positions only after all "noisy" (tactical/forcing) moves are resolved,
    /// preventing the horizon effect where a search stops right before a massive blunder.
    fn qsearch<N: NodeType>(
        &mut self,
        searcher: &mut Searcher,
        mut alpha: i32,
        beta: i32,
        ply: usize,
    ) -> Result<i32, SearchAborted> {
        self.stack[ply].pv.len = 0;

        if (searcher.nodes & (NODE_CHECK_INTERVAL - 1)) == 0 && searcher.check_signals() {
            return Err(SearchAborted);
        }

        searcher.nodes += 1;

        if self.is_draw(ply, &searcher.history) {
            return Ok(0);
        }

        if ply as i32 > searcher.sel_depth {
            searcher.sel_depth = ply as i32;
        }

        if ply >= MAX_PLY {
            return Ok(evaluate(&self.pos, &self.accumulator));
        }

        let checkers = self.pos.checkers();
        let in_check = checkers.is_not_empty();
        let stm = self.pos.stm;
        let opp = stm.opposite();

        // ── QSearch Evaluations & Evasions ──
        // If we are in check, the position is forced and static evaluation is meaningless.
        // We drop the stand-pat evaluation (-INF) and the MovePicker will generate
        // all legal evasions instead of just captures/queen promotions.
        let mut best_eval = if in_check {
            -INF
        } else {
            let eval = evaluate(&self.pos, &self.accumulator);
            if eval >= beta {
                return Ok(eval);
            }

            // ── Delta pruning (revisit after NMP+LMR+aspiration) ──
            // Tested: 0.00 Elo with bare search. Needs tighter windows and deeper
            // search to be effective. Code kept for when we revisit.
            // let best_capturable = [PieceType::Queen, PieceType::Rook, PieceType::Bishop, PieceType::Knight, PieceType::Pawn]
            //     .into_iter()
            //     .find(|&pt| self.pos.pieces(pt, opp).is_not_empty())
            //     .map_or(0, |pt| searcher.cfg.mvvlva_v[pt as usize]);
            // if eval + best_capturable + searcher.cfg.search_params.delta_margin < alpha {
            //     return Ok(alpha);
            // }

            alpha = alpha.max(eval);
            eval
        };

        let mut moves_made = 0;
        let ksq = self.pos.pieces(PieceType::King, stm).lsb();
        let pinned = self.pos.king_blockers();

        let mut picker = MovePicker::new_qsearch(None, searcher.cfg, in_check);

        while let Some(mv) = picker.next(&self.pos, &self.history) {
            if !is_legal(&self.pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            let saved_acc = self.accumulator;
            let undo = self.pos.make_move(mv, &mut self.accumulator);

            moves_made += 1;
            searcher.history.push(self.pos.hash);

            let score = match self.qsearch::<N>(searcher, -beta, -alpha, ply + 1) {
                Ok(v) => -v,
                Err(e) => {
                    searcher.history.pop();
                    self.pos.unmake_move(mv, &undo);
                    self.accumulator = saved_acc;
                    return Err(e);
                },
            };

            searcher.history.pop();
            self.pos.unmake_move(mv, &undo);
            self.accumulator = saved_acc;

            if score > best_eval {
                best_eval = score;
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

        Ok(best_eval)
    }

    // ──────── Heuristics & State Queries ────────

    /// Fifty move rule, insufficient material, or repetition → immediate draw.
    ///
    /// NOTE: checked before move generation, so a theoretical checkmate on
    /// exactly the 100th half-move is scored as a draw. This is the
    /// standard compromise — every top engine accepts this edge case.
    #[inline]
    pub fn is_draw(&self, ply: usize, history: &[u64]) -> bool {
        self.pos.is_fifty_move_draw()
            || self.pos.is_draw_by_material()
            || (ply > 0 && self.is_repetition(self.pos.hash, history))
    }

    /// Scans the history for a previous occurrence of the current hash.
    ///
    /// Any second occurrence is treated as a draw.
    /// While technically 3-fold is the rule,
    /// engines score the second to avoid searching infinite
    /// cycles and to prevent the GUI from flagging PVs that repeat.
    ///
    /// NOTE: This uses a raw scan without side-to-move skipping because
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

        // ── Implementation coupling ──
        // We exclude the current position using saturating_sub(2).
        // This works because the caller (search_move) has already pushed the current
        // hash into searcher.history before descending, placing it at len - 1.
        let end = history.len().saturating_sub(2);

        if start > end {
            return false;
        }

        let needle = Vu64x4::splat(key);
        let slice = &history[start..=end];
        let chunks = slice.rchunks_exact(4);
        let remainder = chunks.remainder();

        // ── Performance Note ──
        // We vector-scan the full slice, including opponent-ply hashes.
        // Zobrist already encodes side-to-move, so cross-ply hashes can never match —
        // the STM bit differs for every other entry, making false positives impossible.
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

// ──────── Supporting Structs ────────

/// Display configuration for search output.
#[derive(Clone, Copy, Default)]
pub struct SearchDisplay {
    pub show_wdl:      bool,
    pub go_pretty:     bool,
    pub pretty_print:  bool,
    pub show_currmove: bool,
    pub use_ansi:      bool,
}

impl SearchDisplay {
    pub const SILENT: Self = Self {
        show_wdl:      false,
        go_pretty:     false,
        pretty_print:  false,
        show_currmove: false,
        use_ansi:      false,
    };
    pub const DEFAULT: Self = Self {
        show_currmove: true,
        use_ansi: true,
        ..Self::SILENT
    };
}

#[derive(Clone)]
pub struct SearchConfig {
    pub limits:        Limits,
    pub start_time:    Instant,
    pub stop:          Arc<AtomicBool>,
    pub display:       SearchDisplay,
    pub search_params: SearchParams,
    pub overhead:      u64,
    pub mvvlva_v:      [i32; 8], // victim values, indexed by PieceType
    pub mvvlva_a:      [i32; 8], // attacker penalties, indexed by PieceType
    pub lmr_table:     Box<[[i8; MAX_PLY + 1]; MAX_DEPTH as usize + 1]>,
}

impl SearchConfig {
    /// Creates a configuration with default display settings.
    pub fn new(
        limits: Limits,
        start_time: Instant,
        stop: Arc<AtomicBool>,
        overhead: u64,
        search_params: SearchParams,
    ) -> Self {
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
        let lmr_table = Self::build_lmr_table(&search_params);

        Self {
            limits,
            start_time,
            stop,
            overhead,
            display,
            search_params,
            mvvlva_v,
            mvvlva_a,
            lmr_table,
        }
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

    /// LMR reduction table from tunable base/divisor.
    ///
    /// R(d, m) = base + ln(d) · ln(m) / divisor
    ///
    /// Logarithmic in both depth and move index: deeper searches tolerate
    /// larger reductions, and later moves deserve them. Precomputed so the
    /// inner loop never touches a float.
    fn build_lmr_table(sp: &SearchParams) -> Box<[[i8; MAX_PLY + 1]; MAX_DEPTH as usize + 1]> {
        let base = sp.lmr_base as f64 / 100.0;
        let divisor = sp.lmr_divisor as f64 / 100.0;

        let mut table = Box::new([[0i8; MAX_PLY + 1]; MAX_DEPTH as usize + 1]);
        for d in 1..=MAX_DEPTH as usize {
            for m in 1..=MAX_PLY {
                table[d][m] = (base + (d as f64).ln() * (m as f64).ln() / divisor).floor() as i8;
            }
        }
        table
    }
}

#[derive(Clone, Default, Debug)]
pub struct Limits {
    pub wtime:       u64,
    pub btime:       u64,
    pub winc:        u64,
    pub binc:        u64,
    pub movestogo:   u64,
    pub movetime:    u64,
    pub depth:       i32,
    pub nodes:       u64,
    pub softnodes:   u64,
    pub infinite:    bool,
    pub silent:      bool,
    pub protocol:    Protocol,
    pub mate:        Option<i32>,
    pub perft:       Option<u8>,
    pub searchmoves: Vec<Move>,
}

/// ── Root move ──
/// A legal move at ply 0 paired with its best known score.
/// After each iteration these are sorted by score so the strongest move
/// is searched first next time — the single most important factor for
/// alpha-beta efficiency.
pub struct RootMove {
    pub mv:    Move,
    pub score: i32,
    pub pv:    Box<Line>,
}

impl RootMove {
    #[inline]
    pub fn new(mv: Move) -> Self {
        Self {
            mv,
            score: -INF,
            pv: Box::new(Line::new()),
        }
    }
}

/// ── Principal Variation line ──
///
/// The PV is the engine's predicted best play for both sides.
/// When a new best move is found at any ply, we compose the line:
/// this move first, then the child's continuation, bubbling the full
/// sequence from leaves to root, one ply at a time.
#[derive(Clone, Copy)]
pub struct Line {
    pub moves: [Move; MAX_PLY],
    pub len:   usize,
}

impl Default for Line {
    fn default() -> Self {
        Self::new()
    }
}

impl Line {
    pub const fn new() -> Self {
        Self {
            moves: [Move::null(); MAX_PLY],
            len:   0,
        }
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
        if idx < self.len {
            Some(self.moves[idx])
        } else {
            None
        }
    }
}

// MAX_TRACKED_QUIETS matches MAX_MOVES today, but only because the legal quiet
// count is bounded by the total pseudo-legal count. If MAX_MOVES grows, this
// can shrink independently — we only care that quiet penalties cover all
// quiets searched before a cutoff, not all possible quiets in a position.
pub const MAX_TRACKED_QUIETS: usize = 256;

/// Per-ply scratch data.
#[derive(Clone, Copy)]
pub struct Stack {
    pub pv:          Line,
    pub quiet_moves: [Move; MAX_TRACKED_QUIETS],
    pub quiet_count: usize,
    pub static_eval: i32,
    pub is_null:     bool,
}

impl Default for Stack {
    fn default() -> Self {
        Self {
            pv:          Line::new(),
            quiet_moves: [Move::null(); MAX_TRACKED_QUIETS],
            quiet_count: 0,
            static_eval: tt::SCORE_NONE,
            is_null:     false,
        }
    }
}

/// Search was cut short — time, node limit, or external stop signal.
#[derive(Debug)]
pub struct SearchAborted;
