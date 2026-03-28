//! Thread-local worker state for playing independent games.
//!
//! Runs full games from opening book positions to checkmate or draw, filtering
//! out highly tactical or noisy positions before saving to the dataset.

use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering::Relaxed},
};

use super::{config::GenfensConfig, stats::GlobalStats};
use crate::{
    core::{
        board::Position,
        defs::{Color, GameOutcome},
        moves::Move,
    },
    engine::{
        adjudication::check_adjudication,
        eval::{evaluate_fast, extract_phase},
        history,
        movegen::gen_legal_moves,
        search::{Limits, SearchConfig as EngineSearchConfig, Searcher},
        search_params::SearchParams,
    },
    tools::dataset::SoulEntry,
};

// ──────── Self-Play Worker ────────

/// A self-play worker that generates NNUE training data one game at a time.
///
/// Each worker independently draws openings from a shared book, runs fixed-depth searches,
/// applies adjudication heuristics, filters positions for training quality,
/// and back-propagates the final WDL result once the game concludes.
pub struct WorkerState {
    pub board:                  Position,
    pub accumulator:            crate::weave::Vi16x8,
    /// Position hashes fed to the searcher (in-search repetition detection).
    pub search_history:         Vec<u64>,
    /// Full game hash trail (threefold repetition detection).
    pub game_history:           Vec<u64>,
    /// Positions awaiting the final game result before becoming training data.
    pub pending:                Vec<(SoulEntry, Color)>,
    /// Fully labeled entries, ready for serialization.
    pub confirmed:              Vec<SoulEntry>,
    pub book:                   Arc<Vec<String>>,
    pub config:                 GenfensConfig,
    pub rng:                    fastrand::Rng,
    pub global:                 Arc<GlobalStats>,
    /// Track if the last move made was a capture or promotion to filter out
    /// tactically "hot" positions reached via a trade.
    pub last_move_was_tactical: bool,
    /// Consecutive plies with |eval| ≥ 2500 (win adjudication fires at 4).
    pub win_adj_counter:        usize,
    /// Consecutive plies with |eval| ≤ 4 (draw adjudication fires at 12).
    pub draw_adj_counter:       usize,
    /// Evaluation of the previous ply (to detect sign flips).
    pub last_eval:              i32,
    pub local_attempted:        u64,
    pub local_plies:            u64,
}

/// Each clone gets a fresh, independently seeded RNG and zeroed adjudication
/// counters. Correlated randomness between parallel workers would reduce
/// opening diversity — the opposite of what datagen needs.
impl Clone for WorkerState {
    fn clone(&self) -> Self {
        Self {
            board:                  self.board,
            accumulator:            self.accumulator,
            search_history:         self.search_history.clone(),
            game_history:           self.game_history.clone(),
            pending:                self.pending.clone(),
            confirmed:              self.confirmed.clone(),
            book:                   Arc::clone(&self.book),
            config:                 self.config.clone(),
            rng:                    fastrand::Rng::new(),
            global:                 Arc::clone(&self.global),
            last_move_was_tactical: false,
            win_adj_counter:        0,
            draw_adj_counter:       0,
            last_eval:              0,
            local_attempted:        0,
            local_plies:            0,
        }
    }
}

thread_local! {
    static NEVER_STOP: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

impl WorkerState {
    pub fn new(book: Arc<Vec<String>>, config: GenfensConfig, global: Arc<GlobalStats>) -> Self {
        let board = Position::new();
        Self {
            accumulator: board.get_initial_accumulator(),
            board,
            search_history: Vec::with_capacity(config.max_plies),
            game_history: Vec::with_capacity(300),
            pending: Vec::with_capacity(config.buffer_size),
            confirmed: Vec::with_capacity(config.buffer_size),
            book,
            config,
            rng: fastrand::Rng::new(),
            global,
            last_move_was_tactical: false,
            win_adj_counter: 0,
            draw_adj_counter: 0,
            last_eval: 0,
            local_attempted: 0,
            local_plies: 0,
        }
    }

    /// Wipes all mutable state and sets up the board from a new opening FEN.
    pub fn reset_for_new_game(&mut self, fen: &str) {
        self.board = Position::from_fen(fen);
        self.accumulator = self.board.get_initial_accumulator();

        self.search_history.clear();
        self.search_history.push(self.board.hash);

        self.game_history.clear();
        self.game_history.push(self.board.hash);

        self.pending.clear();
        self.confirmed.clear();
        self.win_adj_counter = 0;
        self.draw_adj_counter = 0;
        self.last_move_was_tactical = false;
    }

    /// Plays one complete self-play game and returns labeled training positions.
    ///
    /// Flow: random opening → ply loop (search, adjudicate, filter, move) →
    /// back-propagate WDL result → return confirmed entries.
    pub fn play_game(&mut self) -> Vec<SoulEntry> {
        let opening = self.book[self.rng.usize(..self.book.len())].clone();
        self.reset_for_new_game(&opening);
        self.global.games.fetch_add(1, Relaxed);

        let mut outcome = GameOutcome::Draw;

        for _ in 0..self.config.max_plies {
            self.local_attempted += 1;
            self.local_plies += 1;

            let moves = gen_legal_moves(&self.board);

            // ──────── Draw Detection ────────

            // ── Checkmate / Stalemate ──
            if moves.is_empty() {
                outcome = if self.board.checkers().is_not_empty() {
                    self.global.term_check.fetch_add(1, Relaxed);
                    match self.board.stm {
                        Color::White => GameOutcome::BlackWins,
                        Color::Black => GameOutcome::WhiteWins,
                    }
                } else {
                    self.global.term_stale.fetch_add(1, Relaxed);
                    GameOutcome::Draw
                };
                break;
            }

            // ── 50-move rule ──
            if self.board.halfmove_clock >= 100 {
                outcome = GameOutcome::Draw;
                self.global.term_d50.fetch_add(1, Relaxed);
                break;
            }

            // ── Threefold repetition ──
            if self.board.is_threefold_repetition(&self.game_history) {
                outcome = GameOutcome::Draw;
                self.global.term_drep.fetch_add(1, Relaxed);
                break;
            }

            // ── Insufficient material ──
            if self.board.is_draw_by_material() {
                outcome = GameOutcome::Draw;
                self.global.term_dmat.fetch_add(1, Relaxed);
                break;
            }

            // ── Fixed depth search ──
            self.search_history.clear();
            let irrev_idx = self
                .game_history
                .len()
                .saturating_sub(self.board.halfmove_clock as usize);
            if irrev_idx < self.game_history.len() {
                self.search_history
                    .extend_from_slice(&self.game_history[irrev_idx..]);
            }
            self.search_history.push(self.board.hash);

            let (static_eval, search_eval, search_move) = self.search_position();

            let best_move = search_move.unwrap_or_else(|| {
                self.global.search_fail.fetch_add(1, Relaxed);
                moves[0] // graceful fallback to first legal move
            });

            // Track consecutive plies of extreme or dead-equal evals
            // to terminate hopelessly decided or drawn games early.
            let abs_eval = search_eval.abs();

            // ── Adjudication ──
            let ply = (self.board.fullmove_number as usize - 1) * 2 + (self.board.stm as usize);
            if let Some(res) = check_adjudication(
                search_eval,
                self.last_eval,
                &mut self.win_adj_counter,
                &mut self.draw_adj_counter,
                self.board.stm,
                ply,
            ) {
                outcome = res;
                if res == GameOutcome::Draw {
                    self.global.term_draw_adj.fetch_add(1, Relaxed);
                } else {
                    self.global.term_resign.fetch_add(1, Relaxed);
                }
                break;
            }
            self.last_eval = search_eval;

            // ── Position quality filter ──
            let is_quiet = self.board.checkers().is_empty() && !best_move.is_tactical();
            let prev_was_tactical = self.last_move_was_tactical;

            let within_score_window = abs_eval <= self.config.score_filter;

            let should_save = if !self.config.filter_quiet {
                true
            } else {
                match (is_quiet, prev_was_tactical, within_score_window) {
                    (true, false, true) => {
                        self.global.passed_filters.fetch_add(1, Relaxed);
                        true
                    },
                    (true, false, false) => {
                        self.global.filtered_score.fetch_add(1, Relaxed);
                        false
                    },
                    (true, true, _) => {
                        // Position itself is quiet, but reached via capture/check evasion.
                        // Skip to avoid "hot" tactical noise.
                        self.global.filtered_quiet.fetch_add(1, Relaxed);
                        false
                    },
                    (false, ..) => {
                        self.global.filtered_quiet.fetch_add(1, Relaxed);
                        false
                    },
                }
            };

            // Stochastic subsampling for training diversity.
            let sampled = self.config.sample_rate >= 1.0 || self.rng.f64() < self.config.sample_rate;

            if should_save && sampled {
                let entry = SoulEntry::from_board(
                    &self.board,
                    0.0, // placeholder WDL — back-filled once the game ends
                    Some(static_eval),
                    Some(search_eval),
                );
                self.pending.push((entry, self.board.stm));
            }

            // Advance the game
            self.last_move_was_tactical = best_move.is_tactical() || self.board.checkers().is_not_empty();
            self.board.make_move(best_move, &mut self.accumulator);
            self.search_history.push(self.board.hash);
            self.game_history.push(self.board.hash);
        }

        // Back-propagate game result to every saved position
        for (mut entry, stm) in self.pending.drain(..) {
            entry.result = outcome.relative_to(stm);
            self.confirmed.push(entry);
        }

        self.global
            .saved
            .fetch_add(self.confirmed.len() as u64, Relaxed);

        // Flush local stats to global
        self.global
            .attempted
            .fetch_add(self.local_attempted, Relaxed);
        self.global.plies.fetch_add(self.local_plies, Relaxed);
        self.local_attempted = 0;
        self.local_plies = 0;

        self.confirmed.clone()
    }

    /// Evaluates the current position with both a static eval and a fixed-depth search.
    ///
    /// Returns `(static_eval, search_eval, best_move)`, all from the side-to-move's
    /// perspective.
    fn search_position(&self) -> (i32, i32, Option<Move>) {
        let acc = self.board.get_initial_accumulator();
        let phase = extract_phase(&acc);
        let static_eval = evaluate_fast(&self.board, &acc, phase);

        let limits = Limits {
            depth: self.config.depth,
            nodes: self.config.hard_nodes.unwrap_or(0),
            softnodes: self.config.soft_nodes.unwrap_or(0),
            silent: true,
            ..Default::default()
        };

        let cfg = EngineSearchConfig::new_full(
            limits,
            std::time::Instant::now(),
            NEVER_STOP.with(Arc::clone),
            0,
            crate::engine::search::SearchDisplay::SILENT,
            SearchParams::default(),
        );

        let mut searcher = Searcher::new(&cfg, &self.board, &self.search_history, history::History::new());
        searcher.iterative_deepening();

        (static_eval, searcher.best_score().unwrap_or(0), searcher.best_move())
    }
}
