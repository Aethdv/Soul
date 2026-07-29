//! Thread-local worker state for playing independent games.
//!
//! Runs full games from opening book positions to checkmate or draw, filtering
//! out highly tactical or noisy positions before saving to the dataset.

use std::{
    mem,
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    time::Instant,
};

use super::{config::GenfensConfig, stats::GlobalStats};
use crate::{
    core::{
        board::Position,
        defs::{Color, GameOutcome, PieceType},
        moves::Move,
    },
    engine::{
        adjudication::check_adjudication,
        eval::{evaluate_fast, extract_phase},
        history::History,
        movegen::gen_legal_moves,
        search::{Limits, SearchConfig as EngineSearchConfig, SearchDisplay, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    tools::dataset::SoulEntry,
    weave::Vi16x8,
};

/// A self-play worker that generates training data one game at a time.
///
/// Each worker independently draws openings from a shared book, runs fixed-depth searches,
/// applies adjudication heuristics, filters positions for training quality,
/// and back-propagates the final WDL result once the game concludes.
pub struct WorkerState {
    pub board: Position,
    pub accumulator: Vi16x8,
    /// Position hashes fed to the searcher (in-search repetition detection).
    pub search_history: Vec<u64>,
    /// Full game hash trail (threefold repetition detection).
    pub game_history: Vec<u64>,
    /// Positions awaiting the final game result before becoming training data.
    pub pending: Vec<(SoulEntry, Color)>,
    /// Fully labeled entries, ready for serialization.
    pub confirmed: Vec<SoulEntry>,
    pub book: Arc<Vec<String>>,
    pub config: GenfensConfig,
    pub rng: fastrand::Rng,
    pub global: Arc<GlobalStats>,
    pub tt: Arc<TranspositionTable>,
    /// Persistent history table: reused across positions within a game,
    /// cleared between games. Avoids per-position heap allocation.
    pub history_table: History,
    /// Track if the last move made was a capture or promotion to filter out
    /// tactically "hot" positions reached via a trade.
    pub last_move_was_tactical: bool,
    /// Consecutive plies with |eval| ≥ 2500 (win adjudication fires at 4).
    pub win_adj_counter: usize,
    /// Consecutive plies with |eval| ≤ 4 (draw adjudication fires at 12).
    pub draw_adj_counter: usize,
    /// Evaluation of the previous ply (to detect sign flips).
    pub last_eval: i32,
    pub local_attempted: u64,
    pub local_plies: u64,
}

/// Each clone gets a fresh, independently seeded RNG and zeroed adjudication
/// counters. Correlated randomness between parallel workers would reduce
/// opening diversity, the opposite of what datagen needs.
impl Clone for WorkerState {
    fn clone(&self) -> Self {
        Self {
            board: self.board,
            accumulator: self.accumulator,
            search_history: self.search_history.clone(),
            game_history: self.game_history.clone(),
            pending: self.pending.clone(),
            confirmed: self.confirmed.clone(),
            book: Arc::clone(&self.book),
            config: self.config.clone(),
            rng: fastrand::Rng::new(),
            global: Arc::clone(&self.global),
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
            last_move_was_tactical: false,
            win_adj_counter: 0,
            draw_adj_counter: 0,
            last_eval: 0,
            local_attempted: 0,
            local_plies: 0,
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
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
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
        self.history_table.clear();
        self.win_adj_counter = 0;
        self.draw_adj_counter = 0;
        self.last_eval = 0;
        self.last_move_was_tactical = false;
    }

    /// Plays one game: either a full self-play game
    /// or a single random-restart position, depending on config.
    pub fn play_game(&mut self) -> Vec<SoulEntry> {
        if self.config.random_restart { self.play_random_position() } else { self.play_full_game() }
    }

    /// Random-restart: pick a random book FEN, play N random moves,
    /// run search, apply quality filters, emit one position.
    /// Each call produces at most one entry.
    fn play_random_position(&mut self) -> Vec<SoulEntry> {
        self.global.attempted.fetch_add(1, Relaxed);

        let opening = self.book[self.rng.usize(..self.book.len())].clone();
        self.reset_for_new_game(&opening);

        for _ in 0..self.config.random_plies {
            let moves = gen_legal_moves(&self.board);

            if moves.is_empty() {
                return Vec::new();
            }

            let mv = moves[self.rng.usize(0..moves.len())];

            self.board.make_move(mv, &mut self.accumulator);
            self.search_history.push(self.board.hash);
            self.game_history.push(self.board.hash);
        }

        if gen_legal_moves(&self.board).is_empty() {
            return Vec::new();
        }

        let (static_eval, search_eval, search_move) = self.search_position();

        let Some(best_move) = search_move else {
            self.global.search_fail.fetch_add(1, Relaxed);
            eprintln!("Warning: random-restart search returned no best move, skipping position");
            return Vec::new();
        };

        let abs_eval = search_eval.abs();

        let is_quiet = self.board.checkers().is_empty() && !best_move.is_tactical();
        let within_score_window = abs_eval <= self.config.score_filter;

        if !is_quiet {
            self.global.filtered_quiet.fetch_add(1, Relaxed);
            return Vec::new();
        }

        if !within_score_window {
            self.global.filtered_score.fetch_add(1, Relaxed);
            return Vec::new();
        }

        self.global.passed_filters.fetch_add(1, Relaxed);

        // The qsearch filter catches positions where the search sees something
        // the static eval doesn't: unresolved tactics the HCE can't learn.
        // `eval_contradiction_limit` is not applied here because random-restart
        // has no game outcome to contradict against; the filter only applies in
        // full-game mode where a back-propagated outcome exists.
        let delta = (search_eval - static_eval).abs();
        if delta > self.config.qsearch_filter {
            self.global.filtered_tactical.fetch_add(1, Relaxed);
            return Vec::new();
        }

        // No game outcome: set result to the draw prior (0.5).
        // At training time, the tuner's instance-confidence WDL blending
        // (training.rs:107-121) treats this as a weak anchor: near-zero
        // search scores keep the 0.5 prior, high-magnitude scores converge
        // to sigmoid(k · score). With wdl_blend=1.0, the draw prior only
        // sticks when the search itself is uncertain.
        let entry = SoulEntry::from_board(&self.board, 0.5, Some(search_eval));
        self.global.saved.fetch_add(1, Relaxed);
        vec![entry]
    }

    /// Plays one complete self-play game and returns labeled training positions.
    ///
    /// Flow: random opening → ply loop (search, adjudicate, filter, move) →
    /// back-propagate WDL result → return confirmed entries.
    fn play_full_game(&mut self) -> Vec<SoulEntry> {
        let opening = self.book[self.rng.usize(..self.book.len())].clone();
        self.reset_for_new_game(&opening);
        self.global.games.fetch_add(1, Relaxed);

        let mut outcome = GameOutcome::Draw;

        for _ in 0..self.config.max_plies {
            self.local_attempted += 1;
            self.local_plies += 1;

            let moves = gen_legal_moves(&self.board);

            // Checkmate / Stalemate
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

            // 50-move rule
            if self.board.halfmove_clock >= 100 {
                outcome = GameOutcome::Draw;
                self.global.term_d50.fetch_add(1, Relaxed);
                break;
            }

            // Threefold repetition
            if self.board.is_threefold_repetition(&self.game_history) {
                outcome = GameOutcome::Draw;
                self.global.term_drep.fetch_add(1, Relaxed);
                break;
            }

            // Insufficient material
            if self.board.is_draw_by_material() {
                outcome = GameOutcome::Draw;
                self.global.term_dmat.fetch_add(1, Relaxed);
                break;
            }

            // Fixed depth search
            self.search_history.clear();
            let irrev_idx = self.game_history.len().saturating_sub(self.board.halfmove_clock as usize);
            if irrev_idx < self.game_history.len() {
                self.search_history.extend_from_slice(&self.game_history[irrev_idx..]);
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

            // Adjudication
            let ply = (self.board.fullmove_number as usize - 1) * 2 + (self.board.stm as usize);

            if let Some(res) = check_adjudication(
                search_eval, self.last_eval, &mut self.win_adj_counter, &mut self.draw_adj_counter, self.board.stm, ply,
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

            // Position quality filter
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

            let pieces: u32 = PieceType::ALL.iter().map(|&pt| self.board.piece_count(pt) as u32).sum();

            let should_save = should_save && ply >= self.config.min_ply && pieces >= self.config.min_pieces;

            if !should_save {
                if ply < self.config.min_ply {
                    self.global.filtered_ply.fetch_add(1, Relaxed);
                } else if pieces < self.config.min_pieces {
                    self.global.filtered_pieces.fetch_add(1, Relaxed);
                }
            }

            // Skip positions where static eval diverges from search eval
            // by more than the threshold. i32::MAX disables this gate.
            let should_save = if should_save {
                let delta = (search_eval - static_eval).abs();
                if delta > self.config.qsearch_filter {
                    self.global.filtered_tactical.fetch_add(1, Relaxed);
                    false
                } else {
                    true
                }
            } else {
                false
            };

            if should_save && sampled {
                let entry = SoulEntry::from_board(&self.board, 0.0, Some(search_eval));
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
            entry.result = (outcome.relative_to(stm) * 2.0) as u8;

            let search_eval = entry.score as i32;
            let contradictory = match (outcome, stm) {
                // STM won but eval said STM was losing badly.
                (GameOutcome::WhiteWins, Color::White) | (GameOutcome::BlackWins, Color::Black) => {
                    search_eval < -self.config.eval_contradiction_limit
                },
                // STM lost but eval said STM was winning.
                (GameOutcome::WhiteWins, Color::Black) | (GameOutcome::BlackWins, Color::White) => {
                    search_eval > self.config.eval_contradiction_limit
                },
                // Draw, but eval was decisively non-draw.
                (GameOutcome::Draw, _) => search_eval.abs() > self.config.eval_contradiction_limit,
            };

            if contradictory {
                self.global.filtered_incorrect.fetch_add(1, Relaxed);
                continue;
            }

            self.confirmed.push(entry);
        }

        self.global.saved.fetch_add(self.confirmed.len() as u64, Relaxed);

        // Flush local stats to global
        self.global.attempted.fetch_add(self.local_attempted, Relaxed);
        self.global.plies.fetch_add(self.local_plies, Relaxed);
        self.local_attempted = 0;
        self.local_plies = 0;

        let mut out = Vec::new();
        mem::swap(&mut self.confirmed, &mut out);
        out
    }

    /// Evaluates the current position with both a static eval and a fixed-depth search.
    ///
    /// Returns `(static_eval, search_eval, best_move)`, all from the side-to-move's
    /// perspective.
    fn search_position(&mut self) -> (i32, i32, Option<Move>) {
        let phase = extract_phase(&self.accumulator);
        let static_eval = evaluate_fast(&self.board, &self.accumulator, phase);

        let limits = Limits {
            depth: self.config.depth,
            nodes: self.config.hard_nodes.unwrap_or(0),
            softnodes: self.config.soft_nodes.unwrap_or(0),
            silent: true,
            ..Default::default()
        };

        let cfg = EngineSearchConfig::new_full(
            limits,
            Instant::now(),
            NEVER_STOP.with(Arc::clone),
            0,
            SearchDisplay::SILENT,
            SearchParams::default(),
        );

        // History accumulates across positions within a game for better ordering.
        let mut searcher = Searcher::new(&cfg, &self.board, &self.search_history, Arc::clone(&self.tt));
        searcher.iterative_deepening(&mut self.history_table);
        let best_score = searcher.best_score().unwrap_or(0);
        let best_move = searcher.best_move();

        (static_eval, best_score, best_move)
    }
}
