//! Thread-local worker state for playing independent games.
//!
//! Runs full games from opening book positions to a terminal position or an
//! adjudicated one, recording every ply. Which of those plies become training
//! positions is decided at load time by `ReplayFilter`, not here.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering::Relaxed},
    },
    time::Instant,
};

use super::{config::DatagenConfig, stats::GlobalStats};
use crate::{
    core::{
        board::Position,
        defs::{Color, GameOutcome},
        moves::Move,
    },
    engine::{
        adjudication::check_adjudication,
        history::History,
        movegen::gen_legal_moves,
        search::{Limits, SearchConfig as EngineSearchConfig, SearchDisplay, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    tools::dataset::flip_score,
    weave::I16x8,
};

/// One finished game, in the shape viriformat stores.
///
/// `result` and every score are White-relative; the search reports side-to-move, so
/// each ply's score is flipped on the way in.
pub struct Game {
    pub opening: Position,
    pub result: u8,
    pub moves: Vec<(Move, i16)>,
}

/// A self-play worker that generates training data one game at a time.
///
/// Each worker independently draws openings from a shared book, runs fixed-depth searches,
/// and applies adjudication heuristics.
pub struct WorkerState {
    pub board: Position,
    pub accumulator: I16x8,
    /// Position hashes fed to the searcher (in-search repetition detection).
    pub search_history: Vec<u64>,
    /// Full game hash trail (threefold repetition detection).
    pub game_history: Vec<u64>,
    pub book: Arc<Vec<String>>,
    pub config: DatagenConfig,
    pub rng: fastrand::Rng,
    pub global: Arc<GlobalStats>,
    pub tt: Arc<TranspositionTable>,
    /// Reused across positions within a game, cleared between games.
    pub history_table: History,
    /// Consecutive plies with |eval| ≥ 2500
    pub win_adj_counter: usize,
    /// Consecutive plies with |eval| ≤ 4
    pub draw_adj_counter: usize,
    /// Evaluation of the previous ply (to detect sign flips).
    pub last_eval: i32,
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
            book: Arc::clone(&self.book),
            config: self.config.clone(),
            rng: fastrand::Rng::new(),
            global: Arc::clone(&self.global),
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
            win_adj_counter: 0,
            draw_adj_counter: 0,
            last_eval: 0,
            local_plies: 0,
        }
    }
}

thread_local! {
    static NEVER_STOP: Arc<AtomicBool> = Arc::new(AtomicBool::new(false));
}

impl WorkerState {
    pub fn new(book: Arc<Vec<String>>, config: DatagenConfig, global: Arc<GlobalStats>) -> Self {
        let board = Position::new();
        Self {
            accumulator: board.initial_accumulator(),
            board,
            search_history: Vec::with_capacity(config.max_plies),
            game_history: Vec::with_capacity(300),
            book,
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
            config,
            rng: fastrand::Rng::new(),
            global,
            win_adj_counter: 0,
            draw_adj_counter: 0,
            last_eval: 0,
            local_plies: 0,
        }
    }

    /// Wipes all mutable state and sets up the board from a new opening FEN.
    pub fn reset_for_new_game(&mut self, fen: &str) {
        self.board = Position::from_fen(fen);
        self.accumulator = self.board.initial_accumulator();

        self.search_history.clear();
        self.search_history.push(self.board.hash);

        self.game_history.clear();
        self.game_history.push(self.board.hash);

        self.history_table.clear();
        self.win_adj_counter = 0;
        self.draw_adj_counter = 0;
        self.last_eval = 0;
    }

    /// Plays one game from a random book opening and returns every ply of it.
    ///
    /// Flow: opening → ply loop (search, adjudicate, move) → the finished game.
    pub fn play_game(&mut self) -> Game {
        let fen = self.book[self.rng.usize(..self.book.len())].clone();
        self.reset_for_new_game(&fen);
        self.global.games.fetch_add(1, Relaxed);

        let opening = self.board;
        let mut moves = Vec::with_capacity(self.config.max_plies);
        let mut outcome = GameOutcome::Draw;

        for _ in 0..self.config.max_plies {
            self.local_plies += 1;

            let legal = gen_legal_moves(&self.board);
            // Checkmate / Stalemate
            if legal.is_empty() {
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

            let (search_eval, search_move) = self.search_position();

            let best_move = search_move.unwrap_or_else(|| {
                self.global.search_fail.fetch_add(1, Relaxed);
                legal[0] // graceful fallback to first legal move
            });

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

            // The file stores White-relative scores; the search reports side-to-move.
            let white_score = flip_score(search_eval, self.board.stm).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
            moves.push((best_move, white_score as i16));

            // Advance the game
            self.board.make_move(best_move, &mut self.accumulator);
            self.search_history.push(self.board.hash);
            self.game_history.push(self.board.hash);
        }

        self.global.plies.fetch_add(self.local_plies, Relaxed);
        self.local_plies = 0;
        Game { opening, result: outcome.packed(), moves }
    }

    /// Runs the fixed-depth search, returning `(score, best_move)` from the side to
    /// move's perspective.
    fn search_position(&mut self) -> (i32, Option<Move>) {
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
        (best_score, best_move)
    }
}
