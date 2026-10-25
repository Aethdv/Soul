use std::{
    sync::{Arc, atomic::Ordering::Relaxed},
    time::Instant,
};

use super::{config::DatagenConfig, stats::RunStats};
use crate::{
    core::{
        board::Position,
        defs::{Color, GameOutcome},
        moves::Move,
    },
    engine::{
        adjudication::Adjudicator,
        history::History,
        movegen::gen_legal_moves,
        search::{Limits, SearchConfig as EngineSearchConfig, SearchDisplay, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    tools::dataset::flip_score,
    weave::I16x8,
};

pub(super) struct Game {
    pub(super) opening: Position,
    pub(super) result: u8,
    pub(super) moves: Vec<(Move, i16)>,
}

pub(super) struct SelfPlayWorker {
    board: Position,
    accumulator: I16x8,
    game_history: Vec<u64>,
    book: Arc<Vec<String>>,
    config: DatagenConfig,
    rng: fastrand::Rng,
    stats: Arc<RunStats>,
    tt: Arc<TranspositionTable>,
    history_table: History,
}

impl SelfPlayWorker {
    pub(super) fn new(book: Arc<Vec<String>>, config: DatagenConfig, stats: Arc<RunStats>) -> Self {
        let board = Position::empty();
        Self {
            accumulator: board.psqt_accumulator(),
            board,
            game_history: Vec::with_capacity(300),
            book,
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
            config,
            rng: fastrand::Rng::new(),
            stats,
        }
    }

    pub(super) fn play_game(&mut self) -> Game {
        self.board = Position::from_fen(&self.book[self.rng.usize(..self.book.len())]);
        self.accumulator = self.board.psqt_accumulator();

        self.game_history.clear();
        self.game_history.push(self.board.hash);

        self.history_table.clear();

        let opening = self.board;
        let mut moves = Vec::with_capacity(self.config.max_plies);
        let mut outcome = GameOutcome::Draw;
        let mut adjudicator = self.config.adjudicate.then(Adjudicator::default);

        for _ in 0..self.config.max_plies {
            let legal_moves = gen_legal_moves(&self.board);
            if legal_moves.is_empty() {
                outcome = if self.board.checkers().is_not_empty() {
                    self.stats.checkmates.fetch_add(1, Relaxed);
                    match self.board.stm {
                        Color::White => GameOutcome::BlackWins,
                        Color::Black => GameOutcome::WhiteWins,
                    }
                } else {
                    self.stats.stalemates.fetch_add(1, Relaxed);
                    GameOutcome::Draw
                };
                break;
            }

            if self.board.is_fifty_move_draw() {
                outcome = GameOutcome::Draw;
                self.stats.fifty_move_draws.fetch_add(1, Relaxed);
                break;
            }

            if self.board.is_threefold_repetition(&self.game_history) {
                outcome = GameOutcome::Draw;
                self.stats.repetition_draws.fetch_add(1, Relaxed);
                break;
            }

            if self.board.is_draw_by_material() {
                outcome = GameOutcome::Draw;
                self.stats.material_draws.fetch_add(1, Relaxed);
                break;
            }

            let (search_score, search_move) = self.search_position();

            let played_move = search_move.unwrap_or_else(|| {
                self.stats.search_failures.fetch_add(1, Relaxed);
                legal_moves[0]
            });

            if let Some(adjudicator) = adjudicator.as_mut() {
                let ply = (self.board.fullmove_number.saturating_sub(1) as usize) * 2 + (self.board.stm as usize);

                if let Some(adjudicated) = adjudicator.update(search_score, self.board.stm, ply) {
                    outcome = adjudicated;

                    if adjudicated == GameOutcome::Draw {
                        self.stats.adjudicated_draws.fetch_add(1, Relaxed);
                    } else {
                        self.stats.adjudicated_wins.fetch_add(1, Relaxed);
                    }
                    break;
                }
            }

            let white_score = flip_score(search_score, self.board.stm).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
            moves.push((played_move, white_score as i16));

            self.board.make_move(played_move, &mut self.accumulator);
            self.game_history.push(self.board.hash);
        }

        Game { opening, result: outcome.packed(), moves }
    }

    fn search_position(&mut self) -> (i32, Option<Move>) {
        let limits = Limits {
            depth: self.config.depth,
            nodes: self.config.hard_nodes.unwrap_or(0),
            softnodes: self.config.soft_nodes.unwrap_or(0),
            silent: true,
            ..Default::default()
        };

        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let search_config =
            EngineSearchConfig::new_full(limits, Instant::now(), stop, 0, SearchDisplay::SILENT, SearchParams::default());

        let mut searcher = Searcher::new(&search_config, &self.board, &self.game_history, Arc::clone(&self.tt));
        searcher.iterative_deepening(&mut self.history_table);
        let best_score = searcher.best_score().unwrap_or(0);
        let best_move = searcher.best_move();
        (best_score, best_move)
    }
}
