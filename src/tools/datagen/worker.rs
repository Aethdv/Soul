use std::{
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

use super::{config::DatagenConfig, stats::Termination};
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
        search::{Limits, SearchConfig, SearchDisplay, Searcher},
        search_params::SearchParams,
        tt::TranspositionTable,
    },
    tools::dataset::flip_score,
};

pub(super) struct Game {
    pub(super) opening: Position,
    pub(super) result: u8,
    pub(super) moves: Vec<(Move, i16)>,
    pub(super) termination: Termination,
    pub(super) search_failures: u64,
}

pub(super) struct SelfPlayWorker {
    game_history: Vec<u64>,
    book: Arc<Vec<String>>,
    max_plies: usize,
    adjudicate: bool,
    rng: fastrand::Rng,
    tt: Arc<TranspositionTable>,
    history_table: History,
    search_config: SearchConfig,
}

impl SelfPlayWorker {
    pub(super) fn new(book: Arc<Vec<String>>, config: &DatagenConfig) -> Self {
        let limits = Limits {
            depth: config.depth,
            nodes: config.hard_nodes.unwrap_or(0),
            softnodes: config.soft_nodes.unwrap_or(0),
            silent: true,
            ..Default::default()
        };
        let stop = Arc::new(AtomicBool::new(false));
        let search_config = SearchConfig::new_full(limits, Instant::now(), stop, 0, SearchDisplay::SILENT, SearchParams::default());

        Self {
            game_history: Vec::with_capacity(config.max_plies + 1),
            book,
            max_plies: config.max_plies,
            adjudicate: config.adjudicate,
            rng: fastrand::Rng::new(),
            tt: Arc::new(TranspositionTable::new(16, 1)),
            history_table: History::new(),
            search_config,
        }
    }

    pub(super) fn play_game(&mut self) -> Game {
        let mut board = Position::from_fen(&self.book[self.rng.usize(..self.book.len())]);
        let mut acc = board.psqt_accumulator();

        self.game_history.clear();
        self.game_history.push(board.hash);
        self.history_table.clear();

        let opening = board;
        let mut moves = Vec::with_capacity(self.max_plies);
        let mut outcome = GameOutcome::Draw;
        let mut termination = Termination::PlyCap;
        let mut search_failures = 0;
        let mut adjudicator = self.adjudicate.then(Adjudicator::default);

        for _ in 0..self.max_plies {
            let legal_moves = gen_legal_moves(&board);
            if legal_moves.is_empty() {
                if board.checkers().is_not_empty() {
                    termination = Termination::Checkmate;
                    outcome = match board.stm {
                        Color::White => GameOutcome::BlackWins,
                        Color::Black => GameOutcome::WhiteWins,
                    };
                } else {
                    termination = Termination::Stalemate;
                }
                break;
            }

            if board.is_fifty_move_draw() {
                termination = Termination::FiftyMove;
                break;
            }
            if board.is_threefold_repetition(&self.game_history) {
                termination = Termination::Repetition;
                break;
            }
            if board.is_draw_by_material() {
                termination = Termination::Material;
                break;
            }

            let (search_score, search_move) = self.search_position(&board);

            let played_move = search_move.unwrap_or_else(|| {
                search_failures += 1;
                legal_moves[0]
            });

            if let Some(adjudicator) = adjudicator.as_mut() {
                let ply = (board.fullmove_number.saturating_sub(1) as usize) * 2 + (board.stm as usize);

                if let Some(adjudicated) = adjudicator.update(search_score, board.stm, ply) {
                    outcome = adjudicated;
                    termination =
                        if adjudicated == GameOutcome::Draw { Termination::AdjudicatedDraw } else { Termination::AdjudicatedWin };
                    break;
                }
            }

            let white_score = flip_score(search_score, board.stm).clamp(i32::from(i16::MIN), i32::from(i16::MAX));
            moves.push((played_move, white_score as i16));

            board.make_move(played_move, &mut acc);
            self.game_history.push(board.hash);
        }

        Game { opening, result: outcome.packed(), moves, termination, search_failures }
    }

    fn search_position(&mut self, board: &Position) -> (i32, Option<Move>) {
        self.search_config.reset();
        let mut searcher = Searcher::new(&self.search_config, board, &self.game_history, Arc::clone(&self.tt));
        searcher.iterative_deepening(&mut self.history_table);
        (searcher.best_score().unwrap_or(0), searcher.best_move())
    }
}
