//! Lazy SMP thread pool and thread voting.
//!
//! Helpers are persistent, parked on an SPMC broadcast channel between
//! searches. `launch` sends one payload to all helpers; each receives it
//! inside a handler that runs the search, and the channel auto-signals
//! completion when the handler returns. `wait` blocks until every helper
//! has finished. `wake` (via `Drop`) shuts them down.
//!
//! Threads searching the same root from different orders reach different
//! answers, and the pool plays the one they agree on rather than main's alone.
//! [`winner`] is where that tally happens, once the pool has joined.

use std::{
    collections::HashMap,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use crate::{
    core::{
        board::Position,
        defs::{INF, is_loss, is_mate, is_win},
    },
    engine::{
        history::History,
        search::{SearchConfig, Searcher, ThreadResult},
        tt::TranspositionTable,
    },
    protocols::spmc,
};

/// The thread whose move the pool plays.
///
/// Under a fixed depth every thread finishes the same one, so the tally would
/// turn on thread timing alone and the move printed for a position would stop
/// being reproducible between runs. There the vote is skipped.
pub fn winner(cfg: &SearchConfig) -> usize {
    if cfg.limits.depth > 0 || cfg.limits.mate.is_some() {
        return 0;
    }
    vote(&cfg.result_slots)
}

/// Tallies the pool's picks and returns the index of the thread that won.
///
/// A thread votes for its own move with `(score - min_score + 10) · depth`.
/// Depth alone would let several shallow threads agreeing on a weak move
/// outvote one deep thread that found a better one; measuring each score
/// against the pool's worst is what keeps the evaluation in the tally. The
/// floor of 10 leaves that worst thread a vote proportional to its depth
/// instead of none at all.
///
/// Thread 0 is the incumbent and an inconclusive tally keeps its move.
pub fn vote(result_slots: &[AtomicU64]) -> usize {
    let results: Vec<ThreadResult> = result_slots.iter().map(|s| ThreadResult::unpack(s.load(Ordering::Acquire))).collect();

    // A thread that never finished a depth published -INF and has no opinion.
    // The minimum runs over the voters alone: let a -INF in and every real score
    // sits about 32000 above it, so the spread between them stops separating
    // anything and depth decides by itself.
    let Some(min_score) = results.iter().filter(|r| r.score != -INF).map(|r| r.score).min() else {
        return 0;
    };

    let weight = |r: &ThreadResult| (r.score - min_score + 10) * r.depth;

    // Every thread is entered, so no lookup below can miss.
    // A non-voter has depth 0 and so weight 0.
    let mut votes: HashMap<u16, i32> = HashMap::new();

    for r in &results {
        *votes.entry(r.mv.inner()).or_default() += weight(r);
    }

    let mut best = 0;

    for cur in 1..results.len() {
        let incumbent = &results[best];
        let candidate = &results[cur];

        let take = if is_win(incumbent.score) {
            // A proven win only yields to a faster one.
            candidate.score > incumbent.score
        } else if candidate.score != -INF && incumbent.score != -INF && is_loss(incumbent.score) {
            // Already lost: take the thread that proved the shortest path to it.
            candidate.score < incumbent.score
        } else if candidate.score != -INF && is_mate(candidate.score) {
            // A proved mate outranks the tally.
            true
        } else {
            let incumbent_votes = votes[&incumbent.mv.inner()];
            let candidate_votes = votes[&candidate.mv.inner()];

            // Two threads on one move share a tally entry and so always tie,
            // leaving the weight test to pick between identical outcomes.
            candidate.mv != incumbent.mv
                && !is_loss(candidate.score)
                && (candidate_votes > incumbent_votes
                    || (candidate_votes == incumbent_votes && weight(candidate) > weight(incumbent)))
        };

        if take {
            best = cur;
        }
    }

    best
}

pub struct LazySmpPool {
    tx: spmc::Sender<(SearchConfig, Position, Vec<u64>)>,
    handles: Vec<thread::JoinHandle<()>>,
}

/// A table sized to `hash_mb`, and a pool bound to it.
///
/// Both come back together because every helper captures the table by `Arc` for
/// the pool's lifetime: replace one without the other and the helpers go on
/// probing the old table while main writes the new one, neither side seeing what
/// the other stores.
pub fn table_and_pool(hash_mb: usize, threads: usize) -> (Arc<TranspositionTable>, Arc<LazySmpPool>) {
    let tt = Arc::new(TranspositionTable::new(hash_mb, threads));
    let pool = LazySmpPool::new(threads, tt.clone());
    (tt, pool)
}

impl LazySmpPool {
    pub fn new(threads: usize, tt: Arc<TranspositionTable>) -> Arc<Self> {
        if threads <= 1 {
            let (tx, _) = spmc::channel::<(SearchConfig, Position, Vec<u64>)>(0);
            return Arc::new(Self { tx, handles: Vec::new() });
        }

        let n = threads - 1;
        let (tx, rxs) = spmc::channel::<(SearchConfig, Position, Vec<u64>)>(n as u32);
        let mut handles = Vec::with_capacity(n);

        for (i, mut rx) in rxs.into_iter().enumerate() {
            let helper_id = i + 1;
            let tt = Arc::clone(&tt);

            handles.push(thread::spawn(move || {
                // Pin before the recv loop so the per-search History allocated below
                // lands on this thread's own node by first-touch, local not remote.
                tt.bind_search_thread(helper_id, threads);

                while rx
                    .recv(|(config, board, history)| {
                        let mut local_cfg = config.clone();
                        local_cfg.thread_id = helper_id;
                        let mut history_table = History::new();
                        let mut ctx = Searcher::new(&local_cfg, board, history, tt.clone());
                        ctx.iterative_deepening(&mut history_table);
                    })
                    .is_some()
                {}
            }));
        }
        Arc::new(Self { tx, handles })
    }

    pub fn launch(&self, cfg: &SearchConfig, board: Position, history: &[u64]) {
        if self.handles.is_empty() {
            return;
        }

        let mut helper_cfg = (*cfg).clone();

        helper_cfg.limits.silent = true;
        self.tx.send((helper_cfg, board, history.to_vec()));
    }

    pub fn wait(&self) {
        if self.handles.is_empty() {
            return;
        }
        self.tx.wait();
    }
}

impl Drop for LazySmpPool {
    fn drop(&mut self) {
        self.tx.wait();
        self.tx.wake();

        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{sync::atomic::AtomicBool, time::Instant};

    use super::*;
    use crate::{
        core::{board::STARTPOS, defs::mate_in, moves::Move},
        engine::{movegen::gen_legal_moves, search::Limits, search_params::SearchParams},
    };

    fn slots(picks: &[(u16, i32, i32)]) -> Vec<AtomicU64> {
        picks
            .iter()
            .map(|&(mv, score, depth)| AtomicU64::new(ThreadResult { mv: Move::from_u16(mv), score, depth }.pack()))
            .collect()
    }

    #[test]
    fn a_packed_result_survives_the_round_trip() {
        for score in [-INF, -29_999, -1, 0, 1, 29_999, INF] {
            let sent = ThreadResult { mv: Move::from_u16(0x1234), score, depth: 246 };
            let back = ThreadResult::unpack(sent.pack());
            assert_eq!((back.mv.inner(), back.score, back.depth), (0x1234, score, 246));
        }
    }

    #[test]
    fn depth_alone_does_not_outvote_a_better_score() {
        let winner = vote(&slots(&[(0xAAAA, 100, 12), (0xBBBB, -50, 10), (0xBBBB, -50, 10), (0xBBBB, -50, 10)]));
        assert_eq!(winner, 0, "three shallow threads outvoted a deeper, far better score");
    }

    #[test]
    fn agreement_at_a_comparable_score_carries_the_pool() {
        let winner = vote(&slots(&[(0xAAAA, 20, 12), (0xBBBB, 18, 12), (0xBBBB, 18, 12), (0xBBBB, 18, 12)]));
        assert_eq!(winner, 1, "a move three threads agree on at nearly the same score lost the tally");
    }

    #[test]
    fn a_proven_mate_beats_the_tally() {
        let winner = vote(&slots(&[(0xAAAA, 300, 20), (0xBBBB, mate_in(5), 9)]));
        assert_eq!(winner, 1, "a proven mate lost to a large eval");
    }

    #[test]
    fn a_proven_mate_only_yields_to_a_faster_one() {
        let kept = vote(&slots(&[(0xAAAA, mate_in(5), 20), (0xBBBB, mate_in(9), 20), (0xBBBB, mate_in(9), 20)]));
        assert_eq!(kept, 0, "a slower mate displaced a faster one");

        let taken = vote(&slots(&[(0xAAAA, mate_in(9), 20), (0xBBBB, mate_in(3), 10)]));
        assert_eq!(taken, 1, "a faster mate failed to displace a slower one");
    }

    #[test]
    fn a_thread_that_never_finished_a_depth_never_wins() {
        let winner = vote(&slots(&[(0xAAAA, 5, 11), (0xBBBB, -INF, 0), (0xBBBB, -INF, 0), (0xBBBB, -INF, 0)]));
        assert_eq!(winner, 0, "threads with no result won the vote");
    }

    #[test]
    fn a_helper_carries_the_pool_when_main_never_finished_a_depth() {
        let winner = vote(&slots(&[(0x0000, -INF, 0), (0xBBBB, 12, 14)]));
        assert_eq!(winner, 1, "main published nothing and still won");
    }

    /// Unguarded, 214 of 320 tallies landed off main while 19 changed the move.
    #[test]
    fn a_thread_naming_mains_own_move_never_takes_the_win() {
        let winner = vote(&slots(&[(0xAAAA, 5, 12), (0xAAAA, 90, 20), (0xAAAA, 90, 20)]));
        assert_eq!(winner, 0, "a helper won the tally without changing the move");
    }

    #[test]
    fn every_thread_silent_keeps_main() {
        assert_eq!(vote(&slots(&[(0x0000, -INF, 0), (0x0000, -INF, 0)])), 0);
    }

    #[test]
    fn every_thread_in_the_pool_publishes_its_own_slot() {
        let board = Position::from_fen(STARTPOS);
        let threads = 4;
        let (tt, pool) = table_and_pool(16, threads);

        let limits = Limits { movetime: 200, silent: true, ..Default::default() };
        let mut cfg = SearchConfig::new(limits, Instant::now(), Arc::new(AtomicBool::new(false)), 0, SearchParams::default());
        cfg.threads = threads;
        cfg.node_slots = SearchConfig::node_slots(threads);
        cfg.result_slots = SearchConfig::result_slots(threads);

        let trail = vec![board.hash];
        pool.launch(&cfg, board, &trail);

        let mut ctx = Searcher::new(&cfg, &board, &trail, tt.clone());
        ctx.iterative_deepening(&mut History::new());

        cfg.stop.store(true, Ordering::Relaxed);
        pool.wait();
        cfg.stop.store(false, Ordering::Relaxed);

        let legal: Vec<u16> = gen_legal_moves(&board).iter().map(|mv| mv.inner()).collect();

        for (id, slot) in cfg.result_slots.iter().enumerate() {
            let result = ThreadResult::unpack(slot.load(Ordering::Acquire));
            assert!(legal.contains(&result.mv.inner()), "thread {id} published a move that is not legal at the root");
            assert!(result.depth >= 1, "thread {id} published no completed depth");
        }

        assert!(winner(&cfg) < threads, "the vote named a thread that does not exist");
    }
}
