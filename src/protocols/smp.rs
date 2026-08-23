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
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    thread,
};

use crate::{
    core::{
        board::Position,
        defs::{INF, is_loss, is_mate},
        moves::Move,
    },
    engine::{
        history::History,
        search::{RESULT_NONE, SearchConfig, Searcher, ThreadResult},
        tt::TranspositionTable,
    },
    protocols::spmc,
};

/// The thread whose move the pool plays.
pub fn await_results(cfg: &SearchConfig) {
    const SPINS: usize = 1 << 14;

    for _ in 0..SPINS {
        if cfg.result_slots.iter().all(|slot| slot.load(Ordering::Acquire) != RESULT_NONE) {
            return;
        }
        thread::yield_now();
    }
}

pub fn winner(cfg: &SearchConfig) -> usize {
    if cfg.limits.depth > 0 || cfg.limits.mate.is_some() {
        return 0;
    }
    vote(&cfg.result_slots)
}

/// Tallies the pool's picks and returns the index of the thread that won.
pub fn vote(result_slots: &[AtomicU64]) -> usize {
    if result_slots.len() < 2 {
        return 0;
    }

    let results: Vec<ThreadResult> = result_slots
        .iter()
        .map(|slot| match slot.load(Ordering::Acquire) {
            RESULT_NONE => ThreadResult { mv: Move::null(), score: -INF, depth: 0 },
            packed => ThreadResult::unpack(packed),
        })
        .collect();

    let Some(min_score) = results.iter().filter(|r| r.score != -INF).map(|r| r.score).min() else {
        return 0;
    };

    let weight = |r: &ThreadResult| r.score - min_score + 14;
    let votes = |mv| results.iter().filter(|r| r.mv == mv && r.score != -INF).map(weight).sum::<i32>();

    let mut best = 0;

    for cur in 1..results.len() {
        let incumbent = &results[best];
        let candidate = &results[cur];

        let incumbent_decisive = incumbent.score != -INF && is_mate(incumbent.score);
        let candidate_decisive = candidate.score != -INF && is_mate(candidate.score);

        let take = if incumbent_decisive {
            // Two correct searches cannot prove opposite results, so both scores
            // here have one sign and the larger magnitude is the shorter mate.
            // Shortest is what we want lost as well as won: the thread that found
            // the mate against us is the one that searched the danger.
            candidate_decisive && candidate.score.abs() > incumbent.score.abs()
        } else if candidate_decisive {
            true
        } else {
            let incumbent_votes = votes(incumbent.mv);
            let candidate_votes = votes(candidate.mv);

            !is_loss(candidate.score)
                && (candidate_votes > incumbent_votes || (candidate_votes == incumbent_votes && candidate.depth > incumbent.depth))
        };

        if take {
            best = cur;
        }
    }

    // Threads naming one move share a tally entry, so the scan
    // can end on a thread that agrees with main.
    if results[best].mv == results[0].mv { 0 } else { best }
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
        core::{
            board::STARTPOS,
            defs::{mate_in, mated_in},
            moves::Move,
        },
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
        let picks = [(0xAAAA, 20, 12), (0xBBBB, 18, 12), (0xBBBB, 18, 12), (0xBBBB, 18, 12)];
        let winner = vote(&slots(&picks));
        assert_eq!(picks[winner].0, 0xBBBB, "a move three threads agree on at nearly the same score lost the tally");
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

    #[test]
    fn a_proof_outranks_the_tally_even_when_it_is_a_loss() {
        let winner = vote(&slots(&[(0xAAAA, 350, 20), (0xBBBB, mated_in(9), 11)]));
        assert_eq!(winner, 1, "a proven loss lost to an unproven score");
    }

    #[test]
    fn a_lost_pool_takes_the_shortest_mate() {
        let winner = vote(&slots(&[(0xAAAA, mated_in(21), 18), (0xBBBB, mated_in(5), 18)]));
        assert_eq!(winner, 1, "the pool kept a mate it had not proved");
    }

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
            let packed = slot.load(Ordering::Acquire);
            assert_ne!(packed, RESULT_NONE, "thread {id} parked without publishing");

            let result = ThreadResult::unpack(packed);
            assert!(legal.contains(&result.mv.inner()), "thread {id} published a move that is not legal at the root");
            assert!(result.depth >= 1, "thread {id} published no completed depth");
        }

        assert!(winner(&cfg) < threads, "the vote named a thread that does not exist");
    }
}
