//! Lazy SMP thread pool.

use std::{
    sync::{Arc, atomic::Ordering},
    thread,
};

use crate::{
    core::board::Position,
    engine::{
        history::History,
        search::{SearchConfig, Searcher},
        tt::TranspositionTable,
    },
    protocols::spmc,
};

pub(super) const DEFAULT_HASH_MB: usize = 16;
pub(super) const MAX_HASH_MB: usize = 524288;
pub(super) const MAX_THREADS: usize = 1024;

type SearchJob = (SearchConfig, Position, Vec<u64>);

#[derive(Clone)]
pub(super) struct TableAndPool {
    tt: Arc<TranspositionTable>,
    pool: Arc<LazySmpPool>,
    hash_mb: usize,
    threads: usize,
}

struct LazySmpPool {
    tx: spmc::Sender<SearchJob>,
    handles: Vec<thread::JoinHandle<()>>,
}

impl Default for TableAndPool {
    fn default() -> Self {
        let tt = Arc::new(TranspositionTable::new(DEFAULT_HASH_MB, 1));
        let pool = LazySmpPool::new(1, tt.clone());
        Self { tt, pool, hash_mb: DEFAULT_HASH_MB, threads: 1 }
    }
}

impl TableAndPool {
    pub(super) fn tt(&self) -> &TranspositionTable { &self.tt }
    pub(super) fn threads(&self) -> usize { self.threads }

    pub(super) fn set_hash(&mut self, mb: usize) {
        self.hash_mb = mb.clamp(1, MAX_HASH_MB);
        self.tt = Arc::new(TranspositionTable::new(self.hash_mb, self.threads));
        self.pool = LazySmpPool::new(self.threads, self.tt.clone());
    }

    pub(super) fn set_threads(&mut self, threads: usize) {
        let threads = threads.clamp(1, MAX_THREADS);
        if threads == self.threads {
            return;
        }

        self.threads = threads;

        // The table's placement on the nodes depends on the thread count.
        if self.tt.spans_nodes() {
            self.tt = Arc::new(TranspositionTable::new(self.hash_mb, threads));
        }

        self.pool = LazySmpPool::new(threads, self.tt.clone());
    }

    /// Runs the main search alongside the helpers and waits for them to finish.
    pub(super) fn search(&self, cfg: &SearchConfig, board: Position, trail: &[u64], history: &mut History) {
        if !self.pool.handles.is_empty() {
            let mut helper_cfg = cfg.clone();
            helper_cfg.limits.silent = true;
            self.pool.tx.send((helper_cfg, board, trail.to_vec()));
        }

        self.tt.bind_search_thread(0, cfg.threads);
        Searcher::new(cfg, &board, trail, self.tt.clone()).iterative_deepening(history);

        cfg.stop.store(true, Ordering::Relaxed);
        if !self.pool.handles.is_empty() {
            self.pool.tx.wait();
        }

        cfg.stop.store(false, Ordering::Relaxed);
    }
}

impl LazySmpPool {
    fn new(threads: usize, tt: Arc<TranspositionTable>) -> Arc<Self> {
        let n = threads - 1;
        let (tx, rxs) = spmc::channel::<SearchJob>(n as u32);
        let mut handles = Vec::with_capacity(n);

        for (i, mut rx) in rxs.into_iter().enumerate() {
            let helper_id = i + 1;
            let tt = Arc::clone(&tt);

            handles.push(thread::spawn(move || {
                // Pin before the loop so each search's History is first-touched on the helper's NUMA node.
                tt.bind_search_thread(helper_id, threads);

                while rx
                    .recv(|(config, board, trail)| {
                        let mut local_cfg = config.clone();
                        local_cfg.thread_id = helper_id;
                        let mut history = History::new();
                        let mut searcher = Searcher::new(&local_cfg, board, trail, tt.clone());
                        searcher.iterative_deepening(&mut history);
                    })
                    .is_some()
                {}
            }));
        }
        Arc::new(Self { tx, handles })
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
