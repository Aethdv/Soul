//! Lazy SMP thread pool.
//!
//! Helpers are persistent, parked on an SPMC broadcast channel between
//! searches. `launch` sends one payload to all helpers; each receives it
//! inside a handler that runs the search, and the channel auto-signals
//! completion when the handler returns. `wait` blocks until every helper
//! has finished. `wake` (via `Drop`) shuts them down.

use std::{sync::Arc, thread};

use crate::{
    core::board::Position,
    engine::{
        history::History,
        search::{SearchConfig, Searcher},
        tt::TranspositionTable,
    },
    protocols::spmc,
};

pub struct LazySmpPool {
    tx: spmc::Sender<(SearchConfig, Position, Vec<u64>)>,
    handles: Vec<thread::JoinHandle<()>>,
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
                        let mut htable = History::new();
                        let mut ctx = Searcher::new(&local_cfg, board, history, tt.clone());
                        ctx.iterative_deepening(&mut htable);
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

        let mut hcfg = (*cfg).clone();

        hcfg.limits.silent = true;
        self.tx.send((hcfg, board, history.to_vec()));
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
