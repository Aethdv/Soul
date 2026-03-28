//! Node counting and NPS benchmarking utility.
//!
//! Runs the search engine at fixed depths across a standard suite of positions
//! to measure raw search throughput and verify deterministic node counts.

use std::{
    io::{self, Write},
    sync::{Arc, atomic::AtomicBool},
    time::Instant,
};

use crate::{
    core::{board::Position, defs::Protocol},
    engine::{
        history,
        search::{Limits, SearchConfig, Searcher},
        search_params::SearchParams,
    },
};

const FENS: &str = include_str!("../data/bench.fens");

pub fn run(depth: u8) {
    let start = Instant::now();
    let mut total_nodes = 0;

    let stop_signal = Arc::new(AtomicBool::new(false));

    let fens: Vec<&str> = FENS.lines().collect();

    println!("Running bench on {} positions (Depth {})...", fens.len(), depth);
    io::stdout().flush().ok();

    let limits = Limits {
        depth,
        silent: true,
        protocol: Protocol::Uci,
        ..Default::default()
    };

    for (i, fen) in fens.iter().enumerate() {
        let board = Position::from_fen(fen);
        let history = vec![board.hash];

        print!("\rPosition {}/{}... ", i + 1, fens.len());
        io::stdout().flush().ok();

        let cfg = SearchConfig::new(
            limits.clone(),
            Instant::now(),
            stop_signal.clone(),
            0,
            SearchParams::default(),
        );
        let mut searcher = Searcher::new(&cfg, &board, &history, history::History::new());
        searcher.iterative_deepening();

        let nodes = searcher.nodes;
        total_nodes += nodes;

        println!(" {nodes} nodes");
        io::stdout().flush().ok();
    }

    let elapsed = start.elapsed();
    let nps = (total_nodes as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64;

    println!("Bench: {total_nodes} nodes {nps} nps");
}
