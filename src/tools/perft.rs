//! Performance test (perft) for move generator validation.
//!
//! Counts strictly legal moves to a given depth to verify movegen correctness.

use std::{
    io::{self, Write},
    time::Instant,
};

use crate::{core::board::Position, engine::movegen::gen_legal_moves, weave::Vi16x8};

pub fn run(board: &Position, depth: u8, divide: bool) {
    let start = Instant::now();
    let mut board_clone = *board;
    let mut acc = board_clone.get_initial_accumulator();

    if divide {
        println!("Perft Divide Depth {depth}");
        let moves = gen_legal_moves(&board_clone);
        let mut total_nodes = 0;

        for mv in &moves {
            let saved_acc = acc;
            let undo = board_clone.make_move(*mv, &mut acc);
            let nodes = perft(&mut board_clone, depth - 1, &mut acc);
            board_clone.unmake_move(*mv, &undo);
            acc = saved_acc;

            println!("{}: {nodes}", mv.to_uci(board_clone.is_frc));
            total_nodes += nodes;
        }
        let elapsed = start.elapsed();
        let nps = (total_nodes as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64;
        println!("\nNodes: {total_nodes}");
        println!("Time:  {elapsed:?}");
        println!("NPS:   {nps}");
    } else {
        let nodes = perft(&mut board_clone, depth, &mut acc);
        let elapsed = start.elapsed();
        let nps = (nodes as f64 / elapsed.as_secs_f64().max(0.000_001)) as u64;
        println!("nodes: {nodes} time: {elapsed:?} nps: {nps}");
    }
    io::stdout().flush().ok();
}

const BULK_COUNTING: bool = false;

pub fn perft(board: &mut Position, depth: u8, acc: &mut Vi16x8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = gen_legal_moves(board);

    if BULK_COUNTING && depth == 1 {
        return moves.len() as u64;
    }

    let mut nodes = 0;
    for mv in &moves {
        let saved_acc = *acc;
        let undo = board.make_move(*mv, acc);
        nodes += perft(board, depth - 1, acc);
        board.unmake_move(*mv, &undo);
        *acc = saved_acc;
    }
    nodes
}

#[cfg(test)]
mod tests {
    use super::perft;
    use crate::core::board::{Position, STARTPOS};

    #[test]
    fn perft_startpos_depth5() {
        let mut board = Position::from_fen(STARTPOS);
        let mut acc = board.get_initial_accumulator();
        assert_eq!(perft(&mut board, 5, &mut acc), 4_865_609);
    }

    #[test]
    fn perft_kiwipete_depth4() {
        let fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -";
        let mut board = Position::from_fen(fen);
        let mut acc = board.get_initial_accumulator();
        assert_eq!(perft(&mut board, 4, &mut acc), 4_085_603);
    }

    #[test]
    fn perft_position2_depth5() {
        let fen = "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -";
        let mut board = Position::from_fen(fen);
        let mut acc = board.get_initial_accumulator();
        assert_eq!(perft(&mut board, 5, &mut acc), 674_624);
    }

    #[test]
    fn perft_position3_depth5() {
        let fen = "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -";
        let mut board = Position::from_fen(fen);
        let mut acc = board.get_initial_accumulator();
        assert_eq!(perft(&mut board, 5, &mut acc), 15_833_292);
    }

    #[test]
    fn perft_depth_1_2() {
        let tests = [
            (STARTPOS, [20, 400]),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", [48, 2039]),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", [14, 191]),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", [6, 264]),
        ];

        for (fen, expected) in tests {
            let mut board = Position::from_fen(fen);
            let mut acc = board.get_initial_accumulator();
            assert_eq!(perft(&mut board, 1, &mut acc), expected[0], "Failed depth 1 for {fen}");
            assert_eq!(perft(&mut board, 2, &mut acc), expected[1], "Failed depth 2 for {fen}");
        }
    }
}
