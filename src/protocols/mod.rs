mod notation;
mod smp;
mod spmc;
pub mod uci;
pub mod xboard;

use std::io::{self, BufRead};

use crate::{
    core::{
        board::{Position, STARTPOS},
        moves::Move,
    },
    engine::history::History,
    weave::I16x8,
};

struct GameState {
    board:       Position,
    accumulator: I16x8,
    trail:       Vec<u64>,
    history:     Option<Box<History>>,
}

impl GameState {
    fn new() -> Self {
        let board = Position::from_fen(STARTPOS);
        let trail = vec![board.hash];
        Self { accumulator: board.psqt_accumulator(), board, trail, history: Some(Box::new(History::new())) }
    }

    fn load(&mut self, mut board: Position, frc_notation: bool) {
        board.frc_notation = frc_notation;
        self.board = board;
        self.accumulator = self.board.psqt_accumulator();
        self.trail.clear();
        self.trail.push(self.board.hash);
    }

    fn make_move(&mut self, mv: Move) {
        self.board.make_move(mv, &mut self.accumulator);
        self.trail.push(self.board.hash);
    }

    fn take_history(&mut self) -> Box<History> { self.history.take().unwrap_or_else(|| Box::new(History::new())) }

    fn clear_history(&mut self) {
        if let Some(history) = &mut self.history {
            history.clear();
        }
    }
}

pub fn read_line(reader: &mut impl BufRead, raw: &mut Vec<u8>) -> Option<String> {
    loop {
        raw.clear();
        match reader.read_until(b'\n', raw) {
            Ok(0) => return None,
            Ok(_) => return Some(clean_line(raw)),
            Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
            Err(_) => return None,
        }
    }
}

fn clean_line(raw: &[u8]) -> String {
    let mut out = Vec::with_capacity(raw.len());
    let mut i = 0;

    while i < raw.len() {
        match raw[i] {
            0x1B => {
                i += 1;
                match raw.get(i) {
                    Some(&b'[') => {
                        i += 1;
                        while raw.get(i).is_some_and(|b| (0x20..=0x3F).contains(b)) {
                            i += 1;
                        }
                        i += usize::from(raw.get(i).is_some_and(|b| (0x40..=0x7E).contains(b)));
                    },
                    Some(&b'O') => {
                        i += 1 + usize::from(raw.get(i + 1).is_some_and(|b| (0x40..=0x7E).contains(b)));
                    },
                    _ => {},
                }
            },
            b'\t' | b' ' => {
                out.push(raw[i]);
                i += 1;
            },
            0x00..=0x1F | 0x7F => i += 1,
            b => {
                out.push(b);
                i += 1;
            },
        }
    }

    String::from_utf8_lossy(&out).trim().to_owned()
}
