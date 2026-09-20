//! Openings drawn on demand, a fresh pool per run rather than one fixed list.
//!
//! `genfens N seed S book <path|None> <extra>` prints N lines of
//! `info string genfens <fen>` and exits. The `book` supplies the lines to play
//! out of, not the pool itself.

use std::{
    fs,
    io::{self, BufRead},
};

use crate::{
    core::{
        board::{Position, STARTPOS},
        util::{Rng, mulhi64},
    },
    engine::movegen::gen_legal_moves,
};

const MARKER: &str = "info string genfens";

const MIN_PLIES: usize = 6;
const MAX_PLIES: usize = 9;

/// Mate or stalemate along the way ends an attempt.
const MAX_ATTEMPTS: usize = 100;

pub fn run(args: &[&str]) {
    let request = Request::parse(args);
    let Some(book) = request.book() else {
        eprintln!("genfens: no positions loaded from '{}'", request.book.as_deref().unwrap_or_default());
        return;
    };

    let mut rng = Rng::new(request.seed);
    for _ in 0..request.count {
        let Some(fen) = opening(&book, &mut rng, request.plies) else {
            eprintln!("genfens: no playable opening in {MAX_ATTEMPTS} attempts");
            return;
        };
        println!("{MARKER} {fen}");
    }
}

struct Request {
    count: usize,
    seed: u64,
    book: Option<String>,
    plies: Option<usize>,
}

impl Request {
    /// `N seed S book <path|None> [extra]`, the count positional and the rest keyed.
    ///
    /// The runner passes `extra` through unread, so it contains arguments written
    /// for other engines. Skipping an unknown key alone leaves its value to be
    /// skipped in turn, so a lone flag cannot swallow the key behind it.
    fn parse(args: &[&str]) -> Self {
        let mut request = Self { count: 1, seed: 0, book: None, plies: None };
        if let Some(count) = args.first().and_then(|n| n.parse().ok()) {
            request.count = count;
        }

        let mut rest = args.iter().skip(1).copied();
        while let Some(key) = rest.next() {
            match key {
                "seed" => {
                    if let Some(seed) = rest.next().and_then(|v| v.parse().ok()) {
                        request.seed = seed;
                    }
                },
                "plies" => request.plies = rest.next().and_then(|v| v.parse().ok()),
                "book" => request.book = rest.next().filter(|&path| path != "None").map(str::to_owned),
                _ => {},
            }
        }
        request
    }

    /// A named book that loads nothing fails here rather than falling back to the
    /// start position, which would hand the run a pool nobody asked for.
    fn book(&self) -> Option<Vec<String>> {
        let Some(path) = self.book.as_deref() else {
            return Some(vec![STARTPOS.to_owned()]);
        };

        let mut fens = Vec::new();
        for line in io::BufReader::new(fs::File::open(path).ok()?).lines() {
            let line = line.ok()?;
            if Position::try_from_fen(&line).is_ok() {
                fens.push(line);
            }
        }

        Some(fens).filter(|fens| !fens.is_empty())
    }
}

fn opening(book: &[String], rng: &mut Rng, plies: Option<usize>) -> Option<String> {
    (0..MAX_ATTEMPTS).find_map(|_| {
        let plies = plies.unwrap_or_else(|| MIN_PLIES + mulhi64(rng.splitmix64(), MAX_PLIES - MIN_PLIES + 1));
        play_out(book, rng, plies)
    })
}

/// A random book line, `plies` random legal moves, and the position that results.
/// `None` if it mates or stalemates on the way or on arrival.
fn play_out(book: &[String], rng: &mut Rng, plies: usize) -> Option<String> {
    let mut pos = Position::from_fen(&book[mulhi64(rng.splitmix64(), book.len())]);
    let mut acc = pos.initial_accumulator();
    for _ in 0..plies {
        let moves = gen_legal_moves(&pos);
        if moves.is_empty() {
            return None;
        }
        pos.make_move(moves[mulhi64(rng.splitmix64(), moves.len())], &mut acc);
    }
    (!gen_legal_moves(&pos).is_empty()).then(|| pos.as_fen())
}

#[cfg(test)]
mod tests {
    use super::{Request, opening};
    use crate::{
        core::{
            board::{Position, STARTPOS},
            util::Rng,
        },
        engine::movegen::gen_legal_moves,
    };

    fn pool(seed: u64, count: usize) -> Vec<String> {
        let book = vec![STARTPOS.to_owned()];
        let mut rng = Rng::new(seed);
        (0..count).filter_map(|_| opening(&book, &mut rng, None)).collect()
    }

    #[test]
    fn a_seed_reproduces_its_pool() {
        assert_eq!(pool(42, 8), pool(42, 8));
        assert_ne!(pool(42, 8), pool(43, 8), "neighboring seeds draw their own openings");
        let zero = pool(0, 8);
        assert!(zero.iter().skip(1).any(|fen| fen != &zero[0]), "seed 0 repeats one opening");
    }

    #[test]
    fn every_opening_has_a_move_to_play() {
        let pool = pool(7, 32);
        assert_eq!(pool.len(), 32, "the whole count, none dropped");
        for fen in pool {
            let pos = Position::try_from_fen(&fen).expect("an emitted opening parses as a FEN");
            assert!(!gen_legal_moves(&pos).is_empty(), "{fen} has no move to play");
        }
    }

    #[test]
    fn both_sides_get_the_move() {
        let pool = pool(3, 32);
        assert!(pool.iter().any(|fen| fen.contains(" w ")), "no White to move in 32 openings");
        assert!(pool.iter().any(|fen| fen.contains(" b ")), "no Black to move in 32 openings");
    }

    #[test]
    fn the_runner_invocation_parses() {
        let request = Request::parse(&["8", "seed", "42", "book", "None", "depth", "6"]);
        assert_eq!(request.count, 8);
        assert_eq!(request.seed, 42);
        assert_eq!(request.book, None, "the literal None is no book");
        assert_eq!(request.plies, None, "an argument meant for another engine is skipped");
    }

    #[test]
    fn a_book_that_loads_nothing_has_no_openings() {
        let request = Request::parse(&["8", "seed", "42", "book", "/nonexistent.epd"]);
        assert_eq!(request.book(), None, "no falling back to the start position");
    }

    #[test]
    fn a_book_line_keeps_its_epd_opcodes() {
        let path = std::env::temp_dir().join(format!("soul_genfens_{}.epd", std::process::id()));
        let opcodes = "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - bm Nf3; id \"sicilian\";";
        std::fs::write(&path, format!("{opcodes}\nnot a position at all\n{STARTPOS}\n")).expect("writing the book");
        let request = Request::parse(&["1", "book", path.to_str().expect("a utf-8 temp path")]);
        let book = request.book().expect("an opcode line is a position");
        std::fs::remove_file(&path).ok();
        assert_eq!(book, vec![opcodes.to_owned(), STARTPOS.to_owned()]);
        assert_eq!(Position::from_fen(&book[0]).as_fen(), "rnbqkbnr/pp1ppppp/8/2p5/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1");
    }
}
