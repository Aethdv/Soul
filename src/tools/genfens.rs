//! Openings minted on demand, a fresh pool per run rather than one fixed list.
//!
//! `genfens N seed S book <path|None> <extra>` prints N lines of
//! `info string genfens <fen>` and exits. The `book` supplies the lines to play
//! out of, not the pool itself.

use fastrand::Rng;

use crate::{
    core::board::{Position, STARTPOS},
    engine::movegen::gen_legal_moves,
    tools::dataset::load_epd_fens,
};

const MARKER: &str = "info string genfens";

/// A fixed count fixes the side to move: an even number of plies out of the
/// start position always lands on White.
const MIN_PLIES: usize = 6;
const MAX_PLIES: usize = 9;

/// Mate or stalemate along the way ends an attempt. Past this many, the book has
/// nothing playable in it and a short count is the right answer.
const MAX_ATTEMPTS: usize = 100;

pub fn run(args: &[&str]) {
    let request = Request::parse(args);

    let Some(book) = request.book() else {
        eprintln!("genfens: no positions loaded from '{}'", request.book.as_deref().unwrap_or_default());
        return;
    };

    let mut rng = Rng::with_seed(request.seed);

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
    /// The runner passes `extra` through unread, so it carries arguments written
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

        load_epd_fens(path).ok().filter(|lines| !lines.is_empty())
    }
}

fn opening(book: &[String], rng: &mut Rng, plies: Option<usize>) -> Option<String> {
    (0..MAX_ATTEMPTS).find_map(|_| {
        let plies = plies.unwrap_or_else(|| rng.usize(MIN_PLIES..=MAX_PLIES));
        play_out(book, rng, plies)
    })
}

/// A random book line, `plies` random legal moves, and the position that results.
/// `None` if it mates or stalemates on the way or on arrival.
fn play_out(book: &[String], rng: &mut Rng, plies: usize) -> Option<String> {
    let mut pos = Position::from_fen(&book[rng.usize(..book.len())]);
    let mut acc = pos.get_initial_accumulator();

    for _ in 0..plies {
        let moves = gen_legal_moves(&pos);
        if moves.is_empty() {
            return None;
        }
        pos.make_move(moves[rng.usize(..moves.len())], &mut acc);
    }
    (!gen_legal_moves(&pos).is_empty()).then(|| pos.as_fen())
}

#[cfg(test)]
mod tests {
    use fastrand::Rng;

    use super::{Request, opening};
    use crate::{
        core::board::{Position, STARTPOS},
        engine::movegen::gen_legal_moves,
    };

    fn pool(seed: u64, count: usize) -> Vec<String> {
        let book = vec![STARTPOS.to_owned()];
        let mut rng = Rng::with_seed(seed);

        (0..count).filter_map(|_| opening(&book, &mut rng, None)).collect()
    }

    #[test]
    fn a_seed_reproduces_its_pool() {
        assert_eq!(pool(42, 8), pool(42, 8));
        assert_ne!(pool(42, 8), pool(43, 8), "neighboring seeds draw their own openings");
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
}
