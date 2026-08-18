//! EPD text parsing.
//!
//! EPD lines are read in two contexts:
//! - Position and result/eval labels extracted into [`SoulEntry`] training records.
//! - Raw or normalized FEN strings extracted for opening books.

use std::{
    fs,
    io::{self, BufRead},
};

use crate::{
    core::board::Position,
    tools::dataset::{SoulEntry, flip_score, flip_wdl},
};

/// Intermediate parsed representation of an EPD line with White-relative labels.
pub struct EpdEntry {
    pub board: Position,
    /// White-relative outcome: `1.0 = White win, 0.5 = draw, 0.0 = Black win`.
    pub result: f64,
    /// White-relative centipawn evaluation, if present.
    pub eval: Option<i32>,
}

/// Parses an EPD line supporting pipe-delimited or classic result notation.
///
/// Supported formats:
/// - Pipe format: `fen | score | wdl` (where `wdl` is a float in `[0.0, 1.0]`).
/// - Classic EPD: standard FEN followed by outcome tokens (`1-0`, `0-1`, `1/2-1/2`,
///   numeric floats `1.0`/`0.5`/`0.0`, or opcode shorthands `;w`, `;b`, `;d`).
///
/// Returned evaluations and outcomes are always White-relative.
pub fn parse_epd_str(line: &str) -> Option<EpdEntry> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    // Pipe format: "fen | score | wdl"
    if line.contains('|') {
        let mut fields = line.split('|').map(str::trim);
        // split always yields at least one element, whatever the line holds.
        let fen = fields.next().unwrap();
        let eval = fields.next().and_then(|e| e.parse::<i32>().ok());

        if let Some(wdl) = fields.next() {
            let result = wdl.parse::<f64>().ok()?;
            if let Ok(board) = Position::try_from_fen(fen) {
                return Some(EpdEntry { board, result, eval });
            }
        }
    }

    const RESULT_SUFFIXES: &[(&str, f64)] = &[
        ("1-0", 1.0),
        ("0-1", 0.0),
        ("1/2-1/2", 0.5),
        ("1.0", 1.0),
        ("0.0", 0.0),
        ("0.5", 0.5),
        (";w", 1.0),
        ("; w", 1.0),
        (";b", 0.0),
        ("; b", 0.0),
        (";d", 0.5),
        ("; d", 0.5),
    ];

    let (result, fen_raw) = RESULT_SUFFIXES
        .iter()
        .find_map(|&(suffix, val)| line.strip_suffix(suffix).map(|s| (val, s.to_string())))
        .unwrap_or((0.5, line.to_string()));

    // Strip trailing EPD operations
    let fen = fen_raw.split(';').next().unwrap_or(&fen_raw).trim();
    Position::try_from_fen(fen).ok().map(|board| EpdEntry { board, result, eval: None })
}

/// Parses an EPD line into a [`SoulEntry`], converting scores and outcomes to side-to-move perspective.
pub fn parse_epd_entry(line: &str) -> Option<SoulEntry> {
    let EpdEntry { board, result, eval } = parse_epd_str(line)?;
    let stm = board.stm;
    Some(SoulEntry::from_board(&board, flip_wdl(result, stm), eval.map(|e| flip_score(e, stm))))
}

/// Extracts normalized FEN strings from an EPD or raw FEN file.
/// Unparseable lines and comments are ignored.
pub fn load_epd_fens(path: &str) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut fens = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some(EpdEntry { board, .. }) = parse_epd_str(&line) {
            fens.push(board.as_fen());
        } else if Position::try_from_fen(&line).is_ok() {
            fens.push(line);
        }
    }
    Ok(fens)
}

/// Validates that outcome codes stay within the legal range (`0..=2`) to prevent underflow in perspective flips.
#[cfg(test)]
mod tests {
    use super::{SoulEntry, parse_epd_entry, parse_epd_str};

    #[test]
    fn black_to_move_epd_negates_eval() {
        let white = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 476 | 1.0";
        let black = "4k3/8/8/8/8/8/8/4K3 b - - 0 1 | 476 | 1.0";
        assert_eq!(parse_epd_str(white).expect("White to move").eval, Some(476));
        assert_eq!(parse_epd_str(black).expect("Black to move").eval, Some(476));
        assert_eq!(parse_epd_entry(white).expect("White entry").score, 476);
        assert_eq!(parse_epd_entry(black).expect("Black entry").score, -476);
    }

    #[test]
    fn epd_without_eval_sets_no_score_sentinel() {
        let entry = parse_epd_entry("4k3/8/8/8/8/8/8/4K3 w - - 0 1;d").expect("Classic EPD draw");
        assert_eq!(entry.score, SoulEntry::NO_SCORE);
        assert_eq!(entry.result, 1);
    }
}
