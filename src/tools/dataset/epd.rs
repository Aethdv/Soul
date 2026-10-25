//! EPD text parsing.

#[cfg(feature = "datagen")]
use std::{
    fs,
    io::{self, BufRead},
};

use crate::core::board::Position;

pub struct EpdEntry {
    pub board: Position,
    pub result: f64,
    /// White-relative centipawn evaluation, if present.
    pub eval: Option<i32>,
}

/// Parses a pipe-delimited `fen | score | wdl` line or a classic EPD line with an
/// outcome token. WDL must be `0.0`, `0.5`, or `1.0`; a bare FEN has no label and is rejected.
pub fn parse_epd_str(line: &str) -> Option<EpdEntry> {
    let line = line.trim();
    if line.is_empty() {
        return None;
    }

    if let Some((fen, rest)) = line.split_once('|') {
        let mut fields = rest.split('|').map(str::trim);
        let eval = match fields.next() {
            None | Some("") => None,
            Some(field) => Some(field.parse::<i32>().ok()?),
        };
        let result = fields.next()?.parse::<f64>().ok()?;
        if !valid_result(result) {
            return None;
        }

        return packed_entry(Position::try_from_fen(fen.trim()).ok()?, result, eval);
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
        .find_map(|&(suffix, val)| line.strip_suffix(suffix).map(|fen| (val, fen)))?;

    let fen = fen_raw.split(';').next().unwrap_or(fen_raw).trim();
    packed_entry(Position::try_from_fen(fen).ok()?, result, None)
}

fn valid_result(result: f64) -> bool { result == 0.0 || result == 0.5 || result == 1.0 }

fn packed_entry(board: Position, result: f64, eval: Option<i32>) -> Option<EpdEntry> {
    (board.occ.popcount() <= 32).then_some(EpdEntry { board, result, eval })
}

#[cfg(feature = "datagen")]
pub(crate) fn load_epd_fens(path: &str) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut fens = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some(EpdEntry { board, .. }) = parse_epd_str(&line) {
            fens.push(board.to_fen());
        } else if let Ok(board) = Position::try_from_fen(line.trim())
            && board.occ.popcount() <= 32
        {
            fens.push(board.to_fen());
        }
    }

    Ok(fens)
}

#[cfg(test)]
mod tests {
    use super::parse_epd_str;

    #[test]
    fn epd_eval_is_white_relative() {
        let white = "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 476 | 1.0";
        let black = "4k3/8/8/8/8/8/8/4K3 b - - 0 1 | 476 | 1.0";
        assert_eq!(parse_epd_str(white).expect("White to move").eval, Some(476));
        assert_eq!(parse_epd_str(black).expect("Black to move").eval, Some(476));
    }

    #[test]
    fn epd_without_eval_parses_as_none() {
        let entry = parse_epd_str("4k3/8/8/8/8/8/8/4K3 w - - 0 1;d").expect("Classic EPD draw");
        assert_eq!(entry.eval, None);
        assert_eq!(entry.result, 0.5);
    }

    #[test]
    fn epd_rejects_results_it_cannot_encode() {
        for wdl in ["0.75", "2.0", "-1.0", "NaN", "inf"] {
            assert!(parse_epd_str(&format!("4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 0 | {wdl}")).is_none());
        }
        assert!(parse_epd_str("4k3/8/8/8/8/8/8/4K3 w - - 0 1").is_none());
    }
}
