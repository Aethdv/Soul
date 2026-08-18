//! Serialization for zstd-compressed dataset frames and EPD text formats.
//!
//! EPD lines are parsed in two contexts:
//! - Position and result/eval labels extracted into [`SoulEntry`] training records.
//! - Raw or normalized FEN strings extracted for opening books.

use std::{
    fs,
    io::{self, BufRead, Read, Write},
    mem,
};

use zerocopy::IntoBytes;

use crate::{
    core::board::Position,
    tools::dataset::{SoulEntry, flip_score, flip_wdl},
};

pub const MAGIC_V6: &[u8; 8] = b"SOULENC6";

/// Loads all [`SoulEntry`] records from a zstd-compressed dataset file.
///
/// Layout per frame:
/// ```text
/// ┌──────────┬──────────────┬────────────────────────────┐
/// │ 8B magic │ 8B LE count  │ count · sizeof(SoulEntry)  │
/// └──────────┴──────────────┴────────────────────────────┘
/// ```
///
/// Supports concatenated compressed frames produced by [`append_encoded`].
/// Decompresses directly into the target vector to avoid intermediate allocations.
pub fn load_encoded(path: &str) -> io::Result<Vec<SoulEntry>> {
    let file = fs::File::open(path)?;
    let mut decoder = zstd::Decoder::new(file)?;
    let mut entries = Vec::new();

    loop {
        let mut magic = [0u8; 8];
        if decoder.read_exact(&mut magic).is_err() {
            break;
        }

        if magic == *MAGIC_V6 {
            let mut count_bytes = [0u8; 8];
            decoder.read_exact(&mut count_bytes)?;

            let count = u64::from_le_bytes(count_bytes) as usize;
            let base = entries.len();

            entries.resize(base + count, SoulEntry::default());
            decoder.read_exact(entries[base..].as_mut_bytes())?;

            check_results(&entries[base..], base)?;
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic in frame"));
        }
    }
    Ok(entries)
}

/// Counts total records in a dataset by reading frame headers and skipping payloads.
pub fn count_encoded(path: &str) -> io::Result<usize> {
    let file = fs::File::open(path)?;
    let mut decoder = zstd::Decoder::new(file)?;
    let mut total = 0usize;

    loop {
        let mut magic = [0u8; 8];
        if decoder.read_exact(&mut magic).is_err() {
            break;
        }

        if magic != *MAGIC_V6 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic in frame"));
        }

        let mut count_bytes = [0u8; 8];
        decoder.read_exact(&mut count_bytes)?;

        let count = u64::from_le_bytes(count_bytes) as usize;
        let payload_bytes = (count * mem::size_of::<SoulEntry>()) as u64;
        let skipped = io::copy(&mut Read::by_ref(&mut decoder).take(payload_bytes), &mut io::sink())?;

        if skipped != payload_bytes {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame ended before its declared payload"));
        }
        total += count;
    }
    Ok(total)
}

/// Writes entries into a new (or truncated) zstd-compressed dataset file.
pub fn save_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    write_frame(fs::File::create(path)?, entries)
}

/// Appends entries as an independent compressed frame to a dataset file.
pub fn append_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    let file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    write_frame(file, entries)
}

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
fn check_results(entries: &[SoulEntry], base_idx: usize) -> io::Result<()> {
    match entries.iter().position(|e| e.result > 2) {
        Some(offset) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("entry {} has invalid result code {}", base_idx + offset, entries[offset].result),
        )),
        None => Ok(()),
    }
}

fn write_frame(writer: impl Write, entries: &[SoulEntry]) -> io::Result<()> {
    let mut enc = zstd::Encoder::new(writer, 3)?;
    enc.write_all(MAGIC_V6)?;
    enc.write_all(&(entries.len() as u64).to_le_bytes())?;
    enc.write_all(entries.as_bytes())?;
    enc.finish()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{SoulEntry, append_encoded, count_encoded, load_encoded, parse_epd_entry, parse_epd_str, save_encoded};
    use crate::core::board::{Position, STARTPOS};

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

    fn temp_path(tag: &str) -> String {
        std::env::temp_dir()
            .join(format!("soul_{tag}_{}.soul.zst", std::process::id()))
            .display()
            .to_string()
    }

    #[test]
    fn invalid_result_byte_aborts_load() {
        let good_path = temp_path("io_good");
        let bad_path = temp_path("io_bad");
        let board = Position::from_fen(STARTPOS);
        let mut entries = vec![SoulEntry::from_board(&board, 1.0, Some(30)); 4];

        save_encoded(&good_path, &entries).expect("Writing valid frame");
        assert_eq!(load_encoded(&good_path).expect("Reading valid frame").len(), 4);
        entries[2].result = 3;
        save_encoded(&bad_path, &entries).expect("Writing corrupted frame");

        let Err(err) = load_encoded(&bad_path) else {
            panic!("Expected Err on corrupted result code");
        };

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("entry 2"));
        let _ = std::fs::remove_file(&good_path);
        let _ = std::fs::remove_file(&bad_path);
    }

    #[test]
    fn count_encoded_matches_loaded_entry_count() {
        let path = temp_path("io_count");
        let board = Position::from_fen(STARTPOS);
        let entries = vec![SoulEntry::from_board(&board, 1.0, Some(30)); 4];
        save_encoded(&path, &entries).expect("Writing first frame");
        append_encoded(&path, &entries[..3]).expect("Appending second frame");
        assert_eq!(count_encoded(&path).expect("Counting records"), 7);
        assert_eq!(load_encoded(&path).expect("Loading records").len(), 7);
        let _ = std::fs::remove_file(&path);
    }
}
