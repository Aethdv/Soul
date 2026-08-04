//! Serialization for Soul's own zstd-compressed frames, and the EPD text
//! parsers beside them.
//!
//! EPD arrives for two different jobs: a line carrying a game result becomes a
//! training entry; lines read for their FEN alone stock an opening book.

use std::{
    fs,
    io::{self, BufRead, Read, Write},
    mem,
};

use zerocopy::IntoBytes;

use crate::{
    core::board::Position,
    tools::dataset::{SoulEntry, flip_wdl},
};

pub const MAGIC_V6: &[u8; 8] = b"SOULENC6";

/// Loads every [`SoulEntry`] from a zstd-compressed dataset.
///
/// The binary layout for each frame is:
///
/// ```text
///   ┌──────────┬──────────────┬────────────────────────────┐
///   │ 8B magic │ 8B LE count  │ count · sizeof(SoulEntry)  │
///   └──────────┴──────────────┴────────────────────────────┘
/// ```
///
/// Files may contain multiple concatenated compressed frames (an append-only
/// consequence of repeated `append_encoded` calls). Each frame is decompressed
/// independently and collected into a single output `Vec`, growing in-place
/// rather than bouncing through an intermediate buffer.
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
            let mut buf = [0u8; 8];
            decoder.read_exact(&mut buf)?;

            let count = u64::from_le_bytes(buf) as usize;
            let base = entries.len();

            entries.resize(base + count, SoulEntry::default());
            decoder.read_exact(entries[base..].as_mut_bytes())?;

            check_results(&entries[base..], base)?;
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic in frame"));
        }
    }

    Ok(entries)
}

/// Positions in a dataset, without decoding any of them.
///
/// The same frame walk as [`load_encoded`], reading each count and skipping its payload, so an
/// append-only file reports everything it holds rather than what its first frame holds.
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
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic in frame"));
        }

        let mut buf = [0u8; 8];
        decoder.read_exact(&mut buf)?;

        let count = u64::from_le_bytes(buf) as usize;
        let payload = (count * mem::size_of::<SoulEntry>()) as u64;
        let skipped = io::copy(&mut Read::by_ref(&mut decoder).take(payload), &mut io::sink())?;

        if skipped != payload {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "frame ends before its count"));
        }

        total += count;
    }

    Ok(total)
}

/// Creates (or overwrites) a compressed dataset file.
pub fn save_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    write_frame(fs::File::create(path)?, entries)
}

/// Appends an independent compressed frame to a dataset file,
/// creating it if necessary.
pub fn append_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    let file = fs::OpenOptions::new().create(true).append(true).open(path)?;
    write_frame(file, entries)
}

/// Parses a single EPD line into a `(Position, result)` pair.
///
/// Two notation families are recognized:
///
///   - Pipe-delimited; `fen | eval | wdl`. The third field is the
///     WDL outcome as a float (1.0 = white wins, 0.0 = black wins).
///   - Classic EPD; a FEN followed by a result token: `1-0`, `0-1`,
///     `1/2-1/2`, numeric suffixes (`1.0`/`0.5`/`0.0`), or the terse
///     `;w`/`;b`/`;d` convention some tools emit.
///
/// The returned `f64` is always from White's perspective.
pub fn parse_epd_str(line: &str) -> Option<(Position, f64)> {
    let line = line.trim();

    if line.is_empty() {
        return None;
    }

    // Pipe-delimited: "fen | score | wdl"
    if line.contains('|') {
        let mut fields = line.split('|').map(str::trim);
        let fen = fields.next().unwrap(); // split always yields ≥ 1 element
        let _eval = fields.next(); // guaranteed present by the guard

        if let Some(wdl) = fields.next() {
            let result = wdl.parse::<f64>().ok()?;

            if let Ok(board) = Position::try_from_fen(fen) {
                return Some((board, result));
            }
        }
        // Fewer than three fields, or bad FEN → fall through to classic heuristics.
    }

    // Result detection
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

    // Strip trailing EPD opcodes (everything past the first ';').
    let fen = fen_raw.split(';').next().unwrap_or(&fen_raw).trim();
    Position::try_from_fen(fen).ok().map(|board| (board, result))
}

/// Converts an EPD line directly into a [`SoulEntry`].
///
/// Flips the WDL result to the side-to-move perspective: the convention
/// the training pipeline expects.
pub fn parse_epd_entry(line: &str) -> Option<SoulEntry> {
    let (board, wdl) = parse_epd_str(line)?;

    Some(SoulEntry::from_board(&board, flip_wdl(wdl, board.stm), None))
}

/// Loads opening positions from an EPD file, falling back to raw FEN parsing.
/// Both formats are common in the chess datagen ecosystem.
pub fn load_epd_fens(path: &str) -> io::Result<Vec<String>> {
    let file = fs::File::open(path)?;
    let reader = io::BufReader::new(file);
    let mut fens = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some((board, _)) = parse_epd_str(&line) {
            // EPD parsed successfully: re-export as FEN to normalize formatting.
            fens.push(board.as_fen());
        } else if Position::try_from_fen(&line).is_ok() {
            // Fallback: raw FEN line (no EPD operations field).
            fens.push(line);
        }
        // Lines that fail both parses are silently skipped,
        // they're comments, blank lines, or corrupted entries.
    }

    Ok(fens)
}

/// The result byte is whatever the file says. Past 2 it underflows [`flip_result`]
/// and reads as a win in the stats tally, so it stops at the read.
fn check_results(entries: &[SoulEntry], base: usize) -> io::Result<()> {
    match entries.iter().position(|e| e.result > 2) {
        Some(i) => Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("entry {} carries result {}, outside 0..=2", base + i, entries[i].result),
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
    use super::{SoulEntry, append_encoded, count_encoded, load_encoded, save_encoded};
    use crate::core::board::{Position, STARTPOS};

    fn temp(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("soul_{name}_{}.soul.zst", std::process::id()))
            .display()
            .to_string()
    }

    /// A dataset is bytes off a disk, and every consumer reads the result byte as
    /// one of three values.
    #[test]
    fn a_result_byte_past_two_fails_the_load() {
        let good = temp("io_good");
        let bad = temp("io_bad");

        let board = Position::from_fen(STARTPOS);
        let mut entries = vec![SoulEntry::from_board(&board, 1.0, Some(30)); 4];

        save_encoded(&good, &entries).expect("writing a good frame");
        assert_eq!(load_encoded(&good).expect("reading it back").len(), 4);

        entries[2].result = 3;
        save_encoded(&bad, &entries).expect("writing the tampered frame");

        // expect_err would need SoulEntry: Debug to format the Ok side.
        let Err(err) = load_encoded(&bad) else {
            panic!("a result past 2 must fail the load");
        };

        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("entry 2"), "the error must name the entry: {err}");

        let _ = std::fs::remove_file(&good);
        let _ = std::fs::remove_file(&bad);
    }

    #[test]
    fn counting_frames_matches_loading_them() {
        let path = temp("io_count");
        let board = Position::from_fen(STARTPOS);
        let entries = vec![SoulEntry::from_board(&board, 1.0, Some(30)); 4];

        save_encoded(&path, &entries).expect("writing the first frame");
        append_encoded(&path, &entries[..3]).expect("appending a second");

        assert_eq!(count_encoded(&path).expect("counting the frames"), 7);
        assert_eq!(load_encoded(&path).expect("loading the frames").len(), 7);

        let _ = std::fs::remove_file(&path);
    }
}
