//! Serialization layer for `SoulEntry` training data.
//!
//! Two formats are supported:
//!   - `SoulEntry`; 32-byte nibble-array frames, zstd-compressed.
//!   - EPD text; one position per line, with game-result annotations.

use std::{
    fs,
    io::{self, Read, Write},
};

use zerocopy::IntoBytes;

use crate::{
    core::{board::Position, defs::Color},
    tools::dataset::SoulEntry,
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
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic in frame"));
        }
    }

    Ok(entries)
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
    let stm_wdl = if board.stm == Color::White { wdl } else { 1.0 - wdl };

    Some(SoulEntry::from_board(&board, stm_wdl, None, None))
}

fn write_frame(writer: impl Write, entries: &[SoulEntry]) -> io::Result<()> {
    let mut enc = zstd::Encoder::new(writer, 3)?;

    enc.write_all(MAGIC_V6)?;
    enc.write_all(&(entries.len() as u64).to_le_bytes())?;
    enc.write_all(entries.as_bytes())?;
    enc.finish()?;

    Ok(())
}
