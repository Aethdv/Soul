//! Serialization layer for `SoulEntry` training data.
//!
//! Two formats are supported:
//!   * `SoulEntry` — 32-byte nibble-array frames, zstd-compressed.
//!   * EPD text — one position per line, with game-result annotations.

use std::io::{self, Read, Write};

use zerocopy::IntoBytes;

use crate::{
    core::{board::Position, defs::Color},
    tools::dataset::SoulEntry,
};

pub const MAGIC_V5: &[u8; 8] = b"SOULENC5";
pub const MAGIC_V6: &[u8; 8] = b"SOULENC6";

const V5_SIZE: usize = 96;

// ──────── Binary codec ────────

/// Loads every [`SoulEntry`] from a zstd-compressed dataset.
///
/// V5 files are transparently upgraded on load — the legacy 96-byte entries
/// are converted to the 32-byte V6 nibble format. V6 files are read directly.
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
    let file = std::fs::File::open(path)?;
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
        } else if magic == *MAGIC_V5 {
            let mut buf = [0u8; 8];
            decoder.read_exact(&mut buf)?;
            let count = u64::from_le_bytes(buf) as usize;

            let mut v5_chunk = vec![0u8; count * V5_SIZE];
            decoder.read_exact(&mut v5_chunk)?;

            entries.reserve(count);
            for i in 0..count {
                entries.push(v5_to_v6(&v5_chunk[i * V5_SIZE..][..V5_SIZE]));
            }
        } else {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "Invalid magic in frame"));
        }
    }

    Ok(entries)
}

/// V5 → V6 conversion; directly construct the 32-byte nibble layout
/// from the legacy PackedPiece encoding without any intermediate FEN or
/// position reconstruction.
fn v5_to_v6(raw: &[u8]) -> SoulEntry {
    // V5 `repr(C)` layout, fields top to bottom:
    let result = f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]);
    let search_score = i16::from_le_bytes([raw[70], raw[71]]);
    let castling_stm = raw[90];
    let ep_square = raw[91];
    let original_stm = raw[89];
    let piece_count = raw[88] as usize;

    // Map each square to its nibble (pt | colour_bit) and build occupancy.
    let mut nibbles = [0u8; 64];
    let mut occupancy = 0u64;

    for i in 0..piece_count.min(32) {
        let off = 4 + i * 2;
        let p = u16::from_le_bytes([raw[off], raw[off + 1]]);
        let sq_val = (p & 0x3F) as u8;
        let upper = (p >> 6) as usize;
        let pt = upper & 0x07;
        if pt > 5 {
            continue;
        }
        let v5_color = upper & 0x08; // 0=Us/White, 8=Them/Black in V5 normalisation

        // Undo V5 STM-perspective normalisation.
        let mut sq = sq_val;
        if original_stm == 1 {
            sq ^= 0x38; // flip_rank
        }
        let real_black = if original_stm == 0 { v5_color != 0 } else { v5_color == 0 };
        let color_bit = if real_black { 0x08u8 } else { 0x00u8 };

        nibbles[sq as usize] = pt as u8 | color_bit;
        occupancy |= 1u64 << (sq as u64);
    }

    // Pack nibbles in occupancy-LSB order.
    let mut pieces = [0u8; 16];
    let mut occ = occupancy;
    let mut idx = 0usize;
    while occ != 0 {
        let sq = occ.trailing_zeros() as usize;
        occ &= occ - 1;
        pieces[idx / 2] |= nibbles[sq] << ((idx & 1) * 4);
        idx += 1;
    }

    // V5 castling is STM-relative; convert to absolute FEN byte.
    let castling = if original_stm == 0 { castling_stm } else { (castling_stm >> 2) | ((castling_stm & 0x3) << 2) };

    // V5 ep square is STM-relative (rank-flipped for Black); undo to absolute.
    let ep = if ep_square >= 64 || original_stm == 0 {
        ep_square
    } else {
        ep_square ^ 0x38
    };

    SoulEntry {
        occupancy,
        pieces,
        score: search_score,
        result: (f64::from(result) * 2.0) as u8,
        stm_and_ep: (original_stm << 7) | (ep & 0x7F),
        castling,
        _pad: [0u8; 3],
    }
}

/// Creates (or overwrites) a compressed dataset file.
pub fn save_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    write_frame(std::fs::File::create(path)?, entries)
}

/// Appends an independent compressed frame to a dataset file,
/// creating it if necessary.
pub fn append_encoded(path: &str, entries: &[SoulEntry]) -> io::Result<()> {
    let file = std::fs::OpenOptions::new().create(true).append(true).open(path)?;
    write_frame(file, entries)
}

/// Writes a single encoded frame (magic + count + payload) through a zstd compressor.
fn write_frame(writer: impl Write, entries: &[SoulEntry]) -> io::Result<()> {
    let mut enc = zstd::Encoder::new(writer, 3)?;
    enc.write_all(MAGIC_V6)?;
    enc.write_all(&(entries.len() as u64).to_le_bytes())?;
    enc.write_all(entries.as_bytes())?;
    enc.finish()?;
    Ok(())
}

// ──────── EPD text codec ────────

/// Parses a single EPD line into a `(Position, result)` pair.
///
/// Two notation families are recognised:
///
///   * Pipe-delimited — `fen | eval | wdl`. The third field is the
///     WDL outcome as a float (1.0 = white wins, 0.0 = black wins).
///   * Classic EPD — a FEN followed by a result token: `1-0`, `0-1`,
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

    // ── Classic EPD result detection ──
    let (result, fen_raw) = if let Some(stripped) = line.strip_suffix("1-0") {
        (1.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("0-1") {
        (0.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("1/2-1/2") {
        (0.5, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("1.0") {
        (1.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("0.0") {
        (0.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("0.5") {
        (0.5, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix(";w") {
        (1.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("; w") {
        (1.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix(";b") {
        (0.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("; b") {
        (0.0, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix(";d") {
        (0.5, stripped.to_string())
    } else if let Some(stripped) = line.strip_suffix("; d") {
        (0.5, stripped.to_string())
    } else {
        (0.5, line.to_string())
    };

    // Strip trailing EPD opcodes (everything past the first ';').
    let fen = fen_raw.split(';').next().unwrap_or(&fen_raw).trim();
    Position::try_from_fen(fen).ok().map(|board| (board, result))
}

/// Converts an EPD line directly into a [`SoulEntry`].
///
/// Flips the WDL result to the side-to-move perspective — the convention
/// the training pipeline expects.
pub fn parse_epd_entry(line: &str) -> Option<SoulEntry> {
    let (board, wdl) = parse_epd_str(line)?;
    let stm_wdl = if board.stm == Color::White { wdl } else { 1.0 - wdl };
    Some(SoulEntry::from_board(&board, stm_wdl, None, None))
}
