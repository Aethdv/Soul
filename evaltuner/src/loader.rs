//! Dataset ingestion, format dispatch (EPD / viriformat), and validation grouping.
//!
//! Handles streaming decompression, sample weighting, and the dataset fingerprint a resume
//! checks its split against.

use std::{
    fmt,
    fs::{self, File},
    io::{self, BufRead, BufReader},
    iter,
    path::Path,
};

use rayon::prelude::*;
use zerocopy::IntoBytes;

use crate::{
    alarm,
    engine::{EpdEntry, ReplayFilter, SoulEntry, flip_score, flip_wdl, parse_epd_str, parse_viri_file},
    fnv::Fnv1a,
    palette::{self, RESET},
};

/// Loads raw EPD positions, transparently decompressing zstd streams if present.
pub fn load_epd(path: &str) -> io::Result<Vec<EpdEntry>> {
    let file = File::open(path)?;
    let reader = open_reader(file, Path::new(path))?;
    let mut entries = Vec::new();
    let mut unparsed = 0usize;

    for line in reader.lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match parse_epd_str(&line) {
            Some(entry) => entries.push(entry),
            None => unparsed += 1,
        }
    }

    // A truncated or mislabelled file otherwise loads short and trains without saying so.
    if unparsed > 0 {
        alarm!("{path}: {unparsed} lines did not parse");
    }
    Ok(entries)
}

/// - `.viri` / `.vf` → [`parse_viri_file`] with replay gating applied.
/// - Other extensions → [`load_epd`] converted to [`SoulEntry`].
///
/// # Returns
/// `(entries, sample_weights, group_sizes)`
/// - `sample_weights`: Per-position weights, padded to `1.0` if mixing weighted and unweighted sources.
/// - `group_sizes`: Partition spans held out contiguously during train/validation splits.
///   Viriformat keeps entire games grouped to prevent correlated adjacent plies from leaking
///   across the split and artificially deflating validation loss; EPDs default to single-position groups.
pub fn load_datasets(paths: &[String], filter: &ReplayFilter) -> (Vec<SoulEntry>, Vec<f32>, Vec<u32>) {
    let mut all_entries = Vec::new();
    let mut all_weights: Vec<f32> = Vec::new();
    let mut all_groups: Vec<u32> = Vec::new();

    for path in paths {
        if path.ends_with(".viri") || path.ends_with(".vf") {
            println!("Loading viriformat dataset: {path}");
            match parse_viri_file(path, filter) {
                Ok((mut viri_entries, weights, games)) => {
                    if !weights.is_empty() {
                        all_weights.resize(all_entries.len(), 1.0);
                        all_weights.extend(weights);
                    }
                    all_entries.append(&mut viri_entries);
                    all_groups.extend(games);
                },
                Err(err) => bail_dataset(path, &err),
            }
        } else {
            println!("Loading raw dataset: {path}");
            match load_epd(path) {
                Ok(epd_entries) => {
                    // No game boundaries in the format, so every position is its own partition.
                    all_groups.extend(iter::repeat_n(1u32, epd_entries.len()));
                    for entry in &epd_entries {
                        let stm = entry.board.stm;
                        all_entries.push(SoulEntry::from_board(
                            &entry.board,
                            flip_wdl(entry.result, stm),
                            entry.eval.map(|score| flip_score(score, stm)),
                        ));
                    }
                },
                Err(err) => bail_dataset(path, &err),
            }
        }
    }

    if !all_weights.is_empty() {
        all_weights.resize(all_entries.len(), 1.0);
    }
    (all_entries, all_weights, all_groups)
}

/// Computes a parallel digest of loaded dataset entries.
///
/// Hashes raw byte representations prior to shuffling, ensuring checkpointed split
/// seeds deterministically recreate the exact same train/validation partitions.
pub fn dataset_fingerprint(entries: &[SoulEntry]) -> u64 {
    // Fixed chunk size guarantees the digest is invariant to thread count.
    const CHUNK_SIZE: usize = 1 << 16;

    let digests: Vec<u64> = entries
        .par_chunks(CHUNK_SIZE)
        .map(|chunk| {
            let mut fnv = Fnv1a::new();
            fnv.write_bytes(chunk.as_bytes());
            fnv.digest()
        })
        .collect();

    let mut fnv = Fnv1a::new();
    fnv.write_bytes(&(entries.len() as u64).to_le_bytes());
    for digest in &digests {
        fnv.write_bytes(&digest.to_le_bytes());
    }
    fnv.digest()
}

/// Resolves comma-separated dataset paths or auto-discovers viriformat files in `data/`.
pub fn resolve_dataset_paths(input: &str) -> Option<Vec<String>> {
    if input == "default" {
        let mut paths = Vec::new();

        if let Ok(entries) = fs::read_dir("data") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.to_string_lossy();
                if name.ends_with(".vf") || name.ends_with(".viri") {
                    paths.push(name.to_string());
                }
            }
        }

        if paths.is_empty() {
            eprintln!("{}Error: No default dataset found in data/ directory.{RESET}", palette::ALARM);
            eprintln!("Please provide a dataset path using --dataset <path>");
            None
        } else {
            println!("Auto-discovered datasets: {}", paths.join(", "));
            Some(paths)
        }
    } else {
        let paths: Vec<String> = input
            .split(',')
            .map(str::trim)
            .map(|spec| {
                if Path::new(spec).exists() {
                    spec.to_string()
                } else {
                    let prefixed = format!("data/{spec}");
                    if Path::new(&prefixed).exists() { prefixed } else { spec.to_string() }
                }
            })
            .collect();

        Some(paths)
    }
}

fn open_reader(file: File, path: &Path) -> io::Result<Box<dyn BufRead>> {
    if path.extension().is_some_and(|ext| ext == "zst") {
        Ok(Box::new(BufReader::new(zstd::Decoder::new(file)?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// Aborts execution on dataset read errors to prevent training on silent data shortfalls.
fn bail_dataset(path: &str, err: &dyn fmt::Display) -> ! {
    alarm!("Cannot read dataset {path}: {err}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use super::{SoulEntry, dataset_fingerprint};

    #[test]
    fn an_edit_beyond_the_first_chunk_changes_the_fingerprint() {
        let entries = vec![SoulEntry::default(); 70_000];
        let mut edited = entries.clone();
        edited[69_000].castling = 0b1010;
        assert_eq!(dataset_fingerprint(&entries), dataset_fingerprint(&entries), "the same entries must hash the same");
        assert_ne!(
            dataset_fingerprint(&entries),
            dataset_fingerprint(&edited),
            "an edit past the first chunk must reach the digest"
        );
    }
}
