//! Dataset ingestion (EPD and viriformat), zstd streaming, and game-grouped holdout metadata.

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
    report::{ALARM, RESET},
    storage::fnv1a,
};

pub(crate) struct LoadedDataset {
    pub(crate) entries:        Vec<SoulEntry>,
    pub(crate) sample_weights: Vec<f32>,
    /// Positions per independent game span (`1` per position for ungrouped EPD files).
    pub(crate) group_sizes:    Vec<u32>,
}

#[inline]
pub(crate) fn is_viri_path(path: &str) -> bool { path.ends_with(".vf") || path.ends_with(".viri") }

pub(crate) fn load_datasets(paths: &[String], filter: &ReplayFilter) -> LoadedDataset {
    let mut entries = Vec::new();
    let mut sample_weights: Vec<f32> = Vec::new();
    let mut group_sizes: Vec<u32> = Vec::new();

    for path in paths {
        if is_viri_path(path) {
            println!("Loading viriformat dataset: {path}");
            match parse_viri_file(path, filter) {
                Ok((mut viri_entries, weights, games)) => {
                    if !weights.is_empty() {
                        sample_weights.resize(entries.len(), 1.0);
                        sample_weights.extend(weights);
                    }
                    entries.append(&mut viri_entries);
                    group_sizes.extend(games);
                },
                Err(err) => bail_dataset(path, &err),
            }
        } else {
            println!("Loading raw dataset: {path}");
            match load_epd(path) {
                Ok(epd_entries) => {
                    group_sizes.extend(iter::repeat_n(1u32, epd_entries.len()));
                    entries.extend(epd_entries.iter().map(|e| {
                        let stm = e.board.stm();
                        SoulEntry::from_board(&e.board, flip_wdl(e.result, stm), e.eval.map(|s| flip_score(s, stm)))
                    }));
                },
                Err(err) => bail_dataset(path, &err),
            }
        }
    }

    if !sample_weights.is_empty() {
        sample_weights.resize(entries.len(), 1.0);
    }
    LoadedDataset { entries, sample_weights, group_sizes }
}

/// Parallel FNV-1a digest of raw entry bytes before shuffling, guarding split consistency on resume.
pub(crate) fn dataset_fingerprint(entries: &[SoulEntry]) -> u64 {
    let digests: Vec<[u8; 8]> = entries.par_chunks(1 << 16).map(|chunk| fnv1a([chunk.as_bytes()]).to_le_bytes()).collect();
    let len = (entries.len() as u64).to_le_bytes();
    fnv1a(iter::once(&len[..]).chain(digests.iter().map(|d| &d[..])))
}

pub(crate) fn resolve_dataset_paths(input: &str) -> Option<Vec<String>> {
    if input != "default" {
        return Some(
            input
                .split(',')
                .map(str::trim)
                .map(|spec| {
                    let prefixed = format!("data/{spec}");
                    if !Path::new(spec).exists() && Path::new(&prefixed).exists() { prefixed } else { spec.to_string() }
                })
                .collect(),
        );
    }

    let mut paths: Vec<String> = fs::read_dir("data")
        .into_iter()
        .flatten()
        .flatten()
        .map(|e| e.path().to_string_lossy().into_owned())
        .filter(|name| is_viri_path(name))
        .collect();
    paths.sort();

    if paths.is_empty() {
        eprintln!("{ALARM}Error: No default dataset found in data/ directory.{RESET}");
        eprintln!("Please provide a dataset path using --dataset <path>");
        None
    } else {
        println!("Auto-discovered datasets: {}", paths.join(", "));
        Some(paths)
    }
}

fn load_epd(path: &str) -> io::Result<Vec<EpdEntry>> {
    let file = File::open(path)?;
    let reader: Box<dyn BufRead> = if Path::new(path).extension().is_some_and(|ext| ext == "zst") {
        Box::new(BufReader::new(zstd::Decoder::new(file)?))
    } else {
        Box::new(BufReader::new(file))
    };

    let mut entries = Vec::new();
    let mut unparsed = 0usize;

    for line in reader.lines() {
        let line = line?;
        if !line.trim().is_empty() {
            match parse_epd_str(&line) {
                Some(entry) => entries.push(entry),
                None => unparsed += 1,
            }
        }
    }

    if unparsed > 0 {
        alarm!("{path}: {unparsed} lines did not parse");
    }

    Ok(entries)
}

fn bail_dataset(path: &str, err: &dyn fmt::Display) -> ! {
    alarm!("Cannot read dataset {path}: {err}");
    std::process::exit(1);
}

#[cfg(test)]
mod tests {
    use std::{env, fs, process};

    use super::*;

    #[test]
    fn epd_labels_become_stm_relative() {
        let path = env::temp_dir().join(format!("soul-epd-labels-{}.epd", process::id()));
        fs::write(
            &path,
            "4k3/8/8/8/8/8/8/4K3 w - - 0 1 | 476 | 1.0\n\
             4k3/8/8/8/8/8/8/4K3 b - - 0 1 | 476 | 1.0\n\
             4k3/8/8/8/8/8/8/4K3 b - - 0 1;d\n",
        )
        .unwrap();
        let loaded = load_datasets(&[path.to_str().unwrap().to_owned()], &ReplayFilter::UNRESTRICTED);
        let _ = fs::remove_file(path);
        let labels: Vec<_> = loaded.entries.iter().map(|e| (e.score, e.result)).collect();
        assert_eq!(labels, [(476, 2), (-476, 0), (SoulEntry::NO_SCORE, 1)]);
    }

    #[test]
    fn edit_past_first_chunk_changes_fingerprint() {
        let entries = vec![SoulEntry::default(); 70_000];
        let mut edited = entries.clone();
        edited[69_000].castling_rights = 0b1010;
        assert_eq!(dataset_fingerprint(&entries), dataset_fingerprint(&entries));
        assert_ne!(dataset_fingerprint(&entries), dataset_fingerprint(&edited));
    }
}
