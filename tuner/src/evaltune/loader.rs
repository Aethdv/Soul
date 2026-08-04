//! EPD loading ([`Entry`], [`load_epd`]), viriformat, [`encode_epd`], and
//! [`load_datasets`] which dispatches by extension.
//!
//! Re-exports the tuner's datasource types ([`SoulEntry`], [`FeatureRecord`],
//! [`eval_record`]) through [`super::engine`].

use std::{
    fmt, fs,
    fs::File,
    io,
    io::{BufRead, BufReader, Write},
    iter, mem, path,
    path::Path,
    time::Instant,
};

pub use super::engine::{
    FeatureRecord, ReplayFilter, SoulEntry, accumulate_record_grad, eval_record, eval_record_full, flip_wdl, load_encoded,
    parse_epd_str, parse_viri_file, save_encoded,
};
use super::{
    engine::Position as Board,
    palette::{self, RESET},
};
use crate::core::fnv::Fnv1a;

/// A raw EPD position with its game result (1.0 = white, 0.0 = black, 0.5 = draw).
pub struct Entry {
    pub board: Board,
    pub result: f64,
}

/// Load raw EPD positions, decompressing zstd on the fly.
pub fn load_epd(path: &str) -> io::Result<Vec<Entry>> {
    let file = File::open(path)?;
    let reader = open_reader(file, Path::new(path))?;

    let mut entries = Vec::new();

    for line in reader.lines() {
        let line = line?;

        if let Some((board, result)) = parse_epd_str(&line) {
            entries.push(Entry { board, result });
        }
    }

    Ok(entries)
}

/// Encode EPD positions into a zstd-compressed Soul dataset.
///
/// Accepts both plain text and zstd-compressed EPD input.
///
/// # Errors
/// Returns an error if the input file cannot be read or the output cannot be written.
pub fn encode_epd(input: &str, output: &str) -> io::Result<()> {
    let file = File::open(input)?;
    let reader = open_reader(file, Path::new(input))?;

    let mut encoded = Vec::new();
    let mut last_print = Instant::now();

    println!("Parsing EPD positions...");

    for line in reader.lines() {
        let line = line?;
        let Some((board, result)) = parse_epd_str(&line) else {
            continue;
        };

        // Result is white-relative in EPD, we need STM-relative.
        encoded.push(SoulEntry::from_board(&board, flip_wdl(result, board.stm), None));

        if last_print.elapsed().as_millis() > 500 {
            print!("\r\x1b[K  Processed {} positions...", encoded.len());
            let _ = io::stdout().flush();
            last_print = Instant::now();
        }
    }
    println!();

    let path = if output.ends_with(".zst") { output.to_string() } else { format!("{output}.zst") };

    println!("Writing encoded file: {path}");
    save_encoded(&path, &encoded)?;

    let orig_size = encoded.len() * mem::size_of::<SoulEntry>();
    let comp_size = fs::metadata(&path)?.len();
    let ratio = orig_size as f64 / comp_size as f64;

    println!("Done! {} entries ({orig_size} bytes → {comp_size} bytes, {ratio:.1}x compression)", encoded.len());
    println!("Entry size: {} bytes", mem::size_of::<SoulEntry>());

    Ok(())
}

/// Load all dataset files by format, dispatching on extension.
///
/// `.soul` / `.soul.zst` → [`load_encoded`]; `.viri` / `.vf` → [`parse_viri_file`];
/// anything else → [`load_epd`] + [`SoulEntry::from_board`].
///
/// `filter` reaches the viriformat path alone; the other two store positions
/// without the ply or the played move its gates read.
///
/// The second return is one weight per position, empty unless a viriformat file
/// brought weighting gates. Files that bring none contribute ones, so the weights
/// stay aligned with the entries whatever the paths mix.
///
/// The third is the group sizes the split holds out together, summing to the entries: a game where
/// the format records one, a lone position where it does not. Split by position, a held-out ply
/// keeps its siblings in train, one move away and carrying the same label, and `L_val` flatters.
pub fn load_datasets(paths: &[String], filter: &ReplayFilter) -> (Vec<SoulEntry>, Vec<f32>, Vec<u32>) {
    let mut all_entries = Vec::new();
    let mut all_weights: Vec<f32> = Vec::new();
    let mut all_groups: Vec<u32> = Vec::new();

    for path in paths {
        let before = all_entries.len();
        let mut grouped = 0usize;

        if path.ends_with(".soul") || path.ends_with(".soul.zst") {
            println!("Loading encoded dataset: {path}");
            match load_encoded(path) {
                Ok(mut file_entries) => all_entries.append(&mut file_entries),
                Err(e) => bail_dataset(path, &e),
            }
        } else if path.ends_with(".viri") || path.ends_with(".vf") {
            println!("Loading viriformat dataset: {path}");
            match parse_viri_file(path, filter) {
                Ok((mut viri_entries, weights, games)) => {
                    if !weights.is_empty() {
                        all_weights.resize(all_entries.len(), 1.0);
                        all_weights.extend(weights);
                    }

                    all_entries.append(&mut viri_entries);
                    grouped = games.iter().sum::<u32>() as usize;
                    all_groups.extend(games);
                },
                Err(e) => bail_dataset(path, &e),
            }
        } else {
            println!("Loading raw dataset: {path}");
            match load_epd(path) {
                Ok(epd_entries) => {
                    for e in &epd_entries {
                        all_entries.push(SoulEntry::from_board(&e.board, flip_wdl(e.result, e.board.stm), None));
                    }
                },
                Err(e) => bail_dataset(path, &e),
            }
        }

        // A format that records no games contributes one group per position.
        all_groups.extend(iter::repeat_n(1u32, all_entries.len() - before - grouped));
    }

    if !all_weights.is_empty() {
        all_weights.resize(all_entries.len(), 1.0);
    }

    (all_entries, all_weights, all_groups)
}

/// Hashed before shuffle: identifies loaded contents, not a permutation.
/// A checkpoint's split seed replays the same split only over the same entries.
pub fn dataset_fingerprint(entries: &[SoulEntry]) -> u64 {
    let mut fnv = Fnv1a::new();
    fnv.write_bytes(&(entries.len() as u64).to_le_bytes());

    let stride = (entries.len() / 1024).max(1);

    for e in entries.iter().step_by(stride) {
        fnv.write_bytes(&e.occupancy.to_le_bytes());
        fnv.write_bytes(&e.score.to_le_bytes());
        fnv.write_bytes(&[e.result, e.stm_and_ep]);
    }

    fnv.digest()
}

pub fn resolve_dataset_paths(input: &str) -> Option<Vec<String>> {
    if input == "default" {
        let mut paths = Vec::new();

        if let Ok(entries) = fs::read_dir("data") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.to_string_lossy();

                if name.ends_with(".soul.zst") || name.ends_with(".soul") {
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
            .map(|s| {
                if path::Path::new(s).exists() {
                    s.to_string()
                } else {
                    let data_prefixed = format!("data/{s}");
                    if path::Path::new(&data_prefixed).exists() { data_prefixed } else { s.to_string() }
                }
            })
            .collect();

        Some(paths)
    }
}

// Opens a file for buffered line reading, transparently decompressing zstd if needed.
fn open_reader(file: File, path: &Path) -> io::Result<Box<dyn BufRead>> {
    if path.extension().is_some_and(|e| e == "zst") {
        Ok(Box::new(BufReader::new(zstd::Decoder::new(file)?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}

/// A file that cannot be read would train the run on less data than it was asked for, and the
/// shortfall reaches the fingerprint, the split and `L_val` with nothing left to say why.
fn bail_dataset(path: &str, e: &dyn fmt::Display) -> ! {
    eprintln!("{}[!] Cannot read dataset {path}: {e}{RESET}", palette::ALARM);
    std::process::exit(1);
}
