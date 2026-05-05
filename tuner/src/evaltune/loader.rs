use std::{
    fs::File,
    io::{BufRead, BufReader, Write},
    path::Path,
    time::Instant,
};

use soul::core::board::Position as Board;
pub use soul::tools::dataset::{SoulEntry, accumulate_gradient_cached, eval_soul_cached, load_encoded, save_encoded};

/// A raw EPD position with its game result (1.0 = white, 0.0 = black, 0.5 = draw).
pub struct Entry {
    pub board: Board,
    pub result: f64,
}

/// Loads a raw EPD dataset into a list of [`Entry`].
/// Supports both plain text and zstd-compressed EPD files.
pub fn load_epd(path: &str) -> std::io::Result<Vec<Entry>> {
    let file = File::open(path)?;
    let reader = open_reader(file, Path::new(path))?;

    let mut entries = Vec::new();

    for line in reader.lines() {
        let line = line?;
        if let Some((board, result)) = soul::tools::dataset::parse_epd_str(&line) {
            entries.push(Entry { board, result });
        }
    }

    Ok(entries)
}

/// Encodes an EPD file into a zstd-compressed Soul dataset.
/// Supports both plain text and zstd-compressed EPD input files.
///
/// # Errors
/// Returns an error if the input file cannot be read or the output cannot be written.
pub fn encode_epd(input: &str, output: &str) -> std::io::Result<()> {
    let file = File::open(input)?;
    let reader = open_reader(file, Path::new(input))?;

    let mut encoded = Vec::new();
    let mut last_print = Instant::now();

    println!("Parsing EPD positions...");

    for line in reader.lines() {
        let line = line?;
        let Some((board, result)) = soul::tools::dataset::parse_epd_str(&line) else {
            continue;
        };

        // Result is white-relative in EPD, we need STM-relative.
        let stm_result = if board.stm == soul::core::defs::Color::Black { 1.0 - result } else { result };
        encoded.push(SoulEntry::from_board(&board, stm_result, None, None));

        if last_print.elapsed().as_millis() > 500 {
            print!("\r\x1b[K  Processed {} positions...", encoded.len());
            let _ = std::io::stdout().flush();
            last_print = Instant::now();
        }
    }
    println!();

    let path = if output.ends_with(".zst") { output.to_string() } else { format!("{output}.zst") };

    println!("Writing encoded file: {path}");
    save_encoded(&path, &encoded)?;

    let orig_size = encoded.len() * std::mem::size_of::<SoulEntry>();
    let comp_size = std::fs::metadata(&path)?.len();
    let ratio = orig_size as f64 / comp_size as f64;

    println!("Done! {} entries ({orig_size} bytes → {comp_size} bytes, {ratio:.1}x compression)", encoded.len());
    println!("Entry size: {} bytes", std::mem::size_of::<SoulEntry>());

    Ok(())
}

/// Opens a file for buffered line reading, transparently decompressing zstd if needed.
fn open_reader(file: File, path: &Path) -> std::io::Result<Box<dyn BufRead>> {
    if path.extension().is_some_and(|e| e == "zst") {
        Ok(Box::new(BufReader::new(zstd::Decoder::new(file)?)))
    } else {
        Ok(Box::new(BufReader::new(file)))
    }
}
