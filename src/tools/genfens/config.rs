//! Configuration parameters for self-play data generation.

use std::{
    fs::{File, read_to_string, rename},
    io::{Error, ErrorKind, Result, Write},
    path::Path,
};

pub const CONFIG_FILENAME: &str = "genfens_config.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenfensConfig {
    pub target_count: u64,
    pub output_path: String,
    pub book_paths: Vec<String>,
    pub depth: i32,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub resign_cp: i32,
    pub score_filter: i32,
    pub max_plies: usize,
    pub buffer_size: usize,
    pub thread_count: Option<usize>,
    pub save_interval: usize,
    pub filter_quiet: bool,
    pub sample_rate: f64,
    /// Skip positions before this ply count. 0 = disabled.
    pub min_ply: usize,
    /// Skip positions with fewer pieces than this. 0 = disabled.
    pub min_pieces: u32,
    /// Skip positions where search eval contradicts game outcome
    /// by more than this many centipawns. i32::MAX = disabled.
    pub eval_contradiction_limit: i32,
    /// Skip positions where |search_eval - static_eval| exceeds this
    /// threshold. Catches positions with unresolved tactics the HCE
    /// cannot learn. i32::MAX = disabled.
    pub qsearch_filter: i32,
    /// Use random-restart generation; each position is independently
    /// generated from a random book line or startpos + N random moves
    /// instead of playing full self-play game.
    pub random_restart: bool,
    /// Plies (half-moves) of random legal moves from the book position
    /// before running the verification search.
    pub random_plies: usize,
    /// Use the standard start position instead of book files.
    pub startpos: bool,
    pub generated_count: u64,
    pub last_update: i64,
}

impl Default for GenfensConfig {
    fn default() -> Self {
        Self {
            target_count: 8_000_000,
            output_path: "data.soul.zst".to_string(),
            book_paths: vec!["UHO_Lichess_4852_v1.epd".to_string()],
            depth: 6,
            soft_nodes: None,
            hard_nodes: None,
            resign_cp: 800,
            score_filter: 450,
            max_plies: 300,
            buffer_size: 256,
            thread_count: None,
            save_interval: 5000,
            filter_quiet: true,
            sample_rate: 0.7,
            min_ply: 0,
            min_pieces: 4,
            eval_contradiction_limit: i32::MAX,
            qsearch_filter: i32::MAX,
            random_restart: true,
            random_plies: 6,
            startpos: false,
            generated_count: 0,
            last_update: 0,
        }
    }
}

impl GenfensConfig {
    pub fn load() -> Result<Self> {
        let path = CONFIG_FILENAME;
        if !Path::new(path).exists() {
            return Ok(Self::default());
        }
        let content = read_to_string(path)?;
        serde_json::from_str(&content).map_err(|e| Error::new(ErrorKind::InvalidData, format!("Invalid config: {e}")))
    }

    pub fn save(&self) -> Result<()> {
        let tmp_path = format!("{CONFIG_FILENAME}.tmp");
        let content = serde_json::to_string_pretty(self)?;
        let mut tmp_file = File::create(&tmp_path)?;

        tmp_file.write_all(content.as_bytes())?;
        rename(&tmp_path, CONFIG_FILENAME)?;
        Ok(())
    }

    pub fn update_count(&mut self, count: u64) {
        self.generated_count = count;
        let _ = self.save();
    }
}
