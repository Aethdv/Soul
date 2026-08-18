//! Configuration parameters for self-play data generation.

use std::{
    fs::{File, read_to_string, rename},
    io::{Error, ErrorKind, Result, Write},
    path::Path,
};

pub const CONFIG_FILENAME: &str = "datagen_config.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DatagenConfig {
    pub target_count: u64,
    pub output_path: String,
    pub book_paths: Vec<String>,
    pub depth: i32,
    pub soft_nodes: Option<u64>,
    pub hard_nodes: Option<u64>,
    pub resign_cp: i32,
    pub max_plies: usize,
    pub thread_count: Option<usize>,
    pub save_interval: usize,
    pub startpos: bool,
    pub generated_count: u64,
    pub last_update: i64,
}

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            target_count: 8_000_000,
            output_path: "data.vf".to_string(),
            book_paths: vec!["UHO_Lichess_4852_v1.epd".to_string()],
            depth: 6,
            soft_nodes: None,
            hard_nodes: None,
            resign_cp: 800,
            max_plies: 300,
            thread_count: None,
            save_interval: 5000,
            startpos: false,
            generated_count: 0,
            last_update: 0,
        }
    }
}

impl DatagenConfig {
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
