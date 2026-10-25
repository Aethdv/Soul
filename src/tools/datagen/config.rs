//! The datagen settings, saved in the working directory for a later resume.

use std::{
    fs::{File, read_to_string, rename},
    io::{Error, ErrorKind, Result, Write},
    path::Path,
};

pub(super) const CONFIG_FILENAME: &str = "datagen_config.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub(super) struct DatagenConfig {
    pub(super) target_count: u64,
    pub(super) output_path: String,
    pub(super) book_paths: Vec<String>,
    pub(super) depth: i32,
    pub(super) soft_nodes: Option<u64>,
    pub(super) hard_nodes: Option<u64>,
    pub(super) max_plies: usize,
    #[serde(default = "adjudication_enabled")]
    pub(super) adjudicate: bool,
    pub(super) thread_count: Option<usize>,
    pub(super) save_interval: usize,
    pub(super) startpos: bool,
}

fn adjudication_enabled() -> bool { true }

impl Default for DatagenConfig {
    fn default() -> Self {
        Self {
            target_count: 8_000_000,
            output_path: "data.vf".to_string(),
            book_paths: vec!["UHO_Lichess_4852_v1.epd".to_string()],
            depth: 6,
            soft_nodes: None,
            hard_nodes: None,
            max_plies: 300,
            adjudicate: true,
            thread_count: None,
            save_interval: 5000,
            startpos: false,
        }
    }
}

impl DatagenConfig {
    pub(super) fn load() -> Result<Option<Self>> {
        let path = Path::new(CONFIG_FILENAME);
        if !path.exists() {
            return Ok(None);
        }

        let content = read_to_string(path)?;
        let config =
            serde_json::from_str(&content).map_err(|e| Error::new(ErrorKind::InvalidData, format!("invalid config: {e}")))?;
        Ok(Some(config))
    }

    pub(super) fn save(&self) -> Result<()> {
        let tmp_path = format!("{CONFIG_FILENAME}.tmp");
        let content = serde_json::to_string_pretty(self)?;
        let mut tmp_file = File::create(&tmp_path)?;
        tmp_file.write_all(content.as_bytes())?;
        rename(&tmp_path, CONFIG_FILENAME)?;
        Ok(())
    }
}
