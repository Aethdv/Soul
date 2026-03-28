//! Configuration parameters for self-play data generation.

use std::{fs::File, io::Write, path::Path};

pub const CONFIG_FILENAME: &str = "genfens_config.json";

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GenfensConfig {
    pub target_count:    u64,
    pub output_path:     String,
    pub book_paths:      Vec<String>,
    pub depth:           u8,
    pub soft_nodes:      Option<u64>,
    pub hard_nodes:      Option<u64>,
    pub resign_cp:       i32,
    pub score_filter:    i32,
    pub max_plies:       usize,
    pub buffer_size:     usize,
    pub thread_count:    Option<usize>,
    pub save_interval:   usize,
    pub filter_quiet:    bool,
    pub sample_rate:     f64,
    pub generated_count: u64,
    pub last_update:     i64,
}

impl Default for GenfensConfig {
    fn default() -> Self {
        Self {
            target_count:    8_000_000,
            output_path:     "data.soul.zst".to_string(),
            book_paths:      vec!["UHO_Lichess_4852_v1.epd".to_string()],
            depth:           6,
            soft_nodes:      None,
            hard_nodes:      None,
            resign_cp:       800,
            score_filter:    450,
            max_plies:       300,
            buffer_size:     256,
            thread_count:    None,
            save_interval:   5000,
            filter_quiet:    true,
            sample_rate:     0.7,
            generated_count: 0,
            last_update:     0,
        }
    }
}

impl GenfensConfig {
    pub fn load() -> std::io::Result<Self> {
        let path = CONFIG_FILENAME;
        if !Path::new(path).exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(path)?;
        serde_json::from_str(&content)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("Invalid config: {e}")))
    }

    pub fn save(&self) -> std::io::Result<()> {
        let tmp_path = format!("{CONFIG_FILENAME}.tmp");
        let content = serde_json::to_string_pretty(self)?;
        let mut tmp_file = File::create(&tmp_path)?;

        tmp_file.write_all(content.as_bytes())?;
        std::fs::rename(&tmp_path, CONFIG_FILENAME)?;
        Ok(())
    }

    pub fn update_count(&mut self, count: u64) {
        self.generated_count = count;
        let _ = self.save();
    }
}

#[derive(Debug, Clone)]
pub struct GenfensArgs {
    pub target_count:  u64,
    pub output_path:   String,
    pub book_paths:    Vec<String>,
    pub depth:         u8,
    pub soft_nodes:    Option<u64>,
    pub hard_nodes:    Option<u64>,
    pub resign_cp:     i32,
    pub score_filter:  i32,
    pub max_plies:     usize,
    pub buffer_size:   usize,
    pub thread_count:  Option<usize>,
    pub save_interval: usize,
    pub filter_quiet:  bool,
    pub sample_rate:   f64,
    pub resume:        bool,
}

impl From<GenfensArgs> for GenfensConfig {
    fn from(args: GenfensArgs) -> Self {
        Self {
            target_count:    args.target_count,
            output_path:     args.output_path,
            book_paths:      args.book_paths,
            depth:           args.depth,
            soft_nodes:      args.soft_nodes,
            hard_nodes:      args.hard_nodes,
            resign_cp:       args.resign_cp,
            score_filter:    args.score_filter,
            max_plies:       args.max_plies,
            buffer_size:     args.buffer_size,
            thread_count:    args.thread_count,
            save_interval:   args.save_interval,
            filter_quiet:    args.filter_quiet,
            sample_rate:     args.sample_rate,
            generated_count: 0,
            last_update:     0,
        }
    }
}
