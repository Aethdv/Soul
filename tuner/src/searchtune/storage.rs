//! Checkpoint serialization for CMA-ES state.

use std::{
    collections::BTreeMap,
    fs::{File, rename},
    io::{self, BufReader, BufWriter, ErrorKind, Write},
};

use serde::{Deserialize, Serialize};

use crate::core::error::CheckpointError;

pub const CHECKPOINT_FILE: &str = "searchtune_checkpoint.json";
pub const CHECKPOINT_VERSION: u32 = 1;

/// Checkpoint for CMA-ES state.
///
/// Engine tuning takes days. Hardware crashes, power flickers, and impatient humans
/// terminating the process are inevitable.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub epoch: usize,
    pub best_elo: f64,
    pub best_params: Vec<f64>,
    pub mean: Vec<f64>,
    pub sigma: f64,
    pub variances: Vec<f64>,
    pub p_sigma: Vec<f64>,
    pub p_c: Vec<f64>,
    /// Denormalized integer values.
    /// Intentionally redundant with `best_params` — derivable via `param.denormalize()`,
    /// but included for `jq`-friendliness without needing to rerun the engine.
    pub best_values: BTreeMap<String, i32>,
}

impl Checkpoint {
    pub fn save(&self) -> Result<(), CheckpointError> {
        let tmp_file = format!("{CHECKPOINT_FILE}.tmp");
        let mut writer = BufWriter::new(File::create(&tmp_file)?);

        serde_json::to_writer(&mut writer, self)?;
        Write::flush(&mut writer)?;
        rename(&tmp_file, CHECKPOINT_FILE)?;
        Ok(())
    }

    pub fn load() -> Result<Option<Self>, CheckpointError> {
        let file = match File::open(CHECKPOINT_FILE) {
            Ok(f) => f,
            Err(e) if e.kind() == ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(e.into()),
        };

        let reader = BufReader::new(file);
        let cp: Self = serde_json::from_reader(reader)?;

        if cp.version != CHECKPOINT_VERSION {
            return Err(io::Error::new(
                ErrorKind::InvalidData,
                format!("Checkpoint version mismatch! (Expected: {}, Found: {})", CHECKPOINT_VERSION, cp.version),
            )
            .into());
        }

        Ok(Some(cp))
    }
}
