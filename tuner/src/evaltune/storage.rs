use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufReader, BufWriter},
};

use serde::{Deserialize, Serialize};
use soul::engine::eval_params::Tunable;

use crate::core::error::CheckpointError;

pub const CHECKPOINT_VERSION: u32 = 2;

/// Serialisable training checkpoint: everything needed to resume a run.
///
/// Parameters are keyed by name to ensure robustness against layout changes
/// (e.g. adding or reordering evaluation terms).
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub epoch: usize,
    #[serde(default = "default_lr_scale")]
    pub lr_scale: f64,
    pub values: BTreeMap<String, f64>,
    pub momentum: BTreeMap<String, f64>,
    pub hash: u64,
    pub rng_seed: u64,
}

/// A frozen parameter snapshot at a specific epoch.
#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub epoch: usize,
    pub params: BTreeMap<String, f64>,
    pub error: f64,
}

/// Save training state to a JSON checkpoint file.
///
/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn save_checkpoint(
    path: &str,
    epoch: usize,
    lr_scale: f64,
    values: &[f64],
    momentum: &[f64],
    tunables: &[Tunable],
    rng_seed: u64,
) -> Result<(), CheckpointError> {
    let mut val_map = BTreeMap::new();
    let mut mom_map = BTreeMap::new();

    for t in tunables {
        val_map.insert(t.name.clone(), values[t.idx]);
        mom_map.insert(t.name.clone(), momentum[t.idx]);
    }

    let cp = Checkpoint {
        version: CHECKPOINT_VERSION,
        epoch,
        lr_scale,
        values: val_map,
        momentum: mom_map,
        hash: compute_layout_hash(tunables),
        rng_seed,
    };

    let tmp = format!("{path}.tmp");
    let file = File::create(&tmp)?;
    serde_json::to_writer(BufWriter::new(file), &cp)?;
    std::fs::rename(&tmp, path)?; // atomic on Linux
    Ok(())
}

/// Load training state from a JSON checkpoint file.
///
/// Maps saved parameter names back to their current indices. New parameters
/// (missing from checkpoint) default to 0.0 momentum and current code values.
///
/// # Errors
/// Returns an error if the file cannot be opened or parsed.
pub struct CheckpointData {
    pub epoch: usize,
    pub lr_scale: f64,
    pub values: Vec<f64>,
    pub momentum: Vec<f64>,
    pub rng_seed: u64,
}

pub fn load_checkpoint(path: &str, tunables: &[Tunable], current_values: &[f64]) -> Result<CheckpointData, CheckpointError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let cp: Checkpoint = serde_json::from_reader(reader)?;
    if cp.version != CHECKPOINT_VERSION {
        eprintln!(
            "\x1b[91m[!] Error: Checkpoint version mismatch! (Expected: {}, Found: {})\x1b[0m",
            CHECKPOINT_VERSION, cp.version
        );
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Checkpoint version mismatch").into());
    }

    let mut values = current_values.to_vec();
    let mut momentum = vec![0.0; values.len()];

    let current_hash = compute_layout_hash(tunables);
    if cp.hash != current_hash {
        eprintln!("\x1b[33mWarning: Checkpoint layout hash mismatch! (Saved: {:x}, Current: {:x})\x1b[0m", cp.hash, current_hash);
        eprintln!("New parameters will use current default values.");
    }

    for t in tunables {
        if let Some(&v) = cp.values.get(&t.name) {
            values[t.idx] = v;
        }
        if let Some(&m) = cp.momentum.get(&t.name) {
            momentum[t.idx] = m;
        }
    }

    Ok(CheckpointData { epoch: cp.epoch, lr_scale: cp.lr_scale, values, momentum, rng_seed: cp.rng_seed })
}

/// FNV-1a hash over parameter names to detect layout changes.
pub fn compute_layout_hash(tunables: &[Tunable]) -> u64 {
    let mut fnv = crate::core::fnv::Fnv1a::new();
    for t in tunables {
        fnv.write_bytes(t.name.as_bytes());
    }
    fnv.digest()
}

/// Snapshot hall of fame: keep the N best checkpoints by validation loss.
pub fn update_snapshots(
    snapshots: &mut Vec<Snapshot>,
    epoch: usize,
    values: &[f64],
    tunables: &[Tunable],
    error: f64,
    limit: usize,
) -> bool {
    let mut params = BTreeMap::new();
    for t in tunables {
        params.insert(t.name.clone(), values[t.idx]);
    }

    let snap = Snapshot { epoch, params, error };

    let admitted = if snapshots.len() < limit {
        snapshots.push(snap);
        true
    } else if error < snapshots.last().unwrap().error {
        *snapshots.last_mut().unwrap() = snap; // replace worst in-place
        true
    } else {
        false
    };

    if admitted {
        let last_idx = snapshots.len() - 1;
        let new_err = snapshots[last_idx].error;
        let pos = snapshots[..last_idx].partition_point(|s| s.error <= new_err);
        snapshots[pos..].rotate_right(1);
    }

    admitted
}

fn default_lr_scale() -> f64 {
    1.0
}
