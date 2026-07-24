//! Checkpoint save/load with name-based parameter remapping.
//!
//! Parameters are keyed by name so adding or reordering evaluation terms
//! doesn't corrupt the load. [`peek_checkpoint`] reads the seed and dataset
//! fingerprint before the train/val split is set up.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufReader, BufWriter},
};

use serde::{Deserialize, Serialize};
use soul::{color, engine::eval_params::Tunable};

use crate::{
    core::{error::CheckpointError, fnv::Fnv1a},
    evaltune::palette,
};

pub const CHECKPOINT_VERSION: u32 = 5;

/// Parameters are keyed by name so adding or reordering evaluation terms
/// doesn't corrupt the load. The flat best-* vectors share the same ordering
/// as `param_names` and are remapped by name on resume.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    #[serde(default)]
    pub k_momentum: f64,
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    pub plateau_count: usize,
    pub params: BTreeMap<String, ParamState>,
    pub hash: u64,
    pub rng_seed: u64,
    pub dataset: u64,
    #[serde(default)]
    pub dataset_path: String,
    pub param_names: Vec<String>,
    pub best_val_params: Vec<f64>,
    pub best_train_params: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub struct ParamState {
    pub value: f64,
    pub momentum: f64,
    pub ema: f64,
    pub grad_ema: f64,
    pub stagnant: usize,
    pub frozen: bool,
}

/// Borrowed by [`save_checkpoint`]. Per-parameter slices indexed by `Tunable::idx`;
/// the save maps them to names.
pub struct TrainerState<'a> {
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    pub k_momentum: f64,
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    pub plateau_count: usize,
    pub rng_seed: u64,
    pub dataset: u64,
    pub dataset_path: &'a str,
    pub values: &'a [f64],
    pub momentum: &'a [f64],
    pub ema: &'a [f64],
    pub grad_ema: &'a [f64],
    pub stagnant: &'a [usize],
    pub frozen: &'a [bool],
    pub best_val_params: &'a [f64],
    pub best_train_params: &'a [f64],
}

/// # Errors
/// Returns an error if the file cannot be created or written.
pub fn save_checkpoint(path: &str, tunables: &[Tunable], state: &TrainerState) -> Result<(), CheckpointError> {
    let mut params = BTreeMap::new();

    for t in tunables {
        params.insert(t.name.clone(), ParamState {
            value: state.values[t.idx],
            momentum: state.momentum[t.idx],
            ema: state.ema[t.idx],
            grad_ema: state.grad_ema[t.idx],
            stagnant: state.stagnant[t.idx],
            frozen: state.frozen[t.idx],
        });
    }

    let mut param_names = vec![String::new(); tunables.len()];

    for t in tunables {
        param_names[t.idx] = t.name.clone();
    }

    let cp = Checkpoint {
        version: CHECKPOINT_VERSION,
        epoch: state.epoch,
        lr_scale: state.lr_scale,
        k: state.k,
        k_ref: state.k_ref,
        k_momentum: state.k_momentum,
        best_val_loss: state.best_val_loss,
        best_val_epoch: state.best_val_epoch,
        best_train_loss: state.best_train_loss,
        best_train_epoch: state.best_train_epoch,
        plateau_count: state.plateau_count,
        params,
        hash: compute_layout_hash(tunables),
        rng_seed: state.rng_seed,
        dataset: state.dataset,
        dataset_path: state.dataset_path.to_string(),
        param_names,
        best_val_params: state.best_val_params.to_vec(),
        best_train_params: state.best_train_params.to_vec(),
    };

    let tmp = format!("{path}.tmp");
    let file = File::create(&tmp)?;

    serde_json::to_writer(BufWriter::new(file), &cp)?;
    std::fs::rename(&tmp, path)?; // atomic on Linux
    Ok(())
}

/// Per-parameter state mapped back to current `Tunable::idx` order, ready for the trainer to adopt.
pub struct CheckpointData {
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    pub k_momentum: f64,
    pub best_val_loss: f64,
    pub best_val_epoch: usize,
    pub best_train_loss: f64,
    pub best_train_epoch: usize,
    pub plateau_count: usize,
    pub rng_seed: u64,
    pub dataset: u64,
    pub dataset_path: String,
    pub values: Vec<f64>,
    pub momentum: Vec<f64>,
    pub ema: Vec<f64>,
    pub grad_ema: Vec<f64>,
    pub stagnant: Vec<usize>,
    pub frozen: Vec<bool>,
    pub best_val_params: Vec<f64>,
    pub best_train_params: Vec<f64>,
}

// Missing parameters keep their `fallback` (current code value), so new tunables
// don't silently zero out on resume.
fn remap_flat_params(checkpoint_names: &[String], checkpoint_vals: &[f64], tunables: &[Tunable], fallback: &[f64]) -> Vec<f64> {
    let saved: BTreeMap<&str, f64> = checkpoint_names.iter().zip(checkpoint_vals).map(|(n, &v)| (n.as_str(), v)).collect();

    let mut out = fallback.to_vec();

    for t in tunables {
        if let Some(&v) = saved.get(t.name.as_str()) {
            out[t.idx] = v;
        }
    }

    out
}

/// A parameter missing from the checkpoint gets a fresh start: current code value, zero
/// momentum and gradient history, frozen only if the code says so.
///
/// # Errors
/// Returns an error if the file cannot be opened or parsed.
pub fn load_checkpoint(path: &str, tunables: &[Tunable], current_values: &[f64]) -> Result<CheckpointData, CheckpointError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let cp: Checkpoint = serde_json::from_reader(reader)?;

    if cp.version != CHECKPOINT_VERSION {
        eprintln!(
            "{}[!] Error: Checkpoint version mismatch! (Expected: {}, Found: {}){}",
            color::ansi_fg((225, 89, 91)),
            CHECKPOINT_VERSION,
            cp.version,
            palette::RESET,
        );

        return Err(io::Error::new(io::ErrorKind::InvalidData, "Checkpoint version mismatch").into());
    }

    let mut values = current_values.to_vec();
    let mut momentum = vec![0.0; values.len()];
    let mut ema = current_values.to_vec();
    let mut grad_ema = vec![0.0; values.len()];
    let mut stagnant = vec![0usize; values.len()];
    let mut frozen: Vec<bool> = tunables.iter().map(|t| t.is_fixed).collect();

    let current_hash = compute_layout_hash(tunables);

    if cp.hash != current_hash {
        eprintln!(
            "{}Warning: Checkpoint layout hash mismatch! (Saved: {:x}, Current: {:x}){}",
            color::ansi_fg((218, 165, 32)),
            cp.hash,
            current_hash,
            palette::RESET,
        );
        eprintln!("New parameters will use current default values.");
    }

    for t in tunables {
        if let Some(p) = cp.params.get(&t.name) {
            values[t.idx] = p.value;
            momentum[t.idx] = p.momentum;
            ema[t.idx] = p.ema;
            grad_ema[t.idx] = p.grad_ema;
            stagnant[t.idx] = p.stagnant;
            frozen[t.idx] = p.frozen || t.is_fixed;
        }
    }

    let best_val_params = remap_flat_params(&cp.param_names, &cp.best_val_params, tunables, &values);
    let best_train_params = remap_flat_params(&cp.param_names, &cp.best_train_params, tunables, &values);

    Ok(CheckpointData {
        epoch: cp.epoch,
        lr_scale: cp.lr_scale,
        k: cp.k,
        k_ref: cp.k_ref,
        k_momentum: cp.k_momentum,
        best_val_loss: cp.best_val_loss,
        best_val_epoch: cp.best_val_epoch,
        best_train_loss: cp.best_train_loss,
        best_train_epoch: cp.best_train_epoch,
        plateau_count: cp.plateau_count,
        rng_seed: cp.rng_seed,
        dataset: cp.dataset,
        dataset_path: cp.dataset_path.clone(),
        values,
        momentum,
        ema,
        grad_ema,
        stagnant,
        frozen,
        best_val_params,
        best_train_params,
    })
}

/// The train/val split happens before the full checkpoint load, and resuming
/// under a different seed reshuffles it: former training positions land in
/// val and the validation loss goes optimistic. The peek hands the shuffle
/// its seed, and the dataset fingerprint, before any of that runs.
///
/// # Errors
/// Returns an error if the file cannot be opened or parsed.
pub fn peek_checkpoint(path: &str) -> Result<Checkpoint, CheckpointError> {
    let file = File::open(path)?;
    let cp: Checkpoint = serde_json::from_reader(BufReader::new(file))?;

    Ok(cp)
}

/// FNV-1a hash over parameter names to detect layout changes.
pub fn compute_layout_hash(tunables: &[Tunable]) -> u64 {
    let mut fnv = Fnv1a::new();

    for t in tunables {
        fnv.write_bytes(t.name.as_bytes());
    }

    fnv.digest()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tunable(name: &str, idx: usize, value: f64, is_fixed: bool) -> Tunable {
        Tunable { name: name.into(), value, idx, is_fixed, freeze_resistant: false }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_trainer_state() {
        let tunables = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true)];

        let state = TrainerState {
            epoch: 42,
            lr_scale: 0.5,
            k: 1.23,
            k_ref: 1.11,
            k_momentum: 0.42,
            best_val_loss: 0.2,
            best_val_epoch: 18,
            best_train_loss: 0.3,
            best_train_epoch: 37,
            plateau_count: 3,
            rng_seed: 999,
            dataset: 777,
            dataset_path: "data/test.txt",
            values: &[10.0, 20.0],
            momentum: &[0.1, 0.2],
            ema: &[9.0, 19.0],
            grad_ema: &[0.01, 0.02],
            stagnant: &[4, 5],
            frozen: &[true, true],
            best_val_params: &[1.5, 2.5],
            best_train_params: &[3.5, 4.5],
        };

        let path = std::env::temp_dir().join(format!("soul_ckpt_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        // Load against a grown layout: gamma is new, so it must come back fresh.
        let grown = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true), tunable("gamma", 2, 30.0, false)];

        let d = load_checkpoint(path, &grown, &[1.0, 2.0, 30.0]).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!((d.epoch, d.lr_scale, d.k, d.k_ref, d.k_momentum), (42, 0.5, 1.23, 1.11, 0.42));
        assert_eq!(
            (d.best_val_loss, d.best_val_epoch, d.best_train_loss, d.best_train_epoch, d.plateau_count),
            (0.2, 18, 0.3, 37, 3)
        );
        assert_eq!((d.rng_seed, d.dataset), (999, 777));
        assert_eq!(d.dataset_path, "data/test.txt");
        assert_eq!(d.values, [10.0, 20.0, 30.0]);
        assert_eq!(d.momentum, [0.1, 0.2, 0.0]);
        assert_eq!(d.ema, [9.0, 19.0, 30.0]);
        assert_eq!(d.grad_ema, [0.01, 0.02, 0.0]);
        assert_eq!(d.stagnant, [4, 5, 0]);
        assert_eq!(d.frozen, [true, true, false]);
        assert_eq!(d.best_val_params, [1.5, 2.5, 30.0]);
        assert_eq!(d.best_train_params, [3.5, 4.5, 30.0]);
    }
}
