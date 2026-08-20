//! Checkpoint persistence with name-based parameter remapping.
//!
//! Parameters are serialized by name to maintain compatibility across layout additions,
//! removals, and reorderings. [`peek_checkpoint`] extracts seeds and dataset fingerprints
//! prior to dataset partitioning.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{self, BufReader, BufWriter, Write},
};

use serde::{Deserialize, Serialize};

use crate::{engine::Tunable, fnv::Fnv1a, training::Progress};

pub const CHECKPOINT_VERSION: u32 = 5;

/// Optimization state, sigmoid scale parameters, and loss progression metrics.
///
/// `k`, `k_ref` and `k_momentum` are adjacent `f64`s copied field by field at every hop, where a
/// transposed pair compiles and resumes onto the wrong scale.
#[derive(Clone, Serialize, Deserialize)]
pub struct RunState {
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    #[serde(default)]
    pub k_momentum: f64,
    #[serde(flatten)]
    pub progress: Progress,
}

/// Serialized checkpoint structure. The flat `best_*` vectors have no names of their own and are
/// remapped through `param_names` on resume.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    #[serde(flatten)]
    pub run: RunState,
    pub params: BTreeMap<String, ParamState>,
    pub hash: u64,
    pub rng_seed: u64,
    /// Seed used for the train/val split.
    #[serde(default)]
    pub split_seed: Option<u64>,
    pub dataset: u64,
    #[serde(default)]
    pub dataset_path: String,
    pub param_names: Vec<String>,
    pub best_val_params: Vec<f64>,
    pub best_train_params: Vec<f64>,
}

/// Optimizer and regularization state for a single parameter.
#[derive(Serialize, Deserialize)]
pub struct ParamState {
    pub value: f64,
    pub momentum: f64,
    pub ema: f64,
    pub grad_ema: f64,
    pub stagnant: usize,
    pub frozen: bool,
}

/// Trainer state slices borrowed for checkpoint serialization, each indexed by `Tunable::idx`.
pub struct TrainerState<'a> {
    pub run: RunState,
    pub rng_seed: u64,
    pub split_seed: u64,
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

/// Restored checkpoint state remapped to the current layout's parameter indices (`Tunable::idx`).
pub struct CheckpointData {
    pub run: RunState,
    pub values: Vec<f64>,
    pub momentum: Vec<f64>,
    pub ema: Vec<f64>,
    pub grad_ema: Vec<f64>,
    pub stagnant: Vec<usize>,
    pub frozen: Vec<bool>,
    pub best_val_params: Vec<f64>,
    pub best_train_params: Vec<f64>,
    /// Number of active parameters not present in the checkpoint and initialized from defaults.
    pub fresh_params: usize,
}

#[derive(Debug, thiserror::Error)]
pub enum CheckpointError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Checkpoint version {found}, expected {CHECKPOINT_VERSION}")]
    VersionMismatch { found: u32 },
    #[error("Layout renamed parameters: {0}. Start fresh, or check out the layout that wrote it")]
    LayoutRenamed(String),
}

/// Serializes trainer state to a temporary file and atomically moves it to `path`.
///
/// # Errors
/// Returns an error if serialization or atomic write operations fail.
pub fn save_checkpoint(path: &str, tunables: &[Tunable], state: &TrainerState) -> Result<(), CheckpointError> {
    let mut params = BTreeMap::new();
    let mut param_names = vec![String::new(); tunables.len()];

    for t in tunables {
        param_names[t.idx] = t.name.clone();
        params.insert(t.name.clone(), ParamState {
            value: state.values[t.idx],
            momentum: state.momentum[t.idx],
            ema: state.ema[t.idx],
            grad_ema: state.grad_ema[t.idx],
            stagnant: state.stagnant[t.idx],
            frozen: state.frozen[t.idx],
        });
    }

    let checkpoint = Checkpoint {
        version: CHECKPOINT_VERSION,
        run: state.run.clone(),
        params,
        hash: compute_layout_hash(tunables),
        rng_seed: state.rng_seed,
        split_seed: Some(state.split_seed),
        dataset: state.dataset,
        dataset_path: state.dataset_path.to_string(),
        param_names,
        best_val_params: state.best_val_params.to_vec(),
        best_train_params: state.best_train_params.to_vec(),
    };

    let tmp_path = format!("{path}.tmp");
    let mut writer = BufWriter::new(File::create(&tmp_path)?);
    serde_json::to_writer(&mut writer, &checkpoint)?;
    writer.flush()?;
    std::fs::rename(&tmp_path, path)?;
    Ok(())
}

/// Loads a checkpoint and remaps parameter states into contiguous index order.
///
/// Missing parameters inherit values from `current_values` and reset optimizer state.
///
/// # Errors
/// Returns [`CheckpointError`] if deserialization fails, versions mismatch, or if a
/// simultaneous addition and deletion indicates a parameter rename.
pub fn load_checkpoint(path: &str, tunables: &[Tunable], current_values: &[f64]) -> Result<CheckpointData, CheckpointError> {
    let checkpoint = peek_checkpoint(path)?;

    let mut values = current_values.to_vec();
    let mut momentum = vec![0.0; values.len()];
    let mut ema = current_values.to_vec();
    let mut grad_ema = vec![0.0; values.len()];
    let mut stagnant = vec![0usize; values.len()];
    let mut frozen: Vec<bool> = tunables.iter().map(|t| t.is_fixed).collect();

    let current_hash = compute_layout_hash(tunables);
    let mut fresh_params = 0;

    if checkpoint.hash != current_hash {
        let live_names: BTreeSet<&str> = tunables.iter().map(|t| t.name.as_str()).collect();
        let saved_names: BTreeSet<&str> = checkpoint.params.keys().map(String::as_str).collect();
        let dropped_names: Vec<&str> = saved_names.difference(&live_names).copied().collect();

        fresh_params = live_names.difference(&saved_names).count();

        // Simultaneous additions and removals suggest unmigrated parameter renames: the trained
        // values sit unreachable under the old names while the new ones resume at defaults.
        if !dropped_names.is_empty() && fresh_params > 0 {
            return Err(CheckpointError::LayoutRenamed(format!(
                "{} dropped ({}) while {fresh_params} added",
                dropped_names.len(),
                dropped_names[..dropped_names.len().min(3)].join(", ")
            )));
        }
    }

    for t in tunables {
        if let Some(p) = checkpoint.params.get(&t.name) {
            values[t.idx] = p.value;
            momentum[t.idx] = p.momentum;
            ema[t.idx] = p.ema;
            grad_ema[t.idx] = p.grad_ema;
            stagnant[t.idx] = p.stagnant;
            frozen[t.idx] = p.frozen || t.is_fixed;
        }
    }

    let best_val_params = remap_flat_params(&checkpoint.param_names, &checkpoint.best_val_params, tunables, &values);
    let best_train_params = remap_flat_params(&checkpoint.param_names, &checkpoint.best_train_params, tunables, &values);

    Ok(CheckpointData {
        run: checkpoint.run,
        values,
        momentum,
        ema,
        grad_ema,
        stagnant,
        frozen,
        best_val_params,
        best_train_params,
        fresh_params,
    })
}

/// Reads a checkpoint's seeds and dataset fingerprint before the train/val split is built.
///
/// A resume under a different split seed reshuffles that split, moving trained positions into val
/// and making the validation loss optimistic.
///
/// # Errors
/// Returns an error if the file cannot be parsed or if the format version mismatches.
pub fn peek_checkpoint(path: &str) -> Result<Checkpoint, CheckpointError> {
    let checkpoint: Checkpoint = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    if checkpoint.version != CHECKPOINT_VERSION {
        return Err(CheckpointError::VersionMismatch { found: checkpoint.version });
    }
    Ok(checkpoint)
}

/// Computes an FNV-1a digest over ordered parameter names to detect layout modifications.
fn compute_layout_hash(tunables: &[Tunable]) -> u64 {
    let mut fnv = Fnv1a::new();
    for t in tunables {
        fnv.write_bytes(t.name.as_bytes());
    }
    fnv.digest()
}

/// Remaps a dense parameter vector from serialized order to the active layout index order.
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tunable(name: &str, idx: usize, value: f64, is_fixed: bool) -> Tunable {
        Tunable { name: name.into(), value, idx, is_fixed, freeze_resistant: false }
    }

    fn temp_path(tag: &str) -> String {
        let name = format!("soul_ckpt_{tag}_{}.json", std::process::id());
        std::env::temp_dir().join(name).display().to_string()
    }

    fn saved_state() -> TrainerState<'static> {
        TrainerState {
            run: sample_run(),
            rng_seed: 1,
            split_seed: 3,
            dataset: 2,
            dataset_path: "data/test.txt",
            values: &[10.0],
            momentum: &[0.0],
            ema: &[10.0],
            grad_ema: &[0.0],
            stagnant: &[0],
            frozen: &[false],
            best_val_params: &[1.5],
            best_train_params: &[3.5],
        }
    }

    fn sample_run() -> RunState {
        RunState {
            epoch: 7,
            lr_scale: 1.0,
            k: 1.0,
            k_ref: 1.0,
            k_momentum: 0.0,
            progress: Progress {
                best_val_loss: 0.2,
                best_val_epoch: 3,
                val_smooth: 0.21,
                best_val_smooth: 0.205,
                best_train_loss: 0.3,
                best_train_epoch: 5,
                train_smooth: 0.31,
                best_train_smooth: 0.305,
                plateau_count: 0,
            },
        }
    }

    #[test]
    fn checkpoint_roundtrip_preserves_trainer_state() {
        let tunables = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true)];
        let state = TrainerState {
            run: RunState {
                epoch: 42,
                lr_scale: 0.5,
                k: 1.23,
                k_ref: 1.11,
                k_momentum: 0.42,
                progress: Progress {
                    best_val_loss: 0.2,
                    best_val_epoch: 18,
                    val_smooth: 0.21,
                    best_val_smooth: 0.205,
                    best_train_loss: 0.3,
                    best_train_epoch: 37,
                    train_smooth: 0.31,
                    best_train_smooth: 0.305,
                    plateau_count: 3,
                },
            },
            rng_seed: 999,
            split_seed: 888,
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

        let path = temp_path("roundtrip");
        save_checkpoint(&path, &tunables, &state).unwrap();

        let grown = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true), tunable("gamma", 2, 30.0, false)];
        let d = load_checkpoint(&path, &grown, &[1.0, 2.0, 30.0]).unwrap();
        let checkpoint = peek_checkpoint(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!((checkpoint.rng_seed, checkpoint.split_seed, checkpoint.dataset), (999, Some(888), 777));
        assert_eq!(checkpoint.dataset_path, "data/test.txt");
        assert_eq!((d.run.epoch, d.run.lr_scale, d.run.k, d.run.k_ref, d.run.k_momentum), (42, 0.5, 1.23, 1.11, 0.42));
        assert_eq!(
            (
                d.run.progress.best_val_loss,
                d.run.progress.best_val_epoch,
                d.run.progress.best_train_loss,
                d.run.progress.best_train_epoch,
                d.run.progress.plateau_count
            ),
            (0.2, 18, 0.3, 37, 3)
        );
        assert_eq!((d.run.progress.val_smooth, d.run.progress.best_val_smooth), (0.21, 0.205));
        assert_eq!((d.run.progress.train_smooth, d.run.progress.best_train_smooth), (0.31, 0.305));
        assert_eq!(d.values, [10.0, 20.0, 30.0]);
        assert_eq!(d.momentum, [0.1, 0.2, 0.0]);
        assert_eq!(d.ema, [9.0, 19.0, 30.0]);
        assert_eq!(d.grad_ema, [0.01, 0.02, 0.0]);
        assert_eq!(d.stagnant, [4, 5, 0]);
        assert_eq!(d.frozen, [true, true, false]);
        assert_eq!(d.best_val_params, [1.5, 2.5, 30.0]);
        assert_eq!(d.best_train_params, [3.5, 4.5, 30.0]);
        assert_eq!(d.fresh_params, 1);
    }

    #[test]
    fn an_unseeded_trail_resumes() {
        let tunables = [tunable("alpha", 0, 1.0, false)];
        let state = TrainerState {
            run: RunState { epoch: 20, lr_scale: 1.0, k: 1.0, k_ref: 1.0, k_momentum: 0.0, progress: Progress::default() },
            ..saved_state()
        };
        let path = temp_path("unseeded");
        save_checkpoint(&path, &tunables, &state).unwrap();
        let d = load_checkpoint(&path, &tunables, &[1.0]);
        let _ = std::fs::remove_file(&path);
        let progress = d.expect("unseeded trail must resume").run.progress;
        assert!(progress.val_smooth.is_nan(), "trail must resume as NaN");
        assert!(progress.train_smooth.is_nan());
    }

    #[test]
    fn a_rename_refuses_to_resume_but_a_removal_does_not() {
        let tunables = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true)];
        let state = TrainerState {
            values: &[10.0, 20.0],
            momentum: &[0.1, 0.2],
            ema: &[9.0, 19.0],
            grad_ema: &[0.01, 0.02],
            stagnant: &[0, 0],
            frozen: &[false, true],
            best_val_params: &[1.5, 2.5],
            best_train_params: &[3.5, 4.5],
            ..saved_state()
        };
        let path = temp_path("renamed");
        save_checkpoint(&path, &tunables, &state).unwrap();
        let renamed = [tunable("alpha", 0, 1.0, false), tunable("gamma", 1, 30.0, false)];
        let refused = load_checkpoint(&path, &renamed, &[1.0, 30.0]);
        let shrunk = [tunable("alpha", 0, 1.0, false)];
        let resumed = load_checkpoint(&path, &shrunk, &[1.0]);
        let _ = std::fs::remove_file(&path);
        assert!(refused.is_err(), "parameter rename must be rejected");
        let d = resumed.expect("parameter removal must resume remaining tunables");
        assert_eq!(d.values, [10.0]);
        assert_eq!(d.momentum, [0.1]);
        assert_eq!(d.fresh_params, 0);
    }

    #[test]
    fn checkpoint_without_smoothing_resumes_with_an_open_record() {
        let tunables = [tunable("alpha", 0, 1.0, false)];
        let state = saved_state();
        let path = temp_path("legacy");
        save_checkpoint(&path, &tunables, &state).unwrap();
        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let obj = raw.as_object_mut().unwrap();
        for key in ["val_smooth", "best_val_smooth", "train_smooth", "best_train_smooth"] {
            obj.remove(key);
        }

        std::fs::write(&path, serde_json::to_string(&raw).unwrap()).unwrap();
        let d = load_checkpoint(&path, &tunables, &[1.0]).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!(d.run.progress.val_smooth.is_nan());
        assert!(d.run.progress.train_smooth.is_nan());
        assert_eq!((d.run.progress.best_val_smooth, d.run.progress.best_train_smooth), (f64::MAX, f64::MAX));
        assert_eq!(d.run.progress.best_val_loss, 0.2);
    }

    #[test]
    fn a_checkpoint_without_a_split_seed_reads_as_none() {
        let tunables = [tunable("alpha", 0, 1.0, false)];
        let state = TrainerState { rng_seed: 12345, ..saved_state() };
        let path = temp_path("split");
        save_checkpoint(&path, &tunables, &state).unwrap();
        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        raw.as_object_mut().unwrap().remove("split_seed");
        std::fs::write(&path, serde_json::to_string(&raw).unwrap()).unwrap();
        let checkpoint = peek_checkpoint(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(checkpoint.split_seed, None);
        assert_eq!(checkpoint.rng_seed, 12345);
    }
}
