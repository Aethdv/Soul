//! Checkpoint save/load with name-based parameter remapping.
//!
//! Parameters are keyed by name so adding or reordering evaluation terms
//! doesn't corrupt the load. [`peek_checkpoint`] reads the seed and dataset
//! fingerprint before the train/val split is set up.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{self, BufReader, BufWriter},
};

use serde::{Deserialize, Serialize};

use super::{engine::Tunable, training::Progress};
use crate::core::{error::CheckpointError, fnv::Fnv1a};

pub const CHECKPOINT_VERSION: u32 = 5;

/// `k`, `k_ref` and `k_momentum` are three `f64` meaning the live scale, the frozen
/// reference and the scale's momentum. Copied field by field at every hop, a transposed
/// pair compiles and resumes onto the wrong one.
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

/// The flat best-* vectors carry no names of their own; they share `param_names`
/// ordering and are remapped through it on resume.
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    #[serde(flatten)]
    pub run: RunState,
    pub params: BTreeMap<String, ParamState>,
    pub hash: u64,
    pub rng_seed: u64,
    /// None in a checkpoint from when `rng_seed` drew the val slice as well. A resume reads it as
    /// `rng_seed` there, or it validates on positions the checkpoint trained on.
    #[serde(default)]
    pub split_seed: Option<u64>,
    pub dataset: u64,
    #[serde(default)]
    pub dataset_path: String,
    pub param_names: Vec<String>,
    pub best_val_params: Vec<f64>,
    pub best_train_params: Vec<f64>,
}

/// Everything the optimizer holds about one parameter, keyed by its name.
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

/// Per-parameter state mapped back to current `Tunable::idx` order, ready for the trainer to adopt.
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
    /// Current tunables the checkpoint never held, resuming from code defaults.
    pub fresh_params: usize,
}

/// # Errors
/// Returns an error if the file cannot be created or written.
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

    let cp = Checkpoint {
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

    let tmp = format!("{path}.tmp");
    let file = File::create(&tmp)?;

    serde_json::to_writer(BufWriter::new(file), &cp)?;
    std::fs::rename(&tmp, path)?; // atomic on Linux
    Ok(())
}

/// A parameter missing from the checkpoint gets a fresh start: current code value, zero
/// momentum and gradient history, frozen only if the code says so.
///
/// # Errors
/// Returns an error if the file cannot be opened or parsed, or if the layout
/// renamed a parameter it holds.
pub fn load_checkpoint(path: &str, tunables: &[Tunable], current_values: &[f64]) -> Result<CheckpointData, CheckpointError> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let cp: Checkpoint = serde_json::from_reader(reader)?;

    if cp.version != CHECKPOINT_VERSION {
        let mismatch = format!("checkpoint is version {}, this build writes {CHECKPOINT_VERSION}", cp.version);

        return Err(io::Error::new(io::ErrorKind::InvalidData, mismatch).into());
    }

    let mut values = current_values.to_vec();
    let mut momentum = vec![0.0; values.len()];
    let mut ema = current_values.to_vec();
    let mut grad_ema = vec![0.0; values.len()];
    let mut stagnant = vec![0usize; values.len()];
    let mut frozen: Vec<bool> = tunables.iter().map(|t| t.is_fixed).collect();

    let current_hash = compute_layout_hash(tunables);

    let mut fresh_params = 0;

    if cp.hash != current_hash {
        let live: BTreeSet<&str> = tunables.iter().map(|t| t.name.as_str()).collect();
        let saved: BTreeSet<&str> = cp.params.keys().map(String::as_str).collect();
        let dropped: Vec<&str> = saved.difference(&live).copied().collect();

        fresh_params = live.difference(&saved).count();

        // Names leaving while others arrive is a rename: the trained values sit
        // unreachable under the old names while the new ones resume at defaults.
        if !dropped.is_empty() && fresh_params > 0 {
            return Err(CheckpointError::LayoutRenamed(format!(
                "{} left and {fresh_params} arrived, starting with {}",
                dropped.len(),
                dropped[..dropped.len().min(3)].join(", "),
            )));
        }
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
        run: cp.run,
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

/// The train/val split happens before the full checkpoint load, and resuming
/// under a different split seed reshuffles it: former training positions land
/// in val and the validation loss goes optimistic. The peek hands the shuffle
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

#[cfg(test)]
mod tests {
    use super::*;

    fn tunable(name: &str, idx: usize, value: f64, is_fixed: bool) -> Tunable {
        Tunable { name: name.into(), value, idx, is_fixed, freeze_resistant: false }
    }

    fn a_run() -> RunState {
        RunState {
            epoch: 7,
            lr_scale: 1.0,
            k: 1.0,
            k_ref: 1.0,
            k_momentum: 0.0,
            // Finite on purpose. `Progress::default` seeds its trails with NaN, and
            // serde_json writes NaN as null, which no `f64` reads back.
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

        let path = std::env::temp_dir().join(format!("soul_ckpt_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        // Load against a grown layout: gamma is new, so it must come back fresh.
        let grown = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true), tunable("gamma", 2, 30.0, false)];

        let d = load_checkpoint(path, &grown, &[1.0, 2.0, 30.0]).unwrap();

        // The seeds and the dataset identity are read off `Checkpoint` by `peek_checkpoint`, ahead
        // of the split and therefore ahead of this load, so that is where they are checked.
        let cp = peek_checkpoint(path).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!((cp.rng_seed, cp.split_seed, cp.dataset), (999, Some(888), 777));
        assert_eq!(cp.dataset_path, "data/test.txt");

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
        assert_eq!(d.fresh_params, 1, "gamma is the one parameter the checkpoint never held");
    }

    #[test]
    fn a_rename_refuses_to_resume_but_a_removal_does_not() {
        let tunables = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true)];

        let state = TrainerState {
            run: a_run(),
            rng_seed: 1,
            split_seed: 2,
            dataset: 3,
            dataset_path: "data/test.txt",
            values: &[10.0, 20.0],
            momentum: &[0.1, 0.2],
            ema: &[9.0, 19.0],
            grad_ema: &[0.01, 0.02],
            stagnant: &[0, 0],
            frozen: &[false, true],
            best_val_params: &[1.5, 2.5],
            best_train_params: &[3.5, 4.5],
        };

        let path = std::env::temp_dir().join(format!("soul_ckpt_renamed_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        let renamed = [tunable("alpha", 0, 1.0, false), tunable("gamma", 1, 30.0, false)];
        let refused = load_checkpoint(path, &renamed, &[1.0, 30.0]);

        // Nothing left in the layout wants beta's state.
        let shrunk = [tunable("alpha", 0, 1.0, false)];
        let resumed = load_checkpoint(path, &shrunk, &[1.0]);

        std::fs::remove_file(path).ok();

        assert!(refused.is_err(), "a renamed parameter must not resume");

        let d = resumed.expect("a removed parameter must still resume the rest");
        assert_eq!(d.values, [10.0], "alpha keeps its trained value across the removal");
        assert_eq!(d.momentum, [0.1], "and its momentum");
        assert_eq!(d.fresh_params, 0, "a removal leaves nothing to start from defaults");
    }

    #[test]
    fn checkpoint_without_smoothing_resumes_with_an_open_record() {
        // Round-trips through a written file with the four fields stripped: the serde defaults
        // are the whole subject, and building a Checkpoint directly would step over them.
        let tunables = [tunable("alpha", 0, 1.0, false)];

        let state = TrainerState {
            run: a_run(),
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
        };

        let path = std::env::temp_dir().join(format!("soul_ckpt_legacy_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        let obj = raw.as_object_mut().unwrap();
        for key in ["val_smooth", "best_val_smooth", "train_smooth", "best_train_smooth"] {
            obj.remove(key);
        }

        std::fs::write(path, serde_json::to_string(&raw).unwrap()).unwrap();

        let d = load_checkpoint(path, &tunables, &[1.0]).unwrap();
        std::fs::remove_file(path).ok();

        assert!(d.run.progress.val_smooth.is_nan(), "an absent val trail must re-seed, got {}", d.run.progress.val_smooth);
        assert!(
            d.run.progress.train_smooth.is_nan(),
            "an absent train trail must re-seed, got {}",
            d.run.progress.train_smooth
        );
        assert_eq!((d.run.progress.best_val_smooth, d.run.progress.best_train_smooth), (f64::MAX, f64::MAX));
        assert_eq!(d.run.progress.best_val_loss, 0.2, "the rest of the checkpoint must survive the older format");
    }

    #[test]
    fn a_checkpoint_without_a_split_seed_reads_as_none() {
        // A bare u64 would default to 0, a split no older checkpoint ever drew.
        // Option is what lets a resume recognize the absence and fall back to rng_seed.
        let tunables = [tunable("alpha", 0, 1.0, false)];

        let state = TrainerState {
            run: a_run(),
            rng_seed: 12345,
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
        };

        let path = std::env::temp_dir().join(format!("soul_ckpt_split_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        raw.as_object_mut().unwrap().remove("split_seed");
        std::fs::write(path, serde_json::to_string(&raw).unwrap()).unwrap();

        let cp = peek_checkpoint(path).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!(cp.split_seed, None);
        assert_eq!(cp.rng_seed, 12345, "the fallback the resume uses must survive the older format");
    }
}
