use std::{
    collections::BTreeMap,
    fs::File,
    io::{self, BufReader, BufWriter},
};

use serde::{Deserialize, Serialize};
use soul::{color, engine::eval_params::Tunable};

use crate::core::{error::CheckpointError, fnv::Fnv1a};

pub const CHECKPOINT_VERSION: u32 = 3;

/// Serialisable training checkpoint: everything needed to resume a run.
///
/// A resume that reconstructs K, the EMA trail, the freeze mask, or the
/// snapshot hall from defaults trains a subtly different run wearing the old
/// one's epoch counter.
///
/// Parameters are keyed by name to ensure robustness against layout changes
/// (e.g. adding or reordering evaluation terms).
#[derive(Serialize, Deserialize)]
pub struct Checkpoint {
    pub version: u32,
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    pub best_val_loss: f64,
    pub plateau_count: usize,
    pub params: BTreeMap<String, ParamState>,
    pub snapshots: Vec<Snapshot>,
    pub hash: u64,
    pub rng_seed: u64,
    pub dataset: u64,
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

/// A frozen parameter snapshot at a specific epoch.
#[derive(Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub epoch: usize,
    pub params: BTreeMap<String, f64>,
    pub error: f64,
}

/// The trainer's live state, borrowed by [`save_checkpoint`]. Per-parameter
/// slices are indexed by `Tunable::idx`; the save maps them to names.
pub struct TrainerState<'a> {
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    pub best_val_loss: f64,
    pub plateau_count: usize,
    pub rng_seed: u64,
    pub dataset: u64,
    pub values: &'a [f64],
    pub momentum: &'a [f64],
    pub ema: &'a [f64],
    pub grad_ema: &'a [f64],
    pub stagnant: &'a [usize],
    pub frozen: &'a [bool],
    pub snapshots: &'a [Snapshot],
}

/// Save training state to a JSON checkpoint file.
///
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

    let cp = Checkpoint {
        version: CHECKPOINT_VERSION,
        epoch: state.epoch,
        lr_scale: state.lr_scale,
        k: state.k,
        k_ref: state.k_ref,
        best_val_loss: state.best_val_loss,
        plateau_count: state.plateau_count,
        params,
        snapshots: state.snapshots.to_vec(),
        hash: compute_layout_hash(tunables),
        rng_seed: state.rng_seed,
        dataset: state.dataset,
    };

    let tmp = format!("{path}.tmp");
    let file = File::create(&tmp)?;

    serde_json::to_writer(BufWriter::new(file), &cp)?;
    std::fs::rename(&tmp, path)?; // atomic on Linux
    Ok(())
}

/// A loaded checkpoint with per-parameter state mapped back to current
/// `Tunable::idx` order, ready for the trainer to adopt.
pub struct CheckpointData {
    pub epoch: usize,
    pub lr_scale: f64,
    pub k: f64,
    pub k_ref: f64,
    pub best_val_loss: f64,
    pub plateau_count: usize,
    pub rng_seed: u64,
    pub dataset: u64,
    pub values: Vec<f64>,
    pub momentum: Vec<f64>,
    pub ema: Vec<f64>,
    pub grad_ema: Vec<f64>,
    pub stagnant: Vec<usize>,
    pub frozen: Vec<bool>,
    pub snapshots: Vec<Snapshot>,
}

/// Load training state from a JSON checkpoint file.
///
/// Maps saved parameter names back to their current indices. A parameter
/// missing from the checkpoint gets a fresh start: current code value, zero
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
            "{}[!] Error: Checkpoint version mismatch! (Expected: {}, Found: {})\x1b[0m",
            color::ansi_fg((225, 89, 91)),
            CHECKPOINT_VERSION,
            cp.version
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
            "{}Warning: Checkpoint layout hash mismatch! (Saved: {:x}, Current: {:x})\x1b[0m",
            color::ansi_fg((218, 165, 32)),
            cp.hash,
            current_hash
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
            // A param the code now declares fixed stays fixed, whatever
            // the checkpoint remembers.
            frozen[t.idx] = p.frozen || t.is_fixed;
        }
    }

    Ok(CheckpointData {
        epoch: cp.epoch,
        lr_scale: cp.lr_scale,
        k: cp.k,
        k_ref: cp.k_ref,
        best_val_loss: cp.best_val_loss,
        plateau_count: cp.plateau_count,
        rng_seed: cp.rng_seed,
        dataset: cp.dataset,
        values,
        momentum,
        ema,
        grad_ema,
        stagnant,
        frozen,
        snapshots: cp.snapshots,
    })
}

/// Parse a checkpoint without mapping it onto the current layout.
///
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

/// Snapshot hall of fame: keep the N best checkpoints by validation loss.
///
/// Returns whether `error` beats the previous best, not whether the snapshot
/// was admitted: while the hall is still filling, every epoch is admitted, and
/// reporting those as best would mark the whole warmup tail ✦.
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
    let is_best = snapshots.first().is_none_or(|s| error < s.error);

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

    is_best
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
        let snapshots = vec![Snapshot { epoch: 7, params: BTreeMap::from([("alpha".into(), 1.5)]), error: 0.25 }];

        let state = TrainerState {
            epoch: 42,
            lr_scale: 0.5,
            k: 1.23,
            k_ref: 1.11,
            best_val_loss: 0.2,
            plateau_count: 3,
            rng_seed: 999,
            dataset: 777,
            values: &[10.0, 20.0],
            momentum: &[0.1, 0.2],
            ema: &[9.0, 19.0],
            grad_ema: &[0.01, 0.02],
            stagnant: &[4, 5],
            frozen: &[true, true],
            snapshots: &snapshots,
        };

        let path = std::env::temp_dir().join(format!("soul_ckpt_test_{}.json", std::process::id()));
        let path = path.to_str().unwrap();
        save_checkpoint(path, &tunables, &state).unwrap();

        // Load against a grown layout: gamma is new, so it must come back fresh.
        let grown = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true), tunable("gamma", 2, 30.0, false)];

        let d = load_checkpoint(path, &grown, &[1.0, 2.0, 30.0]).unwrap();
        std::fs::remove_file(path).ok();

        assert_eq!((d.epoch, d.lr_scale, d.k, d.k_ref), (42, 0.5, 1.23, 1.11));
        assert_eq!((d.best_val_loss, d.plateau_count), (0.2, 3));
        assert_eq!((d.rng_seed, d.dataset), (999, 777));
        assert_eq!(d.values, [10.0, 20.0, 30.0]);
        assert_eq!(d.momentum, [0.1, 0.2, 0.0]);
        assert_eq!(d.ema, [9.0, 19.0, 30.0]);
        assert_eq!(d.grad_ema, [0.01, 0.02, 0.0]);
        assert_eq!(d.stagnant, [4, 5, 0]);
        assert_eq!(d.frozen, [true, true, false]);
        assert_eq!(d.snapshots.len(), 1);
        assert_eq!(d.snapshots[0].epoch, 7);
    }
}
