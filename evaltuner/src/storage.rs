//! Checkpoint persistence, stable FNV-1a digests, and append-only JSON-lines run logging.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{File, OpenOptions},
    io::{self, BufReader, BufWriter, Write},
    path::Path,
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{engine::Tunable, training::Progress};

const CHECKPOINT_VERSION: u32 = 5;

pub(crate) struct JsonLog(File);

impl JsonLog {
    pub(crate) fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        Ok(Self(OpenOptions::new().create(true).append(true).open(path)?))
    }

    pub(crate) fn write<T: Serialize>(&mut self, event: &str, data: &T) {
        #[derive(Serialize)]
        struct Entry<'a, T> {
            timestamp: u64,
            event:     &'a str,
            #[serde(flatten)]
            data:      &'a T,
        }

        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs();
        if let Ok(mut json) = serde_json::to_string(&Entry { timestamp, event, data }) {
            json.push('\n');
            self.0.write_all(json.as_bytes()).ok();
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub(crate) struct RunState {
    pub(crate) epoch:      usize,
    pub(crate) lr_scale:   f64,
    pub(crate) k:          f64,
    pub(crate) k_ref:      f64,
    #[serde(default)]
    pub(crate) k_momentum: f64,
    #[serde(flatten)]
    pub(crate) progress:   Progress,
}

/// Flat `best_*` vectors carry no inline keys and remap through `param_names` on resume.
#[derive(Serialize, Deserialize)]
pub(crate) struct Checkpoint {
    pub(crate) version:           u32,
    #[serde(flatten)]
    pub(crate) run:               RunState,
    pub(crate) params:            BTreeMap<String, ParamState>,
    pub(crate) hash:              u64,
    pub(crate) rng_seed:          u64,
    #[serde(default)]
    pub(crate) split_seed:        Option<u64>,
    pub(crate) dataset:           u64,
    #[serde(default)]
    pub(crate) dataset_path:      String,
    pub(crate) param_names:       Vec<String>,
    pub(crate) best_val_params:   Vec<f64>,
    pub(crate) best_train_params: Vec<f64>,
}

#[derive(Serialize, Deserialize)]
pub(crate) struct ParamState {
    pub(crate) value:    f64,
    pub(crate) momentum: f64,
    pub(crate) ema:      f64,
    pub(crate) grad_ema: f64,
    pub(crate) stagnant: usize,
    pub(crate) frozen:   bool,
}

pub(crate) struct TrainerState<'a> {
    pub(crate) run:               RunState,
    pub(crate) rng_seed:          u64,
    pub(crate) split_seed:        u64,
    pub(crate) dataset:           u64,
    pub(crate) dataset_path:      &'a str,
    pub(crate) values:            &'a [f64],
    pub(crate) momentum:          &'a [f64],
    pub(crate) ema:               &'a [f64],
    pub(crate) grad_ema:          &'a [f64],
    pub(crate) stagnant:          &'a [usize],
    pub(crate) frozen:            &'a [bool],
    pub(crate) best_val_params:   &'a [f64],
    pub(crate) best_train_params: &'a [f64],
}

pub(crate) struct CheckpointData {
    pub(crate) run:               RunState,
    pub(crate) values:            Vec<f64>,
    pub(crate) momentum:          Vec<f64>,
    pub(crate) ema:               Vec<f64>,
    pub(crate) grad_ema:          Vec<f64>,
    pub(crate) stagnant:          Vec<usize>,
    pub(crate) frozen:            Vec<bool>,
    pub(crate) best_val_params:   Vec<f64>,
    pub(crate) best_train_params: Vec<f64>,
    pub(crate) fresh_params:      usize,
}

#[derive(Debug, thiserror::Error)]
pub(crate) enum CheckpointError {
    #[error("I/O error: {0}")]
    Io(#[from] io::Error),
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),
    #[error("Checkpoint version {found}, expected {CHECKPOINT_VERSION}")]
    VersionMismatch { found: u32 },
    #[error("Layout renamed parameters: {0}. Start fresh, or check out the layout that wrote it")]
    LayoutRenamed(String),
}

/// FNV-1a over byte slices (`DefaultHasher` is not stable across releases).
pub(crate) fn fnv1a<'a>(slices: impl IntoIterator<Item = &'a [u8]>) -> u64 {
    let mut h = 0xCBF2_9CE4_8422_2325u64;
    for slice in slices {
        for &b in slice {
            h = (h ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01B3);
        }
    }
    h
}

pub(crate) fn save_checkpoint(path: &str, tunables: &[Tunable], state: &TrainerState) -> Result<(), CheckpointError> {
    let mut params = BTreeMap::new();
    let mut param_names = vec![String::new(); tunables.len()];

    for t in tunables {
        param_names[t.idx] = t.name.clone();
        params.insert(t.name.clone(), ParamState {
            value:    state.values[t.idx],
            momentum: state.momentum[t.idx],
            ema:      state.ema[t.idx],
            grad_ema: state.grad_ema[t.idx],
            stagnant: state.stagnant[t.idx],
            frozen:   state.frozen[t.idx],
        });
    }

    let checkpoint = Checkpoint {
        version: CHECKPOINT_VERSION,
        run: state.run.clone(),
        params,
        hash: layout_hash(tunables),
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

pub(crate) fn load_checkpoint(path: &str, tunables: &[Tunable], current_values: &[f64]) -> Result<CheckpointData, CheckpointError> {
    let cp = peek_checkpoint(path)?;
    let n = current_values.len();

    let (mut values, mut ema) = (current_values.to_vec(), current_values.to_vec());
    let (mut momentum, mut grad_ema) = (vec![0.0; n], vec![0.0; n]);

    let mut stagnant = vec![0usize; n];
    let mut frozen: Vec<bool> = tunables.iter().map(|t| t.is_fixed).collect();
    let mut fresh_params = 0;

    if cp.hash != layout_hash(tunables) {
        let live: BTreeSet<&str> = tunables.iter().map(|t| t.name.as_str()).collect();
        let saved: BTreeSet<&str> = cp.params.keys().map(String::as_str).collect();
        let dropped: Vec<&str> = saved.difference(&live).copied().collect();
        fresh_params = live.difference(&saved).count();

        // Simultaneous drops and additions point to an unmigrated rename rather than an append or prune.
        if !dropped.is_empty() && fresh_params > 0 {
            return Err(CheckpointError::LayoutRenamed(format!(
                "{} dropped ({}) while {fresh_params} added",
                dropped.len(),
                dropped[..dropped.len().min(3)].join(", ")
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

    Ok(CheckpointData {
        run: cp.run,
        best_val_params: remap_flat_params(&cp.param_names, &cp.best_val_params, tunables, &values),
        best_train_params: remap_flat_params(&cp.param_names, &cp.best_train_params, tunables, &values),
        values,
        momentum,
        ema,
        grad_ema,
        stagnant,
        frozen,
        fresh_params,
    })
}

/// Reads checkpoint metadata before partitioning so a resume reuses the original holdout split seed.
pub(crate) fn peek_checkpoint(path: &str) -> Result<Checkpoint, CheckpointError> {
    let cp: Checkpoint = serde_json::from_reader(BufReader::new(File::open(path)?))?;
    if cp.version != CHECKPOINT_VERSION {
        return Err(CheckpointError::VersionMismatch { found: cp.version });
    }
    Ok(cp)
}

fn layout_hash(tunables: &[Tunable]) -> u64 { fnv1a(tunables.iter().map(|t| t.name.as_bytes())) }

fn remap_flat_params(names: &[String], vals: &[f64], tunables: &[Tunable], fallback: &[f64]) -> Vec<f64> {
    let saved: BTreeMap<&str, f64> = names.iter().zip(vals).map(|(n, &v)| (n.as_str(), v)).collect();
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
        std::env::temp_dir()
            .join(format!("soul_ckpt_{tag}_{}.json", std::process::id()))
            .display()
            .to_string()
    }

    fn saved_state(n: usize) -> TrainerState<'static> {
        const VALS: [f64; 2] = [10.0, 20.0];
        const MOM: [f64; 2] = [0.1, 0.2];
        const EMA: [f64; 2] = [9.0, 19.0];
        const GEMA: [f64; 2] = [0.01, 0.02];
        const STAG: [usize; 2] = [4, 5];
        const FROZ: [bool; 2] = [true, true];
        const BVAL: [f64; 2] = [1.5, 2.5];
        const BTRN: [f64; 2] = [3.5, 4.5];

        let progress = Progress {
            best_val_loss:     0.2,
            best_val_epoch:    3,
            val_smooth:        0.21,
            best_val_smooth:   0.205,
            best_train_loss:   0.3,
            best_train_epoch:  5,
            train_smooth:      0.31,
            best_train_smooth: 0.305,
            plateau_count:     0,
        };
        TrainerState {
            run:               RunState { epoch: 42, lr_scale: 0.5, k: 1.23, k_ref: 1.11, k_momentum: 0.42, progress },
            rng_seed:          999,
            split_seed:        888,
            dataset:           777,
            dataset_path:      "data/test.txt",
            values:            &VALS[..n],
            momentum:          &MOM[..n],
            ema:               &EMA[..n],
            grad_ema:          &GEMA[..n],
            stagnant:          &STAG[..n],
            frozen:            &FROZ[..n],
            best_val_params:   &BVAL[..n],
            best_train_params: &BTRN[..n],
        }
    }

    #[test]
    fn roundtrip_rename_guard_and_legacy_fields() {
        let two = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true)];
        let path = temp_path("roundtrip");
        save_checkpoint(&path, &two, &saved_state(2)).unwrap();

        let grown = [tunable("alpha", 0, 1.0, false), tunable("beta", 1, 2.0, true), tunable("gamma", 2, 30.0, false)];
        let d = load_checkpoint(&path, &grown, &[1.0, 2.0, 30.0]).unwrap();
        let cp = peek_checkpoint(&path).unwrap();
        assert_eq!((cp.rng_seed, cp.split_seed, cp.dataset, cp.dataset_path.as_str()), (999, Some(888), 777, "data/test.txt"));
        assert_eq!((d.run.epoch, d.run.lr_scale, d.run.k, d.run.k_ref, d.run.k_momentum), (42, 0.5, 1.23, 1.11, 0.42));
        assert_eq!((d.run.progress.val_smooth, d.run.progress.best_val_smooth, d.fresh_params), (0.21, 0.205, 1));
        assert_eq!(
            (d.values, d.momentum, d.ema, d.grad_ema),
            (vec![10.0, 20.0, 30.0], vec![0.1, 0.2, 0.0], vec![9.0, 19.0, 30.0], vec![0.01, 0.02, 0.0])
        );
        assert_eq!(
            (d.stagnant, d.frozen, d.best_val_params, d.best_train_params),
            (vec![4, 5, 0], vec![true, true, false], vec![1.5, 2.5, 30.0], vec![3.5, 4.5, 30.0])
        );

        // Simultaneous drop + add rejects as unmigrated rename; pure removal succeeds.
        assert!(
            load_checkpoint(&path, &[tunable("alpha", 0, 1.0, false), tunable("gamma", 1, 30.0, false)], &[1.0, 30.0]).is_err()
        );
        let pruned = load_checkpoint(&path, &[tunable("alpha", 0, 1.0, false)], &[1.0]).unwrap();
        assert_eq!((pruned.values.as_slice(), pruned.momentum.as_slice(), pruned.fresh_params), (&[10.0][..], &[0.1][..], 0));

        // Unseeded NaN trails and stripped legacy keys fall back cleanly.
        let one = [tunable("alpha", 0, 1.0, false)];
        let base = saved_state(1);
        save_checkpoint(&path, &one, &TrainerState { run: RunState { progress: Progress::default(), ..base.run }, ..base })
            .unwrap();
        let d = load_checkpoint(&path, &one, &[1.0]).unwrap();
        assert!(d.run.progress.val_smooth.is_nan() && d.run.progress.train_smooth.is_nan());

        save_checkpoint(&path, &one, &TrainerState { rng_seed: 12345, ..saved_state(1) }).unwrap();
        let mut raw: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        for key in ["val_smooth", "best_val_smooth", "train_smooth", "best_train_smooth", "split_seed"] {
            raw.as_object_mut().unwrap().remove(key);
        }
        std::fs::write(&path, serde_json::to_string(&raw).unwrap()).unwrap();
        let (cp, d) = (peek_checkpoint(&path).unwrap(), load_checkpoint(&path, &one, &[1.0]).unwrap());
        let _ = std::fs::remove_file(&path);
        assert_eq!(
            (cp.split_seed, cp.rng_seed, d.run.progress.best_val_smooth, d.run.progress.best_val_loss),
            (None, 12345, f64::MAX, 0.2)
        );
        assert!(d.run.progress.val_smooth.is_nan() && d.run.progress.train_smooth.is_nan());
    }

    #[test]
    fn json_log_appends_one_object_per_line() {
        let path = std::env::temp_dir().join(format!("soul_log_test_{}.jsonl", std::process::id()));
        let mut log = JsonLog::open(&path).unwrap();
        log.write("epoch", &serde_json::json!({ "n": 1 }));
        log.write("final", &serde_json::json!({ "n": 2 }));
        let text = std::fs::read_to_string(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        let events: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!((events[0]["event"].as_str(), events[1]["event"].as_str()), (Some("epoch"), Some("final")));
    }
}
