//! Seed spread: how much of a tuned parameter set is the data, and how much is the draw.
//!
//! Every retune comparison rests on an assumption nobody has measured, that two runs of one
//! config land in the same place. This runs N seeds and reports where they disagree, in
//! validation loss and in the integers that ship. Two numbers come out of it: the loss spread,
//! which says whether `best_val_loss` can rank anything, and the parameter spread, which says
//! how much of a retune's diff is signal.
//!
//! A floor is expected rather than zero. Parameters parked near a rounding boundary flip on
//! nothing, so disagreement of ±1 on a minority of them is the resolution of the instrument.

use std::{cmp::Reverse, env, fs, process::Command, time::Instant};

use serde::Deserialize;

use super::{
    engine::eval_params,
    palette::{LAB, RESET, VAL},
    report::fmt_loss,
};

/// How many of the most load-bearing parameters get their own line in the report. A spread
/// concentrated below this rank costs nothing; a spread above it is the tuner handing you a
/// different engine per seed.
const LOAD_BEARING: usize = 50;

#[derive(Deserialize)]
struct Final {
    seed: u64,
    /// Absent in a record written while the training seed still carved out the val slice.
    #[serde(default)]
    split_seed: Option<u64>,
    /// `None` when the run had no validation split: nothing to rank it by.
    #[serde(default)]
    best_val_loss: Option<f64>,
    best_val_epoch: usize,
    /// `params` in a log written before the key said which vector it holds.
    #[serde(rename = "best_val_params", alias = "params")]
    params: Vec<i32>,
    /// Per-parameter |gradient| EMA, the same figure `sensitivity-report.txt` ranks on.
    #[serde(default)]
    sensitivity: Vec<f64>,
}

pub fn run_seed_spread(dataset: &str, config_path: &str, epochs: usize, count: usize, log_path: &str) {
    let mut rng = fastrand::Rng::with_seed(0xA5EE_D000);
    let seeds: Vec<u64> = (0..count).map(|_| rng.u64(..)).collect();

    println!("\n{LAB}Seed spread:{RESET} {VAL}{count}{RESET} seeds × {VAL}{epochs}{RESET} epochs on {dataset}\n");
    for (i, &seed) in seeds.iter().enumerate() {
        let start = Instant::now();
        let ok = spawn_trial(dataset, config_path, epochs, log_path, &[("--seed", seed.to_string())]);
        let elapsed = start.elapsed().as_secs_f32();

        let kept = if ok { fs::copy("evaltune_best.txt", format!("seed_{seed}_best.txt")).is_ok() } else { false };

        let status = if ok { "done" } else { "FAILED" };
        let note = if ok && !kept { "  (no evaltune_best.txt to keep)" } else { "" };

        println!("  [{}/{count}] {seed:<22} {status:<8} {elapsed:.1}s{note}", i + 1);
    }

    let runs = collect(log_path, &seeds);

    if runs.len() < 2 {
        eprintln!("Only {} of {count} runs reported a final record; nothing to compare.", runs.len());
        return;
    }

    report(&runs);
}

/// One training run in its own process, so a failure costs one trial rather than the sweep.
///
/// `extra` is whatever the caller varies. Results come back through `log_path`, where
/// the child appends its `final` record; reading them off stdout instead would tie every
/// caller to a print format.
pub fn spawn_trial(dataset: &str, config_path: &str, epochs: usize, log_path: &str, extra: &[(&str, String)]) -> bool {
    let Ok(exe) = env::current_exe() else {
        eprintln!("Cannot locate the running binary to spawn a trial.");
        return false;
    };

    let quiet = env::temp_dir().join(format!("evaltune_trial_{}.txt", std::process::id()));
    let Ok(sink) = fs::File::create(&quiet) else { return false };

    let mut cmd = Command::new(exe);

    cmd.arg("--dataset").arg(dataset);
    cmd.arg("--config").arg(config_path);
    cmd.arg("--epochs").arg(epochs.to_string());
    cmd.arg("--log").arg(log_path);

    for (flag, value) in extra {
        cmd.arg(flag).arg(value);
    }

    let status = cmd.stdout(sink).stderr(std::process::Stdio::inherit()).status();
    let _ = fs::remove_file(&quiet);

    status.is_ok_and(|s| s.success())
}

/// What the last run in `log_path` reached, for a caller spawning one trial at a time.
#[must_use]
pub fn last_best_val(log_path: &str) -> Option<f64> {
    finals(log_path).last().and_then(|r| r.best_val_loss)
}

/// Every `final` record in an append-only log, in the order they were written.
fn finals(log_path: &str) -> Vec<Final> {
    let Ok(text) = fs::read_to_string(log_path) else { return Vec::new() };

    text.lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|v| v.get("event").and_then(serde_json::Value::as_str) == Some("final"))
        .filter_map(|v| serde_json::from_value::<Final>(v).ok())
        .collect()
}

/// Last record per requested seed. The log is append-only across runs, so matching by seed
/// and keeping the latest is what makes a rerun overwrite rather than accumulate.
fn collect(log_path: &str, seeds: &[u64]) -> Vec<Final> {
    let mut found: Vec<Option<Final>> = seeds.iter().map(|_| None).collect();

    for record in finals(log_path) {
        if let Some(slot) = seeds.iter().position(|&s| s == record.seed) {
            found[slot] = Some(record);
        }
    }

    found.into_iter().flatten().collect()
}

fn report(runs: &[Final]) {
    // On big3, which tenth got held out moves best_val_loss by 1.6e-3 at identical parameters,
    // eighty times what 4000 epochs buy. A table mixing holdouts therefore ranks the holdouts.
    if runs.iter().any(|r| r.split_seed != runs[0].split_seed) {
        eprintln!("  Runs held out different validation slices; the L_val column below ranks that, not the seed.");
    }

    let losses: Vec<f64> = runs.iter().filter_map(|r| r.best_val_loss).collect();

    if losses.len() < runs.len() {
        eprintln!("  Runs without a holdout show —; the L_val spread ranks the rest.");
    }

    println!("\n  {LAB}seed{RESET}                    {LAB}L_val{RESET}       {LAB}epoch{RESET}");

    for r in runs {
        println!("  {:<22}  {}    {}", r.seed, fmt_loss(r.best_val_loss), r.best_val_epoch);
    }

    if losses.is_empty() {
        println!("\n  {LAB}L_val{RESET}   (no run had a validation split)");
    } else {
        let lo = losses.iter().copied().fold(f64::MAX, f64::min);
        let hi = losses.iter().copied().fold(f64::MIN, f64::max);

        println!("\n  {LAB}L_val{RESET}   min {VAL}{lo:.6}{RESET}  max {VAL}{hi:.6}{RESET}  spread {VAL}{:.2e}{RESET}", hi - lo);
    }

    let params = eval_params::collect_parameters();
    let np = runs[0].params.len();

    if runs.iter().any(|r| r.params.len() != np) {
        eprintln!("  Runs disagree on parameter count; the log mixes incompatible builds.");
        return;
    }

    // Range across seeds, per parameter. A parameter every run agreed on is one the data
    // pinned; the rest is what a retune diff would have shown you and called a change.
    let mut ranges: Vec<(i32, usize)> = (0..np)
        .map(|i| {
            let lo = runs.iter().map(|r| r.params[i]).min().unwrap_or(0);
            let hi = runs.iter().map(|r| r.params[i]).max().unwrap_or(0);

            (hi - lo, i)
        })
        .collect();

    let identical = ranges.iter().filter(|(r, _)| *r == 0).count();
    let jitter = ranges.iter().filter(|(r, _)| *r == 1).count();
    let real = ranges.iter().filter(|(r, _)| *r > 1).count();
    let total: i32 = ranges.iter().map(|(r, _)| *r).sum();

    println!(
        "  {LAB}Params{RESET}  identical {VAL}{identical}{RESET}  ±1 on {VAL}{jitter}{RESET}  \
         wider on {VAL}{real}{RESET}  of {np}, total spread {VAL}{total}{RESET}"
    );

    // Range grows with the seed count even at fixed variance, 2.06σ at four seeds against 3.53σ
    // at sixteen, so the counts above compare only within one sweep size. Deviation does not.
    let n = runs.len() as f64;
    let deviation = |i: usize| {
        let mean = runs.iter().map(|r| f64::from(r.params[i])).sum::<f64>() / n;

        (runs.iter().map(|r| (f64::from(r.params[i]) - mean).powi(2)).sum::<f64>() / n).sqrt()
    };

    let unsettled = (0..np).filter(|&i| deviation(i) > 0.5).count();
    let total_deviation: f64 = (0..np).map(deviation).sum();

    println!(
        "  {LAB}Spread{RESET}  sd over half a unit on {VAL}{unsettled}{RESET}  total sd {VAL}{total_deviation:.1}{RESET}  \
         {LAB}(comparable across sweep sizes){RESET}"
    );

    // The same spread split by whether the loss notices the parameter at all. Seeds disagreeing
    // on a term with ΔL near 1e-8 is free; disagreeing on mobility or king safety is not.
    if runs.iter().all(|r| r.sensitivity.len() == np) {
        let mut by_impact: Vec<usize> = (0..np).collect();
        let mean = |i: usize| runs.iter().map(|r| r.sensitivity[i]).sum::<f64>() / runs.len() as f64;

        by_impact.sort_unstable_by(|&a, &b| mean(b).total_cmp(&mean(a)));

        let top = &by_impact[..LOAD_BEARING.min(np)];
        let same = top.iter().filter(|&&i| ranges[i].0 == 0).count();
        let wider = top.iter().filter(|&&i| ranges[i].0 > 1).count();
        let spread: i32 = top.iter().map(|&i| ranges[i].0).sum();

        println!(
            "  {LAB}Top {LOAD_BEARING}{RESET}  identical {VAL}{same}{RESET}  ±1 on {VAL}{}{RESET}  \
             wider on {VAL}{wider}{RESET}, total spread {VAL}{spread}{RESET}",
            top.len() - same - wider
        );
    }

    ranges.sort_unstable_by_key(|&(range, _)| Reverse(range));

    let worst: Vec<String> = ranges
        .iter()
        .take(6)
        .filter(|(r, _)| *r > 1)
        .map(|&(r, i)| format!("{} {r}", params.get(i).map_or("?", |p| p.name.as_str())))
        .collect();

    if !worst.is_empty() {
        println!("  {LAB}Widest{RESET}  {}", worst.join("   "));
    }

    // The pair to spend games on: furthest apart in shipped integers, so an SPRT between them
    // asks the largest version of the question the sweep raised.
    let mut far = (0usize, 0usize, 0i64);

    for a in 0..runs.len() {
        for b in a + 1..runs.len() {
            let d: i64 = runs[a].params.iter().zip(&runs[b].params).map(|(x, y)| i64::from((x - y).abs())).sum();

            if d > far.2 {
                far = (a, b, d);
            }
        }
    }

    let (a, b, distance) = far;

    println!(
        "\n  {LAB}Furthest pair{RESET}  {VAL}{}{RESET} and {VAL}{}{RESET}, {VAL}{distance}{RESET} apart, \
         L_val {} vs {}\n  paste seed_{}_best.txt against seed_{}_best.txt",
        runs[a].seed, runs[b].seed,
        fmt_loss(runs[a].best_val_loss),
        fmt_loss(runs[b].best_val_loss),
        runs[a].seed, runs[b].seed
    );
}
