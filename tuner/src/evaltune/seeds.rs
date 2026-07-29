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
    palette::{self, RESET},
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
    best_val_loss: f64,
    best_val_epoch: usize,
    params: Vec<i32>,
    /// Per-parameter |gradient| EMA, the same figure `sensitivity-report.txt` ranks on.
    #[serde(default)]
    sensitivity: Vec<f64>,
}

pub fn run_seed_spread(dataset: &str, config_path: &str, epochs: usize, count: usize, log_path: &str) {
    let mut rng = fastrand::Rng::with_seed(0xA5EE_D000);
    let seeds: Vec<u64> = (0..count).map(|_| rng.u64(..)).collect();

    let lab = palette::fg(palette::LABEL);
    let val = palette::fg(palette::VALUE);

    println!("\n{lab}Seed spread:{RESET} {val}{count}{RESET} seeds × {val}{epochs}{RESET} epochs on {dataset}\n");
    for (i, &seed) in seeds.iter().enumerate() {
        let start = Instant::now();
        let ok = run_one(dataset, config_path, epochs, seed, log_path);
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

/// One training run in its own process, so a failure costs one seed rather than the sweep.
fn run_one(dataset: &str, config_path: &str, epochs: usize, seed: u64, log_path: &str) -> bool {
    let Ok(exe) = env::current_exe() else {
        eprintln!("Cannot locate the running binary to spawn a trial.");
        return false;
    };

    let quiet = env::temp_dir().join(format!("seed_spread_{seed}.txt"));
    let Ok(sink) = fs::File::create(&quiet) else { return false };

    let status = Command::new(exe)
        .arg("--dataset")
        .arg(dataset)
        .arg("--config")
        .arg(config_path)
        .arg("--epochs")
        .arg(epochs.to_string())
        .arg("--seed")
        .arg(seed.to_string())
        // The children have to append where `collect` reads, so the caller's path
        // goes down with them.
        .arg("--log")
        .arg(log_path)
        .stdout(sink)
        .stderr(std::process::Stdio::inherit())
        .status();

    let _ = fs::remove_file(&quiet);

    status.is_ok_and(|s| s.success())
}

/// Last `final` record per requested seed. The log is append-only across runs, so matching by
/// seed and keeping the latest is what makes a rerun overwrite rather than accumulate.
fn collect(log_path: &str, seeds: &[u64]) -> Vec<Final> {
    let Ok(text) = fs::read_to_string(log_path) else {
        eprintln!("No log at {log_path}; nothing to compare.");
        return Vec::new();
    };

    let mut found: Vec<Option<Final>> = seeds.iter().map(|_| None).collect();

    for line in text.lines() {
        let Ok(value) = serde_json::from_str::<serde_json::Value>(line) else { continue };

        if value.get("event").and_then(serde_json::Value::as_str) != Some("final") {
            continue;
        }

        let Ok(record) = serde_json::from_value::<Final>(value) else { continue };

        if let Some(slot) = seeds.iter().position(|&s| s == record.seed) {
            found[slot] = Some(record);
        }
    }

    found.into_iter().flatten().collect()
}

fn report(runs: &[Final]) {
    let lab = palette::fg(palette::LABEL);
    let val = palette::fg(palette::VALUE);

    // On big3, which tenth got held out moves best_val_loss by 1.6e-3 at identical parameters,
    // eighty times what 4000 epochs buy. A table mixing holdouts therefore ranks the holdouts.
    if runs.iter().any(|r| r.split_seed != runs[0].split_seed) {
        eprintln!("  Runs held out different validation slices; the L_val column below ranks that, not the seed.");
    }

    let lo = runs.iter().map(|r| r.best_val_loss).fold(f64::MAX, f64::min);
    let hi = runs.iter().map(|r| r.best_val_loss).fold(f64::MIN, f64::max);

    println!("\n  {lab}seed{RESET}                    {lab}L_val{RESET}       {lab}epoch{RESET}");

    for r in runs {
        println!("  {:<22}  {:.6}    {}", r.seed, r.best_val_loss, r.best_val_epoch);
    }

    println!("\n  {lab}L_val{RESET}   min {val}{lo:.6}{RESET}  max {val}{hi:.6}{RESET}  spread {val}{:.2e}{RESET}", hi - lo);

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
        "  {lab}Params{RESET}  identical {val}{identical}{RESET}  ±1 on {val}{jitter}{RESET}  \
         wider on {val}{real}{RESET}  of {np}, total spread {val}{total}{RESET}"
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
        "  {lab}Spread{RESET}  sd over half a unit on {val}{unsettled}{RESET}  total sd {val}{total_deviation:.1}{RESET}  \
         {lab}(comparable across sweep sizes){RESET}"
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
            "  {lab}Top {LOAD_BEARING}{RESET}  identical {val}{same}{RESET}  ±1 on {val}{}{RESET}  \
             wider on {val}{wider}{RESET}, total spread {val}{spread}{RESET}",
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
        println!("  {lab}Widest{RESET}  {}", worst.join("   "));
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
        "\n  {lab}Furthest pair{RESET}  {val}{}{RESET} and {val}{}{RESET}, {val}{distance}{RESET} apart, \
         L_val {:.6} vs {:.6}\n  paste seed_{}_best.txt against seed_{}_best.txt",
        runs[a].seed, runs[b].seed, runs[a].best_val_loss, runs[b].best_val_loss, runs[a].seed, runs[b].seed
    );
}
