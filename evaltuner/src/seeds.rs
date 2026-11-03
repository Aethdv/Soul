//! Out-of-process multi-run drivers: random-seed variance (`seed-spread`) and
//! `lr_mult` grid refinement (`sweep-lr-mult`).

use std::{
    cmp::Reverse,
    env, fs,
    path::PathBuf,
    process::{self, Command},
    time::Instant,
};

use serde::Deserialize;

use crate::{
    engine::eval_params,
    report::{LAB, RESET, VAL, fmt_loss},
    run::ARTIFACT_DIR,
};

const TOP_SENSITIVE_COUNT: usize = 50;
const SWEEP_HEADER: &str = "  lr_mult    Best L_val    Time\n  -------    ----------    ----";

pub(crate) fn run_seed_spread(dataset: &str, config_path: &str, epochs: usize, count: usize, log_path: &str) {
    let mut rng = fastrand::Rng::with_seed(0xA5EE_D000);
    let seeds: Vec<u64> = (0..count).map(|_| rng.u64(..)).collect();

    println!("\n{LAB}Seed spread:{RESET} {VAL}{count}{RESET} seeds × {VAL}{epochs}{RESET} epochs on {dataset}\n");
    let mut failures = 0usize;
    for (idx, &seed) in seeds.iter().enumerate() {
        let start = Instant::now();
        let ok = spawn_trial(dataset, config_path, epochs, log_path, &[("--seed", seed.to_string())]);
        let copied = ok && fs::copy(trial_dir().join("evaltune_best.txt"), format!("seed_{seed}_best.txt")).is_ok();
        failures += usize::from(!ok);
        println!(
            "  [{}/{count}] {seed:<22} {:<8} {:.1}s{}",
            idx + 1,
            if ok { "done" } else { "FAILED" },
            start.elapsed().as_secs_f32(),
            if ok && !copied { "  (no evaltune_best.txt found)" } else { "" }
        );
    }

    // A failed trial's artifact is never copied out, so its working directory is all that is left of it.
    if failures == 0 {
        let _ = fs::remove_dir_all(trial_dir());
    } else {
        eprintln!("  {failures} of {count} trials failed; their working directory is {}", trial_dir().display());
    }

    let mut resolved: Vec<Option<FinalRecord>> = seeds.iter().map(|_| None).collect();
    for record in read_final_records(log_path) {
        if let Some(pos) = seeds.iter().position(|&s| s == record.seed) {
            resolved[pos] = Some(record);
        }
    }

    let runs: Vec<FinalRecord> = resolved.into_iter().flatten().collect();
    if runs.len() < 2 {
        eprintln!("Only {} of {count} runs reported valid results; comparison requires at least 2.", runs.len());
        return;
    }
    print_spread_report(&runs);
}

/// Each round doubles the epoch budget and re-centers a three-point grid on the winner.
pub(crate) fn sweep_lr_mult(
    dataset: &str,
    config_path: &str,
    values: Option<Vec<f64>>,
    (min, max, count): (f64, f64, usize),
    base_epochs: usize,
    refine_rounds: usize,
    seed: Option<u64>,
) {
    let mut grid = match values {
        Some(v) if !v.is_empty() => v,
        _ => log_space(min, max, count),
    };
    let mut results: Vec<(f64, f64, f32)> = Vec::new();
    let mut best = (f64::MAX, grid[0]);

    let print_row = |lr_mult: f64, loss: f64, elapsed: f32| {
        let loss_str = if loss == f64::MAX { "FAILED".to_string() } else { format!("{loss:>10.6}") };
        println!("  {lr_mult:>7.4}    {loss_str}    {elapsed:.1}s");
    };

    for round in 0..=refine_rounds {
        let epochs = base_epochs * (1 << round);
        let log_step = if grid.len() > 1 { (grid[grid.len() - 1].ln() - grid[0].ln()) / (grid.len() - 1) as f64 } else { 0.5 };
        println!("── Round {round}: {epochs} epochs\n{SWEEP_HEADER}");

        for &lr_mult in &grid {
            let (loss, duration) = run_sweep_trial(lr_mult, dataset, config_path, epochs, seed);
            print_row(lr_mult, loss, duration);
            results.push((lr_mult, loss, duration));

            if loss < best.0 {
                best = (loss, lr_mult);
            }
        }

        if round < refine_rounds && grid.len() > 1 {
            let (half, lb) = (log_step / 2.0, best.1.ln());
            grid = [-half, 0.0, half].into_iter().map(|d| (lb + d).exp().clamp(min, max)).collect();
        }
    }

    results.sort_by(|a, b| a.1.total_cmp(&b.1));
    println!("\n  Sorted by L_val:\n{SWEEP_HEADER}");
    for &(lr_mult, loss, duration) in &results {
        print_row(lr_mult, loss, duration);
    }
    println!("\nBest lr_mult = {:.4} (L_val = {:.6})", best.1, best.0);
}

#[derive(Deserialize)]
struct FinalRecord {
    seed: u64,
    #[serde(default)]
    split_seed: Option<u64>,
    #[serde(default)]
    best_val_loss: Option<f64>,
    best_val_epoch: usize,
    #[serde(rename = "best_val_params", alias = "params")]
    params: Vec<i32>,
    #[serde(default)]
    sensitivity: Vec<f64>,
}

fn log_space(min: f64, max: f64, points: usize) -> Vec<f64> {
    if points <= 1 {
        return vec![min];
    }
    let (lo, hi) = (min.ln(), max.ln());
    (0..points).map(|i| (lo + (hi - lo) * i as f64 / (points - 1) as f64).exp()).collect()
}

fn run_sweep_trial(lr_mult: f64, dataset: &str, config_path: &str, epochs: usize, seed: Option<u64>) -> (f64, f32) {
    let log_str = env::temp_dir()
        .join(format!("sweep_lr_mult_{}.jsonl", process::id()))
        .to_string_lossy()
        .into_owned();
    let _ = fs::remove_file(&log_str);
    let mut extra = vec![("--lr-mult", lr_mult.to_string())];

    if let Some(s) = seed {
        extra.push(("--seed", s.to_string()));
    }

    let start = Instant::now();
    let ok = spawn_trial(dataset, config_path, epochs, &log_str, &extra);
    let elapsed = start.elapsed().as_secs_f32();
    let val_loss = if ok {
        read_final_records(&log_str).last().and_then(|r| r.best_val_loss).unwrap_or(f64::MAX)
    } else {
        f64::MAX
    };
    let _ = fs::remove_file(&log_str);
    (val_loss, elapsed)
}

fn spawn_trial(dataset: &str, config_path: &str, epochs: usize, log_path: &str, extra_args: &[(&str, String)]) -> bool {
    let Ok(exe) = env::current_exe() else {
        eprintln!("Failed to locate binary path to spawn trial process.");
        return false;
    };

    let work_dir = trial_dir();
    if let Err(err) = fs::create_dir_all(&work_dir) {
        eprintln!("Failed to create trial working directory {}: {err}", work_dir.display());
        return false;
    }

    let mut cmd = Command::new(exe);
    cmd.env(ARTIFACT_DIR, &work_dir).args([
        "--dataset",
        dataset,
        "--config",
        config_path,
        "--epochs",
        &epochs.to_string(),
        "--log",
        log_path,
    ]);

    for (flag, val) in extra_args {
        cmd.arg(flag).arg(val);
    }
    // The parent prints its own per-trial line; the child's progress would interleave with it.
    cmd.stdout(process::Stdio::null())
        .stderr(process::Stdio::inherit())
        .status()
        .is_ok_and(|s| s.success())
}

fn trial_dir() -> PathBuf { env::temp_dir().join(format!("evaltune_trial_{}", process::id())) }

fn read_final_records(log_path: &str) -> Vec<FinalRecord> {
    fs::read_to_string(log_path)
        .unwrap_or_default()
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|val| val.get("event").and_then(serde_json::Value::as_str) == Some("final"))
        .filter_map(|val| serde_json::from_value::<FinalRecord>(val).ok())
        .collect()
}

fn span_counts(indices: impl Iterator<Item = usize>, spans: &[i32]) -> (usize, usize, usize, i32) {
    let (mut n, mut identical, mut wide, mut total) = (0usize, 0usize, 0usize, 0i32);
    for i in indices {
        n += 1;
        identical += usize::from(spans[i] == 0);
        wide += usize::from(spans[i] > 1);
        total += spans[i];
    }
    (identical, n - identical - wide, wide, total)
}

fn print_spread_report(runs: &[FinalRecord]) {
    if runs.iter().any(|r| r.split_seed != runs[0].split_seed) {
        eprintln!("  Warning: Validation splits differed across runs; loss differences reflect partition shifts.");
    }

    let val_losses: Vec<f64> = runs.iter().filter_map(|r| r.best_val_loss).collect();
    if val_losses.len() < runs.len() {
        eprintln!("  Warning: Some runs lack validation splits (marked with '—').");
    }

    println!("\n  {LAB}seed{RESET}                    {LAB}L_val{RESET}       {LAB}epoch{RESET}");
    for run in runs {
        println!("  {:<22}  {}    {}", run.seed, fmt_loss(run.best_val_loss), run.best_val_epoch);
    }

    if val_losses.is_empty() {
        println!("\n  {LAB}L_val{RESET}   (no validation loss available)");
    } else {
        let min_loss = val_losses.iter().copied().fold(f64::MAX, f64::min);
        let max_loss = val_losses.iter().copied().fold(f64::MIN, f64::max);
        println!(
            "\n  {LAB}L_val{RESET}   min {VAL}{min_loss:.6}{RESET}  max {VAL}{max_loss:.6}{RESET}  spread {VAL}{:.2e}{RESET}",
            max_loss - min_loss
        );
    }

    let params = eval_params::collect_parameters();
    let num_params = runs[0].params.len();

    if runs.iter().any(|r| r.params.len() != num_params) {
        eprintln!("  Error: Parameter layout mismatch across runs. Log contains mixed configurations.");
        return;
    }

    let spans: Vec<i32> = (0..num_params)
        .map(|i| {
            let (lo, hi) = runs
                .iter()
                .map(|r| r.params[i])
                .fold((i32::MAX, i32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
            hi - lo
        })
        .collect();

    let (identical, jitter, wide, total_span) = span_counts(0..num_params, &spans);
    println!(
        "  {LAB}Params{RESET}  identical {VAL}{identical}{RESET}  ±1 on {VAL}{jitter}{RESET}  \
         wider on {VAL}{wide}{RESET}  of {num_params}, total spread {VAL}{total_span}{RESET}"
    );

    // Bessel's correction keeps the figure comparable across run counts.
    let n = runs.len() as f64;
    let std_devs: Vec<f64> = (0..num_params)
        .map(|i| {
            let mean = runs.iter().map(|r| f64::from(r.params[i])).sum::<f64>() / n;
            (runs.iter().map(|r| (f64::from(r.params[i]) - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt()
        })
        .collect();
    println!(
        "  {LAB}Spread{RESET}  sd > 0.5 on {VAL}{}{RESET}  total sd {VAL}{:.1}{RESET}  {LAB}(comparable across sample sizes){RESET}",
        std_devs.iter().filter(|&&sd| sd > 0.5).count(),
        std_devs.iter().sum::<f64>()
    );

    if runs.iter().all(|r| r.sensitivity.len() == num_params) {
        let mean_sens: Vec<f64> = (0..num_params).map(|i| runs.iter().map(|r| r.sensitivity[i]).sum::<f64>() / n).collect();
        let mut ranked: Vec<usize> = (0..num_params).collect();
        ranked.sort_unstable_by(|&a, &b| mean_sens[b].total_cmp(&mean_sens[a]));
        let (ti, tj, tw, ts) = span_counts(ranked[..TOP_SENSITIVE_COUNT.min(num_params)].iter().copied(), &spans);
        println!(
            "  {LAB}Top {TOP_SENSITIVE_COUNT}{RESET}  identical {VAL}{ti}{RESET}  ±1 on {VAL}{tj}{RESET}  \
             wider on {VAL}{tw}{RESET}, total spread {VAL}{ts}{RESET}"
        );
    }

    let mut widest: Vec<usize> = (0..num_params).collect();
    widest.sort_unstable_by_key(|&i| Reverse(spans[i]));
    let widest_labels: Vec<String> = widest
        .into_iter()
        .take(6)
        .filter(|&i| spans[i] > 1)
        .map(|i| format!("{} {}", params.get(i).map_or("?", |p| p.name.as_str()), spans[i]))
        .collect();
    if !widest_labels.is_empty() {
        println!("  {LAB}Widest{RESET}  {}", widest_labels.join("   "));
    }

    let mut furthest = (0usize, 0usize, 0i64);

    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            let l1: i64 = runs[i].params.iter().zip(&runs[j].params).map(|(&a, &b)| i64::from((a - b).abs())).sum();
            if l1 > furthest.2 {
                furthest = (i, j, l1);
            }
        }
    }

    let (a, b, dist) = furthest;
    println!(
        "\n  {LAB}Furthest pair{RESET}  {VAL}{}{RESET} and {VAL}{}{RESET}, {VAL}{dist}{RESET} apart, \
         L_val {} vs {}\n  diff seed_{}_best.txt against seed_{}_best.txt",
        runs[a].seed,
        runs[b].seed,
        fmt_loss(runs[a].best_val_loss),
        fmt_loss(runs[b].best_val_loss),
        runs[a].seed,
        runs[b].seed
    );
}
