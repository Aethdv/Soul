//! Parameter variance across random seeds ("seed spread").
//!
//! Evaluates whether differences between retuned weights reflect data-driven
//! signal or optimization noise (stochastic batch order, initialization drift).
//! Reports the spread in validation loss and in the integer weights themselves.

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
    palette::{LAB, RESET, VAL},
    report::fmt_loss,
    run::ARTIFACT_DIR,
};

/// Number of top loss-sensitive parameters highlighted individually in the report.
const RANKED_PARAM_COUNT: usize = 50;

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
    /// Exponential moving average of per-parameter absolute gradients.
    #[serde(default)]
    sensitivity: Vec<f64>,
}

pub fn run_seed_spread(dataset: &str, config_path: &str, epochs: usize, count: usize, log_path: &str) {
    let mut rng = fastrand::Rng::with_seed(0xA5EE_D000);
    let seeds: Vec<u64> = (0..count).map(|_| rng.u64(..)).collect();

    println!("\n{LAB}Seed spread:{RESET} {VAL}{count}{RESET} seeds × {VAL}{epochs}{RESET} epochs on {dataset}\n");
    let mut failures = 0usize;
    for (idx, &seed) in seeds.iter().enumerate() {
        let start = Instant::now();
        let success = spawn_trial(dataset, config_path, epochs, log_path, &[("--seed", seed.to_string())]);
        let elapsed_sec = start.elapsed().as_secs_f32();

        let preserved_artifact = if success {
            fs::copy(trial_dir().join("evaltune_best.txt"), format!("seed_{seed}_best.txt")).is_ok()
        } else {
            false
        };

        failures += usize::from(!success);
        let status = if success { "done" } else { "FAILED" };
        let note = if success && !preserved_artifact { "  (no evaltune_best.txt found)" } else { "" };
        println!("  [{}/{count}] {seed:<22} {status:<8} {elapsed_sec:.1}s{note}", idx + 1);
    }

    // A failed trial's artifact is never copied out, so its working directory is all that is left of it.
    if failures == 0 {
        let _ = fs::remove_dir_all(trial_dir());
    } else {
        eprintln!("  {failures} of {count} trials failed; their working directory is {}", trial_dir().display());
    }

    let runs = collect_completed_runs(log_path, &seeds);
    if runs.len() < 2 {
        eprintln!("Only {} of {count} runs reported valid results; comparison requires at least 2.", runs.len());
        return;
    }
    print_spread_report(&runs);
}

/// Runs each trial out-of-process to isolate crashes and prevent artifact collisions.
/// Results are written directly to `log_path`.
pub fn spawn_trial(dataset: &str, config_path: &str, epochs: usize, log_path: &str, extra_args: &[(&str, String)]) -> bool {
    let Ok(exe_path) = env::current_exe() else {
        eprintln!("Failed to locate binary path to spawn trial process.");
        return false;
    };

    let work_dir = trial_dir();
    if let Err(err) = fs::create_dir_all(&work_dir) {
        eprintln!("Failed to create trial working directory {}: {err}", work_dir.display());
        return false;
    }

    let mut cmd = Command::new(exe_path);
    cmd.env(ARTIFACT_DIR, &work_dir);
    cmd.arg("--dataset").arg(dataset);
    cmd.arg("--config").arg(config_path);
    cmd.arg("--epochs").arg(epochs.to_string());
    cmd.arg("--log").arg(log_path);
    for (flag, val) in extra_args {
        cmd.arg(flag).arg(val);
    }

    // The parent prints its own per-trial line; the child's progress would interleave with it.
    let status = cmd.stdout(process::Stdio::null()).stderr(process::Stdio::inherit()).status();
    status.is_ok_and(|s| s.success())
}

fn trial_dir() -> PathBuf { env::temp_dir().join(format!("evaltune_trial_{}", process::id())) }

#[must_use]
pub fn last_best_val(log_path: &str) -> Option<f64> { read_final_records(log_path).last().and_then(|record| record.best_val_loss) }

/// Parses all `final` records from an append-only JSON-lines log file.
fn read_final_records(log_path: &str) -> Vec<FinalRecord> {
    let Ok(content) = fs::read_to_string(log_path) else {
        return Vec::new();
    };

    content
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .filter(|val| val.get("event").and_then(serde_json::Value::as_str) == Some("final"))
        .filter_map(|val| serde_json::from_value::<FinalRecord>(val).ok())
        .collect()
}

/// Extracts the latest run record for each requested seed, preserving request order.
fn collect_completed_runs(log_path: &str, seeds: &[u64]) -> Vec<FinalRecord> {
    let mut resolved: Vec<Option<FinalRecord>> = seeds.iter().map(|_| None).collect();
    for record in read_final_records(log_path) {
        if let Some(pos) = seeds.iter().position(|&seed| seed == record.seed) {
            resolved[pos] = Some(record);
        }
    }
    resolved.into_iter().flatten().collect()
}

fn print_spread_report(runs: &[FinalRecord]) {
    // Differing validation split seeds invalidate cross-run loss comparisons.
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

    // Per-parameter min-max spread. ±0 indicates data saturation; ±1 is rounding quantization noise.
    let spans: Vec<i32> = (0..num_params)
        .map(|param_idx| {
            let values = runs.iter().map(|r| r.params[param_idx]);
            let (min_val, max_val) = values.fold((i32::MAX, i32::MIN), |(lo, hi), v| (lo.min(v), hi.max(v)));
            max_val - min_val
        })
        .collect();

    let identical_count = spans.iter().filter(|&&span| span == 0).count();
    let jitter_count = spans.iter().filter(|&&span| span == 1).count();
    let wide_count = spans.iter().filter(|&&span| span > 1).count();
    let total_span_sum: i32 = spans.iter().sum();

    println!(
        "  {LAB}Params{RESET}  identical {VAL}{identical_count}{RESET}  ±1 on {VAL}{jitter_count}{RESET}  \
         wider on {VAL}{wide_count}{RESET}  of {num_params}, total spread {VAL}{total_span_sum}{RESET}"
    );

    // Sample standard deviation (scale-invariant with respect to run count, unlike raw span).
    // Bessel's correction: the 1/n form is biased low by (n-1)/n, so it would shrink with the
    // very run count this figure exists to be compared across.
    let sample_size = runs.len() as f64;
    let std_devs: Vec<f64> = (0..num_params)
        .map(|param_idx| {
            let mean = runs.iter().map(|r| f64::from(r.params[param_idx])).sum::<f64>() / sample_size;
            let variance = runs.iter().map(|r| (f64::from(r.params[param_idx]) - mean).powi(2)).sum::<f64>() / (sample_size - 1.0);
            variance.sqrt()
        })
        .collect();

    let high_variance_count = std_devs.iter().filter(|&&sd| sd > 0.5).count();
    let cumulative_std_dev: f64 = std_devs.iter().sum();
    println!(
        "  {LAB}Spread{RESET}  sd > 0.5 on {VAL}{high_variance_count}{RESET}  total sd {VAL}{cumulative_std_dev:.1}{RESET}  \
         {LAB}(comparable across sample sizes){RESET}"
    );

    // Isolate variance on top loss-sensitive parameters.
    if runs.iter().all(|r| r.sensitivity.len() == num_params) {
        let mean_sensitivity: Vec<f64> = (0..num_params)
            .map(|idx| runs.iter().map(|r| r.sensitivity[idx]).sum::<f64>() / sample_size)
            .collect();
        let mut sorted_by_sensitivity: Vec<usize> = (0..num_params).collect();
        sorted_by_sensitivity.sort_unstable_by(|&a, &b| mean_sensitivity[b].total_cmp(&mean_sensitivity[a]));

        let top_subset = &sorted_by_sensitivity[..RANKED_PARAM_COUNT.min(num_params)];
        let top_identical = top_subset.iter().filter(|&&idx| spans[idx] == 0).count();
        let top_wide = top_subset.iter().filter(|&&idx| spans[idx] > 1).count();
        let top_spread_sum: i32 = top_subset.iter().map(|&idx| spans[idx]).sum();

        println!(
            "  {LAB}Top {RANKED_PARAM_COUNT}{RESET}  identical {VAL}{top_identical}{RESET}  ±1 on {VAL}{}{RESET}  \
             wider on {VAL}{top_wide}{RESET}, total spread {VAL}{top_spread_sum}{RESET}",
            top_subset.len() - top_identical - top_wide
        );
    }

    let mut widest: Vec<usize> = (0..num_params).collect();
    widest.sort_unstable_by_key(|&idx| Reverse(spans[idx]));
    let widest_labels: Vec<String> = widest
        .iter()
        .take(6)
        .filter(|&&idx| spans[idx] > 1)
        .map(|&idx| format!("{} {}", params.get(idx).map_or("?", |p| p.name.as_str()), spans[idx]))
        .collect();

    if !widest_labels.is_empty() {
        println!("  {LAB}Widest{RESET}  {}", widest_labels.join("   "));
    }

    // Identify maximally divergent parameter pair (L1 norm) for A/B game validation.
    let mut max_divergent_pair = (0usize, 0usize, 0i64);
    for i in 0..runs.len() {
        for j in i + 1..runs.len() {
            let l1_distance: i64 = runs[i].params.iter().zip(&runs[j].params).map(|(&a, &b)| i64::from((a - b).abs())).sum();
            if l1_distance > max_divergent_pair.2 {
                max_divergent_pair = (i, j, l1_distance);
            }
        }
    }

    let (run_a, run_b, max_distance) = max_divergent_pair;
    println!(
        "\n  {LAB}Furthest pair{RESET}  {VAL}{}{RESET} and {VAL}{}{RESET}, {VAL}{max_distance}{RESET} apart, \
         L_val {} vs {}\n  diff seed_{}_best.txt against seed_{}_best.txt",
        runs[run_a].seed,
        runs[run_b].seed,
        fmt_loss(runs[run_a].best_val_loss),
        fmt_loss(runs[run_b].best_val_loss),
        runs[run_a].seed,
        runs[run_b].seed
    );
}
