//! Main CMA-ES search tuning loop.
//!
//! Each epoch:
//!  1. Sample λ candidates from the current distribution.
//!  2. Partially impute fitness from the EloCache surrogate (≤25%).
//!  3. Play the remaining candidates against the current elite.
//!  4. Penalize by speed, OOB, and centering; feed the Elo cache.
//!  5. Update CMA-ES with the penalized fitness vector.
//!  6. If the best candidate beats `H2H_ELO_THRESHOLD`, challenge the elite.
//!  7. Periodically re-verify the elite against baseline.
//!  8. IPOP restart if σ collapses; σ boost if stagnation detected.

use std::{
    collections::BTreeMap,
    io::{Write, stdout},
    sync::atomic::{AtomicUsize, Ordering},
    time::Instant,
};

use rayon::prelude::*;
use soul::engine::search_params::{self, SearchParams};

use super::{
    cache::MatchCache,
    cmaes::{CmaEs, clamp_normalized, default_lambda},
    elo::{EloCache, elo_color},
    selfplay,
    storage::Checkpoint,
};
use crate::core::config::SearchTuneConfig;

/// Minimum Elo gain required to trigger a Head-to-Head validation match.
const H2H_ELO_THRESHOLD: f64 = 4.0;

/// # Panics
/// if opening file cannot be read or if internal logic fails.
pub fn run(openings_path: &str, config: &SearchTuneConfig, resume: bool) {
    let params = search_params::tunable_param_defs();
    let n = params.len();

    let lambda = default_lambda(n, config.population_scale);

    let mut cmaes = CmaEs::new_with_lambda(n, lambda, config.active_softness);

    // Convert default ParamDefs into normalized f64 vector
    let mut initial_mean = vec![0.0; n];
    for (i, p) in params.iter().enumerate() {
        initial_mean[i] = p.normalize(p.default);
    }
    cmaes.set_mean(&initial_mean);
    cmaes.set_sigma(config.sigma_init);

    let mut start_epoch = 1;

    let mut best_elo = 0.0;
    let mut best_params = cmaes.current_mean();
    let mut restart_count = 0usize;
    let mut epochs_without_improvement = 0usize;
    let mut last_best_elo = f64::NEG_INFINITY;
    let mut elo_cache = EloCache::default();
    let match_cache = MatchCache::new(config.tc.as_deref().unwrap_or("4+0.04"));

    // ── Adaptive Budget State ──
    // Tracks the optimal number of match pairs to play based on signal quality.
    let mut adaptive_pairs = config.pairs as f64 * 0.6; // Start at 60% budget

    if resume {
        match Checkpoint::load() {
            Ok(Some(ckpt)) => {
                start_epoch = ckpt.epoch + 1;
                best_elo = ckpt.best_elo;
                cmaes.set_mean(&ckpt.mean);
                cmaes.set_sigma(ckpt.sigma);

                cmaes.restore_state(ckpt.variances, ckpt.p_sigma, ckpt.p_c);

                best_params = ckpt.best_params;
                println!("\x1b[1;33m>> Resumed from epoch {} (Best: {best_elo:+.1} Elo)\x1b[0m", ckpt.epoch);
                println!();
            },
            Ok(None) => {
                println!("\x1b[33m>> No checkpoint found, starting fresh.\x1b[0m");
            },
            Err(e) => {
                eprintln!("\x1b[91m[!] Fatal Error reading checkpoint: {}\x1b[0m", e);
                std::process::exit(1);
            },
        }
    }
    println!("\x1b[90m ─── Configuration ───\x1b[0m");
    println!("  Openings:        {openings_path}");
    println!("  Time Control:    {}", config.tc.as_deref().unwrap_or("4+0.04"));
    println!("  Epochs:          {}", config.epochs);
    let lambda = cmaes.lambda();
    println!("  Workload:        {} candidates x {} pairs", lambda, config.pairs);
    println!("                   (Total {} games/gen at 100% budget)", lambda * config.pairs * 2);
    println!();
    println!("\x1b[90m ─── Soft Active CMA-ES (Separable) ───\x1b[0m");
    println!("  Population (λ):  {}", cmaes.lambda());
    println!("  Elite (μ):       {}", cmaes.lambda() / 2);
    println!("  Step size (σ₀):  {:.2}", cmaes.sigma());
    println!();
    println!("\x1b[90m ─── Parameters ({n}) ───\x1b[0m");
    for p in &params {
        println!(
            " \x1b[38;5;250m{:<22}\x1b[0m \x1b[1;37m{:>6}\x1b[0m   \x1b[90m{:>6} .. {:<6}\x1b[0m",
            p.name, p.default as i32, p.min as i32, p.max as i32
        );
    }
    println!("\n\x1b[1;32m>> Loading...\x1b[0m\n");

    let openings = match selfplay::load_openings(openings_path) {
        Ok(o) => {
            if o.is_empty() {
                eprintln!("\x1b[31mError:\x1b[0m Openings file is empty.");
                return;
            }

            println!("  Loaded {} openings.\n", o.len());
            o
        },

        Err(e) => {
            eprintln!("\x1b[31mError:\x1b[0m Failed to load openings: {e}");
            return;
        },
    };

    let mut rng = fastrand::Rng::new();
    let total_start = Instant::now();

    let mut verified_elite_params = best_params.clone();
    // On resume, verified_elite_state is the loaded CMA-ES state (same as cmaes).
    // If interrupted mid-reeval, the most recent state snapshot is conservatively lost.
    let mut verified_elite_state = cmaes.clone();
    let mut verified_elo = best_elo;

    // For fresh runs, ground the initial parameters to baseline immediately.
    // This prevents the "huh" effect where Best: +0.0 jumps to Best: -34.5 on the first win.
    if !resume {
        print!("\x1b[94m>> Grounding initial baseline Elo...\x1b[0m");
        let _ = stdout().flush();
        let elite = SearchParams::from_normalized(&best_params);
        let baseline = SearchParams::default();
        let (grounded_elo, ..) = selfplay::run_head_to_head(
            &elite,
            &baseline,
            &openings,
            config.h2h_pairs,
            config.h2h_tc.as_deref().unwrap_or("1.0+0.01"),
        );
        best_elo = grounded_elo;
        verified_elo = best_elo;
        last_best_elo = best_elo;
        println!("\r\x1b[K\x1b[90m>> Initial baseline grounded at {best_elo:+.1} Elo\x1b[0m\n");
    }

    for epoch in start_epoch..=config.epochs {
        let epoch_start = Instant::now();
        let lambda = cmaes.lambda();
        let epoch_start_best_elo = best_elo;

        // ── Adaptive Budgeting ──
        // The evaluation budget (matches per candidate) is scaled by the Signal-to-Noise Ratio.
        // If the mean shift is clear (high signal), we can get away with fewer matches.
        // As we converge and the signal shrinks, we automatically ramp up games to reduce noise.
        let effective_pairs = if epoch == start_epoch {
            // Cold start: use the config default for the first epoch
            config.pairs
        } else {
            // Signal = ||m_new - m_old|| / sigma
            let signal = cmaes.mean_shift_norm();
            let noise = 15.0; // Target SE threshold for budget expansion

            // M* = C · (Noise / Signal)^2
            // We use an EMA to smooth the transitions and clamp to [60%, 150%] of config.pairs.
            let optimal = (config.pairs as f64 * (noise / signal.max(1.0)).powi(2)).round();
            adaptive_pairs = 0.2_f64.mul_add(optimal, 0.8 * adaptive_pairs); // 20% alpha EMA

            let min_p = (config.pairs as f64 * 0.6) as usize;
            let max_p = (config.pairs as f64 * 1.5) as usize;
            (adaptive_pairs as usize).clamp(min_p, max_p)
        };

        // ── Sample Candidates ──
        let population = cmaes.sample_population(&mut rng);

        let opponent_norm = clamp_normalized(&best_params); // Anchor to the stable Elite
        let mut all_indices: Vec<usize> = (0..openings.len()).collect();
        rng.shuffle(&mut all_indices);

        let mut epoch_openings = Vec::with_capacity(effective_pairs);
        for i in 0..effective_pairs {
            epoch_openings.push(openings[all_indices[i % openings.len()]].clone());
        }

        let opponent_params = SearchParams::from_normalized(&opponent_norm);

        // ── Adaptive Surrogate Bandwidth ──
        // Silverman's rule naturally "zooms in" the kernel as the optimizer converges.
        // We blend it with the config value to ensure we don't start too narrow.
        let silverman_h = elo_cache.silverman_bandwidth();
        let adaptive_radius = if elo_cache.len() > 10 { silverman_h } else { config.smoothing_radius };

        // ── Surrogate-Assisted Rank Imputation ──
        // We use the EloCache to predict results for candidates in well-explored regions.
        // This saves massive amounts of compute without skewing the optimizer's distribution.
        let mut fitness_results = vec![(0.0, 0.0, 0, 0); lambda];
        let mut needs_play = Vec::new();

        // Softens the surrogate kernel: higher values extend the effective neighborhood.
        const SURROGATE_TEMPERATURE: f64 = 2.0;

        let mut min_surrogate_err = 1000.0;
        let mut candidates_with_err = Vec::with_capacity(lambda);

        for (i, candidate) in population.iter().enumerate() {
            let (surrogate_elo, surrogate_err) = match elo_cache.weighted_elo(
                candidate,
                &opponent_norm,
                cmaes.variances(),
                cmaes.sigma(),
                adaptive_radius,
                SURROGATE_TEMPERATURE,
            ) {
                // weight = 1 / (SE² + 1), so sum_weight ≈ Σ 1/SE².
                // surrogate_err = 1/√(sum_weight) therefore approximates the effective SE
                // in the same Elo units as the Pentanomial estimator. The < 15.0 threshold
                // corresponds to "confident enough that we trust the surrogate over a real match."
                Some((e, w)) => (e, (1.0 / w.max(1e-9).sqrt()).min(99.9)),
                None => (0.0, 99.9),
            };

            if surrogate_err < min_surrogate_err {
                min_surrogate_err = surrogate_err;
            }
            candidates_with_err.push((i, surrogate_elo, surrogate_err));
        }

        // Sort by confidence (lowest surrogate err first)
        candidates_with_err.sort_unstable_by(|a, b| a.2.total_cmp(&b.2));

        // Cap imputation at 25% to avoid chasing noise.
        // Also strictly require < 15.0 Elo std_err to even consider imputing.
        let max_imputed = (lambda as f64 * 0.25).max(1.0) as usize;
        let mut imputed_count = 0;

        for (i, surrogate_elo, surrogate_err) in candidates_with_err {
            if surrogate_err < 15.0 && imputed_count < max_imputed {
                fitness_results[i] = (surrogate_elo, surrogate_err, 0, 0);
                imputed_count += 1;
            } else {
                needs_play.push(i);
            }
        }

        let progress = AtomicUsize::new(0);
        let total_to_play = needs_play.len() * effective_pairs;

        macro_rules! fmt_progress {
            ($done:expr) => {
                format!(
                    "  Epoch {:>3}/{} | Budget: {:>3} | Pairs {:>3}/{} | Imputed: {:>2} | SE: {:>4.1} | Elo: ...",
                    epoch,
                    config.epochs,
                    effective_pairs,
                    $done,
                    total_to_play,
                    lambda - needs_play.len(),
                    min_surrogate_err
                )
            };
        }

        print!("{}", fmt_progress!(0));
        let _ = stdout().flush();

        // Physically play the matches for unknown candidates
        let played_results: Vec<(usize, f64, f64, u64, u64)> = needs_play
            .par_iter()
            .map(|&idx| {
                let candidate = &population[idx];
                let clamped: Vec<f64> = candidate.iter().map(|v| v.clamp(0.0, 1.0)).collect();
                let candidate_params = SearchParams::from_normalized(&clamped);

                let progress_ref = &progress;
                let on_pair = || {
                    let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    // Throttle updates to minimize lock contention on stdout().flush()
                    if done.is_multiple_of(10) || done == total_to_play {
                        print!("\r{}", fmt_progress!(done));
                        let _ = stdout().flush();
                    }
                };

                let req = super::cache::MatchRequest {
                    params: candidate_params,
                    normalized: &clamped,
                    opponent_params,
                    opponent_normalized: &opponent_norm,
                    openings: &epoch_openings,
                    min_pairs: effective_pairs,
                };
                let (penta, c_nodes, b_nodes) = match_cache.get_or_run(req, on_pair);

                (idx, penta.mle_elo(), penta.std_err_hybrid(), c_nodes, b_nodes)
            })
            .collect();

        for (idx, elo, err, c_nodes, b_nodes) in played_results {
            fitness_results[idx] = (elo, err, c_nodes, b_nodes);
            // Update cache with fresh measurement — remember who we played against!
            let weight = 1.0 / (err.powi(2) + 1.0);
            elo_cache.add(population[idx].clone(), opponent_norm.clone(), elo, weight);
        }

        let avg_std_err: f64 = fitness_results.iter().map(|&(_, err, ..)| err).sum::<f64>() / lambda as f64;

        // ── Penalization & Bayesian Consensus ──
        let penalized_elo: Vec<f64> = population
            .iter()
            .zip(&fitness_results)
            .map(|(candidate, (raw_elo, std_err, c_nodes, b_nodes))| {
                // — Efficiency Penalty —
                // We penalize slow nodes, but we DO NOT reward fast nodes to prevent the
                // optimizer from converging on instant-fail parameters.
                let tc = config.tc.as_deref().unwrap_or("4+0.04");
                let uses_clock = tc.contains('+') || tc.parse::<f64>().is_ok() || tc.starts_with("movetime=");

                let efficiency_ratio = if uses_clock {
                    (*c_nodes).max(1) as f64 / (*b_nodes).max(1) as f64
                } else {
                    (*b_nodes).max(1) as f64 / (*c_nodes).max(1) as f64
                };
                let efficiency_penalty = -efficiency_ratio.ln().min(0.0) * config.speed_penalty;

                // OOB Penalty: Heavy quadratic crush for boundary violations
                let oob_penalty: f64 = candidate
                    .iter()
                    .map(|&v| {
                        if v < 0.0 {
                            v.powi(2) * 5000.0
                        } else if v > 1.0 {
                            (v - 1.0).powi(2) * 5000.0
                        } else {
                            0.0
                        }
                    })
                    .sum();

                // ── Continuous Centering Penalty ──
                // Snap to integers to avoid plateau blindness.
                // We use the raw unsnapped value to provide a micro-gradient that pulls
                // the continuous vector toward the exact center of the discrete bucket.
                let mut centering_penalty = 0.0;
                for (i, &v) in candidate.iter().enumerate() {
                    let p = &params[i];
                    let raw = v.mul_add(p.max - p.min, p.min);
                    let snapped = p.denormalize(v);
                    let err_norm = (raw - snapped) / (p.max - p.min).max(1.0);
                    centering_penalty += err_norm.powi(2);
                }

                *raw_elo
                    - (config.confidence_factor * std_err) // Lower-confidence bound: penalize by k·SE to prefer statistically certain gains.
                    - efficiency_penalty
                    - oob_penalty
                    - (centering_penalty * config.centering_penalty)
            })
            .collect();

        let raw_elos: Vec<f64> = fitness_results.iter().map(|&(r, ..)| r).collect();
        cmaes.update(&population, &penalized_elo, &raw_elos, avg_std_err);

        // ── Learning Rate Adaptation (LRA) ──
        // Scale the global learning rate based on the estimated Signal-to-Noise Ratio.
        let snr = cmaes.update_snr();
        let target_snr = 0.5; // Baseline alpha for default lambda
        let damping = 0.1; // Prevents wild oscillations in eta

        // eta_{t+1} = eta_t · exp(damping · (snr / target_snr - 1))
        let snr_ratio = (snr / target_snr - 1.0).clamp(-1.0, 1.0);
        let adaptive_factor = (damping * snr_ratio).exp();
        let new_eta = cmaes.learning_rate() * adaptive_factor;
        cmaes.set_lr(new_eta);

        let (gen_best_idx, _) = penalized_elo.iter().enumerate().max_by(|(_, a), (_, b)| a.total_cmp(b)).unwrap();

        let gen_best_elo = fitness_results[gen_best_idx].0;
        let avg_raw_elo = fitness_results.iter().map(|&(r, ..)| r).sum::<f64>() / lambda as f64;

        let mut h2h_result: Option<(bool, f64)> = None;

        if gen_best_elo > H2H_ELO_THRESHOLD {
            let challenger = SearchParams::from_normalized(&clamp_normalized(&population[gen_best_idx]));
            let defender = SearchParams::from_normalized(&clamp_normalized(&best_params));

            print!("\r\x1b[K        \x1b[1;94m⟳ VALIDATING CHALLENGER...\x1b[0m");
            let _ = stdout().flush();

            let (h2h_elo, ..) = selfplay::run_head_to_head(
                &challenger,
                &defender,
                &openings,
                config.h2h_pairs,
                config.h2h_tc.as_deref().unwrap_or("1.0+0.01"),
            );

            if h2h_elo > 0.0 {
                // Challenger won. Update params and ground the Elo estimate against the baseline.
                best_params = clamp_normalized(&population[gen_best_idx]);

                // Snapshot the entire optimizer state for the new elite.
                verified_elite_params = best_params.clone();
                verified_elite_state = cmaes.clone();

                let elite = SearchParams::from_normalized(&best_params);
                let baseline = SearchParams::default();
                let (grounded_elo, ..) = selfplay::run_head_to_head(
                    &elite,
                    &baseline,
                    &openings,
                    config.h2h_pairs,
                    config.h2h_tc.as_deref().unwrap_or("1.0+0.01"),
                );

                // ── Dampened Grounding ──
                // Blend the H2H gain with the absolute grounding match.
                // This prevents a single noisy match from causing a massive absolute jump.
                best_elo = (best_elo + h2h_elo + grounded_elo) / 3.0;
                verified_elo = best_elo;
                epochs_without_improvement = 0;

                // Feed the grounding match back to the cache too
                let weight = 1.0 / (2.5f64.powi(2) + 1.0);
                elo_cache.add(best_params.clone(), SearchParams::to_normalized(), verified_elo, weight);
            }

            h2h_result = Some((h2h_elo > 0.0, h2h_elo));

            // Record H2H result in the cache – crucial feedback loop for the surrogate!
            // Assuming SE ≈ 2.5 for H2H (1000 pairs).
            let weight = 1.0 / (2.5f64.powi(2) + 1.0);
            elo_cache.add(population[gen_best_idx].clone(), opponent_norm.clone(), h2h_elo, weight);
        }

        let epoch_duration = epoch_start.elapsed();

        print!("\r\x1b[K");
        // Status color for the Epoch header:
        // If we are winning significantly (>20 Elo) but this generation stalled,
        // use a neutral 'Holding' color (Steel) instead of the negative blue/red.
        let status_color = if epoch_start_best_elo > 20.0 && gen_best_elo <= 0.0 {
            "\x1b[38;2;176;196;222m".to_string() // STEEL
        } else {
            elo_color(gen_best_elo)
        };

        let best_val_color = elo_color(epoch_start_best_elo);
        let avg_val_color = elo_color(avg_raw_elo);

        println!(
            "{status_color}Epoch {epoch:>3}\x1b[0m | \
             \x1b[90mBest:\x1b[0m {best_val_color}{epoch_start_best_elo:>+5.1}\x1b[0m | \
             \x1b[90mAvg:\x1b[0m {avg_val_color}{avg_raw_elo:>+5.1}\x1b[0m | \
             \x1b[90mBudget:\x1b[0m \x1b[33m{effective_pairs:>3}\x1b[0m | \
             \x1b[90mσ:\x1b[0m \x1b[36m{:.3}\x1b[0m | \
             \x1b[90mη:\x1b[0m \x1b[36m{:.2}\x1b[0m | \
             \x1b[90mTime:\x1b[0m \x1b[37m{:>5.1}s\x1b[0m",
            cmaes.sigma(),
            cmaes.learning_rate(),
            epoch_duration.as_secs_f64()
        );

        if let Some((won, h2h_elo)) = h2h_result {
            if won {
                println!(
                    "        └─ \x1b[38;2;255;215;0m✓\x1b[0m \x1b[1;96mH2H: Challenger wins ({h2h_elo:+.1} \
                     Elo) → New elite!\x1b[0m"
                );
            } else {
                println!(
                    "        └─ \x1b[38;2;255;100;100m✗\x1b[0m \x1b[38;2;255;180;80mH2H: Defender holds \
                     ({h2h_elo:+.1} Elo) → Keeping elite.\x1b[0m"
                );
            }
        }

        let best_clamped = clamp_normalized(&best_params);

        print!("        └─ ");
        let indent = "           ";
        let mut line_len = 11;

        for (i, param) in params.iter().enumerate() {
            let val = param.denormalize(best_clamped[i]).round() as i32;
            let entry = format!("{}={}", param.name, val);

            if i > 0 {
                if line_len + 2 + entry.len() > 110 {
                    println!(",");
                    print!("{}", indent);
                    line_len = indent.len();
                } else {
                    print!(", ");
                    line_len += 2;
                }
            }
            print!("{}", entry);
            line_len += entry.len();
        }
        println!();

        // if sigma collapses, restart with 2x population from best
        if cmaes.sigma() < config.min_sigma && restart_count < config.max_restarts {
            restart_count += 1;
            let old_lambda = cmaes.lambda();
            let new_lambda = old_lambda * 2;
            println!(
                "\x1b[93m>> IPOP Restart #{restart_count}: σ collapsed ({:.2e} < {:.0e}), λ: {old_lambda} → \
                 {new_lambda}\x1b[0m",
                cmaes.sigma(),
                config.min_sigma
            );
            cmaes.restart_from(best_params.clone(), new_lambda, config.sigma_restart);
            // After restart: old Elo estimates are relative to a different optimizer region.
            // Clear the cache to let the surrogate rebuild cleanly from the new mean.
            elo_cache.clear();
            epochs_without_improvement = 0;

            // Update the verified state to reflect the restart
            verified_elite_state = cmaes.clone();
        }

        // if no improvement for N epochs, boost σ
        if best_elo > last_best_elo {
            epochs_without_improvement = 0;
            last_best_elo = best_elo;
        } else {
            epochs_without_improvement += 1;
        }

        if epochs_without_improvement >= config.stagnation_threshold {
            let old_sigma = cmaes.sigma();
            cmaes.set_sigma((old_sigma * config.sigma_boost_factor).min(0.5));
            println!(
                "\x1b[93m>> Stagnation detected ({epochs_without_improvement} epochs): σ boosted \
                 {old_sigma:.3} → {:.3}\x1b[0m",
                cmaes.sigma()
            );
            epochs_without_improvement = 0;

            // Update the verified state to reflect the sigma boost
            verified_elite_state = cmaes.clone();
        }

        // — Periodic Elite Cross-Examination —
        // Re-verify elite against baseline to catch flukes.
        if epoch % config.reeval_interval == 0 && epoch > 1 {
            print!("\x1b[94m        └─ Re-evaluating elite...\x1b[0m");
            let _ = stdout().flush();
            let elite = SearchParams::from_normalized(&clamp_normalized(&best_params));
            let baseline = SearchParams::default();
            let (reeval_elo, ..) = selfplay::run_head_to_head(
                &elite,
                &baseline,
                &openings,
                config.h2h_pairs,
                config.h2h_tc.as_deref().unwrap_or("1.0+0.01"),
            );

            let color = elo_color(reeval_elo);
            println!("\r\x1b[K{color}        └─ Elite re-eval: {reeval_elo:+.1} Elo vs baseline\x1b[0m");

            // If the elite regresses during re-evaluation, rollback to the last
            // verified elite parameters and state.
            if reeval_elo < -25.0 {
                println!("\x1b[38;2;255;100;100m        └─ ⚠ Elite regressed! Reverting to last likely best elite.\x1b[0m");
                best_params.clone_from(&verified_elite_params);
                cmaes = verified_elite_state.clone(); // Full state rollback!
                best_elo = verified_elo;
                epochs_without_improvement = 0;
            } else {
                best_elo = (best_elo + reeval_elo) * 0.5;
                verified_elite_params.clone_from(&best_params);
                verified_elite_state = cmaes.clone();
                verified_elo = best_elo;
            }
        }

        // ── Save Checkpoint ──
        let mut best_values_map = BTreeMap::new();
        let current_best_clamped = clamp_normalized(&best_params);
        for (i, param) in params.iter().enumerate() {
            let best_val = param.denormalize(current_best_clamped[i]).round() as i32;
            best_values_map.insert(param.name.to_string(), best_val);
        }

        let ckpt = Checkpoint {
            version: super::storage::CHECKPOINT_VERSION,
            epoch,
            best_elo,
            best_params: best_params.clone(),
            mean: cmaes.mean().to_vec(),
            sigma: cmaes.sigma(),
            variances: cmaes.variances().to_vec(),
            p_sigma: cmaes.p_sigma().to_vec(),
            p_c: cmaes.p_c().to_vec(),
            best_values: best_values_map,
        };

        if let Err(e) = ckpt.save() {
            eprintln!("\n\x1b[31m[!] Failed to save checkpoint: {e}\x1b[0m");
        }
    }

    println!("\n\x1b[1;36m>> Search Tuning Complete ({:.1}s)\x1b[0m", total_start.elapsed().as_secs_f64());
    println!("\x1b[1;33m>> Peak Elo (noisy estimate): {best_elo:+.1}\x1b[0m");
    println!("\x1b[90m   Elo cache: {} samples collected\x1b[0m", elo_cache.len());

    let (mc_entries, mc_pairs, mc_hits, mc_misses) = match_cache.stats();
    println!(
        "\x1b[90m   Match cache: {mc_entries} entries, {mc_pairs} pairs | hits: {mc_hits}, misses: \
         {mc_misses}\x1b[0m"
    );

    let final_radius = if elo_cache.len() > 10 { elo_cache.silverman_bandwidth() } else { config.smoothing_radius };
    if let Some((denoised, count)) = elo_cache.denoised_elo(&best_params, final_radius) {
        println!("\x1b[90m   Denoised estimate: {denoised:+.1} Elo (from {count} nearby samples)\x1b[0m");
    }

    println!(
        "\n\x1b[1;35m>> Running validation ({} pairs each at tc: {})...\x1b[0m",
        config.validation_pairs,
        config.val_tc.as_deref().unwrap_or("1.0+0.01")
    );

    let best_final = SearchParams::from_normalized(&clamp_normalized(&best_params));
    let baseline = SearchParams::default();

    let (best_vs_baseline, ..) = selfplay::run_head_to_head(
        &best_final,
        &baseline,
        &openings,
        config.validation_pairs,
        config.val_tc.as_deref().unwrap_or("1.0+0.01"),
    );

    println!("\n\x1b[1;36m>> Validation Results:\x1b[0m");
    println!("   Best vs Baseline: {best_vs_baseline:+.1} Elo");

    println!("\n\x1b[1;36m>> Parameter Sensitivity Report (Relative to matrix max):\x1b[0m");

    let variances = cmaes.variances();
    let max_var = variances.iter().copied().fold(0.0, f64::max);
    let sigma = cmaes.sigma();

    for (i, param) in params.iter().enumerate() {
        let v = variances[i];

        // 1. Relative variance dictates the shape of the landscape.
        // A low relative variance means CMA-ES hit a steep wall in this dimension.
        let rel_var = v / max_var.max(1e-6);

        // 2. Physical search radius in normalized [0, 1] space.
        let search_radius = sigma * v.sqrt();

        // A parameter is considered highly sensitive if it has been squashed relative to others,
        // OR if the entire search radius has physically collapsed into a tight needle.
        // We use 5 tiers to gracefully show differentiation even in the early epochs.
        let (sensitivity, color) = if search_radius < 0.015 || rel_var < 0.20 {
            ("Critical", "\x1b[31;1m") // Red
        } else if search_radius < 0.04 || rel_var < 0.45 {
            ("High", "\x1b[33;1m") // Yellow
        } else if search_radius < 0.08 || rel_var < 0.70 {
            ("Moderate", "\x1b[32;1m") // Green
        } else if search_radius < 0.12 || rel_var < 0.90 {
            ("Low", "\x1b[36m") // Cyan
        } else {
            ("Dead", "\x1b[90m") // Gray
        };

        println!(
            "   {:<18} : {}{:<18}\x1b[0m \x1b[90m(Rel Var: {:.2} | Radius: ±{:.3})\x1b[0m",
            param.name, color, sensitivity, rel_var, search_radius
        );
    }

    println!("\n// --- Best Values ({best_vs_baseline:+.1} vs baseline) ---");
    let best_clamped_final = clamp_normalized(&best_params);
    for (i, param) in params.iter().enumerate() {
        let value = param.denormalize(best_clamped_final[i]).round() as i32;
        println!("{:<14} = {value},", param.name);
    }
    println!("// -------------------------------\n");
}
