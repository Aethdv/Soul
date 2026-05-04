use std::{
    fs::File,
    io::{BufWriter, Write},
    time::Instant,
};

use rayon::prelude::*;
use soul::{
    core::psqt,
    engine::eval_params::{self, Tunable},
};

use super::{lion::Lion, loader, report::*, storage::*, tape, training::*};
use crate::core::{
    config::{EvalTuneConfig, LrScheduleConfig},
    logger::JsonLogger,
};

/// Hard clamp for mobility parameters to prevent drift from unbounded features.
const MOB_CLAMP: f64 = 100.0;

pub fn run(dataset_path: Option<&str>, config: &EvalTuneConfig, resume_path: Option<&str>) {
    let total_start = Instant::now();

    // 1. Enable FTZ/DAZ on the MAIN thread immediately.
    #[cfg(target_arch = "x86_64")]
    unsafe {
        enable_ftz_daz();
    }

    // 2. Configure the Rayon thread pool (catches new worker threads).
    rayon::ThreadPoolBuilder::new()
        .start_handler(|_| unsafe {
            #[cfg(target_arch = "x86_64")]
            enable_ftz_daz();
        })
        .build_global()
        .ok(); // because it might already be initialized

    let paths = resolve_dataset_paths(dataset_path.unwrap_or("default"));
    let Some(paths) = paths else {
        return;
    };
    let path_str = paths.join(",");
    let is_encoded = paths.iter().all(|p| p.ends_with(".soul") || p.ends_with(".soul.zst"));

    if is_encoded {
        run_encoded(&paths, config, resume_path);
    } else {
        println!("Loading raw datasets: {path_str}");
        run_raw(&paths, config, resume_path);
    }

    let elapsed = total_start.elapsed().as_secs_f32();
    println!("\n\x1b[93mDone in {elapsed:.2}s\x1b[0m");
}

/// Enable Flush-to-Zero and Denormals-are-Zero for performance.
#[cfg(target_arch = "x86_64")]
#[inline]
unsafe fn enable_ftz_daz() {
    use std::arch::asm;
    let mut mxcsr: u32 = 0;
    unsafe {
        asm!("stmxcsr [{}]", in(reg) &mut mxcsr, options(nostack, preserves_flags));
        mxcsr |= 0x8040; // FTZ | DAZ
        asm!("ldmxcsr [{}]", in(reg) &mxcsr, options(nostack, preserves_flags));
    }
}

/// Training loop for the `.soul.zst` encoded dataset path.
///
/// Encoded entries carry pre-computed features — no attack generation or board
/// analysis at training time. Uses `TrainableEntry` trait dispatch for gradient
/// computation.
fn run_encoded(paths: &[String], config: &EvalTuneConfig, resume_path: Option<&str>) {
    let mut entries = Vec::new();
    for path in paths {
        println!("Loading encoded dataset: {path}");
        let mut file_entries = loader::load_encoded(path).expect("Failed to load .soul dataset");
        entries.append(&mut file_entries);
    }

    fastrand::shuffle(&mut entries);

    let val_count = entries.len() / 10;
    let train_count = entries.len() - val_count;
    let (train, val) = entries.split_at(train_count);

    print_dataset_stats(train, val, entries.len(), |e| {
        if e.original_stm == soul::tools::dataset::STM_WHITE { e.result as f64 } else { 1.0 - e.result as f64 }
    });

    // Batch gradient closure.
    // Uses TrainableEntry trait dispatch for generic accumulation,
    // though encoded datasets use stored features directly.
    let batch_grad = |batch_indices: &[usize], values: &[f64], k: f64, blend: f64, grads: &mut [f64]| -> f64 {
        let reduce_res = batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0),
                |(mut g, mut loss), chunk| {
                    for &i in chunk {
                        let entry = &train[i];
                        let score = entry.eval_with_state(values, &mut ());
                        let sig = sigmoid(score, k);
                        let target = entry.target(k, blend);

                        // MSE Loss gradient:
                        // J = (S(x) - y)² where S(x) is the sigmoid eval and y is the target.
                        //
                        // Chain rule: dJ/dx = dJ/dS · dS/dx
                        // 1. dJ/dS = 2 · (S(x) - y)
                        // 2. dS/dx = K · S(x) · (1 - S(x))
                        //
                        // dJ/dx = 2 · (S(x) - y) · K · S(x) · (1 - S(x))
                        let err = sig - target;
                        let d = 2.0 * err * sig * (1.0 - sig) * k;

                        entry.accumulate_grad(values, d, &mut g, &());
                        loss = err.mul_add(err, loss);
                    }
                    (g, loss)
                },
            )
            .reduce(|| (vec![0.0; values.len()], 0.0), grad_combine);

        reduce_res.pipe_grads(grads)
    };

    // Validation eval closure
    let val_eval = |values: &[f64], k: f64| -> f64 {
        val.par_iter()
            .map(|e: &loader::SoulEntry| {
                let score = loader::eval_soul(e, values);
                let sig = sigmoid(score, k);
                let target = e.target(k, 0.0); // Hardcoded to 0.0 for validation
                let err = sig - target;
                err * err
            })
            .sum::<f64>()
            / val.len() as f64
    };

    train_loop(train.len(), "Encoded (no attack gen)", config, resume_path, batch_grad, val_eval);
}

/// Training loop for raw EPD datasets.
///
/// Computes full mobility, king safety, and zone defense from scratch for each position.
/// Uses `eval_linear_grad` for direct gradient extraction — exploiting the eval's linearity
/// to compute feature coefficients directly instead of propagating gradient arrays.
fn run_raw(paths: &[String], config: &EvalTuneConfig, resume_path: Option<&str>) {
    let mut entries = Vec::new();
    for path in paths {
        println!("Loading:    \x1b[33m{path}\x1b[0m");
        match loader::load_epd(path) {
            Ok(e) => entries.extend(e),
            Err(e) => {
                eprintln!("\x1b[31mFailed: {e}\x1b[0m");
                return;
            },
        }
    }

    if config.wdl_schedule.is_active() {
        println!("\x1b[93m[!] Warning: WDL blend > 0.0 requested on raw dataset.\x1b[0m");
        println!("\x1b[93m    Raw EPD entries lack search scores; blend will be ignored.\x1b[0m");
    }

    fastrand::shuffle(&mut entries);

    let val_count = entries.len() / 10;
    let train_count = entries.len() - val_count;
    let (train, val) = entries.split_at(train_count);

    print_dataset_stats(train, val, entries.len(), |e| e.result);

    // Batch gradient closure: uses eval_linear_grad (exploits eval linearity)
    let batch_grad = |batch_indices: &[usize], values: &[f64], k: f64, blend: f64, grads: &mut [f64]| -> f64 {
        let reduce_res = batch_indices
            .par_chunks(256)
            .fold(
                || (vec![0.0; values.len()], 0.0),
                |(mut g, mut loss), chunk| {
                    for &i in chunk {
                        let entry = &train[i];
                        let target = entry.target(k, blend);
                        let sq_err = tape::eval_linear_grad(&entry.board, values, target, k, &mut g);
                        loss += sq_err;
                    }
                    (g, loss)
                },
            )
            .reduce(|| (vec![0.0; values.len()], 0.0), grad_combine);

        reduce_res.pipe_grads(grads)
    };

    // Validation eval closure
    let val_eval = |values: &[f64], k: f64| -> f64 {
        val.par_iter()
            .map(|e: &loader::Entry| {
                let score = e.eval(values);
                let sig = sigmoid(score, k);
                let target = e.target(k, 0.0); // Hardcoded to 0.0 for validation
                let err = sig - target;
                err * err
            })
            .sum::<f64>()
            / val.len() as f64
    };

    // ── Sanity check: Split-Brain Gradient Trap ──
    //
    // Verifies that the optimized linear gradient extraction (tape::eval_linear_grad)
    // matches the reference dual-number fused evaluation (tape::eval_dual_fused).
    //
    // If these drift, the eval logic in src/engine/eval.rs and the gradient logic
    // in tuner/src/evaltune/tape.rs have diverged — fixing it then is mandatory.
    let all_params = eval_params::collect_parameters();
    let default_values: Vec<f64> = all_params.iter().map(|p| p.value).collect();
    for entry in train.iter().take(10) {
        let target = entry.target(1.0, 0.0);
        let mut linear_g = vec![0.0; default_values.len()];
        let mut dual_g = vec![0.0; default_values.len()];
        tape::eval_linear_grad(&entry.board, &default_values, target, 1.0, &mut linear_g);
        tape::eval_dual_fused(&entry.board, &default_values, target, 1.0, &mut dual_g);

        for (i, (lin, dual)) in linear_g.iter().zip(dual_g.iter()).enumerate() {
            assert!((lin - dual).abs() < 1e-4, "Gradient drift detected at index {}! linear: {}, dual: {}", i, lin, dual);
        }
    }

    train_loop(train.len(), "Raw (full attack gen)", config, resume_path, batch_grad, val_eval);
}

// ──────── Shared training infrastructure ────────

/// Scatter rayon-reduced gradients into the caller's accumulator.
trait PipeGrads {
    fn pipe_grads(self, out: &mut [f64]) -> f64;
}

impl PipeGrads for (Vec<f64>, f64) {
    fn pipe_grads(self, out: &mut [f64]) -> f64 {
        let (grads, loss) = self;
        for (o, g) in out.iter_mut().zip(grads.iter()) {
            *o += g;
        }
        loss
    }
}

fn grad_combine((mut g1, l1): (Vec<f64>, f64), (g2, l2): (Vec<f64>, f64)) -> (Vec<f64>, f64) {
    for (a, b) in g1.iter_mut().zip(g2) {
        *a += b;
    }
    (g1, l1 + l2)
}

fn print_dataset_stats<T, F: Fn(&T) -> f64>(train: &[T], val: &[T], total: usize, result_fn: F) {
    let train_count = train.len();
    let val_count = val.len();
    println!("Positions:  \x1b[32m{}\x1b[0m ({} train / {} val)", total, train_count, val_count);

    let (ww, bw, dr) = train.iter().fold((0, 0, 0), |(w, b, d), entry| {
        let r = result_fn(entry);
        if (r - 1.0).abs() < 1e-4 {
            (w + 1, b, d)
        } else if r.abs() < 1e-4 {
            (w, b + 1, d)
        } else {
            (w, b, d + 1)
        }
    });
    println!("  White wins: \x1b[32m{ww}\x1b[0m");
    println!("  Black wins: \x1b[32m{bw}\x1b[0m");
    println!("  Draws:      \x1b[32m{dr}\x1b[0m");

    assert!(!train.is_empty(), "Dataset cannot be empty!");
}

/// ── Shared training loop ──
/// Gradient computation and validation eval are injected as closures,
/// everything else (schedulers, optimizer, clipping, logging, checkpointing) is common.
fn train_loop<G, V>(
    train_len: usize,
    mode_label: &str,
    config: &EvalTuneConfig,
    resume_path: Option<&str>,
    batch_grad: G,
    val_eval: V,
) where
    G: Fn(&[usize], &[f64], f64, f64, &mut [f64]) -> f64,
    V: Fn(&[f64], f64) -> f64,
{
    let all_params = eval_params::collect_parameters();
    let default_values: Vec<f64> = all_params.iter().map(|p| p.value).collect();

    let (start_epoch, mut lr_scale, mut values, mut momentum, rng_seed) = resume_path.map_or_else(
        || {
            let momentum = vec![0.0; default_values.len()];
            let seed = config.seed.unwrap_or_else(|| fastrand::u64(..));
            (1, 1.0_f64, default_values.clone(), momentum, seed)
        },
        |path| {
            println!("Resuming from checkpoint: \x1b[33m{path}\x1b[0m");
            let data = load_checkpoint(path, &all_params, &default_values).unwrap_or_else(|e| {
                eprintln!("Failed to load checkpoint: {e}");
                std::process::exit(1);
            });
            (data.epoch, data.lr_scale, data.values, data.momentum, data.rng_seed)
        },
    );

    let is_constant_schedule = matches!(config.lr_schedule, LrScheduleConfig::Constant { .. });
    let lr_scheduler = config.lr_schedule.clone().into_scheduler();
    let wdl_scheduler = config.wdl_schedule.clone().into_scheduler();

    // ── K line search ──
    // the sigmoid scaling constant that best maps
    // raw centipawn scores to win/draw/loss outcomes
    println!("Optimizing K...");
    let mut k = find_optimal_k(&values, config, |v, k| val_eval(v, k));
    let win_rate_100cp = sigmoid(100.0, k);
    println!("K Factor:   \x1b[36m{k:.6}\x1b[0m (100cp -> {:.1}%)   ", win_rate_100cp * 100.0);

    // Frozen reference K - never re-optimized.
    // L_ref uses this K so that loss numbers are comparable across epochs
    // and runs regardless of K-reopt drift.
    let k_ref = k;
    println!("Ref K:      \x1b[36m{k_ref:.6}\x1b[0m");
    let seed_label = if config.seed.is_some() { " (deterministic)" } else { "" };
    println!("Seed:       \x1b[36m{rng_seed}\x1b[0m{seed_label}");

    let initial_values = values.clone();

    // Setup optimizer state and convergence tracking
    let mut fixed_mask: Vec<bool> = all_params.iter().map(|p| p.is_fixed).collect();
    let decay_mask = build_decay_mask(&all_params);
    let beta2_mask = build_beta2_mask(&all_params, config.beta2);

    // Zero init - the EMA decay (0.99 per batch) extinguishes any seed value
    // before epoch 500 when auto-freeze activates, so the auto-freeze sees
    // only real gradient history. A non-zero seed only delays detection of
    // genuinely dead parameters.
    let mut grad_ema_per_param = vec![0.0_f64; values.len()];
    let mut stagnant_epochs = vec![0usize; values.len()];

    let snapshot_limit = (config.epochs / 10).max(1);
    let mut snapshots: Vec<Snapshot> = Vec::with_capacity(snapshot_limit);

    println!("Parameters: {}", all_params.len());
    println!("Mode:       \x1b[36m{mode_label}\x1b[0m");
    println!("LR Sched:   \x1b[36m{}\x1b[0m", lr_scheduler.describe());
    println!("WDL Sched:  \x1b[36m{}\x1b[0m", wdl_scheduler.describe());
    println!("Optimizer:  \x1b[36mLion\x1b[0m (Batch: {}, WD: {})", config.batch_size, config.weight_decay);

    let log_file = File::create("evaltune_log.txt").ok();
    let mut logger = log_file.map(BufWriter::new);
    if let Some(ref mut w) = logger {
        writeln!(w, "# Seed: {rng_seed}").unwrap();
        writeln!(w).unwrap();
        writeln!(w, "epoch   L_train     L_val       L_ref       LR").unwrap();
    }
    let mut json_logger = JsonLogger::new("evaltune.jsonl").ok();

    let mut rng = fastrand::Rng::with_seed(rng_seed);
    let mut optimizer = Lion::new(config.beta1, lr_scheduler.rate(start_epoch, config.epochs), config.weight_decay);

    let mut grad_stats = GradientStats::new(100);
    let mut indices: Vec<usize> = (0..train_len).collect();

    let mut ema_values = values.clone();
    let lr_peak = (1..=config.epochs).fold(0.0f64, |m, e| m.max(lr_scheduler.rate(e, config.epochs)));

    // Tail-only EMA doesn't apply to constant schedules,
    // there is no "tail" phase. Fall back to uniform Polyak averaging.
    let mut ema_active = is_constant_schedule;
    let ema_threshold = if is_constant_schedule { 0.0 } else { 0.3 * lr_peak };
    let mut best_val_loss = f64::MAX;
    let mut plateau_count = 0usize;

    let psqt_end = psqt::LAYOUT.material_offset;
    let base_end = psqt_end + psqt::LAYOUT.material_len;
    let mob_start = psqt::LAYOUT.mobility_open_offset;
    let mob_end = psqt::LAYOUT.weight_offset;

    // ── Progressive unfreeze: material-only warmup ──
    // Freeze all non-psqt/mat parameters for the first unfreeze_epoch epochs,
    // so PSQT + material settle before the refinements join.
    if config.unfreeze_epoch > 0 {
        for i in base_end..fixed_mask.len() {
            fixed_mask[i] = true;
        }
        println!("Progressive unfreeze: params {base_end}+ frozen until epoch {}", config.unfreeze_epoch);
    }

    for epoch in start_epoch..=config.epochs {
        let t0 = Instant::now();

        // ── Periodic K factor re-optimization ──
        // Re-align the sigmoid scaling as parameters drift from their initial state.
        if epoch % 500 == 0 {
            k = find_optimal_k(&ema_values, config, |v, kk| val_eval(v, kk));
            println!("  Reoptimized K: {k:.6}");
            if let Some(ref mut w) = logger {
                writeln!(w, "# K re-opt @ epoch {epoch}: {k:.6}").ok();
            }
        }

        let scheduled_lr = lr_scheduler.rate(epoch, config.epochs) * lr_scale;
        let lr = scheduled_lr.max(0.00001);

        if !ema_active && lr < ema_threshold {
            println!("  EMA activated at epoch {epoch} (lr {lr:.6} < {ema_threshold:.6})");
            ema_active = true;
        }

        let is_restart = epoch > 1 && {
            let prev_scheduled_lr = lr_scheduler.rate(epoch - 1, config.epochs) * lr_scale;
            // A ≥50% jump in LR indicates a scheduler restart (e.g. cosine SGDR cycle boundary).
            // This threshold is intentionally generous: normal LR decay is monotone, so a
            // 50% increase can only mean a deliberate reset point.
            //
            // Correct for cosine-with-restarts, but would false-fire on any scheduler that
            // legitimately increases LR during training (e.g. warmup phases).
            // If a new scheduler with a genuine LR increase is added,
            // gate this on scheduler type or add an LrScheduler::is_restart_boundary method to the trait.
            scheduled_lr > prev_scheduled_lr * 1.5
        };

        let blend = wdl_scheduler.blend(epoch, config.epochs);
        optimizer.set_lr(lr);

        if is_restart {
            plateau_count = 0;
        }

        rng.shuffle(&mut indices);

        let mut train_loss = 0.0;
        let mut total_grads = vec![0.0; values.len()];

        for batch in indices.chunks(config.batch_size) {
            let mut grads = vec![0.0; values.len()];
            let batch_loss = batch_grad(batch, &values, k, blend, &mut grads);

            train_loss += batch_loss;

            let n = batch.len() as f64;
            let norm: f64 = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
            let avg_norm = norm / n;

            // ── Dynamic Gradient Clipping ──
            // Clips outliers based on the distribution of recent batch norms.
            grad_stats.update(avg_norm);
            let clip_thresh = grad_stats.clip_threshold(config.grad_clip);
            let threshold = clip_thresh * n;

            let scale = if norm > threshold { threshold / norm } else { 1.0 };
            for (i, g) in grads.iter_mut().enumerate() {
                *g = *g / n * scale;
                total_grads[i] += *g;
            }

            optimizer.update(&mut values, &mut momentum, &grads, &decay_mask, &fixed_mask, &beta2_mask);

            for value in &mut values[mob_start..mob_end] {
                *value = value.clamp(-MOB_CLAMP, MOB_CLAMP);
            }

            // ── Per-parameter Convergence Tracking ──
            // Freeze parameters that have statistically converged to reduce noise.
            for i in 0..values.len() {
                if !fixed_mask[i] {
                    grad_ema_per_param[i] = 0.99_f64.mul_add(grad_ema_per_param[i], 0.01 * grads[i].abs());
                }
            }

            // ── Tail-only EMA ──
            // Skip the noisy high-LR phase; only average once LR has
            // decayed below 30 % of its peak. Before that, snapshot
            // the live weights directly.
            if ema_active {
                for i in 0..values.len() {
                    ema_values[i] = config.ema_decay.mul_add(ema_values[i], (1.0 - config.ema_decay) * values[i]);
                }
            } else {
                ema_values.copy_from_slice(&values);
            }
        }

        // Progressive unfreeze; lift the material-only gate.
        if config.unfreeze_epoch > 0 && epoch == config.unfreeze_epoch {
            for (i, p) in all_params.iter().enumerate() {
                fixed_mask[i] = p.is_fixed;
            }
            println!("  Unfrozen all remaining parameters at epoch {epoch}");
        }

        // ── Auto-freeze stagnant parameters ──
        if epoch > 500 && epoch % 100 == 0 {
            let mut frozen = 0;
            for i in 0..values.len() {
                if !fixed_mask[i] && grad_ema_per_param[i] < 1e-7 && !all_params[i].freeze_resistant {
                    stagnant_epochs[i] += 1;
                    if stagnant_epochs[i] >= 2 {
                        fixed_mask[i] = true;
                        frozen += 1;
                    }
                } else {
                    stagnant_epochs[i] = 0;
                }
            }
            if frozen > 0 {
                println!("  Auto-frozen {frozen} stagnant parameters.");
            }
        }

        let val_loss = val_eval(&ema_values, k);
        let ref_loss = val_eval(&ema_values, k_ref);
        let train_loss = train_loss / train_len as f64;

        // ── Validation Plateau Detection ──
        // Reduce LR if validation loss stalls for Constant schedule.
        if val_loss < best_val_loss - 1e-6 {
            best_val_loss = val_loss;
            plateau_count = 0;
        } else {
            plateau_count += 1;
            // Plateau LR halving is gated to constant schedules only.
            // Cosine/WSD/etc don't need a separate stall-response mechanism.
            // Reducing LR when the scheduler is already responsible for decay would overcorrect.
            if is_constant_schedule {
                if plateau_count >= config.patience {
                    lr_scale *= 0.5;
                    plateau_count = 0;
                    println!("  Plateau detected — LR scale → {lr_scale:.3}");
                }
            }
        }

        let overfit_warn = if val_loss > best_val_loss * 1.02 { " \x1b[31;1m⚠ OVERFIT\x1b[0m" } else { "" };

        let is_best = update_snapshots(&mut snapshots, epoch, &ema_values, &all_params, val_loss, snapshot_limit);

        // Group-wise gradient norms for diagnostics
        let psqt_norm = total_grads[..psqt_end].iter().map(|g| g * g).sum::<f64>().sqrt();
        let mob_norm = total_grads[mob_start..mob_end].iter().map(|g| g * g).sum::<f64>().sqrt();

        if let Some(ref mut w) = logger {
            writeln!(
                w,
                "{:>3}     {:.6}    {:.6}    {:.6}    {:.4}{}",
                epoch,
                train_loss,
                val_loss,
                ref_loss,
                lr,
                if is_best { " *" } else { "" }
            )
            .ok();
        }
        if let Some(ref mut l) = json_logger {
            if is_restart {
                l.log("restart", &serde_json::json!({ "epoch": epoch }));
            }
            l.log(
                "epoch",
                &serde_json::json!({
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "ref_loss": ref_loss,
                    "lr": lr,
                    "is_best": is_best,
                    "psqt_norm": psqt_norm,
                    "mob_norm": mob_norm,
                    "overfit": overfit_warn.contains("OVERFIT")
                }),
            );
        }

        let elapsed = t0.elapsed().as_secs_f32();
        let color = if is_best { "\x1b[32m" } else { "\x1b[0m" };
        println!(
            "\r{}Epoch {:>3}/{} | L_train: {:.6} | L_val: {:.6} | L_ref: {:.6} | LR: {:.4} | {:.2}s{}\x1b[0m\x1b[K",
            color, epoch, config.epochs, train_loss, val_loss, ref_loss, lr, elapsed, overfit_warn
        );

        if epoch % 20 == 0 || epoch == config.epochs {
            print_params(&all_params, &initial_values, &ema_values);

            if let Err(e) = save_checkpoint(
                "evaltune_checkpoint.json",
                epoch + 1, // resume starts here; the current epoch is already done
                lr_scale,
                &values,
                &momentum,
                &all_params,
                rng_seed,
            ) {
                eprintln!("Failed to save checkpoint: {e}");
            }
        }
    }

    // Flush the log before writing final reports,
    // since print_results opens its own handle to the same file.
    drop(logger);

    sensitivity_report(&all_params, &grad_ema_per_param, &fixed_mask);
    print_results(&snapshots, &all_params, &initial_values, &ema_values, config.epochs);
}

/// Sensitivity Analysis — writes `sensitivity-report.txt`.
fn sensitivity_report(params: &[Tunable], grad_ema: &[f64], fixed_mask: &[bool]) {
    let Ok(mut f) = std::fs::File::create("sensitivity-report.txt") else { return };
    let mut w = std::io::BufWriter::new(&mut f);
    writeln!(w, "Sensitivity Analysis").ok();
    writeln!(w).ok();
    let mut sensitivities = Vec::new();
    let mut frozen = Vec::new();

    for p in params {
        let delta = grad_ema[p.idx];
        if p.is_fixed || fixed_mask[p.idx] {
            frozen.push((delta, p.idx, p.name.as_str()));
        } else {
            sensitivities.push((delta, p.idx, p.name.as_str()));
        }
    }

    sensitivities.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
    frozen.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));

    let max_width = |list: &[(f64, usize, &str)]| list.iter().take(10).map(|r| r.2.len()).max().unwrap_or(20);
    let active_width = max_width(&sensitivities) + 1;
    let frozen_width = frozen.iter().take(10).map(|r| r.2.len()).max().unwrap_or(20) + 1;

    writeln!(w, "  Top Load-Bearing Parameters:").ok();
    for (i, (delta, _, name)) in sensitivities.iter().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    writeln!(w).ok();
    writeln!(w, "  Lowest-Impact Parameters:").ok();
    for (i, (delta, _, name)) in sensitivities.iter().rev().take(10).enumerate() {
        writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = active_width).ok();
    }

    if !frozen.is_empty() {
        writeln!(w).ok();
        writeln!(w, "  Highest Sensitivity Auto-Frozen/Fixed Parameters:").ok();
        for (i, (delta, _, name)) in frozen.iter().take(10).enumerate() {
            writeln!(w, "    {:>3}. {:<name_width$} ΔL: {:.8}", i + 1, name, delta, name_width = frozen_width).ok();
        }
    }
}

/// K line search via golden-section search.
fn find_optimal_k<F: Fn(&[f64], f64) -> f64>(values: &[f64], config: &EvalTuneConfig, eval_fn: F) -> f64 {
    let phi = (5.0_f64.sqrt() - 1.0) / 2.0;
    let mut lo = config.k_min;
    let mut hi = config.k_max;
    let tol = 1e-6 * (hi - lo).max(1e-9);

    let mut a = hi - phi * (hi - lo);
    let mut b = lo + phi * (hi - lo);
    let mut fa = eval_fn(values, a);
    let mut fb = eval_fn(values, b);

    while (hi - lo).abs() > tol {
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - phi * (hi - lo);
            fa = eval_fn(values, a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + phi * (hi - lo);
            fb = eval_fn(values, b);
        }
    }
    (lo + hi) / 2.0
}

fn resolve_dataset_paths(input: &str) -> Option<Vec<String>> {
    if input == "default" {
        let mut paths = Vec::new();
        if let Ok(entries) = std::fs::read_dir("data") {
            for entry in entries.flatten() {
                let path = entry.path();
                let name = path.to_string_lossy();
                if name.ends_with(".soul.zst") || name.ends_with(".soul") {
                    paths.push(name.to_string());
                }
            }
        }

        if paths.is_empty() {
            eprintln!("\x1b[31mError: No default dataset found in data/ directory.\x1b[0m");
            eprintln!("Please provide a dataset path using --dataset <path>");
            None
        } else {
            println!("Auto-discovered datasets: {}", paths.join(", "));
            Some(paths)
        }
    } else {
        let paths: Vec<String> = input
            .split(',')
            .map(str::trim)
            .map(|s| {
                if std::path::Path::new(s).exists() {
                    s.to_string()
                } else {
                    let data_prefixed = format!("data/{s}");
                    if std::path::Path::new(&data_prefixed).exists() { data_prefixed } else { s.to_string() }
                }
            })
            .collect();
        Some(paths)
    }
}

/// Weight decay mask: not all parameters deserve equal punishment.
///
/// - PSQT center squares decay at 0.5× (central values are more structurally
///   significant — aggressive decay risks flattening critical gradients).
/// - Mobility weights decay at 1.5× (these can drift without bound since their
///   features are unbounded integer counts).
/// - Everything else decays at 1.0×.
fn build_decay_mask(params: &[Tunable]) -> Vec<f64> {
    let psqt_end = psqt::LAYOUT.material_offset;
    let mat_end = psqt::LAYOUT.mobility_open_offset;
    let mob_end = psqt::LAYOUT.weight_offset;

    (0..params.len())
        .map(|i| {
            if i < psqt_end {
                let sq = i % 32;
                let row = sq / 4;
                let col = sq % 4;
                let is_center = (2..=5).contains(&row) && (2..=3).contains(&col);
                if is_center { 0.5 } else { 1.0 }
            } else if i < mat_end {
                1.0
            } else if i < mob_end {
                1.5
            } else {
                1.0
            }
        })
        .collect()
}

/// Per-group momentum decay mask.
///
/// Different parameter groups have different natural gradient timescales.
/// - PSQT (0.995): squares only see updates when a piece of that type lands
///   there — longer momentum smooths sparse signal across positions.
/// - Mobility (0.95): features are computed every position; shorter momentum
///   lets weights track the faster dynamics without lag.
/// - Everything else (0.99): the existing default from the config.
fn build_beta2_mask(params: &[Tunable], default_beta2: f64) -> Vec<f64> {
    let psqt_end = psqt::LAYOUT.material_offset;
    let mat_end = psqt::LAYOUT.mobility_open_offset;
    let mob_end = psqt::LAYOUT.weight_offset;

    (0..params.len())
        .map(|i| {
            if i < psqt_end {
                0.995
            } else if i < mat_end {
                default_beta2
            } else if i < mob_end {
                0.95
            } else {
                default_beta2
            }
        })
        .collect()
}
