//! Diagnostic benchmarks and analytical passes the tuner runs in place of training.
//!
//! Each is a task of `run` itself, so a probe reads the dataset through the same load, shuffle
//! and split a training run would, and reports instead of stepping.

use std::{hint::black_box, time::Instant};

use rayon::prelude::*;

use crate::{
    config::EvalTuneConfig,
    curvature::Curvature,
    engine::{Tunable, accumulate_record_grad, eval_params, eval_record_full},
    groups::build_lr_mask,
    palette::{DIM, LAB, RESET, VAL},
    run::TrainerContext,
    scale::KController,
    shuffle::Shuffler,
    training::{sigmoid, wdl_target},
};

/// Computes and reports the empirical loss Hessian at baseline parameters over the training set.
///
/// Passes a unit scalar multiplier (`1.0`) to `accumulate_record_grad` to extract raw linear
/// feature coefficients directly into the gradient scratch buffer without altering the eval path.
pub fn curvature_report(ctx: &TrainerContext, config: &EvalTuneConfig) {
    let (params, values, k, blend) = shipped(ctx, config);
    let free_indices: Vec<usize> = params.iter().filter(|p| !p.is_fixed).map(|p| p.idx).collect();
    let is_trainable: Vec<bool> = params.iter().map(|p| !p.is_fixed).collect();
    let feature_dim = params.len();

    // Filtered inline so sample counts strictly reflect positions contributing to the Hessian.
    let (curvature, contributing_count) = (0..ctx.train_count)
        .into_par_iter()
        .fold(
            || (Curvature::zeros(feature_dim), 0usize, vec![0.0; feature_dim], Vec::with_capacity(64)),
            |(mut hessian, mut count, mut grad_buf, mut sparse_features), index| {
                let record = &ctx.train[index];
                if ctx.passes_vol_filter(record) {
                    let eval = eval_record_full(record, &values);
                    let pred = sigmoid(eval.score, k);
                    let (target, _) = wdl_target(record, k, blend);
                    let sample_weight = if ctx.phase_weights.is_empty() { 1.0 } else { ctx.phase_weights[index] };

                    accumulate_record_grad(record, &eval, 1.0, &mut grad_buf);

                    // Drain non-zero entries and reset scratch slots for the next iteration in a single pass.
                    sparse_features.clear();
                    for (param_idx, coefficient) in grad_buf.iter_mut().enumerate() {
                        if *coefficient != 0.0 {
                            if is_trainable[param_idx] {
                                sparse_features.push((param_idx, *coefficient));
                            }
                            *coefficient = 0.0;
                        }
                    }

                    let hessian_scale = ctx.loss_fn().hessian_scale(pred, target, k);
                    hessian.add_outer(sample_weight * hessian_scale, &sparse_features);
                    count += 1;
                }
                (hessian, count, grad_buf, sparse_features)
            },
        )
        .map(|(hessian, count, ..)| (hessian, count))
        .reduce(
            || (Curvature::zeros(feature_dim), 0),
            |(mut acc_hessian, acc_count), (thread_hessian, thread_count)| {
                acc_hessian.merge(&thread_hessian);
                (acc_hessian, acc_count + thread_count)
            },
        );

    curvature.symmetrized().spectrum(&free_indices).report(&params, contributing_count, k);
}

/// Draws per large-batch size, and the small batch its chunks approximate.
const TRIALS: usize = 64;

/// Batch sizes on the sign-error ladder, each double the last.
const RUNGS: usize = 6;
const SMALL_CHUNK: usize = 2048;

/// Doublings of the large batch allowed while it sits under the scale being estimated.
const CLIMBS: usize = 4;

/// The shipped parameters, and the K and blend a cold run would open with.
///
/// Every probe measures at the same point, or their numbers cannot be read against each other.
fn shipped(ctx: &TrainerContext, config: &EvalTuneConfig) -> (Vec<Tunable>, Vec<f64>, f64, f64) {
    let params = eval_params::collect_parameters();
    let values = eval_params::default_values(&params);
    let blend = config.wdl_schedule.clone().into_scheduler().blend(1, config.epochs);
    let k = KController::bootstrap(config, ctx, &values, &values, blend, None).k();
    (params, values, k, blend)
}

/// Gradient noise scale: the batch size past which averaging stops buying signal.
///
/// `E|G_B|² = |g|² + tr(Σ)/B`, so one large batch and its chunks give both terms. Numerator and
/// denominator pool across trials before dividing, a ratio of noisy estimates being biased.
///
/// McCandlish et al., An Empirical Model of Large-Batch Training, 2018. <https://arxiv.org/pdf/1812.06162>
/// derived for a step proportional to the gradient. Lion's is `±lr` by sign, so this bounds where a batch
/// stops improving the gradient, not the step. Grows as `|g|` shrinks; measured at the shipped values.
pub fn batch_size_report(ctx: &TrainerContext, config: &EvalTuneConfig) {
    let (_, values, k, blend) = shipped(ctx, config);

    let mut indices: Vec<u32> = (0..ctx.train_count as u32).collect();
    let mut shuffler = Shuffler::new(ctx.train_count);
    let mut big = config.batch_size.min(ctx.train_count);
    let mut ladder = [Rung::default(); RUNGS];

    // The subtraction cancels to nothing while `b_big · |g|²` sits under `tr(Σ)`, so the large
    // batch climbs until it reaches the scale it is measuring or runs out of positions.
    for climb in 0..=CLIMBS {
        ladder.iter_mut().for_each(|rung| *rung = Rung::default());
        let (signal, noise, degenerate) = sample(ctx, &values, k, blend, &mut indices, &mut shuffler, big, TRIALS, &mut ladder);
        let resolved = signal > 0.0 && noise > 0.0 && noise / signal <= big as f64;
        if resolved || climb == CLIMBS || big == ctx.train_count {
            report(config, k, big, signal, noise, degenerate, TRIALS, &ladder);
            return;
        }
        big = (big * 2).min(ctx.train_count);
    }
}

/// Pools the signal and noise terms over `trials` draws at one large-batch size.
fn sample(
    ctx: &TrainerContext,
    values: &[f64],
    k: f64,
    blend: f64,
    indices: &mut [u32],
    shuffler: &mut Shuffler,
    big: usize,
    trials: usize,
    ladder: &mut [Rung],
) -> (f64, f64, usize) {
    let chunks = (big / SMALL_CHUNK).max(2);
    let chunk_len = big / chunks;
    let (mut pooled_signal, mut pooled_noise, mut degenerate) = (0.0, 0.0, 0usize);
    let (mut chunk_sums, mut chunk_counts): (Vec<Vec<f64>>, Vec<usize>) = (Vec::new(), Vec::new());

    for trial in 0..trials {
        shuffler.fill(indices, 0x0BAD_5EED ^ trial as u64);

        // Σ over chunks of |mean gradient|², against the same quantity for their union.
        let mut union_sum = vec![0.0; values.len()];
        let (mut union_count, mut small_sq, mut small_count) = (0usize, 0.0, 0usize);
        chunk_sums.clear();
        chunk_counts.clear();

        for chunk in indices[..big].chunks(chunk_len) {
            let (grad, .., count) = ctx.batch_grad(chunk, values, k, blend);
            if count == 0 {
                continue;
            }
            small_sq += grad.iter().map(|g| (g / count as f64).powi(2)).sum::<f64>();
            small_count += count;
            for (acc, g) in union_sum.iter_mut().zip(&grad) {
                *acc += g;
            }
            union_count += count;
            chunk_sums.push(grad);
            chunk_counts.push(count);
        }

        if union_count == 0 || small_count == 0 {
            continue;
        }

        accumulate_sign_error(&chunk_sums, &chunk_counts, &union_sum, union_count, ladder);

        // The vol filter drops positions, so the sizes are counted rather than assumed.
        let b_small = small_count as f64 / chunks as f64;
        let b_big = union_count as f64;
        let small_sq = small_sq / chunks as f64;
        let big_sq: f64 = union_sum.iter().map(|g| (g / b_big).powi(2)).sum();

        let signal = b_big.mul_add(big_sq, -(b_small * small_sq)) / (b_big - b_small);
        let noise = (small_sq - big_sq) / (b_small.recip() - b_big.recip());
        degenerate += usize::from(signal <= 0.0 || noise <= 0.0);
        pooled_signal += signal;
        pooled_noise += noise;
    }
    (pooled_signal, pooled_noise, degenerate)
}

/// One batch size on the sign-error ladder, pooled over every pair and trial that fed it.
#[derive(Clone, Copy, Default)]
struct Rung {
    positions: u64,
    groups: u64,
    flipped: u64,
    compared: u64,
    /// The same by magnitude, a share of the gradient rather than of the parameter count.
    flipped_mass: f64,
    mass: f64,
}

impl Rung {
    /// One sample's error rate, from the rate at which two of them disagree. `2p(1 − p) = m` has
    /// no real root past `m = 0.5`, where the two are telling each other nothing.
    fn error(disagreement: f64) -> f64 { 0.5 * (1.0 - (1.0 - 2.0 * disagreement).max(0.0).sqrt()) }
}

/// How often a batch of each size gets a coordinate's sign wrong.
///
/// Between disjoint pairs of equal size: two independent samples each wrong with probability `p`
/// disagree at `2p(1 − p)`, which [`Rung::error`] inverts. Against a superset, the batch's error,
/// the reference's and their correlation all arrive as one number.
///
/// Weighted by the whole draw, since `|a + b|` shrinks exactly when the pair disagrees and would
/// score every flip as though it barely mattered.
fn accumulate_sign_error(sums: &[Vec<f64>], counts: &[usize], union_sum: &[f64], union_count: usize, ladder: &mut [Rung]) {
    let weights: Vec<f64> = union_sum.iter().map(|g| (g / union_count as f64).abs()).collect();

    for (rung, group) in ladder.iter_mut().zip(std::iter::successors(Some(1usize), |g| Some(g * 2))) {
        if group * 2 > sums.len() {
            break;
        }

        for (pair, pair_counts) in sums.chunks(group * 2).zip(counts.chunks(group * 2)) {
            if pair.len() < group * 2 {
                continue;
            }

            let (left, right) = pair.split_at(group);
            let (left_n, right_n) = pair_counts.split_at(group);
            let (left_n, right_n) = (left_n.iter().sum::<usize>() as f64, right_n.iter().sum::<usize>() as f64);
            if left_n == 0.0 || right_n == 0.0 {
                continue;
            }

            rung.groups += 2;
            rung.positions += (left_n + right_n) as u64;
            for slot in 0..sums[0].len() {
                let a: f64 = left.iter().map(|sum| sum[slot]).sum::<f64>() / left_n;
                let b: f64 = right.iter().map(|sum| sum[slot]).sum::<f64>() / right_n;
                if a == 0.0 && b == 0.0 {
                    continue;
                }

                let weight = weights[slot];
                let flipped = a * b < 0.0;
                rung.compared += 1;
                rung.flipped += u64::from(flipped);
                rung.mass += weight;
                rung.flipped_mass += if flipped { weight } else { 0.0 };
            }
        }
    }
}

fn report(
    config: &EvalTuneConfig,
    k: f64,
    big: usize,
    pooled_signal: f64,
    pooled_noise: f64,
    degenerate: usize,
    trials: usize,
    ladder: &[Rung],
) {
    println!("\n{LAB}Noise scale{RESET} {DIM}({trials} trials against {big} positions at K = {k:.6}){RESET}");
    if pooled_noise <= 0.0 {
        println!("  {LAB}unresolved{RESET}    the larger batch measured no less noise, so raise the trial count");
        return;
    }
    if pooled_signal <= 0.0 {
        println!("  {LAB}unresolved{RESET}    both batches are pure noise here, putting the scale above {big}");
        return;
    }

    let b_simple = pooled_noise / pooled_signal;
    println!("  {LAB}|g|²{RESET}          {VAL}{:.4e}{RESET}", pooled_signal / trials as f64);
    println!("  {LAB}tr(Σ){RESET}         {VAL}{:.4e}{RESET}", pooled_noise / trials as f64);
    println!("  {LAB}B_simple{RESET}      {VAL}{b_simple:.0}{RESET}  {DIM}positions past which averaging stops paying{RESET}");
    println!(
        "  {LAB}configured{RESET}    {VAL}{}{RESET}  {DIM}{:.2}× B_simple{RESET}",
        config.batch_size,
        config.batch_size as f64 / b_simple
    );
    if degenerate > 0 {
        println!("  {LAB}unstable{RESET}      {VAL}{degenerate}{RESET} of {trials} {DIM}trials estimated a negative term{RESET}");
    }

    println!("\n{LAB}Sign error{RESET} {DIM}(what a batch that size gets wrong, which is what Lion pays for){RESET}");
    println!("  {LAB}batch      coordinates    gradient mass{RESET}");
    for rung in ladder.iter().filter(|r| r.compared > 0) {
        println!(
            "  {VAL}{:<9}{RESET}    {VAL}{:5.1}%{RESET}          {VAL}{:5.1}%{RESET}",
            rung.positions / rung.groups,
            100.0 * Rung::error(rung.flipped as f64 / rung.compared as f64),
            100.0 * Rung::error(rung.flipped_mass / rung.mass)
        );
    }
}

/// How much history the step direction should hold.
///
/// `m` is an EMA with weights `(1 − β₂)β₂ʲ`, so it averages `B·(1 + β₂)/(1 − β₂)` positions and
/// lags the current gradient by `β₂/(1 − β₂)` steps of travel. Writing `a = 1 − β₂`, its squared
/// error against the true gradient is `(a/2)·tr(Σ)/B + (d/a)²`, minimized at `a = (4·d²·B / tr(Σ))^(1/3)`.
///
/// `d` is measured rather than derived from the Hessian: one Lion-shaped step, then the same batch
/// again, so the sampling noise is identical on both sides and the difference is curvature alone.
/// It assumes a coordinate keeps its sign across the window, the most drift available and so the
/// shortest window; the true optimum is at or above what this prints.
pub fn momentum_report(ctx: &TrainerContext, config: &EvalTuneConfig) {
    const TRIALS: usize = 32;

    let (params, values, k, blend) = shipped(ctx, config);
    let lr_mask = build_lr_mask(values.len(), config);

    let batch = config.batch_size.min(ctx.train_count / 2).max(1);
    let mut indices: Vec<u32> = (0..ctx.train_count as u32).collect();
    let mut shuffler = Shuffler::new(ctx.train_count);
    let (mut variance, mut drift_sq, mut samples) = (0.0, 0.0, 0usize);

    for trial in 0..TRIALS {
        shuffler.fill(&mut indices, 0x_D1FF_5EED ^ trial as u64);
        let (first, second) = (&indices[..batch], &indices[batch..batch * 2]);

        let (a_sum, .., a_count) = ctx.batch_grad(first, &values, k, blend);
        let (b_sum, .., b_count) = ctx.batch_grad(second, &values, k, blend);
        if a_count == 0 || b_count == 0 {
            continue;
        }

        let gradient: Vec<f64> = a_sum.iter().map(|g| g / a_count as f64).collect();
        let other: Vec<f64> = b_sum.iter().map(|g| g / b_count as f64).collect();
        variance += 0.5 * gradient.iter().zip(&other).map(|(x, y)| (x - y).powi(2)).sum::<f64>();

        let stepped: Vec<f64> = (0..values.len())
            .map(|i| if params[i].is_fixed { values[i] } else { gradient[i].signum().mul_add(-lr_mask[i], values[i]) })
            .collect();
        let (moved_sum, .., moved_count) = ctx.batch_grad(first, &stepped, k, blend);
        if moved_count == 0 {
            continue;
        }

        drift_sq += gradient
            .iter()
            .zip(&moved_sum)
            .map(|(g, moved)| (moved / moved_count as f64 - g).powi(2))
            .sum::<f64>();
        samples += 1;
    }

    println!("\n{LAB}Momentum{RESET} {DIM}({samples} trials of {batch} positions at K = {k:.6}){RESET}");
    if samples == 0 || variance <= 0.0 || drift_sq <= 0.0 {
        println!("  {LAB}unresolved{RESET}    no batch pair produced both a variance and a drift");
        return;
    }

    let variance = variance / samples as f64;
    let drift = (drift_sq / samples as f64).sqrt();
    let row = |label: &str, value: &str, note: &str| println!("  {LAB}{label:<12}{RESET}{VAL}{value:<11}{RESET}{DIM}{note}{RESET}");

    let coefficient = (4.0 * drift.powi(2) / variance).cbrt();
    row("variance", &format!("{variance:.4e}"), "tr(Σ)/B, one batch against another");
    row("drift", &format!("{drift:.4e}"), "gradient moved per unit of learning rate");
    row("coefficient", &format!("{coefficient:.4}"), "beta2_lr_coefficient, 1 - β₂ = c·lr^(2/3)");

    let scheduler = config.lr_schedule.clone().into_scheduler();
    let (first, last) = (scheduler.rate(1, config.epochs), scheduler.rate(config.epochs, config.epochs));
    for (label, lr) in [("β₂ first", first), ("β₂ last", last)] {
        let a = (4.0 * (drift * lr).powi(2) / variance).cbrt().clamp(1e-6, 1.0);
        row(label, &format!("{:.4}", 1.0 - a), &format!("lr {lr:.5}"));
    }
    row("configured", &format!("{:.4}", config.beta2), "");
}

/// Benchmarks gradient throughput across sequential, cache-blocked, and fully randomized access orders.
///
/// Isolates random-gather overhead by keeping data and arithmetic constant while varying
/// index traversal. A wide gap indicates memory latency / TLB bottlenecks (pointing to record
/// packing or hugepages as optimizations); a narrow gap indicates arithmetic saturation.
/// The blocked arm uses the configured `shuffle_block`, so it shows how much of that gap a
/// run already avoids.
pub fn gather_cost(ctx: &TrainerContext, config: &EvalTuneConfig) {
    const MAX_BATCHES: usize = 200;

    let values = eval_params::default_values(&eval_params::collect_parameters());
    let k = 0.5 * (config.k_min + config.k_max);
    let batches = (ctx.train_count / config.batch_size).clamp(1, MAX_BATCHES);
    let total_positions = (batches * config.batch_size).min(ctx.train_count) as f64;
    let mut indices: Vec<u32> = (0..ctx.train_count as u32).collect();

    let bench_pass = |order: &[u32]| {
        let start = Instant::now();
        for batch in order.chunks(config.batch_size).take(batches) {
            black_box(ctx.batch_grad(batch, &values, k, 0.0));
        }
        start.elapsed().as_secs_f64()
    };

    // Warm up page caches sequentially before measuring shuffled access.
    let mut shuffler = Shuffler::new(ctx.train_count);
    let sequential_sec = bench_pass(&indices);
    let blocked_sec = (config.shuffle_block > 0).then(|| {
        shuffler.fill_blocked(&mut indices, 0xC0FFEE, config.shuffle_block, ctx.train_count);
        bench_pass(&indices)
    });
    shuffler.fill(&mut indices, 0xC0FFEE);
    let shuffled_sec = bench_pass(&indices);

    println!("\n{LAB}Gather cost{RESET} {DIM}({batches} batches of {}){RESET}", config.batch_size);
    let print_row = |label: &str, elapsed_sec: f64| {
        println!(
            "  {LAB}{label:<10}{RESET}  {VAL}{elapsed_sec:6.2}s{RESET}  {VAL}{:5.1}M pos/s{RESET}  {VAL}{:5.2}×{RESET}",
            total_positions / elapsed_sec / 1e6,
            elapsed_sec / sequential_sec
        );
    };

    print_row("sequential", sequential_sec);
    if let Some(elapsed_sec) = blocked_sec {
        print_row(&format!("block {}", config.shuffle_block), elapsed_sec);
    }
    print_row("shuffled", shuffled_sec);
}

/// Measures the throughput advantage of evaluating dual-target validation probes in a single fused pass.
pub fn val_cost(ctx: &TrainerContext, config: &EvalTuneConfig) {
    const TRIALS: usize = 7;

    let values = eval_params::default_values(&eval_params::collect_parameters());
    let k = 0.5 * (config.k_min + config.k_max);

    let eval_fused = || {
        let start = Instant::now();
        black_box(ctx.val_eval(&values, [(k, 1.0), (k, 0.0)]));
        start.elapsed().as_secs_f64()
    };

    let eval_split = || {
        let start = Instant::now();
        black_box(ctx.val_eval(&values, [(k, 1.0)]));
        black_box(ctx.val_eval(&values, [(k, 0.0)]));
        start.elapsed().as_secs_f64()
    };

    // Warm up to fault in validation memory slices.
    let _ = eval_fused();
    let _ = eval_split();

    // Interleave trials and record the minimum time to reject OS scheduling jitter.
    let (mut min_fused_sec, mut min_split_sec) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..TRIALS {
        min_fused_sec = min_fused_sec.min(eval_fused());
        min_split_sec = min_split_sec.min(eval_split());
    }

    println!("\n{LAB}Val cost{RESET} {DIM}({} positions, best of {TRIALS}){RESET}", ctx.val.len());
    println!("  {LAB}fused{RESET}    {VAL}{:7.2} ms{RESET}  {DIM}one traversal, two probes{RESET}", min_fused_sec * 1e3);
    println!(
        "  {LAB}split{RESET}    {VAL}{:7.2} ms{RESET}  {DIM}two traversals, one probe each{RESET}",
        min_split_sec * 1e3
    );
    println!(
        "  {LAB}saved{RESET}    {VAL}{:7.2} ms{RESET}  {VAL}{:.2}×{RESET} {DIM}per epoch{RESET}",
        (min_split_sec - min_fused_sec) * 1e3,
        min_split_sec / min_fused_sec
    );
}
