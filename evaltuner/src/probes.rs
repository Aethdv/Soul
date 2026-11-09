//! Diagnostic benchmarks and analytical passes the tuner runs in place of training.
//!
//! Each runs through the same load, shuffle, and split a training run would, and reports
//! instead of stepping.

use std::{hint::black_box, iter::successors, time::Instant};

use rayon::prelude::*;

use crate::{
    config::EvalTuneConfig,
    curvature::Curvature,
    engine::{Tunable, accumulate_record_grad, eval_params, eval_record_full},
    report::{DIM, LAB, RESET, VAL},
    run::Objective,
    scale::KController,
    shuffle::Shuffler,
    training::{add_slice, lr_mask, sigmoid, wdl_target},
};

const TRIALS: usize = 64;
const RUNGS: usize = 6;
const SMALL_CHUNK: usize = 2048;
const CLIMBS: usize = 4;

/// Passes `d_score = 1.0` to `accumulate_record_grad` so raw linear feature coefficients land
/// directly in the scratch buffer without a separate extraction path.
pub(crate) fn curvature_report(obj: &Objective, config: &EvalTuneConfig) {
    let ProbePoint { params, values, k, blend } = baseline_point(obj, config);
    let free: Vec<usize> = params.iter().filter(|p| !p.is_fixed).map(|p| p.idx).collect();
    let trainable: Vec<bool> = params.iter().map(|p| !p.is_fixed).collect();
    let dim = params.len();

    let (curvature, count) = (0..obj.train.len())
        .into_par_iter()
        .fold(
            || (Curvature::zeros(dim), 0usize, vec![0.0; dim], Vec::with_capacity(64)),
            |(mut hessian, mut count, mut grad_buf, mut nonzeros), idx| {
                let record = &obj.train[idx];
                if obj.passes_volatility_filter(record) {
                    let eval = eval_record_full(record, &values);
                    let pred = sigmoid(eval.score, k);
                    let (target, _) = wdl_target(record, k, blend);
                    let w = if obj.sample_weights.is_empty() { 1.0 } else { obj.sample_weights[idx] };

                    accumulate_record_grad(record, &eval, 1.0, &mut grad_buf);

                    nonzeros.clear();
                    for (i, coeff) in grad_buf.iter_mut().enumerate() {
                        if *coeff != 0.0 {
                            if trainable[i] {
                                nonzeros.push((i, *coeff));
                            }
                            *coeff = 0.0;
                        }
                    }

                    let h_scale = obj.loss_fn().hessian_scale(pred, target, k);
                    hessian.add_outer(w * h_scale, &nonzeros);
                    count += 1;
                }

                (hessian, count, grad_buf, nonzeros)
            },
        )
        .map(|(hessian, count, ..)| (hessian, count))
        .reduce(
            || (Curvature::zeros(dim), 0),
            |(mut acc, n1), (other, n2)| {
                acc.merge(&other);
                (acc, n1 + n2)
            },
        );

    curvature.symmetrized().spectrum(&free).report(&params, count, k);
}

/// Gradient noise scale: the batch size past which averaging stops buying signal.
///
/// `E|G_B|² = |g|² + tr(Σ)/B`, so one large batch and its chunks give both terms. Numerator and
/// denominator pool across trials before dividing, a ratio of noisy estimates being biased.
///
/// McCandlish et al. An Empirical Model of Large-Batch Training, 2018. <https://arxiv.org/pdf/1812.06162>
/// derived for a step proportional to the gradient. Lion's is `±lr` by sign, so this bounds where a batch
/// stops improving the gradient, not the step. Grows as `|g|` shrinks; measured at the shipped values.
pub(crate) fn batch_size_report(obj: &Objective, config: &EvalTuneConfig) {
    let point = baseline_point(obj, config);
    let train_len = obj.train.len();

    let mut indices: Vec<u32> = (0..train_len as u32).collect();
    let mut shuffler = Shuffler::new(train_len);
    let mut big = config.batch_size.min(train_len);
    let mut ladder = [Rung::default(); RUNGS];

    // The subtraction cancels to nothing while b_big · |g|² sits under tr(Σ), so the large
    // batch climbs until it reaches the scale it is measuring or runs out of positions.
    for climb in 0..=CLIMBS {
        ladder.fill(Rung::default());
        let est = pool_noise_trials(obj, &point, big, &mut indices, &mut shuffler, &mut ladder);
        let resolved = est.signal > 0.0 && est.noise > 0.0 && est.noise / est.signal <= big as f64;
        if resolved || climb == CLIMBS || big == train_len {
            print_noise_report(config, point.k, big, est, &ladder);
            return;
        }
        big = (big * 2).min(train_len);
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
pub(crate) fn momentum_report(obj: &Objective, config: &EvalTuneConfig) {
    const MOMENTUM_TRIALS: usize = 32;

    let ProbePoint { params, values, k, blend } = baseline_point(obj, config);
    let lr_mults = lr_mask(values.len(), config);
    let train_len = obj.train.len();
    let batch = config.batch_size.min(train_len / 2).max(1);

    let mut indices: Vec<u32> = (0..train_len as u32).collect();
    let mut shuffler = Shuffler::new(train_len);
    let (mut variance, mut drift_sq, mut samples) = (0.0, 0.0, 0usize);

    for trial in 0..MOMENTUM_TRIALS {
        shuffler.fill(&mut indices, 0xD1FF_5EED ^ trial as u64);
        let (first, second) = (&indices[..batch], &indices[batch..batch * 2]);
        let (a_sum, .., a_count) = obj.batch_grad(first, &values, k, blend);
        let (b_sum, .., b_count) = obj.batch_grad(second, &values, k, blend);

        if a_count == 0 || b_count == 0 {
            continue;
        }

        let gradient: Vec<f64> = a_sum.iter().map(|g| g / a_count as f64).collect();
        let other: Vec<f64> = b_sum.iter().map(|g| g / b_count as f64).collect();
        variance += 0.5 * gradient.iter().zip(&other).map(|(x, y)| (x - y).powi(2)).sum::<f64>();

        let stepped: Vec<f64> = (0..values.len())
            .map(|i| if params[i].is_fixed { values[i] } else { gradient[i].signum().mul_add(-lr_mults[i], values[i]) })
            .collect();

        let (moved_sum, .., moved_count) = obj.batch_grad(first, &stepped, k, blend);
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

    let (first, last) = (config.lr_schedule.rate(1, config.epochs), config.lr_schedule.rate(config.epochs, config.epochs));
    for (label, lr) in [("β₂ first", first), ("β₂ last", last)] {
        let a = (4.0 * (drift * lr).powi(2) / variance).cbrt().clamp(1e-6, 1.0);
        row(label, &format!("{:.4}", 1.0 - a), &format!("lr {lr:.5}"));
    }
    row("configured", &format!("{:.4}", config.beta2), "");
}

/// Times sequential, cache-blocked, and shuffled index gathers over identical records and math.
pub(crate) fn gather_cost(obj: &Objective, config: &EvalTuneConfig) {
    const MAX_BATCHES: usize = 200;

    let train_len = obj.train.len();
    let values = eval_params::default_values(&eval_params::collect_parameters());
    let k = 0.5 * (config.k_min + config.k_max);
    let batches = (train_len / config.batch_size).clamp(1, MAX_BATCHES);
    let total_positions = (batches * config.batch_size).min(train_len) as f64;
    let mut indices: Vec<u32> = (0..train_len as u32).collect();

    let bench_pass = |order: &[u32]| {
        let start = Instant::now();
        for batch in order.chunks(config.batch_size).take(batches) {
            black_box(obj.batch_grad(batch, &values, k, 0.0));
        }
        start.elapsed().as_secs_f64()
    };

    let mut shuffler = Shuffler::new(train_len);
    let sequential_sec = bench_pass(&indices);
    let blocked_sec = (config.shuffle_block > 0).then(|| {
        shuffler.fill_blocked(&mut indices, 0xC0FFEE, config.shuffle_block, train_len);
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

pub(crate) fn val_cost(obj: &Objective, config: &EvalTuneConfig) {
    const VAL_TRIALS: usize = 7;

    let values = eval_params::default_values(&eval_params::collect_parameters());
    let k = 0.5 * (config.k_min + config.k_max);

    let eval_fused = || {
        let start = Instant::now();
        black_box(obj.val_eval(&values, [(k, 1.0), (k, 0.0)]));
        start.elapsed().as_secs_f64()
    };

    let eval_split = || {
        let start = Instant::now();
        black_box(obj.val_eval(&values, [(k, 1.0)]));
        black_box(obj.val_eval(&values, [(k, 0.0)]));
        start.elapsed().as_secs_f64()
    };

    let _ = eval_fused();
    let _ = eval_split();

    // Interleave trials and keep the minimum to reject scheduling jitter.
    let (mut min_fused, mut min_split) = (f64::INFINITY, f64::INFINITY);
    for _ in 0..VAL_TRIALS {
        min_fused = min_fused.min(eval_fused());
        min_split = min_split.min(eval_split());
    }

    println!("\n{LAB}Val cost{RESET} {DIM}({} positions, best of {VAL_TRIALS}){RESET}", obj.val.len());
    println!("  {LAB}fused{RESET}    {VAL}{:7.2} ms{RESET}  {DIM}one traversal, two probes{RESET}", min_fused * 1e3);
    println!("  {LAB}split{RESET}    {VAL}{:7.2} ms{RESET}  {DIM}two traversals, one probe each{RESET}", min_split * 1e3);
    println!(
        "  {LAB}saved{RESET}    {VAL}{:7.2} ms{RESET}  {VAL}{:.2}×{RESET} {DIM}per epoch{RESET}",
        (min_split - min_fused) * 1e3,
        min_split / min_fused
    );
}

struct ProbePoint {
    params: Vec<Tunable>,
    values: Vec<f64>,
    k: f64,
    blend: f64,
}

#[derive(Clone, Copy)]
struct NoiseEstimate {
    signal: f64,
    noise: f64,
    degenerate: usize,
}

#[derive(Clone, Copy, Default)]
struct Rung {
    positions: u64,
    groups: u64,
    flipped: u64,
    compared: u64,
    flipped_mass: f64,
    mass: f64,
}

impl Rung {
    // One sample's error rate from pairwise disagreement m = 2p(1 − p), clamped past m = 0.5.
    fn error(disagreement: f64) -> f64 { 0.5 * (1.0 - (1.0 - 2.0 * disagreement).max(0.0).sqrt()) }
}

// Every probe measures at the shipped parameters and opening (K, blend) so their figures
// compare on a common footing.
fn baseline_point(obj: &Objective, config: &EvalTuneConfig) -> ProbePoint {
    let params = eval_params::collect_parameters();
    let values = eval_params::default_values(&params);
    let blend = config.wdl_schedule.blend(1, config.epochs);
    let k = KController::bootstrap(config, obj, &values, &values, blend, None).k();
    ProbePoint { params, values, k, blend }
}

fn pool_noise_trials(
    obj: &Objective,
    point: &ProbePoint,
    big: usize,
    indices: &mut [u32],
    shuffler: &mut Shuffler,
    ladder: &mut [Rung],
) -> NoiseEstimate {
    let chunks = (big / SMALL_CHUNK).max(2);
    let chunk_len = big / chunks;

    let (mut pooled_signal, mut pooled_noise, mut degenerate) = (0.0, 0.0, 0usize);
    let (mut chunk_sums, mut chunk_counts): (Vec<Vec<f64>>, Vec<usize>) = (Vec::new(), Vec::new());

    for trial in 0..TRIALS {
        shuffler.fill(indices, 0x0BAD_5EED ^ trial as u64);

        let mut union_sum = vec![0.0; point.values.len()];
        let (mut union_count, mut small_sq, mut small_count) = (0usize, 0.0, 0usize);
        chunk_sums.clear();
        chunk_counts.clear();

        for chunk in indices[..big].chunks(chunk_len) {
            let (grad, .., count) = obj.batch_grad(chunk, &point.values, point.k, point.blend);
            if count == 0 {
                continue;
            }

            small_sq += grad.iter().map(|g| (g / count as f64).powi(2)).sum::<f64>();
            small_count += count;

            add_slice(&mut union_sum, &grad);

            union_count += count;
            chunk_sums.push(grad);
            chunk_counts.push(count);
        }

        if union_count == 0 || small_count == 0 {
            continue;
        }

        accumulate_sign_error(&chunk_sums, &chunk_counts, &union_sum, union_count, ladder);

        // The vol filter drops positions, so sizes are counted rather than assumed.
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

    NoiseEstimate { signal: pooled_signal, noise: pooled_noise, degenerate }
}

// Disjoint equal-size pairs disagree at 2p(1 − p), which Rung::error inverts; weighting
// by the whole draw avoids |a + b| shrinking exactly when a pair flips.
fn accumulate_sign_error(sums: &[Vec<f64>], counts: &[usize], union_sum: &[f64], union_count: usize, ladder: &mut [Rung]) {
    let weights: Vec<f64> = union_sum.iter().map(|g| (g / union_count as f64).abs()).collect();

    for (rung, group) in ladder.iter_mut().zip(successors(Some(1usize), |g| Some(g * 2))) {
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

fn print_noise_report(config: &EvalTuneConfig, k: f64, big: usize, est: NoiseEstimate, ladder: &[Rung]) {
    println!("\n{LAB}Noise scale{RESET} {DIM}({TRIALS} trials against {big} positions at K = {k:.6}){RESET}");
    if est.noise <= 0.0 {
        println!("  {LAB}unresolved{RESET}    the larger batch measured no less noise, so raise the trial count");
        return;
    }
    if est.signal <= 0.0 {
        println!("  {LAB}unresolved{RESET}    both batches are pure noise here, putting the scale above {big}");
        return;
    }

    let b_simple = est.noise / est.signal;
    println!("  {LAB}|g|²{RESET}          {VAL}{:.4e}{RESET}", est.signal / TRIALS as f64);
    println!("  {LAB}tr(Σ){RESET}         {VAL}{:.4e}{RESET}", est.noise / TRIALS as f64);
    println!("  {LAB}B_simple{RESET}      {VAL}{b_simple:.0}{RESET}  {DIM}positions past which averaging stops paying{RESET}");
    println!(
        "  {LAB}configured{RESET}    {VAL}{}{RESET}  {DIM}{:.2}× B_simple{RESET}",
        config.batch_size,
        config.batch_size as f64 / b_simple
    );

    if est.degenerate > 0 {
        println!(
            "  {LAB}unstable{RESET}      {VAL}{}{RESET} of {TRIALS} {DIM}trials estimated a negative term{RESET}",
            est.degenerate
        );
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
