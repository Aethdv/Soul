//! One-shot measurements the tuner can be asked for instead of a training run.
//!
//! Each answers a question the epoch line cannot: what the data determines at all,
//! what binds the gradient loop, and what the fused validation pass saves.

use std::time::Instant;

use rayon::prelude::*;
use soul::{engine::eval_params, tools::dataset::accumulate_record_grad};

use super::{
    curvature::Curvature,
    loader,
    palette::{self, RESET},
    run::TrainerContext,
    scale::KController,
    training::sigmoid,
};
use crate::core::{config::EvalTuneConfig, shuffle::Shuffler};

/// Curvature of the loss at the shipped parameters, over the training split.
///
/// The Hessian needs the eval's raw coefficient vector per position, and `accumulate_record_grad`
/// already produces exactly that: its `gradient` argument is a scalar multiplier, so passing 1.0
/// leaves the coefficients themselves in the scratch buffer. Nothing about the training path has
/// to change to read them.
pub fn curvature_report(ctx: &TrainerContext, config: &EvalTuneConfig) {
    let params = eval_params::collect_parameters();
    let values: Vec<f64> = params.iter().map(|p| p.value).collect();

    let blend = config.wdl_schedule.clone().into_scheduler().blend(1, config.epochs);
    let k = KController::bootstrap(config, ctx, &values, &values, blend, None).k();

    let free: Vec<usize> = params.iter().filter(|p| !p.is_fixed).map(|p| p.idx).collect();
    let n = params.len();
    let trainable: Vec<bool> = {
        let mut mask = vec![false; n];

        for &i in &free {
            mask[i] = true;
        }

        mask
    };

    let curvature = (0..ctx.train_count)
        .into_par_iter()
        .fold(
            || (Curvature::zeros(n), vec![0.0; n], Vec::with_capacity(64)),
            |(mut acc, mut scratch, mut nonzeros), i| {
                let (entry, record) = (&ctx.train[i], &ctx.records[i]);

                if ctx.passes_vol_filter(entry, record.static_eval) {
                    let eval = loader::eval_record_full(record, &values);
                    let p = sigmoid(eval.score, k);
                    let w = if ctx.phase_weights.is_empty() { 1.0 } else { ctx.phase_weights[i] };

                    accumulate_record_grad(record, &eval, 1.0, &mut scratch);

                    // One walk drains the buffer and collects it, so the next position starts from
                    // zero without paying to clear all 490 slots again.
                    nonzeros.clear();

                    for (j, coefficient) in scratch.iter_mut().enumerate() {
                        if *coefficient != 0.0 {
                            if trainable[j] {
                                nonzeros.push((j, *coefficient));
                            }

                            *coefficient = 0.0;
                        }
                    }

                    acc.add_outer(k * k * w * p * (1.0 - p), &nonzeros);
                }

                (acc, scratch, nonzeros)
            },
        )
        .map(|(acc, ..)| acc)
        .reduce(
            || Curvature::zeros(n),
            |mut acc, other| {
                acc.merge(&other);
                acc
            },
        )
        .symmetrized();

    curvature.spectrum(&free).report(&params, ctx.train_count, k);
}

/// Gradient throughput over sequential batches against shuffled ones.
///
/// `grad` is 90% of an epoch and the open question is what binds it. This holds the arithmetic and
/// the records fixed and varies only the order they are visited in, so the gap is the whole cost of
/// gathering `FeatureRecord`s at random: a wide gap points at latency and the TLB, and names the
/// record layout and hugepages as the levers, while a narrow one says the math is the wall and
/// none of that pays.
pub fn gather_cost(ctx: &TrainerContext, config: &EvalTuneConfig) {
    const BATCHES: usize = 200;

    let params = eval_params::collect_parameters();
    let values: Vec<f64> = params.iter().map(|p| p.value).collect();
    let k = 0.5 * (config.k_min + config.k_max);

    let batches = (ctx.train_count / config.batch_size).clamp(1, BATCHES);
    let positions = (batches * config.batch_size) as f64;

    let mut indices: Vec<u32> = (0..ctx.train_count as u32).collect();

    let time_pass = |order: &[u32]| {
        let start = Instant::now();

        for batch in order.chunks(config.batch_size).take(batches) {
            // Discarded: only the read pattern is under measurement.
            let _ = ctx.batch_grad(batch, &values, k, 0.0);
        }

        start.elapsed().as_secs_f64()
    };

    // Sequential first, so the shuffled arm cannot be the one paying for a cold page cache.
    let sequential = time_pass(&indices);

    Shuffler::new(ctx.train_count).fill(&mut indices, 0xC0FFEE);
    let shuffled = time_pass(&indices);

    let lab = palette::fg(palette::LABEL);
    let v = palette::fg(palette::VALUE);
    let dim = palette::fg(palette::DIM);

    println!("\n{lab}Gather cost{RESET} {dim}({batches} batches of {}){RESET}", config.batch_size);
    println!("  {lab}sequential{RESET}  {v}{sequential:6.2}s{RESET}  {v}{:5.1}M pos/s{RESET}", positions / sequential / 1e6);
    println!("  {lab}shuffled{RESET}    {v}{shuffled:6.2}s{RESET}  {v}{:5.1}M pos/s{RESET}", positions / shuffled / 1e6);
    println!("  {lab}ratio{RESET}       {v}{:6.2}×{RESET}", shuffled / sequential);
}

pub fn val_cost(ctx: &TrainerContext, config: &EvalTuneConfig) {
    const REPEATS: usize = 7;

    let params = eval_params::collect_parameters();
    let values: Vec<f64> = params.iter().map(|p| p.value).collect();
    let k = 0.5 * (config.k_min + config.k_max);

    let fused = || {
        let start = Instant::now();
        let _ = ctx.val_eval(&values, [(k, 1.0), (k, 0.0)]);
        start.elapsed().as_secs_f64()
    };

    let split = || {
        let start = Instant::now();
        let _ = ctx.val_eval(&values, [(k, 1.0)]);
        let _ = ctx.val_eval(&values, [(k, 0.0)]);
        start.elapsed().as_secs_f64()
    };

    // Both arms untimed once, or the first one pays to fault in the val slice.
    let _ = fused();
    let _ = split();

    // Interleaved, and scored on the minimum: noise only ever adds time.
    let (mut best_fused, mut best_split) = (f64::INFINITY, f64::INFINITY);

    for _ in 0..REPEATS {
        best_fused = best_fused.min(fused());
        best_split = best_split.min(split());
    }

    let lab = palette::fg(palette::LABEL);
    let v = palette::fg(palette::VALUE);
    let dim = palette::fg(palette::DIM);

    println!("\n{lab}Val cost{RESET} {dim}({} positions, best of {REPEATS}){RESET}", ctx.val.len());
    println!("  {lab}fused{RESET}    {v}{:7.2} ms{RESET}  {dim}one traversal, two probes{RESET}", best_fused * 1e3);
    println!("  {lab}split{RESET}    {v}{:7.2} ms{RESET}  {dim}two traversals, one probe each{RESET}", best_split * 1e3);
    println!(
        "  {lab}saved{RESET}    {v}{:7.2} ms{RESET}  {v}{:.2}×{RESET} {dim}per epoch{RESET}",
        (best_split - best_fused) * 1e3,
        best_split / best_fused
    );
}
