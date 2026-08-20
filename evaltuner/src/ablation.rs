//! Measures the loss delta from zeroing one term's parameter group without refitting.
//!
//! The delta reflects a group's marginal contribution at the current parameter point.
//! Redundant groups each report small individual deltas because the unablated features
//! mask the omission; this interaction is captured by `curvature`.

use std::ops::Range;

use rayon::prelude::*;

use crate::{
    engine::{BLOCKS, EpdEntry, Group as BlockGroup, ReplayFilter, SoulEntry, TABLE_SQUARES, eval_params, parse_viri_file},
    loader::load_epd,
    report::PIECE_NAMES,
    scale::golden_search_k,
    training::{self, TunableData},
};

pub fn run_ablation(dataset_paths: &[String], filter: &ReplayFilter) {
    let values = eval_params::default_values(&eval_params::collect_parameters());

    println!("Loading dataset...");

    let mut soul_entries: Vec<SoulEntry> = Vec::new();
    let mut epd_entries: Vec<EpdEntry> = Vec::new();

    for path in dataset_paths {
        if path.ends_with(".vf") || path.ends_with(".viri") {
            match parse_viri_file(path, filter) {
                Ok((batch, ..)) => soul_entries.extend(batch),
                Err(e) => eprintln!("Skipping {path}: {e}"),
            }
        } else if path.ends_with(".epd") || path.ends_with(".txt") {
            match load_epd(path) {
                Ok(batch) => epd_entries.extend(batch),
                Err(e) => eprintln!("Skipping {path}: {e}"),
            }
        } else {
            eprintln!("Skipping {path}: unknown format");
        }
    }

    if soul_entries.is_empty() && epd_entries.is_empty() {
        eprintln!("No data loaded.");
        return;
    }

    if !soul_entries.is_empty() {
        println!("  {} positions (packed)", soul_entries.len());
        ablate(&soul_entries, &values);
    }
    if !epd_entries.is_empty() {
        if !soul_entries.is_empty() {
            println!();
        }
        println!("  {} positions (.epd)", epd_entries.len());
        ablate(&epd_entries, &values);
    }
}

struct Group {
    name: String,
    range: Range<usize>,
}

struct AblationResult {
    name: String,
    delta: f64,
}

fn ablate<T: TunableData>(entries: &[T], values: &[f64]) {
    let groups = build_groups();

    println!("Optimizing K...");
    let k = golden_search_k(0.001, 0.012, 1e-8, |k| compute_loss(entries, values, k));
    let baseline = compute_loss(entries, values, k);
    println!("  K = {k:.6}    baseline MSE = {baseline:.6}\n");

    let mut params = values.to_vec();
    let mut results: Vec<AblationResult> = Vec::with_capacity(groups.len());

    for group in &groups {
        params[group.range.clone()].fill(0.0);
        let delta = compute_loss(entries, &params, k) - baseline;
        params[group.range.clone()].copy_from_slice(&values[group.range.clone()]);

        results.push(AblationResult { name: group.name.clone(), delta });
    }

    results.sort_by(|a, b| b.delta.total_cmp(&a.delta));
    print_report(&results);
}

fn print_report(results: &[AblationResult]) {
    const BAR_WIDTH: usize = 32;
    let peak = results.first().map_or(0.0, |r| r.delta);

    println!("Term Ablation Report\n");
    println!("  {:<20} {:>11}  Impact", "Group", "dL");
    println!("  {}", "─".repeat(20 + 13 + BAR_WIDTH + 3));

    for r in results {
        let bar = if peak < 1e-15 || r.delta < peak * 0.001 || r.delta < 1e-5 {
            "·".to_string()
        } else {
            let w = ((r.delta / peak).min(1.0) * BAR_WIDTH as f64) as usize;
            "█".repeat(w.max(1))
        };

        println!("  {:<20} {:>+11.8}  {bar}", r.name, r.delta);
    }
}

/// Partitions evaluation parameters into disjoint, named ablation groups.
fn build_groups() -> Vec<Group> {
    // Layout keeps PSQT as one 384-slot block; ablation wants a piece table at a time,
    // so these six are built by hand and the block itself is filtered out below.
    // One piece's table is both phases of the mirrored half-board.
    const TABLE: usize = 2 * TABLE_SQUARES;
    let psqt = eval_params::LAYOUT.psqt_offset;

    let mut groups: Vec<Group> = PIECE_NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| Group { name: format!("{name} PSQT"), range: psqt + i * TABLE..psqt + (i + 1) * TABLE })
        .collect();

    groups.extend(
        BLOCKS
            .iter()
            .filter(|b| b.group != BlockGroup::Psqt)
            .map(|b| Group { name: title_case(b.name), range: b.offset..b.offset + b.len }),
    );

    groups
}

fn title_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for word in name.split('_') {
        if !out.is_empty() {
            out.push(' ');
        }
        let mut chars = word.chars();
        if let Some(first) = chars.next() {
            out.extend(first.to_uppercase());
            out.push_str(chars.as_str());
        }
    }
    out
}

fn compute_loss<T: TunableData>(entries: &[T], values: &[f64], k: f64) -> f64 {
    let sum_sq_err: f64 = entries
        .par_iter()
        .map(|e| {
            let err = training::sigmoid(e.eval(values), k) - e.result();
            err * err
        })
        .sum();

    sum_sq_err / entries.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_tile_the_parameter_vector() {
        let mut covered = vec![false; eval_params::LAYOUT.total];
        for g in build_groups() {
            for i in g.range {
                assert!(!covered[i], "slot {i} belongs to multiple ablation groups");
                covered[i] = true;
            }
        }
        if let Some(i) = covered.iter().position(|&c| !c) {
            panic!("slot {i} is unassigned to any ablation group");
        }
    }
}
