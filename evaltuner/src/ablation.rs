//! Evaluation term ablation via sequential parameter group zeroing.
//!
//! Quantifies the marginal predictive contribution of individual evaluation terms
//! (e.g. piece-square tables, mobility, pawn structure) by zeroing each parameter group
//! against a baseline evaluation vector and measuring the validation loss delta.

use std::ops::Range;

use rayon::prelude::*;

use crate::{
    engine::{BLOCKS, Group as BlockGroup, eval_params},
    loader::{self, EpdEntry, ReplayFilter, SoulEntry},
    scale::golden_search_k,
    training::{self, TunableData},
};

/// Loads dataset files and executes parameter ablation across all detected formats.
pub fn run_ablation(dataset_paths: &[String], filter: &ReplayFilter) {
    let values = eval_params::default_values(&eval_params::collect_parameters());

    println!("Loading dataset...");

    let mut soul_entries: Vec<SoulEntry> = Vec::new();
    let mut epd_entries: Vec<EpdEntry> = Vec::new();

    for path in dataset_paths {
        if path.ends_with(".vf") || path.ends_with(".viri") {
            match loader::parse_viri_file(path, filter) {
                Ok((batch, ..)) => soul_entries.extend(batch),
                Err(e) => eprintln!("Skipping {path}: {e}"),
            }
        } else if path.ends_with(".epd") || path.ends_with(".txt") {
            match loader::load_epd(path) {
                Ok(batch) => epd_entries.extend(batch),
                Err(e) => eprintln!("Skipping {path}: {e}"),
            }
        } else {
            eprintln!("Skipping {path}: unknown format");
        }
    }

    let mut ran = false;

    if !soul_entries.is_empty() {
        println!("  {} positions (packed)", soul_entries.len());
        run_generic(&soul_entries, &values);
        ran = true;
    }
    if !epd_entries.is_empty() {
        if ran {
            println!();
        }
        println!("  {} positions (.epd)", epd_entries.len());
        run_generic(&epd_entries, &values);
        ran = true;
    }

    if !ran {
        eprintln!("No data loaded.");
    }
}

/// Contiguous parameter slice representing a cohesive evaluation term.
struct Group {
    name: String,
    range: Range<usize>,
}

struct AblationResult {
    name: String,
    delta: f64,
}

fn run_generic<T: TunableData>(entries: &[T], values: &[f64]) {
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

/// Prints the sorted ablation impact table with proportional magnitude bars.
fn print_report(results: &[AblationResult]) {
    const BAR_WIDTH: usize = 32;
    let peak = results.first().map_or(0.0, |r| r.delta);

    println!("Term Ablation Report\n");
    println!("  {:<20} {:>11}  Impact", "Group", "dL_val");
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
    // Layout keeps PSQT as one 384-slot block; ablation wants a piece table at a time, so
    // these six are built by hand and the block itself is filtered out below.
    const PIECE_PSQTS: [&str; 6] = ["Pawn PSQT", "Knight PSQT", "Bishop PSQT", "Rook PSQT", "Queen PSQT", "King PSQT"];

    let mut groups: Vec<Group> = PIECE_PSQTS
        .iter()
        .enumerate()
        .map(|(i, &name)| Group { name: name.to_string(), range: i * 64..(i + 1) * 64 })
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

/// Computes dataset MSE for a given parameter vector and sigmoid scale `k`.
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
