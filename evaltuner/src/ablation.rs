//! Marginal loss delta from zeroing one parameter group at a time without refitting.

use std::ops::Range;

use rayon::prelude::*;

use crate::{
    engine::{BLOCKS, FeatureRecord, Group as BlockGroup, ReplayFilter, TABLE_SQUARES, eval_params, eval_record, sigmoid},
    loader::{load_datasets, resolve_dataset_paths},
    report::PIECE_NAMES,
    scale::golden_search_k,
};

const ABLATION_K: (f64, f64) = (0.001, 0.012);
const K_TOL: f64 = 1e-8;

pub(crate) fn run_ablation(dataset_paths: &[String], filter: &ReplayFilter) {
    let Some(paths) = resolve_dataset_paths(&dataset_paths.join(",")) else {
        return;
    };

    let loaded = load_datasets(&paths, filter);
    if loaded.entries.is_empty() {
        eprintln!("No data loaded.");
        return;
    }

    println!("  {} positions", loaded.entries.len());
    let records: Vec<FeatureRecord> = loaded.entries.par_iter().map(FeatureRecord::from_entry).collect();
    drop(loaded);

    let params = eval_params::collect_parameters();
    let defaults = eval_params::default_values(&params);
    let groups = term_groups()
        .into_iter()
        .filter(|(_, range)| params[range.clone()].iter().any(|p| !p.is_fixed));

    println!("Optimizing K...");
    let k = golden_search_k(ABLATION_K.0, ABLATION_K.1, K_TOL, |k| mse_loss(&records, &defaults, k));
    let baseline = mse_loss(&records, &defaults, k);
    println!("  K = {k:.6}    baseline MSE = {baseline:.6}\n");

    let mut ablated = defaults.clone();
    let mut results: Vec<(String, f64)> = groups
        .map(|(name, range)| {
            ablated[range.clone()].fill(0.0);
            let delta = mse_loss(&records, &ablated, k) - baseline;
            ablated[range.clone()].copy_from_slice(&defaults[range]);
            (name, delta)
        })
        .collect();

    results.sort_by(|a, b| b.1.total_cmp(&a.1));
    print_report(&results);
}

fn print_report(results: &[(String, f64)]) {
    const BAR_WIDTH: usize = 32;
    let peak = results.first().map_or(0.0, |r| r.1);

    println!("Term Ablation Report\n");
    println!("  {:<20} {:>11}  Impact", "Group", "dL");
    println!("  {}", "─".repeat(20 + 13 + BAR_WIDTH + 3));

    for (name, delta) in results {
        let bar = if peak < 1e-15 || *delta < peak * 0.001 || *delta < 1e-5 {
            "·".to_string()
        } else {
            let w = ((*delta / peak).min(1.0) * BAR_WIDTH as f64) as usize;
            "█".repeat(w.max(1))
        };
        println!("  {name:<20} {delta:>+11.8}  {bar}");
    }
}

fn term_groups() -> Vec<(String, Range<usize>)> {
    const TABLE: usize = 2 * TABLE_SQUARES;
    let psqt = eval_params::LAYOUT.psqt_offset;

    let mut groups: Vec<(String, Range<usize>)> = PIECE_NAMES
        .iter()
        .enumerate()
        .map(|(i, &name)| (format!("{name} PSQT"), psqt + i * TABLE..psqt + (i + 1) * TABLE))
        .collect();

    groups.extend(
        BLOCKS
            .iter()
            .filter(|b| b.group != BlockGroup::Psqt)
            .map(|b| (title_case(b.name), b.offset..b.offset + b.len)),
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

fn mse_loss(records: &[FeatureRecord], values: &[f64], k: f64) -> f64 {
    let sum_sq_err: f64 = records
        .par_iter()
        .map(|r| {
            let err = sigmoid(eval_record(r, values), k) - f64::from(r.result) / 2.0;
            err * err
        })
        .sum();
    sum_sq_err / records.len() as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn groups_tile_parameter_vector() {
        let mut covered = vec![false; eval_params::LAYOUT.total];
        for (_, range) in term_groups() {
            for i in range {
                assert!(!covered[i], "slot {i} belongs to multiple ablation groups");
                covered[i] = true;
            }
        }
        if let Some(i) = covered.iter().position(|&c| !c) {
            panic!("slot {i} is unassigned to any ablation group");
        }
    }
}
