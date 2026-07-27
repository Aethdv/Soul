//! Term ablation: zeroes each parameter group and measures the MSE delta
//! against the baseline.
//!
//! Groups are defined by layout ranges (PSQT per piece type, then each
//! eval term). Uses [`TunableData`] to run on both `.soul` and raw EPD.

use std::{mem, ops::Range};

use soul::{core::psqt, engine::eval_params};

use super::evaltuner::golden_search_k;
use crate::evaltune::{
    loader::{self, Entry, SoulEntry},
    training::{self, TunableData},
};

pub fn run_ablation(dataset_paths: &[String]) {
    let values: Vec<f64> = eval_params::collect_parameters().iter().map(|p| p.value).collect();

    println!("Loading dataset...");

    let mut soul_entries: Vec<SoulEntry> = Vec::new();
    let mut epd_entries: Vec<Entry> = Vec::new();

    for path in dataset_paths {
        if path.ends_with(".soul") || path.ends_with(".soul.zst") {
            match loader::load_encoded(path) {
                Ok(batch) => soul_entries.extend(batch),
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
        println!("  {} positions (.soul)", soul_entries.len());
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

struct Group {
    name: &'static str,
    range: Range<usize>,
}

struct AblationResult {
    name: &'static str,
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
        // Zero this group, measure impact, restore.
        let orig: Vec<f64> = group.range.clone().map(|i| mem::replace(&mut params[i], 0.0)).collect();

        let delta = compute_loss(entries, &params, k) - baseline;

        for (i, v) in group.range.clone().zip(orig) {
            params[i] = v;
        }

        results.push(AblationResult { name: group.name, delta });
    }

    results.sort_by(|a, b| b.delta.total_cmp(&a.delta));
    print_report(&results);
}

/// Print the sorted ablation table with proportional bars.
fn print_report(results: &[AblationResult]) {
    const BAR_W: usize = 32;
    let peak = results.first().map_or(0.0, |r| r.delta);

    println!("Term Ablation Report\n");
    println!("  {:<20} {:>11}  Impact", "Group", "dL_val");
    println!("  {}", "─".repeat(20 + 13 + BAR_W + 3));

    for r in results {
        let bar = if peak < 1e-15 || r.delta < peak * 0.001 || r.delta < 1e-5 {
            "·".into()
        } else {
            let w = ((r.delta / peak).min(1.0) * BAR_W as f64) as usize;
            "█".repeat(w.max(1))
        };

        let sign = if r.delta >= 0.0 { "+" } else { "-" };
        println!("  {:<20} {}{:>.8}  {}", r.name, sign, r.delta.abs(), bar);
    }
}

fn build_groups() -> Vec<Group> {
    let l = &psqt::LAYOUT;

    let mut groups: Vec<Group> = ["Pawn PSQT", "Knight PSQT", "Bishop PSQT", "Rook PSQT", "Queen PSQT", "King PSQT"]
        .iter()
        .enumerate()
        .map(|(i, &name)| Group { name, range: i * 64..(i + 1) * 64 })
        .collect();

    #[rustfmt::skip]
    let terms: &[(&str, usize, usize)] = &[
        ("Material",         l.material_offset,           12),
        ("Mobility Open",    l.mobility_open_offset,       8),
        ("Mobility Closed",  l.mobility_closed_offset,     8),
        ("Phase Weights",    l.phase_offset,               6),
        ("Attacker Weights", l.attacker_offset,            6),
        ("King Safety",      l.king_safety_offset,         3),
        ("Xray",             l.xray_offset,                1),
        ("Bishop Pair",      l.bishop_pair_offset,         2),
        ("Rook Open",        l.rook_open_offset,           2),
        ("Passed Pawn",      l.passed_pawn_mg_offset,     12),
        ("Enemy King Dist",  l.enemy_king_dist_mg_offset, 12),
    ];

    groups.extend(terms.iter().map(|&(name, off, len)| Group { name, range: off..off + len }));
    groups
}

/// MSE over the dataset for a given parameter set and K.
fn compute_loss<T: TunableData>(entries: &[T], values: &[f64], k: f64) -> f64 {
    let n = entries.len() as f64;
    entries
        .iter()
        .map(|e| {
            let err = training::sigmoid(e.eval(values), k) - e.result();
            err * err
        })
        .sum::<f64>()
        / n
}
