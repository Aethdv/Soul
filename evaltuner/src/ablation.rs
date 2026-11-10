//! Marginal loss delta from zeroing one parameter group at a time without refitting.

use std::ops::Range;

use crate::{
    config::EvalTuneConfig,
    engine::{BLOCKS, Group as BlockGroup, TABLE_SQUARES, eval_params},
    probes::{ProbePoint, baseline_point},
    report::PIECE_NAMES,
    run::Objective,
};

pub(crate) fn ablation_report(obj: &Objective, config: &EvalTuneConfig) {
    let ProbePoint { params, values, k, blend } = baseline_point(obj, config);
    let probe = [(k, blend)];
    let baseline = obj.all_eval(&values, probe)[0];
    println!("  loss = {}    K = {k:.6}    blend = {blend}    baseline = {baseline:.6}\n", obj.loss_fn());

    let mut ablated = values.clone();
    let mut results: Vec<(String, f64)> = term_groups()
        .into_iter()
        .filter(|(_, range)| params[range.clone()].iter().any(|p| !p.is_fixed))
        .map(|(name, range)| {
            ablated[range.clone()].fill(0.0);
            let delta = obj.all_eval(&ablated, probe)[0] - baseline;
            ablated[range.clone()].copy_from_slice(&values[range]);
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
