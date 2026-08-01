//! What a dataset says, read off its labels before anything trains on them.
//!
//! Three reports. `profile` counts what the positions and the games offer a weight
//! to sit on. `material` fits ten coefficients to the labels alone, outside the tuner: a set whose
//! midgame queen sits at twice its endgame rook will not produce a shippable evaluation, and ten
//! weights say so in one pass instead of four thousand epochs. `score` puts parameter vectors
//! against datasets at each pair's own K, which is the only honest way to rank two runs that
//! trained on different data, since a validation loss describes its own split and nothing else.
//!
//! Every report takes several datasets and prints a row each, since these numbers mean little
//! alone. None of them splits, weights or shuffles: an assay describes a file, and a holdout would
//! describe a tenth of one.

use std::path::Path;

use rayon::prelude::*;

use super::{
    engine::{
        DECISIVE_ENDING, FeatureRecord, GameScan, LAYOUT, QUIET_ENDING, SoulEntry, TOTAL_PHASE, collect_parameters, default_values,
        eval_record, format_comma, scan_viri_games, sigmoid,
    },
    loader::{load_datasets, resolve_dataset_paths},
    palette::{ALARM, COUNT, DIM, LAB, RESET, VAL},
    run::replay_filter,
    scale::golden_search_k,
    storage::load_checkpoint,
    training::{phase_of, phase_weights},
};
use crate::core::config::{EvalTuneConfig, LossFn};

/// A constant 0.5 prediction, in nats.
const COIN_FLIP: f64 = std::f64::consts::LN_2;

/// The bracket the yardstick searches for K, wider than a training run's.
///
/// A cold-start vector is logged before the gauge normalizes it and can sit a factor of four off
/// the centipawn scale, so a bracket cut to what a shipped-scale eval needs would clamp such a
/// candidate at the edge and report a loss that is about its units rather than its opinions.
const SCORE_K: (f64, f64) = (1e-5, 0.2);

/// The sigmoid scale the material fit holds fixed.
///
/// Weights and K trade off exactly, so fixing one identifies the other, and the printed table is
/// normalized past both.
const FIT_K: f64 = 0.0025;

const FIT_STEPS: usize = 24;
const FIT_TOL: f64 = 1e-10;

/// The five piece types a material table has an opinion about; the king is fixed at zero.
const PIECES: [&str; 5] = ["pawn", "knight", "bishop", "rook", "queen"];

/// The same five where a column header has room for one letter.
const SYMBOLS: [&str; 5] = ["P", "N", "B", "R", "Q"];

/// Which question the datasets are being asked.
pub enum Assay {
    /// Loss of each parameter vector on each set.
    Score { params: Vec<String>, loss: LossFn, shipped: String },
    /// Tapered material-only logistic fit, one row per set against the shipped table.
    Material { shipped: String },
    /// Label, design and game-shape counts.
    Profile,
}

/// One dataset argument, loaded and featurized.
struct Set {
    label: String,
    paths: Vec<String>,
    entries: Vec<SoulEntry>,
    records: Vec<FeatureRecord>,
    /// Positions the file yielded, which `entries` stops holding once a sample is taken.
    loaded: usize,
}

/// Load each argument as its own set, then report.
///
/// One argument is one row, so a comma-joined argument loads as one set: the spelling that trains
/// on several files assays them as the mixture they would train as.
pub fn run(report: &Assay, datasets: &[String], config: &EvalTuneConfig, sample: Option<usize>) {
    let filter = replay_filter(config);
    let mut sets = Vec::new();

    for dataset in datasets {
        let Some(paths) = resolve_dataset_paths(dataset) else {
            continue;
        };

        let (mut entries, ..) = load_datasets(&paths, &filter);

        if entries.is_empty() {
            eprintln!("{ALARM}[!] No positions loaded from {dataset}.{RESET}");
            continue;
        }

        let loaded = entries.len();

        if let Some(cap) = sample {
            entries = strided(entries, cap);
        }

        println!("  Extracting features ({} entries)...", format_comma(entries.len() as u64));
        let records = entries.par_iter().map(FeatureRecord::from_entry).collect();

        sets.push(Set { label: label_for(&paths), paths, entries, records, loaded });
    }

    if sets.is_empty() {
        eprintln!("{ALARM}[!] Nothing to assay.{RESET}");
        return;
    }

    match report {
        Assay::Score { params, loss, shipped } => score(&sets, params, *loss, shipped),
        Assay::Material { shipped } => material(&sets, shipped),
        Assay::Profile => profile(&sets),
    }
}

/// Parameter vectors against datasets, each cell at the K that suits that pair best.
///
/// Refitting K per cell is what makes the numbers comparable: a cold-start run lands on whatever
/// scale its gauge left it at, and a vector three times too large is not three times worse at
/// predicting outcomes. The target is the game result alone; blending the search score in would
/// score a candidate against the generating engine's opinion.
fn score(sets: &[Set], params: &[String], loss: LossFn, shipped: &str) {
    let tunables = collect_parameters();
    let defaults = default_values(&tunables);
    let mut candidates = vec![(shipped.to_string(), defaults.clone())];

    for path in params {
        match load_checkpoint(path, &tunables, &defaults) {
            Ok(data) => {
                let values = if data.best_val_params.len() == tunables.len() { data.best_val_params } else { data.values };

                candidates.push((label_for(std::slice::from_ref(path)), values));
            },
            Err(e) => eprintln!("{ALARM}[!] Cannot read parameters from {path}: {e}{RESET}"),
        }
    }

    // One dataset is the yardstick case, where K and the headroom under a coin flip are the two
    // figures worth having. Several are a ranking, and K varies per cell rather than per row, so
    // there is no column for it to sit in.
    let single = sets.len() == 1;
    let mut columns: Vec<String> = sets.iter().map(|set| set.label.clone()).collect();

    if single {
        columns.push("K".to_string());
        columns.push("ln2 − L".to_string());
    }

    let mut table = Table::new("parameters", columns);

    for (name, values) in &candidates {
        let mut cells = Vec::new();
        let mut last = (0.0, 0.0);

        for set in sets {
            last = best_loss(set, values, loss);
            cells.push(format!("{:.6}", last.1));
        }

        if single {
            cells.push(format!("{:.6}", last.0));
            cells.push(format!("{:.6}", COIN_FLIP - last.1));
        }

        table.push(name, cells);
    }

    println!("\n{LAB}Loss of a parameter vector on a dataset, at that pair's own K{RESET}");
    table.print();

    if candidates.len() > 1 {
        println!("\n{DIM}  A checkpoint is read at its best-validation vector.{RESET}");
    }

    println!("\n{DIM}  {loss}, measured against the game result.");
    println!("  A constant 0.5 scores ln 2 = 0.693147; under 0.05 of headroom, a set has no outcome signal.{RESET}");
}

/// Ten coefficients over piece counts, fitted outside the tuner.
///
/// Material is nearly separable from the rest of the evaluation, so this recovers a set's opinion
/// of a piece without the other 480 parameters trading against it. Normalizing the midgame pawn to
/// 100 is what makes two sets comparable when their sigmoids landed on different slopes.
fn material(sets: &[Set], shipped: &str) {
    let columns: Vec<String> = SYMBOLS
        .iter()
        .map(|p| format!("{p}mg"))
        .chain(SYMBOLS.iter().map(|p| format!("{p}eg")))
        .collect();

    let mut table = Table::new("dataset", columns);
    let mut notes = Vec::new();

    table.split = Some(5);

    for set in sets {
        let (weights, loss, steps) = newton_fit(&material_rows(set));

        if weights[0].abs() < 1e-12 {
            eprintln!("{ALARM}[!] {}: the fit puts a midgame pawn at zero, so nothing normalizes.{RESET}", set.label);
            continue;
        }

        let scale = 100.0 / weights[0];

        table.push(&set.label, (0..10).map(|i| format!("{:.0}", weights[i] * scale)).collect());
        notes.push((
            set.label.clone(),
            format!(
                "{COUNT}{}{RESET} positions, L_fit {VAL}{loss:.4}{RESET}, {steps} Newton steps",
                format_comma(set.entries.len() as u64),
            ),
        ));
    }

    let defaults = default_values(&collect_parameters());
    let ship_at = |i: usize| defaults[LAYOUT.material_offset + i];
    let ship_scale = 100.0 / ship_at(0);

    table.push_dim(
        shipped,
        (0..10)
            .map(|i| format!("{:.0}", ship_at(if i < 5 { i } else { i + 1 }) * ship_scale))
            .collect(),
    );

    println!("\n{LAB}Material-only logistic fit, midgame beside endgame, the pawn pinned to 100{RESET}");
    table.print();
    println!();

    let width = notes.iter().map(|(label, _)| label.chars().count()).max().unwrap_or_default();

    for (label, note) in notes {
        println!("  {LAB}{label:<width$}{RESET}  {note}");
    }

    println!("{DIM}  L_fit is each set's loss on its own positions and does not compare across sets.{RESET}");
}

/// Counts: what the labels say, what the positions offer, and how the games ended.
fn profile(sets: &[Set]) {
    let stats: Vec<Counts> = sets.iter().map(count).collect();

    let mut labels = Table::new("dataset", strings(&["positions", "win", "draw", "loss", "STM", "score≠result"]));
    let mut design = Table::new("dataset", strings(&["phase", "under 12", "20 or more", "pieces", "scored", "|score|"]));
    let mut imbalance = Table::new("dataset", PIECES.iter().map(|p| format!("{p}≠")).collect());
    let mut at_phase = Table::new("dataset", PIECES.iter().map(|p| (*p).to_string()).collect());

    for (set, counts) in sets.iter().zip(&stats) {
        let n = set.entries.len() as f64;
        let pct = |count: usize| format!("{:.2}%", 100.0 * count as f64 / n);

        labels.push(&set.label, vec![
            format_comma(set.entries.len() as u64),
            pct(counts.results[2]),
            pct(counts.results[1]),
            pct(counts.results[0]),
            format!("{:.2}%", 100.0 * (counts.results[2] as f64 + 0.5 * counts.results[1] as f64) / n),
            pct(counts.contradictions),
        ]);

        design.push(&set.label, vec![
            format!("{:.2}", counts.phase_sum / n),
            pct(counts.phase_low),
            pct(counts.phase_high),
            format!("{:.1}", counts.pieces as f64 / n),
            pct(counts.scored),
            match counts.scored {
                0 => "-".to_string(),
                scored => format!("{:.0}", counts.score_sum / scored as f64),
            },
        ]);

        imbalance.push(&set.label, (0..5).map(|pt| pct(counts.unequal[pt])).collect());

        at_phase.push(
            &set.label,
            (0..5)
                .map(|pt| match counts.unequal[pt] {
                    0 => "-".to_string(),
                    seen => format!("{:.1}", counts.unequal_phase[pt] / seen as f64),
                })
                .collect(),
        );
    }

    println!("\n{LAB}Labels{RESET}");
    labels.print();

    println!("\n{LAB}Positions{RESET}");
    design.print();

    // A coefficient is fitted on the positions where its piece count differs, so a set with no
    // queen imbalances has nothing to say about a queen however many positions it holds.
    println!("\n{LAB}Imbalance: the share of positions where the count differs{RESET}");
    imbalance.print();

    println!("\n{LAB}And the mean phase where it does{RESET}");
    at_phase.print();

    print_games(sets);
}

/// The game table, for whichever sets came from a replay.
fn print_games(sets: &[Set]) {
    let scans: Vec<(&Set, GameScan)> = sets
        .iter()
        .filter_map(|set| {
            let path = set.paths.iter().find(|p| p.ends_with(".vf") || p.ends_with(".viri"))?;

            match scan_viri_games(path) {
                Ok(scan) if scan.games > 0 => Some((set, scan)),
                Ok(_) => None,
                Err(e) => {
                    eprintln!("{ALARM}[!] Cannot scan games in {path}: {e}{RESET}");
                    None
                },
            }
        })
        .collect();

    if scans.is_empty() {
        return;
    }

    let mut games = Table::new("dataset", {
        let mut columns = strings(&["games", "plies", "kept/game", "pieces left", "draws", "mate"]);

        columns.push(format!("past {DECISIVE_ENDING}"));
        columns.push(format!("inside {QUIET_ENDING}"));
        columns
    });

    for (set, scan) in &scans {
        let n = scan.games as f64;
        let pct = |count: usize| format!("{:.1}%", 100.0 * count as f64 / n);

        games.push(&set.label, vec![
            format_comma(scan.games as u64),
            format!("{:.1}", scan.plies as f64 / n),
            format!("{:.1}", set.loaded as f64 / n),
            format!("{:.1}", scan.pieces_left as f64 / n),
            pct(scan.results[1]),
            pct(scan.mate_endings),
            pct(scan.decisive_endings),
            pct(scan.quiet_endings),
        ]);
    }

    println!("\n{LAB}Games, unfiltered: the replay's independent unit{RESET}");
    games.print();
    println!("{DIM}  The last three are what the final score said; a game in none of them stopped for");
    println!("  something other than its score.{RESET}");
}

/// The K that suits this vector on this set, and the loss there.
///
/// The eval is the whole cost of the pass and does not depend on K, so it is paid once and the
/// golden search walks over cached scores.
fn best_loss(set: &Set, values: &[f64], loss: LossFn) -> (f64, f64) {
    let scored: Vec<(f64, f64)> = set
        .records
        .par_iter()
        .zip(&set.entries)
        .map(|(record, entry)| (eval_record(record, values), f64::from(entry.result) / 2.0))
        .collect();

    let loss_at = |k: f64| {
        let sum: f64 = scored.par_iter().map(|&(s, y)| loss.loss(sigmoid(s, k), y)).sum();

        sum / scored.len() as f64
    };

    let (k_min, k_max) = SCORE_K;
    let k = golden_search_k(k_min, k_max, 1e-6 * (k_max - k_min), loss_at);

    if k <= k_min * 1.01 || k >= k_max * 0.99 {
        eprintln!(
            "{ALARM}[!] K stopped at the edge of its bracket on {}: this loss is not the candidate's best.{RESET}",
            set.label
        );
    }

    (k, loss_at(k))
}

/// `(design, target)` per position: the five count differentials tapered into a midgame half and
/// an endgame half, against the STM-relative outcome.
fn material_rows(set: &Set) -> Vec<([f64; 10], f64)> {
    let phase_w = phase_weights();

    set.records
        .par_iter()
        .zip(&set.entries)
        .map(|(record, entry)| {
            let mg = f64::from(phase_of(record, &phase_w) as u32) / f64::from(TOTAL_PHASE);
            let mut design = [0.0; 10];

            for pt in 0..5 {
                let diff = f64::from(record.mat_diffs[pt]);

                design[pt] = diff * mg;
                design[pt + 5] = diff * (1.0 - mg);
            }

            (design, f64::from(entry.result) / 2.0)
        })
        .collect()
}

/// Newton on the log-likelihood: ten weights, the exact Hessian, and step halving for the
/// iterations where the quadratic model overshoots. Returns the weights, the loss and the steps.
///
/// Cross-entropy is not a setting here the way it is for the yardstick. The derivatives below are
/// the canonical-link forms, exact for this loss and wrong for any other, so a fit under a smoothed
/// or focal objective wants its own derivation rather than a different constant.
fn newton_fit(rows: &[([f64; 10], f64)]) -> ([f64; 10], f64, usize) {
    let mut w = [0.0f64; 10];
    let mut loss = fit_loss(rows, &w);

    for step in 1..=FIT_STEPS {
        let (gradient, hessian) = fit_derivatives(rows, &w);

        let Some(delta) = solve(hessian, gradient) else {
            return (w, loss, step);
        };

        // Full step first; a logistic fit on nearly separable columns is where it overshoots.
        let mut scale = 1.0;
        let mut moved = false;

        for _ in 0..8 {
            let candidate = std::array::from_fn(|i| w[i] - scale * delta[i]);
            let candidate_loss = fit_loss(rows, &candidate);

            if candidate_loss <= loss {
                w = candidate;
                loss = candidate_loss;
                moved = true;
                break;
            }

            scale *= 0.5;
        }

        let norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt() * scale;

        if !moved || norm < FIT_TOL {
            return (w, loss, step);
        }
    }

    (w, loss, FIT_STEPS)
}

/// Gradient and Hessian of the mean log-loss at `w`.
fn fit_derivatives(rows: &[([f64; 10], f64)], w: &[f64; 10]) -> ([f64; 10], [f64; 100]) {
    let (gradient, hessian) = rows
        .par_iter()
        .fold(
            || ([0.0f64; 10], [0.0f64; 100]),
            |(mut g, mut h), (x, y)| {
                let p = sigmoid(dot(x, w), FIT_K);
                let residual = FIT_K * (p - y);
                let curvature = FIT_K * FIT_K * p * (1.0 - p);

                for i in 0..10 {
                    if x[i] == 0.0 {
                        continue;
                    }

                    g[i] += residual * x[i];

                    for j in i..10 {
                        h[i * 10 + j] += curvature * x[i] * x[j];
                    }
                }

                (g, h)
            },
        )
        .reduce(
            || ([0.0f64; 10], [0.0f64; 100]),
            |(mut g, mut h), (dg, dh)| {
                for i in 0..10 {
                    g[i] += dg[i];
                }

                for i in 0..100 {
                    h[i] += dh[i];
                }

                (g, h)
            },
        );

    let n = rows.len() as f64;
    let gradient = std::array::from_fn(|i| gradient[i] / n);
    let mut hessian: [f64; 100] = std::array::from_fn(|i| hessian[i] / n);

    for i in 0..10 {
        for j in 0..i {
            hessian[i * 10 + j] = hessian[j * 10 + i];
        }
    }

    // A set that never varies one piece count leaves that row empty, which is a singular matrix
    // rather than a bad answer. The ridge keeps the solve alive and pins the coefficient near zero.
    for i in 0..10 {
        hessian[i * 10 + i] += 1e-12;
    }

    (gradient, hessian)
}

fn fit_loss(rows: &[([f64; 10], f64)], w: &[f64; 10]) -> f64 {
    let sum: f64 = rows
        .par_iter()
        .map(|(x, y)| LossFn::CrossEntropy.loss(sigmoid(dot(x, w), FIT_K), *y))
        .sum();

    sum / rows.len() as f64
}

fn dot(x: &[f64; 10], w: &[f64; 10]) -> f64 {
    (0..10).map(|i| x[i] * w[i]).sum()
}

/// Gaussian elimination with partial pivoting, row-major and ten wide.
fn solve(mut a: [f64; 100], mut b: [f64; 10]) -> Option<[f64; 10]> {
    for col in 0..10 {
        let pivot = (col..10).max_by(|&i, &j| a[i * 10 + col].abs().total_cmp(&a[j * 10 + col].abs()))?;

        if a[pivot * 10 + col].abs() < 1e-18 {
            return None;
        }

        if pivot != col {
            for k in 0..10 {
                a.swap(col * 10 + k, pivot * 10 + k);
            }

            b.swap(col, pivot);
        }

        for row in col + 1..10 {
            let factor = a[row * 10 + col] / a[col * 10 + col];

            if factor == 0.0 {
                continue;
            }

            for k in col..10 {
                a[row * 10 + k] -= factor * a[col * 10 + k];
            }

            b[row] -= factor * b[col];
        }
    }

    let mut x = [0.0f64; 10];

    for row in (0..10).rev() {
        let known: f64 = (row + 1..10).map(|k| a[row * 10 + k] * x[k]).sum();

        x[row] = (b[row] - known) / a[row * 10 + row];
    }

    Some(x)
}

fn count(set: &Set) -> Counts {
    let phase_w = phase_weights();

    set.records
        .par_iter()
        .zip(&set.entries)
        .fold(Counts::default, |mut acc, (record, entry)| {
            acc.observe(entry, record, &phase_w);
            acc
        })
        .reduce(Counts::default, Counts::merged)
}

/// Every nth position, for the sets whose features do not fit in memory whole.
///
/// A stride rather than a draw: it is reproducible without carrying a seed, and it spreads the
/// sample evenly over a file whose halves differ measurably in how predictable they are. Games sit
/// contiguously, so it also lands on many games rather than on a few of them entire.
fn strided(entries: Vec<SoulEntry>, cap: usize) -> Vec<SoulEntry> {
    if cap == 0 || entries.len() <= cap {
        return entries;
    }

    let stride = entries.len().div_ceil(cap);
    let sampled: Vec<SoulEntry> = entries.iter().copied().step_by(stride).collect();

    println!(
        "  Sampling {} of {} positions, every {stride}",
        format_comma(sampled.len() as u64),
        format_comma(entries.len() as u64),
    );

    sampled
}

/// A row's name: the file without its format suffixes, and how many more files came with it.
///
/// A name may carry dots of its own, so the suffixes come off one at a time.
fn label_for(paths: &[String]) -> String {
    const SUFFIXES: [&str; 7] = [".zst", ".soul", ".vf", ".viri", ".epd", ".txt", ".json"];

    let stem = |path: &String| {
        let mut name = Path::new(path).file_name().unwrap_or_default().to_string_lossy().into_owned();

        while let Some(shorter) = SUFFIXES.iter().find_map(|suffix| name.strip_suffix(suffix)) {
            name = shorter.to_string();
        }

        name
    };

    match paths {
        [] => "unnamed".to_string(),
        [one] => stem(one),
        [first, rest @ ..] => format!("{} +{}", stem(first), rest.len()),
    }
}

fn strings(items: &[&str]) -> Vec<String> {
    items.iter().map(|s| (*s).to_string()).collect()
}

/// Everything one pass over the positions counts.
#[derive(Default)]
struct Counts {
    results: [usize; 3],
    scored: usize,
    score_sum: f64,
    contradictions: usize,
    phase_sum: f64,
    phase_low: usize,
    phase_high: usize,
    pieces: u64,
    unequal: [usize; 5],
    unequal_phase: [f64; 5],
}

impl Counts {
    fn observe(&mut self, entry: &SoulEntry, record: &FeatureRecord, phase_w: &[f64; 6]) {
        self.results[usize::from(entry.result.min(2))] += 1;

        if entry.score != SoulEntry::NO_SCORE {
            self.scored += 1;
            self.score_sum += f64::from(entry.score.abs());

            // Both are STM-relative, so a winner the search scored below zero is a position whose
            // label and whose eval disagree about who stands better.
            let winner = i32::from(entry.result) - 1;

            if winner != 0 && i32::from(entry.score) * winner < 0 {
                self.contradictions += 1;
            }
        }

        let phase = f64::from(phase_of(record, phase_w) as u32);

        self.phase_sum += phase;
        self.pieces += u64::from(entry.occupancy.count_ones());

        if phase < 12.0 {
            self.phase_low += 1;
        } else if phase >= 20.0 {
            self.phase_high += 1;
        }

        for pt in 0..5 {
            if record.mat_diffs[pt] != 0 {
                self.unequal[pt] += 1;
                self.unequal_phase[pt] += phase;
            }
        }
    }

    fn merged(mut self, other: Self) -> Self {
        for i in 0..3 {
            self.results[i] += other.results[i];
        }

        for i in 0..5 {
            self.unequal[i] += other.unequal[i];
            self.unequal_phase[i] += other.unequal_phase[i];
        }

        self.scored += other.scored;
        self.score_sum += other.score_sum;
        self.contradictions += other.contradictions;
        self.phase_sum += other.phase_sum;
        self.phase_low += other.phase_low;
        self.phase_high += other.phase_high;
        self.pieces += other.pieces;

        self
    }
}

/// Right-aligned columns under their headers, one row per dataset or candidate.
struct Table {
    corner: String,
    columns: Vec<String>,
    rows: Vec<Row>,
    /// Column the gap widens before, for a table read in two halves.
    split: Option<usize>,
}

struct Row {
    label: String,
    cells: Vec<String>,
    /// Context rather than a result: the reference the other rows are read against.
    reference: bool,
}

impl Table {
    const GAP: usize = 3;

    fn new(corner: &str, columns: Vec<String>) -> Self {
        Self { corner: corner.to_string(), columns, rows: Vec::new(), split: None }
    }

    fn push(&mut self, label: &str, cells: Vec<String>) {
        self.rows.push(Row { label: label.to_string(), cells, reference: false });
    }

    fn push_dim(&mut self, label: &str, cells: Vec<String>) {
        self.rows.push(Row { label: label.to_string(), cells, reference: true });
    }

    fn print(&self) {
        let label_width = self
            .rows
            .iter()
            .map(|row| row.label.chars().count())
            .chain([self.corner.chars().count()])
            .max()
            .unwrap_or_default();

        let widths: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, head)| {
                self.rows
                    .iter()
                    .filter_map(|row| row.cells.get(i))
                    .map(|cell| cell.chars().count())
                    .chain([head.chars().count()])
                    .max()
                    .unwrap_or_default()
            })
            .collect();

        let mut out = format!("  {LAB}{:<label_width$}", self.corner);

        for (i, head) in self.columns.iter().enumerate() {
            out.push_str(&self.gap_before(i));
            out.push_str(&format!("{head:>width$}", width = widths[i]));
        }

        out.push_str(RESET);

        for row in &self.rows {
            let pen = if row.reference { DIM } else { VAL };
            let name = if row.reference { DIM } else { LAB };

            out.push_str(&format!("\n  {name}{:<label_width$}{RESET}", row.label));

            for (i, cell) in row.cells.iter().enumerate() {
                out.push_str(&self.gap_before(i));
                out.push_str(&format!("{pen}{cell:>width$}{RESET}", width = widths[i]));
            }
        }

        println!("{out}");
    }

    fn gap_before(&self, column: usize) -> String {
        " ".repeat(if self.split == Some(column) { Self::GAP * 2 } else { Self::GAP })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A dataset generated from known piece values, so the fit has a right answer to find.
    ///
    /// Targets are the model's own probabilities rather than sampled outcomes: the log-loss
    /// minimizer of soft targets is exactly the vector that produced them, so any gap the fit
    /// leaves is the solver's and not the sample's.
    fn planted(weights: &[f64; 10]) -> Vec<([f64; 10], f64)> {
        let mut rows = Vec::new();

        for phase in 0..=24 {
            for piece in 0..5 {
                for diff in [-2.0, -1.0, 1.0, 2.0] {
                    let mg = f64::from(phase) / f64::from(TOTAL_PHASE);
                    let mut design = [0.0; 10];

                    design[piece] = diff * mg;
                    design[piece + 5] = diff * (1.0 - mg);

                    rows.push((design, sigmoid(dot(&design, weights), FIT_K)));
                }
            }
        }

        rows
    }

    #[test]
    fn the_fit_recovers_the_values_it_was_generated_from() {
        let planted_weights = [100.0, 370.0, 382.0, 471.0, 752.0, 162.0, 513.0, 550.0, 970.0, 1897.0];
        let (fitted, _, steps) = newton_fit(&planted(&planted_weights));

        assert!(steps < FIT_STEPS, "the fit ran out of Newton steps");

        for (fitted, planted) in fitted.iter().zip(&planted_weights) {
            assert!((fitted - planted).abs() < 1.0, "fitted {fitted:.1} against a planted {planted:.1}");
        }
    }

    /// The piece a dataset never varies is the one whose column is empty, and an empty column is
    /// a singular matrix. It has to come back as no opinion rather than as a crash or a wild value.
    #[test]
    fn a_piece_that_never_varies_gets_no_opinion() {
        let mut rows = planted(&[100.0, 370.0, 382.0, 471.0, 752.0, 162.0, 513.0, 550.0, 970.0, 1897.0]);

        rows.retain(|(design, _)| design[4] == 0.0 && design[9] == 0.0);

        let (fitted, ..) = newton_fit(&rows);

        assert!(fitted[4].abs() < 1e-6, "a queen nothing constrains came out at {}", fitted[4]);
        assert!(fitted[9].abs() < 1e-6, "a queen nothing constrains came out at {}", fitted[9]);
        assert!((fitted[0] - 100.0).abs() < 1.0, "the pawn moved when the queen dropped out");
    }

    #[test]
    fn a_sample_takes_the_whole_span_and_stops_at_the_cap() {
        let entries: Vec<SoulEntry> = (0..100).map(|i| SoulEntry { occupancy: i, ..SoulEntry::default() }).collect();

        let sampled = strided(entries.clone(), 10);

        assert_eq!(sampled.len(), 10);
        assert_eq!(sampled.first().map(|e| e.occupancy), Some(0));
        assert_eq!(sampled.last().map(|e| e.occupancy), Some(90));
        assert_eq!(strided(entries.clone(), 100).len(), 100, "a cap at the count keeps every entry");
        assert_eq!(strided(entries, 0).len(), 100, "a cap of zero is no cap");
    }
}
