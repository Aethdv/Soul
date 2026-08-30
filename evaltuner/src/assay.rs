//! What a dataset says before anything trains on it.
//!
//! Assays read a set in its natural order, with no resampling and no holdout: they describe what
//! is there rather than fitting to part of it.

use std::path::Path;

use rayon::prelude::*;

use crate::{
    alarm,
    config::{EvalTuneConfig, LossFn},
    engine::{
        Color, DECISIVE_ENDING, FeatureRecord, GameScan, LAYOUT, OPEN_UNITY, PieceType, QUIET_ENDING, SoulEntry, TOTAL_PHASE,
        collect_parameters, default_values, eval_record, format_comma, pct, piece_mobility_counts, scan_viri_games, sigmoid,
    },
    loader::{load_datasets, resolve_dataset_paths},
    palette::{ALARM, COUNT, DIM, LAB, RESET, VAL},
    report::PIECE_NAMES,
    run::replay_filter,
    scale::golden_search_k,
    storage::load_checkpoint,
    training::{phase_of, phase_weights},
};

/// Binary cross-entropy loss for a uniform `p = 0.5` prediction (`ln(2)` nats).
const COIN_FLIP: f64 = std::f64::consts::LN_2;

/// The bracket the score assay searches for K, wider than a training run's.
///
/// A cold-start vector is logged before the gauge normalizes it and can sit a factor of four off
/// the centipawn scale. Cutting the bracket to what a shipped-scale eval needs would clamp such a
/// candidate at the edge and report a loss about its units rather than its opinions.
const SCORE_K: (f64, f64) = (1e-5, 0.2);

/// K for the material fit, held fixed.
///
/// K and the weights are collinear, pinning K leaves the weights
/// to be read against `P_mg = 100` afterwards.
const FIT_K: f64 = 0.0025;
const FIT_STEPS: usize = 24;
const FIT_TOL: f64 = 1e-10;

const FITTED: usize = 5;

/// Algebraic letters for those five, read off the engine's own table.
const SYMBOLS: [char; FITTED] = [
    PieceType::Pawn.to_char(Color::White),
    PieceType::Knight.to_char(Color::White),
    PieceType::Bishop.to_char(Color::White),
    PieceType::Rook.to_char(Color::White),
    PieceType::Queen.to_char(Color::White),
];

/// Diagnostic report to execute.
pub enum Assay {
    /// Loss of each parameter vector on each set, with K refit per pair so scale cannot flatter one.
    Score { params: Vec<String>, loss: LossFn, shipped: String },
    /// Ten tapered material coefficients fitted to outcomes alone, which shows a skewed material
    /// prior in one pass instead of thousands of tuning epochs.
    Material { shipped: String },
    /// Labels, phase, material imbalances, and how the games ended.
    Profile,
    /// The distributions that decide how wide a table should be: per-class mobility
    /// counts, openness, and the phase a rank-7 passer appears at.
    Spread,
}

/// Comma-joined paths load as one mixture and report as one row, not as several sets.
pub fn run(report: &Assay, datasets: &[String], config: &EvalTuneConfig, sample: Option<usize>) {
    let filter = replay_filter(config);
    let mut sets = Vec::new();

    for dataset in datasets {
        let Some(paths) = resolve_dataset_paths(dataset) else {
            continue;
        };

        let (mut entries, ..) = load_datasets(&paths, &filter);
        if entries.is_empty() {
            alarm!("No positions loaded from {dataset}.");
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
        alarm!("Nothing to assay.");
        return;
    }

    match report {
        Assay::Score { params, loss, shipped } => score(&sets, params, *loss, shipped),
        Assay::Material { shipped } => material(&sets, shipped),
        Assay::Profile => profile(&sets),
        Assay::Spread => spread(&sets),
    }
}

/// In-memory dataset and its extracted evaluation features.
struct Set {
    label: String,
    paths: Vec<String>,
    entries: Vec<SoulEntry>,
    records: Vec<FeatureRecord>,
    /// Total entries parsed from disk prior to subsampling.
    loaded: usize,
}

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

/// Plaintext terminal table with right-aligned numeric columns.
struct Table {
    corner: String,
    columns: Vec<String>,
    rows: Vec<Row>,
    /// Column index before which an extra visual separator is inserted.
    split: Option<usize>,
}

struct Row {
    label: String,
    cells: Vec<String>,
    /// Reference row rendered with dimmed styling.
    reference: bool,
}

impl Counts {
    fn observe(&mut self, entry: &SoulEntry, record: &FeatureRecord, phase_w: &[f64; 6]) {
        self.results[usize::from(entry.result.min(2))] += 1;

        if entry.score != SoulEntry::NO_SCORE {
            self.scored += 1;
            self.score_sum += f64::from(entry.score.abs());

            // Side-to-move relative: tracks positions where search score sign opposes the final outcome.
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
        let gap = if self.split == Some(column) { Self::GAP * 2 } else { Self::GAP };
        " ".repeat(gap)
    }
}

/// Evaluates cross-entropy or MSE of parameter vectors, refitting K per dataset.
///
/// Optimizing K per cell decouples relative ordering accuracy from arbitrary centipawn
/// scale differences across checkpoints. Evaluated solely against game outcomes to avoid
/// scoring candidates against search engine evaluation artifacts.
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

/// Fits 10 tapered material coefficients (5 middlegame, 5 endgame) directly to game results,
/// isolating base piece values from every positional term.
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

fn profile(sets: &[Set]) {
    let stats: Vec<Counts> = sets.iter().map(count).collect();

    let mut labels = Table::new("dataset", strings(&["positions", "win", "draw", "loss", "STM", "score≠result"]));
    let mut design = Table::new("dataset", strings(&["phase", "under 12", "20 or more", "pieces", "scored", "|score|"]));
    let pieces = || PIECE_NAMES[..FITTED].iter().map(|p| p.to_lowercase());
    let mut imbalance = Table::new("dataset", pieces().map(|p| format!("{p}≠")).collect());
    let mut at_phase = Table::new("dataset", pieces().collect());

    for (set, counts) in sets.iter().zip(&stats) {
        let n = set.entries.len() as u64;
        let share = |count: usize| format!("{:.2}%", pct(count as u64, n));

        labels.push(&set.label, vec![
            format_comma(n),
            share(counts.results[2]),
            share(counts.results[1]),
            share(counts.results[0]),
            format!("{:.2}%", 100.0 * (counts.results[2] as f64 + 0.5 * counts.results[1] as f64) / n as f64),
            share(counts.contradictions),
        ]);

        design.push(&set.label, vec![
            format!("{:.2}", counts.phase_sum / n as f64),
            share(counts.phase_low),
            share(counts.phase_high),
            format!("{:.1}", counts.pieces as f64 / n as f64),
            share(counts.scored),
            match counts.scored {
                0 => "-".to_string(),
                scored => format!("{:.0}", counts.score_sum / scored as f64),
            },
        ]);

        imbalance.push(&set.label, (0..5).map(|pt| share(counts.unequal[pt])).collect());
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
    println!("\n{LAB}Imbalance: the share of positions where the count differs{RESET}");
    imbalance.print();
    println!("\n{LAB}And the mean phase where it does{RESET}");
    at_phase.print();
    print_games(sets);
}

fn print_games(sets: &[Set]) {
    let scans: Vec<(&Set, GameScan)> = sets
        .iter()
        .filter_map(|set| {
            let path = set.paths.iter().find(|p| p.ends_with(".vf") || p.ends_with(".viri"))?;
            match scan_viri_games(path) {
                Ok(scan) if scan.games > 0 => Some((set, scan)),
                Ok(_) => None,
                Err(e) => {
                    alarm!("Cannot scan games in {path}: {e}");
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
        let n = scan.games as u64;
        let share = |count: usize| format!("{:.1}%", pct(count as u64, n));
        games.push(&set.label, vec![
            format_comma(n),
            format!("{:.1}", scan.plies as f64 / n as f64),
            format!("{:.1}", set.loaded as f64 / n as f64),
            format!("{:.1}", scan.pieces_left as f64 / n as f64),
            share(scan.results[1]),
            share(scan.mate_endings),
            share(scan.decisive_endings),
            share(scan.quiet_endings),
        ]);
    }
    println!("\n{LAB}Games, unfiltered: the replay's independent unit{RESET}");
    games.print();
    println!("{DIM}  The last three are what the final score said; a game in none of them stopped for");
    println!("  something other than its score.{RESET}");
}

/// Finds the optimal scale K and minimal loss for a parameter vector on a dataset.
///
/// Raw evaluations are computed once in parallel; golden-section search then optimizes K
/// over cached predictions.
fn best_loss(set: &Set, values: &[f64], loss: LossFn) -> (f64, f64) {
    let eval_targets: Vec<(f64, f64)> = set
        .records
        .par_iter()
        .zip(&set.entries)
        .map(|(record, entry)| (eval_record(record, values), f64::from(entry.result) / 2.0))
        .collect();

    let loss_at = |k: f64| {
        let sum: f64 = eval_targets.par_iter().map(|&(s, y)| loss.loss(sigmoid(s, k), y)).sum();
        sum / eval_targets.len() as f64
    };

    let (k_min, k_max) = SCORE_K;
    let k = golden_search_k(k_min, k_max, 1e-6 * (k_max - k_min), loss_at);
    if k <= k_min * 1.01 || k >= k_max * 0.99 {
        eprintln!("{ALARM}[!] K converged to boundary [{k_min}, {k_max}] on {}: loss may not be optimal.{RESET}", set.label);
    }
    (k, loss_at(k))
}

/// Constructs design matrix rows and target outcomes for tapered material regression.
///
/// Feature layout: `[d_0·phi, ..., d_4·phi, d_0·(1-phi), ..., d_4·(1-phi)]`,
/// where `d_i` is the piece differential and `phi` in `[0.0, 1.0]` is game phase.
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

/// Newton-Raphson with a backtracking line search, on exact analytic gradients and Hessians under
/// the canonical logit link. Returns `(weights, cross_entropy_loss, iterations)`.
fn newton_fit(rows: &[([f64; 10], f64)]) -> ([f64; 10], f64, usize) {
    let mut w = [0.0f64; 10];
    let mut loss = fit_loss(rows, &w);

    for step in 1..=FIT_STEPS {
        let (gradient, hessian) = fit_derivatives(rows, &w);

        let Some(delta) = solve(hessian, gradient) else {
            return (w, loss, step);
        };

        // Backtracking line search prevents quadratic overshoot near separable boundaries.
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

        let step_norm: f64 = delta.iter().map(|d| d * d).sum::<f64>().sqrt() * scale;
        if !moved || step_norm < FIT_TOL {
            return (w, loss, step);
        }
    }
    (w, loss, FIT_STEPS)
}

/// Computes the empirical gradient and Hessian of the cross-entropy loss at `w`.
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

    // Ridge regularization (1e-12) guarantees invertibility when specific piece imbalances are unobserved.
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

#[inline(always)]
fn dot(x: &[f64; 10], w: &[f64; 10]) -> f64 { (0..10).map(|i| x[i] * w[i]).sum() }

/// Solves a 10x10 linear system `Ax = b` via Gaussian elimination with partial pivoting.
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

/// Pawns are in the list because dropping them from the sum is its own experiment,
/// and their share is what that would cost.
const MOBILITY_CLASSES: [PieceType; 5] = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen];

/// A queen on an open board reaches 27 squares, and the area only ever removes some.
const MOBILITY_MAX: usize = 28;
const OPENNESS_BINS: usize = 8;
const PHASE_BINS: usize = 6;

/// Upper bound of each bin, inclusive.
const SHAPE_BINS: [u32; 7] = [0, 2, 4, 6, 8, 12, 16];

#[derive(Clone)]
struct Spread {
    /// Pieces observed at each count, per class.
    mobility: [[u64; MOBILITY_MAX]; MOBILITY_CLASSES.len()],
    openness: [u64; OPENNESS_BINS],
    openness_floor: u64,
    openness_ceiling: u64,
    openness_sum: f64,
    phase_all: [u64; PHASE_BINS],
    phase_rank7: [u64; PHASE_BINS],
    phase_all_sum: f64,
    phase_rank7_sum: f64,
    positions: u64,
    rank7_positions: u64,
}

impl Default for Spread {
    fn default() -> Self {
        Self {
            mobility: [[0; MOBILITY_MAX]; MOBILITY_CLASSES.len()],
            openness: [0; OPENNESS_BINS],
            openness_floor: 0,
            openness_ceiling: 0,
            openness_sum: 0.0,
            phase_all: [0; PHASE_BINS],
            phase_rank7: [0; PHASE_BINS],
            phase_all_sum: 0.0,
            phase_rank7_sum: 0.0,
            positions: 0,
            rank7_positions: 0,
        }
    }
}

impl Spread {
    fn observe(&mut self, entry: &SoulEntry, record: &FeatureRecord, phase_w: &[f64; 6]) {
        self.positions += 1;

        let board = entry.to_board();
        for color in [Color::White, Color::Black] {
            piece_mobility_counts(&board, color, |piece, count| {
                if let Some(row) = MOBILITY_CLASSES.iter().position(|c| *c == piece) {
                    self.mobility[row][(count as usize).min(MOBILITY_MAX - 1)] += 1;
                }
            });
        }

        let openness = record.open_raw.clamp(0, OPEN_UNITY);
        self.openness_sum += f64::from(openness);
        if openness == 0 {
            self.openness_floor += 1;
        } else if openness == OPEN_UNITY {
            self.openness_ceiling += 1;
        }
        let bin = (openness as usize * OPENNESS_BINS / (OPEN_UNITY as usize + 1)).min(OPENNESS_BINS - 1);
        self.openness[bin] += 1;

        let phase = phase_of(record, phase_w);
        let phase_bin = (phase * PHASE_BINS / (TOTAL_PHASE as usize + 1)).min(PHASE_BINS - 1);
        self.phase_all[phase_bin] += 1;
        self.phase_all_sum += phase as f64;

        // The rank-7 entry is a differential, so nonzero is exactly when PASSED_PAWN[5] takes
        // gradient. Whether that set concentrates in phase decides if its MG and EG halves
        // are one parameter or two.
        if record.passed_pawn[5] != 0 {
            self.rank7_positions += 1;
            self.phase_rank7[phase_bin] += 1;
            self.phase_rank7_sum += phase as f64;
        }
    }

    fn merged(mut self, other: Self) -> Self {
        for (row, add) in self.mobility.iter_mut().zip(&other.mobility) {
            for (cell, plus) in row.iter_mut().zip(add) {
                *cell += plus;
            }
        }
        for (bin, plus) in self.openness.iter_mut().zip(&other.openness) {
            *bin += plus;
        }
        for (bin, plus) in self.phase_all.iter_mut().zip(&other.phase_all) {
            *bin += plus;
        }
        for (bin, plus) in self.phase_rank7.iter_mut().zip(&other.phase_rank7) {
            *bin += plus;
        }
        self.openness_floor += other.openness_floor;
        self.openness_ceiling += other.openness_ceiling;
        self.openness_sum += other.openness_sum;
        self.phase_all_sum += other.phase_all_sum;
        self.phase_rank7_sum += other.phase_rank7_sum;
        self.positions += other.positions;
        self.rank7_positions += other.rank7_positions;
        self
    }

    /// The smallest count holding at least `fraction` of a class's pieces, which is the
    /// table width that would lose no more than the remainder off its top.
    fn cutoff(&self, row: usize, fraction: f64) -> usize {
        let total: u64 = self.mobility[row].iter().sum();
        if total == 0 {
            return 0;
        }
        let target = fraction * total as f64;
        let mut seen = 0u64;
        for (count, pieces) in self.mobility[row].iter().enumerate() {
            seen += pieces;
            if seen as f64 >= target {
                return count;
            }
        }
        MOBILITY_MAX - 1
    }
}

/// Distributions behind the table-sizing decisions.
fn spread(sets: &[Set]) {
    for set in sets {
        let stats = gather_spread(set);
        println!();

        let mut summary = Table::new(&set.label, strings(&["pieces", "mean", "max", "p99", "p99.9", "at 0"]));
        let mut shape = Table::new("share at count", shape_headings());

        for (row, piece) in MOBILITY_CLASSES.iter().enumerate() {
            let counts = &stats.mobility[row];
            let pieces: u64 = counts.iter().sum();
            if pieces == 0 {
                continue;
            }

            let weighted: u64 = counts.iter().enumerate().map(|(count, n)| count as u64 * n).sum();
            let max = counts.iter().rposition(|n| *n > 0).unwrap_or(0);
            let label = piece.to_char(Color::White).to_lowercase().to_string();

            summary.push(&label, vec![
                format_comma(pieces),
                format!("{:.2}", weighted as f64 / pieces as f64),
                max.to_string(),
                stats.cutoff(row, 0.99).to_string(),
                stats.cutoff(row, 0.999).to_string(),
                format!("{:.2}%", pct(counts[0], pieces)),
            ]);

            let mut cells = Vec::new();
            let mut low = 0u32;
            for high in SHAPE_BINS.into_iter().skip(1) {
                let slice: u64 = counts[low as usize..=(high as usize).min(MOBILITY_MAX - 1)].iter().sum();
                cells.push(format!("{:.1}%", pct(slice, pieces)));
                low = high + 1;
            }
            let tail: u64 = counts[(low as usize).min(MOBILITY_MAX - 1)..].iter().sum();
            cells.push(format!("{:.1}%", pct(tail, pieces)));
            shape.push(&label, cells);
        }

        summary.print();
        println!();
        shape.print();
        println!();

        let n = stats.positions;
        let mut openness = Table::new("openness", openness_headings());
        let mut cells: Vec<String> = stats.openness.iter().map(|bin| format!("{:.1}%", pct(*bin, n))).collect();
        cells.push(format!("{:.2}%", pct(stats.openness_floor, n)));
        cells.push(format!("{:.2}%", pct(stats.openness_ceiling, n)));
        cells.push(format!("{:.0}", stats.openness_sum / n as f64));
        openness.push(&set.label, cells);
        openness.print();
        println!();

        let mut phase = Table::new("phase", phase_headings());
        let row = |bins: &[u64; PHASE_BINS], total: u64, sum: f64| {
            let mut cells: Vec<String> = bins.iter().map(|bin| format!("{:.1}%", pct(*bin, total.max(1)))).collect();
            cells.push(format_comma(total));
            cells.push(format!("{:.2}", sum / total.max(1) as f64));
            cells
        };
        phase.push_dim("every position", row(&stats.phase_all, n, stats.phase_all_sum));
        phase.push("rank-7 passer", row(&stats.phase_rank7, stats.rank7_positions, stats.phase_rank7_sum));
        phase.print();
    }
}

fn shape_headings() -> Vec<String> {
    let mut heads = Vec::new();
    let mut low = 0u32;
    for high in SHAPE_BINS.into_iter().skip(1) {
        heads.push(if low == high { low.to_string() } else { format!("{low}-{high}") });
        low = high + 1;
    }
    heads.push(format!("{low}+"));
    heads
}

fn openness_headings() -> Vec<String> {
    let step = (OPEN_UNITY as usize + 1) / OPENNESS_BINS;
    let mut heads: Vec<String> = (0..OPENNESS_BINS).map(|bin| format!("{}", bin * step)).collect();
    heads.extend(strings(&["closed", "open", "mean"]));
    heads
}

fn phase_headings() -> Vec<String> {
    let step = (TOTAL_PHASE as usize + 1) / PHASE_BINS;
    let mut heads: Vec<String> = (0..PHASE_BINS).map(|bin| format!("{}", bin * step)).collect();
    heads.extend(strings(&["positions", "mean"]));
    heads
}

fn gather_spread(set: &Set) -> Spread {
    let phase_w = phase_weights();
    set.records
        .par_iter()
        .zip(&set.entries)
        .fold(Spread::default, |mut acc, (record, entry)| {
            acc.observe(entry, record, &phase_w);
            acc
        })
        .reduce(Spread::default, Spread::merged)
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

/// Subsamples entries by selecting every k-th position.
///
/// Uniform striding preserves the natural opening-to-endgame progression across games
/// without requiring PRNG state.
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

/// Generates a display label by stripping compound file extensions from the base dataset path.
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

fn strings(items: &[&str]) -> Vec<String> { items.iter().map(|s| (*s).to_string()).collect() }

#[cfg(test)]
mod tests {
    use super::*;

    /// Generates a synthetic dataset from ground-truth piece values.
    ///
    /// Evaluates exact sigmoid probabilities as targets so zero training error corresponds
    /// strictly to the planted weights.
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
