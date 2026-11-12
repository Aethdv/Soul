//! What a dataset says before anything trains on it.
//!
//! Assays read a set in its natural order, with no resampling and no holdout.

use std::{f64::consts::LN_2, path::Path};

use rayon::prelude::*;

use crate::{
    alarm,
    config::{EvalTuneConfig, LossFn},
    engine::{
        Color, FeatureRecord, GameScan, LAYOUT, MAX_QUIET_ENDING, MIN_DECISIVE_ENDING, OPEN_UNITY, PieceType, SoulEntry,
        TOTAL_PHASE, collect_parameters, default_values, eval_record, fmt_comma, pct, piece_mobility_counts, scan_viri_games,
        sigmoid,
    },
    loader::{is_viri_path, load_datasets, resolve_dataset_paths},
    report::{COUNT, DIM, LAB, PIECE_NAMES, RESET, VAL},
    run::replay_filter,
    scale::golden_search_k,
    storage::load_checkpoint,
    training::{add_slice, l2_norm, phase_of, piece_phase_weights},
};

const COIN_FLIP: f64 = LN_2;

// Wider than a training run's K bracket: a cold-start checkpoint logged before gauge
// normalization can sit 4× off centipawn scale and would otherwise clamp at the bracket edge.
const SCORE_K: (f64, f64) = (1e-5, 0.2);

// Held fixed during material fitting because K and the weights are collinear.
// Results normalize against P_mg = 100 afterwards.
const FIT_K: f64 = 0.0025;
const FIT_STEPS: usize = 24;
const FIT_TOL: f64 = 1e-10;

// TODO: Pawns are included because dropping them from the mobility sum is its own experiment.
// Got attempted but to no avail.
const PIECES: [PieceType; 5] = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen];
const MOBILITY_MAX: usize = 28;
const OPENNESS_BINS: usize = 8;
const PHASE_BINS: usize = 6;
const SHAPE_HIGHS: [usize; 6] = [2, 4, 6, 8, 12, 16];

pub(crate) enum Assay {
    Score { params: Vec<String>, loss: LossFn, shipped: String },
    Material { shipped: String },
    Profile,
    Spread,
}

/// Comma-joined paths load as one mixture and report as one row.
pub(crate) fn run(report: &Assay, datasets: &[String], config: &EvalTuneConfig, sample: Option<usize>) {
    let filter = replay_filter(config);
    let mut sets = Vec::new();

    for dataset in datasets {
        let Some(paths) = resolve_dataset_paths(dataset) else { continue };
        let mut entries = load_datasets(&paths, &filter).entries;

        if entries.is_empty() {
            alarm!("No positions loaded from {dataset}.");
            continue;
        }

        let loaded = entries.len();

        if let Some(cap) = sample {
            entries = strided(entries, cap);
        }

        println!("  Extracting features ({} entries)...", fmt_comma(entries.len() as u64));
        let records = entries.par_iter().map(FeatureRecord::from_entry).collect();
        sets.push(Dataset { label: label_for(&paths), paths, entries, records, loaded });
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

struct Dataset {
    label:   String,
    paths:   Vec<String>,
    entries: Vec<SoulEntry>,
    records: Vec<FeatureRecord>,
    loaded:  usize,
}

#[derive(Default)]
struct DatasetProfile {
    results:        [usize; 3],
    scored:         usize,
    score_sum:      f64,
    contradictions: usize,
    phase_sum:      f64,
    phase_low:      usize,
    phase_high:     usize,
    pieces:         u64,
    unequal:        [usize; PIECES.len()],
    unequal_phase:  [f64; PIECES.len()],
}

#[derive(Clone, Copy, Default)]
struct PhaseHist {
    bins: [u64; PHASE_BINS],
    sum:  f64,
    n:    u64,
}

impl PhaseHist {
    fn observe(&mut self, bin: usize, phase: usize) {
        self.bins[bin] += 1;
        self.sum += phase as f64;
        self.n += 1;
    }

    fn merge(&mut self, other: &Self) {
        add_slice(&mut self.bins, &other.bins);
        self.sum += other.sum;
        self.n += other.n;
    }

    fn cells(&self) -> Vec<String> {
        let denom = self.n.max(1);
        let mut out: Vec<String> = self.bins.iter().map(|&b| format!("{:.1}%", pct(b, denom))).collect();
        out.push(fmt_comma(self.n));
        out.push(format!("{:.2}", self.sum / denom as f64));
        out
    }
}

#[derive(Clone, Default)]
struct FeatureSpread {
    mobility:         [[u64; MOBILITY_MAX]; PIECES.len()],
    openness:         [u64; OPENNESS_BINS],
    openness_floor:   u64,
    openness_ceiling: u64,
    openness_sum:     f64,
    phase_all:        PhaseHist,
    phase_rank7:      PhaseHist,
}

struct Table {
    corner:  String,
    columns: Vec<String>,
    rows:    Vec<(String, Vec<String>, bool)>,
    split:   Option<usize>,
}

impl DatasetProfile {
    fn observe(&mut self, entry: &SoulEntry, record: &FeatureRecord, phase_w: &[f64; 6]) {
        self.results[usize::from(entry.result.min(2))] += 1;
        if entry.score != SoulEntry::NO_SCORE {
            self.scored += 1;
            self.score_sum += f64::from(entry.score.abs());
            let winner = i32::from(entry.result) - 1;
            self.contradictions += usize::from(winner != 0 && i32::from(entry.score) * winner < 0);
        }

        let phase = f64::from(phase_of(record, phase_w) as u32);
        self.phase_sum += phase;
        self.pieces += u64::from(entry.occupancy.count_ones());
        self.phase_low += usize::from(phase < 12.0);
        self.phase_high += usize::from(phase >= 20.0);

        for pt in 0..PIECES.len() {
            if record.material_diff[pt] != 0 {
                self.unequal[pt] += 1;
                self.unequal_phase[pt] += phase;
            }
        }
    }

    fn merged(mut self, o: Self) -> Self {
        add_slice(&mut self.results, &o.results);
        add_slice(&mut self.unequal, &o.unequal);
        add_slice(&mut self.unequal_phase, &o.unequal_phase);
        self.scored += o.scored;
        self.score_sum += o.score_sum;
        self.contradictions += o.contradictions;
        self.phase_sum += o.phase_sum;
        self.phase_low += o.phase_low;
        self.phase_high += o.phase_high;
        self.pieces += o.pieces;
        self
    }
}

impl FeatureSpread {
    fn observe(&mut self, entry: &SoulEntry, record: &FeatureRecord, phase_w: &[f64; 6]) {
        let board = entry.to_board();
        for color in [Color::White, Color::Black] {
            piece_mobility_counts(&board, color, |piece, count| {
                if let Some(row) = PIECES.iter().position(|&c| c == piece) {
                    self.mobility[row][(count as usize).min(MOBILITY_MAX - 1)] += 1;
                }
            });
        }

        let openness = record.openness.clamp(0, OPEN_UNITY);
        self.openness_sum += f64::from(openness);
        self.openness_floor += u64::from(openness == 0);
        self.openness_ceiling += u64::from(openness == OPEN_UNITY);
        self.openness[(openness as usize * OPENNESS_BINS / (OPEN_UNITY as usize + 1)).min(OPENNESS_BINS - 1)] += 1;

        let phase = phase_of(record, phase_w);
        let phase_bin = (phase * PHASE_BINS / (TOTAL_PHASE as usize + 1)).min(PHASE_BINS - 1);
        self.phase_all.observe(phase_bin, phase);

        // Nonzero differential is where PASSED_PAWN[5] takes gradient; phase concentration decides MG/EG identifiability.
        if record.passed_pawn[5] != 0 {
            self.phase_rank7.observe(phase_bin, phase);
        }
    }

    fn merged(mut self, o: Self) -> Self {
        for (row, add) in self.mobility.iter_mut().zip(&o.mobility) {
            add_slice(row, add);
        }
        add_slice(&mut self.openness, &o.openness);
        self.openness_floor += o.openness_floor;
        self.openness_ceiling += o.openness_ceiling;
        self.openness_sum += o.openness_sum;
        self.phase_all.merge(&o.phase_all);
        self.phase_rank7.merge(&o.phase_rank7);
        self
    }

    fn cutoff(&self, row: usize, fraction: f64) -> usize {
        let total: u64 = self.mobility[row].iter().sum();
        if total == 0 {
            return 0;
        }

        let target = fraction * total as f64;
        let mut seen = 0u64;

        self.mobility[row]
            .iter()
            .position(|&pieces| {
                seen += pieces;
                seen as f64 >= target
            })
            .unwrap_or(MOBILITY_MAX - 1)
    }
}

impl Table {
    fn new(corner: &str, columns: impl IntoIterator<Item = impl Into<String>>) -> Self {
        Self {
            corner:  corner.into(),
            columns: columns.into_iter().map(Into::into).collect(),
            rows:    Vec::new(),
            split:   None,
        }
    }

    fn push(&mut self, label: &str, cells: Vec<String>) { self.rows.push((label.into(), cells, false)); }
    fn push_dim(&mut self, label: &str, cells: Vec<String>) { self.rows.push((label.into(), cells, true)); }

    fn print(&self) {
        let lw = self
            .rows
            .iter()
            .map(|(l, ..)| l.chars().count())
            .chain([self.corner.chars().count()])
            .max()
            .unwrap_or(0);
        let widths: Vec<usize> = self
            .columns
            .iter()
            .enumerate()
            .map(|(i, h)| {
                self.rows
                    .iter()
                    .filter_map(|(_, c, _)| c.get(i))
                    .map(|s| s.chars().count())
                    .chain([h.chars().count()])
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        let gap = |col: usize| if self.split == Some(col) { "      " } else { "   " };

        let mut out = format!("  {LAB}{:<lw$}", self.corner);
        for (i, head) in self.columns.iter().enumerate() {
            out.push_str(&format!("{}{head:>w$}", gap(i), w = widths[i]));
        }

        out.push_str(RESET);

        for (label, cells, dim) in &self.rows {
            let (name, pen) = if *dim { (DIM, DIM) } else { (LAB, VAL) };
            out.push_str(&format!("\n  {name}{label:<lw$}{RESET}"));
            for (i, cell) in cells.iter().enumerate() {
                out.push_str(&format!("{}{pen}{cell:>w$}{RESET}", gap(i), w = widths[i]));
            }
        }
        println!("{out}");
    }
}

fn fold_dataset<T: Default + Send>(
    set: &Dataset,
    observe: impl Fn(&mut T, &SoulEntry, &FeatureRecord, &[f64; 6]) + Sync,
    merge: impl Fn(T, T) -> T + Send + Sync,
) -> T {
    let phase_w = piece_phase_weights();
    set.records
        .par_iter()
        .zip(&set.entries)
        .fold(T::default, |mut acc, (record, entry)| {
            observe(&mut acc, entry, record, &phase_w);
            acc
        })
        .reduce(T::default, merge)
}

// Refits K per (candidate, dataset) cell against game outcomes alone.
fn score(sets: &[Dataset], params: &[String], loss: LossFn, shipped: &str) {
    let tunables = collect_parameters();
    let defaults = default_values(&tunables);
    let mut candidates = vec![(shipped.to_string(), defaults.clone())];

    for path in params {
        match load_checkpoint(path, &tunables, &defaults) {
            Ok(d) => {
                let v = if d.best_val_params.len() == tunables.len() { d.best_val_params } else { d.values };
                candidates.push((label_for(std::slice::from_ref(path)), v));
            },
            Err(e) => alarm!("Cannot read parameters from {path}: {e}"),
        }
    }

    let single = sets.len() == 1;
    let mut cols: Vec<String> = sets.iter().map(|s| s.label.clone()).collect();

    if single {
        cols.extend(["K".into(), "ln2 − L".into()]);
    }

    let mut table = Table::new("parameters", cols);
    for (name, values) in &candidates {
        let mut cells = Vec::new();
        let mut last = (0.0, 0.0);

        for set in sets {
            last = best_loss(set, values, loss);
            cells.push(format!("{:.6}", last.1));
        }

        if single {
            cells.extend([format!("{:.6}", last.0), format!("{:.6}", COIN_FLIP - last.1)]);
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

fn material(sets: &[Dataset], shipped: &str) {
    let syms = PIECES.map(|p| p.to_char(Color::White));
    let mut table = Table::new("dataset", syms.iter().map(|p| format!("{p}mg")).chain(syms.iter().map(|p| format!("{p}eg"))));
    let mut notes = Vec::new();
    table.split = Some(5);

    for set in sets {
        let (weights, loss, steps) = newton_fit(&material_rows(set));
        if weights[0].abs() < 1e-12 {
            alarm!("{}: the fit puts a midgame pawn at zero, so nothing normalizes.", set.label);
            continue;
        }

        let scale = 100.0 / weights[0];
        table.push(&set.label, (0..10).map(|i| format!("{:.0}", weights[i] * scale)).collect());
        notes.push((
            set.label.clone(),
            format!(
                "{COUNT}{}{RESET} positions, L_fit {VAL}{loss:.4}{RESET}, {steps} Newton steps",
                fmt_comma(set.entries.len() as u64)
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

    let width = notes.iter().map(|(l, _)| l.chars().count()).max().unwrap_or(0);
    for (label, note) in notes {
        println!("  {LAB}{label:<width$}{RESET}  {note}");
    }
    println!("{DIM}  L_fit is each set's loss on its own positions and does not compare across sets.{RESET}");
}

fn profile(sets: &[Dataset]) {
    let stats: Vec<DatasetProfile> = sets
        .iter()
        .map(|s| fold_dataset(s, DatasetProfile::observe, DatasetProfile::merged))
        .collect();

    let mut labels = Table::new("dataset", ["positions", "win", "draw", "loss", "STM", "score≠result"]);
    let mut design = Table::new("dataset", ["phase", "under 12", "20 or more", "pieces", "scored", "|score|"]);

    let pieces = || PIECE_NAMES[..PIECES.len()].iter().map(|p| p.to_lowercase());

    let mut imbalance = Table::new("dataset", pieces().map(|p| format!("{p}≠")));
    let mut at_phase = Table::new("dataset", pieces());

    for (set, c) in sets.iter().zip(&stats) {
        let n = set.entries.len() as u64;
        let share = |count: usize| format!("{:.2}%", pct(count as u64, n));

        labels.push(&set.label, vec![
            fmt_comma(n),
            share(c.results[2]),
            share(c.results[1]),
            share(c.results[0]),
            format!("{:.2}%", 100.0 * (c.results[2] as f64 + 0.5 * c.results[1] as f64) / n as f64),
            share(c.contradictions),
        ]);
        design.push(&set.label, vec![
            format!("{:.2}", c.phase_sum / n as f64),
            share(c.phase_low),
            share(c.phase_high),
            format!("{:.1}", c.pieces as f64 / n as f64),
            share(c.scored),
            if c.scored == 0 { "-".into() } else { format!("{:.0}", c.score_sum / c.scored as f64) },
        ]);
        imbalance.push(&set.label, (0..5).map(|pt| share(c.unequal[pt])).collect());
        at_phase.push(
            &set.label,
            (0..5)
                .map(
                    |pt| if c.unequal[pt] == 0 { "-".into() } else { format!("{:.1}", c.unequal_phase[pt] / c.unequal[pt] as f64) },
                )
                .collect(),
        );
    }

    for (title, tbl) in [
        ("Labels", &labels),
        ("Positions", &design),
        ("Imbalance: the share of positions where the count differs", &imbalance),
        ("And the mean phase where it does", &at_phase),
    ] {
        println!("\n{LAB}{title}{RESET}");
        tbl.print();
    }
    print_games(sets);
}

fn print_games(sets: &[Dataset]) {
    let scans: Vec<(&Dataset, GameScan)> = sets
        .iter()
        .filter_map(|set| {
            let path = set.paths.iter().find(|p| is_viri_path(p))?;
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

    let mut cols: Vec<String> = ["games", "plies", "kept/game", "pieces left", "draws", "mate"]
        .into_iter()
        .map(Into::into)
        .collect();
    cols.extend([format!("past {MIN_DECISIVE_ENDING}"), format!("inside {MAX_QUIET_ENDING}")]);

    let mut games = Table::new("dataset", cols);
    for (set, scan) in &scans {
        let n = scan.games as u64;
        let share = |count: usize| format!("{:.1}%", pct(count as u64, n));
        games.push(&set.label, vec![
            fmt_comma(n),
            format!("{:.1}", scan.plies as f64 / n as f64),
            format!("{:.1}", set.loaded as f64 / n as f64),
            format!("{:.1}", scan.pieces_left_sum as f64 / n as f64),
            share(scan.draws),
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

fn best_loss(set: &Dataset, values: &[f64], loss: LossFn) -> (f64, f64) {
    let pairs: Vec<(f64, f64)> = set
        .records
        .par_iter()
        .zip(&set.entries)
        .map(|(r, e)| (eval_record(r, values), f64::from(e.result) / 2.0))
        .collect();
    let loss_at = |k: f64| pairs.par_iter().map(|&(s, y)| loss.loss(sigmoid(s, k), y)).sum::<f64>() / pairs.len() as f64;
    let (k_min, k_max) = SCORE_K;
    let k = golden_search_k(k_min, k_max, 1e-6 * (k_max - k_min), loss_at);

    if k <= k_min * 1.01 || k >= k_max * 0.99 {
        alarm!("K converged to boundary [{k_min}, {k_max}] on {}: loss may not be optimal.", set.label);
    }

    (k, loss_at(k))
}

fn material_rows(set: &Dataset) -> Vec<([f64; 10], f64)> {
    let phase_w = piece_phase_weights();
    set.records
        .par_iter()
        .zip(&set.entries)
        .map(|(record, entry)| {
            let mg = f64::from(phase_of(record, &phase_w) as u32) / f64::from(TOTAL_PHASE);
            let mut design = [0.0; 10];
            for pt in 0..5 {
                let diff = f64::from(record.material_diff[pt]);
                (design[pt], design[pt + 5]) = (diff * mg, diff * (1.0 - mg));
            }
            (design, f64::from(entry.result) / 2.0)
        })
        .collect()
}

fn newton_fit(rows: &[([f64; 10], f64)]) -> ([f64; 10], f64, usize) {
    let mut w = [0.0f64; 10];
    let mut loss = fit_loss(rows, &w);

    for step in 1..=FIT_STEPS {
        let (gradient, hessian) = fit_derivatives(rows, &w);
        let Some(delta) = solve(hessian, gradient) else { return (w, loss, step) };

        let (mut scale, mut moved) = (1.0, false);
        for _ in 0..8 {
            let candidate = std::array::from_fn(|i| w[i] - scale * delta[i]);
            let candidate_loss = fit_loss(rows, &candidate);
            if candidate_loss <= loss {
                (w, loss, moved) = (candidate, candidate_loss, true);
                break;
            }
            scale *= 0.5;
        }

        if !moved || l2_norm(&delta) * scale < FIT_TOL {
            return (w, loss, step);
        }
    }

    (w, loss, FIT_STEPS)
}

fn fit_derivatives(rows: &[([f64; 10], f64)], w: &[f64; 10]) -> ([f64; 10], [f64; 100]) {
    let (g, h) = rows
        .par_iter()
        .fold(
            || ([0.0f64; 10], [0.0f64; 100]),
            |(mut g, mut h), (x, y)| {
                let p = sigmoid(dot(x, w), FIT_K);
                let (res, curv) = (FIT_K * (p - y), FIT_K * FIT_K * p * (1.0 - p));
                for i in 0..10 {
                    if x[i] != 0.0 {
                        g[i] += res * x[i];
                        for j in i..10 {
                            h[i * 10 + j] += curv * x[i] * x[j];
                        }
                    }
                }
                (g, h)
            },
        )
        .reduce(
            || ([0.0f64; 10], [0.0f64; 100]),
            |(mut g, mut h), (dg, dh)| {
                add_slice(&mut g, &dg);
                add_slice(&mut h, &dh);
                (g, h)
            },
        );

    let n = rows.len() as f64;
    let mut hessian: [f64; 100] = std::array::from_fn(|i| h[i] / n);

    for i in 0..10 {
        // 1e-12 ridge keeps the system invertible when a piece imbalance never occurs in the set.
        hessian[i * 10 + i] += 1e-12;
        for j in 0..i {
            hessian[i * 10 + j] = hessian[j * 10 + i];
        }
    }

    (std::array::from_fn(|i| g[i] / n), hessian)
}

fn fit_loss(rows: &[([f64; 10], f64)], w: &[f64; 10]) -> f64 {
    rows.par_iter()
        .map(|(x, y)| LossFn::CrossEntropy.loss(sigmoid(dot(x, w), FIT_K), *y))
        .sum::<f64>()
        / rows.len() as f64
}

#[inline(always)]
fn dot(x: &[f64; 10], w: &[f64; 10]) -> f64 { (0..10).map(|i| x[i] * w[i]).sum() }

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
            if factor != 0.0 {
                for k in col..10 {
                    a[row * 10 + k] -= factor * a[col * 10 + k];
                }
                b[row] -= factor * b[col];
            }
        }
    }

    let mut x = [0.0f64; 10];

    for row in (0..10).rev() {
        x[row] = (b[row] - (row + 1..10).map(|k| a[row * 10 + k] * x[k]).sum::<f64>()) / a[row * 10 + row];
    }
    Some(x)
}

fn spread(sets: &[Dataset]) {
    let bin_headings = |bins: usize, max: usize, tail: &[&str]| -> Vec<String> {
        let step = (max + 1) / bins;
        (0..bins)
            .map(|b| (b * step).to_string())
            .chain(tail.iter().map(|&s| s.to_string()))
            .collect()
    };

    let mut lo = 0;
    let mut shape_bins = Vec::with_capacity(SHAPE_HIGHS.len() + 1);

    for hi in SHAPE_HIGHS {
        let head = if lo == hi { lo.to_string() } else { format!("{lo}-{hi}") };
        shape_bins.push((lo..hi + 1, head));
        lo = hi + 1;
    }

    shape_bins.push((lo..MOBILITY_MAX, format!("{lo}+")));

    for set in sets {
        let stats = fold_dataset(set, FeatureSpread::observe, FeatureSpread::merged);
        println!();

        let mut summary = Table::new(&set.label, ["pieces", "mean", "max", "p99", "p99.9", "at 0"]);
        let mut shape = Table::new("share at count", shape_bins.iter().map(|(_, h)| h.as_str()));

        for (row, piece) in PIECES.iter().enumerate() {
            let counts = &stats.mobility[row];
            let pieces: u64 = counts.iter().sum();

            if pieces == 0 {
                continue;
            }

            let weighted: u64 = counts.iter().enumerate().map(|(c, n)| c as u64 * n).sum();
            let max = counts.iter().rposition(|&n| n > 0).unwrap_or(0);
            let label = piece.to_char(Color::White).to_lowercase().to_string();

            summary.push(&label, vec![
                fmt_comma(pieces),
                format!("{:.2}", weighted as f64 / pieces as f64),
                max.to_string(),
                stats.cutoff(row, 0.99).to_string(),
                stats.cutoff(row, 0.999).to_string(),
                format!("{:.2}%", pct(counts[0], pieces)),
            ]);

            let cells: Vec<String> = shape_bins
                .iter()
                .map(|(range, _)| format!("{:.1}%", pct(counts[range.clone()].iter().sum(), pieces)))
                .collect();
            shape.push(&label, cells);
        }

        summary.print();
        println!();
        shape.print();
        println!();

        let n = stats.phase_all.n;
        let mut openness = Table::new("openness", bin_headings(OPENNESS_BINS, OPEN_UNITY as usize, &["closed", "open", "mean"]));
        let mut cells: Vec<String> = stats.openness.iter().map(|&b| format!("{:.1}%", pct(b, n))).collect();

        cells.extend([
            format!("{:.2}%", pct(stats.openness_floor, n)),
            format!("{:.2}%", pct(stats.openness_ceiling, n)),
            format!("{:.0}", stats.openness_sum / n as f64),
        ]);

        openness.push(&set.label, cells);
        openness.print();
        println!();

        let mut phase = Table::new("phase", bin_headings(PHASE_BINS, TOTAL_PHASE as usize, &["positions", "mean"]));
        phase.push_dim("every position", stats.phase_all.cells());
        phase.push("rank-7 passer", stats.phase_rank7.cells());
        phase.print();
    }
}

// Uniform striding preserves opening-to-endgame progression across games without PRNG state.
fn strided(entries: Vec<SoulEntry>, cap: usize) -> Vec<SoulEntry> {
    if cap == 0 || entries.len() <= cap {
        return entries;
    }
    let stride = entries.len().div_ceil(cap);
    let sampled: Vec<SoulEntry> = entries.iter().copied().step_by(stride).collect();
    println!(
        "  Sampling {} of {} positions, every {stride}",
        fmt_comma(sampled.len() as u64),
        fmt_comma(entries.len() as u64)
    );
    sampled
}

fn label_for(paths: &[String]) -> String {
    const SUFFIXES: [&str; 7] = [".zst", ".soul", ".vf", ".viri", ".epd", ".txt", ".json"];

    let stem = |path: &String| {
        let mut name = Path::new(path).file_name().unwrap_or_default().to_string_lossy().into_owned();
        while let Some(shorter) = SUFFIXES.iter().find_map(|s| name.strip_suffix(s)) {
            name = shorter.to_string();
        }
        name
    };
    match paths {
        [] => "unnamed".into(),
        [one] => stem(one),
        [first, rest @ ..] => format!("{} +{}", stem(first), rest.len()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn newton_fit_recovers_planted_weights_and_zeros_unconstrained_slots() {
        let target = [100.0, 370.0, 382.0, 471.0, 752.0, 162.0, 513.0, 550.0, 970.0, 1897.0];
        let mut rows = Vec::new();

        for phase in 0..=24 {
            let mg = f64::from(phase) / f64::from(TOTAL_PHASE);

            for piece in 0..5 {
                for diff in [-2.0, -1.0, 1.0, 2.0] {
                    let mut design = [0.0; 10];
                    (design[piece], design[piece + 5]) = (diff * mg, diff * (1.0 - mg));
                    rows.push((design, sigmoid(dot(&design, &target), FIT_K)));
                }
            }
        }

        let (fitted, _, steps) = newton_fit(&rows);
        assert!(steps < FIT_STEPS && fitted.iter().zip(&target).all(|(g, w)| (g - w).abs() < 1.0));

        rows.retain(|(d, _)| d[4] == 0.0 && d[9] == 0.0);
        let (no_queen, ..) = newton_fit(&rows);
        assert!(no_queen[4].abs() < 1e-6 && no_queen[9].abs() < 1e-6 && (no_queen[0] - 100.0).abs() < 1.0);
    }

    #[test]
    fn strided_spans_dataset_and_honors_cap() {
        let entries: Vec<SoulEntry> = (0..100)
            .map(|i| {
                let mut e = SoulEntry::default();
                e.occupancy = i;
                e
            })
            .collect();
        let s = strided(entries.clone(), 10);
        assert_eq!((s.len(), s.first().map(|e| e.occupancy), s.last().map(|e| e.occupancy)), (10, Some(0), Some(90)));
        assert_eq!((strided(entries.clone(), 100).len(), strided(entries, 0).len()), (100, 100));
    }
}
