//! Hessian accumulation and spectral analysis of the parameter objective.
//!
//! Under the logistic link with scaling factor `k`, the exact cross-entropy Hessian is:
//!
//! ```text
//! H = k² · Σ_i w_i · p_i(1 − p_i) · a_i a_iᵀ
//! ```
//!
//! where `a_i` is the evaluation feature gradient at position `i`, `w_i` is the sample weight,
//! and `p_i` is the predicted win probability. The Gauss-Newton weight `p²(1 − p)²` is wrong
//! even under squared error, whose exact weight includes a residual cross-term and can turn negative.
//!
//! Diagonalizing the parameter correlation matrix identifies unconstrained directions (null space),
//! ill-conditioned parameter combinations, and collinearities (variance inflation factors).

use crate::{
    engine::Tunable,
    report::{DIM, LAB, RESET, VAL},
    training::add_slice,
};

const MAX_SWEEPS: usize = 40;
const REPORT_LIMIT: usize = 6;
const LOADINGS: usize = 3;

// Jacobi rotation leaves −8e−16 behind on a positive semi-definite matrix, so the null-space
// cutoff sits four orders above rounding noise.
const RELATIVE_NULL_THRESHOLD: f64 = 1e-12;

pub(crate) struct Curvature {
    dim:  usize,
    data: Vec<f64>,
}

/// Eigendecomposition of `C_ij = H_ij / √(H_ii · H_jj)`, sorted descending by eigenvalue so
/// parameter interactions are compared free of physical units.
pub(crate) struct Spectrum {
    values:    Vec<f64>,
    // Column-major eigenvectors: component i of eigenvector j is at i * live.len() + j.
    vectors:   Vec<f64>,
    live:      Vec<usize>,
    diagonal:  Vec<f64>,
    // Parameters in the considered set with zero or negative diagonal curvature. Negative occurs
    // under MeanSquaredError, whose residual cross-term can outweigh S(1 − S).
    untouched: Vec<usize>,
}

impl Curvature {
    pub(crate) fn zeros(dim: usize) -> Self { Self { dim, data: vec![0.0; dim * dim] } }

    /// Accumulates `weight · a aᵀ` into the upper triangle; requires `nonzeros` sorted ascending
    /// by parameter index, followed by one [`Self::symmetrized`] call after reduction.
    pub(crate) fn add_outer(&mut self, weight: f64, nonzeros: &[(usize, f64)]) {
        debug_assert!(nonzeros.windows(2).all(|w| w[0].0 < w[1].0), "nonzeros must be in ascending index order");

        for (col, &(j, aj)) in nonzeros.iter().enumerate() {
            let scaled = weight * aj;
            for &(i, ai) in &nonzeros[..=col] {
                self.data[i * self.dim + j] += scaled * ai;
            }
        }
    }

    pub(crate) fn merge(&mut self, other: &Self) { add_slice(&mut self.data, &other.data); }

    pub(crate) fn symmetrized(mut self) -> Self {
        for i in 0..self.dim {
            for j in i + 1..self.dim {
                self.data[j * self.dim + i] = self.data[i * self.dim + j];
            }
        }
        self
    }

    pub(crate) fn spectrum(&self, considered: &[usize]) -> Spectrum {
        let diagonal: Vec<f64> = (0..self.dim).map(|i| self.data[i * self.dim + i]).collect();
        let live: Vec<usize> = considered.iter().copied().filter(|&i| diagonal[i] > 0.0).collect();
        let untouched: Vec<usize> = considered.iter().copied().filter(|&i| diagonal[i] <= 0.0).collect();
        let m = live.len();
        let scales: Vec<f64> = live.iter().map(|&i| diagonal[i].sqrt()).collect();

        let mut corr = vec![0.0; m * m];

        for (row, &i) in live.iter().enumerate() {
            for (col, &j) in live.iter().enumerate() {
                corr[row * m + col] = self.data[i * self.dim + j] / (scales[row] * scales[col]);
            }
        }

        let mut vectors = vec![0.0; m * m];
        for i in 0..m {
            vectors[i * m + i] = 1.0;
        }

        jacobi(&mut corr, &mut vectors, m);

        let mut order: Vec<usize> = (0..m).collect();
        order.sort_unstable_by(|&x, &y| corr[y * m + y].total_cmp(&corr[x * m + x]));

        let values: Vec<f64> = order.iter().map(|&j| corr[j * m + j]).collect();
        let mut sorted_vectors = vec![0.0; m * m];

        for (col, &j) in order.iter().enumerate() {
            for row in 0..m {
                sorted_vectors[row * m + col] = vectors[row * m + j];
            }
        }

        Spectrum { values, vectors: sorted_vectors, live, diagonal, untouched }
    }
}

impl Spectrum {
    pub(crate) fn report(&self, params: &[Tunable], positions: usize, k: f64) {
        let m = self.live.len();
        let name = |i: usize| params.get(i).map_or("?", |p| p.name.as_str());

        let Some(&largest) = self.values.first() else {
            eprintln!("No curvature: objective has no dependence on active parameters.");
            return;
        };

        // Correlation matrix trace equals m, guaranteeing λ_max >= 1.0. Asserted because the
        // index below underflows if the threshold is ever raised past every eigenvalue.
        let determined = self.determined();
        assert!(determined > 0, "active parameters must have at least one determined direction");
        let smallest = self.values[determined - 1];

        println!("\n{LAB}Curvature{RESET} {DIM}({positions} train positions at K = {k:.6}){RESET}");
        println!(
            "  {LAB}parameters{RESET}      {VAL}{m}{RESET} have curvature, {VAL}{}{RESET} never appear",
            self.untouched.len()
        );
        println!("  {LAB}directions{RESET}      {VAL}{determined}{RESET} determined, {VAL}{}{RESET} free", m - determined);
        println!(
            "  {LAB}eigenvalues{RESET}     max {VAL}{largest:.3e}{RESET}  min {VAL}{smallest:.3e}{RESET}  \
             condition {VAL}{:.2e}{RESET}",
            largest / smallest
        );

        // Three thresholds rather than one rank: a finite sample's smallest eigenvalues are never
        // exactly zero, so the question is how far above that floor the signal reaches.
        let rank = |relative: f64| self.values.iter().filter(|&&x| x > largest * relative).count();
        println!(
            "  {LAB}effective rank{RESET}  {VAL}{}{RESET} above 1e-3 of max, {VAL}{}{RESET} above 1e-6, {VAL}{}{RESET} above 1e-9",
            rank(1e-3),
            rank(1e-6),
            rank(1e-9)
        );

        // Per position, since add_outer sums over the split and the other probes report means.
        let per_position = 1.0 / positions as f64;
        let raw_max = self.diagonal.iter().copied().fold(0.0_f64, f64::max) * per_position;
        let raw_min = self.diagonal.iter().copied().filter(|&x| x > 0.0).fold(f64::MAX, f64::min) * per_position;

        println!(
            "  {LAB}raw diagonal{RESET}    max {VAL}{raw_max:.3e}{RESET}  min {VAL}{raw_min:.3e}{RESET} {DIM}per position{RESET}"
        );

        if !self.untouched.is_empty() {
            let names: Vec<&str> = self.untouched.iter().take(REPORT_LIMIT).map(|&i| name(i)).collect();
            let more = self.untouched.len().saturating_sub(names.len());
            let tail = if more > 0 { format!(" and {more} more") } else { String::new() };
            println!("  {LAB}never appear{RESET}    {}{tail}", names.join(", "));
        }

        if determined < m {
            println!(
                "\n{LAB}Undetermined{RESET} {DIM}(share of each parameter lying in the {} free directions){RESET}",
                m - determined
            );
            for &(share, name) in self.ranked(&self.participation(), params).iter().take(REPORT_LIMIT) {
                println!("  {VAL}{share:5.2}{RESET}  {name}");
            }
        }

        println!("\n{LAB}Flattest determined directions{RESET} {DIM}(heaviest loadings){RESET}");
        let flattest: Vec<(f64, Vec<(&str, f64)>)> = (determined.saturating_sub(REPORT_LIMIT)..determined)
            .rev()
            .map(|eig_idx| {
                let mut loadings: Vec<(f64, usize)> = (0..m).map(|row| (self.vectors[row * m + eig_idx], self.live[row])).collect();
                loadings.sort_unstable_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));
                (self.values[eig_idx], loadings.iter().take(LOADINGS).map(|&(w, i)| (name(i), w)).collect())
            })
            .collect();

        // Each column is padded to its own longest name, so the signed loadings form a column.
        let mut widths = [0usize; LOADINGS];
        for (_, row) in &flattest {
            for (col, (parameter, _)) in row.iter().enumerate() {
                widths[col] = widths[col].max(parameter.len());
            }
        }

        for (value, row) in &flattest {
            let cells: Vec<String> = row.iter().zip(&widths).map(|((n, w), &pad)| format!("{n:<pad$} {w:+.2}")).collect();
            println!("  {VAL}{value:.3e}{RESET}  {}", cells.join("   ").trim_end());
        }

        println!("\n{LAB}Least curvature{RESET} {DIM}(raw diagonal, the freeze normalizer){RESET}");
        let diag: Vec<f64> = self.live.iter().map(|&i| self.diagonal[i]).collect();
        for &(value, name) in self.ranked(&diag, params).iter().rev().take(REPORT_LIMIT) {
            println!("  {VAL}{:9.3e}{RESET}  {name}", value);
        }

        println!("\n{LAB}Most collinear parameters{RESET} {DIM}(variance inflation){RESET}");
        for &(factor, name) in self.ranked(&self.inflation(), params).iter().take(REPORT_LIMIT) {
            println!("  {VAL}{factor:9.3e}{RESET}  {name}");
        }
    }

    fn determined(&self) -> usize {
        let cutoff = self.values.first().copied().unwrap_or(0.0) * RELATIVE_NULL_THRESHOLD;
        self.values.iter().filter(|&&x| x > cutoff).count()
    }

    // Fraction of each parameter's variance lying in the null space (Σ_k V_jk² for k >= determined),
    // invariant under orthogonal rotations of the null subspace.
    fn participation(&self) -> Vec<f64> {
        let d = self.determined();
        self.vectors
            .chunks_exact(self.live.len())
            .map(|row| row[d..].iter().map(|&v| v * v).sum())
            .collect()
    }

    // Variance inflation factor (C⁻¹)_jj = Σ_k (V_jk² / λ_k) across determined directions.
    // Null directions are excluded: a parameter there is unidentifiable rather than
    // ill-conditioned, and 1/λ would report an arbitrarily large number for it.
    fn inflation(&self) -> Vec<f64> {
        let d = self.determined();
        self.vectors
            .chunks_exact(self.live.len())
            .map(|row| row[..d].iter().zip(&self.values).map(|(&v, &l)| v * v / l).sum())
            .collect()
    }

    fn ranked<'a>(&self, figures: &[f64], params: &'a [Tunable]) -> Vec<(f64, &'a str)> {
        let mut ranked: Vec<(f64, &str)> = figures
            .iter()
            .zip(&self.live)
            .map(|(&figure, &i)| (figure, params.get(i).map_or("?", |p| p.name.as_str())))
            .collect();

        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        ranked
    }
}

#[inline]
fn rotate_cols(mat: &mut [f64], p: usize, q: usize, n: usize, c: f64, s: f64) {
    for k in 0..n {
        let (kp, kq) = (mat[k * n + p], mat[k * n + q]);
        mat[k * n + p] = c * kp - s * kq;
        mat[k * n + q] = s * kp + c * kq;
    }
}

#[inline]
fn rotate_rows(mat: &mut [f64], p: usize, q: usize, n: usize, c: f64, s: f64) {
    for k in 0..n {
        let (pk, qk) = (mat[p * n + k], mat[q * n + k]);
        mat[p * n + k] = c * pk - s * qk;
        mat[q * n + k] = s * pk + c * qk;
    }
}

fn jacobi(matrix: &mut [f64], vectors: &mut [f64], n: usize) {
    let tolerance = 1e-14 * matrix.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);
    for _ in 0..MAX_SWEEPS {
        let off_diagonal_sq: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| matrix[p * n + q].powi(2))
            .sum();

        if off_diagonal_sq <= tolerance {
            return;
        }

        for p in 0..n {
            for q in p + 1..n {
                let apq = matrix[p * n + q];
                if apq == 0.0 {
                    continue;
                }

                // Numerically stable t = tan(θ), avoiding cancellation when |τ| is large.
                let tau = (matrix[q * n + q] - matrix[p * n + p]) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + tau.mul_add(tau, 1.0).sqrt());
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;

                rotate_cols(matrix, p, q, n, c, s);
                rotate_rows(matrix, p, q, n, c, s);
                rotate_cols(vectors, p, q, n, c, s);
            }
        }
    }

    eprintln!("Jacobi rotation did not converge within {MAX_SWEEPS} sweeps; eigenvalues are approximate.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes_known_matrix() {
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let mut v = vec![1.0, 0.0, 0.0, 1.0];

        jacobi(&mut a, &mut v, 2);

        let mut found = [a[0], a[3]];
        found.sort_by(f64::total_cmp);
        assert!((found[0] - 1.0).abs() < 1e-12, "smallest eigenvalue: {}", found[0]);
        assert!((found[1] - 3.0).abs() < 1e-12, "largest eigenvalue: {}", found[1]);

        for col in 0..2 {
            let norm = (v[col].powi(2) + v[2 + col].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "eigenvector {col} norm: {norm}");
        }
    }

    #[test]
    fn duplicated_column_leaves_one_null_direction() {
        let mut curvature = Curvature::zeros(3);
        for x in [-2.0, -1.0, 1.0, 3.0] {
            curvature.add_outer(1.0, &[(0, 1.0), (1, 1.0), (2, x)]);
        }

        let spectrum = curvature.symmetrized().spectrum(&[0, 1, 2]);
        assert!(spectrum.untouched.is_empty());
        assert_eq!(spectrum.values.len(), 3);
        let smallest = *spectrum.values.last().unwrap();
        assert!(smallest < 1e-12, "aliased pair must leave a null direction, got {smallest}");

        let (first, second, third) = (spectrum.vectors[2], spectrum.vectors[5], spectrum.vectors[8]);
        assert!((first + second).abs() < 1e-9, "loadings must cancel: {first} and {second}");
        assert!(first.abs() > 0.5, "aliased pair must dominate: {first}");
        assert!(third.abs() < 1e-9, "independent parameter must not load: {third}");
    }
}
