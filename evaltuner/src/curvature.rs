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
    palette::{DIM, LAB, RESET, VAL},
};

/// Maximum cyclic Jacobi sweeps before terminating.
const MAX_SWEEPS: usize = 40;

/// Number of top entries to display in summary reports.
const REPORT_LIMIT: usize = 6;

/// Relative eigenvalue threshold below which a direction is treated as unconstrained (null space).
///
/// The rotation leaves −8e−16 behind on a positive semi-definite matrix, so the cutoff sits four
/// orders above that rounding rather than on top of it.
const RELATIVE_NULL_THRESHOLD: f64 = 1e-12;

/// Dense, row-major symmetric curvature (Hessian) matrix over engine parameters.
pub struct Curvature {
    dim: usize,
    data: Vec<f64>,
}

/// Eigendecomposition of the parameter correlation matrix, sorted descending by eigenvalue.
///
/// Decomposition is performed on the correlation matrix (`C_ij = H_ij / √(H_ii · H_jj)`)
/// rather than the raw Hessian to evaluate parameter interactions independent of their physical units.
pub struct Spectrum {
    values: Vec<f64>,
    /// Column-major eigenvectors: component `i` of eigenvector `j` is at `i * live.len() + j`.
    vectors: Vec<f64>,
    /// Indices of parameters with strictly positive diagonal curvature.
    live: Vec<usize>,
    /// Raw Hessian diagonal across all parameters.
    diagonal: Vec<f64>,
    /// Parameters in the considered set with zero or negative diagonal curvature. Negative occurs
    /// under `MeanSquaredError`, whose residual cross-term can outweigh `S(1 − S)`.
    untouched: Vec<usize>,
}

impl Curvature {
    #[must_use]
    pub fn zeros(dim: usize) -> Self { Self { dim, data: vec![0.0; dim * dim] } }

    /// Accumulates `weight · a aᵀ` for a sparse gradient vector `a`.
    ///
    /// Requires `nonzeros` to be sorted by parameter index in ascending order.
    /// Updates only the upper triangle; call [`Self::symmetrized`] once after accumulation completes.
    pub fn add_outer(&mut self, weight: f64, nonzeros: &[(usize, f64)]) {
        debug_assert!(nonzeros.windows(2).all(|w| w[0].0 < w[1].0), "nonzeros must be in ascending index order");

        for (col, &(j, aj)) in nonzeros.iter().enumerate() {
            let scaled = weight * aj;
            for &(i, ai) in &nonzeros[..=col] {
                self.data[i * self.dim + j] += scaled * ai;
            }
        }
    }

    /// Accumulates another curvature matrix elementwise.
    pub fn merge(&mut self, other: &Self) {
        for (lhs, rhs) in self.data.iter_mut().zip(&other.data) {
            *lhs += rhs;
        }
    }

    /// Copies the upper triangle into the lower triangle, producing a full symmetric matrix.
    #[must_use]
    pub fn symmetrized(mut self) -> Self {
        for i in 0..self.dim {
            for j in i + 1..self.dim {
                self.data[j * self.dim + i] = self.data[i * self.dim + j];
            }
        }
        self
    }

    /// Computes the eigendecomposition of the correlation matrix for the active parameter subset.
    /// Parameters with non-positive diagonal curvature are excluded to avoid zero division.
    #[must_use]
    pub fn spectrum(&self, considered: &[usize]) -> Spectrum {
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
        let sorted_vectors = {
            let mut v = vec![0.0; m * m];
            for (col, &j) in order.iter().enumerate() {
                for row in 0..m {
                    v[row * m + col] = vectors[row * m + j];
                }
            }
            v
        };
        Spectrum { values, vectors: sorted_vectors, live, diagonal, untouched }
    }
}

impl Spectrum {
    /// Number of constrained directions with correlation eigenvalues above `RELATIVE_NULL_THRESHOLD · λ_max`.
    fn determined(&self) -> usize {
        let cutoff = self.values.first().copied().unwrap_or(0.0) * RELATIVE_NULL_THRESHOLD;
        self.values.iter().filter(|&&x| x > cutoff).count()
    }

    /// Fraction of each parameter's variance lying in the null space (`Σ_k V_jk²` for `k >= determined`).
    /// Invariant under orthogonal rotations of the null subspace.
    fn participation(&self) -> Vec<f64> {
        let m = self.live.len();
        let determined = self.determined();
        (0..m).map(|j| (determined..m).map(|k| self.vectors[j * m + k].powi(2)).sum()).collect()
    }

    /// Variance inflation factor (VIF) for each parameter across determined directions:
    /// `(C⁻¹)_jj = Σ_k (V_jk² / λ_k)`.
    ///
    /// Null directions are excluded: a parameter there is unidentifiable rather than
    /// ill-conditioned, and `1/λ` would report an arbitrarily large number for it.
    fn inflation(&self) -> Vec<f64> {
        let m = self.live.len();
        let determined = self.determined();
        (0..m)
            .map(|j| (0..determined).map(|k| self.vectors[j * m + k].powi(2) / self.values[k]).sum())
            .collect()
    }

    /// Pairs metric values with parameter names, sorted in descending order.
    fn ranked<'a>(&self, figures: &[f64], params: &'a [Tunable]) -> Vec<(f64, &'a str)> {
        let mut ranked: Vec<(f64, &str)> = figures
            .iter()
            .zip(&self.live)
            .map(|(&figure, &i)| (figure, params.get(i).map_or("?", |p| p.name.as_str())))
            .collect();

        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        ranked
    }

    pub fn report(&self, params: &[Tunable], positions: usize, k: f64) {
        let m = self.live.len();
        let name = |i: usize| params.get(i).map_or("?", |p| p.name.as_str());

        let Some(&largest) = self.values.first() else {
            eprintln!("No curvature: objective has no dependence on active parameters.");
            return;
        };

        // Correlation matrix trace equals `m`, guaranteeing λ_max >= 1.0. Asserted because the
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

        // Per position, since `add_outer` sums over the split and the other probes report means.
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
        for eig_idx in (determined.saturating_sub(REPORT_LIMIT)..determined).rev() {
            let mut loadings: Vec<(f64, usize)> = (0..m).map(|row| (self.vectors[row * m + eig_idx], self.live[row])).collect();
            loadings.sort_unstable_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));

            let named: Vec<String> = loadings.iter().take(3).map(|&(w, i)| format!("{} {w:+.2}", name(i))).collect();
            println!("  {VAL}{:.3e}{RESET}  {}", self.values[eig_idx], named.join("   "));
        }

        println!("\n{LAB}Least curvature{RESET} {DIM}(raw diagonal, the freeze normalizer){RESET}");
        let neg_diag: Vec<f64> = self.live.iter().map(|&i| -self.diagonal[i]).collect();
        for &(negated, name) in self.ranked(&neg_diag, params).iter().take(REPORT_LIMIT) {
            println!("  {VAL}{:9.3e}{RESET}  {name}", -negated);
        }

        println!("\n{LAB}Most collinear parameters{RESET} {DIM}(variance inflation){RESET}");
        for &(factor, name) in self.ranked(&self.inflation(), params).iter().take(REPORT_LIMIT) {
            println!("  {VAL}{factor:9.3e}{RESET}  {name}");
        }
    }
}

/// Cyclic Jacobi rotation diagonalizing a symmetric matrix in-place.
/// Accumulates orthonormal eigenvectors into `vectors`.
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

                for k in 0..n {
                    let (kp, kq) = (matrix[k * n + p], matrix[k * n + q]);
                    matrix[k * n + p] = c * kp - s * kq;
                    matrix[k * n + q] = s * kp + c * kq;
                }
                for k in 0..n {
                    let (pk, qk) = (matrix[p * n + k], matrix[q * n + k]);
                    matrix[p * n + k] = c * pk - s * qk;
                    matrix[q * n + k] = s * pk + c * qk;
                }
                for k in 0..n {
                    let (kp, kq) = (vectors[k * n + p], vectors[k * n + q]);
                    vectors[k * n + p] = c * kp - s * kq;
                    vectors[k * n + q] = s * kp + c * kq;
                }
            }
        }
    }

    eprintln!("Jacobi rotation did not converge within {MAX_SWEEPS} sweeps; eigenvalues are approximate.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jacobi_diagonalizes_a_known_matrix() {
        // [[2, 1], [1, 2]] has eigenvalues 3 and 1 on (1, 1)/√2 and (1, −1)/√2.
        let mut a = vec![2.0, 1.0, 1.0, 2.0];
        let mut v = vec![1.0, 0.0, 0.0, 1.0];

        jacobi(&mut a, &mut v, 2);

        let mut found = [a[0], a[3]];
        found.sort_by(f64::total_cmp);
        assert!((found[0] - 1.0).abs() < 1e-12, "smallest eigenvalue: {}", found[0]);
        assert!((found[1] - 3.0).abs() < 1e-12, "largest eigenvalue: {}", found[1]);

        for col in 0..2 {
            let norm = (v[col].powi(2) + v[2 + col].powi(2)).sqrt();
            assert!((norm - 1.0).abs() < 1e-12, "eigenvector {col} is not a unit vector: {norm}");
        }
    }

    #[test]
    fn a_duplicated_column_leaves_one_flat_direction() {
        let mut curvature = Curvature::zeros(3);
        // a = (1, 1, x): parameters 0 and 1 are perfectly collinear.
        for x in [-2.0, -1.0, 1.0, 3.0] {
            curvature.add_outer(1.0, &[(0, 1.0), (1, 1.0), (2, x)]);
        }

        let spectrum = curvature.symmetrized().spectrum(&[0, 1, 2]);
        assert!(spectrum.untouched.is_empty(), "every parameter here has positive curvature");
        assert_eq!(spectrum.values.len(), 3);
        let smallest = *spectrum.values.last().unwrap();
        assert!(smallest < 1e-12, "aliased pair must leave a null direction, got {smallest}");
        let m = 3;
        let col = m - 1;
        let (first, second, third) = (spectrum.vectors[col], spectrum.vectors[m + col], spectrum.vectors[2 * m + col]);
        assert!((first + second).abs() < 1e-9, "loadings must cancel: {first} and {second}");
        assert!(first.abs() > 0.5, "aliased pair must dominate the direction: {first}");
        assert!(third.abs() < 1e-9, "independent parameter must not load in the null direction: {third}");
    }
}
