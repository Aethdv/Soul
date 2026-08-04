//! What the data actually determines: the curvature of the training objective.
//!
//! At fixed K the Hessian is built exactly, its per-position weight read from
//! `LossFn::hessian_scale` so it can never disagree with the loss the run trains.
//! For the canonical logit link, CE's weight is the clean closed form:
//!
//! ```text
//! H = k² · Σ_i w_i · p_i(1 − p_i) · a_i a_iᵀ
//! ```
//!
//! where `a_i` is the eval's gradient at position i, the same sparse coefficient vector the
//! training scatter already produces, and `p_i` is the predicted win probability. The naive
//! squared-error guess of `p²(1 − p)²` is wrong even for MSE: its exact weight carries the
//! residual cross-term and can go negative, the loss's own non-convexity showing up where a
//! Gauss-Newton approximation would hide it.
//!
//! At 491 parameters it is a two-megabyte object, so it can simply be built: how many directions
//! the data constrains, which combinations it leaves free, and how much of one parameter's column
//! another already explains. A seed sweep sees those flat directions from the outside, as
//! parameters that disagree at equal loss; this sees them from the inside and can name them.

use super::{
    engine::Tunable,
    palette::{DIM, LAB, RESET, VAL},
};

/// Sweeps the Jacobi rotation is allowed before it gives up on an unconverged matrix.
const MAX_SWEEPS: usize = 40;

/// How many of the flattest directions and most collinear parameters get printed.
const LISTED: usize = 6;

/// Eigenvalue below this fraction of the largest counts as zero.
///
/// A positive semi-definite matrix has no negative eigenvalues, so the −8e−16 the rotation leaves
/// behind is rounding, and the direction it belongs to is one the data does not constrain.
/// The cutoff sits four orders above that debris rather than on top of it.
const NULL: f64 = 1e-12;

/// Symmetric curvature over the free parameters, row-major and dense.
pub struct Curvature {
    n: usize,
    h: Vec<f64>,
}

/// Eigendecomposition of the correlation form, largest eigenvalue first.
///
/// The correlation form rather than the raw matrix, because parameters range from a tempo bonus
/// near 30 to a queen near 1000, and raw eigenvectors would rank directions by their units.
pub struct Spectrum {
    values: Vec<f64>,
    /// Eigenvectors as columns: component `i` of eigenvector `j` at `i · live.len() + j`.
    vectors: Vec<f64>,
    /// Parameter index of each row, since parameters the data never touches are left out.
    live: Vec<usize>,
    /// Raw curvature diagonal, over every free parameter. This is the figure a freeze threshold
    /// wants to normalize against.
    diagonal: Vec<f64>,
    /// Free parameters with no positive curvature. Zero is the data never touching them;
    /// negative is `MeanSquaredError`, whose residual cross-term can outweigh `S(1−S)`.
    untouched: Vec<usize>,
}

impl Curvature {
    #[must_use]
    pub fn zeros(n: usize) -> Self {
        Self { n, h: vec![0.0; n * n] }
    }

    /// Adds `weight · a aᵀ` for one position, given `a`'s nonzeros in ascending index order.
    ///
    /// Only the upper triangle is written; [`Self::symmetrized`] mirrors it once at the end rather
    /// than doing twice the work on every one of tens of millions of positions.
    pub fn add_outer(&mut self, weight: f64, nonzeros: &[(usize, f64)]) {
        for (col, &(j, aj)) in nonzeros.iter().enumerate() {
            let scaled = weight * aj;

            for &(i, ai) in &nonzeros[..=col] {
                self.h[i * self.n + j] += scaled * ai;
            }
        }
    }

    pub fn merge(&mut self, other: &Self) {
        for (lhs, rhs) in self.h.iter_mut().zip(&other.h) {
            *lhs += rhs;
        }
    }

    /// Mirrors the upper triangle down, leaving a full symmetric matrix.
    #[must_use]
    pub fn symmetrized(mut self) -> Self {
        for i in 0..self.n {
            for j in i + 1..self.n {
                self.h[j * self.n + i] = self.h[i * self.n + j];
            }
        }
        self
    }

    /// Diagonalizes the correlation form over `considered`, the parameters actually being trained.
    ///
    /// Those with zero curvature drop out first: they would divide the correlation form by zero,
    /// and they are an answer in themselves rather than a numerical nuisance.
    #[must_use]
    pub fn spectrum(&self, considered: &[usize]) -> Spectrum {
        let diagonal: Vec<f64> = (0..self.n).map(|i| self.h[i * self.n + i]).collect();
        let live: Vec<usize> = considered.iter().copied().filter(|&i| diagonal[i] > 0.0).collect();
        let untouched: Vec<usize> = considered.iter().copied().filter(|&i| diagonal[i] <= 0.0).collect();

        let m = live.len();
        let scale: Vec<f64> = live.iter().map(|&i| diagonal[i].sqrt()).collect();

        let mut a = vec![0.0; m * m];

        for (row, &i) in live.iter().enumerate() {
            for (col, &j) in live.iter().enumerate() {
                a[row * m + col] = self.h[i * self.n + j] / (scale[row] * scale[col]);
            }
        }

        let mut vectors = vec![0.0; m * m];

        for i in 0..m {
            vectors[i * m + i] = 1.0;
        }

        jacobi(&mut a, &mut vectors, m);

        let mut order: Vec<usize> = (0..m).collect();
        order.sort_unstable_by(|&x, &y| a[y * m + y].total_cmp(&a[x * m + x]));

        let values: Vec<f64> = order.iter().map(|&j| a[j * m + j]).collect();
        let sorted = {
            let mut v = vec![0.0; m * m];

            for (col, &j) in order.iter().enumerate() {
                for row in 0..m {
                    v[row * m + col] = vectors[row * m + j];
                }
            }
            v
        };

        Spectrum { values, vectors: sorted, live, diagonal, untouched }
    }
}

impl Spectrum {
    /// Number of leading directions the data constrains.
    fn determined(&self) -> usize {
        let cutoff = self.values.first().copied().unwrap_or(0.0) * NULL;

        self.values.iter().filter(|&&x| x > cutoff).count()
    }

    /// How much of each parameter lies in the null space, `Σ_k V_jk²` over the null block.
    ///
    /// The individual null eigenvectors are arbitrary, since any rotation within a null space
    /// diagonalizes it just as well, so naming them would report an artifact of the solver.
    /// This sum is invariant under that rotation. The figures total the null dimension.
    fn participation(&self) -> Vec<f64> {
        let m = self.live.len();

        (0..m)
            .map(|j| (self.determined()..m).map(|k| self.vectors[j * m + k].powi(2)).sum())
            .collect()
    }

    /// Variance inflation of every live parameter: how much of its column the others explain.
    ///
    /// `(C⁻¹)_jj = Σ_k V_jk² / λ_k` on the correlation form, whose unit diagonal makes that the
    /// inflation factor directly. Summed over the determined directions only: a parameter inside
    /// the null space is not badly conditioned, it is unidentifiable, which participation reports
    /// and a reciprocal would only turn into a large arbitrary number.
    fn inflation(&self) -> Vec<f64> {
        let m = self.live.len();

        (0..m)
            .map(|j| (0..self.determined()).map(|k| self.vectors[j * m + k].powi(2) / self.values[k]).sum())
            .collect()
    }

    /// Parameters ranked by a per-parameter figure, largest first, as `(figure, name)`.
    fn ranked<'a>(&self, figures: &[f64], params: &'a [Tunable]) -> Vec<(f64, &'a str)> {
        let mut ranked: Vec<(f64, &str)> = figures
            .iter()
            .zip(&self.live)
            .map(|(&figure, &i)| (figure, params.get(i).map_or("?", |p| p.name.as_str())))
            .collect();

        ranked.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        ranked
    }

    /// Everything the spectrum has to say.
    pub fn report(&self, params: &[Tunable], positions: usize, k: f64) {
        let m = self.live.len();
        let name = |i: usize| params.get(i).map_or("?", |p| p.name.as_str());

        let Some(&largest) = self.values.first() else {
            eprintln!("No curvature at all: the objective does not depend on any free parameter.");
            return;
        };

        // A unit diagonal makes the trace the live-row count, so the largest eigenvalue is at
        // least 1 and one direction always clears the cutoff. Asserted so that a change to NULL
        // fails loudly rather than wrapping the index below.
        let determined = self.determined();
        assert!(determined > 0, "a live parameter with no determined direction is impossible");

        let smallest = self.values[determined - 1];

        println!("\n{LAB}Curvature{RESET} {DIM}({positions} train positions at K = {k:.6}){RESET}");
        println!(
            "  {LAB}parameters{RESET}      {VAL}{m}{RESET} carry curvature, {VAL}{}{RESET} never appear",
            self.untouched.len()
        );
        println!("  {LAB}directions{RESET}      {VAL}{determined}{RESET} determined, {VAL}{}{RESET} free", m - determined);
        println!(
            "  {LAB}eigenvalues{RESET}     max {VAL}{largest:.3e}{RESET}  min {VAL}{smallest:.3e}{RESET}  \
             condition {VAL}{:.2e}{RESET}",
            largest / smallest
        );

        // Rank against a threshold rather than against zero: the small end of a finite sample's
        // spectrum is never exactly zero, and the question worth asking is how many directions
        // carry signal well clear of it.
        let rank = |relative: f64| self.values.iter().filter(|&&x| x > largest * relative).count();
        println!(
            "  {LAB}effective rank{RESET}  {VAL}{}{RESET} above 1e-3 of max, {VAL}{}{RESET} above 1e-6, {VAL}{}{RESET} above 1e-9",
            rank(1e-3),
            rank(1e-6),
            rank(1e-9)
        );

        let raw_max = self.diagonal.iter().copied().fold(0.0_f64, f64::max);
        let raw_min = self.diagonal.iter().copied().filter(|&x| x > 0.0).fold(f64::MAX, f64::min);
        println!("  {LAB}raw diagonal{RESET}    max {VAL}{raw_max:.3e}{RESET}  min {VAL}{raw_min:.3e}{RESET}");

        if !self.untouched.is_empty() {
            let names: Vec<&str> = self.untouched.iter().take(LISTED).map(|&i| name(i)).collect();
            let more = self.untouched.len().saturating_sub(names.len());
            let tail = if more > 0 { format!(" and {more} more") } else { String::new() };

            println!("  {LAB}never appear{RESET}    {}{tail}", names.join(", "));
        }

        if determined < m {
            println!(
                "\n{LAB}Undetermined{RESET} {DIM}(share of each parameter lying in the {} free directions){RESET}",
                m - determined
            );

            for &(share, name) in self.ranked(&self.participation(), params).iter().take(LISTED) {
                println!("  {VAL}{share:5.2}{RESET}  {name}");
            }
        }

        println!("\n{LAB}Flattest determined directions{RESET} {DIM}(heaviest loadings){RESET}");

        for k in (determined.saturating_sub(LISTED)..determined).rev() {
            let mut loadings: Vec<(f64, usize)> = (0..m).map(|row| (self.vectors[row * m + k], self.live[row])).collect();

            loadings.sort_unstable_by(|a, b| b.0.abs().total_cmp(&a.0.abs()));

            let named: Vec<String> = loadings.iter().take(3).map(|&(w, i)| format!("{} {w:+.2}", name(i))).collect();

            println!("  {VAL}{:.3e}{RESET}  {}", self.values[k], named.join("   "));
        }

        // Raw rather than correlation-scaled, so the bottom of this list is the sparse
        // parameters a global gradient threshold mistakes for converged.
        println!("\n{LAB}Least curvature{RESET} {DIM}(raw diagonal, the freeze normalizer){RESET}");

        let raw: Vec<f64> = self.live.iter().map(|&i| -self.diagonal[i]).collect();

        for &(negated, name) in self.ranked(&raw, params).iter().take(LISTED) {
            println!("  {VAL}{:9.3e}{RESET}  {name}", -negated);
        }

        println!("\n{LAB}Most collinear parameters{RESET} {DIM}(variance inflation){RESET}");

        for &(factor, name) in self.ranked(&self.inflation(), params).iter().take(LISTED) {
            println!("  {VAL}{factor:9.3e}{RESET}  {name}");
        }
    }
}

/// Cyclic Jacobi rotation to a diagonal matrix, eigenvectors accumulated into `vectors`.
///
/// Every rotation zeroes one off-diagonal pair and is orthogonal, so the eigenvalues stay on the
/// diagonal and the accumulated product stays an orthonormal basis; no pivoting, no deflation, and
/// nothing to tune. Slower than a tridiagonal reduction and irrelevantly so at this size.
fn jacobi(a: &mut [f64], vectors: &mut [f64], n: usize) {
    // Convergence against the matrix's own scale, so it means the same thing on a Hessian scaled
    // by any dataset size.
    let tolerance = 1e-14 * a.iter().map(|x| x * x).sum::<f64>().max(f64::MIN_POSITIVE);

    for _ in 0..MAX_SWEEPS {
        let off: f64 = (0..n)
            .flat_map(|p| (p + 1..n).map(move |q| (p, q)))
            .map(|(p, q)| a[p * n + q].powi(2))
            .sum();

        if off <= tolerance {
            return;
        }

        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];

                if apq == 0.0 {
                    continue;
                }

                // Rotate by the angle that zeroes (p, q). The reciprocal form of tan
                // is stable at small angles, where the quadratic formula would cancel.
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let t = theta.signum() / (theta.abs() + theta.mul_add(theta, 1.0).sqrt());
                let c = 1.0 / t.mul_add(t, 1.0).sqrt();
                let s = t * c;

                for k in 0..n {
                    let (kp, kq) = (a[k * n + p], a[k * n + q]);

                    a[k * n + p] = c * kp - s * kq;
                    a[k * n + q] = s * kp + c * kq;
                }

                for k in 0..n {
                    let (pk, qk) = (a[p * n + k], a[q * n + k]);

                    a[p * n + k] = c * pk - s * qk;
                    a[q * n + k] = s * pk + c * qk;
                }

                for k in 0..n {
                    let (kp, kq) = (vectors[k * n + p], vectors[k * n + q]);

                    vectors[k * n + p] = c * kp - s * kq;
                    vectors[k * n + q] = s * kp + c * kq;
                }
            }
        }
    }
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

    /// Two parameters the data can only ever see the sum of. The flat direction
    /// is their difference, and it has to come out as the smallest eigenvalue.
    #[test]
    fn a_duplicated_column_leaves_one_flat_direction() {
        let mut curvature = Curvature::zeros(3);

        // a = (1, 1, x): the first two coefficients always move together, the third varies.
        for x in [-2.0, -1.0, 1.0, 3.0] {
            curvature.add_outer(1.0, &[(0, 1.0), (1, 1.0), (2, x)]);
        }

        let spectrum = curvature.symmetrized().spectrum(&[0, 1, 2]);

        assert!(spectrum.untouched.is_empty(), "every parameter here carries curvature");
        assert_eq!(spectrum.values.len(), 3);

        let smallest = *spectrum.values.last().unwrap();
        assert!(smallest < 1e-12, "the duplicated pair must leave a null direction, got {smallest}");

        // Its eigenvector is the difference of the two aliased parameters,
        // so they load equal and opposite while the third barely appears.
        let m = 3;
        let col = m - 1;
        let (first, second, third) = (spectrum.vectors[col], spectrum.vectors[m + col], spectrum.vectors[2 * m + col]);

        assert!((first + second).abs() < 1e-9, "loadings must cancel: {first} and {second}");
        assert!(first.abs() > 0.5, "the aliased pair must carry the direction: {first}");
        assert!(third.abs() < 1e-9, "the third parameter is determined: {third}");
    }
}
