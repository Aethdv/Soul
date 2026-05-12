//! Acquisition functions for surrogate-assisted candidate ranking.
//!
//! Currently implements Expected Improvement (EI) for maximization,
//! which is used to rank sampled candidates before expensive self-play evaluation.
//! EI naturally balances exploitation (high mean) and exploration (high uncertainty).

/// Rank candidates by Expected Improvement (descending).
#[must_use]
pub fn rank_with_ei(means: &[f64], std_errs: &[f64], best_elo: f64) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..means.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        let ei_a = expected_improvement(means[a], std_errs[a], best_elo);
        let ei_b = expected_improvement(means[b], std_errs[b], best_elo);
        ei_b.total_cmp(&ei_a)
    });
    indices
}

/// Expected Improvement for maximization.
///
/// EI(x) = (μ - best) · Φ(Z) + σ · φ(Z)
/// where Z = (μ - best) / σ
#[must_use]
pub fn expected_improvement(mean: f64, sigma: f64, best: f64) -> f64 {
    if sigma <= 0.0 {
        return (mean - best).max(0.0);
    }

    let z = (mean - best) / sigma;
    let phi = normal_pdf(z);
    let cdf = normal_cdf(z);

    (mean - best).mul_add(cdf, sigma * phi)
}

/// Gaussian Probability Density Function
fn normal_pdf(x: f64) -> f64 {
    let inv_sqrt_2pi = 0.398_942_280_401_432_7;
    (-0.5 * x * x).exp() * inv_sqrt_2pi
}

/// Abramowitz & Stegun 7.1.26 polynomial approximation to erf.
/// Max error < 1.5e-7.
fn approx_erf(x: f64) -> f64 {
    const A1: f64 = 0.254_829_592;
    const A2: f64 = -0.284_496_736;
    const A3: f64 = 1.421_413_741;
    const A4: f64 = -1.453_152_027;
    const A5: f64 = 1.061_405_429;
    const P: f64 = 0.327_591_1;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = P.mul_add(x, 1.0).recip();
    let poly = A5.mul_add(t, A4).mul_add(t, A3).mul_add(t, A2).mul_add(t, A1);
    let val = (poly * t).mul_add(-(-x * x).exp(), 1.0);

    sign * val
}

/// Gaussian Cumulative Distribution Function,
/// using Abramowitz & Stegun 7.1.26 approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + approx_erf(x / std::f64::consts::SQRT_2))
}
