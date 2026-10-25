//! Win/Draw/Loss probability model.

/// Material-dependent logistic curves. `a` scales the score so +100 cp is 50% win; `b` controls the spread.
const A_COEFFS: [f64; 4] = [-72.32565836, 185.93832038, -144.58862193, 416.44950446];
const B_COEFFS: [f64; 4] = [83.86794042, -136.06112997, 69.98820887, 47.62901433];

pub(crate) fn wdl_model(score: i32, material: u32) -> (f64, f64, f64) {
    let m = f64::from(material.clamp(17, 78)) / 58.0;
    let a = A_COEFFS[0].mul_add(m, A_COEFFS[1]).mul_add(m, A_COEFFS[2]).mul_add(m, A_COEFFS[3]);
    let b = B_COEFFS[0].mul_add(m, B_COEFFS[1]).mul_add(m, B_COEFFS[2]).mul_add(m, B_COEFFS[3]);
    let v = f64::from(score) * a / 100.0;

    let win = 1.0 / (1.0 + ((a - v) / b).exp());
    let loss = 1.0 / (1.0 + ((a + v) / b).exp());
    let draw = (1.0 - win - loss).max(0.0);

    (win, draw, loss)
}

/// Logistic score to win probability at scale `k`, used by tuner and the gradient oracle.
#[inline]
#[must_use]
pub fn sigmoid(score: f64, k: f64) -> f64 {
    // Keep exp out of subnormal and overflow ranges; libm may bypass FTZ/DAZ.
    let x = (-k * score).clamp(-700.0, 700.0);
    1.0 / (1.0 + x.exp())
}
