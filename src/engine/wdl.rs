//! Win/Draw/Loss probability model.

/// Cubics in material: `a` is the score sitting at 50% win probability,
/// `b` the spread around it, so a larger `b` flattens the curve.
const A_COEFFS: [f64; 4] = [-72.32565836, 185.93832038, -144.58862193, 416.44950446];
const B_COEFFS: [f64; 4] = [83.86794042, -136.06112997, 69.98820887, 47.62901433];

/// Maps a side-to-move centipawn score and piece material count to `(win, draw, loss)` probabilities.
pub fn wdl_model(score: i32, material: u32) -> (f64, f64, f64) {
    let m = f64::from(material.clamp(17, 78)) / 58.0;
    let a = A_COEFFS[0].mul_add(m, A_COEFFS[1]).mul_add(m, A_COEFFS[2]).mul_add(m, A_COEFFS[3]);
    let b = B_COEFFS[0].mul_add(m, B_COEFFS[1]).mul_add(m, B_COEFFS[2]).mul_add(m, B_COEFFS[3]);
    // The fit lives in the units `a` normalizes, and the caller hands us centipawns,
    // so this undoes the `100 · v / a` conversion the score arrived through.
    let v = f64::from(score) * a / 100.0;

    let win = 1.0 / (1.0 + ((a - v) / b).exp());
    let loss = 1.0 / (1.0 + ((a + v) / b).exp());
    let draw = (1.0 - win - loss).max(0.0);

    (win, draw, loss)
}

/// The plain logistic link, score to win probability at scale `k`.
///
/// `S(x) = 1 / (1 + exp(-k · x))`, the material-free form of the model above: what a tuner's
/// loss compares its label against, and what the gradient oracle rebuilds a reference loss from.
#[inline]
#[must_use]
pub fn sigmoid(score: f64, k: f64) -> f64 {
    // Clamp to avoid libm subnormal slow paths near underflow which ignore CPU FTZ/DAZ flags.
    let x = (-k * score).clamp(-700.0, 700.0);
    1.0 / (1.0 + x.exp())
}
