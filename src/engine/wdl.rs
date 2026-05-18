//! Win/Draw/Loss probability model.
//!
//! Maps a STM-relative centipawn score and material count to (win, draw, loss)
//! probabilities via Stockfish's logistic coefficients (`win_rate_params`).

const A_COEFFS: [f64; 4] = [-72.32565836, 185.93832038, -144.58862193, 416.44950446];
const B_COEFFS: [f64; 4] = [83.86794042, -136.06112997, 69.98820887, 47.62901433];

/// Win, draw, and loss probabilities from the side-to-move's
/// centipawn score. The material count scales the logistic
/// function — a given score is more decisive with fewer pieces
/// on the board.
pub fn wdl_model(score: i32, material: u32) -> (f64, f64, f64) {
    let m = f64::from(material.clamp(17, 78)) / 58.0;
    let a = A_COEFFS[0].mul_add(m, A_COEFFS[1]).mul_add(m, A_COEFFS[2]).mul_add(m, A_COEFFS[3]);
    let b = B_COEFFS[0].mul_add(m, B_COEFFS[1]).mul_add(m, B_COEFFS[2]).mul_add(m, B_COEFFS[3]);
    let v = f64::from(score) * a / 100.0;

    let w = 1.0 / (1.0 + ((a - v) / b).exp());
    let l = 1.0 / (1.0 + ((a + v) / b).exp());
    let d = (1.0 - w - l).max(0.0);

    (w, d, l)
}
