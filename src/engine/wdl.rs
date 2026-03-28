//! Win/Draw/Loss probability model.
//!
//! Provides a polynomial regression model to estimate game outcomes based on
//! evaluation scores and remaining material.

const A_COEFFS: [f64; 4] = [-13.500_301_98, 40.927_808_83, -36.827_535_45, 386.830_040_70];
const B_COEFFS: [f64; 4] = [96.533_548_96, -165.790_583_88, 90.896_790_19, 49.295_618_89];

/// Sigmoid-like model where the win/loss probability is a function of the
/// STM-relative centipawn score, scaled by the remaining material on the board.
///
/// NOTE: Coefficients are ported from Stockfish. Soul's internal scale differs,
/// so displayed WDL probabilities are systematically wrong.
///
/// Returns (Win, Draw, Loss) probabilities as floats in [0, 1].
pub fn wdl_model(score: i32, material: u32) -> (f64, f64, f64) {
    let m = f64::from(material.clamp(17, 78)) / 58.0;
    let a = A_COEFFS[0]
        .mul_add(m, A_COEFFS[1])
        .mul_add(m, A_COEFFS[2])
        .mul_add(m, A_COEFFS[3]);
    let b = B_COEFFS[0]
        .mul_add(m, B_COEFFS[1])
        .mul_add(m, B_COEFFS[2])
        .mul_add(m, B_COEFFS[3]);
    let v = f64::from(score) * (a / 100.0);

    let w = 1.0 / (1.0 + ((a - v) / b).exp());
    let l = 1.0 / (1.0 + ((a + v) / b).exp());
    let d = (1.0 - w - l).max(0.0);

    (w, d, l)
}
