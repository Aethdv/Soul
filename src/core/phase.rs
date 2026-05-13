//! Game phase calculation for tapered evaluation.
//!
//! Interpolates between middlegame and endgame weights based on remaining material.

use crate::core::{defs::TOTAL_PHASE, psqt};

/// Compute phase weights from piece counts.
/// Used directly during standard engine evaluation.
#[inline]
pub fn compute_phase_weights(piece_counts: &[i32; 6], values: &[f64]) -> (f64, f64) {
    compute_phase_weights_inner(piece_counts, values)
}

/// Compute phase weights from f64 piece counts (for gradient tracing).
#[inline]
pub fn compute_phase_weights_f64(piece_counts: &[f64; 6], values: &[f64]) -> (f64, f64) {
    compute_phase_weights_inner(piece_counts, values)
}

/// Core phase interpolation. Generic over the count type so both the
/// engine (`i32` piece counts from PSQT accumulators) and the tuner
/// (`f64` counts from gradient traces) share the same formula.
///
/// Returns `(mg_weight, eg_weight)` where both are in `[0.0, 1.0]`
/// and `mg_weight + eg_weight == 1.0`.
#[inline]
fn compute_phase_weights_inner<T: Into<f64> + Copy>(piece_counts: &[T; 6], values: &[f64]) -> (f64, f64) {
    let mut phase_raw = 0.0;
    for (pt, &count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;
        if phase_idx < values.len() {
            phase_raw += count.into() * values[phase_idx];
        }
    }
    let t_phase = TOTAL_PHASE as f64;
    let phase = phase_raw.clamp(0.0, t_phase).trunc();
    let mg_w = phase / t_phase;
    let eg_w = 1.0 - mg_w;

    (mg_w, eg_w)
}
