//! Game phase calculation for tapered evaluation.
//!
//! Interpolates between middlegame and endgame weights based on remaining material.

use crate::core::{defs::TOTAL_PHASE, psqt};

/// Compute `(mg_weight, eg_weight)` from f64 piece counts, both in `[0.0, 1.0]`
/// and summing to `1.0`. The tuner feeds `f64` counts from gradient traces;
/// the engine reads its phase from the PSQT accumulator lane, not this formula.
#[inline]
pub fn compute_phase_weights_f64(piece_counts: &[f64; 6], values: &[f64]) -> (f64, f64) {
    let mut phase_raw = 0.0;

    for (pt, &count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.weight_offset + pt;

        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }

    let t_phase = TOTAL_PHASE as f64;
    let phase = phase_raw.clamp(0.0, t_phase).trunc();
    let mg_w = phase / t_phase;
    let eg_w = 1.0 - mg_w;

    (mg_w, eg_w)
}
