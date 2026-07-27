//! Game phase calculation for tapered evaluation.
//!
//! Interpolates between middlegame and endgame weights based on remaining material.

use crate::core::{defs::TOTAL_PHASE, psqt};

/// Compute the clamped phase from f64 piece counts. The tuner feeds `f64` counts
/// from gradient traces; the engine reads its phase from the PSQT accumulator
/// lane, not this formula.
#[inline]
pub fn compute_phase_f64(piece_counts: &[f64; 6], values: &[f64]) -> f64 {
    let mut phase_raw = 0.0;

    for (pt, &count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = psqt::LAYOUT.phase_offset + pt;

        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }

    phase_raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc()
}

/// Compute `(mg_weight, eg_weight)`, both in `[0.0, 1.0]` and summing to `1.0`.
///
/// A tapered value built from these two rounds differently than
/// [`crate::engine::combiner::taper`], which divides by `TOTAL_PHASE` once at the
/// end. Anything that has to agree with the engine's score takes the phase.
#[inline]
pub fn compute_phase_weights_f64(piece_counts: &[f64; 6], values: &[f64]) -> (f64, f64) {
    let mg_w = compute_phase_f64(piece_counts, values) / f64::from(TOTAL_PHASE);

    (mg_w, 1.0 - mg_w)
}
