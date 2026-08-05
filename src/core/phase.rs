//! Game phase calculation for tapered evaluation.
//!
//! Interpolates between middlegame and endgame weights based on remaining material.

use crate::{core::defs::TOTAL_PHASE, engine::eval_params::LAYOUT};

/// Compute the clamped phase from f64 piece counts. The tuner feeds `f64` counts
/// from gradient traces; the engine reads its phase from the PSQT accumulator
/// lane, not this formula.
///
/// Anything scoring against the engine takes the phase itself and hands it to
/// [`crate::engine::combiner::taper`]. Splitting it into `(mg_w, eg_w)` first and
/// multiplying rounds differently, because the combiner divides by `TOTAL_PHASE`
/// once at the end.
#[inline]
pub fn compute_phase_f64(piece_counts: &[f64; 6], values: &[f64]) -> f64 {
    debug_assert!(values.len() > LAYOUT.phase_offset + 5, "a short vector silently drops piece types from the phase");

    let mut phase_raw = 0.0;

    for (pt, &count) in piece_counts.iter().enumerate().take(6) {
        let phase_idx = LAYOUT.phase_offset + pt;

        if phase_idx < values.len() {
            phase_raw += count * values[phase_idx];
        }
    }

    phase_raw.clamp(0.0, f64::from(TOTAL_PHASE)).trunc()
}
