//! The parameter groups the tuner treats differently, and the per-slot masks it
//! builds from them.
//!
//! Groups are contiguous runs of the layout, so a per-group statistic is a subslice
//! and a per-group policy is a lookup by index.

use std::ops::Range;

use crate::{config::EvalTuneConfig, engine::LAYOUT};

/// Hard clamp for mobility parameters to prevent drift from unbounded features.
pub const MOB_CLAMP: f64 = 100.0;
/// `p + c·p²/DANGER_SCALE` turns over and starts falling at `p = DANGER_SCALE/2c`,
/// so a negative curvature puts that turnover inside the reachable pressure range
/// and a besieged king scores safer than a lightly pressed one. Zero is the
/// linear block, so a run parked on the floor has answered the question.
pub const DANGER_CURVE_CLAMP: (f64, f64) = (0.0, 256.0);

/// Lion's parameter groups in layout order, for anything reported per group.
pub const GROUP_NAMES: [&str; 4] = ["psqt", "material", "mobility", "other"];

/// Parameter group by position in the layout, the axis the per-group optimizer
/// masks (decay, momentum, learning rate) all key off.
pub enum ParamGroup {
    Psqt,
    Material,
    Mobility,
    Other,
}

/// Classify a parameter index into its layout group: the single source of the
/// group boundaries the masks below share.
pub fn param_group(i: usize) -> ParamGroup {
    if i < LAYOUT.material_offset {
        ParamGroup::Psqt
    } else if i < LAYOUT.mobility_open_offset {
        ParamGroup::Material
    } else if i < LAYOUT.mobility_closed_offset + LAYOUT.mobility_closed_len {
        ParamGroup::Mobility
    } else {
        ParamGroup::Other
    }
}

/// Index of a group in [`GROUP_NAMES`] and in anything else reported per group.
pub const fn group_index(group: &ParamGroup) -> usize {
    match group {
        ParamGroup::Psqt => 0,
        ParamGroup::Material => 1,
        ParamGroup::Mobility => 2,
        ParamGroup::Other => 3,
    }
}

/// Each group as an index range, read back off [`param_group`] rather than restated.
///
/// Restating the cuts is how a per-group report drifts into reporting the wrong parameters
/// after a layout change, silently, since every number it prints stays plausible.
/// The contiguity the ranges assume is asserted here rather than assumed.
pub fn group_ranges(slots: usize) -> [Range<usize>; GROUP_NAMES.len()] {
    let mut span = [(usize::MAX, 0usize); GROUP_NAMES.len()];
    let mut counts = [0usize; GROUP_NAMES.len()];

    for i in 0..slots {
        let g = group_index(&param_group(i));
        span[g].0 = span[g].0.min(i);
        span[g].1 = i + 1;
        counts[g] += 1;
    }

    std::array::from_fn(|g| {
        if counts[g] == 0 {
            return 0..0;
        }

        let (lo, hi) = span[g];
        assert_eq!(hi - lo, counts[g], "{} does not occupy a contiguous range of the layout", GROUP_NAMES[g]);
        lo..hi
    })
}

/// Weight decay mask: not all parameters deserve equal punishment.
///
/// - PSQT center squares decay at 0.5× (central values are more structurally
///   significant; aggressive decay risks flattening critical gradients).
/// - Mobility weights decay at 1.5× (these can drift without bound since their
///   features are unbounded integer counts).
/// - Everything else decays at 1.0×.
pub fn build_decay_mask(slots: usize) -> Vec<f64> {
    (0..slots)
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => {
                let sq = i % 32;
                let (row, col) = (sq / 4, sq % 4);
                let is_center = (2..=5).contains(&row) && (2..=3).contains(&col);
                if is_center { 0.5 } else { 1.0 }
            },
            ParamGroup::Mobility => 1.5,
            ParamGroup::Material | ParamGroup::Other => 1.0,
        })
        .collect()
}

/// Per-group learning-rate mask: PSQT, material, mobility, and the rest each scale
/// by their configured rate, so groups on different gradient scales tune independently.
pub fn build_lr_mask(slots: usize, config: &EvalTuneConfig) -> Vec<f64> {
    (0..slots)
        .map(|i| match param_group(i) {
            ParamGroup::Psqt => config.lr_psqt,
            ParamGroup::Material => config.lr_material,
            ParamGroup::Mobility => config.lr_mobility,
            ParamGroup::Other => config.lr_other,
        })
        .collect()
}

/// Per-parameter range the sign step may not leave, unbounded outside mobility
/// and the king-danger curvature.
pub fn build_clip_mask(slots: usize) -> Vec<(f64, f64)> {
    let danger = LAYOUT.king_danger_offset;
    (0..slots)
        .map(|i| {
            if i == danger {
                return DANGER_CURVE_CLAMP;
            }

            match param_group(i) {
                ParamGroup::Mobility => (-MOB_CLAMP, MOB_CLAMP),
                ParamGroup::Psqt | ParamGroup::Material | ParamGroup::Other => (f64::NEG_INFINITY, f64::INFINITY),
            }
        })
        .collect()
}
