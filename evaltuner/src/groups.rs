//! Parameter group classification and per-parameter optimizer masks.
//!
//! Groups correspond to contiguous layout slices, enabling slice-based metrics
//! and flat lookup masks for optimizer hyperparameters.

use std::ops::Range;

use crate::{
    config::EvalTuneConfig,
    engine::{LAYOUT, TABLE_SQUARES},
};

/// Bound on each mobility weight. The four lanes multiply raw counts of safe squares and
/// attacked pieces, where a PSQT coefficient is one or two, so an excursion costs far more score.
pub const MOB_CLAMP: f64 = 100.0;

/// Monotonicity bounds for quadratic king-danger scaling: `p + c · p² / SCALE`.
///
/// A negative curvature `c` places the turnover point `p = -SCALE / (2c)` within
/// reachable pressure values, causing an intensely attacked king to score safer
/// than a moderately attacked one. Clamping `c >= 0` ensures strictly monotonic danger.
pub const DANGER_CURVE_CLAMP: (f64, f64) = (0.0, 256.0);

/// Evaluator parameter group names in layout order.
pub const GROUP_NAMES: [&str; 4] = ["psqt", "material", "mobility", "other"];

/// Parameter group classifications matching layout partitions.
#[repr(usize)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ParamGroup {
    Psqt = 0,
    Material = 1,
    Mobility = 2,
    Other = 3,
}

/// Resolves the [`ParamGroup`] for a given flat layout index.
pub fn param_group(index: usize) -> ParamGroup {
    if index < LAYOUT.material_offset {
        ParamGroup::Psqt
    } else if index < LAYOUT.mobility_open_offset {
        ParamGroup::Material
    } else if index < LAYOUT.mobility_closed_offset + LAYOUT.mobility_closed_len {
        ParamGroup::Mobility
    } else {
        ParamGroup::Other
    }
}

/// Derives the index range of each parameter group from [`param_group`].
///
/// Deriving spans directly from group queries guarantees consistency if layout
/// offsets change, while asserting that every group occupies a contiguous slice.
pub fn group_ranges(slots: usize) -> [Range<usize>; GROUP_NAMES.len()] {
    let mut spans = [(usize::MAX, 0usize); GROUP_NAMES.len()];
    let mut counts = [0usize; GROUP_NAMES.len()];

    for index in 0..slots {
        let group = param_group(index) as usize;
        spans[group].0 = spans[group].0.min(index);
        spans[group].1 = index + 1;
        counts[group] += 1;
    }

    std::array::from_fn(|group| {
        if counts[group] == 0 {
            return 0..0;
        }

        let (start, end) = spans[group];
        assert_eq!(end - start, counts[group], "Group '{}' is not contiguous in the parameter layout", GROUP_NAMES[group]);
        start..end
    })
}

/// The per-parameter weight decay multiplier mask.
///
/// - PSQT (0.5×).
/// - Mobility (1.5×).
pub fn build_decay_mask(slots: usize) -> Vec<f64> {
    (0..slots)
        .map(|index| match param_group(index) {
            ParamGroup::Psqt => {
                let sq = index % TABLE_SQUARES;
                let (rank, file) = (sq / 4, sq % 4);
                // Half-table files 2..=3 map to files c-d (mirrored as e-f), forming the c3-f6 central box.
                let is_center = (2..=5).contains(&rank) && (2..=3).contains(&file);
                if is_center { 0.5 } else { 1.0 }
            },
            ParamGroup::Mobility => 1.5,
            ParamGroup::Material | ParamGroup::Other => 1.0,
        })
        .collect()
}

/// The per-parameter second-moment decay (`beta2`) mask.
///
/// - PSQT (0.995).
/// - Mobility (0.95).
pub fn build_beta2_mask(slots: usize, default_beta2: f64) -> Vec<f64> {
    (0..slots)
        .map(|index| match param_group(index) {
            ParamGroup::Psqt => 0.995,
            ParamGroup::Mobility => 0.95,
            ParamGroup::Material | ParamGroup::Other => default_beta2,
        })
        .collect()
}

/// The per-parameter learning rate mask.
pub fn build_lr_mask(slots: usize, config: &EvalTuneConfig) -> Vec<f64> {
    (0..slots)
        .map(|index| match param_group(index) {
            ParamGroup::Psqt => config.lr_psqt,
            ParamGroup::Material => config.lr_material,
            ParamGroup::Mobility => config.lr_mobility,
            ParamGroup::Other => config.lr_other,
        })
        .collect()
}

/// Allowed parameter value bounds `(min, max)` for optimizer steps.
pub fn build_clip_mask(slots: usize) -> Vec<(f64, f64)> {
    let king_danger_offset = LAYOUT.king_danger_offset;
    (0..slots)
        .map(|index| {
            if index == king_danger_offset {
                return DANGER_CURVE_CLAMP;
            }

            match param_group(index) {
                ParamGroup::Mobility => (-MOB_CLAMP, MOB_CLAMP),
                ParamGroup::Psqt | ParamGroup::Material | ParamGroup::Other => (f64::NEG_INFINITY, f64::INFINITY),
            }
        })
        .collect()
}
