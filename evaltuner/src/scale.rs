//! What holds the eval's units still.
//!
//! K maps a centipawn to a win probability, the gauge holds the parameter vector
//! on the scale the search reads, and the fold collapses the one direction between
//! material and PSQT that no evaluation can see.

use crate::{
    config::{EvalTuneConfig, KMode},
    engine::{FeatureRecord, LAYOUT, PIECE_TABLES, PieceType, TABLE_SQUARES, Tunable, eval_record},
    lion::Lion,
    run::TrainerContext,
    storage::CheckpointData,
};

/// Positions the [`Gauge`] reads the eval's scale from, every batch.
pub const GAUGE_PROBE: usize = 1024;

/// The [`Gauge`] reads its probe at this magnification.
///
/// `taper` truncates in the f64 path, so Σ|score| is a staircase in the scale and the gauge
/// oscillates between two treads for the rest of the run. Twelve doublings sink a centipawn of
/// truncation to 2⁻¹² of the score, and a power of two keeps the lift exact.
const MEASURE_GAIN: f64 = 4096.0;

pub struct KController {
    k: f64,
    k_ref: f64,
    mode: KMode,
    k_min: f64,
    k_max: f64,
    beta1: f64,
    beta2: f64,
    pub momentum: f64,
}

impl KController {
    pub fn bootstrap(
        config: &EvalTuneConfig,
        ctx: &TrainerContext,
        values: &[f64],
        defaults: &[f64],
        init_blend: f64,
        resume: Option<&CheckpointData>,
    ) -> Self {
        let (k, k_ref, k_momentum) = match resume {
            Some(data) => (data.run.k, data.run.k_ref, data.run.k_momentum),
            None => match config.k_mode {
                KMode::Fixed { value } => (value, value, 0.0),
                _ => {
                    println!("Optimizing K...");

                    let fit = |weights: &[f64]| {
                        golden_search_k(config.k_min, config.k_max, 1e-6 * (config.k_max - config.k_min), |candidate_k| {
                            ctx.k_fit_eval(weights, candidate_k, init_blend)
                        })
                    };

                    // k_ref is fitted on the defaults: a cold start begins near a zero eval
                    // where the loss is flat in K, and one frozen at that bracket edge makes
                    // ref_loss climb all run as the eval grows a scale.
                    let k = fit(values);
                    let k_ref = if values == defaults { k } else { fit(defaults) };
                    (k, k_ref, 0.0)
                },
            },
        };

        Self {
            k,
            k_ref,
            mode: config.k_mode,
            k_min: config.k_min,
            k_max: config.k_max,
            beta1: config.beta1,
            beta2: config.beta2,
            momentum: k_momentum,
        }
    }

    #[inline]
    #[must_use]
    pub const fn k(&self) -> f64 { self.k }

    #[inline]
    #[must_use]
    pub const fn k_ref(&self) -> f64 { self.k_ref }

    pub fn on_epoch(&mut self, epoch: usize, ctx: &TrainerContext, ema_values: &[f64], blend: f64) -> Option<f64> {
        let KMode::Sweep { interval } = self.mode else {
            return None;
        };

        if !epoch.is_multiple_of(interval.max(1)) {
            return None;
        }

        self.k = golden_search_k(self.k_min, self.k_max, 1e-6 * (self.k_max - self.k_min), |candidate_k| {
            ctx.k_fit_eval(ema_values, candidate_k, blend)
        });

        Some(self.k)
    }

    pub fn on_batch(&mut self, k_grad: f64, batch_count: usize, lr: f64, scale: f64, weight_decay: f64) {
        let KMode::Learned { lr_mult } = self.mode else {
            return;
        };

        let n = batch_count.max(1) as f64;
        let scaled_grad = k_grad / n * scale;
        let eff_lr = lr * lr_mult;

        let c = self.beta1.mul_add(self.momentum, (1.0 - self.beta1) * scaled_grad);
        // sign(0.0) is 1.0: a cold start's zero eval keeps the blended direction
        // at exactly zero, and without the gate K would slide down by eff_lr
        // every batch. Only the sign step is dead; decay still fires.
        let direction = if c.abs() < 1e-9 { 0.0 } else { c.signum() };

        self.k -= eff_lr * (direction + weight_decay * self.k);
        self.momentum = self.beta2.mul_add(self.momentum, (1.0 - self.beta2) * scaled_grad);
        self.k = self.k.clamp(self.k_min, self.k_max);
    }

    /// Takes the other half of a [`Gauge`] rescale: parameters just moved by `factor`,
    /// so K moves by `1/factor` and the product K·score is where it was. `k_ref` is
    /// deliberately untouched, which is what leaves `ref_loss` able to see drift
    /// the gauge could not absorb.
    pub fn rescale(&mut self, factor: f64) {
        self.k = (self.k / factor).clamp(self.k_min, self.k_max);
        self.momentum *= factor;
    }
}

/// Folds each piece's mean PSQT into its material term.
///
/// `material[p] + psqt[p][sq]` is a piece's whole contribution, so a constant moved between them
/// is invisible to the loss. Lion drifts along it anyway: thirty-two square gradients cancelling
/// one material gradient in sum do not cancel in sign, and the step is `±lr` either way. Nothing
/// bounds that drift and it enters every statistic over the vector, weight decay included.
///
/// The king is dropped rather than folded: both sides field one, so its constant cancels. The
/// shift is an integer because `taper` truncates, and ties round up because `f64::round` breaks
/// translation-equivariance at −0.5, where the table would fold back and forth forever.
pub fn canonicalize(values: &mut [f64], params: &[Tunable]) {
    let layout = &LAYOUT;
    for piece in 0..PIECE_TABLES {
        for phase in 0..2 {
            let table_offset = layout.psqt_offset + piece * 2 * TABLE_SQUARES + phase * TABLE_SQUARES;
            let free_squares = || (table_offset..table_offset + TABLE_SQUARES).filter(|&i| !params[i].is_fixed);
            let material_idx = (piece != PieceType::King as usize).then(|| layout.material_offset + phase * PIECE_TABLES + piece);

            // Nowhere to put the fold, and shifting the table alone moves the score.
            if material_idx.is_some_and(|idx| params[idx].is_fixed) {
                continue;
            }

            // The fold shifts material for every square, so a fixed square keeps its value while
            // a piece standing there collects the shift. Only the pawn ranks nobody can occupy
            // are fixed, which is what makes the fold invisible to the score.
            debug_assert!(
                (table_offset..table_offset + TABLE_SQUARES)
                    .filter(|&i| params[i].is_fixed)
                    .all(|i| piece == PieceType::Pawn as usize && (i - table_offset < 4 || i - table_offset >= TABLE_SQUARES - 4)),
                "a fixed PSQT square a piece can stand on takes the fold's shift"
            );

            let count = free_squares().count();
            if count == 0 {
                continue;
            }

            let mean_shift = (free_squares().map(|i| values[i]).sum::<f64>() / count as f64 + 0.5).floor();
            if mean_shift == 0.0 {
                continue;
            }

            for i in free_squares() {
                values[i] -= mean_shift;
            }
            if let Some(mat_idx) = material_idx {
                values[mat_idx] += mean_shift;
            }
        }
    }
}

/// Holds the eval's overall scale still.
///
/// The score is homogeneous of degree one in every parameter outside the phase block, so scaling
/// those by `c` and dividing K by `c` leaves the loss where it was. Lion moves along that freely
/// at `±lr` whatever the gradient says, while K answers at `lr_mult` and cannot keep up, and
/// `search_params` is in fixed centipawns where a drifting eval is not. The anchor is the shipped
/// scale, so a run's output lands where the search expects it.
pub struct Gauge<'a> {
    pub probe: Vec<&'a FeatureRecord>,
    pub reference: f64,
    /// Every correction multiplied together, so 1.0 is a run that never pulled on
    /// the scale and anything else is budget the loss could not price.
    pub applied: f64,
}

impl<'a> Gauge<'a> {
    /// Σ|score| over a fixed set of positions, the scale the search actually pays in.
    ///
    /// Σ|θ| answers a different question, the vector holding directions no evaluation can see:
    /// fold a piece's PSQT mean into its material and the norm moves while every score stands
    /// still. Sizing a correction from that hands a run 20% of scale it never paid for.
    /// [`canonicalize`] closes that direction; measuring the score covers whatever else is there.
    ///
    /// The same positions every time, so the ratio is a rescale.
    pub fn measure(probe: &[&FeatureRecord], values: &[f64]) -> f64 {
        let lifted: Vec<f64> = values.iter().enumerate().map(|(i, &v)| v * Self::slot_scale(i, MEASURE_GAIN)).collect();

        probe.iter().map(|r| eval_record(r, &lifted).abs()).sum()
    }

    #[must_use]
    pub fn new(probe: Vec<&'a FeatureRecord>, defaults: &[f64]) -> Self {
        let reference = Self::measure(&probe, defaults);
        Self { probe, reference, applied: 1.0 }
    }

    /// Whether holding the scale during training is meaningful for this run.
    ///
    /// Only for a run already standing on the reference. A cold start is learning the scale rather
    /// than drifting off one: `Init::Zero` scores every position at zero, where the correction is
    /// undefined, and `Init::Random` an order of magnitude under it. Those get
    /// [`Gauge::normalize`] on the way out instead.
    #[must_use]
    pub fn holds(&self, values: &[f64]) -> bool {
        (Self::measure(&self.probe, values) - self.reference).abs() <= 1e-9 * self.reference
    }

    /// What slot `i` takes when the vector is scaled by `f`.
    ///
    /// The phase block sits out as an interpolation coordinate rather than a score term: scaled
    /// to zero it would clamp `phase_raw` on every position, tapering the eval to its endgame half
    /// without failing anything. Every other fixed slot is zero, where scaling is a no-op. The
    /// king-danger curvature moves the other way, multiplying `pressure²`, so `p + c·p²/S` is
    /// homogeneous only when `c` takes `1/f` against everything else's `f`.
    fn slot_scale(i: usize, f: f64) -> f64 {
        let (phase_lo, phase_hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);
        if (phase_lo..phase_hi).contains(&i) {
            1.0
        } else if i == LAYOUT.king_danger_offset {
            f.recip()
        } else {
            f
        }
    }

    /// Scales `values` back onto the reference, returning the factor applied.
    pub fn normalize(&self, values: &mut [f64]) -> f64 {
        let current = Self::measure(&self.probe, values);
        if !(current.is_finite() && current > 0.0 && self.reference > 0.0) {
            return 1.0;
        }

        let factor = self.reference / current;
        for (i, v) in values.iter_mut().enumerate() {
            *v *= Self::slot_scale(i, factor);
        }
        factor
    }

    /// [`Gauge::normalize`] with the rest of the optimizer state taken along.
    pub fn restore(&mut self, values: &mut [f64], optimizer: &mut Lion, k_ctrl: &mut KController) {
        let factor = self.normalize(values);
        optimizer.rescale(|i| Self::slot_scale(i, factor));
        k_ctrl.rescale(factor);
        self.applied *= factor;
    }
}

/// Golden-section search for K.
///
/// Two interior probes, `a` and `b`; the loser becomes the new boundary, so each iteration costs
/// one fresh eval. The offset is `INV_PHI · range` rather than `range` because the surviving probe
/// already sits at `INV_PHI²` of the old width. Assumes `eval` is unimodal on `[lo, hi]`.
pub fn golden_search_k<F: Fn(f64) -> f64>(lo: f64, hi: f64, tol: f64, eval: F) -> f64 {
    assert!(lo < hi, "golden_search_k: lo ({lo}) must be < hi ({hi})");
    assert!(tol > 0.0, "golden_search_k: tol ({tol}) must be positive");
    if hi - lo <= tol {
        return (lo + hi) / 2.0;
    }

    const INV_PHI: f64 = 0.618_033_988_749_894_9; // (√5 − 1) / 2

    let mut lo = lo;
    let mut hi = hi;
    let mut width = hi - lo;
    let mut a = hi - INV_PHI * width;
    let mut b = lo + INV_PHI * width;
    let mut fa = eval(a);
    let mut fb = eval(b);

    while width > tol {
        width *= INV_PHI;
        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - INV_PHI * width;
            fa = eval(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + INV_PHI * width;
            fb = eval(b);
        }
    }
    (lo + hi) / 2.0
}

#[cfg(test)]
mod tests {
    use super::{
        super::{
            engine::{Position, SoulEntry, eval_params},
            run::seed_values,
        },
        *,
    };
    use crate::config::Init;

    const PROBE_FENS: &[&str] = &[
        "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "2kr3r/ppp2ppp/8/8/8/8/PPP2PPP/2KR3R w - - 0 1",
        "rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR b KQkq - 0 1",
        "rnbqkbn1/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQq - 0 1",
        "4k3/8/8/8/8/8/PPPPPPPP/4K3 w - - 0 1",
        "4k3/pppppppp/8/8/8/8/8/Q3K3 w - - 0 1",
    ];

    /// The result label is unused: the gauge reads scores, never targets.
    fn probe_records() -> Vec<FeatureRecord> {
        PROBE_FENS
            .iter()
            .map(|fen| FeatureRecord::from_entry(&SoulEntry::from_board(&Position::from_fen(fen), 0.5, Some(20))))
            .collect()
    }

    #[test]
    fn canonicalizing_moves_no_score() {
        let params = eval_params::collect_parameters();
        let layout = &LAYOUT;
        let table = layout.psqt_offset + PieceType::Queen as usize * 2 * TABLE_SQUARES;
        let records = probe_records();

        let mut values = eval_params::default_values(&params);
        for v in &mut values[table..table + TABLE_SQUARES] {
            *v += 60.0;
        }

        let mut folded = values.clone();
        canonicalize(&mut folded, &params);
        assert_ne!(folded, values, "a table 60 off its mean has to move");

        for (fen, record) in PROBE_FENS.iter().zip(&records) {
            let (before, after) = (eval_record(record, &values), eval_record(record, &folded));
            assert!((before - after).abs() < 1e-9, "{fen}: score changed from {before} to {after}");
        }

        let mut twice = folded.clone();
        canonicalize(&mut twice, &params);
        for (i, (once, again)) in folded.iter().zip(&twice).enumerate() {
            assert!((once - again).abs() < 1e-9, "the canonical point is not fixed under its own fold: slot {i}");
        }
    }

    #[test]
    fn canonicalizing_collapses_the_flat_direction() {
        let params = eval_params::collect_parameters();
        let layout = &LAYOUT;
        let squares = TABLE_SQUARES;
        let queen = PieceType::Queen as usize;
        let table = layout.psqt_offset + queen * 2 * squares;

        let mut base = eval_params::default_values(&params);
        let mut drifted = base.clone();
        for v in &mut drifted[table..table + squares] {
            *v += 37.0;
        }
        drifted[layout.material_offset + queen] -= 37.0;

        for (fen, record) in PROBE_FENS.iter().zip(&probe_records()) {
            let (flat, moved) = (eval_record(record, &base), eval_record(record, &drifted));
            assert!((flat - moved).abs() < 1e-9, "{fen}: evaluations differ: {flat} vs {moved}");
        }

        canonicalize(&mut base, &params);
        canonicalize(&mut drifted, &params);

        for (i, (want, got)) in base.iter().zip(&drifted).enumerate() {
            assert!((want - got).abs() < 1e-9, "slot {i}: expected {want}, got {got}");
        }

        let mut tied = eval_params::default_values(&params);
        for (n, i) in (table..table + squares).enumerate() {
            tied[i] = if n < squares / 2 { -1.0 } else { 0.0 };
        }

        canonicalize(&mut tied, &params);
        let once = tied.clone();
        canonicalize(&mut tied, &params);
        assert_eq!(tied, once, "a table tied at half a unit never settles");
    }

    #[test]
    fn the_gauge_returns_the_scale_and_pays_k_for_it() {
        let params = eval_params::collect_parameters();
        let mut values = eval_params::default_values(&params);
        let mut optimizer = Lion::new(values.len(), 0.9, 0.1, 0.0);
        optimizer.restore_momentum(&vec![0.5; values.len()]);

        let records = probe_records();
        let probe: Vec<&FeatureRecord> = records.iter().collect();
        let gauge_ref = Gauge::measure(&probe, &values);
        let (phase_lo, phase_hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);
        let phase_before: Vec<f64> = values[phase_lo..phase_hi].to_vec();

        let mut gauge = Gauge::new(probe.clone(), &values);
        assert!(gauge.holds(&values), "a run standing on the reference must gauge");

        let mut k_ctrl = KController {
            k: 0.004,
            k_ref: 0.004,
            mode: KMode::Learned { lr_mult: 0.001 },
            k_min: 0.0001,
            k_max: 1.0,
            beta1: 0.9,
            beta2: 0.99,
            momentum: 0.0,
        };

        for i in (0..values.len()).filter(|i| !(phase_lo..phase_hi).contains(i)) {
            values[i] *= 1.23;
        }

        gauge.restore(&mut values, &mut optimizer, &mut k_ctrl);
        assert!((Gauge::measure(&probe, &values) - gauge_ref).abs() < 1e-6 * gauge_ref, "scale not restored");
        assert_eq!(&values[phase_lo..phase_hi], &phase_before[..], "phase is a coordinate, not a score term");
        assert!((k_ctrl.k() / (0.004 * 1.23) - 1.0).abs() < 1e-6, "K did not take the other half: {}", k_ctrl.k());
        assert!((optimizer.momentum()[0] / (0.5 * 1.23) - 1.0).abs() < 1e-6, "momentum did not follow the gradient rescale");
        assert!((gauge.applied * 1.23 - 1.0).abs() < 1e-6, "the pull was not recorded");
    }

    #[test]
    fn the_gauge_restores_a_live_curvature_too() {
        let params = eval_params::collect_parameters();
        let defaults = eval_params::default_values(&params);
        let curve = LAYOUT.king_danger_offset;

        let mut want = defaults.clone();
        want[curve] = 64.0;

        let records = probe_records();
        let probe: Vec<&FeatureRecord> = records.iter().collect();
        let gauge = Gauge::new(probe.clone(), &want);
        let mut drifted = want.clone();
        let (phase_lo, phase_hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);
        for (i, v) in drifted.iter_mut().enumerate() {
            if (phase_lo..phase_hi).contains(&i) {
                continue;
            }
            *v *= if i == curve { 1.0 / 1.23 } else { 1.23 };
        }

        gauge.normalize(&mut drifted);
        assert!((Gauge::measure(&probe, &drifted) - gauge.reference).abs() < 1e-6 * gauge.reference, "no fixed point");
        for (i, (got, expect)) in drifted.iter().zip(&want).enumerate() {
            assert!((got - expect).abs() < 1e-6 * expect.abs().max(1.0), "slot {i}: expected {expect}, got {got}");
        }
    }

    #[test]
    fn a_cold_start_does_not_gauge() {
        let params = eval_params::collect_parameters();
        let defaults = eval_params::default_values(&params);
        let records = probe_records();
        let gauge = Gauge::new(records.iter().collect(), &defaults);
        for init in [Init::Zero, Init::Random] {
            assert!(!gauge.holds(&seed_values(&params, init, 7)), "{init:?} must not gauge during training");
        }
        assert!(gauge.holds(&seed_values(&params, Init::Default, 7)), "a warm start must");
    }

    #[test]
    fn a_zero_gradient_does_not_slide_learned_k() {
        let mut ctrl = KController {
            k: 0.01,
            k_ref: 0.005,
            mode: KMode::Learned { lr_mult: 0.001 },
            k_min: 0.001,
            k_max: 0.020,
            beta1: 0.9,
            beta2: 0.99,
            momentum: 0.0,
        };

        for _ in 0..100 {
            ctrl.on_batch(0.0, 256, 0.01, 1.0, 0.0);
        }
        assert_eq!(ctrl.k, 0.01, "zero gradient on fresh momentum must not move K");
    }

    #[test]
    fn a_zero_gradient_still_steps_with_momentum() {
        let mut ctrl = KController {
            k: 0.01,
            k_ref: 0.005,
            mode: KMode::Learned { lr_mult: 0.001 },
            k_min: 0.001,
            k_max: 0.020,
            beta1: 0.9,
            beta2: 0.99,
            momentum: 0.5,
        };

        ctrl.on_batch(0.0, 256, 0.01, 1.0, 0.0);
        let expected = 0.01 - 0.01 * 0.001; // eff_lr = lr · lr_mult
        assert!((ctrl.k - expected).abs() < 1e-15, "momentum should still step K: {} against {expected}", ctrl.k);
    }

    #[test]
    fn a_zero_gradient_still_decays_k() {
        let mut ctrl = KController {
            k: 0.01,
            k_ref: 0.005,
            mode: KMode::Learned { lr_mult: 0.001 },
            k_min: 0.001,
            k_max: 0.020,
            beta1: 0.9,
            beta2: 0.99,
            momentum: 0.0,
        };

        let before = ctrl.k;
        ctrl.on_batch(0.0, 256, 0.01, 1.0, 0.00001);
        let expected = before - (0.01 * 0.001) * (0.00001 * before);
        assert!((ctrl.k - expected).abs() < 1e-16, "decay must fire in the dead zone: {} against {expected}", ctrl.k);
    }
}
