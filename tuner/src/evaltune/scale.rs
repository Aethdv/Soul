//! What holds the eval's units still.
//!
//! K maps a centipawn to a win probability, the gauge holds the parameter vector
//! on the scale the search reads, and the fold collapses the one direction between
//! material and PSQT that no evaluation can see.

use super::{
    engine::{FeatureRecord, LAYOUT, PIECE_TABLES, PieceType, TABLE_SQUARES, Tunable, eval_record},
    lion::Lion,
    run::TrainerContext,
    storage::CheckpointData,
};
use crate::core::config::{EvalTuneConfig, KMode};

/// Positions the [`Gauge`] reads the eval's scale from, every batch.
pub const GAUGE_PROBE: usize = 1024;

/// The [`Gauge`] reads its probe at this magnification.
///
/// `taper` truncates in the f64 path too, which makes Σ|score| a staircase in the
/// scale: a correction lands on one tread, the next comes back onto the other, and
/// the gauge oscillates between them for the rest of the run. Twelve doublings sink
/// a centipawn of truncation to 2⁻¹² of the score it perturbs; a power of two keeps
/// the lift itself exact.
const MEASURE_GAIN: f64 = 4096.0;

/// State machine for K.
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
            Some(d) => (d.run.k, d.run.k_ref, d.run.k_momentum),
            None => match config.k_mode {
                KMode::Fixed { value } => (value, value, 0.0),
                _ => {
                    println!("Optimizing K...");

                    let fit = |v: &[f64]| {
                        golden_search_k(config.k_min, config.k_max, 1e-6 * (config.k_max - config.k_min), |kk| {
                            ctx.k_fit_eval(v, kk, init_blend)
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

    pub fn k(&self) -> f64 {
        self.k
    }
    pub fn k_ref(&self) -> f64 {
        self.k_ref
    }

    pub fn on_epoch(&mut self, epoch: usize, ctx: &TrainerContext, ema_values: &[f64], blend: f64) -> Option<f64> {
        let KMode::Sweep { interval } = self.mode else { return None };
        if !epoch.is_multiple_of(interval.max(1)) {
            return None;
        }

        self.k = golden_search_k(self.k_min, self.k_max, 1e-6 * (self.k_max - self.k_min), |kk| {
            ctx.k_fit_eval(ema_values, kk, blend)
        });

        Some(self.k)
    }

    pub fn on_batch(&mut self, k_grad: f64, batch_count: usize, lr: f64, scale: f64, weight_decay: f64) {
        let KMode::Learned { lr_mult } = self.mode else { return };

        let n = batch_count.max(1) as f64;
        let kg = k_grad / n * scale;
        let eff_lr = lr * lr_mult;
        let c = self.beta1.mul_add(self.momentum, (1.0 - self.beta1) * kg);

        // `sign(0.0)` is 1.0: a cold start's zero eval keeps the blended direction
        // at exactly zero, and without the gate K would walk down by `eff_lr`
        // every batch. Only the sign step is dead; decay still fires.
        let direction = if c.abs() < 1e-9 { 0.0 } else { c.signum() };

        self.k -= eff_lr * (direction + weight_decay * self.k);
        self.momentum = self.beta2.mul_add(self.momentum, (1.0 - self.beta2) * kg);
        self.k = self.k.clamp(self.k_min, self.k_max);
    }

    /// Takes the other half of a [`Gauge`] rescale: parameters just moved by `f`,
    /// so K moves by `1/f` and the product K·score is where it was. `k_ref` is
    /// deliberately untouched, which is what leaves `ref_loss` able to see drift
    /// the gauge could not absorb.
    pub fn rescale(&mut self, f: f64) {
        self.k = (self.k / f).clamp(self.k_min, self.k_max);
        self.momentum *= f;
    }
}

/// Folds each piece's mean PSQT into its material term.
///
/// `material[p] + psqt[p][sq]` is the whole of a piece's contribution, so a
/// constant carried from one to the other is invisible to the loss. Lion walks
/// that direction anyway: thirty-two square gradients that cancel against one
/// material gradient in sum do not cancel in sign, and its step is `±lr` either
/// way. Nothing bounds the walk, and it lands in every statistic taken over the
/// vector, weight decay included.
///
/// The king pays nobody: both sides field exactly one, so its constant cancels in
/// the difference and is dropped rather than folded.
///
/// The fold is an integer, so `taper`'s truncation cannot see it: a fractional
/// shift moves any score sitting exactly on an integer by a whole centipawn, and
/// the file ships integers.
///
/// Half rounds up, the tie rule that makes `fold(m + k) = fold(m) + k` hold for
/// integer `k`; without it two vectors an integer apart fold to different points.
/// `f64::round` breaks that at exactly −0.5, which a mean over integer squares hits
/// often enough to see, and the table folds back and forth from then on.
pub fn canonicalize(values: &mut [f64], params: &[Tunable]) {
    let l = &LAYOUT;
    let (pieces, squares) = (PIECE_TABLES, TABLE_SQUARES);

    for piece in 0..pieces {
        for phase in 0..2 {
            let table = l.psqt_offset + piece * 2 * squares + phase * squares;
            let free = || (table..table + squares).filter(|&i| !params[i].is_fixed);
            let sink = (piece != PieceType::King as usize).then(|| l.material_offset + phase * pieces + piece);

            // Nowhere to put the fold, and shifting the table alone moves the score.
            if sink.is_some_and(|s| params[s].is_fixed) {
                continue;
            }

            let n = free().count();

            if n == 0 {
                continue;
            }

            let fold = (free().map(|i| values[i]).sum::<f64>() / n as f64 + 0.5).floor();

            if fold == 0.0 {
                continue;
            }

            for i in free() {
                values[i] -= fold;
            }

            if let Some(s) = sink {
                values[s] += fold;
            }
        }
    }
}
/// Holds the eval's overall scale still.
///
/// The score is homogeneous of degree one in every parameter outside the phase
/// block, so multiplying those by `c` and dividing K by `c` leaves the loss
/// exactly where it was. Lion walks that direction freely, its step being `±lr`
/// whatever the gradient says, while K answers at `lr_mult` and cannot keep up.
/// What the loss cannot price the search pays for: `search_params` is in fixed
/// centipawns and a drifting eval is not.
///
/// The anchor is the shipped scale, so a run's output lands where the search
/// expects it whatever the run started from.
pub struct Gauge<'a> {
    pub probe: Vec<&'a FeatureRecord>,
    pub reference: f64,
    /// Every correction multiplied together, so 1.0 is a run that never pulled on
    /// the scale and anything else is budget the loss could not price.
    pub applied: f64,
}

impl<'a> Gauge<'a> {
    /// Σ|score| over a fixed set of positions, the scale the search actually pays
    /// in.
    ///
    /// Σ|θ| answers a different question, the vector holding directions no
    /// evaluation can see: fold a piece's PSQT mean into its material and the norm
    /// moves while every score stands still. Sizing a correction from that hands a
    /// run 20% of scale for a rebalance it never paid for. [`canonicalize`] closes
    /// the one such direction we know of; measuring the score closes the ones we
    /// don't.
    ///
    /// The same positions every time, so the ratio is a rescale: the score is
    /// homogeneous of degree one under [`Gauge::slot`], and so is any fixed sum of
    /// its magnitudes.
    pub fn measure(probe: &[&FeatureRecord], values: &[f64]) -> f64 {
        let lifted: Vec<f64> = values.iter().enumerate().map(|(i, v)| v * Self::slot(i, MEASURE_GAIN)).collect();

        probe.iter().map(|r| eval_record(r, &lifted).abs()).sum()
    }

    pub fn new(probe: Vec<&'a FeatureRecord>, defaults: &[f64]) -> Self {
        let reference = Self::measure(&probe, defaults);

        Self { probe, reference, applied: 1.0 }
    }

    /// Whether holding the scale during training is meaningful for this run.
    ///
    /// Only for a run already standing on the reference. A cold start is learning
    /// the scale rather than drifting off one, and has none to hold: `Init::Zero`
    /// scores every position at zero, where the correction is undefined, and `Init::Random`
    /// an order of magnitude under it. Those get [`Gauge::normalize`] on the way
    /// out instead.
    pub fn holds(&self, values: &[f64]) -> bool {
        (Self::measure(&self.probe, values) - self.reference).abs() <= 1e-9 * self.reference
    }

    /// What slot `i` takes when the vector is scaled by `f`.
    ///
    /// The phase block sits out, being an interpolation coordinate rather than a
    /// score term; every other fixed slot is zero, where scaling is a no-op. The
    /// king-danger curvature moves the other way: it multiplies `pressure²`, so
    /// `p + c·p²/S` is homogeneous only when `c` takes `1/f` against everything
    /// else's `f`, and scaling it alike leaves the curve trading unevenly against K.
    fn slot(i: usize, f: f64) -> f64 {
        let (lo, hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);

        if (lo..hi).contains(&i) {
            1.0
        } else if i == LAYOUT.king_danger_offset {
            f.recip()
        } else {
            f
        }
    }

    /// Scales `values` back onto the reference, returning the factor applied.
    pub fn normalize(&self, values: &mut [f64]) -> f64 {
        let now = Self::measure(&self.probe, values);

        if !(now.is_finite() && now > 0.0 && self.reference > 0.0) {
            return 1.0;
        }

        let f = self.reference / now;

        for (i, v) in values.iter_mut().enumerate() {
            *v *= Self::slot(i, f);
        }

        f
    }

    /// [`Gauge::normalize`] with the rest of the optimizer state carried along.
    pub fn restore(&mut self, values: &mut [f64], optimizer: &mut Lion, k_ctrl: &mut KController) {
        let f = self.normalize(values);

        optimizer.rescale(|i| Self::slot(i, f));
        k_ctrl.rescale(f);
        self.applied *= f;
    }
}

/// Golden-section search for K.
///
/// Maintains two interior probes `a` and `b`; the losing probe becomes the new
/// boundary, requiring only one fresh eval per iteration. The probe offset is
/// `C · range` (not `range`) because the surviving probe already sits at `C²`
/// of the old width, placing the new probe at `C` of the new width.
///
/// Assumes `eval` is unimodal on `[lo, hi]`; otherwise the result is not
/// guaranteed to be a global minimum.
pub fn golden_search_k<F: Fn(f64) -> f64>(lo: f64, hi: f64, tol: f64, eval: F) -> f64 {
    assert!(lo < hi, "golden_search_k: lo ({lo}) must be < hi ({hi})");
    assert!(tol > 0.0, "golden_search_k: tol ({tol}) must be positive");

    if hi - lo <= tol {
        return (lo + hi) / 2.0;
    }

    const C: f64 = 0.618_033_988_749_894_9; // (√5 − 1) / 2

    let mut lo = lo;
    let mut hi = hi;
    let mut width = hi - lo;
    let mut a = hi - C * width;
    let mut b = lo + C * width;
    let mut fa = eval(a);
    let mut fb = eval(b);

    while width > tol {
        width *= C;

        if fa < fb {
            hi = b;
            b = a;
            fb = fa;
            a = hi - C * width;
            fa = eval(a);
        } else {
            lo = a;
            a = b;
            fa = fb;
            b = lo + C * width;
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
    use crate::core::config::Init;

    /// Stands in for a run's thousand-position probe: the lopsided positions carry
    /// the magnitude, the quiet ones stop it measuring a scale no game reaches.
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

    /// The fold changes coordinates and nothing else.
    #[test]
    fn canonicalizing_moves_no_score() {
        let params = eval_params::collect_parameters();
        let l = &LAYOUT;
        let table = l.psqt_offset + PieceType::Queen as usize * 2 * TABLE_SQUARES;
        let records = probe_records();

        // Off canon by 60 on the queen's midgame table, so the fold has work to do
        // whatever the shipped file happens to hold.
        let mut values = eval_params::default_values(&params);

        for i in table..table + TABLE_SQUARES {
            values[i] += 60.0;
        }

        let mut folded = values.clone();
        canonicalize(&mut folded, &params);

        assert_ne!(folded, values, "a table 60 off its mean has to move");

        for (fen, record) in PROBE_FENS.iter().zip(&records) {
            let (before, after) = (eval_record(record, &values), eval_record(record, &folded));
            assert!((before - after).abs() < 1e-9, "{fen}: {before} became {after}");
        }

        let mut twice = folded.clone();
        canonicalize(&mut twice, &params);

        for (i, (once, again)) in folded.iter().zip(&twice).enumerate() {
            assert!((once - again).abs() < 1e-9, "the canonical point is not fixed under its own fold: slot {i}");
        }
    }

    /// Two vectors differing only along the direction the loss cannot see have to
    /// land on the same point.
    #[test]
    fn canonicalizing_collapses_the_flat_direction() {
        let params = eval_params::collect_parameters();
        let l = &LAYOUT;
        let squares = TABLE_SQUARES;
        let queen = PieceType::Queen as usize;
        let table = l.psqt_offset + queen * 2 * squares;

        let mut base = eval_params::default_values(&params);
        let mut drifted = base.clone();

        // Onto every queen square and off the queen's material: nothing to the eval.
        for i in table..table + squares {
            drifted[i] += 37.0;
        }

        drifted[l.material_offset + queen] -= 37.0;

        for (fen, record) in PROBE_FENS.iter().zip(&probe_records()) {
            let (flat, moved) = (eval_record(record, &base), eval_record(record, &drifted));
            assert!((flat - moved).abs() < 1e-9, "{fen} is not on the flat direction: {flat} against {moved}");
        }

        canonicalize(&mut base, &params);
        canonicalize(&mut drifted, &params);

        for (i, (want, got)) in base.iter().zip(&drifted).enumerate() {
            assert!((want - got).abs() < 1e-9, "slot {i}: {got} against {want}");
        }

        // A mean of exactly half a unit, which integer squares hit often enough to
        // see. Rounding away from zero folds it back and forth from here on.
        let mut tied = eval_params::default_values(&params);

        for (n, i) in (table..table + squares).enumerate() {
            tied[i] = if n < squares / 2 { -1.0 } else { 0.0 };
        }

        canonicalize(&mut tied, &params);
        let once = tied.clone();
        canonicalize(&mut tied, &params);

        assert_eq!(tied, once, "a table tied at half a unit never settles");
    }

    /// A cold start that zeroed `phase` would clamp `phase_raw` to 0 on every
    /// position, tapering the eval to its endgame half without failing anything.
    /// Both halves of one rescale: the parameters land back on the reference and
    /// K takes the reciprocal, so `K·score` is where it started.
    #[test]
    fn the_gauge_returns_the_scale_and_pays_k_for_it() {
        let params = eval_params::collect_parameters();
        let mut values = eval_params::default_values(&params);
        let mut optimizer = Lion::new(values.len(), 0.9, 0.1, 0.0);

        optimizer.restore_momentum(&vec![0.5; values.len()]);

        let records = probe_records();
        let probe: Vec<&FeatureRecord> = records.iter().collect();
        let gauge_ref = Gauge::measure(&probe, &values);
        let (lo, hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);
        let phase_before: Vec<f64> = values[lo..hi].to_vec();

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

        for i in (0..values.len()).filter(|i| !(lo..hi).contains(i)) {
            values[i] *= 1.23;
        }

        gauge.restore(&mut values, &mut optimizer, &mut k_ctrl);

        assert!((Gauge::measure(&probe, &values) - gauge_ref).abs() < 1e-6 * gauge_ref, "scale not restored");
        assert_eq!(&values[lo..hi], &phase_before[..], "phase is a coordinate, not a score term");
        assert!((k_ctrl.k() / (0.004 * 1.23) - 1.0).abs() < 1e-6, "K did not take the other half: {}", k_ctrl.k());
        assert!((optimizer.momentum()[0] / (0.5 * 1.23) - 1.0).abs() < 1e-6, "momentum did not follow the gradient rescale");
        assert!((gauge.applied * 1.23 - 1.0).abs() < 1e-6, "the pull was not recorded");
    }

    /// Gauging a cold start would multiply the vector by zero on the first batch.
    /// The gauge's whole mandate, run with a live curvature: the drift the loss is
    /// blind to comes back, and nothing else does.
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
        let (lo, hi) = (LAYOUT.phase_offset, LAYOUT.phase_offset + LAYOUT.phase_len);

        // Spelled out rather than routed through `Gauge::slot`, so a wrong rule
        // there cannot cancel against itself on both sides of the test.
        for (i, v) in drifted.iter_mut().enumerate() {
            if (lo..hi).contains(&i) {
                continue;
            }
            *v *= if i == curve { 1.0 / 1.23 } else { 1.23 };
        }

        gauge.normalize(&mut drifted);

        assert!((Gauge::measure(&probe, &drifted) - gauge.reference).abs() < 1e-6 * gauge.reference, "no fixed point");

        for (i, (got, expect)) in drifted.iter().zip(&want).enumerate() {
            assert!((got - expect).abs() < 1e-6 * expect.abs().max(1.0), "slot {i}: {got} against {expect}");
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
    fn a_zero_gradient_does_not_walk_learned_k() {
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
