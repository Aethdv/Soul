//! Sigmoid scale `K`, the score-scale `Gauge`, and the PSQT-into-material fold.

use crate::{
    config::{EvalTuneConfig, KMode},
    engine::{FeatureRecord, LAYOUT, PIECE_TABLES, PieceType, TABLE_SQUARES, Tunable, eval_record},
    lion::Lion,
    run::Objective,
    storage::CheckpointData,
};

pub(crate) const GAUGE_PROBE: usize = 1024;

// taper truncates in the f64 path, turning Σ|score| into a staircase at 1x scale;
// 2¹² sinks 1 cp of truncation to 2⁻¹² without rounding error.
const MEASURE_GAIN: f64 = 4096.0;

pub(crate) struct KController {
    k: f64,
    k_ref: f64,
    mode: KMode,
    k_min: f64,
    k_max: f64,
    beta1: f64,
    beta2: f64,
    pub(crate) momentum: f64,
}

impl KController {
    pub(crate) fn bootstrap(
        cfg: &EvalTuneConfig,
        obj: &Objective,
        values: &[f64],
        defaults: &[f64],
        init_blend: f64,
        resume: Option<&CheckpointData>,
    ) -> Self {
        let (k, k_ref, momentum) = match resume {
            Some(d) => (d.run.k, d.run.k_ref, d.run.k_momentum),
            None => match cfg.k_mode {
                KMode::Fixed { value } => (value, value, 0.0),
                _ => {
                    println!("Optimizing K...");
                    let fit = |w: &[f64]| {
                        golden_search_k(cfg.k_min, cfg.k_max, 1e-6 * (cfg.k_max - cfg.k_min), |ck| {
                            obj.k_fit_eval(w, ck, init_blend)
                        })
                    };
                    // Fit k_ref on defaults even on cold starts, where values ≈ 0 leaves K flat at a bracket edge.
                    let k = fit(values);
                    (k, if values == defaults { k } else { fit(defaults) }, 0.0)
                },
            },
        };
        Self {
            k,
            k_ref,
            mode: cfg.k_mode,
            k_min: cfg.k_min,
            k_max: cfg.k_max,
            beta1: cfg.beta1,
            beta2: cfg.beta2,
            momentum,
        }
    }

    #[inline]
    pub(crate) const fn k(&self) -> f64 { self.k }

    #[inline]
    pub(crate) const fn k_ref(&self) -> f64 { self.k_ref }

    pub(crate) fn on_epoch(&mut self, epoch: usize, obj: &Objective, ema_values: &[f64], blend: f64) -> Option<f64> {
        let KMode::Sweep { interval } = self.mode else { return None };
        if !epoch.is_multiple_of(interval.max(1)) {
            return None;
        }
        self.k =
            golden_search_k(self.k_min, self.k_max, 1e-6 * (self.k_max - self.k_min), |ck| obj.k_fit_eval(ema_values, ck, blend));
        Some(self.k)
    }

    pub(crate) fn on_batch(&mut self, k_grad: f64, batch_count: usize, lr: f64, clip_scale: f64, weight_decay: f64) {
        let KMode::Learned { lr_mult } = self.mode else { return };
        let g = k_grad / batch_count.max(1) as f64 * clip_scale;
        let eff_lr = lr * lr_mult;
        let c = self.beta1.mul_add(self.momentum, (1.0 - self.beta1) * g);
        let dir = if c.abs() < 1e-9 { 0.0 } else { c.signum() };
        self.k = (self.k - eff_lr * (dir + weight_decay * self.k)).clamp(self.k_min, self.k_max);
        self.momentum = self.beta2.mul_add(self.momentum, (1.0 - self.beta2) * g);
    }

    /// Counterpart to a parameter rescale by `factor`: preserves `K · score` while leaving `k_ref` anchored.
    pub(crate) fn rescale(&mut self, factor: f64) {
        self.k = (self.k / factor).clamp(self.k_min, self.k_max);
        self.momentum *= factor;
    }
}

/// Folds each piece's mean free PSQT offset into its material weight (dropping king, which cancels across sides).
///
/// Shifts by an integer so `taper` truncation is preserved, rounding ties up via `(x + 0.5).floor()`
/// rather than `f64::round()` (which breaks translation-equivariance at `-0.5` and oscillates).
pub(crate) fn canonicalize(values: &mut [f64], params: &[Tunable]) {
    for piece in 0..PIECE_TABLES {
        for phase in 0..2 {
            let offset = LAYOUT.psqt_offset + piece * 2 * TABLE_SQUARES + phase * TABLE_SQUARES;
            let mat_idx = (piece != PieceType::King as usize).then(|| LAYOUT.material_offset + phase * PIECE_TABLES + piece);
            if mat_idx.is_some_and(|idx| params[idx].is_fixed) {
                continue;
            }

            // A fixed square that a piece can actually occupy would collect the material shift without losing it from PSQT.
            debug_assert!(
                (offset..offset + TABLE_SQUARES)
                    .filter(|&i| params[i].is_fixed)
                    .all(|i| piece == PieceType::Pawn as usize && (i - offset < 4 || i - offset >= TABLE_SQUARES - 4)),
                "occupiable fixed PSQT square would take the material fold shift"
            );

            let (mut sum, mut count) = (0.0, 0usize);
            for i in offset..offset + TABLE_SQUARES {
                if !params[i].is_fixed {
                    sum += values[i];
                    count += 1;
                }
            }

            if count == 0 {
                continue;
            }

            let shift = (sum / count as f64 + 0.5).floor();
            if shift != 0.0 {
                for i in offset..offset + TABLE_SQUARES {
                    if !params[i].is_fixed {
                        values[i] -= shift;
                    }
                }
                if let Some(idx) = mat_idx {
                    values[idx] += shift;
                }
            }
        }
    }
}

/// Pins `Σ|score|` over a fixed probe to the shipped eval scale so tuned weights match `search_params` centipawns.
pub(crate) struct Gauge<'a> {
    pub(crate) probe: Vec<&'a FeatureRecord>,
    pub(crate) reference: f64,
    pub(crate) applied: f64,
}

impl<'a> Gauge<'a> {
    pub(crate) fn measure(probe: &[&FeatureRecord], values: &[f64]) -> f64 {
        let lifted: Vec<f64> = values.iter().enumerate().map(|(i, &v)| v * Self::slot_scale(i, MEASURE_GAIN)).collect();
        probe.iter().map(|r| eval_record(r, &lifted).abs()).sum()
    }

    pub(crate) fn new(probe: Vec<&'a FeatureRecord>, defaults: &[f64]) -> Self {
        let reference = Self::measure(&probe, defaults);
        Self { probe, reference, applied: 1.0 }
    }

    /// Cold starts (`Zero`, `Random`) have no scale to hold mid-run and normalize on exit instead.
    pub(crate) fn holds(&self, values: &[f64]) -> bool {
        (Self::measure(&self.probe, values) - self.reference).abs() <= 1e-9 * self.reference
    }

    // Phase weights are interpolation coordinates (1.0); king_danger multiplies pressure² (1/f).
    fn slot_scale(i: usize, f: f64) -> f64 {
        if (LAYOUT.phase_offset..LAYOUT.phase_offset + LAYOUT.phase_len).contains(&i) {
            1.0
        } else if i == LAYOUT.king_danger_offset {
            f.recip()
        } else {
            f
        }
    }

    pub(crate) fn normalize(&self, values: &mut [f64]) -> f64 {
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

    pub(crate) fn restore(&mut self, values: &mut [f64], optimizer: &mut Lion, k_ctrl: &mut KController) {
        let factor = self.normalize(values);
        optimizer.rescale(|i| Self::slot_scale(i, factor));
        k_ctrl.rescale(factor);
        self.applied *= factor;
    }
}

/// Golden-section search on a unimodal `[lo, hi]` bracket; each step costs one `eval` call.
pub(crate) fn golden_search_k<F: Fn(f64) -> f64>(mut lo: f64, mut hi: f64, tol: f64, eval: F) -> f64 {
    assert!(lo < hi && tol > 0.0, "golden_search_k: invalid bracket [{lo}, {hi}] or tol {tol}");
    if hi - lo <= tol {
        return 0.5 * (lo + hi);
    }

    const INV_PHI: f64 = 0.618_033_988_749_894_9; // (√5 − 1) / 2
    let mut width = hi - lo;
    let (mut a, mut b) = (hi - INV_PHI * width, lo + INV_PHI * width);
    let (mut fa, mut fb) = (eval(a), eval(b));

    while width > tol {
        width *= INV_PHI;
        if fa < fb {
            (hi, b, fb) = (b, a, fa);
            a = hi - INV_PHI * width;
            fa = eval(a);
        } else {
            (lo, a, fa) = (a, b, fb);
            b = lo + INV_PHI * width;
            fb = eval(b);
        }
    }

    0.5 * (lo + hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        config::Init,
        engine::{Position, SoulEntry, eval_params},
        run::seed_values,
    };

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

    fn learned_k(momentum: f64) -> KController {
        KController {
            k: 0.01,
            k_ref: 0.005,
            mode: KMode::Learned { lr_mult: 0.001 },
            k_min: 0.001,
            k_max: 0.020,
            beta1: 0.9,
            beta2: 0.99,
            momentum,
        }
    }

    fn probe_records() -> Vec<FeatureRecord> {
        PROBE_FENS
            .iter()
            .map(|fen| FeatureRecord::from_entry(&SoulEntry::from_board(&Position::from_fen(fen), 0.5, Some(20))))
            .collect()
    }

    #[test]
    fn canonicalize_preserves_scores_and_collapses_offset() {
        let params = eval_params::collect_parameters();
        let queen = PieceType::Queen as usize;
        let table = LAYOUT.psqt_offset + queen * 2 * TABLE_SQUARES;
        let records = probe_records();

        let mut base = eval_params::default_values(&params);
        let mut shifted = base.clone();
        shifted[table..table + TABLE_SQUARES].iter_mut().for_each(|v| *v += 60.0);

        let mut folded = shifted.clone();
        canonicalize(&mut folded, &params);
        assert_ne!(folded, shifted);
        for (fen, rec) in PROBE_FENS.iter().zip(&records) {
            assert!((eval_record(rec, &shifted) - eval_record(rec, &folded)).abs() < 1e-9, "{fen}");
        }

        let mut drifted = base.clone();
        drifted[table..table + TABLE_SQUARES].iter_mut().for_each(|v| *v += 37.0);
        drifted[LAYOUT.material_offset + queen] -= 37.0;
        canonicalize(&mut base, &params);
        canonicalize(&mut drifted, &params);
        for (i, (want, got)) in base.iter().zip(&drifted).enumerate() {
            assert!((want - got).abs() < 1e-9, "slot {i}: {want} vs {got}");
        }

        // Half-unit tie settles in one pass rather than oscillating around -0.5.
        let mut tied = base;
        for (n, i) in (table..table + TABLE_SQUARES).enumerate() {
            tied[i] = if n < TABLE_SQUARES / 2 { -1.0 } else { 0.0 };
        }
        canonicalize(&mut tied, &params);
        let once = tied.clone();
        canonicalize(&mut tied, &params);
        assert_eq!(tied, once);
    }

    #[test]
    fn gauge_restores_scale_k_and_curvature() {
        let params = eval_params::collect_parameters();
        let mut values = eval_params::default_values(&params);
        let curve = LAYOUT.king_danger_offset;
        values[curve] = 64.0;
        let want = values.clone();

        let mut opt = Lion::new(values.len(), 0.9, 0.1, 0.0);
        opt.restore_momentum(&vec![0.5; values.len()]);
        let records = probe_records();
        let probe: Vec<&FeatureRecord> = records.iter().collect();

        let mut gauge = Gauge::new(probe.clone(), &values);
        assert!(gauge.holds(&values));

        let mut k_ctrl = KController { k: 0.004, k_ref: 0.004, k_min: 0.0001, k_max: 1.0, ..learned_k(0.0) };
        for (i, v) in values.iter_mut().enumerate() {
            if !(LAYOUT.phase_offset..LAYOUT.phase_offset + LAYOUT.phase_len).contains(&i) {
                *v *= if i == curve { 1.0 / 1.23 } else { 1.23 };
            }
        }
        gauge.restore(&mut values, &mut opt, &mut k_ctrl);
        assert!((Gauge::measure(&probe, &values) - gauge.reference).abs() < 1e-6 * gauge.reference);
        for (i, (got, expect)) in values.iter().zip(&want).enumerate() {
            assert!((got - expect).abs() < 1e-6 * expect.abs().max(1.0), "slot {i}: {got} vs {expect}");
        }
        assert!((k_ctrl.k() / (0.004 * 1.23) - 1.0).abs() < 1e-6);
        assert!((opt.momentum()[0] / (0.5 * 1.23) - 1.0).abs() < 1e-6);
        assert!((gauge.applied * 1.23 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cold_start_skips_in_run_gauge() {
        let params = eval_params::collect_parameters();
        let records = probe_records();
        let gauge = Gauge::new(records.iter().collect(), &eval_params::default_values(&params));
        for init in [Init::Zero, Init::Random] {
            assert!(!gauge.holds(&seed_values(&params, init, 7)), "{init:?}");
        }
        assert!(gauge.holds(&seed_values(&params, Init::Default, 7)));
    }

    #[test]
    fn learned_k_zero_gradient() {
        let mut flat = learned_k(0.0);
        for _ in 0..100 {
            flat.on_batch(0.0, 256, 0.01, 1.0, 0.0);
        }
        assert_eq!(flat.k, 0.01);

        let mut moving = learned_k(0.5);
        moving.on_batch(0.0, 256, 0.01, 1.0, 0.0);
        assert!((moving.k - (0.01 - 0.01 * 0.001)).abs() < 1e-15);

        let mut decaying = learned_k(0.0);
        decaying.on_batch(0.0, 256, 0.01, 1.0, 0.00001);
        assert!((decaying.k - (0.01 - 0.01 * 0.001 * 0.00001 * 0.01)).abs() < 1e-16);
    }
}
