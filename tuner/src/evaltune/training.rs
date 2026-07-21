use soul::core::{board::Position, defs::Color};

use crate::evaltune::{loader, tape::eval_f64};

/// Online 95th percentile estimation of gradient norms via SGD.
///
/// The estimation happens in log-space to ensure the threshold remains positive and
/// tracks the order-of-magnitude changes in norms as the learning rate decays.
pub struct GradientStats {
    p95: f64,
    alpha: f64,
    count: usize,
}

impl GradientStats {
    #[must_use]
    pub fn new(window: usize) -> Self {
        Self { p95: 1.0, alpha: 2.0 / (window as f64 + 1.0), count: 0 }
    }

    pub fn update(&mut self, norm: f64) {
        if self.count == 0 {
            self.p95 = norm.max(1e-6);
            self.count = 1;
            return;
        }
        self.count += 1;

        // ── Online 95th percentile estimation.
        // We move up by (1-p) when norm > p95, and down by p when norm < p95.
        // Balancing: 0.05 * 0.95 (up) + 0.95 * -0.05 (down) = 0.
        let step = if norm > self.p95 { 0.95 } else { -0.05 };

        // Update in log-space. Stays positive regardless of how
        // small norms get, and the multiplicative step scales with
        // the current magnitude. This eliminates the need for a floor
        // at initialization or late in training.
        self.p95 *= (self.alpha * step).exp();
    }

    /// Returns the estimated 95th percentile of recent gradient norms.
    ///
    /// Falls back to `default` until 10 observations are collected:
    /// estimating a distribution from fewer points is noise, not signal.
    #[must_use]
    pub fn clip_threshold(&self, default: f64) -> f64 {
        if self.count < 10 {
            return default;
        }
        self.p95.max(0.1)
    }
}

/// Scaling the sigmoid constant K
/// S(x) = 1 / (1 + exp(-K · x))
#[inline]
#[must_use]
pub fn sigmoid(score: f64, k: f64) -> f64 {
    // Clamp the exponent to avoid libm's extremely slow subnormal fallback path
    // for values between -708 and -744, which ignores CPU FTZ/DAZ flags.
    let x = (-k * score).clamp(-700.0, 700.0);
    1.0 / (1.0 + x.exp())
}

/// Read-only evaluation trait: can compute a score and knows the game result.
///
/// Implemented by both `Entry` (raw EPD) and `SoulEntry` (encoded); the ablation
/// tool is generic over it.
pub trait TunableData: Sync + Send {
    fn eval(&self, values: &[f64]) -> f64;
    fn result(&self) -> f64;
}

/// Training target for an encoded entry: the game result, blended toward the
/// search eval by instance-confidence WDL.
///
/// Near-zero search scores (low engine confidence) fall back to the game result;
/// high-magnitude scores trust the eval fully. `wdl_blend >= 1.0` bypasses the
/// instance scaling: the target is pure `sigmoid(score)`, for random-restart
/// data with no game outcome. 400 cp is the empirical confidence saturation point.
pub fn wdl_target(entry: &loader::SoulEntry, k: f64, wdl_blend: f64) -> f64 {
    const CONFIDENCE_THRESHOLD: f64 = 400.0;

    // i16::MAX sentinel = EPD data with no search score: pure outcome.
    if entry.score == i16::MAX {
        return f64::from(entry.result) / 2.0;
    }
    let score = f64::from(entry.score);

    let instance_blend = if wdl_blend >= 1.0 { 1.0 } else { wdl_blend * (score.abs() / CONFIDENCE_THRESHOLD).min(1.0) };

    let expected = sigmoid(score, k);
    // result in {0,1,2} → normalize to [0.0, 1.0] for sigmoid target.
    (1.0 - instance_blend).mul_add(f64::from(entry.result) / 2.0, instance_blend * expected)
}

impl TunableData for loader::SoulEntry {
    /// Evaluation via FEN round-trip: valid but slow.
    /// Production code uses `eval_record` with a packed `FeatureRecord`.
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 {
        let board = Position::from_fen(&self.to_fen());
        eval_f64(&board, values)
    }

    #[inline]
    fn result(&self) -> f64 {
        f64::from(self.result) / 2.0
    }
}

impl TunableData for loader::Entry {
    #[inline]
    fn eval(&self, values: &[f64]) -> f64 {
        eval_f64(&self.board, values)
    }

    #[inline]
    fn result(&self) -> f64 {
        // EPD results are from White's perspective.
        // The eval produces a score relative to the side-to-move,
        // so we flip for Black.
        if self.board.stm == Color::Black { 1.0 - self.result } else { self.result }
    }
}
