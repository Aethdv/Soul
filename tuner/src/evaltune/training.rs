use soul::core::{board::Position, defs::Color};

use crate::evaltune::{loader, tape, tape::eval_f64};

/// Adaptive gradient clipping based on exponentially weighted percentile estimation.
///
/// Rather than a fixed clip threshold or a sliding window (which requires sorting),
/// this uses Stochastic Gradient Descent (SGD) to maintain an O(1) estimate of the
/// 95th percentile of gradient norms.
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

        // Online 95th percentile estimation.
        // We move up by (1-p) when norm > p95, and down by p when norm < p95.
        // Balancing: 0.05 * 0.95 (up) + 0.95 * -0.05 (down) = 0.
        let step = if norm > self.p95 { 0.95 } else { -0.05 };

        // Update in log-space to handle the exponential decay of norms.
        self.p95 *= (self.alpha * step).exp();
    }

    /// Returns the estimated 95th percentile of recent gradient norms.
    ///
    /// Falls back to `default` until 10 observations are collected —
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
/// Implemented by both `Entry` (raw EPD) and `SoulEntry` (encoded).
/// This is the base layer — `TrainableEntry` extends it with gradient support.
pub trait TunableData: Sync + Send {
    fn eval(&self, values: &[f64]) -> f64;
    fn result(&self) -> f64;
}

/// Gradient-capable evaluation trait.
///
/// Adds `target()` (result with optional WDL blending), `eval_with_state()`
/// (evaluation that may capture internal state for gradient scatter), and
/// `accumulate_grad()` (scatter gradients into the parameter vector).
///
/// For raw EPD entries, this trait is **not used in the production loop** —
/// `eval_linear_grad` bypasses it entirely. It remains active for the
/// encoded `.soul.zst` pipeline.
pub trait TrainableEntry: TunableData {
    type GradState: Default + Send;

    fn target(&self, k: f64, wdl_blend: f64) -> f64;
    fn eval_with_state(&self, values: &[f64], state: &mut Self::GradState) -> f64;

    fn accumulate_grad(&self, values: &[f64], gradient: f64, grads: &mut [f64], state: &Self::GradState);
}

impl TunableData for loader::SoulEntry {
    /// Evaluation via FEN round-trip — valid but slow.
    /// Production code uses `eval_soul_cached` with `FeatureSlots`.
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

impl TrainableEntry for loader::SoulEntry {
    type GradState = ();

    #[inline]
    fn target(&self, k: f64, wdl_blend: f64) -> f64 {
        // i16::MAX sentinel = EPD data with no search score — pure outcome.
        if self.score == i16::MAX {
            return f64::from(self.result) / 2.0;
        }
        let score = f64::from(self.score);

        // Instance-Confidence WDL blending:
        // Scale the global `wdl_blend` based on the magnitude of the search score.
        // Near-zero scores (low engine confidence) fall back to the game result;
        // high-magnitude scores trust the search eval fully.
        //
        // wdl_blend >= 1.0 bypasses instance-confidence entirely — the target
        // is pure sigmoid(score). Used for random-restart data with no game outcome.
        // 400 cp is the empirical confidence saturation point.
        let confidence_threshold = 400.0;
        let instance_blend = if wdl_blend >= 1.0 { 1.0 } else { wdl_blend * (score.abs() / confidence_threshold).min(1.0) };

        let expected = sigmoid(score, k);
        // result in {0,1,2} → normalize to [0.0, 1.0] for sigmoid target.
        (1.0 - instance_blend).mul_add(f64::from(self.result) / 2.0, instance_blend * expected)
    }

    #[inline]
    fn eval_with_state(&self, values: &[f64], _: &mut Self::GradState) -> f64 {
        self.eval(values)
    }

    #[inline]
    fn accumulate_grad(&self, _values: &[f64], _gradient: f64, _grads: &mut [f64], _: &Self::GradState) {
        unimplemented!(
            "SoulEntry gradient accumulation requires FeatureSlots; production code uses the cached path via eval_soul_cached / accumulate_gradient_cached"
        );
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

impl TrainableEntry for loader::Entry {
    type GradState = Option<tape::DualEvalResult>;

    #[inline]
    fn target(&self, _k: f64, _wdl_blend: f64) -> f64 {
        self.result()
    }

    #[inline]
    fn eval_with_state(&self, values: &[f64], state: &mut Self::GradState) -> f64 {
        let result = tape::eval_dual_forward(&self.board, values);
        let val = result.score;
        *state = Some(result);
        val
    }

    #[inline]
    fn accumulate_grad(&self, _: &[f64], gradient: f64, grads: &mut [f64], state: &Self::GradState) {
        if let Some(result) = state {
            result.scatter_grads(gradient, grads);
        }
    }
}
