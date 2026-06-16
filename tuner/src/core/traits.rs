/// Feedback type from evaluation: determines what the optimizer receives.
#[derive(Clone, Debug)]
pub enum Feedback {
    /// For population-based methods.
    /// Higher score is better.
    /// `std_err` is the uncertainty (sigma) of the score.
    Scalar { score: f64, raw_score: f64, std_err: f64 },

    /// Lower loss is better: gradient points toward steepest ascent.
    Gradient { loss: f64, grad: Vec<f64> },
}

pub struct TuningConfig {
    pub epochs: usize,
    pub checkpoint_path: Option<String>,
    pub resume: bool,
    pub verbose: bool,
}

impl Default for TuningConfig {
    fn default() -> Self {
        Self { epochs: 100, checkpoint_path: None, resume: false, verbose: true }
    }
}
