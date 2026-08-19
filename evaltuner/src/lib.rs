// Mathematical implementations heavily rely on standard linear algebra notation (x, y, z, w, m).
#![allow(clippy::many_single_char_names)]
// Tuner routines inherently thread many per-parameter scalars through one call.
#![allow(clippy::too_many_arguments)]

pub mod ablation;
pub mod assay;
pub mod config;
pub mod correlation;
pub mod curvature;
pub mod engine;
pub mod error;
pub mod fnv;
pub mod groups;
pub mod lion;
pub mod loader;
pub mod logger;
pub mod palette;
pub mod probes;
pub mod report;
pub mod run;
pub mod scale;
pub mod schedule;
pub mod seeds;
pub mod shuffle;
pub mod storage;
pub mod training;

pub use run::run;
pub use schedule::{LrScheduler, WdlScheduler};
