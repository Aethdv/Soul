//! High-throughput self-play engine for dataset generation.

pub mod config;
pub mod run;
pub mod stats;
pub mod worker;

pub use run::run;
