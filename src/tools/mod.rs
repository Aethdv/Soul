//! Development utilities: benchmarking, perft, and the optional dataset and measurement tools.

pub mod bench;
#[cfg(feature = "rigs")] pub mod byteboard;
#[cfg(feature = "datagen")] pub mod datagen;
#[cfg(feature = "dataset")] pub mod dataset;
#[cfg(feature = "datagen")] pub mod genfens;
#[cfg(feature = "rigs")] pub mod measure;
pub mod perft;
#[cfg(feature = "rigs")] pub mod speedtest;
#[cfg(feature = "rigs")] pub mod votecheck;
