#[cfg(feature = "rigs")] pub mod attack_table_bench;
pub mod bench;
#[cfg(feature = "rigs")]
#[cfg_attr(not(test), expect(dead_code, reason = "byteboard API: the rig reads part of it, its tests the rest"))]
mod byteboard;
#[cfg(feature = "datagen")] pub mod datagen;
#[cfg(feature = "dataset")] pub mod dataset;
pub mod genfens;
pub mod perft;
#[cfg(feature = "rigs")] pub mod speedtest;
pub mod spsa;
