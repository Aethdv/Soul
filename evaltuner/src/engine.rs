//! Everything the eval tuner takes from the engine.
//!
//! Every module here imports from this file, so the surface between the two is one list.
//! The test here enforces it.
//!
//! The dataset formats are in the list because `soul` owns the readers and the writer,
//! not because the eval needs them.

pub use soul::{
    cli::Help,
    color::{self, Rgb, ansi_fg},
    core::{
        board::Position,
        defs::{Color, PieceType, TOTAL_PHASE},
        util::{format_comma, pct},
    },
    engine::{
        eval_params::{
            self, BLOCKS, Block, Group, LAYOUT, PIECE_TABLES, TABLE_SQUARES, Tunable, collect_parameters, default_values,
        },
        wdl::sigmoid,
    },
    tools::dataset::{
        EpdEntry, FeatureRecord, GameScan, ReplayFilter, SoulEntry, accumulate_record_grad, eval_record, eval_record_full,
        flip_score, flip_wdl, parse_epd_str, parse_viri_file, scan_viri_games,
        tape::eval_f64,
        viri_format::{DECISIVE_ENDING, QUIET_ENDING},
    },
};

#[cfg(test)]
mod tests {
    #[test]
    fn nothing_else_reaches_past_this_file() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let mut offenders = Vec::new();
        for entry in std::fs::read_dir(dir).expect("the module's own directory") {
            let entry = entry.expect("a readable entry");
            let name = entry.file_name().to_string_lossy().into_owned();
            if name == "engine.rs" || !name.ends_with(".rs") {
                continue;
            }
            if std::fs::read_to_string(entry.path()).expect("a readable module").contains("soul::") {
                offenders.push(name);
            }
        }
        assert!(offeevaltuner/src/run.rsnders.is_empty(), "these reach past engine.rs into soul: {offenders:?}");
    }
}
