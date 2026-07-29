//! Everything the eval tuner takes from the engine.
//!
//! Every module here imports from this file, so the surface between the two is one list
//! and cannot grow without someone editing it. The test below is what makes that true
//! rather than intended.

pub use soul::{
    color::{self, Rgb, ansi_fg},
    core::{
        board::Position,
        defs::{Color, PieceType, TOTAL_PHASE},
    },
    engine::{
        eval_params::{
            self, BLOCKS, Block, Group, LAYOUT, PIECE_TABLES, TABLE_SQUARES, Tunable, collect_parameters, default_values,
        },
        wdl::sigmoid,
    },
    tools::dataset::{
        FeatureRecord, SoulEntry, accumulate_record_grad, eval_record, eval_record_full, load_encoded, parse_epd_str,
        parse_viri_file, save_encoded, tape::eval_f64,
    },
};

#[cfg(test)]
mod tests {
    /// Anything naming `soul::` outside this file has gone around the door.
    #[test]
    fn nothing_else_reaches_past_this_file() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src/evaltune");
        let mut offenders = Vec::new();

        for entry in std::fs::read_dir(dir).expect("the module's own directory") {
            let path = entry.expect("a readable entry").path();
            let name = path.file_name().unwrap_or_default().to_string_lossy().into_owned();

            if name == "engine.rs" || path.extension().is_none_or(|e| e != "rs") {
                continue;
            }

            if std::fs::read_to_string(&path).is_ok_and(|src| src.contains("soul::")) {
                offenders.push(name);
            }
        }

        assert!(offenders.is_empty(), "these reach past engine.rs into soul: {offenders:?}");
    }
}
