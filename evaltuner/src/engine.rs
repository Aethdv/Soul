//! Single import boundary between `evaltuner` and `soul`.

#[cfg(test)] pub(crate) use soul::core::board::Position;
pub(crate) use soul::{
    cli::Help,
    color,
    core::{
        defs::{Color, PieceType, TOTAL_PHASE},
        util::{fmt_comma, pct},
    },
    engine::{
        eval_params::{self, BLOCKS, Group, LAYOUT, PIECE_TABLES, TABLE_SQUARES, Tunable, collect_parameters, default_values},
        mobility::{OPEN_UNITY, piece_mobility_counts},
        wdl::sigmoid,
    },
    tools::dataset::{
        EpdEntry, FeatureRecord, GameScan, MAX_QUIET_ENDING, MIN_DECISIVE_ENDING, ReplayFilter, SoulEntry, accumulate_record_grad,
        eval_record, eval_record_full, flip_score, flip_wdl, parse_epd_str, parse_viri_file, scan_viri_games,
    },
};

#[cfg(test)]
mod tests {
    use std::fs;

    #[test]
    fn nothing_else_reaches_past_this_file() {
        let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/src");
        let mut offenders = Vec::new();

        for entry in fs::read_dir(dir).expect("read src/") {
            let entry = entry.expect("dir entry");
            let name = entry.file_name().to_string_lossy().into_owned();

            if name != "engine.rs"
                && name.ends_with(".rs")
                && fs::read_to_string(entry.path()).expect("read module").contains("soul::")
            {
                offenders.push(name);
            }
        }
        assert!(offenders.is_empty(), "modules reaching past engine.rs into soul: {offenders:?}");
    }
}
