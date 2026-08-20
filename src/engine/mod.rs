//! Search, evaluation, and time management.

pub mod adjudication;
pub mod autograd;
pub mod combiner;
#[cfg(feature = "corrstats")] pub mod corrstats;
pub mod eval;
pub mod eval_params;
pub mod history;
pub mod mobility;
pub mod movegen;
pub mod movepicker;
#[cfg(feature = "mvpstats")] pub mod mvpstats;
pub mod search;
pub mod search_params;
pub mod see;
pub mod term;
pub mod tm;
pub mod tt;
pub mod tui;
pub mod wdl;
