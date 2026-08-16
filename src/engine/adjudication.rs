//! Score-based game adjudication for self-play and automated testing.
//!
//! Terminates games early when the score has settled to save compute:
//! - Win/loss: eval exceeds [`ADJ_WIN_SCORE`] for [`ADJ_WIN_PLIES`] consecutive plies.
//! - Draw: eval stays below [`ADJ_DRAW_SCORE`] for [`ADJ_DRAW_PLIES`] plies (after [`ADJ_DRAW_START_PLY`]).
//! - Resignation: eval exceeds [`ADJ_RESIGN_SCORE`] on any single ply.
//!
//! Because scores are side-to-move (STM) relative, a persistent advantage alternates sign
//! each ply (+eval on White's turn, -eval on Black's turn). Two identical signs in a row
//! mean the lead changed hands, which zeroes the win counter so an advantage blundered
//! away cannot adjudicate as a win a few plies later.

use crate::core::defs::{
    ADJ_DRAW_PLIES, ADJ_DRAW_SCORE, ADJ_DRAW_START_PLY, ADJ_RESIGN_SCORE, ADJ_WIN_PLIES, ADJ_WIN_SCORE, Color, GameOutcome,
};

/// Updates the counters and returns an outcome once one has held long enough.
pub fn check_adjudication(
    score: i32,
    last_score: i32,
    win_adj_counter: &mut usize,
    draw_adj_counter: &mut usize,
    stm: Color,
    ply: usize,
) -> Option<GameOutcome> {
    let abs_score = score.abs();

    if abs_score >= ADJ_RESIGN_SCORE {
        return Some(GameOutcome::from_stm_score(score, stm));
    }

    if abs_score >= ADJ_WIN_SCORE {
        // Matching signs mean the lead changed hands. The counter guard skips that test on
        // the entry ply, where `last_score` is a quiet eval with no lead to compare against.
        if (score > 0) == (last_score > 0) && *win_adj_counter > 0 {
            *win_adj_counter = 0;
        } else {
            *win_adj_counter += 1;
        }

        if *win_adj_counter >= ADJ_WIN_PLIES {
            return Some(GameOutcome::from_stm_score(score, stm));
        }
        *draw_adj_counter = 0;
    } else if abs_score < ADJ_DRAW_SCORE && ply >= ADJ_DRAW_START_PLY {
        *draw_adj_counter += 1;

        if *draw_adj_counter >= ADJ_DRAW_PLIES {
            return Some(GameOutcome::Draw);
        }
        *win_adj_counter = 0;
    } else {
        // Between the thresholds, or too early to call a draw.
        *win_adj_counter = 0;
        *draw_adj_counter = 0;
    }

    None
}
