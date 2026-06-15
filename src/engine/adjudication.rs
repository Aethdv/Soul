//! Score-based game adjudication for self-play and automated testing.
//!
//! Decides when a game should be terminated early by score stability:
//! a position held decisively for `ADJ_WIN_PLIES` consecutive plies, or
//! held roughly equal for `ADJ_DRAW_PLIES` plies after `ADJ_DRAW_START_PLY`,
//! triggers a result rather than playing to checkmate or 50-move timeout.
//!
//! Scores are always STM-relative. If White is winning with score +3000
//! at White's turn, Black will see −3000 on their turn, same result, opposite sign.
//! The sign-flip detection logic in [`check_adjudication`] accounts for this.

use crate::core::defs::{
    ADJ_DRAW_PLIES, ADJ_DRAW_SCORE, ADJ_DRAW_START_PLY, ADJ_RESIGN_SCORE, ADJ_WIN_PLIES, ADJ_WIN_SCORE, Color, GameOutcome,
};

/// Universal adjudication engine.
///
/// Decides if a game should end early based on score stability or extreme values.
/// Tracking the sign of the score ensures that a sudden blunder flip correctly
/// resets the adjudication progress. :p
pub fn check_adjudication(
    score: i32,
    last_score: i32,
    win_adj_counter: &mut usize,
    draw_adj_counter: &mut usize,
    stm: Color,
    ply: usize,
) -> Option<GameOutcome> {
    let abs_score = score.abs();

    // 1. Hard resignation (instant)
    if abs_score >= ADJ_RESIGN_SCORE {
        return Some(GameOutcome::from_stm_score(score, stm));
    }

    // 2. Decisive adjudication
    if abs_score >= ADJ_WIN_SCORE {
        // Reset counter if the winning side flips.
        //
        // In an STM-relative search, scores are from the perspective of the side whose turn it is.
        // A persistent objective advantage means alternating signs across plies
        // (e.g. White sees +3000, next turn Black sees -3000).
        //
        // Therefore, if the signs are EQUAL across consecutive plies, it means the
        // side with the lead flipped (or the advantage blundered into a disadvantage).
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
        *win_adj_counter = 0;
        *draw_adj_counter = 0;
    }

    None
}
