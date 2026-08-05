//! Score-based game adjudication for self-play and automated testing.
//!
//! A game ends early when its score has settled: held decisively for
//! `ADJ_WIN_PLIES` straight plies calls a win, held near-equal for
//! `ADJ_DRAW_PLIES` plies past `ADJ_DRAW_START_PLY` calls a draw.
//! Either beats playing on to checkmate or the 50-move timeout once
//! the outcome is already certain.
//!
//! Scores are STM-relative, so a persistent advantage shows up as alternating
//! signs: White sees +3000 on its turn, Black sees −3000 on the reply, one
//! result wearing two signs. [`check_adjudication`] reads that flip to tell a
//! real lead from a blunder that handed it back.

use crate::core::defs::{
    ADJ_DRAW_PLIES, ADJ_DRAW_SCORE, ADJ_DRAW_START_PLY, ADJ_RESIGN_SCORE, ADJ_WIN_PLIES, ADJ_WIN_SCORE, Color, GameOutcome,
};

/// Returns the outcome once the score has held stable long enough to call the
/// game, `None` while it's still settling. The counters carry that progress
/// between calls; a sign flip in the lead zeroes the win counter, so an advantage
/// blundered away never adjudicates as a win a few plies later.
pub fn check_adjudication(
    score: i32,
    last_score: i32,
    win_adj_counter: &mut usize,
    draw_adj_counter: &mut usize,
    stm: Color,
    ply: usize,
) -> Option<GameOutcome> {
    let abs_score = score.abs();
    // Resignation: a margin this extreme ends it on the spot.
    if abs_score >= ADJ_RESIGN_SCORE {
        return Some(GameOutcome::from_stm_score(score, stm));
    }

    // Decisive: a winning margin sustained across plies.
    if abs_score >= ADJ_WIN_SCORE {
        // A persistent lead alternates sign every ply (STM-relative); equal signs
        // on consecutive plies mean the lead changed hands, so zero the count.
        // The `> 0` guard skips that check on the entry ply, where last_score is
        // a quiet eval whose sign carries no real lead to compare.
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
        // Draw: a quiet margin sustained past the opening.
        *draw_adj_counter += 1;

        if *draw_adj_counter >= ADJ_DRAW_PLIES {
            return Some(GameOutcome::Draw);
        }
        *win_adj_counter = 0;
    } else {
        // No clear verdict: score between the thresholds, or too early to draw. Reset both.
        *win_adj_counter = 0;
        *draw_adj_counter = 0;
    }
    None
}
