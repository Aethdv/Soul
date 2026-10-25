//! Score-based self-play adjudication.

use crate::core::defs::{Color, GameOutcome};

const RESIGN_SCORE: i32 = 3000;
const WIN_SCORE: i32 = 1200;
const WIN_PLIES: usize = 8;
const DRAW_SCORE: i32 = 10;
const DRAW_PLIES: usize = 24;
const DRAW_START_PLY: usize = 80;

#[derive(Default)]
pub(crate) struct Adjudicator {
    leader: Option<Color>,
    win_plies: usize,
    draw_plies: usize,
}

impl Adjudicator {
    /// Scores are side-to-move relative; supply one score per ply.
    pub(crate) fn update(&mut self, score: i32, stm: Color, ply: usize) -> Option<GameOutcome> {
        let abs_score = score.abs();
        if abs_score >= RESIGN_SCORE {
            return Some(GameOutcome::from_stm_score(score, stm));
        }

        if abs_score >= WIN_SCORE {
            let leader = if score > 0 { stm } else { stm.opposite() };
            self.win_plies = if self.win_plies > 0 && self.leader != Some(leader) { 0 } else { self.win_plies + 1 };
            self.leader = Some(leader);
            self.draw_plies = 0;

            return (self.win_plies >= WIN_PLIES).then(|| GameOutcome::from_stm_score(score, stm));
        }

        self.leader = None;
        self.win_plies = 0;
        self.draw_plies = if abs_score < DRAW_SCORE && ply >= DRAW_START_PLY { self.draw_plies + 1 } else { 0 };
        (self.draw_plies >= DRAW_PLIES).then_some(GameOutcome::Draw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn win_streak_tracks_the_leader_across_turns() {
        let mut adjudicator = Adjudicator::default();
        for ply in 0..WIN_PLIES {
            let stm = if ply % 2 == 0 { Color::White } else { Color::Black };
            let score = if stm == Color::White { WIN_SCORE } else { -WIN_SCORE };
            let expected = (ply + 1 == WIN_PLIES).then_some(GameOutcome::WhiteWins);
            assert_eq!(adjudicator.update(score, stm, ply), expected);
        }
    }

    #[test]
    fn leader_change_discards_the_first_qualifying_ply() {
        let mut adjudicator = Adjudicator::default();
        assert_eq!(adjudicator.update(WIN_SCORE, Color::White, 0), None);
        for offset in 0..=WIN_PLIES {
            let stm = if offset % 2 == 0 { Color::Black } else { Color::White };
            let score = if stm == Color::Black { WIN_SCORE } else { -WIN_SCORE };
            let expected = (offset == WIN_PLIES).then_some(GameOutcome::BlackWins);
            assert_eq!(adjudicator.update(score, stm, offset + 1), expected);
        }
    }

    #[test]
    fn draw_streak_starts_at_the_required_ply() {
        let mut adjudicator = Adjudicator::default();
        for ply in DRAW_START_PLY - 1..DRAW_START_PLY + DRAW_PLIES {
            let stm = if ply % 2 == 0 { Color::White } else { Color::Black };
            let expected = (ply + 1 == DRAW_START_PLY + DRAW_PLIES).then_some(GameOutcome::Draw);
            assert_eq!(adjudicator.update(0, stm, ply), expected);
        }
    }
}
