//! Static Exchange Evaluation.

use crate::{
    core::{
        board::{
            Position,
            attacks::{Pins, all_attackers_to},
            bitboard::{atk_bishop, atk_rook, line_bb},
        },
        defs::{Bitboard, Color, PieceType, Square},
        moves::Move,
    },
    engine::search_params::SearchParams,
};

pub(crate) const fn see_values(sp: &SearchParams) -> [i32; 8] {
    let mut v = [0i32; 8];
    v[PieceType::Pawn.as_usize()] = sp.see_value_pawn;
    v[PieceType::Knight.as_usize()] = sp.see_value_knight;
    v[PieceType::Bishop.as_usize()] = sp.see_value_bishop;
    v[PieceType::Rook.as_usize()] = sp.see_value_rook;
    v[PieceType::Queen.as_usize()] = sp.see_value_queen;
    v
}

const LVA_ORDER: [PieceType; 5] = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen];

/// Whether the mover nets at least `threshold` from the exchange after `mv`.
#[must_use]
pub(crate) fn see_ge(pos: &Position, mv: Move, threshold: i32, pins: &Pins, values: &[i32; 8]) -> bool {
    if mv.is_castling() {
        return threshold <= 0;
    }

    let from = mv.from();
    let to = mv.to();

    // gain includes the capture and promotion gain; attacker is the piece
    // left on `to`, whose value the next recapture would win.
    let (gain, attacker) = if mv.is_en_passant() {
        (values[PieceType::Pawn], PieceType::Pawn)
    } else if let Some(promo) = mv.promo() {
        let captured = values[pos.piece_at(to)];
        let upgrade = values[promo] - values[PieceType::Pawn];
        (captured + upgrade, promo)
    } else {
        (values[pos.piece_at(to)], pos.piece_at(from))
    };

    let mut balance = gain - threshold;
    if balance < 0 {
        return false; // even without a recapture, the move falls short of threshold.
    }

    balance = values[attacker] - balance;
    if balance <= 0 {
        return true; // even if the attacker is lost for nothing, the move still clears threshold.
    }

    // Attacks to `to` don't depend on its occupancy. Vacating `from` and,
    // for en passant, the captured pawn's square can reveal sliders.
    let mut occ = pos.occ ^ from.bitboard();
    if mv.is_en_passant() {
        occ ^= (to ^ 8).bitboard();
    }

    let excluded = pin_excluded(pins, to);
    let mut attackers = all_attackers_to(pos, to, occ) & occ & !excluded;

    let diag = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];
    let orth = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];

    let mut stm = pos.color_at(from);
    let mut flipped = false;

    loop {
        stm = stm.opposite();
        let stm_attackers = attackers & pos.side_bb[stm];
        if stm_attackers.is_empty() {
            // The previous mover keeps their net gain.
            return !flipped;
        }

        let attacker_of = |pt: PieceType| (stm_attackers & pos.role_bb[pt]).into_iter().next().map(|sq| (pt, sq));
        let Some((lva, lva_sq)) = LVA_ORDER.into_iter().find_map(attacker_of) else {
            // A king can recapture only if the target square is undefended.
            let opp_attackers = attackers & pos.side_bb[stm.opposite()];
            return if opp_attackers.is_not_empty() { !flipped } else { flipped };
        };

        occ ^= lva_sq.bitboard();

        // Removing the attacker can uncover a slider behind it.
        // Pawn recaptures uncover diagonal sliders, just like bishops.
        match lva {
            PieceType::Pawn | PieceType::Bishop => {
                attackers |= atk_bishop(to, occ) & diag;
            },
            PieceType::Rook => {
                attackers |= atk_rook(to, occ) & orth;
            },
            PieceType::Queen => {
                attackers |= (atk_bishop(to, occ) & diag) | (atk_rook(to, occ) & orth);
            },
            _ => {},
        }
        attackers &= occ & !excluded;

        // Flip the balance for the next side's perspective. Ties favor the
        // original mover, so the cutoff alternates between < 0 and <= 0.
        balance = values[lva] - balance;
        if balance < i32::from(flipped) {
            return flipped;
        }

        flipped = !flipped;
    }
}

/// Exclude pinned attackers that cannot legally recapture on `to`.
/// A pinned piece can recapture only along its pin line.
#[inline]
fn pin_excluded(pins: &Pins, to: Square) -> Bitboard {
    (pins.pinned(Color::White) & !line_bb(pins.king(Color::White), to))
        | (pins.pinned(Color::Black) & !line_bb(pins.king(Color::Black), to))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::movegen::gen_legal_moves;

    const DEFAULTS: SearchParams = SearchParams::new();
    const VALUES: [i32; 8] = see_values(&DEFAULTS);

    const P: i32 = VALUES[PieceType::Pawn.as_usize()];
    const N: i32 = VALUES[PieceType::Knight.as_usize()];
    const R: i32 = VALUES[PieceType::Rook.as_usize()];
    const Q: i32 = VALUES[PieceType::Queen.as_usize()];

    fn legal_move(pos: &Position, uci: &str) -> Move {
        for mv in gen_legal_moves(pos).iter() {
            if mv.to_uci(pos.frc_notation) == uci {
                return *mv;
            }
        }
        panic!("move {uci} not legal in {}", pos.to_fen());
    }

    #[track_caller]
    fn assert_see(fen: &str, uci: &str, expected: i32) {
        let pos = Position::from_fen(fen);
        let mv = legal_move(&pos, uci);
        let pins = Pins::new(&pos);
        assert!(see_ge(&pos, mv, expected, &pins, &VALUES), "SEE({uci}) ≥ {expected} should hold\n  fen: {fen}");
        assert!(
            !see_ge(&pos, mv, expected + 1, &pins, &VALUES),
            "SEE({uci}) ≥ {} should fail (claimed value: {expected})\n  fen: {fen}",
            expected + 1,
        );
    }

    #[test]
    fn rook_takes_undefended_pawn() { assert_see("7k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", P); }

    #[test]
    fn rook_takes_pawn_defended_by_rook() { assert_see("4r2k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", P - R); }

    #[test]
    fn rook_recaptures_knight() { assert_see("k7/3n4/8/4p2R/8/8/8/4R2K w - - 0 1", "e1e5", P - R + N); }

    #[test]
    fn pawn_takes_knight_undefended() { assert_see("7k/8/8/3n4/4P3/8/8/7K w - - 0 1", "e4d5", N); }

    #[test]
    fn pawn_takes_knight_defended_by_pawn() { assert_see("7k/8/2p5/3n4/4P3/8/8/7K w - - 0 1", "e4d5", N - P); }

    #[test]
    fn pawn_takes_queen_defended_by_king() { assert_see("8/8/2k5/3q4/2P5/8/8/K7 w - - 0 1", "c4d5", Q - P); }

    #[test]
    fn queen_takes_pawn_defended_by_pawn() { assert_see("7k/8/8/8/2p5/3p4/4Q3/7K w - - 0 1", "e2d3", P - Q); }

    #[test]
    fn knight_takes_knight_defended_by_knight() { assert_see("7k/4n3/8/3n4/8/2N5/8/7K w - - 0 1", "c3d5", 0); }

    #[test]
    fn rook_battery_xray_chain() { assert_see("4r3/4r2k/8/4p3/8/8/4R3/4R2K w - - 0 1", "e2e5", P - R); }

    #[test]
    fn en_passant_undefended() { assert_see("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", P); }

    #[test]
    fn en_passant_defended_by_knight() { assert_see("1n2k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", 0); }

    #[test]
    fn quiet_queen_promotion_undefended() { assert_see("7k/4P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", Q - P); }

    #[test]
    fn quiet_queen_promotion_defended_by_knight() { assert_see("7k/2n1P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", -P); }

    #[test]
    fn capture_promotion_undefended() { assert_see("4n2k/3P4/8/8/8/8/8/4K3 w - - 0 1", "d7e8q", N + Q - P); }

    #[test]
    fn castling_captures_nothing() { assert_see("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 0); }

    #[test]
    fn pinned_defender_cannot_recapture() { assert_see("8/8/k7/4p2R/2n5/8/4B3/6K1 w - - 0 1", "h5e5", P); }

    #[test]
    fn king_cannot_recapture_defended_rook() { assert_see("8/8/5k2/4p3/3B4/4R3/8/7K w - - 0 1", "e3e5", P); }
}
