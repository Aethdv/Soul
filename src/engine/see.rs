//! Static Exchange Evaluation.
//!
//! `see_ge(pos, mv, threshold)` answers in time linear in the attacker
//! count: will the moving side net at least `threshold` material if both
//! sides recapture optimally on `mv`'s destination?
//!
//! Used by qsearch to prune losing captures, by the picker for the
//! good/bad-capture split, and by search for SEE pruning and ProbCut.
//!
//! One `balance` integer, perspective-flipped per recapture: no scratch
//! array, no post-loop minimax pass.

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

/// SEE's own scale, deliberately not the eval's `MG_MATERIAL`. Shifting a constant
/// from `material[pt]` into every `PSQT[pt][sq]` leaves the eval unchanged, so
/// the tuner can move material anywhere at no cost to its loss; the thresholds
/// callers pass are in these units and need them to hold still.
///
/// The king keeps its zero: a chain that reaches it has already ended.
const SEE_VALUE: [i32; 8] = {
    let sp = SearchParams::new();
    let mut v = [0i32; 8];
    v[PieceType::Pawn.as_usize()] = sp.see_value_pawn;
    v[PieceType::Knight.as_usize()] = sp.see_value_knight;
    v[PieceType::Bishop.as_usize()] = sp.see_value_bishop;
    v[PieceType::Rook.as_usize()] = sp.see_value_rook;
    v[PieceType::Queen.as_usize()] = sp.see_value_queen;
    v
};

/// Capture order for the exchange loop: cheapest first, the king left out because
/// reaching it ends the chain rather than continuing it.
const LVA_ORDER: [PieceType; 5] = [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen];

/// Is the static exchange on `mv`'s destination square at least
/// `threshold` centipawns for the side making `mv`?
///
/// The pin scan belongs to the caller, so a loop of exchanges at one position
/// pays for it once. En passant, promotions, castling and revealed slider
/// x-rays are all modeled.
#[must_use]
pub fn see_ge(pos: &Position, mv: Move, threshold: i32, pins: &Pins) -> bool {
    // Castling captures nothing, and its to square holds our own rook, so the
    // exchange loop would price a trade that never happens.
    if mv.is_castling() {
        return threshold <= 0;
    }

    let from = mv.from();
    let to = mv.to();

    // Resolve the initial exchange into two scalars:
    //   gain     - material that moves into our column this ply
    //              (captured piece + promotion upgrade, if any)
    //   attacker - piece sitting on to after our move, whose value
    //              will be lost if the opponent recaptures
    let (gain, attacker) = if mv.is_en_passant() {
        (val(PieceType::Pawn), PieceType::Pawn)
    } else if let Some(promo) = mv.promo() {
        // The mover becomes promo; we earn the upgrade on top of
        // whatever (if anything) was captured on to.
        let captured = val(pos.piece_at(to));
        let upgrade = val(promo) - val(PieceType::Pawn);
        (captured + upgrade, promo)
    } else {
        (val(pos.piece_at(to)), pos.piece_at(from))
    };

    // What the side to move still has to gain to beat the previous player's outcome.
    let mut balance = gain - threshold;
    if balance < 0 {
        // Even with the free gift, the move doesn't reach threshold.
        return false;
    }

    balance = val(attacker) - balance;
    if balance <= 0 {
        // Even if the attacker is lost for nothing, the move still clears
        // threshold, so no deeper search is needed.
        return true;
    }

    // Rebuild occupancy with our attacker lifted off its square
    // (and for en passant also the victim pawn on to ^ 8),
    // then recompute the full attacker set from scratch, picking
    // up any slider that was previously blocked by our attacker.
    let mut occ = pos.occ ^ from.bitboard();
    if mv.is_en_passant() {
        occ ^= (to ^ 8).bitboard();
    }

    let excluded = pin_excluded(pins, to);
    let mut attackers = all_attackers_to(pos, to, occ) & occ & !excluded;

    let diag = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];
    let orth = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];

    let caller = pos.color_at(from);
    let mut stm = caller;
    let mut flipped = false;

    loop {
        stm = stm.opposite();
        let stm_attackers = attackers & pos.side_bb[stm];
        if stm_attackers.is_empty() {
            // The side to move has no attacker left, so the trade ends with
            // the previous mover keeping their net.
            return !flipped;
        }

        // Pick the least-valuable attacker.
        let attacker_of = |pt: PieceType| (stm_attackers & pos.role_bb[pt]).into_iter().next().map(|sq| (pt, sq));
        let Some((lva, lva_sq)) = LVA_ORDER.into_iter().find_map(attacker_of) else {
            // A king may not capture into check, so an opposing attacker still on
            // the square ends the chain instead of continuing it.
            let opp_attackers = attackers & pos.side_bb[stm.opposite()];
            return if opp_attackers.is_not_empty() { !flipped } else { flipped };
        };

        occ ^= lva_sq.bitboard();

        // Reveal x-rays through the newly vacated square:
        //   Pawn   - diagonal sliders behind the capturing pawn
        //   Bishop - diagonal sliders collinear with the bishop
        //   Rook   - orthogonal sliders collinear with the rook
        //   Queen  - both
        //   Knight - nothing (leaper, no ray behind it)
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

        // Negamax flip: the next player's running deficit is this attacker's value
        // minus the previous player's. Ties go to the caller, so once the perspective
        // has flipped, break-even no longer clears the threshold.
        balance = val(lva) - balance;
        if balance < i32::from(flipped) {
            return flipped;
        }
        flipped = !flipped;
    }
}

#[inline(always)]
fn val(pt: PieceType) -> i32 {
    SEE_VALUE[pt.as_usize()]
}

/// Pinned attackers that can't legally recapture on `to`: a pinned piece moves
/// only along its pin ray, so it reaches `to` only when `to` lies on that ray.
#[inline]
fn pin_excluded(pins: &Pins, to: Square) -> Bitboard {
    (pins.blockers(Color::White) & !line_bb(pins.king(Color::White), to))
        | (pins.blockers(Color::Black) & !line_bb(pins.king(Color::Black), to))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{core::board::Position, engine::movegen::gen_legal_moves};

    const P: i32 = SEE_VALUE[PieceType::Pawn.as_usize()];
    const N: i32 = SEE_VALUE[PieceType::Knight.as_usize()];
    const R: i32 = SEE_VALUE[PieceType::Rook.as_usize()];
    const Q: i32 = SEE_VALUE[PieceType::Queen.as_usize()];

    #[test]
    fn the_exchange_scale_holds_still() {
        assert_eq!(SEE_VALUE[..6], [92, 373, 372, 568, 1160, 0]);
    }

    /// Resolve a UCI move string against the legal move list.
    fn legal_move(pos: &Position, uci: &str) -> Move {
        for mv in gen_legal_moves(pos).iter() {
            if mv.to_uci(pos.is_frc) == uci {
                return *mv;
            }
        }
        panic!("move {uci} not legal in {}", pos.as_fen());
    }

    #[track_caller]
    fn assert_see(fen: &str, uci: &str, expected: i32) {
        let pos = Position::from_fen(fen);
        let mv = legal_move(&pos, uci);
        let pins = Pins::new(&pos);
        assert!(see_ge(&pos, mv, expected, &pins), "SEE({uci}) ≥ {expected} should hold (claimed value: {expected})\n  fen: {fen}",);
        assert!(
            !see_ge(&pos, mv, expected + 1, &pins),
            "SEE({uci}) ≥ {} should fail (claimed value: {expected})\n  fen: {fen}",
            expected + 1,
        );
    }

    #[test]
    fn rook_takes_undefended_pawn() {
        assert_see("7k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", P);
    }

    #[test]
    fn rook_takes_pawn_defended_by_rook() {
        assert_see("4r2k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", P - R);
    }

    #[test]
    fn pawn_takes_knight_undefended() {
        assert_see("7k/8/8/3n4/4P3/8/8/7K w - - 0 1", "e4d5", N);
    }

    #[test]
    fn pawn_takes_knight_defended_by_pawn() {
        assert_see("7k/8/2p5/3n4/4P3/8/8/7K w - - 0 1", "e4d5", N - P);
    }

    #[test]
    fn queen_takes_pawn_defended_by_pawn() {
        // catastrophic, don't do it
        assert_see("7k/8/8/8/2p5/3p4/4Q3/7K w - - 0 1", "e2d3", P - Q);
    }

    #[test]
    fn knight_takes_knight_defended_by_knight() {
        assert_see("7k/4n3/8/3n4/8/2N5/8/7K w - - 0 1", "c3d5", 0);
    }

    #[test]
    fn rook_battery_xray_chain() {
        // (W) Re2xPe5, (B) Re7xRe5, (W) Re1xRe5 via x-ray, (B) Re8xRe5 via x-ray.
        assert_see("4r3/4r2k/8/4p3/8/8/4R3/4R2K w - - 0 1", "e2e5", P - R);
    }

    #[test]
    fn en_passant_undefended() {
        assert_see("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", P);
    }

    #[test]
    fn en_passant_defended_by_knight() {
        assert_see("1n2k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", 0);
    }

    #[test]
    fn quiet_queen_promotion_undefended() {
        assert_see("7k/4P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", Q - P);
    }

    #[test]
    fn quiet_queen_promotion_defended_by_knight() {
        assert_see("7k/2n1P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", -P);
    }

    #[test]
    fn capture_promotion_undefended() {
        // PxN
        assert_see("4n2k/3P4/8/8/8/8/8/4K3 w - - 0 1", "d7e8q", N + Q - P);
    }

    #[test]
    fn pinned_defender_cannot_recapture() {
        // RxP has no recapture
        assert_see("8/8/k7/4p2R/2n5/8/4B3/6K1 w - - 0 1", "h5e5", P);
    }
}
