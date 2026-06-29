//! Static Exchange Evaluation.
//!
//! `see_ge(pos, mv, threshold)` answers in time linear in the attacker
//! count: will the moving side net at least `threshold` material if both
//! sides recapture optimally on `mv`'s destination?
//!
//! Used in qsearch to prune losing captures; available to main search for
//! the good/bad-capture split, SEE pruning, and ProbCut.
//!
//! One `balance` integer, perspective-flipped per recapture: no scratch
//! array, no post-loop minimax pass. Move type is dispatched once at
//! entry; captures, en passant, promotions, and castling each set their
//! own initial balance before sharing the exchange loop. The king carries
//! zero material: if it would capture while an opponent attacker remains,
//! the chain stops short: illegal recapture, not a trade. A `us` bool
//! tracks who owns the trade; when flipped, break-even isn't good enough.

use crate::{
    core::{
        board::{
            Position,
            attacks::{all_attackers_to, pinned_pieces},
            bitboard::{atk_bishop, atk_rook, line_bb},
        },
        defs::{Bitboard, Color, PieceType, Square},
        moves::Move,
    },
    engine::eval_params::MG_MATERIAL,
};

/// `King = 0` because the king is never captured.
const SEE_VALUE: [i32; 8] = {
    let mut v = [0i32; 8];
    v[PieceType::Pawn.as_usize()] = MG_MATERIAL[0];
    v[PieceType::Knight.as_usize()] = MG_MATERIAL[1];
    v[PieceType::Bishop.as_usize()] = MG_MATERIAL[2];
    v[PieceType::Rook.as_usize()] = MG_MATERIAL[3];
    v[PieceType::Queen.as_usize()] = MG_MATERIAL[4];
    v
};

/// Is the static exchange on `mv`'s destination square at least
/// `threshold` centipawns for the side making `mv`?
///
/// Correctly models en passant (the victim pawn lives on `to ^ 8`),
/// promotion (the pawn transforms into the promoted piece for the rest
/// of the trade, and the side earns the `promo − pawn` upgrade),
/// castling (materially neutral; legality already guaranteed by
/// movegen), and revealed slider x-rays through vacated squares.
#[must_use]
pub fn see_ge(pos: &Position, mv: Move, threshold: i32) -> bool {
    // Castling is the special case the exchange loop would mishandle:
    // the king and the rook both land on movegen-verified-safe squares,
    // and no capture is involved. Material impact is zero, so any
    // non-positive threshold trivially holds.
    if mv.is_castling() {
        return threshold <= 0;
    }

    let from = mv.from();
    let to = mv.to();

    // Resolve the initial exchange into two scalars:
    //   gain      - material that moves into our column this ply
    //               (captured piece + promotion upgrade, if any)
    //   attacker  - piece sitting on to after our move, whose value
    //               will be lost if the opponent recaptures
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

    // balance is the amount the side to move still needs to gain to
    // beat the previous player's outcome. After the first assignment
    // it represents our caller's deficit relative to threshold; after
    // every loop iteration the sign flips via balance = val(lva) - balance,
    // so the same variable tracks both sides.
    let mut balance = gain - threshold;
    if balance < 0 {
        // Even with the free gift, the move doesn't reach threshold.
        return false;
    }

    balance = val(attacker) - balance;
    if balance <= 0 {
        // Even after an optimal recapture, the move still meets
        // threshold, no deeper search needed.
        return true;
    }

    // Rebuild occupancy with our attacker removed from from
    // (and for en passant also the victim pawn on to ^ 8),
    // then recompute the full attacker set from scratch, picking
    // up any slider that was previously blocked by our attacker.
    let mut occ = pos.occ ^ from.bitboard();
    if mv.is_en_passant() {
        occ ^= (to ^ 8).bitboard();
    }

    // Past the early exits, so a trivial exchange skips the pin scan.
    let excluded = pinned_excluded(pos, to);
    let mut attackers = all_attackers_to(pos, to, occ) & occ & !excluded;

    let diag = pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen];
    let orth = pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen];

    let caller = pos.color_at(from);
    let mut stm = caller;
    let mut us = false;

    loop {
        stm = stm.opposite();
        let mine = attackers & pos.side_bb[stm];

        if mine.is_empty() {
            // The side to move has no attacker left, so the trade ends
            // with the previous mover keeping their net. That side is
            // the caller if us has been flipped an even number of
            // times, i.e. if us == false.
            return !us;
        }

        // Pick the least-valuable attacker. A match cascade is the
        // cleanest shape: each arm does exactly the work its piece
        // type requires, and the compiler turns the chain into a tight
        // priority-decoder on the bitboards.
        let (lva, lva_sq) = if let Some(sq) = lsb_of(mine & pos.role_bb[PieceType::Pawn]) {
            (PieceType::Pawn, sq)
        } else if let Some(sq) = lsb_of(mine & pos.role_bb[PieceType::Knight]) {
            (PieceType::Knight, sq)
        } else if let Some(sq) = lsb_of(mine & pos.role_bb[PieceType::Bishop]) {
            (PieceType::Bishop, sq)
        } else if let Some(sq) = lsb_of(mine & pos.role_bb[PieceType::Rook]) {
            (PieceType::Rook, sq)
        } else if let Some(sq) = lsb_of(mine & pos.role_bb[PieceType::Queen]) {
            (PieceType::Queen, sq)
        } else {
            // Only the king is left to capture. If the opposing side
            // still has any attacker on to, the king capture would
            // move the king into check, illegal, so the chain stops
            // here and the previous side's net stands.
            let opp = attackers & pos.side_bb[stm.opposite()];
            return if opp.is_not_empty() { !us } else { us };
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

        // Negamax flip: the next player's running deficit is this
        // attacker's value minus the previous player's deficit.
        balance = val(lva) - balance;

        // Asymmetric break encoding the tie-break rule. See module docs.
        if balance < i32::from(us) {
            return us;
        }

        us = !us;
    }
}

#[inline(always)]
fn val(pt: PieceType) -> i32 {
    SEE_VALUE[pt.as_usize()]
}

/// LSB of a bitboard, or `None` if empty. Separates the emptiness test
/// from the LSB extraction so the `if let` chain in `see_ge` stays tight.
#[inline(always)]
fn lsb_of(bb: Bitboard) -> Option<Square> {
    if bb.is_empty() { None } else { Some(bb.lsb()) }
}

/// Pinned attackers that can't legally recapture on `to`.
///
/// A pinned piece moves only along its pin ray, so it can recapture on `to`
/// only when `to` lies on that ray.
///
/// Pins are read once against the pre-exchange occupancy, so a pin the trade
/// later breaks still excludes its piece. A known approximation.
fn pinned_excluded(pos: &Position, to: Square) -> Bitboard {
    let mut excluded = Bitboard(0);

    for color in [Color::White, Color::Black] {
        let pinned = pinned_pieces(pos, color);

        if pinned.is_not_empty() {
            let king_sq = pos.pieces(PieceType::King, color).lsb();
            excluded |= pinned & !line_bb(king_sq, to);
        }
    }

    excluded
}

#[cfg(test)]
mod tests {
    //! Static test suite for SEE.
    //!
    //! Each case asserts a boundary: SEE-ge passes for `expected` and
    //! fails for `expected + 1`. Together they pin down the exact SEE
    //! value, so any algorithmic regression (wrong piece value, missed
    //! x-ray, mishandled EP square, broken negamax flip) collapses one
    //! of these cases with a one-line failure.

    use super::*;
    use crate::{
        core::board::Position,
        engine::{eval_params::MG_MATERIAL, movegen::gen_legal_moves},
    };

    macro_rules! pc {
        ($($n:ident = $p:expr;)*) => {
            $(const fn $n() -> i32 { MG_MATERIAL[$p as usize] })*
        };
    }

    pc! {
        p = PieceType::Pawn;
        n = PieceType::Knight;
        r = PieceType::Rook;
        q = PieceType::Queen;
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

    /// Pin SEE to an exact value: ge holds at `expected`, fails at `expected + 1`.
    #[track_caller]
    fn assert_see(fen: &str, uci: &str, expected: i32) {
        let pos = Position::from_fen(fen);
        let mv = legal_move(&pos, uci);
        assert!(see_ge(&pos, mv, expected), "SEE({uci}) ≥ {expected} should hold (claimed value: {expected})\n  fen: {fen}",);
        assert!(
            !see_ge(&pos, mv, expected + 1),
            "SEE({uci}) ≥ {} should fail (claimed value: {expected})\n  fen: {fen}",
            expected + 1,
        );
    }

    #[test]
    fn rook_takes_undefended_pawn() {
        // RxP, no recapture → +P
        assert_see("7k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", p());
    }

    #[test]
    fn rook_takes_pawn_defended_by_rook() {
        // RxP, RxR → +P − R
        assert_see("4r2k/8/8/4p3/8/8/8/4R2K w - - 0 1", "e1e5", p() - r());
    }

    #[test]
    fn pawn_takes_knight_undefended() {
        // PxN → +N
        assert_see("7k/8/8/3n4/4P3/8/8/7K w - - 0 1", "e4d5", n());
    }

    #[test]
    fn pawn_takes_knight_defended_by_pawn() {
        // PxN, PxP → +N − P
        assert_see("7k/8/2p5/3n4/4P3/8/8/7K w - - 0 1", "e4d5", n() - p());
    }

    #[test]
    fn queen_takes_pawn_defended_by_pawn() {
        // QxP, PxQ → +P − Q (catastrophic)
        assert_see("7k/8/8/8/2p5/3p4/4Q3/7K w - - 0 1", "e2d3", p() - q());
    }

    #[test]
    fn knight_takes_knight_defended_by_knight() {
        // NxN, NxN → break-even
        assert_see("7k/4n3/8/3n4/8/2N5/8/7K w - - 0 1", "c3d5", 0);
    }

    #[test]
    fn rook_battery_xray_chain() {
        // (W) Re2xPe5, (B) Re7xRe5, (W) Re1xRe5 via x-ray, (B) Re8xRe5 via x-ray.
        // Sequence: +P − R + R − R = P − R
        assert_see("4r3/4r2k/8/4p3/8/8/4R3/4R2K w - - 0 1", "e2e5", p() - r());
    }

    #[test]
    fn en_passant_undefended() {
        // EP capture, no recapture → +P
        assert_see("4k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", p());
    }

    #[test]
    fn en_passant_defended_by_knight() {
        // EP capture, knight recaptures on c6 → break-even
        assert_see("1n2k3/8/8/2pP4/8/8/8/4K3 w - c6 0 1", "d5c6", 0);
    }

    #[test]
    fn quiet_queen_promotion_undefended() {
        // Pawn becomes queen on an empty square: +Q − P
        assert_see("7k/4P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", q() - p());
    }

    #[test]
    fn quiet_queen_promotion_defended_by_knight() {
        // Promote (+Q − P), opponent captures the new queen (− Q) → − P
        assert_see("7k/2n1P3/8/8/8/8/8/7K w - - 0 1", "e7e8q", -p());
    }

    #[test]
    fn capture_promotion_undefended() {
        // PxN with promotion: +N + (Q − P)
        assert_see("4n2k/3P4/8/8/8/8/8/4K3 w - - 0 1", "d7e8q", n() + q() - p());
    }

    #[test]
    fn pinned_defender_cannot_recapture() {
        // Nc4 guards e5 but is pinned to Ka6 by Be2, so RxP has no recapture
        assert_see("8/8/k7/4p2R/2n5/8/4B3/6K1 w - - 0 1", "h5e5", p());
    }
}
