//! Pseudo-legal move generation and strict legality filtering.
//!
//! Generates moves in bulk via bitwise operations, then filters them
//! dynamically against the current board state's pins and checks.

use crate::core::{
    board::{
        B_OO_EMPTY, B_OOO_EMPTY, BLACK_OO, BLACK_OOO, CASTLE_B_KS, CASTLE_B_KS_CHECK, CASTLE_B_QS,
        CASTLE_B_QS_CHECK, CASTLE_W_KS, CASTLE_W_KS_CHECK, CASTLE_W_QS, CASTLE_W_QS_CHECK, Position,
        W_OO_EMPTY, W_OOO_EMPTY, WHITE_OO, WHITE_OOO,
        bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, between_bb, line_bb},
    },
    defs::{Bitboard, Color, Direction, FILE_A, FILE_H, PieceType, RANK_1, RANK_3, RANK_6, RANK_8, Square},
    moves::{Move, MoveList},
};

// Move Generation — pseudo-legal generation followed by legality filter.
//
// Why not generate legal moves directly?
// Bitboard tricks let us compute ALL pawn pushes in one shift,
// ALL knight hops in one lookup — bulk operations oblivious to legality.
// But pins, checks, and en-passant x-rays require per move reasoning.
// Generating too many and filtering beats being careful from the start.

/// Every strictly legal move in the position.
#[inline]
pub fn gen_legal_moves(board: &Position) -> MoveList {
    let mut legal = MoveList::new();
    let pseudo = gen_pseudo_moves(board);

    let stm = board.stm;
    let opp = stm.opposite();
    let k_bb = board.pieces(PieceType::King, stm);

    if k_bb.is_empty() {
        debug_assert!(false, "Position has no king for {stm:?}");
        return legal;
    }

    let ksq = k_bb.lsb();
    let pinned = board.king_blockers();
    let checkers = board.checkers();

    for &mv in &pseudo {
        if is_legal(board, mv, ksq, pinned, checkers, opp) {
            legal.push(mv);
        }
    }

    legal
}

/// Validates that a TT move is pseudo-legal for the current position.
///
/// TT hash collisions can produce moves from completely unrelated positions.
/// This checks: friendly piece on `from`, correct attack pattern for that
/// piece, and flag consistency (captures, promotions, castling, EP, double push).
/// Does NOT check legality (pins, checks) — that's `is_legal`'s job.
#[inline]
pub fn is_pseudo_legal(board: &Position, mv: Move) -> bool {

    let from = mv.from();
    let to = mv.to();
    let stm = board.stm;
    let us = board.side_bb[stm];
    let them = board.side_bb[stm.opposite()];
    let occ = board.occ;

    // Must have a friendly piece on the origin square.
    if !us.check_bit(from) {
        return false;
    }

    let piece = board.piece_at(from);

    // Castling — trust the flag; full validation happens in gen_castling / is_legal.
    if mv.is_castling() {
        return piece == PieceType::King;
    }

    // Destination must not hold a friendly piece.
    if us.check_bit(to) {
        return false;
    }

    // Capture flag consistency.
    let enemy_on_to = them.check_bit(to);
    if mv.is_capture() && !mv.is_en_passant() && !enemy_on_to {
        return false;
    }
    if !mv.is_capture() && enemy_on_to {
        return false;
    }

    // Promotions must come from a pawn.
    if mv.is_promotion() && piece != PieceType::Pawn {
        return false;
    }

    match piece {
        PieceType::Pawn => {
            let fwd = stm.forward_dir().delta();

            if mv.is_en_passant() {
                return board.en_passant == Some(to) && atk_pawn(from, stm).check_bit(to);
            }

            if mv.is_double_push() {
                let mid = Square((from.0 as i8 + fwd) as u8);
                return to.0 as i8 == from.0 as i8 + fwd * 2
                    && !occ.check_bit(mid)
                    && !occ.check_bit(to);
            }

            if mv.is_capture() {
                return atk_pawn(from, stm).check_bit(to);
            }

            // Single push.
            to.0 as i8 == from.0 as i8 + fwd && !occ.check_bit(to)
        },
        PieceType::Knight => atk_knight(from).check_bit(to),
        PieceType::Bishop => atk_bishop(from, occ).check_bit(to),
        PieceType::Rook => atk_rook(from, occ).check_bit(to),
        PieceType::Queen => (atk_bishop(from, occ) | atk_rook(from, occ)).check_bit(to),
        PieceType::King => atk_king(from).check_bit(to),
        _ => false,
    }
}

/// Legality — does this move leave our king safe?
///
/// Most moves pass trivially. The interesting cases:
///   • King moves    — destination must not be attacked.
///   • Double check  — only the king himself can escape.
///   • En passant    — removing two pawns from one rank can unmask a rook.
///   • Pinned pieces — may only slide along the pin ray.
///   • Single check  — must capture the checker or interpose.
#[inline(always)]
pub fn is_legal(
    board: &Position,
    mv: Move,
    ksq: Square,
    pinned: Bitboard,
    checkers: Bitboard,
    opp: Color,
) -> bool {
    let from = mv.from();
    let to = mv.to();

    // ── King moves ──
    if from == ksq {
        if mv.is_castling() {
            // Castling out of check is flatly illegal:
            // all other castling constraints were enforced during generation.
            return checkers.is_empty();
        }
        // is_attacked::<true> removes the king from occupancy so
        // sliding attackers see through his old square. Without this,
        // the king could "hide behind himself" — stepping backward
        // along a rook's file and believing the square is safe.
        return !board.is_attacked::<true>(to, opp, from.bitboard());
    }

    // ── Double check: only the king can escape ──
    if checkers.popcount() > 1 {
        return false;
    }

    // ── En passant ──
    // The one move where the captured piece isn't on the destination square,
    // requiring a special x-ray check.
    if mv.is_en_passant() {
        return is_ep_legal(board, mv, ksq, pinned, checkers, opp);
    }

    // ── Pinned piece: may only slide along its pin ray ──
    // line_bb(from, to) returns the entire line through both squares.
    // If the king isn't on that line, the piece is leaving the ray.
    if pinned.check_bit(from) && !line_bb(from, to).check_bit(ksq) {
        return false;
    }

    // ── Single check: capture the checker or interpose ──
    if checkers.is_empty() {
        return true;
    }

    let checker = checkers.lsb();
    to == checker || between_bb(ksq, checker).check_bit(to)
}

/// En passant creates a unique legality headache: two pawns vanish from
/// the same rank simultaneously (ours departs, theirs is captured).
/// If both were masking a rook or queen from our king along that rank,
/// the capture reveals a discovered check — the only move in chess that can
/// do this without involving the moving piece's own file.
///
/// We simulate the resulting occupancy and check for sliding x-rays.
#[inline]
fn is_ep_legal(
    board: &Position,
    mv: Move,
    ksq: Square,
    pinned: Bitboard,
    checkers: Bitboard,
    opp: Color,
) -> bool {
    let from = mv.from();
    let to = mv.to();

    // The captured pawn is exactly one rank behind the en passant destination square.
    // In our rank-major square encoding (index = rank · 8 + file), ±8 shifts by exactly one rank.
    // This is equivalent to the to ^ 8 trick used in update_accumulator.
    let cap_sq = Square(to.0 ^ 8);

    // If already in check, EP must resolve it:
    // capture the checker (the captured pawn IS the checker)
    // or block the ray.
    if checkers.is_not_empty() {
        let checker = checkers.lsb();
        if checker != cap_sq && !between_bb(ksq, checker).check_bit(to) {
            return false;
        }
    }

    // Pinned fleeing pawn: may only move along its pin ray.
    if pinned.check_bit(from) && !line_bb(from, to).check_bit(ksq) {
        return false;
    }

    // ── Horizontal discovered checks ──
    // En Passant is the only move where two pieces disappear from the same rank
    // simultaneously. If both were masking a horizontal sliding attack against
    // the king, the move is illegal.
    if ksq.rank() == from.rank() {
        let rq = board.pieces(PieceType::Rook, opp) | board.pieces(PieceType::Queen, opp);
        if rq.is_not_empty() {
            let rank_occ = board.occ ^ from.bitboard() ^ cap_sq.bitboard();
            if (atk_rook(ksq, rank_occ) & rq).is_not_empty() {
                return false;
            }
        }
    }

    // ── Diagonal discovered check via captured-pawn removal ──
    //
    // Unlike the horizontal case, the diagonal case involves only the captured pawn
    // (and the capturing pawn moving). We simulate post-EP occupancy and probe for
    // diagonal sliding attacks on the king.
    let bq = board.pieces(PieceType::Bishop, opp) | board.pieces(PieceType::Queen, opp);
    if bq.is_not_empty() {
        let ep_occ = (board.occ ^ from.bitboard() ^ cap_sq.bitboard()) | to.bitboard();
        if (atk_bishop(ksq, ep_occ) & bq).is_not_empty() {
            return false;
        }
    }

    true
}

/// Generate only strictly tactical pseudo-legal moves (captures and promotions).
#[inline]
pub fn gen_tactical_moves(board: &Position) -> MoveList {
    let mut list = MoveList::new();
    match board.stm {
        Color::White => gen_all::<{ Color::White }, true>(board, &mut list),
        Color::Black => gen_all::<{ Color::Black }, true>(board, &mut list),
    }
    list
}

/// Pseudo-Legal Generation
///
/// Const-generic on color: the compiler monomorphizes into two copies with
/// all direction and rank logic resolved at compile time. No branches for
/// "which way do pawns go?" — it's baked into the binary.
#[inline]
pub fn gen_pseudo_moves(board: &Position) -> MoveList {
    let mut list = MoveList::new();
    match board.stm {
        Color::White => gen_all::<{ Color::White }, false>(board, &mut list),
        Color::Black => gen_all::<{ Color::Black }, false>(board, &mut list),
    }
    list
}

/// Master dispatcher for pseudo-legal generation, heavily optimized via compile-time constants.
#[inline(always)]
fn gen_all<const US: Color, const TACTICAL: bool>(board: &Position, acc: &mut MoveList) {
    let us = board.side_bb[US];
    let them = board.side_bb[US.opposite()];
    let occ = board.occ;

    gen_pawns::<US, TACTICAL>(board, acc, them, occ);
    gen_knights::<TACTICAL>(board, acc, us, them);
    gen_sliders::<TACTICAL>(board, acc, occ, us, them);
    gen_king::<TACTICAL>(board, acc, us, them);

    if !TACTICAL {
        gen_castling::<US>(board, acc, occ);
    }
}

/// Pawns — the most irregular piece in chess.
///
/// Every other piece has clean, symmetric movement.
/// Pawns are special in five different ways:
/// color dependent direction, double push from home, diagonal capture,
/// promotion on the back rank (four choices), and en passant.
/// Bitboard parallelism tames this complexity:
/// one shift computes every single push simultaneously.
#[inline]
fn gen_pawns<const US: Color, const TACTICAL: bool>(
    board: &Position,
    acc: &mut MoveList,
    them: Bitboard,
    occ: Bitboard,
) {
    let empty = !occ;
    let pawns = board.role_bb[PieceType::Pawn] & board.side_bb[US];

    // "North" always means toward promotion, regardless of color.
    let up = Direction::North.relative(US);
    let left = Direction::NorthWest.relative(US);
    let right = Direction::NorthEast.relative(US);

    let up_d = up.delta();
    let left_d = left.delta();
    let right_d = right.delta();

    let (promo_rank, third_rank) = if US == Color::White {
        (RANK_8, RANK_3)
    } else {
        (RANK_1, RANK_6)
    };

    // ── Single pushes (non-promoting) ──
    let all_pushes = pawns.shift(up) & empty;

    if !TACTICAL {
        let mut quiet_pushes = all_pushes & !promo_rank;

        while quiet_pushes.is_not_empty() {
            let to = quiet_pushes.pop_lsb();
            acc.push(Move::new(to.offset_unchecked(-up_d), to, Move::QUIET));
        }

        // ── Double pushes ──
        // Must pass through the 3rd rank on the way — can't leap over pieces.
        let mut doubles = (all_pushes & third_rank).shift(up) & empty;

        while doubles.is_not_empty() {
            let to = doubles.pop_lsb();
            acc.push(Move::new(to.offset_unchecked(-up_d * 2), to, Move::DOUBLE_PUSH));
        }
    }

    // ── Diagonal captures ──
    // File masks prevent board wrapping.
    // NOTE: left and right directions are strictly relative to the
    // side-to-move's visual perspective (e.g. for Black, left shifts
    // toward the H-file).
    let (mask_l, mask_r) = if US == Color::White {
        (!FILE_A, !FILE_H)
    } else {
        (!FILE_H, !FILE_A)
    };

    let cap_l = (pawns & mask_l).shift(left) & them;
    let cap_r = (pawns & mask_r).shift(right) & them;

    let mut cap_l_promo = cap_l & promo_rank;
    let mut cap_l_standard = cap_l & !promo_rank;

    while cap_l_promo.is_not_empty() {
        let to = cap_l_promo.pop_lsb();
        emit_promotions(acc, to.offset_unchecked(-left_d), to, true);
    }

    while cap_l_standard.is_not_empty() {
        let to = cap_l_standard.pop_lsb();
        acc.push(Move::new(to.offset_unchecked(-left_d), to, Move::CAPTURE));
    }

    let mut cap_r_promo = cap_r & promo_rank;
    let mut cap_r_standard = cap_r & !promo_rank;

    while cap_r_promo.is_not_empty() {
        let to = cap_r_promo.pop_lsb();
        emit_promotions(acc, to.offset_unchecked(-right_d), to, true);
    }

    while cap_r_standard.is_not_empty() {
        let to = cap_r_standard.pop_lsb();
        acc.push(Move::new(to.offset_unchecked(-right_d), to, Move::CAPTURE));
    }

    // ── Quiet promotions ──
    let mut promo_pushes = all_pushes & promo_rank;
    while promo_pushes.is_not_empty() {
        let to = promo_pushes.pop_lsb();
        emit_promotions(acc, to.offset_unchecked(-up_d), to, false);
    }

    // ── En passant ──
    if let Some(ep_sq) = board.en_passant {
        let mut attackers = board.get_attackers_on(ep_sq, US) & pawns;
        while attackers.is_not_empty() {
            acc.push(Move::new(attackers.pop_lsb(), ep_sq, Move::EP_CAPTURE));
        }
    }
}

// ──────── Knights, Sliders, King ────────

/// Generate all pseudo-legal knight moves (captures and quiets).
#[inline]
fn gen_knights<const TACTICAL: bool>(board: &Position, acc: &mut MoveList, us: Bitboard, them: Bitboard) {
    for from in board.role_bb[PieceType::Knight] & us {
        let mut targets = atk_knight(from) & !us;
        if TACTICAL {
            targets &= them;
        }
        emit_from_mask(acc, from, targets, them);
    }
}

/// We iterate over bishops, rooks and the corresponding queen components
/// to compute attack bitboards. MAGIC lookups are fast, but the real win is
/// in the dense bitboard transformations that follow.
#[inline]
fn gen_sliders<const TACTICAL: bool>(
    board: &Position,
    acc: &mut MoveList,
    occ: Bitboard,
    us: Bitboard,
    them: Bitboard,
) {
    // Diagonal movers: bishops + queen's diagonal component.
    let mut diags = (board.role_bb[PieceType::Bishop] | board.role_bb[PieceType::Queen]) & us;
    while diags.is_not_empty() {
        let from = diags.pop_lsb();
        let mut targets = atk_bishop(from, occ) & !us;
        if TACTICAL {
            targets &= them;
        }
        emit_from_mask(acc, from, targets, them);
    }

    // Orthogonal movers: rooks + queen's orthogonal component.
    let mut orthos = (board.role_bb[PieceType::Rook] | board.role_bb[PieceType::Queen]) & us;
    while orthos.is_not_empty() {
        let from = orthos.pop_lsb();
        let mut targets = atk_rook(from, occ) & !us;
        if TACTICAL {
            targets &= them;
        }
        emit_from_mask(acc, from, targets, them);
    }
}

/// Generate all pseudo-legal king moves (captures and quiets, excluding castling).
#[inline]
fn gen_king<const TACTICAL: bool>(board: &Position, acc: &mut MoveList, us: Bitboard, them: Bitboard) {
    for from in board.role_bb[PieceType::King] & us {
        let mut targets = atk_king(from) & !us;
        if TACTICAL {
            targets &= them;
        }
        emit_from_mask(acc, from, targets, them);
    }
}

/// Castling — Chess960-compatible.
///
/// Encoded as king→rook (not king→destination) so it generalises to
/// Fischer Random positions where the rook can start on either side.
///
/// Three requirements, all checked here during generation:
///   1. Corridor between king and rook is free of other pieces.
///   2. King is not currently in check.
///   3. King does not pass through or land on an attacked square.
#[inline]
fn gen_castling<const US: Color>(board: &Position, acc: &mut MoveList, occ: Bitboard) {
    let k_bb = board.role_bb[PieceType::King] & board.side_bb[US];
    if k_bb.is_empty() {
        return;
    }

    let ksq = k_bb.lsb();
    let opp = US.opposite();

    let (oo_mask, ooo_mask, oo_idx, ooo_idx) = if US == Color::White {
        (WHITE_OO, WHITE_OOO, 0, 1)
    } else {
        (BLACK_OO, BLACK_OOO, 2, 3)
    };

    // Compute both castle data upfront, then check each right.
    let (oo_data, oo_checks, oo_empty, ooo_data, ooo_checks, ooo_empty) = if US == Color::White {
        (
            &CASTLE_W_KS,
            &CASTLE_W_KS_CHECK,
            W_OO_EMPTY,
            &CASTLE_W_QS,
            &CASTLE_W_QS_CHECK,
            W_OOO_EMPTY,
        )
    } else {
        (
            &CASTLE_B_KS,
            &CASTLE_B_KS_CHECK,
            B_OO_EMPTY,
            &CASTLE_B_QS,
            &CASTLE_B_QS_CHECK,
            B_OOO_EMPTY,
        )
    };

    if (board.castling_rights & oo_mask) != 0 {
        let rsq = board.castling_rooks[oo_idx];
        if board.is_castle_legal(occ, ksq, rsq, oo_data, oo_checks, oo_empty, opp) {
            acc.push(Move::new(ksq, rsq, Move::CASTLE));
        }
    }

    if (board.castling_rights & ooo_mask) != 0 {
        let rsq = board.castling_rooks[ooo_idx];
        if board.is_castle_legal(occ, ksq, rsq, ooo_data, ooo_checks, ooo_empty, opp) {
            acc.push(Move::new(ksq, rsq, Move::CASTLE));
        }
    }
}

/// Queen first — correct in 99.9% of positions.
/// Knight underpromotion (the sneaky fork) is next.
/// Rook and Bishop promotions are almost never optimal,
/// but must exist for completeness.
const QUIET_PROMOS: [u16; 4] = [Move::PROM_Q, Move::PROM_R, Move::PROM_B, Move::PROM_N];

/// Same priority ordering for promotion-captures.
const CAPTURE_PROMOS: [u16; 4] = [
    Move::PROM_Q_CAPTURE,
    Move::PROM_R_CAPTURE,
    Move::PROM_B_CAPTURE,
    Move::PROM_N_CAPTURE,
];

/// Push all four promotion variants (Queen, Rook, Bishop, Knight) to the move list.
///
/// Queen is emitted first as it is almost always the strongest move.
/// Rook and Bishop follow, and Knight is last. While underpromotions are
/// rare, they are necessary for tactical completeness.
#[inline]
fn emit_promotions(acc: &mut MoveList, from: Square, to: Square, capture: bool) {
    for &flag in if capture {
        &CAPTURE_PROMOS
    } else {
        &QUIET_PROMOS
    } {
        acc.push(Move::new(from, to, flag));
    }
}

/// Fan out an attack bitboard into individual moves, tagging captures.
#[inline]
fn emit_from_mask(acc: &mut MoveList, from: Square, targets: Bitboard, them: Bitboard) {
    for to in targets {
        let flag = if them.check_bit(to) {
            Move::CAPTURE
        } else {
            Move::QUIET
        };
        acc.push(Move::new(from, to, flag));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::board::Position;

    #[test]
    fn ep_diagonal_discovered_check() {
        let pos = Position::from_fen("K7/8/8/3Pp3/8/8/8/4k2b w - e6 0 1");
        let moves = gen_legal_moves(&pos);

        let ep_move = Move::new(Square::from_coords(3, 4), Square::from_coords(4, 5), Move::EP_CAPTURE);
        assert!(
            !moves.contains(&ep_move),
            "EP move should be illegal due to diagonal discovered check exposing the mover's king"
        );

        let pos2 = Position::from_fen("4k3/8/8/r2Pp2K/8/8/8/8 w - e6 0 1");
        let moves2 = gen_legal_moves(&pos2);
        assert!(
            !moves2.contains(&ep_move),
            "EP move should be illegal due to horizontal discovered check exposing the mover's king"
        );
    }
}
