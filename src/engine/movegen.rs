//! Pseudo-legal move generation and strict legality filtering.
//!
//! A bitboard computes every pawn push in one shift and every knight hop in one
//! lookup, bulk operations oblivious to legality. Pins, checks, and en-passant
//! x-rays need per-move reasoning, so generation runs blind and a legality pass
//! drops the moves that leave the king in check.

use crate::core::{
    board::{
        Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS,
        bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, between_bb, line_bb},
    },
    defs::{Bitboard, Color, Direction, PieceType, RANK_1, RANK_3, RANK_6, RANK_8, Square},
    moves::{Move, MoveList},
};

/// Queen first, almost always right. Knight is the underpromotion that matters, since it
/// checks and forks where a queen cannot; rook and bishop ride along.
const QUIET_PROMOS: [u16; 4] = [Move::PROM_Q, Move::PROM_R, Move::PROM_B, Move::PROM_N];
/// Same priority ordering for promotion-captures.
const CAPTURE_PROMOS: [u16; 4] = [Move::PROM_Q_CAPTURE, Move::PROM_R_CAPTURE, Move::PROM_B_CAPTURE, Move::PROM_N_CAPTURE];

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

/// Dispatches on the runtime side to move, into the const-generic worker
/// below, so the color check happens once per call, not once per piece.
#[inline]
pub fn gen_pseudo_moves(board: &Position) -> MoveList {
    let mut list = MoveList::new();
    match board.stm {
        Color::White => gen_all::<{ Color::White }, false>(board, &mut list),
        Color::Black => gen_all::<{ Color::Black }, false>(board, &mut list),
    }
    list
}

/// Captures and promotions only, still pseudo-legal.
#[inline]
pub fn gen_tactical_moves(board: &Position) -> MoveList {
    let mut list = MoveList::new();
    match board.stm {
        Color::White => gen_all::<{ Color::White }, true>(board, &mut list),
        Color::Black => gen_all::<{ Color::Black }, true>(board, &mut list),
    }
    list
}

/// Validates that a TT move is pseudo-legal for the current position.
///
/// TT hash collisions can produce moves from completely unrelated positions.
/// This checks: friendly piece on `from`, correct attack pattern for that
/// piece, and flag consistency (captures, promotions, castling, EP, double push).
/// Does not check legality (pins, checks). That's `is_legal`'s job.
#[inline]
pub fn is_pseudo_legal(board: &Position, mv: Move) -> bool {
    let from = mv.from();
    let to = mv.to();
    let stm = board.stm;
    let us = board.side_bb[stm];
    let them = board.side_bb[stm.opposite()];
    let occ = board.occ;

    if !us.check_bit(from) {
        return false;
    }

    let piece = board.piece_at(from);

    // `to` names the rook's home square, so a collision with the CASTLE flag and a junk
    // `to` makes apply_castling lift a non-rook and drop a phantom rook, corruption that
    // unmake compounds instead of repairing.
    //
    // Only the mover's own two slots count. Scanning all four reaches a stale enemy slot,
    // and accepts a castle onto the square that rook abandoned and one of ours later took.
    // is_castle_move_legal owns the rest: the right, the corridor, and the king's path.
    if mv.is_castling() {
        let (ks, qs) = if stm == Color::White { (ROOK_W_KS, ROOK_W_QS) } else { (ROOK_B_KS, ROOK_B_QS) };
        return piece == PieceType::King
            && board.piece_at(to) == PieceType::Rook
            && us.check_bit(to)
            && (to == board.castling_rooks[ks] || to == board.castling_rooks[qs])
            && board.is_castle_move_legal(stm, from, to);
    }

    if us.check_bit(to) {
        return false;
    }

    // The capture flag has to agree with the board, en passant excepted: its
    // victim stands beside `to`, never on it.
    let enemy_on_to = them.check_bit(to);
    if mv.is_capture() != enemy_on_to && !mv.is_en_passant() {
        return false;
    }

    // Pawn-only flags must come from a pawn. A slider or knight move flagged as a
    // promotion, en passant, or double push clears the per-piece attack check below,
    // then reaches make_move's pawn paths and arms a phantom en passant square or lifts
    // a victim off the wrong square. The castle flag is king-only and handled above.
    if piece != PieceType::Pawn && (mv.is_promotion() || mv.is_en_passant() || mv.is_double_push()) {
        return false;
    }

    match piece {
        PieceType::Pawn => {
            let fwd = stm.forward_dir().delta();
            let (start_rank, last_rank) = if stm == Color::White { (1u8, 7u8) } else { (6u8, 0u8) };

            // A promotion flag belongs exactly on the last rank, and a non-promoting
            // pawn move must never land there; the generator always promotes.
            // A collision that breaks either would queen mid-board or push
            // a pawn onto the back rank.
            if mv.is_promotion() != (to.rank() == last_rank) {
                return false;
            }

            if mv.is_en_passant() {
                return board.en_passant == Some(to) && atk_pawn(from, stm).check_bit(to);
            }

            if mv.is_double_push() {
                // From the start rank only. Otherwise the move arms a phantom en passant
                // square that a later EP capture turns into board corruption. The rank is
                // tested first because the square between only exists once it holds.
                return from.rank() == start_rank
                    && to.0 as i8 == from.0 as i8 + fwd * 2
                    && !occ.check_bit(Square((from.0 as i8 + fwd) as u8))
                    && !occ.check_bit(to);
            }

            if mv.is_capture() {
                return atk_pawn(from, stm).check_bit(to);
            }
            // Single push
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

/// Does this move leave our king safe?
///
/// Most moves pass trivially. The interesting cases:
///   • King moves    - destination must not be attacked.
///   • Double check  - only the king himself can escape.
///   • En passant    - removing two pawns from one rank can unmask a rook.
///   • Pinned pieces - may only slide along the pin ray.
///   • Single check  - must capture the checker or interpose.
#[inline(always)]
pub fn is_legal(board: &Position, mv: Move, ksq: Square, pinned: Bitboard, checkers: Bitboard, opp: Color) -> bool {
    let from = mv.from();
    let to = mv.to();

    // King moves
    if from == ksq {
        if mv.is_castling() {
            // Castling out of check is flatly illegal. Everything else was settled by
            // is_castle_move_legal, which both the generator and the TT validator call.
            return checkers.is_empty();
        }
        // is_attacked::<true> removes the king from occupancy so
        // sliding attackers see through his old square. Without this,
        // the king could "hide behind himself", stepping backward
        // along a rook's file and believing the square is safe.
        return !board.is_attacked::<true>(to, opp, from.bitboard());
    }

    // Double check
    if checkers.popcount() > 1 {
        return false;
    }

    // En passant: the one move where the captured piece isn't on the destination
    // square, requiring a special x-ray check.
    if mv.is_en_passant() {
        return is_ep_legal(board, mv, ksq, pinned, checkers, opp);
    }

    // Pinned pieces: line_bb(from, to) returns the entire line through both squares,
    // so a king that isn't on it means the piece is leaving the ray.
    if pinned.check_bit(from) && !line_bb(from, to).check_bit(ksq) {
        return false;
    }

    if checkers.is_empty() {
        return true;
    }

    // Single check
    let checker = checkers.lsb();
    to == checker || between_bb(ksq, checker).check_bit(to)
}

/// En passant is the only move that clears two squares on one rank: ours departs, theirs
/// is captured beside it. A pin test on the moving pawn cannot see the second removal, so
/// a rook or queen screened by both pawns gives a discovered check.
///
/// The slider probes run against the post-capture occupancy instead.
#[inline]
fn is_ep_legal(board: &Position, mv: Move, ksq: Square, pinned: Bitboard, checkers: Bitboard, opp: Color) -> bool {
    let from = mv.from();
    let to = mv.to();
    // The captured pawn stands one rank behind the destination, and `^ 8` gets there
    // without knowing which way: an en passant destination is always on rank 3 or rank 6,
    // and flipping the low rank bit walks each of those toward the pawn that just moved.
    let cap_sq = Square(to.0 ^ 8);

    // An en passant capture answers a check only by taking the checker, which is then the
    // pawn that just double-pushed, or by landing on the ray between king and checker.
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

    // Both pawns off, ours onto the destination.
    let ep_occ = (board.occ ^ from.bitboard() ^ cap_sq.bitboard()) | to.bitboard();

    // ── Horizontal discovered check
    // Only the shared rank can lose two screens at once, so the rook probe is gated on it.
    if ksq.rank() == from.rank() {
        let rq = board.pieces(PieceType::Rook, opp) | board.pieces(PieceType::Queen, opp);
        if (atk_rook(ksq, ep_occ) & rq).is_not_empty() {
            return false;
        }
    }

    // ── Diagonal discovered check
    // The captured pawn alone can be the screen, so nothing gates this one.
    let bq = board.pieces(PieceType::Bishop, opp) | board.pieces(PieceType::Queen, opp);
    (atk_bishop(ksq, ep_occ) & bq).is_empty()
}

/// Const-generic on color and `TACTICAL`, so the pawn directions and the promotion rank
/// resolve at compile time and nothing in here branches on which way pawns move.
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
        board.for_each_castle(US, |mv| acc.push(mv));
    }
}

/// Pawns are the one irregular piece: direction depends on color, the first move may
/// double, capture is diagonal, the last rank promotes four ways, and en passant takes
/// a piece that never stood on the destination square.
#[inline]
fn gen_pawns<const US: Color, const TACTICAL: bool>(board: &Position, acc: &mut MoveList, them: Bitboard, occ: Bitboard) {
    let empty = !occ;
    let pawns = board.role_bb[PieceType::Pawn] & board.side_bb[US];

    // "North" always means toward promotion, regardless of color.
    let up = Direction::North.relative(US);
    let left = Direction::NorthWest.relative(US);
    let right = Direction::NorthEast.relative(US);

    let up_d = up.delta();
    let left_d = left.delta();
    let right_d = right.delta();

    let (promo_rank, third_rank) = if US == Color::White { (RANK_8, RANK_3) } else { (RANK_1, RANK_6) };

    // ── Single pushes (non-promoting)
    let all_pushes = pawns.shift(up) & empty;

    if !TACTICAL {
        let mut quiet_pushes = all_pushes & !promo_rank;
        while quiet_pushes.is_not_empty() {
            let to = quiet_pushes.pop_lsb();
            acc.push(Move::new(to.offset_unchecked(-up_d), to, Move::QUIET));
        }

        // ── Double pushes
        // A pawn cannot leap, so the second step starts only from the pushes that landed
        // on the third rank.
        let mut doubles = (all_pushes & third_rank).shift(up) & empty;
        while doubles.is_not_empty() {
            let to = doubles.pop_lsb();
            acc.push(Move::new(to.offset_unchecked(-up_d * 2), to, Move::DOUBLE_PUSH));
        }
    }

    // ── Diagonal captures
    // Left and right are relative to the side-to-move's view of the board, so for Black,
    // left shifts toward the H-file. The wrap-around file mask lives in `shift`.
    let cap_l = pawns.shift(left) & them;
    let cap_r = pawns.shift(right) & them;
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

    // ── Quiet promotions
    let mut promo_pushes = all_pushes & promo_rank;
    while promo_pushes.is_not_empty() {
        let to = promo_pushes.pop_lsb();
        emit_promotions(acc, to.offset_unchecked(-up_d), to, false);
    }

    // ── En passant
    // The squares an enemy pawn on `ep_sq` would strike are the squares ours strike it
    // from, so the lookup runs inverted.
    if let Some(ep_sq) = board.en_passant {
        let mut attackers = atk_pawn(ep_sq, US.opposite()) & pawns;
        while attackers.is_not_empty() {
            acc.push(Move::new(attackers.pop_lsb(), ep_sq, Move::EP_CAPTURE));
        }
    }
}

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

/// A queen appears in both loops, once for each half of its move set.
#[inline]
fn gen_sliders<const TACTICAL: bool>(board: &Position, acc: &mut MoveList, occ: Bitboard, us: Bitboard, them: Bitboard) {
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

/// King steps only; castling is generated separately.
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

#[inline]
fn emit_promotions(acc: &mut MoveList, from: Square, to: Square, capture: bool) {
    for &flag in if capture { &CAPTURE_PROMOS } else { &QUIET_PROMOS } {
        acc.push(Move::new(from, to, flag));
    }
}

/// Fan out an attack bitboard into individual moves, tagging captures.
#[inline]
fn emit_from_mask(acc: &mut MoveList, from: Square, targets: Bitboard, them: Bitboard) {
    for to in targets {
        let flag = if them.check_bit(to) { Move::CAPTURE } else { Move::QUIET };
        acc.push(Move::new(from, to, flag));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{
        board::{Position, STARTPOS},
        zobrist::ConstRng,
    };

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

    #[test]
    fn tt_castle_move_validates_corridor() {
        let pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K1NR w KQkq - 0 1");
        let e1 = Square(4);
        let kingside = Move::new(e1, Square(7), Move::CASTLE); // blocked by the knight on g1
        let queenside = Move::new(e1, Square(0), Move::CASTLE); // corridor clear
        assert!(!is_pseudo_legal(&pos, kingside), "kingside castle through the g1 knight must be rejected");
        assert!(is_pseudo_legal(&pos, queenside), "queenside castle with a clear corridor must pass");
    }

    #[test]
    fn tt_castle_rejects_the_opponents_stale_rook_slot() {
        let mut pos = Position::from_fen("r3k3/1P6/8/8/8/8/8/R3K3 b Qq - 0 1");
        let mut acc = pos.get_initial_accumulator();
        for mv in [
            Move::new(Square(60), Square(61), Move::QUIET),
            Move::new(Square(49), Square(56), Move::PROM_R_CAPTURE),
            Move::new(Square(61), Square(62), Move::QUIET),
        ] {
            pos.make_move(mv, &mut acc);
        }
        assert_eq!(pos.castling_rooks[ROOK_B_QS], Square(56), "the stale slot is what makes this reachable");
        let cross_board = Move::new(Square(4), Square(56), Move::CASTLE);
        assert!(!is_pseudo_legal(&pos, cross_board), "a rook on the opponent's stale slot is not a castling partner");
        let genuine = Move::new(Square(4), Square(0), Move::CASTLE);
        assert!(is_pseudo_legal(&pos, genuine), "White still holds the queenside right with a clear corridor");
    }

    #[test]
    fn tt_double_push_validates_start_rank() {
        let pos = Position::from_fen("4k3/8/8/8/8/4P3/8/4K3 w - - 0 1"); // pawn on e3
        let phantom = Move::new(Square::from_coords(4, 2), Square::from_coords(4, 4), Move::DOUBLE_PUSH);
        assert!(!is_pseudo_legal(&pos, phantom), "double push from the third rank must be rejected");
        let start = Position::from_fen("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1"); // pawn on e2
        let real = Move::new(Square::from_coords(4, 1), Square::from_coords(4, 3), Move::DOUBLE_PUSH);
        assert!(is_pseudo_legal(&start, real), "double push from the second rank must pass");
    }

    /// Legality perft (from Rose): at each node, sweep every 16-bit move encoding
    /// and recurse on whatever `is_pseudo_legal` + `is_legal` jointly accept.
    #[test]
    fn legality_perft_matches_reference() {
        // (FEN, [perft(0), perft(1), …])
        let cases: &[(&str, &[u64])] = &[
            (STARTPOS, &[1, 20, 400, 8902]),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[1, 48, 2039]),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[1, 14, 191, 2812]),
            ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", &[1, 6, 264]),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[1, 44, 1486]),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[1, 46, 2079]),
        ];

        for (fen, expected) in cases {
            let pos = Position::from_fen(fen);
            for (depth, &want) in expected.iter().enumerate() {
                assert_eq!(legality_perft(&pos, depth), want, "legality perft depth {depth} for {fen}");
            }
        }
    }

    /// The same legality perft, pushed deeper.
    #[test]
    fn legality_perft_deep() {
        let cases: &[(&str, &[u64])] = &[
            (STARTPOS, &[1, 20, 400, 8902, 197281]),
            ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", &[1, 48, 2039, 97862]),
            ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", &[1, 14, 191, 2812, 43238]),
            ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", &[1, 44, 1486, 62379]),
            ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", &[1, 46, 2079, 89890]),
        ];

        for (fen, expected) in cases {
            let pos = Position::from_fen(fen);
            for (depth, &want) in expected.iter().enumerate() {
                assert_eq!(legality_perft(&pos, depth), want, "deep legality perft depth {depth} for {fen}");
            }
        }
    }

    /// Counts legal-move paths by validating every 16-bit encoding through
    /// `is_pseudo_legal` + `is_legal`, the exact pair that guards TT moves.
    fn legality_perft(pos: &Position, depth: usize) -> u64 {
        if depth == 0 {
            return 1;
        }

        let stm = pos.stm;
        let opp = stm.opposite();
        let ksq = pos.pieces(PieceType::King, stm).lsb();
        let pinned = pos.king_blockers();
        let checkers = pos.checkers();

        let mut nodes = 0;

        for raw in 0..=u16::MAX {
            // 8, 9, 13 are unused flag nibbles the generator never emits, so a real
            // collision never produces them; counting them would double-count the
            // quiet/capture move they decode to.
            if matches!(raw >> 12, 8 | 9 | 13) {
                continue;
            }

            let mv = Move::from_u16(raw);
            if !is_pseudo_legal(pos, mv) || !is_legal(pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            let mut child = *pos;
            let mut acc = child.get_initial_accumulator();
            child.make_move(mv, &mut acc);
            nodes += legality_perft(&child, depth - 1);
        }
        nodes
    }

    #[test]
    fn fuzz_make_unmake_integrity() {
        const POSITIONS: usize = 256;

        let mut rng = ConstRng::new(0x9E3779B97F4A7C15);
        let mut pos = Position::from_fen(STARTPOS);
        let mut acc = pos.get_initial_accumulator();

        for _ in 0..POSITIONS {
            check_every_move(&pos);

            let legal = gen_legal_moves(&pos);
            if legal.is_empty() {
                pos = Position::from_fen(STARTPOS);
                acc = pos.get_initial_accumulator();
                continue;
            }

            let mv = legal[rng.next() as usize % legal.len()];
            pos.make_move(mv, &mut acc);
        }
    }

    /// Sweep every 16-bit encoding against `pos`; each move the validators accept must
    /// survive make (consistent board, matching accumulator) and unmake (exact restore).
    fn check_every_move(pos: &Position) {
        let stm = pos.stm;
        let opp = stm.opposite();
        let ksq = pos.pieces(PieceType::King, stm).lsb();
        let pinned = pos.king_blockers();
        let checkers = pos.checkers();

        for raw in 0..=u16::MAX {
            let mv = Move::from_u16(raw);

            if !is_pseudo_legal(pos, mv) || !is_legal(pos, mv, ksq, pinned, checkers, opp) {
                continue;
            }

            let mut child = *pos;
            let mut acc = child.get_initial_accumulator();
            let undo = child.make_move(mv, &mut acc);

            assert_consistent(&child, pos, mv);
            let fresh = child.get_initial_accumulator();
            assert_eq!(acc.to_array(), fresh.to_array(), "accumulator diverged {}", context(pos, mv));
            child.unmake_move(mv, &undo);
            assert!(board_eq(&child, pos), "unmake did not restore {}", context(pos, mv));
        }
    }

    /// Every internal view of a position agrees with the others.
    fn assert_consistent(pos: &Position, before: &Position, mv: Move) {
        let w = pos.side_bb[Color::White];
        let b = pos.side_bb[Color::Black];

        assert_eq!(pos.occ, w | b, "occupancy is not the color union {}", context(before, mv));
        assert!((w & b).is_empty(), "a square is owned by both colors {}", context(before, mv));

        let mut union = Bitboard(0);
        let mut count = 0;

        for pt in [PieceType::Pawn, PieceType::Knight, PieceType::Bishop, PieceType::Rook, PieceType::Queen, PieceType::King] {
            let role = pos.role_bb[pt];
            assert!((role & !pos.occ).is_empty(), "role {pt:?} outside occupancy {}", context(before, mv));
            union |= role;
            count += role.popcount();
        }

        assert_eq!(union, pos.occ, "roles do not cover occupancy {}", context(before, mv));
        assert_eq!(count, pos.occ.popcount(), "two roles share a square {}", context(before, mv));

        for color in [Color::White, Color::Black] {
            assert_eq!(pos.pieces(PieceType::King, color).popcount(), 1, "{color:?} king count {}", context(before, mv));
        }

        for sq in (0..64).map(Square) {
            let pt = pos.piece_at(sq);

            assert_eq!(
                pos.occ.check_bit(sq),
                pt != PieceType::None,
                "mailbox/occupancy disagree at {} {}",
                sq.0,
                context(before, mv)
            );

            if pt != PieceType::None {
                assert!(pos.role_bb[pt].check_bit(sq), "mailbox/role disagree at {} {}", sq.0, context(before, mv));
            }
        }

        assert_eq!(pos.hash, pos.calc_zobrist(), "hash desynced {}", context(before, mv));
        assert_eq!(pos.pawn_key, pos.calc_pawn_hash(), "pawn key desynced {}", context(before, mv));
        assert_eq!(pos.minor_key, pos.calc_minor_hash(), "minor key desynced {}", context(before, mv));
        assert_eq!(pos.major_key, pos.calc_major_hash(), "major key desynced {}", context(before, mv));
    }

    /// Field-wise board equality across everything make/unmake can touch.
    fn board_eq(a: &Position, b: &Position) -> bool {
        a.side_bb == b.side_bb
            && a.role_bb == b.role_bb
            && a.occ == b.occ
            && a.hash == b.hash
            && a.pawn_key == b.pawn_key
            && a.minor_key == b.minor_key
            && a.major_key == b.major_key
            && a.pieces == b.pieces
            && a.castling_rights == b.castling_rights
            && a.stm == b.stm
            && a.en_passant == b.en_passant
            && a.halfmove_clock == b.halfmove_clock
            && a.fullmove_number == b.fullmove_number
    }

    fn context(pos: &Position, mv: Move) -> String {
        format!("after {} from {}", mv.to_uci(pos.is_frc), pos.as_fen())
    }
}
