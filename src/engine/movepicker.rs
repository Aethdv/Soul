//! Staged move generation and heuristic ordering.
//!
//! # Architecture
//!
//! Generates and scores moves lazily to maximize alpha-beta cutoffs.
//!
//! | Stage          | Content               | Sorting             |
//! |----------------|-----------------------|---------------------|
//! | `Hash`         | PV move from prior iteration | Exact match |
//! | `Captures`     | Captures & promotions | MVV-LVA             |
//! | `Quiets`       | Non-captures          | History heuristic   |
//!
//! Moves are bitpacked with their heuristic scores into `u32`s and sorted
//! ascending using Rust's native `sort_unstable` (PDQSort). This allows
//! popping the highest-scored moves from the back in 𝒪(1) time without
//! index shifting.

use std::mem::MaybeUninit;

use crate::{
    core::{
        board::{
            B_OO_EMPTY, B_OOO_EMPTY, BLACK_OO, BLACK_OOO, CASTLE_B_KS, CASTLE_B_KS_CHECK, CASTLE_B_QS,
            CASTLE_B_QS_CHECK, CASTLE_W_KS, CASTLE_W_KS_CHECK, CASTLE_W_QS, CASTLE_W_QS_CHECK, Position,
            W_OO_EMPTY, W_OOO_EMPTY, WHITE_OO, WHITE_OOO,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook},
        },
        defs::{
            Bitboard, Color, MAX_MOVES, MoveScore, NOT_A, NOT_H, PieceType, RANK_1, RANK_3, RANK_6, RANK_8,
            Square,
        },
        moves::Move,
    },
    debug_index,
    engine::{history::History, search::SearchConfig},
};

// ──────── Staged Move Picker ────────
//
// Generates moves lazily to maximize alpha-beta cutoffs.
//
// Pipeline:
//   [ Hash Move ] ──> [ Captures ] ──> [ Quiets ]
//       (𝒪(1))       (MVV-LVA Sort)   (History Sort)
//
// We fully sort the generated stages using Rust's sort_unstable.
// Why not a lazy partial selection sort to save cycles on early cutoffs?
// Because Big-O is a lie when it hits modern hardware.
// A PDQSort beats a branch-heavy selection sort loop, even when K is small.
//
// Moves and their scores are cleanly bitpacked into u32's. Native sorting
// places the highest-scored moves at the end of the array, allowing us
// to pop them off the back (count -= 1) with zero index-shifting overhead.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Hash,
    GenCaptures,
    YieldCaptures,
    GenQuiets,
    YieldQuiets,
    Done,
}

pub struct MovePicker {
    stage:      Stage,
    hash_move:  Option<Move>,
    candidates: [MaybeUninit<u32>; MAX_MOVES],
    count:      usize,
    mvvlva_v:   [i32; 8],
    mvvlva_a:   [i32; 8],
    mvvlva_ep:  i32,
}

// Ensure move bit-packing assumes correctly.
const _: () = assert!(std::mem::size_of::<Move>() == 2);

impl MovePicker {
    #[inline]
    pub fn new(hash_move: Option<Move>, cfg: &SearchConfig) -> Self {
        Self {
            stage: Stage::Hash,
            hash_move,
            candidates: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
            mvvlva_v: cfg.mvvlva_v,
            mvvlva_a: cfg.mvvlva_a,
            mvvlva_ep: cfg.search_params.mvvlva_ep,
        }
    }

    /// Produce the next move in priority order, or `None` when exhausted.
    #[inline(always)]
    pub fn next(&mut self, board: &Position, history: &History) -> Option<Move> {
        loop {
            match self.stage {
                Stage::Hash => {
                    self.stage = Stage::GenCaptures;
                    if let Some(mv) = self.hash_move {
                        return Some(mv);
                    }
                },

                Stage::GenCaptures => {
                    self.gen_captures(board);
                    // ── Stage Segregation ──
                    // We sort captures independently of quiet moves.
                    // Even if a strong quiet move has a high history score,
                    // it will never override a capture because they are processed
                    // in strictly cordoned stages.
                    //
                    // SAFETY: self.count tracks the exact number of initialized elements.
                    // ptr is valid for self.count reads and writes, and memory is exclusively owned.
                    if self.count > 1 {
                        unsafe {
                            let ptr = self.candidates.as_mut_ptr() as *mut u32;
                            // Sort natively ascending. Best elements float to the end.
                            std::slice::from_raw_parts_mut(ptr, self.count).sort_unstable();
                        }
                    }
                    self.stage = Stage::YieldCaptures;
                },
                Stage::YieldCaptures => {
                    if self.count == 0 {
                        // INVARIANT: When YieldCaptures is exhausted, count is exactly 0.
                        // This allows GenQuiets to reuse the candidates array from the beginning
                        // without needing an explicit clear() or reallocation.
                        self.stage = Stage::GenQuiets;
                        continue;
                    }
                    // Pop from the back: since the array is sorted ascending,
                    // the highest-scored moves sit at the end. Popping via count -= 1
                    // is 𝒪(1) and avoids the expensive index-shifting of remove(0).
                    self.count -= 1;
                    // SAFETY: self.count was strictly checked > 0 above, proving this index holds a valid move.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.hash_move {
                        return Some(mv);
                    }
                },

                Stage::GenQuiets => {
                    self.gen_quiets(board, history);
                    // SAFETY: self.count accurately tracks initialized items.
                    // ptr covers valid memory and is sorted in-place.
                    if self.count > 1 {
                        unsafe {
                            let ptr = self.candidates.as_mut_ptr() as *mut u32;
                            std::slice::from_raw_parts_mut(ptr, self.count).sort_unstable();
                        }
                    }
                    self.stage = Stage::YieldQuiets;
                },
                Stage::YieldQuiets => {
                    if self.count == 0 {
                        self.stage = Stage::Done;
                        continue;
                    }
                    // Pop from the back: exploits the ascending sort to yield the strongest moves first.
                    self.count -= 1;
                    // SAFETY: self.count was strictly checked > 0 above, proving this index holds a valid move.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.hash_move {
                        return Some(mv);
                    }
                },

                Stage::Done => return None,
            }
        }
    }

    // ──────── Capture Generation ────────

    /// Generate all pseudo-legal captures and promotions for the side to move.
    /// Defers to specific piece generators.
    fn gen_captures(&mut self, board: &Position) {
        let stm = board.stm;
        let us = board.side_bb[stm];
        // king captures are never legal
        let them = board.side_bb[stm.opposite()] & !board.role_bb[PieceType::King];
        let occ = board.occ;

        self.gen_pawn_caps(board, us, them);
        self.gen_piece_caps::<{ PieceType::Knight }>(board, us, them, occ);
        self.gen_piece_caps::<{ PieceType::Bishop }>(board, us, them, occ);
        self.gen_piece_caps::<{ PieceType::Rook }>(board, us, them, occ);
        self.gen_piece_caps::<{ PieceType::Queen }>(board, us, them, occ);
        self.gen_piece_caps::<{ PieceType::King }>(board, us, them, occ);
    }

    /// Generates pawn captures (diagonal) and promotion captures.
    /// Handles en passant explicitly.
    #[inline(always)]
    fn gen_pawn_caps(&mut self, board: &Position, us: Bitboard, them: Bitboard) {
        let stm = board.stm;
        let pawns = board.role_bb[PieceType::Pawn] & us;

        let targets = if stm == Color::White {
            [
                (9i8, (pawns & NOT_H) << 9 & them),
                (7i8, (pawns & NOT_A) << 7 & them),
            ]
        } else {
            [
                (-7i8, (pawns & NOT_H) >> 7 & them),
                (-9i8, (pawns & NOT_A) >> 9 & them),
            ]
        };

        let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };

        for (delta, victims) in targets {
            let promo = victims & prom_mask;
            let standard = victims & !prom_mask;

            for to in promo {
                let from = Square((to.0 as i8 - delta) as u8);
                self.add_promo_caps(board, from, to);
            }

            for to in standard {
                let from = Square((to.0 as i8 - delta) as u8);
                self.add_cap(board, Move::new(from, to, Move::CAPTURE), PieceType::Pawn);
            }
        }

        // En passant.
        if let Some(ep_sq) = board.en_passant {
            for from in atk_pawn(ep_sq, stm.opposite()) & pawns {
                self.add_cap(board, Move::new(from, ep_sq, Move::EP_CAPTURE), PieceType::Pawn);
            }
        }
    }

    /// Generic piece capture generator.
    /// Const-monomorphized by `PT` (PieceType) to eliminate dynamic dispatch for attack lookups.
    #[inline]
    fn gen_piece_caps<const PT: PieceType>(
        &mut self,
        board: &Position,
        us: Bitboard,
        them: Bitboard,
        occ: Bitboard,
    ) {
        let a_pen = *crate::debug_index!(self.mvvlva_a, PT as usize);
        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(from, occ) & them {
                let victim = board.piece_at(to);
                let v_val = *crate::debug_index!(self.mvvlva_v, victim as usize);
                let score = v_val - a_pen;
                self.add_move_packed(Move::new(from, to, Move::CAPTURE), score as MoveScore);
            }
        }
    }

    /// Appends a move and its score to the candidates list.
    /// Manages bit-packing and boundary checks.
    #[inline(always)]
    fn add_move_packed(&mut self, mv: Move, score: MoveScore) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");
        let sort_score = (score as i32 + 32768).clamp(0, 65535) as u32;
        let packed = (sort_score << 16) | (mv.inner() as u32);
        crate::debug_index_mut!(self.candidates, self.count).write(packed);
        self.count += 1;
    }

    /// Append a capture, tagged with its MVV-LVA score for selection sort.
    #[inline]
    fn add_cap(&mut self, board: &Position, mv: Move, attacker: PieceType) {
        let score = self.mvv_lva(board, mv, attacker);
        self.add_move_packed(mv, score);
    }

    /// Emit all four promotion-captures for one pawn diagonal.
    /// Queen promotion scores highest via MVV-LVA and surfaces first.
    #[inline]
    fn add_promo_caps(&mut self, board: &Position, from: Square, to: Square) {
        let victim = board.piece_at(to);
        let v_val = *crate::debug_index!(self.mvvlva_v, victim as usize);
        let a_pen = *crate::debug_index!(self.mvvlva_a, PieceType::Pawn as usize);
        let base = v_val - a_pen;

        let q = base + *crate::debug_index!(self.mvvlva_v, PieceType::Queen as usize);
        let r = base + *crate::debug_index!(self.mvvlva_v, PieceType::Rook as usize);
        let b = base + *crate::debug_index!(self.mvvlva_v, PieceType::Bishop as usize);
        let n = base + *crate::debug_index!(self.mvvlva_v, PieceType::Knight as usize);

        self.add_move_packed(Move::new(from, to, Move::PROM_Q_CAPTURE), q as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_R_CAPTURE), r as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_B_CAPTURE), b as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_N_CAPTURE), n as MoveScore);
    }

    /// Most Valuable Victim – Least Valuable Attacker.
    ///
    ///   `score = V(victim) - V(attacker) [+ V(promo)]`
    ///
    /// The Stage segregation in `MovePicker::next` ensures all captures are
    /// yielded before any quiet moves, so a global bias is no longer needed.
    #[inline(always)]
    fn mvv_lva(&self, board: &Position, mv: Move, attacker: PieceType) -> MoveScore {
        if mv.is_en_passant() {
            return self.mvvlva_ep as MoveScore;
        }

        let victim = board.piece_at(mv.to());

        debug_assert!(usize::from(victim) < 8);
        debug_assert!(usize::from(attacker) < 8);

        let v = *crate::debug_index!(self.mvvlva_v, victim as usize);
        let a = *crate::debug_index!(self.mvvlva_a, attacker as usize);
        let mut s = v - a;

        // ── Promotion bonus ──
        // Queen promotion capture is worth more than
        // a plain capture of the same victim.
        if let Some(p) = mv.promo() {
            debug_assert!(usize::from(p) < 8);
            s += *crate::debug_index!(self.mvvlva_v, p as usize);
        }

        s as MoveScore
    }

    // ──────── Quiet Move Generation ────────

    /// Generate all non-capturing pseudo-legal moves, including castling.
    fn gen_quiets(&mut self, board: &Position, history: &History) {
        let us = board.side_bb[board.stm];
        let occ = board.occ;
        let empty = !occ;

        self.gen_pawn_quiets(board, us, empty, history);
        self.gen_piece_quiets::<{ PieceType::Knight }>(board, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Bishop }>(board, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Rook }>(board, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Queen }>(board, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::King }>(board, us, empty, occ, history);
        self.gen_castling(board, us, occ, history);
    }

    /// Append a quiet move with history scoring.
    #[inline]
    fn add_quiet(&mut self, board: &Position, mv: Move, history: &History) {
        let pt = board.expect_piece_at(mv.from());
        self.add_quiet_node(mv, pt, board.stm, history);
    }

    #[inline(always)]
    fn add_quiet_node(&mut self, mv: Move, pt: PieceType, stm: Color, history: &History) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");

        let score = history.score_quiet(stm, pt, mv.to());

        // ──────── Move Ordering Heuristics ────────

        // History values are mathematically bounded to [-16384, 16384] by the
        // update gravity. If this ever breaches 32767, the bitpacked sort
        // key will overflow and destroy move ordering.
        debug_assert!(score.abs() <= 16384, "History score overflow: {score}");

        // Added 32768 guarantees a positive value without overflow.
        // History ranges from [-16384, 16384], so logic natively boundaries at [16384, 49152].
        let mut sort_score = (score + 32768).clamp(0, 65535) as u32;

        // Quiet promotions outrank all other quiet moves.
        // Queen first (almost always best), then knight (fork potential),
        // then rook/bishop (needed for correctness, rarely chosen).
        if mv.is_promotion() {
            sort_score = match mv.promo() {
                Some(PieceType::Queen) => 65535,
                Some(PieceType::Knight) => 65534,
                Some(PieceType::Rook) => 65533,
                _ => 65532,
            };
        }

        let packed = (sort_score << 16) | (mv.inner() as u32);

        // SAFETY: count < MAX_MOVES is guarded above.
        unsafe {
            self.candidates
                .as_mut_ptr()
                .add(self.count)
                .write(MaybeUninit::new(packed))
        };
        self.count += 1;
    }

    /// Emit all four quiet promotions for a single pawn push.
    #[inline]
    fn add_promo_quiets(&mut self, from: Square, to: Square, stm: Color, history: &History) {
        self.add_quiet_node(Move::new(from, to, Move::PROM_Q), PieceType::Pawn, stm, history);
        self.add_quiet_node(Move::new(from, to, Move::PROM_R), PieceType::Pawn, stm, history);
        self.add_quiet_node(Move::new(from, to, Move::PROM_B), PieceType::Pawn, stm, history);
        self.add_quiet_node(Move::new(from, to, Move::PROM_N), PieceType::Pawn, stm, history);
    }

    /// Generates pawn pushes: single steps, double steps from the starting rank,
    /// and quiet promotions on the final rank.
    #[inline(always)]
    fn gen_pawn_quiets(&mut self, board: &Position, us: Bitboard, empty: Bitboard, history: &History) {
        let stm = board.stm;
        let up = stm.forward_dir();
        let up_d = up.delta();

        let pawns = board.role_bb[PieceType::Pawn] & us;
        let all_pushes = pawns.shift(up) & empty;

        let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };
        let third_rank = if stm == Color::White { RANK_3 } else { RANK_6 };

        let promo_pushes = all_pushes & prom_mask;
        let quiet_pushes = all_pushes & !prom_mask;

        for to in promo_pushes {
            let from = Square((to.0 as i8 - up_d) as u8);
            self.add_promo_quiets(from, to, stm, history);
        }

        for to in quiet_pushes {
            let from = Square((to.0 as i8 - up_d) as u8);
            self.add_quiet_node(Move::new(from, to, Move::QUIET), PieceType::Pawn, stm, history);
        }

        let doubles = (all_pushes & third_rank).shift(up) & empty;
        for to in doubles {
            let from = Square((to.0 as i8 - up_d * 2) as u8);
            self.add_quiet_node(Move::new(from, to, Move::DOUBLE_PUSH), PieceType::Pawn, stm, history);
        }
    }

    /// Generic piece quiet move generator.
    /// Const-monomorphized by `PT` (PieceType) to eliminate dynamic dispatch for attack lookups.
    #[inline]
    fn gen_piece_quiets<const PT: PieceType>(
        &mut self,
        board: &Position,
        us: Bitboard,
        empty: Bitboard,
        occ: Bitboard,
        history: &History,
    ) {
        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(from, occ) & empty {
                self.add_quiet_node(Move::new(from, to, Move::QUIET), PT, board.stm, history);
            }
        }
    }

    // ──────── Castling (Standard + Chess960) ────────

    /// Evaluates castling rights and attempts to generate legal castling moves.
    #[inline]
    fn gen_castling(&mut self, board: &Position, us: Bitboard, occ: Bitboard, history: &History) {
        let king_bb = board.role_bb[PieceType::King] & us;
        if king_bb.is_empty() {
            return;
        }

        let stm = board.stm;
        let ksq = king_bb.lsb();
        let opp = stm.opposite();

        let (oo_mask, ooo_mask, ks_idx, qs_idx) = if stm == Color::White {
            (WHITE_OO, WHITE_OOO, 0, 1)
        } else {
            (BLACK_OO, BLACK_OOO, 2, 3)
        };

        // Kingside
        if (board.castling_rights & oo_mask) != 0 {
            let rsq = board.castling_rooks[ks_idx];
            let (data, checks, empty_mask) = if stm == Color::White {
                (&CASTLE_W_KS, &CASTLE_W_KS_CHECK, W_OO_EMPTY)
            } else {
                (&CASTLE_B_KS, &CASTLE_B_KS_CHECK, B_OO_EMPTY)
            };
            self.try_castle(board, occ, ksq, rsq, data, checks, empty_mask, opp, Move::CASTLE, history);
        }

        // Queenside
        if (board.castling_rights & ooo_mask) != 0 {
            let rsq = board.castling_rooks[qs_idx];
            let (data, checks, empty_mask) = if stm == Color::White {
                (&CASTLE_W_QS, &CASTLE_W_QS_CHECK, W_OOO_EMPTY)
            } else {
                (&CASTLE_B_QS, &CASTLE_B_QS_CHECK, B_OOO_EMPTY)
            };
            self.try_castle(board, occ, ksq, rsq, data, checks, empty_mask, opp, Move::CASTLE, history);
        }
    }

    /// Validate and emit a single castling move.
    ///
    /// `data` layout: `[king_origin, rook_origin, king_dest, rook_dest]`.
    /// Standard chess takes the fast path (single bitboard AND for emptiness);
    /// Chess960 falls through to the general per-square walk.
    #[allow(clippy::too_many_arguments)]
    fn try_castle(
        &mut self,
        board: &Position,
        occ: Bitboard,
        ksq: Square,
        rsq: Square,
        data: &[u8; 4],
        check_sqs: &[u8],
        empty_mask: Bitboard,
        opp: Color,
        flag: u16,
        history: &History,
    ) {
        if board.is_castle_legal(occ, ksq, rsq, data, check_sqs, empty_mask, opp) {
            self.add_quiet(board, Move::new(ksq, rsq, flag), history);
        }
    }

    // ──────── Bit-Unpacking & Attack Lookups ────────

    /// Extracts the move from the packed item at `i`.
    #[inline(always)]
    unsafe fn read_move(&self, i: usize) -> Move {
        debug_assert!(i < MAX_MOVES);
        // SAFETY: The caller contract of read_move requires that index i has been successfully initialized.
        let packed = unsafe { debug_index!(self.candidates, i).assume_init() };
        Move::from_u16((packed & 0xFFFF) as u16)
    }

    /// Static dispatcher for piece attacks.
    /// Maps the const generic `PT` to the appropriate bitboard attack function.
    #[inline(always)]
    fn attacks<const PT: PieceType>(from: Square, occ: Bitboard) -> Bitboard {
        match PT {
            PieceType::Pawn => {
                debug_assert!(false, "Pawn attacks must use dedicated generators");
                Bitboard(0)
            },
            PieceType::Knight => atk_knight(from),
            PieceType::Bishop => atk_bishop(from, occ),
            PieceType::Rook => atk_rook(from, occ),
            PieceType::Queen => atk_bishop(from, occ) | atk_rook(from, occ),
            PieceType::King => atk_king(from),
            PieceType::None => Bitboard(0),
        }
    }
}
