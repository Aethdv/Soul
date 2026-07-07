//! Staged move generation and heuristic ordering.
//!
//! Most nodes cut off on the hash move or the first good capture, so generating
//! and scoring every quiet up front is work thrown away. The picker yields moves
//! in stages, best guesses first, and generates a stage only once the cheaper
//! ones run dry.
//!
//! | Stage         | Content                       | Sorting           |
//! |---------------|-------------------------------|-------------------|
//! | `Hash`        | PV move from prior iteration  | Exact match       |
//! | `Captures`    | SEE-winning captures & promos | MVV-LVA           |
//! | `Quiets`      | Non-captures                  | History heuristic |
//! | `BadCaptures` | SEE-losing captures, deferred | MVV-LVA           |
//!
//! Moves are bitpacked with their heuristic scores into `u32`s and sorted
//! ascending using Rust's native `sort_unstable` (ipnsort). This allows
//! popping the highest-scored moves from the back in 𝒪(1) time without
//! index shifting.

use std::{mem::MaybeUninit, slice};

use crate::{
    core::{
        board::{
            B_OO_EMPTY, B_OOO_EMPTY, BLACK_OO, BLACK_OOO, CASTLE_B_KS, CASTLE_B_KS_CHECK, CASTLE_B_QS, CASTLE_B_QS_CHECK,
            CASTLE_W_KS, CASTLE_W_KS_CHECK, CASTLE_W_QS, CASTLE_W_QS_CHECK, Position, W_OO_EMPTY, W_OOO_EMPTY, WHITE_OO, WHITE_OOO,
            attacks::Pins,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook},
        },
        defs::{Bitboard, Color, MAX_MOVES, MoveScore, NOT_A, NOT_H, PieceType, RANK_1, RANK_3, RANK_6, RANK_8, Square},
        moves::Move,
    },
    debug_index,
    engine::{
        history::{ContContext, History},
        search::SearchConfig,
        see::see_ge_with,
    },
};

// Ensure move bit-packing assumes correctly.
const _: () = assert!(std::mem::size_of::<Move>() == 2);

// ── Staged Move Picker ──
//
// Pipeline:
//   [ Hash Move ] ──> [ Good Captures ] ──> [ Quiets ] ──> [ Bad Captures ]
//      (𝒪(1))          (MVV-LVA Sort)     (History Sort)  (SEE-losing, deferred)
//
// We fully sort the generated stages using Rust's sort_unstable.
// Why not a lazy partial selection sort to save cycles on early cutoffs?
// Because Big-O is a lie when it hits modern hardware.
// An ipnsort beats a branch-heavy selection sort loop, even when K is small.
//
// Moves and their scores are cleanly bitpacked into u32s. Native sorting
// places the highest-scored moves at the end of the array, allowing us
// to pop them off the back (count -= 1) with zero index-shifting overhead.
#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Hash,
    GenCaptures,
    YieldCaptures,
    GenQSearchQuiets,
    GenQuiets,
    YieldQuiets,
    YieldBadCaptures,
    Done,
}

pub struct MovePicker {
    stage: Stage,
    hash_move: Option<Move>,
    candidates: [MaybeUninit<u32>; MAX_MOVES],
    count: usize,
    /// SEE-losing captures parked at the array top, drained in `YieldBadCaptures`.
    bad_count: usize,
    /// A capture orders before quiets when its SEE is at least `-good_capture_margin`;
    /// losing more than that defers it to the bad-capture stage.
    good_capture_margin: i32,
    mvvlva_v: [i32; 8],
    mvvlva_a: [i32; 8],
    mvvlva_ep: i32,
    capt_hist_divisor: i32,
    /// The node's pins, so the good/bad split's SEE calls don't each rescan.
    pins: Pins,
    killers: [Move; 2],
    threats: Bitboard,
    cont1: ContContext,
    cont2: ContContext,
    cont4: ContContext,
    is_qsearch: bool,
    in_check: bool,
    /// Quiets generated/sorted at this node, for `mvpstats`.
    #[cfg(feature = "mvpstats")]
    quiets_gen: u32,
    /// Quiets yielded before the picker is dropped (cutoff or exhaustion).
    #[cfg(feature = "mvpstats")]
    quiets_used: u32,
}

#[cfg(feature = "mvpstats")]
impl Drop for MovePicker {
    fn drop(&mut self) {
        if self.quiets_gen > 0 {
            crate::engine::mvpstats::record_quiets(self.quiets_gen, self.quiets_used);
        }
    }
}

impl MovePicker {
    #[inline]
    pub fn new(
        hash_move: Option<Move>,
        cfg: &SearchConfig,
        pins: Pins,
        killers: [Move; 2],
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
    ) -> Self {
        Self {
            stage: Stage::Hash,
            hash_move,
            candidates: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
            bad_count: 0,
            good_capture_margin: cfg.search_params.good_capture_margin,
            mvvlva_v: cfg.mvvlva_v,
            mvvlva_a: cfg.mvvlva_a,
            mvvlva_ep: cfg.search_params.mvvlva_ep,
            capt_hist_divisor: cfg.search_params.capt_hist_divisor,
            pins,
            killers,
            threats,
            cont1,
            cont2,
            cont4,
            is_qsearch: false,
            in_check: false,
            #[cfg(feature = "mvpstats")]
            quiets_gen: 0,
            #[cfg(feature = "mvpstats")]
            quiets_used: 0,
        }
    }

    #[inline]
    pub fn new_qsearch(hash_move: Option<Move>, cfg: &SearchConfig, pins: Pins, in_check: bool) -> Self {
        Self {
            stage: Stage::Hash,
            hash_move,
            candidates: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
            bad_count: 0,
            good_capture_margin: cfg.search_params.good_capture_margin,
            mvvlva_v: cfg.mvvlva_v,
            mvvlva_a: cfg.mvvlva_a,
            mvvlva_ep: cfg.search_params.mvvlva_ep,
            capt_hist_divisor: cfg.search_params.capt_hist_divisor,
            pins,
            killers: [Move::null(); 2],
            threats: Bitboard(0),
            cont1: ContContext::default(),
            cont2: ContContext::default(),
            cont4: ContContext::default(),
            is_qsearch: true,
            in_check,
            #[cfg(feature = "mvpstats")]
            quiets_gen: 0,
            #[cfg(feature = "mvpstats")]
            quiets_used: 0,
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
                    self.gen_captures(board, history);
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
                            slice::from_raw_parts_mut(ptr, self.count).sort_unstable();
                        }
                    }
                    self.stage = Stage::YieldCaptures;
                },
                Stage::YieldCaptures => {
                    if self.count == 0 {
                        // INVARIANT: when YieldCaptures is exhausted, count is exactly 0,
                        // so GenQuiets reuses the array from index 0 without a clear. The
                        // deferred bad captures sit at the array top, a region the quiets
                        // never reach (their combined count can't exceed MAX_MOVES).
                        if self.is_qsearch && !self.in_check {
                            self.stage = Stage::GenQSearchQuiets;
                        } else {
                            self.stage = Stage::GenQuiets;
                        }

                        continue;
                    }
                    // Pop from the back; since the array is sorted ascending,
                    // the highest-scored moves sit at the end. Popping via count -= 1
                    // is 𝒪(1) and avoids the expensive index-shifting of remove(0).
                    self.count -= 1;
                    // SAFETY: count was strictly > 0 above, so this index holds a valid packed move.
                    let packed = unsafe { debug_index!(self.candidates, self.count).assume_init() };
                    let mv = Move::from_u16((packed & 0xFFFF) as u16);

                    if Some(mv) == self.hash_move {
                        continue;
                    }

                    // ── Good / Bad Capture Split ──
                    // A SEE-losing capture rarely deserves to jump ahead of every quiet; a
                    // killer or a high-history move usually refutes the node first. Defer it:
                    // park the packed entry at the array top, which the quiets never fill, and
                    // drain it last in YieldBadCaptures. Promotions stay up front regardless,
                    // they win a piece outright. Qsearch opts out, its own SEE prune already
                    // culls losing captures, so a second SEE here would be wasted.
                    if !self.is_qsearch && !mv.is_promotion() {
                        let attacker = board.piece_at(mv.from());
                        let victim = if mv.is_en_passant() { PieceType::Pawn } else { board.piece_at(mv.to()) };
                        let victim_val = *debug_index!(self.mvvlva_v, victim as usize);
                        let attacker_val = *debug_index!(self.mvvlva_v, attacker as usize);

                        // A capture winning material (victim worth at least the attacker) has
                        // SEE >= 0, so it's good without the exchange walk; only material-losing
                        // captures are ambiguous enough to need SEE. The common cutoff-causing
                        // captures sort to the front and skip it entirely.
                        if victim_val < attacker_val && !see_ge_with(board, mv, -self.good_capture_margin, &self.pins) {
                            // SAFETY: count + bad_count <= total moves <= MAX_MOVES, so the park
                            // slot MAX_MOVES-1-bad_count is always >= count, never aliasing the
                            // active region [0, count) nor the quiets that later fill it.
                            unsafe {
                                self.candidates
                                    .as_mut_ptr()
                                    .add(MAX_MOVES - 1 - self.bad_count)
                                    .write(MaybeUninit::new(packed));
                            }
                            self.bad_count += 1;
                            continue;
                        }
                    }

                    return Some(mv);
                },

                Stage::GenQSearchQuiets => {
                    self.gen_qsearch_quiets(board, history);
                    // SAFETY: self.count accurately tracks initialized items.
                    if self.count > 1 {
                        unsafe {
                            let ptr = self.candidates.as_mut_ptr() as *mut u32;
                            std::slice::from_raw_parts_mut(ptr, self.count).sort_unstable();
                        }
                    }
                    self.stage = Stage::YieldQuiets;
                },

                Stage::GenQuiets => {
                    self.gen_quiets(board, history);

                    #[cfg(feature = "mvpstats")]
                    {
                        self.quiets_gen = self.count as u32;
                    }
                    // SAFETY: self.count accurately tracks initialized items.
                    // ptr covers valid memory and is sorted in-place.
                    if self.count > 1 {
                        unsafe {
                            let ptr = self.candidates.as_mut_ptr() as *mut u32;
                            slice::from_raw_parts_mut(ptr, self.count).sort_unstable();
                        }
                    }
                    self.stage = Stage::YieldQuiets;
                },
                Stage::YieldQuiets => {
                    if self.count == 0 {
                        // Quiets done; reuse count as a top-down cursor over the parked bad
                        // captures. It halts at MAX_MOVES - bad_count, draining none when the
                        // split deferred nothing (or in qsearch, where bad_count stays 0).
                        self.count = MAX_MOVES;
                        self.stage = Stage::YieldBadCaptures;
                        continue;
                    }
                    // Pop from the back: exploits the ascending sort to yield the strongest moves first.
                    self.count -= 1;
                    // SAFETY: self.count was strictly checked > 0 above, proving this index holds a valid move.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.hash_move {
                        #[cfg(feature = "mvpstats")]
                        {
                            self.quiets_used += 1;
                        }
                        return Some(mv);
                    }
                },
                Stage::YieldBadCaptures => {
                    // The deferred losing captures occupy [MAX_MOVES - bad_count, MAX_MOVES),
                    // best at the top since YieldCaptures parked them in descending score.
                    // count walks down from MAX_MOVES, yielding best-first, until drained.
                    if self.count == MAX_MOVES - self.bad_count {
                        self.stage = Stage::Done;
                        continue;
                    }

                    self.count -= 1;
                    // SAFETY: count is in (MAX_MOVES - bad_count, MAX_MOVES], every slot
                    // written by the park step in YieldCaptures.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.hash_move {
                        return Some(mv);
                    }
                },
                Stage::Done => return None,
            }
        }
    }

    #[inline]
    fn gen_captures(&mut self, board: &Position, history: &History) {
        let stm = board.stm;
        let us = board.side_bb[stm];
        // king captures are never legal
        let them = board.side_bb[stm.opposite()] & !board.role_bb[PieceType::King];
        let occ = board.occ;

        self.gen_pawn_caps(board, us, them, history);
        self.gen_piece_caps::<{ PieceType::Knight }>(board, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Bishop }>(board, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Rook }>(board, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Queen }>(board, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::King }>(board, us, them, occ, history);
    }

    #[inline(always)]
    fn gen_pawn_caps(&mut self, board: &Position, us: Bitboard, them: Bitboard, history: &History) {
        let stm = board.stm;
        let pawns = board.role_bb[PieceType::Pawn] & us;

        let targets = if stm == Color::White {
            [(9i8, (pawns & NOT_H) << 9 & them), (7i8, (pawns & NOT_A) << 7 & them)]
        } else {
            [(-7i8, (pawns & NOT_H) >> 7 & them), (-9i8, (pawns & NOT_A) >> 9 & them)]
        };

        let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };

        for (delta, victims) in targets {
            let promo = victims & prom_mask;
            let standard = victims & !prom_mask;

            // Promotion-captures bypass capture history entirely (see add_promo_caps).
            for to in promo {
                let from = Square((to.0 as i8 - delta) as u8);
                self.add_promo_caps(board, from, to);
            }

            for to in standard {
                let from = Square((to.0 as i8 - delta) as u8);
                self.add_cap(board, Move::new(from, to, Move::CAPTURE), PieceType::Pawn, history);
            }
        }

        // Victim is always a pawn: the captured pawn sits on an adjacent square, not the destination.
        if let Some(ep_sq) = board.en_passant {
            for from in atk_pawn(ep_sq, stm.opposite()) & pawns {
                self.add_cap(board, Move::new(from, ep_sq, Move::EP_CAPTURE), PieceType::Pawn, history);
            }
        }
    }

    /// Blend MVV-LVA with capture history into a single sort score.
    /// Single source of truth for the capture ordering formula.
    #[inline(always)]
    fn cap_score(&self, mvv: i32, chist: i32) -> i32 {
        mvv + chist / self.capt_hist_divisor
    }

    /// Monomorphized by `PT` for dispatch-free attack lookups.
    #[inline]
    fn gen_piece_caps<const PT: PieceType>(
        &mut self,
        board: &Position,
        us: Bitboard,
        them: Bitboard,
        occ: Bitboard,
        history: &History,
    ) {
        let a_pen = *crate::debug_index!(self.mvvlva_a, PT as usize);
        let stm = board.stm;

        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(from, occ) & them {
                let victim = board.piece_at(to);
                let v_val = *crate::debug_index!(self.mvvlva_v, victim as usize);
                let chist = history.score_capture(stm, PT, to, victim);
                let score = self.cap_score(v_val - a_pen, chist);

                self.add_move_packed(Move::new(from, to, Move::CAPTURE), score as MoveScore);
            }
        }
    }

    #[inline(always)]
    fn add_move_packed(&mut self, mv: Move, score: MoveScore) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");
        let sort_score = (score as i32 + 32768).clamp(0, 65535) as u32;
        let packed = (sort_score << 16) | (mv.inner() as u32);
        crate::debug_index_mut!(self.candidates, self.count).write(packed);

        self.count += 1;
    }

    /// Append a capture, scored as `MVV-LVA + capture_history / divisor`.
    /// Promotion-captures bypass this path entirely; they go through `add_promo_caps`.
    #[inline]
    fn add_cap(&mut self, board: &Position, mv: Move, attacker: PieceType, history: &History) {
        // Victim is always pawn (the captured pawn sits on an adjacent square, not the destination).
        let victim = if mv.is_en_passant() { PieceType::Pawn } else { board.piece_at(mv.to()) };
        let mvv = self.mvv_lva(mv, attacker, victim);
        let chist = history.score_capture(board.stm, attacker, mv.to(), victim);

        self.add_move_packed(mv, self.cap_score(mvv as i32, chist) as MoveScore);
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

    /// Most Valuable Victim - Least Valuable Attacker: `V(victim) - V(attacker)`.
    ///
    /// Promotion-captures never reach this function; they're scored entirely
    /// by `add_promo_caps`, which is why there's no promotion term here.
    /// The Stage segregation in `MovePicker::next` ensures all captures are
    /// yielded before any quiet moves, so a global bias is no longer needed.
    #[inline(always)]
    fn mvv_lva(&self, mv: Move, attacker: PieceType, victim: PieceType) -> MoveScore {
        if mv.is_en_passant() {
            return self.mvvlva_ep as MoveScore;
        }

        debug_assert!(mv.promo().is_none(), "mvv_lva is only reached by non-promotion captures");
        debug_assert!(usize::from(victim) < 8);
        debug_assert!(usize::from(attacker) < 8);

        let v = *crate::debug_index!(self.mvvlva_v, victim as usize);
        let a = *crate::debug_index!(self.mvvlva_a, attacker as usize);

        (v - a) as MoveScore
    }

    /// Generate only quiet queen promotions for QSearch.
    #[inline]
    fn gen_qsearch_quiets(&mut self, board: &Position, history: &History) {
        let us = board.side_bb[board.stm];
        let occ = board.occ;
        let empty = !occ;

        let stm = board.stm;
        let up = stm.forward_dir();
        let up_d = up.delta();

        let pawns = board.role_bb[PieceType::Pawn] & us;
        let all_pushes = pawns.shift(up) & empty;

        let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };
        let promo_pushes = all_pushes & prom_mask;

        for to in promo_pushes {
            let from = Square((to.0 as i8 - up_d) as u8);
            self.add_quiet_node(Move::new(from, to, Move::PROM_Q), PieceType::Pawn, stm, history);
        }
    }

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

    #[inline]
    fn add_quiet(&mut self, board: &Position, mv: Move, history: &History) {
        let pt = board.expect_piece_at(mv.from());
        self.add_quiet_node(mv, pt, board.stm, history);
    }

    #[inline(always)]
    fn add_quiet_node(&mut self, mv: Move, pt: PieceType, stm: Color, history: &History) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");

        let score = history.score_quiet(stm, pt, mv.from(), mv.to(), self.threats, self.cont1, self.cont2, self.cont4);

        // Combined history values stay well inside [-32768, 32768] in practice.
        // Soft-gravity attractors prevent any single table from sitting near its
        // ±16384 cap, so even with four tables the summed range never approaches
        // the sort band edges. Measured on bench: zero saturation in ~11M quiets.
        // Clamped at 63000 so the [64000, 65535] band stays exclusive for
        // killers and promotions.
        let mut sort_score = (score + 32768).clamp(0, 63000) as u32;

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
        } else if mv == self.killers[0] {
            sort_score = 65000;
        } else if mv == self.killers[1] {
            sort_score = 64000;
        }

        let packed = (sort_score << 16) | (mv.inner() as u32);

        // SAFETY: count < MAX_MOVES is guarded above.
        unsafe { self.candidates.as_mut_ptr().add(self.count).write(MaybeUninit::new(packed)) };
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

    /// Monomorphized by `PT` for dispatch-free attack lookups.
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

    #[inline]
    fn gen_castling(&mut self, board: &Position, us: Bitboard, occ: Bitboard, history: &History) {
        let king_bb = board.role_bb[PieceType::King] & us;
        if king_bb.is_empty() {
            return;
        }

        let stm = board.stm;
        let ksq = king_bb.lsb();
        let opp = stm.opposite();

        let (oo_mask, ooo_mask, ks_idx, qs_idx) =
            if stm == Color::White { (WHITE_OO, WHITE_OOO, 0, 1) } else { (BLACK_OO, BLACK_OOO, 2, 3) };

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

    /// Extracts the move from the packed item at `i`.
    #[inline(always)]
    unsafe fn read_move(&self, i: usize) -> Move {
        debug_assert!(i < MAX_MOVES);
        // SAFETY: The caller contract of read_move requires that index i has been successfully initialized.
        let packed = unsafe { debug_index!(self.candidates, i).assume_init() };
        Move::from_u16(packed as u16)
    }

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
