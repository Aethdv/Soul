//! Staged move generation and heuristic ordering.
//!
//! Most search nodes cut off on the hash move or the first good capture; generating
//! quiets eagerly wastes work. The picker yields moves in stages, best guesses first,
//! and generates a stage only when the earlier ones are exhausted.
//!
//! | Stage         | Content                       | Sorting           |
//! |---------------|-------------------------------|-------------------|
//! | `Hash`        | PV move from prior iteration  | Exact match       |
//! | `Captures`    | SEE-winning captures & promos | MVV-LVA           |
//! | `Quiets`      | Non-captures                  | History heuristic |
//! | `BadCaptures` | SEE-losing captures, deferred | MVV-LVA           |
//!
//! Moves are bitpacked with their heuristic scores into `u32` values and sorted
//! ascending with `sort_unstable` (ipnsort). This allows popping the highest-scored
//! moves from the back in 𝒪(1) time without index shifting.

use std::mem::MaybeUninit;

#[cfg(not(feature = "nostore"))]
use crate::core::board::xorboard::XorBoard;
use crate::{
    core::{
        board::{
            Position,
            attacks::Pins,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook},
        },
        defs::{Bitboard, Color, MAX_MOVES, MoveScore, NOT_A, NOT_H, PieceType, RANK_1, RANK_3, RANK_6, RANK_8, Square},
        moves::Move,
    },
    debug_index, debug_index_mut,
    engine::{
        history::{ContContext, History},
        movegen::is_pseudo_legal,
        search::SearchConfig,
        see::see_ge,
    },
};

fn promotion_outranks_killers(board: &Position) -> bool {
    let stm = board.stm;
    let pawns = board.role_bb[PieceType::Pawn] & board.side_bb[stm];
    let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };
    (pawns.shift(stm.forward_dir()) & !board.occ & prom_mask).is_not_empty()
}

/// Where a slider's attacks come from. `nostore` has no store to read, so the argument
/// holds nothing and every slider falls back to its probe.
#[cfg(not(feature = "nostore"))]
type Rows<'a> = &'a XorBoard;
#[cfg(feature = "nostore")]
type Rows<'a> = core::marker::PhantomData<&'a ()>;

const _: () = assert!(std::mem::size_of::<Move>() == 2);

// Sort scores use non-overlapping bands: history quiets < killers < promotions.
// History values are clamped below the killer floor to prevent band intrusion.
const SORT_BIAS: i32 = 32768;
const QUIET_SCORE_MAX: i32 = 63000;
const KILLER_SCORES: [u32; 2] = [65000, 64000];
const PROMO_B_SCORE: u32 = 65532;
const PROMO_R_SCORE: u32 = 65533;
const PROMO_N_SCORE: u32 = 65534;
const PROMO_Q_SCORE: u32 = 65535;

// An edit that overlaps two bands fails the build instead of the search.
const _: () = {
    assert!(QUIET_SCORE_MAX < KILLER_SCORES[1] as i32);
    assert!(KILLER_SCORES[1] < KILLER_SCORES[0]);
    assert!(KILLER_SCORES[0] < PROMO_B_SCORE);
    assert!(PROMO_B_SCORE < PROMO_R_SCORE);
    assert!(PROMO_R_SCORE < PROMO_N_SCORE);
    assert!(PROMO_N_SCORE < PROMO_Q_SCORE);
    assert!(PROMO_Q_SCORE <= u16::MAX as u32);
    // The bias alone puts an i16 capture score inside the band, so packing needs no clamp.
    assert!(MoveScore::MIN as i32 + SORT_BIAS == 0);
    assert!(MoveScore::MAX as i32 + SORT_BIAS <= u16::MAX as i32);
};

#[repr(u8)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Stage {
    Hash,
    GenCaptures,
    YieldCaptures,
    Killers,
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
    killer_idx: usize,
    killers_taken: [bool; 2],
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
            killer_idx: 0,
            killers_taken: [false; 2],
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

    // Duplicating the struct literal avoids a ~1KB stack copy (~1% NPS loss).
    // Rust lacks guaranteed copy elision; returning an explicit literal is the only
    // shape rustc reliably constructs in place within the caller's frame.
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
            killer_idx: 0,
            killers_taken: [false; 2],
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

    /// Yields the next move in priority order, or `None` when exhausted.
    #[inline(always)]
    pub fn next(&mut self, board: &Position, rows: Rows<'_>, history: &History) -> Option<Move> {
        loop {
            match self.stage {
                // ── TT Move Ordering (~56 Elo)
                // The move a previous search stored is the best guess available here,
                // and yielding it ahead of generation means a cutoff can land before a
                // single move has been generated.
                Stage::Hash => {
                    self.stage = Stage::GenCaptures;
                    if let Some(mv) = self.hash_move {
                        return Some(mv);
                    }
                },

                Stage::GenCaptures => {
                    self.gen_captures(board, rows, history);
                    // Captures sort in their own stage; MVV-LVA scores do not need to align with quiet bands.
                    self.sort_candidates();
                    self.stage = Stage::YieldCaptures;
                },

                Stage::YieldCaptures => {
                    if self.count == 0 {
                        // Array is exhausted (count == 0), so GenQuiets reuses index 0 without clearing.
                        // Deferred bad captures sit safely at the high end (total moves <= MAX_MOVES).
                        self.stage = if self.is_qsearch && !self.in_check { Stage::GenQSearchQuiets } else { Stage::Killers };
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: count was non-zero; index holds an initialized packed move.
                    let packed = unsafe { debug_index!(self.candidates, self.count).assume_init() };
                    let mv = Move::from_u16(packed as u16);
                    if Some(mv) == self.hash_move {
                        continue;
                    }

                    // ── Good / Bad Capture Split
                    // Defer SEE-losing captures: killers and strong quiets usually refute the
                    // node first. Promotions bypass this check (winning material is guaranteed).
                    // Qsearch bypasses this check because its standalone SEE prune already handles it.
                    if !self.is_qsearch && !mv.is_promotion() {
                        let attacker = board.piece_at(mv.from());
                        let victim = if mv.is_en_passant() { PieceType::Pawn } else { board.piece_at(mv.to()) };

                        // Use victim table for both to compare values on the same absolute scale.
                        let victim_val = *debug_index!(self.mvvlva_v, victim as usize);
                        let attacker_val = *debug_index!(self.mvvlva_v, attacker as usize);

                        // If victim >= attacker, SEE >= 0 is guaranteed; only material-losing captures need SEE.
                        if victim_val < attacker_val && !see_ge(board, mv, -self.good_capture_margin, &self.pins) {
                            // SAFETY: count + bad_count <= MAX_MOVES, so MAX_MOVES - 1 - bad_count >= count.
                            // The park slot never aliases active captures or subsequent quiets.
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

                // ── Killer Stage
                // Both killers outrank every history-scored quiet, so the order is unchanged;
                // a cutoff here skips generating and scoring the node's quiets entirely.
                Stage::Killers => {
                    if self.killer_idx == 0 && promotion_outranks_killers(board) {
                        self.stage = Stage::GenQuiets;
                        continue;
                    }
                    if self.killer_idx == self.killers.len() {
                        self.stage = Stage::GenQuiets;
                        continue;
                    }
                    let i = self.killer_idx;
                    let mv = self.killers[i];
                    self.killer_idx += 1;
                    if !mv.is_null() && Some(mv) != self.hash_move && is_pseudo_legal(board, mv) {
                        self.killers_taken[i] = true;
                        return Some(mv);
                    }
                },

                Stage::GenQSearchQuiets => {
                    self.gen_qsearch_quiets(board, history);
                    self.sort_candidates();
                    self.stage = Stage::YieldQuiets;
                },

                Stage::GenQuiets => {
                    self.gen_quiets(board, rows, history);
                    #[cfg(feature = "mvpstats")]
                    {
                        self.quiets_gen = self.count as u32;
                    }
                    self.sort_candidates();
                    self.stage = Stage::YieldQuiets;
                },

                Stage::YieldQuiets => {
                    if self.count == 0 {
                        // Quiets exhausted. Reset count to MAX_MOVES to drain parked bad captures top-down.
                        self.count = MAX_MOVES;
                        self.stage = Stage::YieldBadCaptures;
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: count was non-zero; index holds an initialized move.
                    let mv = unsafe { self.read_move(self.count) };

                    let taken =
                        (self.killers_taken[0] && mv == self.killers[0]) || (self.killers_taken[1] && mv == self.killers[1]);
                    if Some(mv) != self.hash_move && !taken {
                        #[cfg(feature = "mvpstats")]
                        {
                            self.quiets_used += 1;
                        }
                        return Some(mv);
                    }
                },

                Stage::YieldBadCaptures => {
                    // Deferred captures occupy [MAX_MOVES - bad_count, MAX_MOVES), ordered descending.
                    // Draining downwards from MAX_MOVES yields best-first.
                    if self.count == MAX_MOVES - self.bad_count {
                        self.stage = Stage::Done;
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: index is within [MAX_MOVES - bad_count, MAX_MOVES), initialized in YieldCaptures.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.hash_move {
                        return Some(mv);
                    }
                },

                Stage::Done => return None,
            }
        }
    }

    /// Sorts the live candidate window ascending, so the best moves land at the end.
    ///
    /// A lazy partial selection sort is the obvious alternative for a node that cuts on its
    /// first move. It loses: the asymptotics say one pass beats a full sort, and the branch
    /// predictor disagrees, so ipnsort wins even when K is small.
    #[inline]
    fn sort_candidates(&mut self) {
        if self.count > 1 {
            // SAFETY: the gen step just wrote the first count entries; the rest
            // of the array stays uninitialized and out of the sorted slice.
            unsafe { self.candidates[..self.count].assume_init_mut() }.sort_unstable();
        }
    }

    #[inline]
    fn gen_captures(&mut self, board: &Position, rows: Rows<'_>, history: &History) {
        let stm = board.stm;
        let us = board.side_bb[stm];
        // king captures are never legal
        let them = board.side_bb[stm.opposite()] & !board.role_bb[PieceType::King];
        let occ = board.occ;

        self.gen_pawn_caps(board, us, them, history);
        self.gen_piece_caps::<{ PieceType::Knight }>(board, rows, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Bishop }>(board, rows, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Rook }>(board, rows, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::Queen }>(board, rows, us, them, occ, history);
        self.gen_piece_caps::<{ PieceType::King }>(board, rows, us, them, occ, history);
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
                let from = Square((to.0.cast_signed() - delta).cast_unsigned());
                self.add_promo_caps(board, from, to);
            }
            for to in standard {
                let from = Square((to.0.cast_signed() - delta).cast_unsigned());
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

    /// Blends MVV-LVA with capture history into one sort score.
    /// Both capture paths score through here, so the formula has one home.
    #[inline(always)]
    fn cap_score(&self, mvv: i32, chist: i32) -> i32 { mvv + chist / self.capt_hist_divisor }

    /// Monomorphized by `PT` for dispatch-free attack lookups.
    #[inline]
    fn gen_piece_caps<const PT: PieceType>(
        &mut self,
        board: &Position,
        rows: Rows<'_>,
        us: Bitboard,
        them: Bitboard,
        occ: Bitboard,
        history: &History,
    ) {
        let a_pen = *debug_index!(self.mvvlva_a, PT as usize);
        let stm = board.stm;

        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(rows, from, occ) & them {
                let victim = board.piece_at(to);
                let v_val = *debug_index!(self.mvvlva_v, victim as usize);
                let chist = history.score_capture(stm, PT, to, victim);
                let score = self.cap_score(v_val - a_pen, chist);
                self.add_move_packed(Move::new(from, to, Move::CAPTURE), score as MoveScore);
            }
        }
    }

    #[inline(always)]
    fn add_move_packed(&mut self, mv: Move, score: MoveScore) { self.write_packed(pack((score as i32 + SORT_BIAS) as u32, mv)); }

    /// Append a pre-packed `(sort_score << 16) | move` entry.
    #[inline(always)]
    fn write_packed(&mut self, packed: u32) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");
        debug_index_mut!(self.candidates, self.count).write(packed);
        self.count += 1;
    }

    /// Appends one capture. Promotion-captures never come through here.
    #[inline]
    fn add_cap(&mut self, board: &Position, mv: Move, attacker: PieceType, history: &History) {
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
        let v_val = *debug_index!(self.mvvlva_v, victim as usize);
        let a_pen = *debug_index!(self.mvvlva_a, PieceType::Pawn as usize);
        let base = v_val - a_pen;

        let q = base + *debug_index!(self.mvvlva_v, PieceType::Queen as usize);
        let r = base + *debug_index!(self.mvvlva_v, PieceType::Rook as usize);
        let b = base + *debug_index!(self.mvvlva_v, PieceType::Bishop as usize);
        let n = base + *debug_index!(self.mvvlva_v, PieceType::Knight as usize);

        self.add_move_packed(Move::new(from, to, Move::PROM_Q_CAPTURE), q as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_R_CAPTURE), r as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_B_CAPTURE), b as MoveScore);
        self.add_move_packed(Move::new(from, to, Move::PROM_N_CAPTURE), n as MoveScore);
    }

    /// Most Valuable Victim - Least Valuable Attacker: `V(victim) - V(attacker)`.
    ///
    /// Promotion-captures are scored entirely by `add_promo_caps`, which is why there is no
    /// promotion term here.
    #[inline(always)]
    fn mvv_lva(&self, mv: Move, attacker: PieceType, victim: PieceType) -> MoveScore {
        if mv.is_en_passant() {
            return self.mvvlva_ep as MoveScore;
        }
        debug_assert!(mv.promo().is_none(), "mvv_lva is only reached by non-promotion captures");
        debug_assert!(usize::from(victim) < self.mvvlva_v.len());
        debug_assert!(usize::from(attacker) < self.mvvlva_a.len());
        let v = *debug_index!(self.mvvlva_v, victim as usize);
        let a = *debug_index!(self.mvvlva_a, attacker as usize);
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
            let from = Square((to.0.cast_signed() - up_d).cast_unsigned());
            self.add_quiet_node(Move::new(from, to, Move::PROM_Q), PieceType::Pawn, stm, history);
        }
    }

    /// Generate all non-capturing pseudo-legal moves, including castling.
    fn gen_quiets(&mut self, board: &Position, rows: Rows<'_>, history: &History) {
        let us = board.side_bb[board.stm];
        let occ = board.occ;
        let empty = !occ;

        self.gen_pawn_quiets(board, us, empty, history);
        self.gen_piece_quiets::<{ PieceType::Knight }>(board, rows, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Bishop }>(board, rows, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Rook }>(board, rows, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::Queen }>(board, rows, us, empty, occ, history);
        self.gen_piece_quiets::<{ PieceType::King }>(board, rows, us, empty, occ, history);
        board.for_each_castle(board.stm, |mv| self.add_quiet(board, mv, history));
    }

    #[inline]
    fn add_quiet(&mut self, board: &Position, mv: Move, history: &History) {
        let pt = board.expect_piece_at(mv.from());
        self.add_quiet_node(mv, pt, board.stm, history);
    }

    #[inline(always)]
    fn add_quiet_node(&mut self, mv: Move, pt: PieceType, stm: Color, history: &History) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");

        // A promotion or a killer outranks any history score, so it takes a fixed band. The
        // knight sits above rook and bishop because its fork is the underpromotion that wins.
        let sort_score = if mv.is_promotion() {
            match mv.promo() {
                Some(PieceType::Queen) => PROMO_Q_SCORE,
                Some(PieceType::Knight) => PROMO_N_SCORE,
                Some(PieceType::Rook) => PROMO_R_SCORE,
                _ => PROMO_B_SCORE,
            }
        } else if mv == self.killers[0] {
            KILLER_SCORES[0]
        } else if mv == self.killers[1] {
            KILLER_SCORES[1]
        } else {
            let score = history.score_quiet(stm, pt, mv.from(), mv.to(), self.threats, self.cont1, self.cont2, self.cont4);
            // Four tables sum here and the clamp still never fires: soft-gravity attractors
            // keep each one away from its own ±16384 cap. Measured on bench, zero saturation
            // in ~11M quiets.
            (score + SORT_BIAS).clamp(0, QUIET_SCORE_MAX) as u32
        };

        self.write_packed(pack(sort_score, mv));
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
            let from = Square((to.0.cast_signed() - up_d).cast_unsigned());
            self.add_promo_quiets(from, to, stm, history);
        }
        for to in quiet_pushes {
            let from = Square((to.0.cast_signed() - up_d).cast_unsigned());
            self.add_quiet_node(Move::new(from, to, Move::QUIET), PieceType::Pawn, stm, history);
        }

        let doubles = (all_pushes & third_rank).shift(up) & empty;
        for to in doubles {
            let from = Square((to.0.cast_signed() - up_d * 2).cast_unsigned());
            self.add_quiet_node(Move::new(from, to, Move::DOUBLE_PUSH), PieceType::Pawn, stm, history);
        }
    }

    /// Monomorphized by `PT` for dispatch-free attack lookups.
    #[inline]
    fn gen_piece_quiets<const PT: PieceType>(
        &mut self,
        board: &Position,
        rows: Rows<'_>,
        us: Bitboard,
        empty: Bitboard,
        occ: Bitboard,
        history: &History,
    ) {
        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(rows, from, occ) & empty {
                self.add_quiet_node(Move::new(from, to, Move::QUIET), PT, board.stm, history);
            }
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
    fn attacks<const PT: PieceType>(rows: Rows<'_>, from: Square, occ: Bitboard) -> Bitboard {
        match PT {
            PieceType::Pawn => {
                debug_assert!(false, "Pawn attacks must use dedicated generators");
                Bitboard(0)
            },
            PieceType::Knight => atk_knight(from),
            PieceType::King => atk_king(from),
            PieceType::None => Bitboard(0),
            _ => Self::slider_attacks::<PT>(rows, from, occ),
        }
    }

    /// Sliders read precomputed attacks from the row store, bypassing magic bitboards.
    /// Leapers (`atk_knight`) use direct lookups instead to avoid the mailbox-then-row
    /// double load.
    ///
    /// The fallback is not dead code. Row data is derived state; because the type
    /// system cannot enforce board-store coherence, the fallback prevents generating
    /// attacks from an empty row if the store desynchronizes.
    #[cfg(not(feature = "nostore"))]
    #[inline(always)]
    fn slider_attacks<const PT: PieceType>(rows: &XorBoard, from: Square, occ: Bitboard) -> Bitboard {
        match rows.id_at(from) {
            Some(id) => rows.row(id),
            None => Self::probe::<PT>(from, occ),
        }
    }

    #[cfg(feature = "nostore")]
    #[inline(always)]
    fn slider_attacks<const PT: PieceType>(_rows: Rows<'_>, from: Square, occ: Bitboard) -> Bitboard {
        Self::probe::<PT>(from, occ)
    }

    #[inline(always)]
    fn probe<const PT: PieceType>(from: Square, occ: Bitboard) -> Bitboard {
        match PT {
            PieceType::Bishop => atk_bishop(from, occ),
            PieceType::Rook => atk_rook(from, occ),
            PieceType::Queen => atk_bishop(from, occ) | atk_rook(from, occ),
            _ => Bitboard(0),
        }
    }
}

/// Sort score in the high half, move in the low, so an ascending integer sort orders by
/// score.
#[inline(always)]
fn pack(sort_score: u32, mv: Move) -> u32 {
    debug_assert!(sort_score <= u32::from(u16::MAX), "sort score outside the packed band");
    (sort_score << 16) | u32::from(mv.inner())
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicBool},
        time::Instant,
    };

    use super::*;
    use crate::{
        core::board::{STARTPOS, xorboard::XorBoard},
        engine::{movegen::gen_pseudo_moves, search::Limits, search_params::SearchParams},
    };

    #[cfg(not(feature = "nostore"))]
    fn rows(store: &XorBoard) -> Rows<'_> { store }

    #[cfg(feature = "nostore")]
    fn rows(_store: &XorBoard) -> Rows<'static> { core::marker::PhantomData }

    #[test]
    fn every_stage_together_yields_what_movegen_does() {
        const FENS: [&str; 6] = [
            STARTPOS,
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
            "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R b KQkq - 0 1",
            "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 1",
            "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
            "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        ];

        let cfg = SearchConfig::new(
            Limits { silent: true, ..Default::default() },
            Instant::now(),
            Arc::new(AtomicBool::new(false)),
            0,
            SearchParams::default(),
        );

        let history = History::new();

        for fen in FENS {
            let board = Position::from_fen(fen);
            let store = XorBoard::new(&board);

            let mut expected: Vec<u16> = gen_pseudo_moves(&board).iter().map(|mv| mv.inner()).collect();
            expected.sort_unstable();

            let quiets: Vec<Move> = gen_pseudo_moves(&board).iter().copied().filter(|mv| mv.is_history_quiet()).collect();
            let first = quiets.first().copied().unwrap_or(Move::null());
            let second = quiets.get(1).copied().unwrap_or(Move::null());
            let last = quiets.last().copied().unwrap_or(Move::null());

            for (case, killers) in [[Move::null(); 2], [first, second], [last, Move::null()]].into_iter().enumerate() {
                let mut picker = MovePicker::new(
                    None,
                    &cfg,
                    Pins::new(&board),
                    killers,
                    Bitboard(0),
                    ContContext::default(),
                    ContContext::default(),
                    ContContext::default(),
                );

                let mut picked = Vec::new();
                while let Some(mv) = picker.next(&board, rows(&store), &history) {
                    picked.push(mv.inner());
                }

                picked.sort_unstable();
                assert_eq!(picked, expected, "{fen} killer case {case}");
            }
        }
    }
}
