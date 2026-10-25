//! Staged move generation and ordering.
//!
//! Later stages are generated only if earlier moves fail to produce a cutoff.

use std::mem::MaybeUninit;

#[cfg(not(feature = "nostore"))]
use crate::core::board::xorboard::XorBoard;
use crate::{
    core::{
        board::{
            Position,
            attacks::Pins,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_queen, atk_rook},
        },
        defs::{Bitboard, Color, Direction, MAX_MOVES, PieceType, RANK_1, RANK_3, RANK_6, RANK_8, Square},
        moves::Move,
    },
    debug_index, debug_index_mut,
    engine::{
        history::{ContContext, History},
        search::SearchConfig,
        see::see_ge,
    },
};

type MoveScore = i16;

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
    TtMove,
    GenCaptures,
    YieldCaptures,
    GenQSearchPromotions,
    GenQuiets,
    YieldQuiets,
    YieldBadCaptures,
    Done,
}

pub(crate) struct MovePicker {
    stage: Stage,
    tt_move: Option<Move>,
    candidates: [MaybeUninit<u32>; MAX_MOVES],
    count: usize,
    /// SEE-losing captures parked at the array top, drained in `YieldBadCaptures`.
    bad_count: usize,
    good_capture_margin: i32,
    mvvlva_victim: [i32; 8],
    mvvlva_attacker: [i32; 8],
    mvvlva_ep: i32,
    see_values: [i32; 8],
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
    pub(super) fn new(
        tt_move: Option<Move>,
        cfg: &SearchConfig,
        pins: Pins,
        killers: [Move; 2],
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
    ) -> Self {
        Self {
            stage: Stage::TtMove,
            tt_move,
            candidates: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
            bad_count: 0,
            good_capture_margin: cfg.search_params.good_capture_margin,
            mvvlva_victim: cfg.mvvlva_victim,
            mvvlva_attacker: cfg.mvvlva_attacker,
            mvvlva_ep: cfg.search_params.mvvlva_ep,
            see_values: cfg.see_values,
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

    // Duplicating the struct literal avoids a ~1KB stack copy (~1% NPS loss).
    #[inline]
    pub(super) fn new_qsearch(tt_move: Option<Move>, cfg: &SearchConfig, pins: Pins, in_check: bool) -> Self {
        Self {
            stage: Stage::TtMove,
            tt_move,
            candidates: [MaybeUninit::uninit(); MAX_MOVES],
            count: 0,
            bad_count: 0,
            good_capture_margin: cfg.search_params.good_capture_margin,
            mvvlva_victim: cfg.mvvlva_victim,
            mvvlva_attacker: cfg.mvvlva_attacker,
            mvvlva_ep: cfg.search_params.mvvlva_ep,
            see_values: cfg.see_values,
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

    #[inline(always)]
    pub(super) fn next(&mut self, board: &Position, rows: Rows<'_>, history: &History) -> Option<Move> {
        loop {
            match self.stage {
                // ── TT Move Ordering (~56 Elo)
                // A cutoff on the stored move avoids generating any moves.
                Stage::TtMove => {
                    self.stage = Stage::GenCaptures;
                    if let Some(mv) = self.tt_move {
                        return Some(mv);
                    }
                },

                Stage::GenCaptures => {
                    self.gen_captures(board, rows, history);
                    self.sort_candidates();
                    self.stage = Stage::YieldCaptures;
                },

                Stage::YieldCaptures => {
                    if self.count == 0 {
                        // The quiets fill from index 0 again. The deferred bad captures sit at the
                        // top of the array, and no position has MAX_MOVES moves to reach them.
                        self.stage = if self.is_qsearch && !self.in_check { Stage::GenQSearchPromotions } else { Stage::GenQuiets };
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: count was non-zero; index holds an initialized packed move.
                    let packed = unsafe { debug_index!(self.candidates, self.count).assume_init() };
                    let mv = Move::from_u16(packed as u16);
                    if Some(mv) == self.tt_move {
                        continue;
                    }

                    // ── Good / Bad Capture Split
                    // Defer SEE-losing captures: killers and strong quiets usually refute the
                    // node first. Promotions are never deferred, and qsearch runs its own SEE prune.
                    if !self.is_qsearch && !mv.is_promotion() {
                        let attacker = board.piece_at(mv.from());
                        let victim = board.captured_piece(mv);

                        // Both read the victim table, so they compare on one scale.
                        let victim_val = *debug_index!(self.mvvlva_victim, victim as usize);
                        let attacker_val = *debug_index!(self.mvvlva_victim, attacker as usize);

                        // Taking something worth at least the attacker cannot lose material.
                        if victim_val < attacker_val && !see_ge(board, mv, -self.good_capture_margin, &self.pins, &self.see_values)
                        {
                            // SAFETY: count + bad_count < MAX_MOVES after popping, so MAX_MOVES - 1 - bad_count >= count
                            // and the slot aliases neither a live capture nor a later quiet.
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

                Stage::GenQSearchPromotions => {
                    self.gen_qsearch_promotions(board, history);
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
                        self.count = MAX_MOVES;
                        self.stage = Stage::YieldBadCaptures;
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: count was non-zero; index holds an initialized move.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.tt_move {
                        #[cfg(feature = "mvpstats")]
                        {
                            self.quiets_used += 1;
                        }
                        return Some(mv);
                    }
                },

                Stage::YieldBadCaptures => {
                    // The deferred captures fill down from the top of the array in the order they
                    // were yielded, so draining down from MAX_MOVES returns them best first.
                    if self.count == MAX_MOVES - self.bad_count {
                        self.stage = Stage::Done;
                        continue;
                    }
                    self.count -= 1;
                    // SAFETY: index is within [MAX_MOVES - bad_count, MAX_MOVES), initialized in YieldCaptures.
                    let mv = unsafe { self.read_move(self.count) };

                    if Some(mv) != self.tt_move {
                        return Some(mv);
                    }
                },

                Stage::Done => return None,
            }
        }
    }

    /// Full sorting beat lazy partial selection even when few moves were searched.
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
            [(9i8, pawns.shift(Direction::NorthEast) & them), (7i8, pawns.shift(Direction::NorthWest) & them)]
        } else {
            [(-7i8, pawns.shift(Direction::SouthEast) & them), (-9i8, pawns.shift(Direction::SouthWest) & them)]
        };

        let prom_mask = if stm == Color::White { RANK_8 } else { RANK_1 };

        for (delta, victims) in targets {
            let promo = victims & prom_mask;
            let standard = victims & !prom_mask;
            for to in promo {
                let from = Square((to.0.cast_signed() - delta).cast_unsigned());
                self.add_promo_caps(board, from, to);
            }
            for to in standard {
                let from = Square((to.0.cast_signed() - delta).cast_unsigned());
                self.add_cap(board, Move::new(from, to, Move::CAPTURE), PieceType::Pawn, history);
            }
        }

        if let Some(ep_sq) = board.en_passant {
            for from in atk_pawn(ep_sq, stm.opposite()) & pawns {
                self.add_cap(board, Move::new(from, ep_sq, Move::EP_CAPTURE), PieceType::Pawn, history);
            }
        }
    }

    #[inline(always)]
    fn cap_score(&self, mvv: i32, chist: i32) -> i32 { mvv + chist / self.capt_hist_divisor }

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
        let attacker_pen = *debug_index!(self.mvvlva_attacker, PT as usize);
        let stm = board.stm;

        for from in board.role_bb[PT as usize] & us {
            for to in Self::attacks::<PT>(rows, from, occ) & them {
                let victim = board.piece_at(to);
                let victim_val = *debug_index!(self.mvvlva_victim, victim as usize);
                let chist = history.score_capture(stm, PT, to, victim);
                let score = self.cap_score(victim_val - attacker_pen, chist);
                self.add_scored(Move::new(from, to, Move::CAPTURE), score as MoveScore);
            }
        }
    }

    #[inline(always)]
    fn add_scored(&mut self, mv: Move, score: MoveScore) { self.write_packed(pack((score as i32 + SORT_BIAS) as u32, mv)); }

    #[inline(always)]
    fn write_packed(&mut self, packed: u32) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");
        debug_index_mut!(self.candidates, self.count).write(packed);
        self.count += 1;
    }

    /// Excludes promotion-captures.
    #[inline]
    fn add_cap(&mut self, board: &Position, mv: Move, attacker: PieceType, history: &History) {
        let victim = board.captured_piece(mv);
        let mvv = if mv.is_en_passant() {
            self.mvvlva_ep as MoveScore
        } else {
            debug_assert!(mv.promo().is_none(), "mvv_lva is only reached by non-promotion captures");
            debug_assert!(usize::from(victim) < self.mvvlva_victim.len());
            debug_assert!(usize::from(attacker) < self.mvvlva_attacker.len());
            let v = *debug_index!(self.mvvlva_victim, victim as usize);
            let a = *debug_index!(self.mvvlva_attacker, attacker as usize);
            (v - a) as MoveScore
        };
        let chist = history.score_capture(board.stm, attacker, mv.to(), victim);
        self.add_scored(mv, self.cap_score(mvv as i32, chist) as MoveScore);
    }

    #[inline]
    fn add_promo_caps(&mut self, board: &Position, from: Square, to: Square) {
        let victim = board.piece_at(to);
        let victim_val = *debug_index!(self.mvvlva_victim, victim as usize);
        let attacker_pen = *debug_index!(self.mvvlva_attacker, PieceType::Pawn as usize);
        let base = victim_val - attacker_pen;

        let q = base + *debug_index!(self.mvvlva_victim, PieceType::Queen as usize);
        let r = base + *debug_index!(self.mvvlva_victim, PieceType::Rook as usize);
        let b = base + *debug_index!(self.mvvlva_victim, PieceType::Bishop as usize);
        let n = base + *debug_index!(self.mvvlva_victim, PieceType::Knight as usize);

        self.add_scored(Move::new(from, to, Move::PROM_Q_CAPTURE), q as MoveScore);
        self.add_scored(Move::new(from, to, Move::PROM_R_CAPTURE), r as MoveScore);
        self.add_scored(Move::new(from, to, Move::PROM_B_CAPTURE), b as MoveScore);
        self.add_scored(Move::new(from, to, Move::PROM_N_CAPTURE), n as MoveScore);
    }

    #[inline]
    fn gen_qsearch_promotions(&mut self, board: &Position, history: &History) {
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
            self.add_quiet(Move::new(from, to, Move::PROM_Q), PieceType::Pawn, stm, history);
        }
    }

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
        board.for_each_castle(board.stm, |mv| {
            let pt = board.expect_piece_at(mv.from());
            self.add_quiet(mv, pt, board.stm, history);
        });
    }

    #[inline(always)]
    fn add_quiet(&mut self, mv: Move, pt: PieceType, stm: Color, history: &History) {
        debug_assert!(self.count < MAX_MOVES, "MovePicker capacity exceeded");

        // Knights can check and fork where a queen cannot, so they rank above the other underpromotions.
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
            // Measured on bench, zero saturation in ~11M quiets.
            (score + SORT_BIAS).clamp(0, QUIET_SCORE_MAX) as u32
        };

        self.write_packed(pack(sort_score, mv));
    }

    #[inline]
    fn add_promo_quiets(&mut self, from: Square, to: Square, stm: Color, history: &History) {
        self.add_quiet(Move::new(from, to, Move::PROM_Q), PieceType::Pawn, stm, history);
        self.add_quiet(Move::new(from, to, Move::PROM_R), PieceType::Pawn, stm, history);
        self.add_quiet(Move::new(from, to, Move::PROM_B), PieceType::Pawn, stm, history);
        self.add_quiet(Move::new(from, to, Move::PROM_N), PieceType::Pawn, stm, history);
    }

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
            self.add_quiet(Move::new(from, to, Move::QUIET), PieceType::Pawn, stm, history);
        }

        let doubles = (all_pushes & third_rank).shift(up) & empty;
        for to in doubles {
            let from = Square((to.0.cast_signed() - up_d * 2).cast_unsigned());
            self.add_quiet(Move::new(from, to, Move::DOUBLE_PUSH), PieceType::Pawn, stm, history);
        }
    }

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
                self.add_quiet(Move::new(from, to, Move::QUIET), PT, board.stm, history);
            }
        }
    }

    /// # Safety
    /// Entry `i` must be initialized.
    #[inline(always)]
    unsafe fn read_move(&self, i: usize) -> Move {
        debug_assert!(i < MAX_MOVES);
        // SAFETY: the caller guarantees entry i is initialized.
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

    /// A slider reads its maintained row instead of a magic lookup. A leaper's table lookup is
    /// a single load, cheaper than the mailbox-then-row pair, so leapers never come here.
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
            PieceType::Queen => atk_queen(from, occ),
            _ => Bitboard(0),
        }
    }
}

/// Equal scores sort by move encoding.
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
            let store = XorBoard::from_position(&board);
            let mut picker = MovePicker::new(
                None,
                &cfg,
                Pins::new(&board),
                [Move::null(); 2],
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
            let mut expected: Vec<u16> = gen_pseudo_moves(&board).iter().map(|mv| mv.inner()).collect();
            expected.sort_unstable();
            assert_eq!(picked, expected, "{fen}");
        }
    }
}
