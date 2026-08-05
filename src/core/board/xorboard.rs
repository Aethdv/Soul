//! The attack relation, stored from-side.
//!
//! Every incremental attack table in the field indexes by square: one entry per
//! square saying who attacks it, whether that entry is Rebel's byte, Rookie's
//! direction bits, KnightCap's id bitset or Rose's colour-split pair. This one
//! indexes by piece. `rows[id]` is the set of squares piece `id` attacks, and
//! the square-indexed table is its transpose, computed where it is wanted
//! instead of kept in step.
//!
//! A move rewrites whole rows rather than scattering bits, roughly one row per
//! move against nine (piece, square) pairs, so the update is a handful of stores
//! with no read-modify-write chain. The union views come free with it: a square
//! ORs a side together and forgets which piece set the bit, which is what forces
//! a square-indexed table to recount, and here the contributor is the index.
//! The unmake record is bounded by pieces touched, not squares attacked.

use core::arch::x86_64::*;

use crate::core::{
    board::{
        Position,
        bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook},
    },
    defs::{Bitboard, Color, PieceType, Square},
    moves::Move,
};

/// Dense slot for a piece, white 0 to 15 and black 16 to 31.
///
/// A side can legally hold nine queens, so no type-major numbering survives and
/// the slot says nothing on its own. Type lives in a LUT beside it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PieceId(u8);

/// The rows, plus the bookkeeping that names them.
#[derive(Clone, Debug)]
pub struct XorBoard {
    /// `rows[id]`: every square piece `id` attacks under the current occupancy.
    /// A vacant slot holds zero, so it contributes nothing to any reduction and
    /// needs no special case.
    rows: [u64; SLOTS],
    /// Square to `1 + id`, so an empty square is zero.
    at: [u8; 64],
    /// Slot to square, `NOWHERE` when the piece is off the board.
    sq: [u8; SLOTS],
    kind: [PieceType; SLOTS],
    /// Slot masks per (type, colour), patched on promotion. Keyed by slot, so
    /// "every rook and queen" is an AND rather than a scan.
    class: [u64; 12],
}

/// What one make overwrote, replayed by unmake.
///
/// Bounded by the affected set, so it counts pieces where a square-indexed diff
/// record would count attacked squares.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    len: u8,
    ids: [u8; UNDO_CAP],
    rows: [u64; UNDO_CAP],
    /// Make already worked out which slots move where; unmake reads it back off
    /// the record rather than deriving it a second time from the flags.
    plan: Plan,
}

const SLOTS: usize = 32;
const NOWHERE: u8 = 0xFF;

/// Slots 0 to 7 and 16 to 23: the eight-wide window each colour's sliders are
/// seeded into. Only sliders can be affected by a move they did not make, so
/// the gather only ever has to look here, and it looks at a fixed four groups
/// rather than all eight.
const SLIDER_WINDOW: u64 = 0x00FF_00FF;

/// Two movers, one victim, and every slider on the board: a side cannot exceed
/// two bishops, two rooks, a queen and eight promotions.
const UNDO_CAP: usize = 29;

impl PieceId {
    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 < 16 { Color::White } else { Color::Black }
    }

    #[inline(always)]
    const fn index(self) -> usize {
        self.0 as usize
    }
}

#[inline(always)]
fn is_slider(piece: PieceType) -> bool {
    matches!(piece, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
}

impl XorBoard {
    /// Slots are assigned by a square walk, so two boards built from the same
    /// position agree slot for slot.
    pub fn new(pos: &Position) -> Self {
        let mut board =
            Self { rows: [0; SLOTS], at: [0; 64], sq: [NOWHERE; SLOTS], kind: [PieceType::None; SLOTS], class: [0; 12] };

        let mut next = [0usize, 16];

        // Sliders first, so they land in the low half of each colour's range and
        // the gather can skip the rest. A side can legally promote its way past
        // eight, which the gather handles by widening rather than by anything
        // here having to care.
        for slider_pass in [true, false] {
            for raw in 0..64u8 {
                let square = Square(raw);
                let piece = pos.piece_at(square);
                if piece == PieceType::None || is_slider(piece) != slider_pass {
                    continue;
                }

                let color = if pos.side_bb[Color::White].check_bit(square) { Color::White } else { Color::Black };
                let slot = next[color as usize];
                next[color as usize] += 1;

                board.at[usize::from(square)] = (slot + 1) as u8;
                board.sq[slot] = raw;
                board.kind[slot] = piece;
                board.class[piece as usize * 2 + color as usize] |= 1 << slot;
            }
        }
        board.refresh(pos);
        board
    }

    /// Recomputes every row from the bookkeeping: the oracle the incremental
    /// path answers to, and how a root position is seeded.
    pub fn refresh(&mut self, pos: &Position) {
        self.rows = [0; SLOTS];
        for slot in 0..SLOTS {
            if self.sq[slot] != NOWHERE {
                self.rows[slot] = self.attacks(PieceId(slot as u8), Square(self.sq[slot]), pos.occ).0;
            }
        }
    }

    /// Every square `color` attacks.
    ///
    /// Not the same set as `Position::threats`, which ends its setwise fill with
    /// `& !generator` over the whole rook-plus-queen union and so cannot see a
    /// rook defending a rook. This one is exact, so swapping it into a consumer
    /// changes search behaviour and wants its own test.
    #[inline(always)]
    pub fn danger(&self, color: Color) -> Bitboard {
        let base = usize::from(color) * 16;
        let mut acc = 0u64;
        for row in &self.rows[base..base + 16] {
            acc |= *row;
        }
        Bitboard(acc)
    }

    /// Every square the given class attacks.
    #[inline(always)]
    pub fn class_attacks(&self, piece: PieceType, color: Color) -> Bitboard {
        self.union(self.class[piece as usize * 2 + color as usize])
    }

    /// What one piece attacks: the from-side query the store exists to answer.
    #[inline(always)]
    pub fn row(&self, id: PieceId) -> Bitboard {
        Bitboard(self.rows[id.index()])
    }

    #[inline(always)]
    pub fn id_at(&self, square: Square) -> Option<PieceId> {
        match self.at[usize::from(square)] {
            0 => None,
            slot => Some(PieceId(slot - 1)),
        }
    }

    /// The transpose, taken one column at a time: which pieces attack any square
    /// in `mask`. Testing a row against a multi-bit mask answers for every square
    /// in it at once, so a castling move's four changed squares cost one pass.
    ///
    /// Written by hand because the shape matters: eight lane tests folded into a
    /// slot set, or four `vptestmq` where the wider registers exist. Left to the
    /// autovectorizer this is a thirty-two iteration dependency chain, and it
    /// sits in the update path of every move.
    #[inline(always)]
    pub fn slider_attackers_of(&self, mask: Bitboard) -> u64 {
        let sliders = self.sliders();

        if sliders & !SLIDER_WINDOW == 0 {
            self.probe_groups::<true>(mask) & sliders
        } else {
            self.probe_groups::<false>(mask) & sliders
        }
    }

    #[inline(always)]
    pub fn attackers_of(&self, mask: Bitboard) -> u64 {
        self.probe_groups::<false>(mask)
    }

    /// `WINDOW` restricts the test to the slider window, four groups instead of
    /// eight, both counts fixed so the loop unrolls either way.
    #[inline(always)]
    fn probe_groups<const WINDOW: bool>(&self, mask: Bitboard) -> u64 {
        // SAFETY: AVX2 is guaranteed for every binary linking this crate by the
        // compile_error gate in weave/mod.rs, and AVX-512 only under its own
        // cfg. Each load covers one whole group of a 32-element array, so the
        // last ends exactly at its end.
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                let want = _mm512_set1_epi64(mask.0 as i64);
                let groups: &[usize] = if WINDOW { &[0, 2] } else { &[0, 1, 2, 3] };
                let mut set = 0u64;

                for &group in groups {
                    let rows = _mm512_loadu_si512(self.rows.as_ptr().add(group * 8).cast());
                    set |= u64::from(_mm512_test_epi64_mask(rows, want)) << (group * 8);
                }

                set
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                let want = _mm256_set1_epi64x(mask.0 as i64);
                let zero = _mm256_setzero_si256();
                let groups: &[usize] = if WINDOW { &[0, 1, 4, 5] } else { &[0, 1, 2, 3, 4, 5, 6, 7] };
                let mut set = 0u64;

                for &group in groups {
                    let rows = _mm256_loadu_si256(self.rows.as_ptr().add(group * 4).cast());
                    let idle = _mm256_cmpeq_epi64(_mm256_and_si256(rows, want), zero);
                    let live = !(_mm256_movemask_pd(_mm256_castsi256_pd(idle)) as u32) & 0xF;
                    set |= u64::from(live) << (group * 4);
                }

                set
            }
        }
    }

    #[inline(always)]
    fn union(&self, slots: u64) -> Bitboard {
        let mut acc = 0u64;
        let mut rest = slots;
        while rest != 0 {
            acc |= self.rows[rest.trailing_zeros() as usize];
            rest &= rest - 1;
        }
        Bitboard(acc)
    }

    #[inline(always)]
    fn sliders(&self) -> u64 {
        self.class_slots(PieceType::Bishop) | self.class_slots(PieceType::Rook) | self.class_slots(PieceType::Queen)
    }

    #[inline(always)]
    fn class_slots(&self, piece: PieceType) -> u64 {
        self.class[piece as usize * 2] | self.class[piece as usize * 2 + 1]
    }

    #[inline(always)]
    fn attacks(&self, id: PieceId, from: Square, occ: Bitboard) -> Bitboard {
        match self.kind[id.index()] {
            PieceType::Pawn => atk_pawn(from, id.color()),
            PieceType::Knight => atk_knight(from),
            PieceType::King => atk_king(from),
            PieceType::Bishop => atk_bishop(from, occ),
            PieceType::Rook => atk_rook(from, occ),
            PieceType::Queen => atk_rook(from, occ) | atk_bishop(from, occ),
            PieceType::None => Bitboard(0),
        }
    }
}

impl Undo {
    pub const fn new() -> Self {
        Self { len: 0, ids: [0; UNDO_CAP], rows: [0; UNDO_CAP], plan: Plan::EMPTY }
    }

    #[inline(always)]
    fn save(&mut self, id: PieceId, row: u64) {
        debug_assert!(usize::from(self.len) < UNDO_CAP, "affected set outgrew two movers, a victim and every slider");
        self.ids[usize::from(self.len)] = id.0;
        self.rows[usize::from(self.len)] = row;
        self.len += 1;
    }
}

impl Default for Undo {
    fn default() -> Self {
        Self::new()
    }
}

/// Which slots the move disturbs and where they end up.
#[derive(Clone, Copy, Debug)]
struct Plan {
    movers: [(PieceId, Square, Square); 2],
    n_movers: usize,
    changed: Bitboard,
    victim: Option<(PieceId, Square)>,
}

impl Plan {
    const EMPTY: Self = Self { movers: [(PieceId(0), Square(0), Square(0)); 2], n_movers: 0, changed: Bitboard(0), victim: None };
}

impl XorBoard {
    /// Brings the rows up to date for `mv`. `pos` is the position after the
    /// move; this board still holds the state before it, which is what the
    /// candidate search has to read.
    ///
    /// A piece's attacks are a function of its square and the occupancy, so an
    /// unmoved piece can only change where its first blocker did, and any such
    /// piece saw one of the changed squares before the move. That makes the
    /// affected set a theorem rather than a scan, and the rows themselves
    /// answer it.
    pub fn make(&mut self, pos: &Position, mv: Move, undo: &mut Undo) {
        let plan = self.read(mv);
        undo.len = 0;
        undo.plan = plan;

        let mut affected = self.slider_attackers_of(plan.changed);
        for m in 0..plan.n_movers {
            affected &= !(1 << plan.movers[m].0.index());
        }

        if let Some((victim, square)) = plan.victim {
            affected &= !(1 << victim.index());
            undo.save(victim, self.rows[victim.index()]);
            self.rows[victim.index()] = 0;
            self.at[usize::from(square)] = 0;
            self.sq[victim.index()] = NOWHERE;
        }

        self.relocate(mv, &plan);

        let mut rest = affected;

        while rest != 0 {
            let id = PieceId(rest.trailing_zeros() as u8);
            rest &= rest - 1;
            undo.save(id, self.rows[id.index()]);
            self.rows[id.index()] = self.attacks(id, Square(self.sq[id.index()]), pos.occ).0;
        }

        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            undo.save(id, self.rows[id.index()]);
            self.rows[id.index()] = self.attacks(id, to, pos.occ).0;
        }
    }

    /// Replays the record and walks the bookkeeping back.
    ///
    /// Every affected piece appears once, because the candidate set has the
    /// movers and the victim masked out of it, so the rows restore in any
    /// order.
    pub fn unmake(&mut self, mv: Move, undo: &Undo) {
        for i in 0..usize::from(undo.len) {
            self.rows[usize::from(undo.ids[i])] = undo.rows[i];
        }

        self.restore(mv, &undo.plan);
    }

    /// Reads the move against pre-move bookkeeping.
    #[inline(always)]
    fn read(&self, mv: Move) -> Plan {
        let (from, to) = (mv.from(), mv.to());
        let mover = PieceId(self.at[usize::from(from)] - 1);
        let mut movers = [(mover, from, to); 2];
        let mut changed = from.bitboard() | to.bitboard();
        let mut n_movers = 1;
        let mut victim = None;

        if mv.is_castling() {
            // The move encodes the rook's home square as its destination, so
            // both pieces and all four squares come out of that one pair.
            let rook = PieceId(self.at[usize::from(to)] - 1);
            let (king_to, rook_to) = super::castling_targets(from, to);
            movers = [(mover, from, king_to), (rook, to, rook_to)];
            changed |= king_to.bitboard() | rook_to.bitboard();
            n_movers = 2;
        } else if mv.is_en_passant() {
            let square = Square(to.0 ^ 8);
            changed |= square.bitboard();
            victim = Some((PieceId(self.at[usize::from(square)] - 1), square));
        } else if mv.is_capture() {
            victim = Some((PieceId(self.at[usize::from(to)] - 1), to));
        }
        Plan { movers, n_movers, changed, victim }
    }

    /// Origins clear before any destination lands, so DFRC castling stays exact
    /// when the king comes to rest on the rook's own origin square.
    #[inline(always)]
    fn relocate(&mut self, mv: Move, plan: &Plan) {
        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].1)] = 0;
        }
        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            self.at[usize::from(to)] = id.0 + 1;
            self.sq[id.index()] = to.0;
        }

        if let Some(promoted) = mv.promo() {
            let id = plan.movers[0].0;
            let color = usize::from(id.color());
            self.class[self.kind[id.index()] as usize * 2 + color] &= !(1 << id.index());
            self.class[promoted as usize * 2 + color] |= 1 << id.index();
            self.kind[id.index()] = promoted;
        }
    }

    /// `relocate` run backwards. The victim lands last: for a capture its square
    /// is the mover's destination, so the later write has to be the one to win.
    #[inline(always)]
    fn restore(&mut self, mv: Move, plan: &Plan) {
        if mv.is_promotion() {
            let id = plan.movers[0].0;
            let color = usize::from(id.color());
            self.class[self.kind[id.index()] as usize * 2 + color] &= !(1 << id.index());
            self.class[PieceType::Pawn as usize * 2 + color] |= 1 << id.index();
            self.kind[id.index()] = PieceType::Pawn;
        }

        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].2)] = 0;
        }
        for m in 0..plan.n_movers {
            let (id, from, _) = plan.movers[m];
            self.at[usize::from(from)] = id.0 + 1;
            self.sq[id.index()] = from.0;
        }

        if let Some((id, square)) = plan.victim {
            self.at[usize::from(square)] = id.0 + 1;
            self.sq[id.index()] = square.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{board::STARTPOS, zobrist::ConstRng},
        engine::movegen::gen_legal_moves,
    };

    const FENS: [&str; 5] = [
        STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 1",
    ];

    /// Rows against a from-scratch rebuild, the danger view against the engine's
    /// own fill, and the record replayed, at every ply.
    #[test]
    fn tracks_the_position() {
        let mut rng = ConstRng::new(0xC0FFEE);

        for fen in FENS {
            let mut pos = Position::from_fen(fen);
            let mut acc = pos.get_initial_accumulator();
            let mut board = XorBoard::new(&pos);
            let mut undo = Undo::new();

            for ply in 0..512 {
                let legal = gen_legal_moves(&pos);
                if legal.is_empty() {
                    break;
                }

                let mv = legal[(rng.next() % legal.len() as u64) as usize];
                let before = board.clone();
                let state = pos.make_move(mv, &mut acc);
                board.make(&pos, mv, &mut undo);

                let mut oracle = board.clone();
                oracle.refresh(&pos);
                assert_eq!(board.rows, oracle.rows, "{fen} ply {ply} move {}\n{pos}", mv.to_uci(pos.is_frc));

                // The fill drops every square holding a same-class slider, so
                // equality is the wrong assertion: it has to be containment, with
                // the gap confined to that side's own sliders.
                for color in [Color::White, Color::Black] {
                    let exact = board.danger(color);
                    let fill = pos.threats(color);
                    let sliders = (pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen])
                        & pos.side_bb[color];

                    assert!((fill & !exact).is_empty(), "fill outside rows, {color:?}, {fen} ply {ply}\n{pos}");
                    assert!((exact & !fill & !sliders).is_empty(), "gap off own sliders, {color:?}, {fen} ply {ply}\n{pos}");
                }

                board.unmake(mv, &undo);
                pos.unmake_move(mv, &state);
                assert_eq!(board.rows, before.rows, "unmake rows, {fen} ply {ply}\n{pos}");
                assert_eq!(board.at, before.at, "unmake mailbox, {fen} ply {ply}\n{pos}");
                assert_eq!(board.kind, before.kind, "unmake types, {fen} ply {ply}\n{pos}");
                pos.make_move(mv, &mut acc);
                board.make(&pos, mv, &mut undo);
            }
        }
    }
}
