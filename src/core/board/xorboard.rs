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
}

const SLOTS: usize = 32;
const NOWHERE: u8 = 0xFF;

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

impl XorBoard {
    /// Slots are assigned by a square walk, so two boards built from the same
    /// position agree slot for slot.
    pub fn new(pos: &Position) -> Self {
        let mut board =
            Self { rows: [0; SLOTS], at: [0; 64], sq: [NOWHERE; SLOTS], kind: [PieceType::None; SLOTS], class: [0; 12] };

        let mut next = [0usize, 16];

        for raw in 0..64u8 {
            let square = Square(raw);
            let piece = pos.piece_at(square);
            if piece == PieceType::None {
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
    #[inline(always)]
    pub fn attackers_of(&self, mask: Bitboard) -> u64 {
        let mut set = 0u64;
        for (slot, row) in self.rows.iter().enumerate() {
            set |= u64::from(row & mask.0 != 0) << slot;
        }
        set
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
        Self { len: 0, ids: [0; UNDO_CAP], rows: [0; UNDO_CAP] }
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
struct Plan {
    movers: [(PieceId, Square, Square); 2],
    n_movers: usize,
    changed: Bitboard,
    victim: Option<(PieceId, Square)>,
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

        let mut affected = self.attackers_of(plan.changed) & self.sliders();
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

        let plan = self.read_back(mv, undo);
        self.restore(mv, &plan);
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

    /// The same read against post-move bookkeeping. The movers sit at their
    /// destinations; the victim's slot has to come from the record, because
    /// once its square is overwritten nothing on the board points at it.
    #[inline(always)]
    fn read_back(&self, mv: Move, undo: &Undo) -> Plan {
        let (from, to) = (mv.from(), mv.to());
        let mut movers = [(PieceId(0), from, to); 2];
        let mut n_movers = 1;

        if mv.is_castling() {
            let (king_to, rook_to) = super::castling_targets(from, to);
            movers = [
                (PieceId(self.at[usize::from(king_to)] - 1), from, king_to),
                (PieceId(self.at[usize::from(rook_to)] - 1), to, rook_to),
            ];
            n_movers = 2;
        } else {
            movers[0].0 = PieceId(self.at[usize::from(to)] - 1);
        }

        // En passant sets the capture bit and castling does not, so the flag
        // alone says whether slot zero of the record is a victim.
        let victim = mv.is_capture().then(|| {
            let square = if mv.is_en_passant() { Square(to.0 ^ 8) } else { to };
            (PieceId(undo.ids[0]), square)
        });

        Plan { movers, n_movers, changed: Bitboard(0), victim }
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
