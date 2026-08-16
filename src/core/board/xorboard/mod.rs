//! The attack relation, stored from-side.
//!
//! Every incremental attack table in the field indexes by square: Rebel's byte,
//! Rookie's direction bits, KnightCap's id bitset, Rose's colour-split pair.
//! This one indexes by piece, so `rows[id]` is what piece `id` attacks and the
//! square-indexed table is a transpose to compute where it is wanted.
//!
//! A move rewrites about one whole row where a square-indexed table scatters
//! nine bits, and the union views come free: a square ORs a side together and
//! forgets which piece set the bit, where here the contributor is the index.
//! The unmake record counts pieces touched, not squares attacked.
//!
//! Maintained through the search's make and unmake.

mod views;

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
    squares: [u8; SLOTS],
    kind: [PieceType; SLOTS],
    /// Slot masks per (type, colour), patched on promotion. Keyed by slot, so
    /// "every rook and queen" is an AND rather than a scan.
    class: [u64; 12],
    /// The three slider classes, unioned once instead of on every gather. Only
    /// a promotion moves a slot in or out: a captured slider keeps its bit and
    /// drops out of the column test on its own, its row being empty.
    slider_slots: u64,
}

/// The store as it stood at the node, restored wholesale by unmake.
///
/// One snapshot serves every move tried at a ply: each make and its unmake put
/// the state back where it was, so all of a node's siblings would otherwise
/// save the same 352 bytes over and over. `kind` and `class` stay out of it,
/// since only a promotion moves them.
#[derive(Clone, Copy, Debug)]
pub struct Undo {
    rows: [u64; SLOTS],
    at: [u8; 64],
    squares: [u8; SLOTS],
}

/// One piece the move itself relocates: the mover, and the rook of a castling.
#[derive(Clone, Copy, Debug)]
struct Mover {
    id: PieceId,
    from: Square,
    to: Square,
}

/// Which slots the move disturbs and where they end up.
#[derive(Clone, Copy, Debug)]
struct Plan {
    movers: [Mover; 2],
    /// The only move that relocates two pieces. Otherwise the second mover is
    /// a copy of the first, which the masks in `make` rely on.
    castling: bool,
    victim: Option<(PieceId, Square)>,
}

const SLOTS: usize = 32;
const NOWHERE: u8 = 0xFF;

/// Slots 0 to 7 and 16 to 23: the eight-wide window each colour's sliders are
/// seeded into. Only sliders can be affected by a move they did not make, so
/// the gather only ever has to look here, and it looks at a fixed four groups
/// rather than all eight.
const SLIDER_WINDOW: u64 = 0x00FF_00FF;

/// Groups of four slots the column test looks at, one bit per group.
/// `WINDOW_GROUPS` is [`SLIDER_WINDOW`] in the same currency.
const ALL_GROUPS: u8 = 0xFF;
const WINDOW_GROUPS: u8 = 0b0011_0011;
const WHITE_GROUPS: u8 = 0x0F;
const BLACK_GROUPS: u8 = 0xF0;

#[cfg(test)]
const fn color_slots(color: Color) -> u64 {
    0xFFFF << (color as usize * 16)
}

/// `class` is keyed by piece type then colour; one place computes that.
#[inline(always)]
const fn class_index(piece: PieceType, color: Color) -> usize {
    piece as usize * 2 + color as usize
}

impl PieceId {
    #[inline(always)]
    pub const fn color(self) -> Color {
        if self.0 < 16 { Color::White } else { Color::Black }
    }

    /// Masked, so every `rows`/`squares`/`kind` access is in range by construction
    /// and the compiler drops the check. A slot out of range would be a bug the
    /// mask hides, and slots only ever come from a 32-bit mask's set bits or
    /// from `at`, both of which are already in range.
    #[inline(always)]
    const fn index(self) -> usize {
        (self.0 & 31) as usize
    }
}

/// The set slots of a mask, low to high.
#[inline(always)]
fn slots(mask: u64) -> impl Iterator<Item = PieceId> {
    let mut rest = mask;

    core::iter::from_fn(move || {
        (rest != 0).then(|| {
            let slot = rest.trailing_zeros() as u8;
            rest &= rest - 1;
            PieceId(slot)
        })
    })
}

#[inline(always)]
fn is_slider(piece: PieceType) -> bool {
    matches!(piece, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
}

impl XorBoard {
    /// Slots are assigned by a square walk, so two boards built from the same
    /// position agree slot for slot.
    pub fn new(pos: &Position) -> Self {
        let mut board = Self {
            rows: [0; SLOTS],
            at: [0; 64],
            squares: [NOWHERE; SLOTS],
            kind: [PieceType::None; SLOTS],
            class: [0; 12],
            slider_slots: 0,
        };

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

                board.set_slot_at(square, (slot + 1) as u8);
                board.squares[slot] = raw;
                board.kind[slot] = piece;
                board.class[class_index(piece, color)] |= 1 << slot;
                board.slider_slots |= u64::from(is_slider(piece)) << slot;
            }
        }
        board.refresh(pos);
        board
    }

    /// Recomputes every row from the bookkeeping: the oracle the incremental
    /// path answers to, and how a root position is seeded.
    fn refresh(&mut self, pos: &Position) {
        self.rows = [0; SLOTS];
        for slot in 0..SLOTS {
            if self.squares[slot] != NOWHERE {
                self.rows[slot] = self.attacks(PieceId(slot as u8), Square(self.squares[slot]), pos.occ).0;
            }
        }
    }

    /// What one piece attacks: the from-side query the store exists to answer.
    #[inline(always)]
    pub fn row(&self, id: PieceId) -> Bitboard {
        Bitboard(self.rows[id.index()])
    }

    #[inline(always)]
    pub fn id_at(&self, square: Square) -> Option<PieceId> {
        match self.slot_at(square) {
            0 => None,
            slot => Some(PieceId(slot - 1)),
        }
    }

    /// `1 + id`, or zero for an empty square. Masked, so every `at` access is in
    /// range by construction and the compiler drops the check.
    #[inline(always)]
    fn slot_at(&self, square: Square) -> u8 {
        self.at[usize::from(square.0 & 63)]
    }

    #[inline(always)]
    fn set_slot_at(&mut self, square: Square, slot: u8) {
        self.at[usize::from(square.0 & 63)] = slot;
    }

    /// The transpose, taken one column at a time: which pieces attack any square
    /// in `mask`. Testing a row against a multi-bit mask answers for every square
    /// in it at once, so a castling move's four changed squares cost one pass.
    ///
    /// Written by hand: left to the autovectorizer it is a thirty-two iteration
    /// dependency chain rather than eight lane tests folded into a slot set.
    #[inline(always)]
    fn slider_attackers_of(&self, mask: Bitboard) -> u64 {
        let sliders = self.slider_slots;
        if sliders & !SLIDER_WINDOW == 0 {
            self.column::<WINDOW_GROUPS>(mask) & sliders
        } else {
            self.column::<ALL_GROUPS>(mask) & sliders
        }
    }

    /// The pieces of `stm`'s opponent giving check.
    ///
    /// Only one side can be checking, so the column test covers that side's
    /// slots and no more. Slots of captured pieces hold an empty row and drop
    /// out of the test on their own.
    #[inline(always)]
    pub fn checkers(&self, pos: &Position) -> Bitboard {
        let king = pos.pieces(PieceType::King, pos.stm);
        if king.is_empty() {
            return Bitboard(0);
        }

        let attackers = match pos.stm.opposite() {
            Color::White => self.column::<WHITE_GROUPS>(king),
            Color::Black => self.column::<BLACK_GROUPS>(king),
        };

        slots(attackers).fold(Bitboard(0), |squares, id| squares | Square(self.squares[id.index()]).bitboard())
    }

    /// Over the groups of four slots `GROUPS` selects.
    ///
    /// Every caller wants a different slice of the thirty-two rows and none of
    /// them wants a runtime bound: a const mask keeps the loop unrolled and the
    /// unwanted groups never reach the machine code. The wide arm loads eight
    /// rows at a time, so a group mask that splits a pair would test slots the
    /// caller excluded, and the assert holds callers to pairs.
    #[inline(always)]
    fn column<const GROUPS: u8>(&self, mask: Bitboard) -> u64 {
        const { assert!(GROUPS & 0x55 == (GROUPS >> 1) & 0x55, "the group mask must select whole pairs") }

        // SAFETY: AVX2 is guaranteed for every binary linking this crate by the
        // compile_error gate in weave/mod.rs, and AVX-512 only under its own
        // cfg. Each load covers whole groups of a 32-element array, so the last
        // ends exactly at its end.
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                let want = _mm512_set1_epi64(mask.0.cast_signed());
                let mut set = 0u64;

                for wide in 0..4 {
                    if GROUPS >> (wide * 2) & 3 != 0 {
                        let rows = _mm512_loadu_si512(self.rows.as_ptr().add(wide * 8).cast());
                        set |= u64::from(_mm512_test_epi64_mask(rows, want)) << (wide * 8);
                    }
                }
                set
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                let want = _mm256_set1_epi64x(mask.0.cast_signed());
                let zero = _mm256_setzero_si256();
                let mut set = 0u64;
                for group in 0..8 {
                    if GROUPS >> group & 1 != 0 {
                        let rows = _mm256_loadu_si256(self.rows.as_ptr().add(group * 4).cast());
                        let idle = _mm256_cmpeq_epi64(_mm256_and_si256(rows, want), zero);
                        let live = !_mm256_movemask_pd(_mm256_castsi256_pd(idle)).cast_unsigned() & 0xF;
                        set |= u64::from(live) << (group * 4);
                    }
                }
                set
            }
        }
    }

    /// Moves a slot from the class it holds to `piece`, on a promotion and on
    /// the unmake that walks one back.
    #[inline(always)]
    fn reclass(&mut self, id: PieceId, piece: PieceType) {
        let slot = id.index();

        self.class[class_index(self.kind[slot], id.color())] &= !(1 << slot);
        self.class[class_index(piece, id.color())] |= 1 << slot;
        self.kind[slot] = piece;
        self.slider_slots = (self.slider_slots & !(1 << slot)) | (u64::from(is_slider(piece)) << slot);
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

    /// Does the maintained store still say what a store built from `pos` says?
    ///
    /// Slots are handed out in seeding order, so the two boards can name the
    /// same piece differently and the rows cannot be compared slot by slot. The
    /// class unions are slot-independent and catch any row that drifted.
    pub fn agrees_with(&self, pos: &Position) -> bool {
        let fresh = Self::new(pos);

        [Color::White, Color::Black].into_iter().all(|color| {
            PieceType::ALL
                .into_iter()
                .all(|pt| self.class_attacks(pt, color) == fresh.class_attacks(pt, color))
        })
    }

    /// Brings the rows up to date for `mv`. `pos` is the position after the
    /// move; this board still holds the state before it, which is what the
    /// candidate search has to read.
    ///
    /// A piece's attacks are a function of its square and the occupancy, so an
    /// unmoved piece can only change where its first blocker did, and any such
    /// piece saw one of the changed squares before the move. That makes the
    /// affected set a theorem rather than a scan, and the rows themselves
    /// answer it.
    ///
    /// [`XorBoard::snapshot`] must have run for this ply, or unmake restores
    /// state that belongs to another node.
    pub fn make(&mut self, pos: &Position, mv: Move) {
        let (plan, changed) = self.decode(mv);
        let (first, second) = (plan.movers[0], plan.movers[1]);
        let castling = plan.castling;

        let mut affected = self.slider_attackers_of(changed) & !(1 << first.id.index()) & !(1 << second.id.index());

        if let Some((victim, square)) = plan.victim {
            affected &= !(1 << victim.index());
            self.rows[victim.index()] = 0;
            self.set_slot_at(square, 0);
            self.squares[victim.index()] = NOWHERE;
        }

        self.relocate(mv, &plan);

        for id in slots(affected) {
            self.rows[id.index()] = self.attacks(id, Square(self.squares[id.index()]), pos.occ).0;
        }

        self.rows[first.id.index()] = self.attacks(first.id, first.to, pos.occ).0;
        if castling {
            self.rows[second.id.index()] = self.attacks(second.id, second.to, pos.occ).0;
        }
    }

    /// Takes the rows a ply's snapshot holds, which is what every move tried
    /// there has to return them to.
    #[inline(always)]
    pub fn snapshot(&self, undo: &mut Undo) {
        undo.rows = self.rows;
        undo.at = self.at;
        undo.squares = self.squares;
    }

    /// The victim, the movers and the mailbox all come back with the arrays. A
    /// promotion's class is the one thing a copy cannot undo, and the restored
    /// mailbox names the piece to undo it for.
    pub fn unmake(&mut self, mv: Move, undo: &Undo) {
        self.rows = undo.rows;
        self.at = undo.at;
        self.squares = undo.squares;

        if mv.is_promotion() {
            self.reclass(PieceId(self.slot_at(mv.from()) - 1), PieceType::Pawn);
        }
    }

    /// Decodes the move against pre-move bookkeeping, and with it the squares
    /// whose occupancy flipped.
    ///
    /// A capture's destination is absent from that mask, since the victim left
    /// it and the mover took it, and the same cancellation covers a DFRC king
    /// landing on the rook's origin.
    #[inline(always)]
    fn decode(&self, mv: Move) -> (Plan, Bitboard) {
        let (from, to) = (mv.from(), mv.to());
        let mover = PieceId(self.slot_at(from) - 1);
        let mut movers = [Mover { id: mover, from, to }; 2];
        let mut victim = None;
        let mut vacated = from.bitboard();
        let mut filled = to.bitboard();
        let castling = mv.is_castling();

        if castling {
            // The move encodes the rook's home square as its destination, so
            // both pieces and all four squares come out of that one pair.
            let rook = PieceId(self.slot_at(to) - 1);
            let (king_to, rook_to) = super::castling_targets(from, to);
            movers = [Mover { id: mover, from, to: king_to }, Mover { id: rook, from: to, to: rook_to }];
            vacated |= to.bitboard();
            filled = king_to.bitboard() | rook_to.bitboard();
        } else if mv.is_en_passant() {
            let square = Square(to.0 ^ 8);
            vacated |= square.bitboard();
            victim = Some((PieceId(self.slot_at(square) - 1), square));
        } else if mv.is_capture() {
            vacated |= to.bitboard();
            victim = Some((PieceId(self.slot_at(to) - 1), to));
        }
        (Plan { movers, castling, victim }, vacated ^ filled)
    }

    /// Origins clear before any destination lands, so DFRC castling stays exact
    /// when the king comes to rest on the rook's own origin square.
    #[inline(always)]
    fn relocate(&mut self, mv: Move, plan: &Plan) {
        let (first, second) = (plan.movers[0], plan.movers[1]);
        let castling = plan.castling;

        self.set_slot_at(first.from, 0);
        if castling {
            self.set_slot_at(second.from, 0);
        }

        self.set_slot_at(first.to, first.id.0 + 1);
        self.squares[first.id.index()] = first.to.0;

        if castling {
            self.set_slot_at(second.to, second.id.0 + 1);
            self.squares[second.id.index()] = second.to.0;
        }

        if let Some(promoted) = mv.promo() {
            self.reclass(first.id, promoted);
        }
    }
}

impl Undo {
    pub const fn new() -> Self {
        Self { rows: [0; SLOTS], at: [0; 64], squares: [NOWHERE; SLOTS] }
    }
}

impl Default for Undo {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        core::{
            board::{
                STARTPOS,
                bitboard::{between_bb, line_bb},
            },
            zobrist::ConstRng,
        },
        engine::movegen::gen_legal_moves,
    };

    const FENS: [&str; 5] = [
        STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
        "4k3/P6p/8/8/8/8/p6P/4K3 w - - 0 1",
        "1rqbkrbn/1ppppp1p/1n6/p1N3p1/8/2P4P/PP1PPPP1/1RQBKRBN w FBfb - 0 1",
    ];

    impl XorBoard {
        fn xray_row(&self, id: PieceId, slider_squares: Bitboard) -> Bitboard {
            let from = Square(self.squares[id.index()]);
            let mut through = Bitboard(0);
            for square in Bitboard(self.rows[id.index()]) & slider_squares {
                let Some(blocker) = self.id_at(square) else { continue };
                let past = line_bb(from, square) & !between_bb(from, square) & !from.bitboard() & !square.bitboard();
                through |= Bitboard(self.rows[blocker.index()]) & past;
            }
            through
        }

        fn legal_rows(&self, color: Color, pinned: Bitboard, ksq: Square) -> impl Iterator<Item = (PieceId, Bitboard)> + '_ {
            let king = self.class[class_index(PieceType::King, color)];

            slots(color_slots(color) & !king).filter_map(move |id| {
                let raw = self.squares[id.index()];
                if raw == NOWHERE {
                    return None;
                }

                let square = Square(raw);
                let row = self.row(id);
                Some((id, if pinned.check_bit(square) { self.pinned_row(id, square, row, ksq) } else { row }))
            })
        }

        fn attack_map(&self, color: Color, pinned: Bitboard, ksq: Square) -> Bitboard {
            self.legal_rows(color, pinned, ksq).fold(Bitboard(0), |acc, (_, row)| acc | row)
        }
    }

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
                board.snapshot(&mut undo);
                board.make(&pos, mv);

                let mut oracle = board.clone();
                oracle.refresh(&pos);
                assert_eq!(board.rows, oracle.rows, "{fen} ply {ply} move {}\n{pos}", mv.to_uci(pos.is_frc));

                // Composition against the probe it replaces, for the blockers
                // it claims: one running along the same line. Lifting such a
                // blocker out of the occupancy has to reach the same squares
                // past it as its own maintained row does.
                let slider_squares = [PieceType::Bishop, PieceType::Rook, PieceType::Queen]
                    .into_iter()
                    .flat_map(|pt| [Color::White, Color::Black].map(|color| pos.pieces(pt, color)))
                    .fold(Bitboard(0), |acc, bb| acc | bb);

                for id in slots(board.slider_slots) {
                    let from = Square(board.squares[id.index()]);
                    let mut want = Bitboard(0);

                    for square in Bitboard(board.rows[id.index()]) & slider_squares {
                        let blocker = board.id_at(square).expect("a slider square holds a piece");
                        let straight = atk_rook(from, Bitboard(0)).check_bit(square);
                        let kind = board.kind[blocker.index()];

                        let aligned = match kind {
                            PieceType::Queen => true,
                            PieceType::Rook => straight,
                            _ => !straight,
                        };

                        if aligned {
                            // Past the blocker, said without a direction table:
                            // the blocker lies between the front piece and it.
                            let lifted = board.attacks(id, from, pos.occ & !square.bitboard());
                            for target in lifted & line_bb(from, square) {
                                if between_bb(from, target).check_bit(square) {
                                    want |= target.bitboard();
                                }
                            }
                        }
                    }
                    assert_eq!(board.xray_row(id, slider_squares), want, "{fen} ply {ply} slot {id:?}\n{pos}");
                }

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

                assert_eq!(board.checkers(&pos), pos.checkers(), "checkers, {fen} ply {ply}\n{pos}");

                // Per-piece mobility against the same count taken the long way.
                // The union cannot produce this number: it drops a square that
                // two pieces both reach to one.
                for color in [Color::White, Color::Black] {
                    let pinned = pos.pinned_pieces(color);
                    let ksq = pos.pieces(PieceType::King, color).lsb();
                    let area = !pos.side_bb[color];
                    let want: i32 = board.legal_rows(color, pinned, ksq).map(|(_, row)| (row & area).popcount() as i32).sum();

                    assert_eq!(board.mobility(color, pinned, ksq, area), want, "mobility {color:?}, {fen} ply {ply}");
                    assert!(
                        board.mobility(color, pinned, ksq, area) >= (board.attack_map(color, pinned, ksq) & area).popcount() as i32
                    );
                }
                board.unmake(mv, &undo);
                pos.unmake_move(mv, &state);
                assert_eq!(board.rows, before.rows, "unmake rows, {fen} ply {ply}\n{pos}");
                assert_eq!(board.at, before.at, "unmake mailbox, {fen} ply {ply}\n{pos}");
                assert_eq!(board.kind, before.kind, "unmake types, {fen} ply {ply}\n{pos}");
                pos.make_move(mv, &mut acc);
                board.snapshot(&mut undo);
                board.make(&pos, mv);
            }
        }
    }
}
