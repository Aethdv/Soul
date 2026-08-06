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
//! Dormant. Maintaining it costs 162 instructions a node and every attack query
//! Soul makes is worth about 53, so hot magics and a setwise eval leave nothing
//! to win. Threat inputs would change what the diff stream is for without
//! settling that: Stockfish generates the same events from probes and keeps no
//! table.

use core::arch::x86_64::*;

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, line_bb},
        },
        defs::{Bitboard, Color, PieceType, Square},
        moves::Move,
    },
    weave::Vu64x4,
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

    /// Masked, so every `rows`/`sq`/`kind` access is in range by construction
    /// and the compiler drops the check. A slot out of range would be a bug the
    /// mask hides, and slots only ever come from a 32-bit mask's set bits or
    /// from `at`, both of which are already in range.
    #[inline(always)]
    const fn index(self) -> usize {
        (self.0 & 31) as usize
    }
}

/// Four lanes of all-ones or all-zeros, indexed by a nibble of the slot mask.
static LANE_MASK: [[u64; 4]; 16] = {
    let mut table = [[0u64; 4]; 16];
    let mut nibble = 0;
    while nibble < 16 {
        let mut lane = 0;
        while lane < 4 {
            table[nibble][lane] = if nibble >> lane & 1 == 1 { u64::MAX } else { 0 };
            lane += 1;
        }
        nibble += 1;
    }
    table
};

/// Square as an index into a 64-element array, masked so the bound is known.
trait SquareIndex {
    fn index(self) -> usize;
}

impl SquareIndex for Square {
    #[inline(always)]
    fn index(self) -> usize {
        usize::from(self.0 & 63)
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
const fn color_slots(color: Color) -> u64 {
    0xFFFF << (color as usize * 16)
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

                board.at[square.index()] = (slot + 1) as u8;
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
    /// Wider than `Position::threats`, which ends its setwise fill with
    /// `& !generator` over the whole rook-plus-queen union and so drops the
    /// squares holding that side's own rooks and queens. Nothing can stand on
    /// those, so the two are interchangeable to a consumer asking about its own
    /// pieces, and not to one asking about the board.
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

    /// Every piece of `color` bar the king, each with the squares it may legally
    /// use.
    ///
    /// One place knows the pin policy and the readers are reductions over it. A
    /// pinned slider may only use its pin ray and a pinned knight has no legal
    /// move at all; crediting either with more is mobility for a move that would
    /// leave the king in check. Pinned pawns are left whole, which is not that
    /// argument but a match for what the tensor does today.
    #[inline(always)]
    pub fn legal_rows(&self, color: Color, pinned: Bitboard, ksq: Square) -> impl Iterator<Item = (PieceId, Bitboard)> + '_ {
        let king = self.class[PieceType::King as usize * 2 + color as usize];

        slots(color_slots(color) & !king).filter_map(move |id| {
            let raw = self.sq[id.index()];
            if raw == NOWHERE {
                return None;
            }

            let square = Square(raw);
            let row = self.row(id);

            if !pinned.check_bit(square) {
                return Some((id, row));
            }

            Some(match self.kind[id.index()] {
                PieceType::Knight => (id, Bitboard(0)),
                PieceType::Bishop | PieceType::Rook | PieceType::Queen => (id, row & line_bb(ksq, square)),
                _ => (id, row),
            })
        })
    }

    /// The eval's attack map for `color`: the union of what its pieces may use.
    #[inline(always)]
    pub fn attack_map(&self, color: Color, pinned: Bitboard, ksq: Square) -> Bitboard {
        self.legal_rows(color, pinned, ksq).fold(Bitboard(0), |acc, (_, row)| acc | row)
    }

    /// Mobility counted per piece rather than over the union.
    ///
    /// A setwise fill cannot produce this: ORing the sides together loses which
    /// piece reached where, so a square two pieces both attack is worth one to
    /// the union and two here. The rows keep the identity.
    ///
    /// Counted over all sixteen slots at once, then corrected. The king is not
    /// a mobility piece and pinned pieces may use less than their row, and both
    /// are rare enough that fixing them up beats branching per piece: dead slots
    /// hold an empty row and correct themselves.
    #[inline(always)]
    pub fn mobility(&self, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
        let base = usize::from(color) * 16;
        let mut total = self.count_rows(base, area);

        if let Some(king) = self.id_at(ksq) {
            total -= (self.row(king) & area).popcount() as i32;
        }

        for square in pinned {
            let Some(id) = self.id_at(square) else { continue };

            let legal = match self.kind[id.index()] {
                PieceType::Knight => Bitboard(0),
                PieceType::Bishop | PieceType::Rook | PieceType::Queen => self.row(id) & line_bb(ksq, square),
                _ => continue,
            };

            total -= (self.row(id) & area).popcount() as i32;
            total += (legal & area).popcount() as i32;
        }
        total
    }

    /// Squares of `area` reached, summed over sixteen consecutive slots.
    #[inline(always)]
    fn count_rows(&self, base: usize, area: Bitboard) -> i32 {
        // SAFETY: AVX2 per the weave/mod.rs gate; `base` is 0 or 16, so the four
        // loads cover slots base..base+16 of a 32-element array.
        unsafe {
            #[cfg(target_feature = "avx512vpopcntdq")]
            {
                let mask = _mm512_set1_epi64(area.0 as i64);
                let mut acc = _mm512_setzero_si512();
                for group in 0..2 {
                    let rows = _mm512_loadu_si512(self.rows.as_ptr().add(base + group * 8).cast());
                    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(rows, mask)));
                }
                _mm512_reduce_add_epi64(acc) as i32
            }

            #[cfg(not(target_feature = "avx512vpopcntdq"))]
            {
                let mask = _mm256_set1_epi64x(area.0 as i64);
                let mut acc = _mm256_setzero_si256();
                for group in 0..4 {
                    let rows = _mm256_loadu_si256(self.rows.as_ptr().add(base + group * 4).cast());
                    acc = _mm256_add_epi64(acc, Vu64x4(_mm256_and_si256(rows, mask)).popcount().0);
                }

                let folded = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
                (_mm_extract_epi64(folded, 0) + _mm_extract_epi64(folded, 1)) as i32
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
        match self.at[square.index()] {
            0 => None,
            slot => Some(PieceId(slot - 1)),
        }
    }

    /// The transpose, taken one column at a time: which pieces attack any square
    /// in `mask`. Testing a row against a multi-bit mask answers for every square
    /// in it at once, so a castling move's four changed squares cost one pass.
    ///
    /// Written by hand: left to the autovectorizer it is a thirty-two iteration
    /// dependency chain rather than eight lane tests folded into a slot set.
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

        let mut slots = self.probe_side(king, pos.stm.opposite());
        let mut squares = Bitboard(0);
        while slots != 0 {
            let slot = slots.trailing_zeros() as usize;
            slots &= slots - 1;
            squares |= Square(self.sq[slot]).bitboard();
        }
        squares
    }

    /// The column test restricted to one colour's half of the slots.
    #[inline(always)]
    fn probe_side(&self, mask: Bitboard, color: Color) -> u64 {
        // SAFETY: as `probe_groups`; the group indices stay inside the half
        // belonging to `color`, so every load covers real rows.
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                let want = _mm512_set1_epi64(mask.0 as i64);
                let group = usize::from(color) * 2;
                let rows = _mm512_loadu_si512(self.rows.as_ptr().add(group * 8).cast());
                let more = _mm512_loadu_si512(self.rows.as_ptr().add((group + 1) * 8).cast());

                (u64::from(_mm512_test_epi64_mask(rows, want)) << (group * 8))
                    | (u64::from(_mm512_test_epi64_mask(more, want)) << ((group + 1) * 8))
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                let want = _mm256_set1_epi64x(mask.0 as i64);
                let zero = _mm256_setzero_si256();
                let base = usize::from(color) * 4;
                let mut set = 0u64;
                for step in 0..4 {
                    let group = base + step;
                    let rows = _mm256_loadu_si256(self.rows.as_ptr().add(group * 4).cast());
                    let idle = _mm256_cmpeq_epi64(_mm256_and_si256(rows, want), zero);
                    let live = !(_mm256_movemask_pd(_mm256_castsi256_pd(idle)) as u32) & 0xF;
                    set |= u64::from(live) << (group * 4);
                }
                set
            }
        }
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

    /// Attacks that pass through exactly one friendly piece, orthogonal and
    /// diagonal kept apart because the eval scores them apart.
    ///
    /// Lifting a slider's own first blockers out of occupancy and probing again
    /// continues each ray from where it stopped, which is what the tensor's
    /// second flood-fill does by feeding the friendly-hit squares back in as
    /// generators. Pinned pieces contribute nothing, matching the tensor.
    ///
    /// Measured slower than the tensor at eval density: two probes per slider
    /// against sixteen setwise fills covering both colours at once.
    #[inline(always)]
    pub fn xray_maps(&self, color: Color, pinned: Bitboard, own: Bitboard, occ: Bitboard) -> (Bitboard, Bitboard) {
        let (mut ortho, mut diag) = (Bitboard(0), Bitboard(0));
        let (mut ortho_direct, mut diag_direct) = (Bitboard(0), Bitboard(0));

        let mut rest =
            (self.class_slots(PieceType::Rook) | self.class_slots(PieceType::Bishop) | self.class_slots(PieceType::Queen))
                & color_slots(color);

        while rest != 0 {
            let id = PieceId(rest.trailing_zeros() as u8);
            rest &= rest - 1;
            // A captured piece keeps its class bit; only its row, square and
            // mailbox entry are cleared, so liveness has to be tested here.
            if self.sq[id.index()] == NOWHERE {
                continue;
            }

            let from = Square(self.sq[id.index()]);

            if pinned.check_bit(from) {
                continue;
            }

            let kind = self.kind[id.index()];
            let behind = occ & !(self.row(id) & own);

            if kind != PieceType::Bishop {
                let direct = atk_rook(from, occ);
                ortho_direct |= direct;
                ortho |= atk_rook(from, behind);
            }
            if kind != PieceType::Rook {
                let direct = atk_bishop(from, occ);
                diag_direct |= direct;
                diag |= atk_bishop(from, behind);
            }
        }
        (ortho & !ortho_direct, diag & !diag_direct)
    }

    /// Reduce-OR over the selected slots. Masked rather than iterated: the
    /// callers pass sixteen-bit selections, and a lane mask off a nibble table
    /// beats walking the set bits at that density.
    #[inline(always)]
    fn union(&self, slots: u64) -> Bitboard {
        // SAFETY: AVX2 per the compile_error gate in weave/mod.rs. Each load
        // covers one group of four of a 32-element array, and the nibble table
        // is indexed by four bits so it stays inside its sixteen rows.
        unsafe {
            let mut acc = _mm256_setzero_si256();
            for group in 0..8 {
                let rows = _mm256_loadu_si256(self.rows.as_ptr().add(group * 4).cast());
                let keep = _mm256_loadu_si256(LANE_MASK.as_ptr().add((slots >> (group * 4)) as usize & 15).cast());
                acc = _mm256_or_si256(acc, _mm256_and_si256(rows, keep));
            }

            let folded = _mm_or_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256(acc, 1));
            Bitboard((_mm_extract_epi64(folded, 0) | _mm_extract_epi64(folded, 1)) as u64)
        }
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
            self.at[square.index()] = 0;
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
        let mover = PieceId(self.at[from.index()] - 1);
        let mut movers = [(mover, from, to); 2];
        let mut changed = from.bitboard() | to.bitboard();
        let mut n_movers = 1;
        let mut victim = None;

        if mv.is_castling() {
            // The move encodes the rook's home square as its destination, so
            // both pieces and all four squares come out of that one pair.
            let rook = PieceId(self.at[to.index()] - 1);
            let (king_to, rook_to) = super::castling_targets(from, to);
            movers = [(mover, from, king_to), (rook, to, rook_to)];
            changed |= king_to.bitboard() | rook_to.bitboard();
            n_movers = 2;
        } else if mv.is_en_passant() {
            let square = Square(to.0 ^ 8);
            changed |= square.bitboard();
            victim = Some((PieceId(self.at[square.index()] - 1), square));
        } else if mv.is_capture() {
            victim = Some((PieceId(self.at[to.index()] - 1), to));
        }
        Plan { movers, n_movers, changed, victim }
    }

    /// Origins clear before any destination lands, so DFRC castling stays exact
    /// when the king comes to rest on the rook's own origin square.
    #[inline(always)]
    fn relocate(&mut self, mv: Move, plan: &Plan) {
        for m in 0..plan.n_movers {
            self.at[plan.movers[m].1.index()] = 0;
        }
        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            self.at[to.index()] = id.0 + 1;
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
            self.at[plan.movers[m].2.index()] = 0;
        }
        for m in 0..plan.n_movers {
            let (id, from, _) = plan.movers[m];
            self.at[from.index()] = id.0 + 1;
            self.sq[id.index()] = from.0;
        }

        if let Some((id, square)) = plan.victim {
            self.at[square.index()] = id.0 + 1;
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

    /// The tensor's x-ray fields for `color`.
    fn tensor_xray(pos: &Position, color: Color) -> (Bitboard, Bitboard) {
        use crate::core::board::spatial::SpatialTensor;

        let t = SpatialTensor::compute(pos, pos.pinned_pieces(Color::White).0, pos.pinned_pieces(Color::Black).0);

        if color == Color::White {
            (Bitboard(t.w_ortho_xray()), Bitboard(t.w_diag_xray()))
        } else {
            (Bitboard(t.b_ortho_xray()), Bitboard(t.b_diag_xray()))
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

                for color in [Color::White, Color::Black] {
                    let pinned = pos.pinned_pieces(color);
                    let own = pos.side_bb[color];
                    let (ortho, diag) = board.xray_maps(color, pinned, own, pos.occ);
                    let (want_o, want_d) = tensor_xray(&pos, color);
                    assert_eq!(ortho, want_o, "xray ortho {color:?}, {fen} ply {ply}\n{pos}");
                    assert_eq!(diag, want_d, "xray diag {color:?}, {fen} ply {ply}\n{pos}");
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
                board.make(&pos, mv, &mut undo);
            }
        }
    }
}
