#![cfg_attr(
    all(feature = "nostore", not(test)),
    expect(
        dead_code,
        reason = "nostore compiles the store without consulting it, so it can be priced against its own absence"
    )
)]

use core::arch::x86_64::*;

use crate::core::{
    board::{
        Position,
        bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_queen, atk_rook, line_bb},
    },
    defs::{Bitboard, Color, PieceType, Square},
    moves::Move,
};
#[cfg(not(target_feature = "avx512vpopcntdq"))]
use crate::weave::U64x4;

const SLOTS: usize = 32;
const NOWHERE: u8 = 0xFF;

/// Sliders are seeded into slots 0..8 and 16..24.
/// The gather usually checks four groups instead of eight.
const SLIDER_WINDOW: u64 = 0x00FF_00FF;

/// One bit per four-slot group in `column`.
const ALL_GROUPS: u8 = 0xFF;
const WINDOW_GROUPS: u8 = 0b0011_0011;
const WHITE_GROUPS: u8 = 0x0F;
const BLACK_GROUPS: u8 = 0xF0;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) struct PieceId(u8);

#[derive(Clone, Debug)]
pub(crate) struct XorBoard {
    /// id → attacked squares. Vacant rows are zero.
    rows: [u64; SLOTS],
    /// square → id + 1. Zero is empty.
    at: [u8; 64],
    /// id → square. `NOWHERE` is vacant.
    squares: [u8; SLOTS],
    kind: [PieceType; SLOTS],
    slider_slots: u64,
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Snapshot {
    rows: [u64; SLOTS],
    at: [u8; 64],
    squares: [u8; SLOTS],
}

#[derive(Clone, Copy, Debug)]
struct Relocation {
    id: PieceId,
    from: Square,
    to: Square,
}

impl PieceId {
    #[inline(always)]
    pub(crate) const fn color(self) -> Color { if self.0 < 16 { Color::White } else { Color::Black } }

    #[inline(always)]
    const fn index(self) -> usize {
        debug_assert!(self.0 < 32);
        (self.0 & 31) as usize
    }
}

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
fn is_slider(piece: PieceType) -> bool { matches!(piece, PieceType::Bishop | PieceType::Rook | PieceType::Queen) }

impl XorBoard {
    pub(crate) fn from_position(pos: &Position) -> Self {
        let mut board = Self {
            rows: [0; SLOTS],
            at: [0; 64],
            squares: [NOWHERE; SLOTS],
            kind: [PieceType::None; SLOTS],
            slider_slots: 0,
        };

        let mut next_slot = [0usize, 16];

        // Seed sliders before the other pieces.
        for slider_pass in [true, false] {
            for raw in 0..64u8 {
                let square = Square(raw);
                let piece = pos.piece_at(square);
                if piece == PieceType::None || is_slider(piece) != slider_pass {
                    continue;
                }

                let color = pos.color_at(square);
                let slot = next_slot[color as usize];
                next_slot[color as usize] += 1;

                board.set_entry_at(square, (slot + 1) as u8);
                board.squares[slot] = raw;
                board.kind[slot] = piece;
                board.slider_slots |= u64::from(is_slider(piece)) << slot;
            }
        }

        board.rebuild_rows(pos);
        board
    }

    fn rebuild_rows(&mut self, pos: &Position) {
        self.rows = [0; SLOTS];
        for slot in 0..SLOTS {
            if self.squares[slot] != NOWHERE {
                self.rows[slot] = self.attacks(PieceId(slot as u8), Square(self.squares[slot]), pos.occ).0;
            }
        }
    }

    #[inline(always)]
    pub(crate) fn row(&self, id: PieceId) -> Bitboard { Bitboard(self.rows[id.index()]) }

    #[inline(always)]
    pub(crate) fn id_at(&self, square: Square) -> Option<PieceId> {
        match self.entry_at(square) {
            0 => None,
            entry => Some(PieceId(entry - 1)),
        }
    }

    #[inline(always)]
    fn entry_at(&self, square: Square) -> u8 { self.at[usize::from(square.0 & 63)] }

    #[inline(always)]
    fn set_entry_at(&mut self, square: Square, slot: u8) { self.at[usize::from(square.0 & 63)] = slot; }

    #[inline(always)]
    fn slider_attackers_of(&self, squares: Bitboard) -> u64 {
        let sliders = self.slider_slots;
        if sliders & !SLIDER_WINDOW == 0 {
            self.column::<WINDOW_GROUPS>(squares) & sliders
        } else {
            self.column::<ALL_GROUPS>(squares) & sliders
        }
    }

    #[inline(always)]
    pub(crate) fn checkers(&self, pos: &Position) -> Bitboard {
        let king = pos.pieces(PieceType::King, pos.stm);
        if king.is_empty() {
            return Bitboard(0);
        }

        let checker_ids = match pos.stm.opposite() {
            Color::White => self.column::<WHITE_GROUPS>(king),
            Color::Black => self.column::<BLACK_GROUPS>(king),
        };

        slots(checker_ids).fold(Bitboard(0), |squares, id| squares | Square(self.squares[id.index()]).bitboard())
    }

    /// Slots in the selected four-row groups whose row intersects `squares`.
    /// Groups come in pairs because the AVX512 path loads eight rows at once.
    #[inline(always)]
    fn column<const GROUPS: u8>(&self, squares: Bitboard) -> u64 {
        const { assert!(GROUPS & 0x55 == (GROUPS >> 1) & 0x55, "groups must come in pairs") }

        // SAFETY: every load stays within the 32 rows.
        unsafe {
            #[cfg(target_feature = "avx512f")]
            {
                let want = _mm512_set1_epi64(squares.0.cast_signed());
                let mut set = 0u64;

                for pair in 0..4 {
                    if GROUPS >> (pair * 2) & 3 != 0 {
                        let rows = _mm512_loadu_si512(self.rows.as_ptr().add(pair * 8).cast());
                        set |= u64::from(_mm512_test_epi64_mask(rows, want)) << (pair * 8);
                    }
                }
                set
            }

            #[cfg(not(target_feature = "avx512f"))]
            {
                let want = _mm256_set1_epi64x(squares.0.cast_signed());
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

    #[inline(always)]
    fn set_kind(&mut self, id: PieceId, kind: PieceType) {
        let slot = id.index();
        self.kind[slot] = kind;
        self.slider_slots = (self.slider_slots & !(1 << slot)) | (u64::from(is_slider(kind)) << slot);
    }

    #[inline(always)]
    fn attacks(&self, id: PieceId, from: Square, occ: Bitboard) -> Bitboard {
        match self.kind[id.index()] {
            PieceType::Pawn => atk_pawn(from, id.color()),
            PieceType::Knight => atk_knight(from),
            PieceType::King => atk_king(from),
            PieceType::Bishop => atk_bishop(from, occ),
            PieceType::Rook => atk_rook(from, occ),
            PieceType::Queen => atk_queen(from, occ),
            PieceType::None => Bitboard(0),
        }
    }

    pub(crate) fn tracks(&self, pos: &Position) -> bool {
        for raw in 0..64u8 {
            let square = Square(raw);

            match self.id_at(square) {
                Some(id) => {
                    let piece = pos.piece_at(square);
                    if piece == PieceType::None
                        || id.color() != pos.color_at(square)
                        || self.kind[id.index()] != piece
                        || (self.slider_slots >> id.index()) & 1 != u64::from(is_slider(piece))
                        || self.row(id) != self.attacks(id, square, pos.occ)
                    {
                        return false;
                    }
                },
                None if pos.piece_at(square) != PieceType::None => return false,
                None => {},
            }
        }

        (0..SLOTS).all(|slot| {
            let raw = self.squares[slot];
            if raw == NOWHERE {
                self.rows[slot] == 0
            } else {
                raw < 64 && self.id_at(Square(raw)) == Some(PieceId(slot as u8))
            }
        })
    }

    /// `pos` has already made `mv`; this board has not. An unmoved slider can only
    /// change if its old row crossed a square whose occupancy changed.
    pub(crate) fn update(&mut self, pos: &Position, mv: Move) {
        let (relocations, captured, occupancy_delta) = self.decode(mv);
        let [first, second] = relocations;
        let castling = mv.is_castling();

        let mut affected = self.slider_attackers_of(occupancy_delta) & !(1 << first.id.index()) & !(1 << second.id.index());

        if let Some((victim, square)) = captured {
            affected &= !(1 << victim.index());
            self.rows[victim.index()] = 0;
            self.set_entry_at(square, 0);
            self.squares[victim.index()] = NOWHERE;
        }

        self.relocate(relocations, castling);
        if let Some(promoted) = mv.promo() {
            self.set_kind(first.id, promoted);
        }

        for id in slots(affected) {
            self.rows[id.index()] = self.attacks(id, Square(self.squares[id.index()]), pos.occ).0;
        }

        self.rows[first.id.index()] = self.attacks(first.id, first.to, pos.occ).0;
        if castling {
            self.rows[second.id.index()] = self.attacks(second.id, second.to, pos.occ).0;
        }
    }

    #[inline(always)]
    pub(crate) fn snapshot(&self, snapshot: &mut Snapshot) {
        snapshot.rows = self.rows;
        snapshot.at = self.at;
        snapshot.squares = self.squares;
    }

    /// Promotion metadata is omitted from the snapshot and restored from `mv`.
    pub(crate) fn restore(&mut self, mv: Move, snapshot: &Snapshot) {
        self.rows = snapshot.rows;
        self.at = snapshot.at;
        self.squares = snapshot.squares;

        if mv.is_promotion() {
            self.set_kind(PieceId(self.entry_at(mv.from()) - 1), PieceType::Pawn);
        }
    }

    #[inline(always)]
    fn decode(&self, mv: Move) -> ([Relocation; 2], Option<(PieceId, Square)>, Bitboard) {
        let (from, to) = (mv.from(), mv.to());
        let mover = PieceId(self.entry_at(from) - 1);

        // Duplicate the sole mover so both ids can be masked unconditionally.
        let mut relocations = [Relocation { id: mover, from, to }; 2];
        let mut captured = None;
        let mut vacated = from.bitboard();
        let mut filled = to.bitboard();

        if mv.is_castling() {
            // Castling encodes the rook's origin as to.
            let rook = PieceId(self.entry_at(to) - 1);
            let (king_to, rook_to) = super::castling_targets(from, to);
            relocations = [Relocation { id: mover, from, to: king_to }, Relocation { id: rook, from: to, to: rook_to }];
            vacated |= to.bitboard();
            filled = king_to.bitboard() | rook_to.bitboard();
        } else if mv.is_en_passant() {
            let victim_square = Square(to.0 ^ 8);
            vacated |= victim_square.bitboard();
            captured = Some((PieceId(self.entry_at(victim_square) - 1), victim_square));
        } else if mv.is_capture() {
            vacated |= to.bitboard();
            captured = Some((PieceId(self.entry_at(to) - 1), to));
        }

        // Squares both vacated and filled leave occupancy unchanged.
        (relocations, captured, vacated ^ filled)
    }

    /// Clear both origins before writing either destination.
    /// Required for overlapping DFRC castling squares.
    #[inline(always)]
    fn relocate(&mut self, [first, second]: [Relocation; 2], castling: bool) {
        self.set_entry_at(first.from, 0);
        if castling {
            self.set_entry_at(second.from, 0);
        }

        self.set_entry_at(first.to, first.id.0 + 1);
        self.squares[first.id.index()] = first.to.0;

        if castling {
            self.set_entry_at(second.to, second.id.0 + 1);
            self.squares[second.id.index()] = second.to.0;
        }
    }
}

impl Default for Snapshot {
    fn default() -> Self { Self { rows: [0; SLOTS], at: [0; 64], squares: [NOWHERE; SLOTS] } }
}

impl XorBoard {
    #[inline(always)]
    pub(crate) fn threats(&self, color: Color) -> Bitboard {
        // SAFETY: four loads cover exactly one 16-row color half.
        unsafe {
            let base = usize::from(color) * 16;
            let mut acc = _mm256_setzero_si256();

            for group in 0..4 {
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.rows.as_ptr().add(base + group * 4).cast()));
            }

            let folded = _mm_or_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            Bitboard((_mm_extract_epi64::<0>(folded) | _mm_extract_epi64::<1>(folded)).cast_unsigned())
        }
    }

    /// Counts attacks per piece; overlapping squares count once for each attacker.
    #[inline(always)]
    pub(crate) fn mobility(&self, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
        let mut total = self.row_attack_count(color, area);

        let king = self.id_at(ksq);
        debug_assert!(king.is_some(), "king square absent from xorboard");
        if let Some(king) = king {
            total -= (self.row(king) & area).popcount() as i32;
        }

        for square in pinned {
            let Some(id) = self.id_at(square) else { continue };
            let row = self.row(id);
            let constrained = self.pinned_row(id, square, row, ksq);

            if constrained == row {
                continue;
            }

            total -= (row & area).popcount() as i32;
            total += (constrained & area).popcount() as i32;
        }

        total
    }

    /// Pawn rows stay unfiltered to match the mobility attack maps.
    #[inline(always)]
    fn pinned_row(&self, id: PieceId, square: Square, row: Bitboard, ksq: Square) -> Bitboard {
        match self.kind[id.index()] {
            PieceType::Knight => Bitboard(0),
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => row & line_bb(ksq, square),
            _ => row,
        }
    }

    #[inline(always)]
    fn row_attack_count(&self, color: Color, area: Bitboard) -> i32 {
        let base = usize::from(color) * 16;

        // SAFETY: base selects one 16-row color half, which both paths load exactly.
        unsafe {
            #[cfg(target_feature = "avx512vpopcntdq")]
            {
                let mask = _mm512_set1_epi64(area.0.cast_signed());
                let mut acc = _mm512_setzero_si512();

                for group in 0..2 {
                    let rows = _mm512_loadu_si512(self.rows.as_ptr().add(base + group * 8).cast());
                    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_and_si512(rows, mask)));
                }

                _mm512_reduce_add_epi64(acc) as i32
            }

            #[cfg(not(target_feature = "avx512vpopcntdq"))]
            {
                let mask = _mm256_set1_epi64x(area.0.cast_signed());
                let mut acc = _mm256_setzero_si256();

                for group in 0..4 {
                    let rows = _mm256_loadu_si256(self.rows.as_ptr().add(base + group * 4).cast());
                    acc = _mm256_add_epi64(acc, U64x4(_mm256_and_si256(rows, mask)).popcount().0);
                }

                let folded = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
                (_mm_extract_epi64::<0>(folded) + _mm_extract_epi64::<1>(folded)) as i32
            }
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

    const fn color_mask(color: Color) -> u64 { 0xFFFF << (color as usize * 16) }

    impl XorBoard {
        fn mobility_rows(&self, color: Color, pinned: Bitboard, ksq: Square) -> impl Iterator<Item = Bitboard> + '_ {
            slots(color_mask(color)).filter_map(move |id| {
                let raw = self.squares[id.index()];
                if raw == NOWHERE || raw == ksq.0 {
                    return None;
                }

                let square = Square(raw);
                let row = self.row(id);
                Some(if pinned.check_bit(square) { self.pinned_row(id, square, row, ksq) } else { row })
            })
        }
    }

    #[test]
    fn tracks_and_restores() {
        let mut rng = ConstRng::new(0xC0FFEE);

        for fen in FENS {
            let mut pos = Position::from_fen(fen);
            let mut acc = pos.psqt_accumulator();
            let mut board = XorBoard::from_position(&pos);
            let mut snapshot = Snapshot::default();

            for ply in 0..512 {
                let legal = gen_legal_moves(&pos);
                if legal.is_empty() {
                    break;
                }

                let mv = legal[(rng.next() % legal.len() as u64) as usize];
                let before = board.clone();
                let saved_acc = acc;

                board.snapshot(&mut snapshot);
                let state = pos.make_move(mv, &mut acc);
                board.update(&pos, mv);

                assert!(board.tracks(&pos), "{fen} ply {ply} move {}\n{pos}", mv.to_uci(pos.frc_notation));

                for color in [Color::White, Color::Black] {
                    assert_eq!(board.threats(color), pos.threats(color), "threats {color:?}, {fen} ply {ply}\n{pos}");
                }

                assert_eq!(board.checkers(&pos), pos.checkers(), "checkers, {fen} ply {ply}\n{pos}");

                for color in [Color::White, Color::Black] {
                    let pinned = pos.pinned_pieces(color);
                    let ksq = pos.pieces(PieceType::King, color).lsb();
                    let area = !pos.side_bb[color];
                    let want: i32 = board.mobility_rows(color, pinned, ksq).map(|row| (row & area).popcount() as i32).sum();

                    assert_eq!(board.mobility(color, pinned, ksq, area), want, "mobility {color:?}, {fen} ply {ply}");
                }

                board.restore(mv, &snapshot);
                pos.unmake_move(mv, &state);
                acc = saved_acc;
                assert_eq!(board.rows, before.rows, "restore rows, {fen} ply {ply}\n{pos}");
                assert_eq!(board.at, before.at, "restore mailbox, {fen} ply {ply}\n{pos}");
                assert_eq!(board.squares, before.squares, "restore squares, {fen} ply {ply}\n{pos}");
                assert_eq!(board.kind, before.kind, "restore kind, {fen} ply {ply}\n{pos}");
                assert_eq!(board.slider_slots, before.slider_slots, "restore slider slots, {fen} ply {ply}\n{pos}");
                pos.make_move(mv, &mut acc);
                board.update(&pos, mv);
            }
        }
    }
}
