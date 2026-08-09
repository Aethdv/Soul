//! What the rows are read for: the union views, the eval's per-piece counts and
//! the x-ray planes.
//!
//! Separate from the store because none of it is storage. These are reductions
//! over `rows` that happen to know what the evaluation wants, most of all the
//! pin policy, which is a scoring decision rather than a fact about the board.

use core::arch::x86_64::*;

use super::{NOWHERE, PieceId, XorBoard, class_index, color_slots, slots};
use crate::{
    core::{
        board::bitboard::{atk_bishop, atk_rook, line_bb},
        defs::{Bitboard, Color, PieceType, Square},
    },
    weave::Vu64x4,
};

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

impl XorBoard {
    /// Every square `color` attacks.
    ///
    /// Wider than `Position::threats`, which ends its setwise fill with
    /// `& !generator` over the whole rook-plus-queen union and so drops the
    /// squares holding that side's own rooks and queens. Nothing can stand on
    /// those, so the two are interchangeable to a consumer asking about its own
    /// pieces, and not to one asking about the board.
    #[inline(always)]
    pub fn danger(&self, color: Color) -> Bitboard {
        self.union(color_slots(color))
    }

    /// Every square the given class attacks.
    #[inline(always)]
    pub fn class_attacks(&self, piece: PieceType, color: Color) -> Bitboard {
        self.union(self.class[class_index(piece, color)])
    }

    /// Every piece of `color` bar the king, each with the squares it may legally
    /// use, pins applied by `pinned_row`.
    #[inline(always)]
    pub fn legal_rows(&self, color: Color, pinned: Bitboard, ksq: Square) -> impl Iterator<Item = (PieceId, Bitboard)> + '_ {
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
            let row = self.row(id);
            let legal = self.pinned_row(id, square, row, ksq);
            if legal == row {
                continue;
            }
            total -= (row & area).popcount() as i32;
            total += (legal & area).popcount() as i32;
        }
        total
    }

    /// What a pinned piece may still use: a slider keeps its pin ray, a pinned
    /// knight has no legal move at all, and a pawn is left whole to match what
    /// the tensor does today. Crediting either of the first two with more is
    /// mobility for a move that would leave the king in check.
    #[inline(always)]
    fn pinned_row(&self, id: PieceId, square: Square, row: Bitboard, ksq: Square) -> Bitboard {
        match self.kind[id.index()] {
            PieceType::Knight => Bitboard(0),
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => row & line_bb(ksq, square),
            _ => row,
        }
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
            if self.squares[id.index()] == NOWHERE {
                continue;
            }

            let from = Square(self.squares[id.index()]);

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
}
