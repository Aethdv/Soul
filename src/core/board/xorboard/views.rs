//! What the rows are read for: the union views, the eval's per-piece counts and
//! the x-ray planes.
//!
//! Separate from the store because none of it is storage. These are reductions
//! over `rows` that happen to know what the evaluation wants, most of all the
//! pin policy, which is a scoring decision rather than a fact about the board.

use core::arch::x86_64::*;

use super::{PieceId, XorBoard, class_index, slots};
use crate::core::{
    board::bitboard::line_bb,
    defs::{Bitboard, Color, PieceType, Square},
};
#[cfg(not(target_feature = "avx512vpopcntdq"))]
use crate::weave::Vu64x4;

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
        // SAFETY: AVX2 per the compile_error gate in weave/mod.rs. A color is
        // slots 0 to 15 or 16 to 31, four whole groups either way, so the four
        // loads end exactly at the half's end.
        //
        // The masked union would load a lane mask per group, and for a whole
        // color those are all-ones or all-zeros: half of them mask in nothing
        // and the rest fetch a constant.
        unsafe {
            let base = color as usize * 16;
            let mut acc = _mm256_setzero_si256();
            for group in 0..4 {
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.rows.as_ptr().add(base + group * 4).cast()));
            }

            let folded = _mm_or_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            Bitboard((_mm_extract_epi64::<0>(folded) | _mm_extract_epi64::<1>(folded)).cast_unsigned())
        }
    }

    /// Scalar: the only caller is the debug-assert oracle.
    pub(super) fn class_attacks(&self, piece: PieceType, color: Color) -> Bitboard {
        slots(self.class[class_index(piece, color)]).fold(Bitboard(0), |acc, id| acc | self.row(id))
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
    pub(super) fn pinned_row(&self, id: PieceId, square: Square, row: Bitboard, ksq: Square) -> Bitboard {
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
                    acc = _mm256_add_epi64(acc, Vu64x4(_mm256_and_si256(rows, mask)).popcount().0);
                }

                let folded = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
                (_mm_extract_epi64::<0>(folded) + _mm_extract_epi64::<1>(folded)) as i32
            }
        }
    }
}
