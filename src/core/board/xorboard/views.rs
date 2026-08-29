//! Reductions over `rows`, as opposed to keeping them. The pin policy lives here
//! rather than in the store, being a scoring decision rather than a fact about
//! the board.

use core::arch::x86_64::*;

use super::{PieceId, XorBoard, class_index, slots};
use crate::core::{
    board::bitboard::line_bb,
    defs::{Bitboard, Color, PieceType, Square},
};
#[cfg(not(target_feature = "avx512vpopcntdq"))]
use crate::weave::U64x4;

impl XorBoard {
    /// Every square `color` attacks. Wider than `Position::threats`, whose fill
    /// ends `& !generator` and so drops the squares holding that side's own rooks
    /// and queens.
    #[inline(always)]
    pub fn danger(&self, color: Color) -> Bitboard {
        // SAFETY: AVX2 per the compile_error gate in weave/mod.rs. A color is
        // slots 0 to 15 or 16 to 31, four whole groups either way, so the four
        // loads end exactly at the half's end.
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

    /// Per piece, so a square two pieces attack counts twice.
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

    /// A pawn is left whole to match the maps; crediting a pinned slider or
    /// knight with more would count moves that leave the king in check.
    #[inline(always)]
    pub(super) fn pinned_row(&self, id: PieceId, square: Square, row: Bitboard, ksq: Square) -> Bitboard {
        match self.kind[id.index()] {
            PieceType::Knight => Bitboard(0),
            PieceType::Bishop | PieceType::Rook | PieceType::Queen => row & line_bb(ksq, square),
            _ => row,
        }
    }

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
                    acc = _mm256_add_epi64(acc, U64x4(_mm256_and_si256(rows, mask)).popcount().0);
                }

                let folded = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
                (_mm_extract_epi64::<0>(folded) + _mm_extract_epi64::<1>(folded)) as i32
            }
        }
    }
}
