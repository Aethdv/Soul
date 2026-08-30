//! Reductions over `rows`, as opposed to keeping them. The pin policy lives here
//! rather than in the store, being a scoring decision rather than a fact about
//! the board.

use core::arch::x86_64::*;

use super::{NOWHERE, PieceId, XorBoard, class_index, slots};
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
    /// The king is out, and so is every class the bucketed tables have taken over.
    #[inline(always)]
    pub fn mobility(&self, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
        let base = usize::from(color) * 16;
        let counted = self.summed_slots(color);
        let mut total = self.count_rows(base, area, counted);

        for square in pinned {
            let Some(id) = self.id_at(square) else { continue };
            if counted >> id.index() & 1 == 0 {
                continue;
            }
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

    /// A class leaves here the moment it gets its own table, or its pieces score twice.
    #[inline(always)]
    fn summed_slots(&self, color: Color) -> u64 {
        !(self.class[class_index(PieceType::Bishop, color)] | self.class[class_index(PieceType::King, color)])
    }

    /// A vacant slot keeps its class bit, so skipping it is what stops a captured bishop
    /// scoring as an immobile one.
    #[inline(always)]
    pub fn bishop_buckets(&self, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard, out: &mut [i32]) {
        for id in slots(self.class[class_index(PieceType::Bishop, color)]) {
            let raw = self.squares[id.index()];
            if raw == NOWHERE {
                continue;
            }

            let square = Square(raw);
            let row = self.row(id);
            let legal = if pinned.check_bit(square) { self.pinned_row(id, square, row, ksq) } else { row };
            out[((legal & area).popcount() as usize).min(out.len() - 1)] += 1;
        }
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

    /// Only bits `base..base + 16` of `counted` are read, so a caller may pass a mask
    /// spanning both colors.
    #[inline(always)]
    fn count_rows(&self, base: usize, area: Bitboard, counted: u64) -> i32 {
        // SAFETY: AVX2 per the weave/mod.rs gate; `base` is 0 or 16, so the four
        // loads cover slots base..base+16 of a 32-element array.
        unsafe {
            #[cfg(target_feature = "avx512vpopcntdq")]
            {
                let mask = _mm512_set1_epi64(area.0.cast_signed());
                let mut acc = _mm512_setzero_si512();
                for group in 0..2 {
                    let rows = _mm512_loadu_si512(self.rows.as_ptr().add(base + group * 8).cast());
                    let keep = ((counted >> (base + group * 8)) & 0xFF) as u8;
                    acc = _mm512_add_epi64(acc, _mm512_popcnt_epi64(_mm512_maskz_and_epi64(keep, rows, mask)));
                }
                _mm512_reduce_add_epi64(acc) as i32
            }

            #[cfg(not(target_feature = "avx512vpopcntdq"))]
            {
                let mask = _mm256_set1_epi64x(area.0.cast_signed());
                // Lane j of a group holds slot base + group * 4 + j, so bit 1 << j selects it.
                let lanes = _mm256_set_epi64x(8, 4, 2, 1);
                let mut acc = _mm256_setzero_si256();
                for group in 0..4 {
                    let rows = _mm256_loadu_si256(self.rows.as_ptr().add(base + group * 4).cast());
                    let nibble = _mm256_set1_epi64x(i64::from((counted >> (base + group * 4)) as u8 & 0xF));
                    let keep = _mm256_cmpeq_epi64(_mm256_and_si256(nibble, lanes), lanes);
                    let selected = _mm256_and_si256(_mm256_and_si256(rows, mask), keep);
                    acc = _mm256_add_epi64(acc, U64x4(selected).popcount().0);
                }

                let folded = _mm_add_epi64(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
                (_mm_extract_epi64::<0>(folded) + _mm_extract_epi64::<1>(folded)) as i32
            }
        }
    }
}
