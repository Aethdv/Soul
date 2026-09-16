//! Reductions over `rows`, as opposed to keeping them. The pin policy lives here
//! rather than in the store, being a scoring decision rather than a fact about
//! the board.

use core::arch::x86_64::*;

use super::{BLACK_GROUPS, PieceId, WHITE_GROUPS, XorBoard, class_index, slots};
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

    #[inline(always)]
    pub fn attackers(&self, square: Square, color: Color) -> u16 {
        match color {
            Color::White => self.column::<WHITE_GROUPS>(square.bitboard()) as u16,
            Color::Black => (self.column::<BLACK_GROUPS>(square.bitboard()) >> 16) as u16,
        }
    }

    pub fn columns(&self, color: Color) -> [u16; 64] {
        let base = usize::from(color) * 16;
        let mut out = [0u16; 64];

        // SAFETY: AVX2 per the weave/mod.rs gate.
        unsafe {
            let p = self.rows.as_ptr().add(base);
            let pair = |a: usize, b: usize| _mm_set_epi64x(*p.add(b) as i64, *p.add(a) as i64);

            // The ladder emits rows in the order 0,2,4,6,1,3,5,7 within each
            // half, so the pairs going in are its inverse.
            let (v0, v1) = (pair(0, 4), pair(1, 5));
            let (v2, v3) = (pair(2, 6), pair(3, 7));
            let (v4, v5) = (pair(8, 12), pair(9, 13));
            let (v6, v7) = (pair(10, 14), pair(11, 15));

            let a0 = _mm_unpacklo_epi8(v0, v1);
            let a1 = _mm_unpackhi_epi8(v0, v1);
            let a2 = _mm_unpacklo_epi8(v2, v3);
            let a3 = _mm_unpackhi_epi8(v2, v3);
            let a4 = _mm_unpacklo_epi8(v4, v5);
            let a5 = _mm_unpackhi_epi8(v4, v5);
            let a6 = _mm_unpacklo_epi8(v6, v7);
            let a7 = _mm_unpackhi_epi8(v6, v7);

            let b0 = _mm_unpacklo_epi16(a0, a2);
            let b1 = _mm_unpackhi_epi16(a0, a2);
            let b2 = _mm_unpacklo_epi16(a1, a3);
            let b3 = _mm_unpackhi_epi16(a1, a3);
            let b4 = _mm_unpacklo_epi16(a4, a6);
            let b5 = _mm_unpackhi_epi16(a4, a6);
            let b6 = _mm_unpacklo_epi16(a5, a7);
            let b7 = _mm_unpackhi_epi16(a5, a7);

            let c0 = _mm_unpacklo_epi32(b0, b2);
            let c1 = _mm_unpackhi_epi32(b0, b2);
            let c2 = _mm_unpacklo_epi32(b1, b3);
            let c3 = _mm_unpackhi_epi32(b1, b3);
            let c4 = _mm_unpacklo_epi32(b4, b6);
            let c5 = _mm_unpackhi_epi32(b4, b6);
            let c6 = _mm_unpacklo_epi32(b5, b7);
            let c7 = _mm_unpackhi_epi32(b5, b7);

            let byte0 = _mm_unpacklo_epi64(c0, c4);
            let byte1 = _mm_unpackhi_epi64(c0, c4);
            let byte2 = _mm_unpacklo_epi64(c1, c5);
            let byte3 = _mm_unpackhi_epi64(c1, c5);
            let byte4 = _mm_unpacklo_epi64(c2, c6);
            let byte5 = _mm_unpackhi_epi64(c2, c6);
            let byte6 = _mm_unpacklo_epi64(c3, c7);
            let byte7 = _mm_unpackhi_epi64(c3, c7);

            let join = |lo, hi| _mm256_inserti128_si256::<1>(_mm256_castsi128_si256(lo), hi);
            let g = [join(byte0, byte1), join(byte2, byte3), join(byte4, byte5), join(byte6, byte7)];

            plane::<7>(&g, &mut out);
            plane::<6>(&g, &mut out);
            plane::<5>(&g, &mut out);
            plane::<4>(&g, &mut out);
            plane::<3>(&g, &mut out);
            plane::<2>(&g, &mut out);
            plane::<1>(&g, &mut out);
            plane::<0>(&g, &mut out);
        }
        out
    }

    #[cfg(test)]
    pub(super) fn columns_scalar(&self, color: Color) -> [u16; 64] {
        let base = usize::from(color) * 16;
        let mut out = [0u16; 64];
        for slot in 0..16 {
            for sq in Bitboard(self.rows[base + slot]) {
                out[usize::from(sq.0)] |= 1 << slot;
            }
        }
        out
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

fn plane<const SHIFT: i32>(g: &[__m256i; 4], out: &mut [u16; 64]) {
    let j = 7 - SHIFT as usize;

    for (t, &rows) in g.iter().enumerate() {
        // SAFETY: AVX2 per the weave/mod.rs gate. The shift drags bits across
        // byte boundaries and only bit 7 of each byte reaches the mask, which
        // took it from bit j of that same byte.
        let m = unsafe { _mm256_movemask_epi8(_mm256_slli_epi64::<SHIFT>(rows)) }.cast_unsigned();
        out[16 * t + j] = m as u16;
        out[16 * t + 8 + j] = (m >> 16) as u16;
    }
}
