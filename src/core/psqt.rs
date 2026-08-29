//! Piece-Square Tables (PSQT) and static evaluation initialization.
//!
//! Precomputes the static material and positional values for each piece on every square,
//! packing them into SIMD-aligned structures for fast incremental updates.

use std::{arch, mem};

use crate::{
    core::defs::{Color, PieceType, Square},
    engine::eval_params::{
        EG_BISHOP, EG_KING, EG_KNIGHT, EG_MATERIAL, EG_PAWN, EG_QUEEN, EG_ROOK, MG_BISHOP, MG_KING, MG_KNIGHT, MG_MATERIAL,
        MG_PAWN, MG_QUEEN, MG_ROOK, PHASE,
    },
    weave::Vi16x8,
};

const _: () = assert!(mem::size_of::<[i16; 8]>().is_multiple_of(16), "SIMD accumulator must be 16-byte aligned");

pub static PSQT: [AlignedTable; 14] = init_psqt();

/// [`PieceType` + `ColorOffset`][`Square`] → [i16; 8]
/// Lanes: [`MG`, `EG`, `Phase`, 0, 0, 0, 0, 0]
/// Alignment: 32 bytes ensures that every 16-byte `[i16; 8]` entry is strictly aligned
/// for safe `_mm_load_si128` vector loads.
#[repr(align(32))]
#[derive(Clone, Copy)]
pub struct AlignedTable(pub [[i16; 8]; 64]);

/// PSQT vector for piece-square table lookup.
#[inline(always)]
pub fn entry(pt: PieceType, sq: Square, c: Color) -> Vi16x8 {
    let idx = usize::from(pt) + (usize::from(c) * 7);
    debug_assert!(idx < 14, "PSQT index out of bounds: {idx} >= 14");
    debug_assert!(usize::from(sq) < 64, "Square index out of bounds: {sq} >= 64");

    // SAFETY: idx = pt + color·7 ≤ 6 + 7 = 13 < 14 (PieceType ≤ 6, Color ≤ 1) and
    // sq < 64 by the Square invariant, so both get_unchecked indices are in bounds.
    let lanes = unsafe { PSQT.get_unchecked(idx).0.get_unchecked(usize::from(sq)) };
    // SAFETY: AlignedTable's 32-byte alignment makes each [i16; 8] entry 16-byte aligned.
    Vi16x8(unsafe { arch::x86_64::_mm_load_si128(lanes.as_ptr() as *const _) })
}

/// Mirror square horizontally; files E-H map to D-A
/// Input: 0..63, Output: 0..31
#[inline]
pub const fn mirror_sq(sq: usize) -> usize {
    const MIRROR_FILE: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

    let file = sq & 7;
    let rank = sq >> 3;
    (rank << 2) + MIRROR_FILE[file]
}

#[inline]
const fn init_psqt() -> [AlignedTable; 14] {
    let mut tables = [AlignedTable([[0; 8]; 64]); 14];
    let mut pt = 0;
    while pt < 6 {
        let mg_w = MG_MATERIAL[pt];
        let eg_w = EG_MATERIAL[pt];
        let ph_w = PHASE[pt];

        let mut sq = 0;
        while sq < 64 {
            let w_visual_idx = sq ^ 0x38;
            let mg_val = clamp_i16(mg_w + raw_mg(pt, w_visual_idx));
            let eg_val = clamp_i16(eg_w + raw_eg(pt, w_visual_idx));

            tables[pt].0[sq] = [mg_val, eg_val, ph_w as i16, 0, 0, 0, 0, 0];

            let b_visual_idx = sq;
            let mg_val_b = clamp_i16(-(mg_w + raw_mg(pt, b_visual_idx)));
            let eg_val_b = clamp_i16(-(eg_w + raw_eg(pt, b_visual_idx)));

            // PSQT values for Black are mirrored using 'sq'.
            // This allows us to use the same tables built from White's perspective,
            // as piece movement symmetry holds when ranks are flipped.
            tables[pt + 7].0[sq] = [mg_val_b, eg_val_b, ph_w as i16, 0, 0, 0, 0, 0];
            sq += 1;
        }
        pt += 1;
    }
    tables
}

/// Saturating i32 → i16 cast.
const fn clamp_i16(v: i32) -> i16 { v.clamp(i16::MIN as i32, i16::MAX as i32) as i16 }

#[inline]
const fn raw_mg(pt: usize, sq64: usize) -> i32 {
    let idx = mirror_sq(sq64);
    match pt {
        0 => MG_PAWN[idx],
        1 => MG_KNIGHT[idx],
        2 => MG_BISHOP[idx],
        3 => MG_ROOK[idx],
        4 => MG_QUEEN[idx],
        5 => MG_KING[idx],
        _ => 0,
    }
}

#[inline]
const fn raw_eg(pt: usize, sq64: usize) -> i32 {
    let idx = mirror_sq(sq64);
    match pt {
        0 => EG_PAWN[idx],
        1 => EG_KNIGHT[idx],
        2 => EG_BISHOP[idx],
        3 => EG_ROOK[idx],
        4 => EG_QUEEN[idx],
        5 => EG_KING[idx],
        _ => 0,
    }
}
