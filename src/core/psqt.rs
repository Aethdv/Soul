//! Precomputed material, piece-square and phase contributions for the incremental accumulator.

use std::arch;

use crate::{
    core::defs::{Color, PieceType, Square},
    engine::eval_params::{
        EG_BISHOP, EG_KING, EG_KNIGHT, EG_MATERIAL, EG_PAWN, EG_QUEEN, EG_ROOK, MG_BISHOP, MG_KING, MG_KNIGHT, MG_MATERIAL,
        MG_PAWN, MG_QUEEN, MG_ROOK, PHASE,
    },
    weave::I16x8,
};

/// `[piece type + 7 · color][square]` with lanes `[MG, EG, phase, 0, 0, 0, 0, 0]`.
pub(crate) static PSQT: [AlignedTable; 14] = init_psqt();

/// 32-byte aligned so every `[i16; 8]` entry is valid for `_mm_load_si128`.
#[repr(align(32))]
#[derive(Clone, Copy)]
pub(crate) struct AlignedTable([[i16; 8]; 64]);

#[inline(always)]
pub(crate) fn entry(pt: PieceType, sq: Square, c: Color) -> I16x8 {
    let idx = usize::from(pt) + usize::from(c) * 7;

    // SAFETY: PieceType <= 6 and Color <= 1, so idx <= 13.
    let table = unsafe { PSQT.get_unchecked(idx) };
    let lanes = &table.0[usize::from(sq)];

    // SAFETY: AlignedTable's 32-byte alignment makes every [i16; 8] entry 16-byte aligned.
    I16x8(unsafe { arch::x86_64::_mm_load_si128(lanes.as_ptr() as *const _) })
}

/// Folds files e-h onto d-a, mapping square indices 0..=63 to 0..=31.
#[inline]
pub(crate) const fn mirror_sq(sq: usize) -> usize {
    const MIRROR_FILE: [usize; 8] = [0, 1, 2, 3, 3, 2, 1, 0];

    let file = sq & 7;
    let rank = sq >> 3;
    (rank << 2) + MIRROR_FILE[file]
}

const fn init_psqt() -> [AlignedTable; 14] {
    let mut tables = [AlignedTable([[0; 8]; 64]); 14];
    let mut pt = 0;
    while pt < 6 {
        let mg_w = MG_MATERIAL[pt];
        let eg_w = EG_MATERIAL[pt];
        let ph_w = PHASE[pt];

        let mut sq = 0;
        while sq < 64 {
            // The raw tables start at rank 8 from White's perspective.
            // White flips the rank; Black uses the square directly and negates the score.
            let w_visual_idx = sq ^ 0x38;
            let (mg, eg) = raw_psqt(pt, w_visual_idx);
            let mg_val = clamp_i16(mg_w + mg);
            let eg_val = clamp_i16(eg_w + eg);

            tables[pt].0[sq] = [mg_val, eg_val, ph_w as i16, 0, 0, 0, 0, 0];

            let b_visual_idx = sq;
            let (mg, eg) = raw_psqt(pt, b_visual_idx);
            let mg_val_b = clamp_i16(-(mg_w + mg));
            let eg_val_b = clamp_i16(-(eg_w + eg));

            tables[pt + 7].0[sq] = [mg_val_b, eg_val_b, ph_w as i16, 0, 0, 0, 0, 0];
            sq += 1;
        }
        pt += 1;
    }
    tables
}

const fn clamp_i16(v: i32) -> i16 { v.clamp(i16::MIN as i32, i16::MAX as i32) as i16 }

const fn raw_psqt(pt: usize, sq64: usize) -> (i32, i32) {
    let idx = mirror_sq(sq64);
    match pt {
        0 => (MG_PAWN[idx], EG_PAWN[idx]),
        1 => (MG_KNIGHT[idx], EG_KNIGHT[idx]),
        2 => (MG_BISHOP[idx], EG_BISHOP[idx]),
        3 => (MG_ROOK[idx], EG_ROOK[idx]),
        4 => (MG_QUEEN[idx], EG_QUEEN[idx]),
        5 => (MG_KING[idx], EG_KING[idx]),
        _ => (0, 0),
    }
}
