//! Bitboard attack generation and precomputed ray lookups.

use crate::core::defs::{Bitboard, Color, Direction, FILE_A, FILE_H, FILE_MASKS, NOT_AB, NOT_GH, RANK_MASKS, Square};

static PAWN_ATTACKS: [[Bitboard; 64]; 2] = init_pawn_atks();
static PASSED_PAWN_MASKS: [[Bitboard; 64]; 2] = init_passed_pawn_masks();
static KNIGHT_ATTACKS: [Bitboard; 64] = init_knight_atks();
static KING_ATTACKS: [Bitboard; 64] = init_king_atks();

pub(super) static ROOK_RAYS: [Bitboard; 64] = init_rook_rays();
pub(super) static BISHOP_RAYS: [Bitboard; 64] = init_bishop_rays();

#[cfg(use_pext)]
struct MagicEntry {
    mask: u64,
    offset: u32,
}

#[cfg(not(use_pext))]
struct MagicEntry {
    mask: u64,
    magic: u64,
    shift: u8,
    offset: u32,
}

include!(concat!(env!("OUT_DIR"), "/magics.rs"));

#[inline(always)]
pub(crate) fn atk_pawn(sq: Square, color: Color) -> Bitboard { PAWN_ATTACKS[color][sq] }

#[inline(always)]
pub(crate) fn is_passed_pawn(sq: Square, color: Color, enemy_pawns: Bitboard) -> bool {
    (PASSED_PAWN_MASKS[color][sq] & enemy_pawns).is_empty()
}

#[inline(always)]
pub(crate) fn atk_knight(sq: Square) -> Bitboard { KNIGHT_ATTACKS[sq] }

#[inline(always)]
pub(crate) fn atk_bishop(sq: Square, occ: Bitboard) -> Bitboard { Bitboard(ATTACK_TABLE[magic_index(&BISHOPS, sq, occ.0)]) }

#[inline(always)]
pub(crate) fn atk_rook(sq: Square, occ: Bitboard) -> Bitboard { Bitboard(ATTACK_TABLE[magic_index(&ROOKS, sq, occ.0)]) }

#[inline(always)]
pub(crate) fn atk_queen(sq: Square, occ: Bitboard) -> Bitboard { atk_rook(sq, occ) | atk_bishop(sq, occ) }

#[inline(always)]
pub(crate) fn atk_king(sq: Square) -> Bitboard { KING_ATTACKS[sq] }

/// The whole line through both squares, rank, file or diagonal.
/// Empty when they do not share one.
#[inline(always)]
pub(crate) fn line_bb(sq1: Square, sq2: Square) -> Bitboard { Bitboard(LINES[sq1][sq2]) }

/// The squares strictly between the two, endpoints excluded.
/// Empty when they do not share a line.
#[inline(always)]
pub(crate) fn between_bb(sq1: Square, sq2: Square) -> Bitboard { Bitboard(BETWEEN[sq1][sq2]) }

#[inline(always)]
fn magic_index(entries: &[MagicEntry; 64], sq: Square, occ: u64) -> usize {
    let entry = &entries[usize::from(sq)];

    #[cfg(use_pext)]
    // SAFETY: use_pext is enabled only when BMI2 is a target feature.
    let key = unsafe { core::arch::x86_64::_pext_u64(occ, entry.mask) } as usize;

    #[cfg(not(use_pext))]
    let key = ((occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;

    key + entry.offset as usize
}

/// ```text
///   . x . x . . . .
///   . . P . . p . .
///   . . . . x . x .
/// ```
const fn init_pawn_atks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard(0); 64]; 2];
    let mut sq = 0usize;

    while sq < 64 {
        let bb = Bitboard(1u64 << sq);
        table[0][sq] = Bitboard(bb.shift(Direction::NorthEast).0 | bb.shift(Direction::NorthWest).0);
        table[1][sq] = Bitboard(bb.shift(Direction::SouthEast).0 | bb.shift(Direction::SouthWest).0);
        sq += 1;
    }

    table
}

const fn init_passed_pawn_masks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard(0); 64]; 2];
    let mut sq = 0usize;

    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;

        let mut files = FILE_MASKS[file].0;
        if file > 0 {
            files |= FILE_MASKS[file - 1].0;
        }
        if file < 7 {
            files |= FILE_MASKS[file + 1].0;
        }

        // On the eighth rank White shifts by all 64 bits, where unbounded_shl gives 0.
        let above = (!0u64).unbounded_shl(8 * (rank as u32 + 1));
        let below = (1u64 << (8 * rank)) - 1;

        table[0][sq] = Bitboard(files & above);
        table[1][sq] = Bitboard(files & below);
        sq += 1;
    }

    table
}

/// ```text
///   . x . x .
///   x . . . x
///   . . N . .
///   x . . . x
///   . x . x .
/// ```
const fn init_knight_atks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;

    while sq < 64 {
        let bb = 1u64 << sq;

        table[sq] = Bitboard(
            ((bb << 17) & !FILE_A.0)  // ↑↑→
          | ((bb << 15) & !FILE_H.0)  // ↑↑←
          | ((bb << 10) &  NOT_AB.0)  // ↑→→
          | ((bb <<  6) &  NOT_GH.0)  // ↑←←
          | ((bb >>  6) &  NOT_AB.0)  // ↓→→
          | ((bb >> 10) &  NOT_GH.0)  // ↓←←
          | ((bb >> 15) & !FILE_A.0)  // ↓↓→
          | ((bb >> 17) & !FILE_H.0), // ↓↓←
        );
        sq += 1;
    }

    table
}

/// ```text
///   . . . . .
///   . x x x .
///   . x K x .
///   . x x x .
///   . . . . .
/// ```
const fn init_king_atks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;

    while sq < 64 {
        let bb = Bitboard(1u64 << sq);

        table[sq] = Bitboard(
            bb.shift(Direction::North).0
                | bb.shift(Direction::South).0
                | bb.shift(Direction::East).0
                | bb.shift(Direction::West).0
                | bb.shift(Direction::NorthEast).0
                | bb.shift(Direction::NorthWest).0
                | bb.shift(Direction::SouthEast).0
                | bb.shift(Direction::SouthWest).0,
        );
        sq += 1;
    }

    table
}

/// ```text
///   . . x . .
///   . . x . .
///   x x R x x
///   . . x . .
///   . . x . .
/// ```
const fn init_rook_rays() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;

    while sq < 64 {
        let rank = sq / 8;
        let file = sq % 8;
        let rank_mask = RANK_MASKS[rank];
        let file_mask = FILE_MASKS[file];
        let sq_bit = 1u64 << sq;

        table[sq] = (rank_mask | file_mask) & !sq_bit;
        sq += 1;
    }

    table
}

/// ```text
///   x . . . x
///   . x . x .
///   . . B . .
///   . x . x .
///   x . . . x
/// ```
const fn init_bishop_rays() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0usize;

    while sq < 64 {
        let rank = (sq / 8) as i8;
        let file = (sq % 8) as i8;

        table[sq] = Bitboard(
            diag_ray(rank, file, 1, 1)    // NE
          | diag_ray(rank, file, 1, -1)   // NW
          | diag_ray(rank, file, -1, 1)   // SE
          | diag_ray(rank, file, -1, -1), // SW
        );
        sq += 1;
    }

    table
}

/// From `(rank, file)` to the edge, stepping `(dr, df)`, the origin excluded.
const fn diag_ray(rank: i8, file: i8, dr: i8, df: i8) -> u64 {
    let (mut r, mut f) = (rank + dr, file + df);
    let mut ray = 0u64;

    while r >= 0 && r < 8 && f >= 0 && f < 8 {
        ray |= 1u64 << (r * 8 + f);
        r += dr;
        f += df;
    }

    ray
}
