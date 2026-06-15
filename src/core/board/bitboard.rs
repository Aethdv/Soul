//! Bitboard attack generation: the computational heart of the move generator.
//!
//! Leapers (pawn, knight, king) use direct lookup tables,
//! fully computed at compile time via `const fn`.
//!
//! Sliders (bishop, rook, queen) index into an attack table
//! built by `build.rs`. The indexing scheme adapts to hardware:
//!
//! | Target        | Method                | Cost      |
//! |---------------|-----------------------|-----------|
//! | BMI2          | `PEXT` instruction    | 1 cycle   |
//! | Fallback      | magic multiplication  | 3 cycles  |
//!
//! Both paths share a single `ATTACK_TABLE`: only the hash function differs.

use crate::core::defs::{Bitboard, Color, FILE_A, FILE_H, FILE_MASKS, NOT_AB, NOT_GH, RANK_MASKS, Square};

/// Pawn captures, indexed by `[color][square]`.
pub static PAWN_ATTACKS: [[Bitboard; 64]; 2] = init_pawn_attacks();
/// Knight jumps, indexed by `[square]`. Occupancy-independent.
pub static KNIGHT_ATTACKS: [Bitboard; 64] = init_knight_attacks();
/// King steps, indexed by `[square]`. Occupancy-independent.
pub static KING_ATTACKS: [Bitboard; 64] = init_king_attacks();
/// Rook rays on an empty board (full rank ∪ file, minus self).
pub static PSEUDO_ROOK_ATTACKS: [Bitboard; 64] = init_pseudo_rook_attacks();
/// Bishop rays on an empty board (all four diagonals through the square).
pub static PSEUDO_BISHOP_ATTACKS: [Bitboard; 64] = init_pseudo_bishop_attacks();
/// Squares ahead of a pawn on its own and adjacent files, indexed `[color][square]`.
/// A pawn is passed when no enemy pawn occupies this mask.
pub static PASSED_PAWN_MASKS: [[Bitboard; 64]; 2] = init_passed_pawn_masks();

// Build-time generated tables: ROOKS, BISHOPS, ATTACK_TABLE, LINES, BETWEEN.
include!(concat!(env!("OUT_DIR"), "/magics.rs"));

/// Per-square lookup key for slider attacks.
///
/// On BMI2 hardware, `PEXT` extracts exactly the relevant occupancy bits
/// in a single cycle, so only `mask` and `offset` are needed.
/// Older CPUs additionally require a magic multiplier and shift.
#[cfg(target_feature = "bmi2")]
#[derive(Clone, Copy, Debug)]
pub struct MagicEntry {
    pub mask: u64,
    pub offset: u32,
}

#[cfg(not(target_feature = "bmi2"))]
#[derive(Clone, Copy, Debug)]
pub struct MagicEntry {
    pub mask: u64,
    pub magic: u64,
    pub shift: u8,
    pub offset: u32,
}

/// Resolve the `ATTACK_TABLE` index for a sliding piece on square `sq`
/// with board occupancy `occ`.
///
/// Compiles to a single `PEXT` on BMI2, a multiply-shift hash otherwise.
macro_rules! magic_index {
    ($entries:expr, $sq:expr, $occ:expr) => {{
        let entry = debug_index!($entries, $sq);

        #[cfg(target_feature = "bmi2")]
        // SAFETY: _pext_u64 is a pure arithmetic intrinsic. Hardware support
        // is verified via the target_feature gate.
        let key = unsafe { core::arch::x86_64::_pext_u64($occ, entry.mask) } as usize;

        #[cfg(not(target_feature = "bmi2"))]
        let key = (($occ & entry.mask).wrapping_mul(entry.magic) >> entry.shift) as usize;

        key + entry.offset as usize
    }};
}

/// Pawn capture mask for a single pawn of the given color.
#[inline(always)]
pub fn atk_pawn(sq: Square, color: Color) -> Bitboard {
    PAWN_ATTACKS[color][sq]
}

/// Forward span a pawn must clear of enemy pawns to count as passed;
/// its own file and both neighbors, every rank ahead toward promotion.
#[inline(always)]
pub fn passed_span(sq: Square, color: Color) -> Bitboard {
    PASSED_PAWN_MASKS[color][sq]
}

/// Knight attack mask: the eight possible L-shaped destinations.
#[inline(always)]
pub fn atk_knight(square: Square) -> Bitboard {
    KNIGHT_ATTACKS[square]
}

/// Bishop attacks along diagonals, blocked by `occupancy`.
#[inline(always)]
pub fn atk_bishop(square: Square, occupancy: Bitboard) -> Bitboard {
    let idx = magic_index!(BISHOPS, usize::from(square), occupancy.0);
    Bitboard(ATTACK_TABLE[idx])
}

/// Rook attacks along ranks and files, blocked by `occupancy`.
#[inline(always)]
pub fn atk_rook(square: Square, occupancy: Bitboard) -> Bitboard {
    let idx = magic_index!(ROOKS, usize::from(square), occupancy.0);
    Bitboard(ATTACK_TABLE[idx])
}

/// King attack mask: one step in each compass direction.
#[inline(always)]
pub fn atk_king(square: Square) -> Bitboard {
    KING_ATTACKS[square]
}

/// The full line through both squares (rank, file, or diagonal).
/// Returns empty if the squares aren't collinear. Used for pin detection.
#[inline(always)]
pub fn line_bb(sq1: Square, sq2: Square) -> Bitboard {
    Bitboard(LINES[sq1][sq2])
}

/// Squares strictly between `sq1` and `sq2` on their shared line.
/// Endpoints excluded. Empty if not collinear. Used for check evasions.
#[inline(always)]
pub fn between_bb(sq1: Square, sq2: Square) -> Bitboard {
    Bitboard(BETWEEN[sq1][sq2])
}

/// Pawn capture masks for both colors.
///
/// ```text
///   White pawn on d4:          Black pawn on d5:
///     . . c . e . . .            . . . . . . . .
///     . . . P . . . .            . . . p . . . .
///     . . . . . . . .            . . c . e . . .
/// ```
const fn init_pawn_attacks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard(0); 64]; 2];
    let mut sq = 0;

    while sq < 64 {
        let bb = 1u64 << sq;
        // White captures northeast & northwest
        table[0][sq as usize] = Bitboard(((bb << 9) & !FILE_A.0) | ((bb << 7) & !FILE_H.0));
        // Black captures southeast & southwest
        table[1][sq as usize] = Bitboard(((bb >> 9) & !FILE_H.0) | ((bb >> 7) & !FILE_A.0));
        sq += 1;
    }
    table
}

/// Passed-pawn spans: own file ∪ adjacent files, every rank ahead of the pawn.
/// White looks toward rank 8, black toward rank 1.
const fn init_passed_pawn_masks() -> [[Bitboard; 64]; 2] {
    let mut table = [[Bitboard(0); 64]; 2];
    let mut sq = 0usize;

    while sq < 64 {
        let file = sq % 8;
        let rank = sq / 8;

        // The pawn's own file plus both neighbors.
        let mut files = FILE_MASKS[file].0;

        if file > 0 {
            files |= FILE_MASKS[file - 1].0;
        }

        if file < 7 {
            files |= FILE_MASKS[file + 1].0;
        }

        let mut above = 0u64;
        let mut r = rank + 1;

        while r < 8 {
            above |= RANK_MASKS[r].0;
            r += 1;
        }

        let mut below = 0u64;
        let mut r = 0;

        while r < rank {
            below |= RANK_MASKS[r].0;
            r += 1;
        }

        table[0][sq] = Bitboard(files & above);
        table[1][sq] = Bitboard(files & below);
        sq += 1;
    }
    table
}

/// Knight attack masks.
/// The eight L-shaped destinations.
///
/// ```text
///   . x . x .
///   x . . . x
///   . . N . .
///   x . . . x
///   . x . x .
/// ```
const fn init_knight_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0;

    while sq < 64 {
        let bb = 1u64 << sq;
        table[sq as usize] = Bitboard(
            ((bb << 17) & !FILE_A.0)  //  ↑↑→
          | ((bb << 15) & !FILE_H.0)  //  ↑↑←
          | ((bb << 10) &  NOT_AB.0)  //  ↑→→
          | ((bb <<  6) &  NOT_GH.0)  //  ↑←←
          | ((bb >>  6) &  NOT_AB.0)  //  ↓→→
          | ((bb >> 10) &  NOT_GH.0)  //  ↓←←
          | ((bb >> 15) & !FILE_A.0)  //  ↓↓→
          | ((bb >> 17) & !FILE_H.0), //  ↓↓←
        );
        sq += 1;
    }
    table
}

/// King attack masks.
/// One step in each compass direction.
///
/// ```text
///   x x x
///   x K x
///   x x x
/// ```
const fn init_king_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0;

    while sq < 64 {
        let bb = 1u64 << sq;
        table[sq as usize] = Bitboard(
            (bb << 8)                // N
          | (bb >> 8)                // S
          | ((bb << 1) & !FILE_A.0)  // E
          | ((bb >> 1) & !FILE_H.0)  // W
          | ((bb << 9) & !FILE_A.0)  // NE
          | ((bb << 7) & !FILE_H.0)  // NW
          | ((bb >> 7) & !FILE_A.0)  // SE
          | ((bb >> 9) & !FILE_H.0), // SW
        );
        sq += 1;
    }
    table
}

/// Rook pseudo-attacks.
/// Full rank ∪ full file through the square, minus self.
/// Useful for quick "could a rook ever reach that square?" filtering.
const fn init_pseudo_rook_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0;

    while sq < 64 {
        let rank = sq / 8;
        let file = sq % 8;

        let rank_mask = RANK_MASKS[rank as usize];
        let file_mask = FILE_MASKS[file as usize];
        let sq_bit = 1u64 << sq;

        table[sq as usize] = (rank_mask | file_mask) & !sq_bit;
        sq += 1;
    }
    table
}

/// Bishop pseudo-attacks.
/// All four diagonal rays through the square.
/// Useful for quick "could a bishop ever reach that square?" filtering.
const fn init_pseudo_bishop_attacks() -> [Bitboard; 64] {
    let mut table = [Bitboard(0); 64];
    let mut sq = 0;

    while sq < 64 {
        let rank = (sq / 8) as i8;
        let file = (sq % 8) as i8;
        let mut attacks = 0u64;

        // Walk each diagonal until we fall off the board.
        let dirs: [(i8, i8); 4] = [(1, 1), (1, -1), (-1, 1), (-1, -1)];
        let mut d = 0;

        while d < 4 {
            let (dr, df) = dirs[d];
            let (mut r, mut f) = (rank + dr, file + df);

            while r >= 0 && r < 8 && f >= 0 && f < 8 {
                attacks |= 1u64 << (r * 8 + f);
                r += dr;
                f += df;
            }
            d += 1;
        }
        table[sq as usize] = Bitboard(attacks);
        sq += 1;
    }
    table
}
