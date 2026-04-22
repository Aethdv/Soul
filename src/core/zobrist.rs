//! Zobrist hashing for incremental position signatures.
//!
//! Generates and stores the 64-bit pseudo-random keys used to uniquely identify
//! board states for transposition tables and repetition detection.

use crate::core::defs::{Color, PieceType, Square};

// ──────── Zobrist Hashing ────────
//
// A high-speed, incremental hashing technique invented by Albert Zobrist in 1970.
// We generate a unique 64-bit random number for every possible feature of a chess board
// (e.g., "White Pawn on e4", "Black can castle Kingside", "e6 is an en passant target").
//
// The hash of a full position is simply the XOR sum of all its features.
// To move a piece, we XOR out the feature at its origin, and XOR in the feature
// at its destination. This updates the hash in 𝒪(1) time without re-scanning the board.

pub const PIECE_KEYS_LEN: usize = 64 * 14; // 64 squares · 7 types (including None) · 2 colors
pub const EP_KEYS_LEN: usize = 8; // files
pub const CASTLING_KEYS_LEN: usize = 16; // 2⁴

/// A compile-time pseudo-random number generator (PRNG).
///
/// We need to generate hundreds of 64-bit random numbers to populate the Zobrist tables,
/// but we want to do it at compile-time to avoid any runtime overhead during engine startup.
pub struct ConstRng {
    state: u64,
}

impl ConstRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Marsaglia's xorshift.
    /// The shifts (13, 7, 17) are optimal constants that create a full-period generator.
    #[inline]
    pub const fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

pub struct ZobristKeys {
    pub pieces: [u64; PIECE_KEYS_LEN],
    pub en_passant: [u64; EP_KEYS_LEN],
    pub castling: [u64; CASTLING_KEYS_LEN],
    pub side: u64,
}

pub static KEYS: ZobristKeys = init_keys();

/// Hash feature for a piece on a square.
#[inline(always)]
pub fn key_piece(pt: PieceType, color: Color, sq: Square) -> u64 {
    // [PieceType + ColorOffset][Square]
    //  White=0, Black=7 (to allow None=6 padding)
    let pt_idx = usize::from(pt) + (usize::from(color) * 7);
    let idx = (pt_idx * 64) + usize::from(sq);

    debug_assert!(usize::from(pt) <= 6 && sq < 64, "key_piece: invalid inputs");
    *debug_index!(KEYS.pieces, idx)
}

/// Hash feature for en passant availability on a given file.
#[inline(always)]
pub fn key_ep(sq: Square) -> u64 {
    let idx = (sq.0 & 7) as usize;
    *debug_index!(KEYS.en_passant, idx)
}

/// Hash feature for the current castling rights bitmask.
#[inline(always)]
pub fn key_castling(rights: u8) -> u64 {
    debug_assert!(rights < 16, "key_castling: rights overflow");
    *debug_index!(KEYS.castling, usize::from(rights))
}

/// Hash feature toggled when it is Black's turn to move.
#[inline(always)]
pub fn key_side() -> u64 {
    KEYS.side
}

#[inline]
const fn init_keys() -> ZobristKeys {
    // Seed = Tord Romstad's birthday (October 7, 1972: 10071972 → 1070372)
    let mut rng = ConstRng::new(1_070_372);

    let mut pieces = [0; PIECE_KEYS_LEN];
    let mut pt = 0;
    while pt < 6 {
        let mut sq = 0;
        while sq < 64 {
            pieces[(pt + Color::White as usize * 7) * 64 + sq] = rng.next();
            pieces[(pt + Color::Black as usize * 7) * 64 + sq] = rng.next();
            sq += 1;
        }
        pt += 1;
    }

    let mut en_passant = [0; EP_KEYS_LEN];
    let mut j = 0;
    while j < EP_KEYS_LEN {
        en_passant[j] = rng.next();
        j += 1;
    }

    let mut castling = [0; CASTLING_KEYS_LEN];
    let mut k = 0;
    while k < CASTLING_KEYS_LEN {
        castling[k] = rng.next();
        k += 1;
    }

    let side = rng.next();

    ZobristKeys { pieces, en_passant, castling, side }
}
