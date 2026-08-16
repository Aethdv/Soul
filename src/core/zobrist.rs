//! Zobrist hashing: incremental 64-bit position signatures, invented by
//! Albert Zobrist in 1970.
//!
//! Every feature a position can carry, a white pawn on e4, an en passant target
//! on e6, the castling rights as a whole, gets its own random 64-bit key, drawn
//! once at compile time. A position's hash is the XOR of the keys for every
//! feature it currently has.
//!
//! XOR being its own inverse is the whole trick: one key toggles a feature on or
//! off with the identical operation. Moving a piece XORs out the key on its
//! origin square and XORs in the key on its destination, so the hash tracks the
//! move in 𝒪(1) without rescanning the board.
//!
//! These signatures key the transposition table and drive repetition detection.

use crate::core::defs::{Color, PieceType, Square};

pub const PIECE_KEYS_LEN: usize = 64 * 14; // 64 squares · 7 types (including None) · 2 colors
pub const EP_KEYS_LEN: usize = 8; // files
pub const CASTLING_KEYS_LEN: usize = 16; // 2⁴

/// A compile-time pseudo-random number generator (PRNG).
///
/// The Zobrist tables need hundreds of 64-bit values, and a `const fn` generator puts them
/// in the binary instead of on the startup path.
pub struct ConstRng {
    state: u64,
}

impl ConstRng {
    pub const fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// Marsaglia's xorshift.
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
    // Seed: a nod to Tord Romstad's birthday (7 Oct 1972).
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
    // Empty rights must hash to nothing. make_move XORs key_castling on every rights
    // change, so a non-zero key at index 0 desyncs the incremental hash from
    // calc_zobrist (which skips zero rights) the moment castling rights run out.
    castling[0] = 0;

    let side = rng.next();
    ZobristKeys { pieces, en_passant, castling, side }
}
