//! Incremental Zobrist hashing for pieces, en passant, castling rights and side to move.

use crate::core::defs::{Color, PieceType, Square};

/// Marsaglia xorshift64 RNG usable in const evaluation.
pub(crate) struct ConstRng {
    state: u64,
}

impl ConstRng {
    pub(crate) const fn new(seed: u64) -> Self { Self { state: seed } }

    #[inline]
    pub(crate) const fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

struct ZobristKeys {
    pieces: [u64; 64 * 7 * 2],
    en_passant: [u64; 8],
    castling: [u64; 1 << 4],
    side: u64,
}

static KEYS: ZobristKeys = init_keys();

#[inline(always)]
pub(super) fn key_piece(pt: PieceType, color: Color, sq: Square) -> u64 {
    // The seventh slot keeps PieceType::None mapped to a zero key.
    let pt_idx = usize::from(pt) + usize::from(color) * 7;
    KEYS.pieces[pt_idx * 64 + usize::from(sq)]
}

/// Keyed by file.
#[inline(always)]
pub(super) fn key_ep(sq: Square) -> u64 {
    let idx = (sq.0 & 7) as usize;
    *debug_index!(KEYS.en_passant, idx)
}

#[inline(always)]
pub(super) fn key_castling(rights: u8) -> u64 { KEYS.castling[usize::from(rights)] }

/// Present when Black is to move.
#[inline(always)]
pub(crate) fn key_side() -> u64 { KEYS.side }

const fn init_keys() -> ZobristKeys {
    // Seed: a nod to Tord Romstad's birthday (7 Oct 1972).
    let mut rng = ConstRng::new(1_070_372);
    let mut pieces = [0; 64 * 7 * 2];
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

    let mut en_passant = [0; 8];
    let mut j = 0;

    while j < en_passant.len() {
        en_passant[j] = rng.next();
        j += 1;
    }

    let mut castling = [0; 1 << 4];
    let mut k = 0;

    while k < castling.len() {
        castling[k] = rng.next();
        k += 1;
    }

    // calc_zobrist skips zero rights, so the incremental hash must use key 0 too.
    castling[0] = 0;

    let side = rng.next();
    ZobristKeys { pieces, en_passant, castling, side }
}
