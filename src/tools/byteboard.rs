//! Rose's ByteBoard, the square-indexed store the rig measures XorBoard against.
//!
//! A byte per square packs type, color and slot, and `attack[color][square]`
//! is the sixteen-bit set of that side's slots hitting the square. Updates run
//! setwise over 64-byte vectors against the ray geometry in [`Geometry`].

use core::arch::x86_64::*;

use crate::core::defs::{Color, PieceType, Square};

const PT_KING: u8 = 0b001;
const PT_PAWN: u8 = 0b010;
const PT_KNIGHT: u8 = 0b011;
const PT_BISHOP: u8 = 0b101;
const PT_ROOK: u8 = 0b110;
const PT_QUEEN: u8 = 0b111;

const PTYPE_SHIFT: u32 = 4;
const SLIDER: u8 = 0b100 << PTYPE_SHIFT;
const DIAG: u8 = 0b001 << PTYPE_SHIFT;
const ORTH: u8 = 0b010 << PTYPE_SHIFT;

const DIRS: [(i32, i32); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHTS: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

const KNIGHT_SLOTS: u64 = 0x0101_0101_0101_0101;
const RAY_SLOTS: u64 = !KNIGHT_SLOTS;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Place(u8);

impl Place {
    pub const EMPTY: Self = Self(0);

    const BLACK: u8 = 0x80;
    const PTYPE: u8 = 0b111 << PTYPE_SHIFT;

    pub fn new(color: Color, pt: PieceType, id: PieceId) -> Self {
        let code = match pt {
            PieceType::King => PT_KING,
            PieceType::Pawn => PT_PAWN,
            PieceType::Knight => PT_KNIGHT,
            PieceType::Bishop => PT_BISHOP,
            PieceType::Rook => PT_ROOK,
            PieceType::Queen => PT_QUEEN,
            PieceType::None => return Self::EMPTY,
        };

        Self((color as u8) << 7 | code << PTYPE_SHIFT | id.0)
    }

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn color(self) -> Color {
        if self.0 & Self::BLACK == 0 { Color::White } else { Color::Black }
    }

    /// The piece code, which is also the row `Geometry::reach` is indexed by.
    fn code(self) -> usize {
        usize::from(self.0 & Self::PTYPE) >> PTYPE_SHIFT
    }

    pub fn id(self) -> PieceId {
        PieceId(self.0 & 0xF)
    }
}

/// A piece's slot within its side, 0 to 15. Two pieces of one color never share
/// one, which is what lets an attack entry be sixteen bits.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PieceId(u8);

impl PieceId {
    pub fn new(slot: u8) -> Self {
        debug_assert!(slot < 16, "a side has sixteen slots");
        Self(slot)
    }

    /// The slot index, for a caller indexing a store laid out the other way.
    pub fn raw(self) -> u8 {
        self.0
    }

    pub fn mask(self) -> PieceMask {
        PieceMask(1 << self.0)
    }
}

/// The slots of one color that reach a square, one bit each.
#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub struct PieceMask(u16);

impl PieceMask {
    pub const EMPTY: Self = Self(0);

    pub fn is_empty(self) -> bool {
        self.0 == 0
    }

    pub fn count(self) -> u32 {
        self.0.count_ones()
    }

    /// The raw bits, for a caller comparing against another representation.
    pub fn bits(self) -> u16 {
        self.0
    }

    pub fn contains(self, id: PieceId) -> bool {
        self.0 & id.mask().0 != 0
    }

    pub fn insert(&mut self, id: PieceId) {
        self.0 |= id.mask().0;
    }
}

impl Iterator for PieceMask {
    type Item = PieceId;

    fn next(&mut self) -> Option<PieceId> {
        (!self.is_empty()).then(|| {
            let id = PieceId(self.0.trailing_zeros() as u8);
            self.0 &= self.0 - 1;
            id
        })
    }
}

/// One color's attack table: for every square, the slots of that color hitting it.
/// The XorBoard holds the same relation transposed, one row per piece.
#[derive(Clone, Copy)]
pub struct Wordboard([u16; 64]);

impl Wordboard {
    pub const EMPTY: Self = Self([0; 64]);

    /// Record that `id` reaches `sq`, for a table built from scratch.
    pub fn insert(&mut self, sq: Square, id: PieceId) {
        self.0[usize::from(sq)] |= id.mask().0;
    }

    pub fn read(&self, sq: Square) -> PieceMask {
        PieceMask(self.0[usize::from(sq)])
    }

    /// The squares this color attacks at all.
    pub fn any(&self) -> u64 {
        self.probe(u16::MAX)
    }

    /// The squares any slot in `mask` attacks.
    pub fn with(&self, mask: PieceMask) -> u64 {
        self.probe(mask.0)
    }

    /// Squares whose entry shares a bit with `mask`, as a bitboard. Sixty-four
    /// words are four vectors; each pair packs down to bytes so one movemask
    /// answers thirty-two squares.
    fn probe(&self, mask: u16) -> u64 {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let m = _mm256_set1_epi16(mask.cast_signed());
            let zero = _mm256_setzero_si256();
            let mut out = 0u64;

            for half in 0..2 {
                let load = |at: usize| _mm256_and_si256(_mm256_loadu_si256(self.0.as_ptr().add(at).cast()), m);
                let lo = _mm256_cmpeq_epi16(load(half * 32), zero);
                let hi = _mm256_cmpeq_epi16(load(half * 32 + 16), zero);

                // packs interleaves the 128-bit halves, so the permute puts the
                // squares back in order before the mask is read off.
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(lo, hi));
                out |= u64::from(!_mm256_movemask_epi8(packed).cast_unsigned()) << (half * 32);
            }
            out
        }
    }
}

pub struct Geometry {
    rays: [[u8; 64]; 64],
    inverse: [[u8; 64]; 64],
    valid: [u64; 64],
    reach: [[u64; 2]; 8],
    lane_kind: [u8; 64],
}

impl Default for Geometry {
    fn default() -> Self {
        Self::new()
    }
}

impl Geometry {
    pub fn new() -> Self {
        let mut g = Self {
            rays: [[0xFF; 64]; 64],
            inverse: [[0xFF; 64]; 64],
            valid: [0; 64],
            reach: [[0; 2]; 8],
            lane_kind: [0; 64],
        };

        for sq in 0..64usize {
            let (f, r) = (sq as i32 % 8, sq as i32 / 8);

            for (lane, &(df, dr)) in DIRS.iter().enumerate() {
                let slot = lane * 8;
                let (kf, kr) = KNIGHTS[lane];

                if (0..8).contains(&(f + kf)) && (0..8).contains(&(r + kr)) {
                    let target = ((r + kr) * 8 + f + kf) as usize;
                    g.rays[sq][slot] = target as u8;
                    g.inverse[sq][target] = slot as u8;
                    g.valid[sq] |= 1 << slot;
                }

                for d in 1..8usize {
                    let (tf, tr) = (f + df * d as i32, r + dr * d as i32);

                    if !(0..8).contains(&tf) || !(0..8).contains(&tr) {
                        break;
                    }

                    let target = (tr * 8 + tf) as usize;
                    g.rays[sq][slot + d] = target as u8;
                    g.inverse[sq][target] = (slot + d) as u8;
                    g.valid[sq] |= 1 << (slot + d);
                }
            }
        }

        for lane in 0..8 {
            for d in 1..8 {
                g.lane_kind[lane * 8 + d] = if lane % 2 == 0 { ORTH } else { DIAG };
            }
        }

        let orth = 0x00FF_00FF_00FF_00FFu64 & RAY_SLOTS;
        let diag = 0xFF00_FF00_FF00_FF00u64 & RAY_SLOTS;
        let near = 0x0202_0202_0202_0202u64;

        g.reach[PT_KNIGHT as usize] = [KNIGHT_SLOTS; 2];
        g.reach[PT_KING as usize] = [near; 2];
        g.reach[PT_BISHOP as usize] = [diag; 2];
        g.reach[PT_ROOK as usize] = [orth; 2];
        g.reach[PT_QUEEN as usize] = [orth | diag; 2];
        g.reach[PT_PAWN as usize] = [0x0200_0000_0000_0200, 0x0000_0200_0200_0000];
        g
    }

    #[inline(always)]
    fn fill(&self, sq: usize, occupied: u64) -> u64 {
        let o = occupied | 0x8181_8181_8181_8181;
        (o ^ o.wrapping_sub(0x0303_0303_0303_0303)) & self.valid[sq]
    }
}

#[derive(Clone, Copy)]
pub struct ByteBoard {
    mailbox: V64,
    attack: [Wordboard; 2],
}

impl ByteBoard {
    pub fn empty() -> Self {
        Self { mailbox: V64::zero(), attack: [Wordboard([0; 64]); 2] }
    }

    pub fn at(&self, sq: Square) -> Place {
        Place(self.mailbox.bytes()[usize::from(sq)])
    }

    pub fn attacks(&self, color: Color) -> &Wordboard {
        &self.attack[usize::from(color)]
    }

    /// Install a table computed from scratch, which is how the rig seeds its oracle.
    pub fn set_attacks(&mut self, attacks: [Wordboard; 2]) {
        self.attack = attacks;
    }

    pub fn occupied(&self) -> u64 {
        self.mailbox.nonzero()
    }

    pub fn pieces(&self, color: Color) -> u64 {
        let black = self.mailbox.signs();
        self.occupied() & if color == Color::Black { black } else { !black }
    }

    /// Squares holding a `pt` of `color`, the color and type nibble compared as one.
    pub fn pieces_of(&self, color: Color, pt: PieceType) -> u64 {
        self.mailbox.match_high(Place::new(color, pt, PieceId(0)).0)
    }

    /// Drop a place onto a square, leaving the attack table alone.
    #[inline(always)]
    pub fn put(&mut self, sq: Square, p: Place) {
        self.mailbox = self.mailbox.write(1u64 << usize::from(sq), p.0);
    }

    /// Lift the piece on `sq`: reopen the lines it blocked, drop its own attacks,
    /// then bare the square.
    #[inline(always)]
    pub fn remove(&mut self, g: &Geometry, sq: Square) {
        let p = self.at(sq);
        self.toggle(g, sq);
        self.clear(p.color(), p.id());
        self.put(sq, Place::EMPTY);
    }

    /// Land a piece on an empty square: close the lines through it, then add its own.
    #[inline(always)]
    pub fn add(&mut self, g: &Geometry, sq: Square, p: Place) {
        self.toggle(g, sq);
        self.land(g, sq, p);
    }

    /// Land a piece on a square another already blocked, so only its own attacks change.
    #[inline(always)]
    pub fn land(&mut self, g: &Geometry, sq: Square, p: Place) {
        self.put(sq, p);
        let s = usize::from(sq);
        let places = self.mailbox.permute(&g.rays[s]);
        let raymask = g.fill(s, places.nonzero());
        let reach = raymask & g.reach[p.code()][usize::from(p.color())];
        self.apply(V64::splat(p.0).keep(reach).permute(&g.inverse[s]));
    }

    #[inline(always)]
    fn toggle(&mut self, g: &Geometry, sq: Square) {
        let s = usize::from(sq);
        let places = self.mailbox.permute(&g.rays[s]);
        let raymask = g.fill(s, places.nonzero());
        let visible = raymask & places.sliders(g);
        if visible == 0 {
            return;
        }

        let rotated = places.keep(visible).lane_spread().rotate();
        self.apply(rotated.keep(raymask & RAY_SLOTS).permute(&g.inverse[s]));
    }

    /// Strike one slot out of its color's whole attack table.
    #[inline(always)]
    pub fn clear(&mut self, color: Color, id: PieceId) {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let m = _mm256_set1_epi16((!id.mask().0).cast_signed());
            let p = self.attack[usize::from(color)].0.as_mut_ptr();
            for q in 0..4 {
                let v = _mm256_loadu_si256(p.add(q * 16).cast());
                _mm256_storeu_si256(p.add(q * 16).cast(), _mm256_and_si256(v, m));
            }
        }
    }

    #[inline(always)]
    fn apply(&mut self, board: V64) {
        let (white, black) = board.to_wordboards();

        // SAFETY: AVX2 per the gate.
        unsafe {
            for (c, src) in [white, black].into_iter().enumerate() {
                let p = self.attack[c].0.as_mut_ptr();
                for (q, x) in src.iter().enumerate() {
                    let v = _mm256_loadu_si256(p.add(q * 16).cast());
                    _mm256_storeu_si256(p.add(q * 16).cast(), _mm256_xor_si256(v, *x));
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
pub struct V64(__m256i, __m256i);

impl V64 {
    #[inline(always)]
    pub fn zero() -> Self {
        // SAFETY: AVX2 per the gate.
        unsafe { Self(_mm256_setzero_si256(), _mm256_setzero_si256()) }
    }

    #[inline(always)]
    pub fn splat(b: u8) -> Self {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let v = _mm256_set1_epi8(b.cast_signed());
            Self(v, v)
        }
    }

    #[inline(always)]
    pub fn load(b: &[u8; 64]) -> Self {
        // SAFETY: AVX2 per the gate.
        unsafe { Self(_mm256_loadu_si256(b.as_ptr().cast()), _mm256_loadu_si256(b.as_ptr().add(32).cast())) }
    }

    #[inline(always)]
    pub fn bytes(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        // SAFETY: AVX2 per the gate.
        unsafe {
            _mm256_storeu_si256(out.as_mut_ptr().cast(), self.0);
            _mm256_storeu_si256(out.as_mut_ptr().add(32).cast(), self.1);
        }
        out
    }

    #[inline(always)]
    pub fn permute(self, idx: &[u8; 64]) -> Self {
        // SAFETY: AVX2 per the gate, AVX-512 under its own cfg.
        unsafe {
            #[cfg(all(target_feature = "avx512vbmi", target_feature = "avx512bw"))]
            {
                let src = _mm512_inserti64x4::<1>(_mm512_castsi256_si512(self.0), self.1);
                let i = _mm512_loadu_si512(idx.as_ptr().cast());
                let live = _mm512_cmplt_epu8_mask(i, _mm512_set1_epi8(64));
                let r = _mm512_maskz_permutexvar_epi8(live, i, src);
                Self(_mm512_castsi512_si256(r), _mm512_extracti64x4_epi64::<1>(r))
            }

            #[cfg(not(all(target_feature = "avx512vbmi", target_feature = "avx512bw")))]
            {
                let chunks = [
                    _mm256_permute2x128_si256::<0x00>(self.0, self.0),
                    _mm256_permute2x128_si256::<0x11>(self.0, self.0),
                    _mm256_permute2x128_si256::<0x00>(self.1, self.1),
                    _mm256_permute2x128_si256::<0x11>(self.1, self.1),
                ];

                let sign = _mm256_set1_epi8(-128);
                let mut out = [_mm256_setzero_si256(); 2];

                for (h, o) in out.iter_mut().enumerate() {
                    let want = _mm256_loadu_si256(idx.as_ptr().add(h * 32).cast());
                    let over = _mm256_slli_epi16::<1>(_mm256_and_si256(want, _mm256_set1_epi8(0x40)));
                    let low = _mm256_or_si256(_mm256_and_si256(want, _mm256_set1_epi8(15)), _mm256_and_si256(want, sign));
                    let low = _mm256_or_si256(low, _mm256_and_si256(over, sign));
                    let which = _mm256_and_si256(want, _mm256_set1_epi8(0x30));

                    for (c, chunk) in chunks.iter().enumerate() {
                        let mine = _mm256_cmpeq_epi8(which, _mm256_set1_epi8((c * 16) as i8));
                        let ctrl = _mm256_or_si256(low, _mm256_andnot_si256(mine, sign));
                        *o = _mm256_or_si256(*o, _mm256_shuffle_epi8(*chunk, ctrl));
                    }
                }
                Self(out[0], out[1])
            }
        }
    }

    #[inline(always)]
    pub fn nonzero(self) -> u64 {
        // SAFETY: AVX2 per the gate, AVX-512VL/BW under its own cfg.
        unsafe {
            #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
            {
                u64::from(_mm256_test_epi8_mask(self.0, self.0)) | u64::from(_mm256_test_epi8_mask(self.1, self.1)) << 32
            }

            #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
            {
                let z = _mm256_setzero_si256();
                let lo = _mm256_movemask_epi8(_mm256_cmpeq_epi8(self.0, z)).cast_unsigned();
                let hi = _mm256_movemask_epi8(_mm256_cmpeq_epi8(self.1, z)).cast_unsigned();
                !(u64::from(lo) | u64::from(hi) << 32)
            }
        }
    }

    /// Bytes with the sign bit set, which under Rose's layout is every black place.
    #[inline(always)]
    pub fn signs(self) -> u64 {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let lo = _mm256_movemask_epi8(self.0).cast_unsigned();
            let hi = _mm256_movemask_epi8(self.1).cast_unsigned();
            u64::from(lo) | u64::from(hi) << 32
        }
    }

    /// Bytes whose color and piece nibble equals `pattern`'s, ignoring the slot.
    #[inline(always)]
    pub fn match_high(self, pattern: u8) -> u64 {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let want = _mm256_set1_epi8(pattern.cast_signed());
            let nibble = _mm256_set1_epi8(0xF0u8.cast_signed());
            let hit = |v| _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(v, nibble), want)).cast_unsigned();
            u64::from(hit(self.0)) | u64::from(hit(self.1)) << 32
        }
    }

    #[inline(always)]
    pub fn sliders(self, g: &Geometry) -> u64 {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let s = _mm256_set1_epi8(SLIDER.cast_signed());
            let z = _mm256_setzero_si256();
            let mut out = 0u64;
            for (h, v) in [self.0, self.1].into_iter().enumerate() {
                let kind = _mm256_loadu_si256(g.lane_kind.as_ptr().add(h * 32).cast());
                let is_slider = _mm256_cmpeq_epi8(_mm256_and_si256(v, s), s);
                let right_way = _mm256_cmpeq_epi8(_mm256_and_si256(v, kind), kind);
                let knight = _mm256_cmpeq_epi8(kind, z);
                let m = _mm256_andnot_si256(knight, _mm256_and_si256(is_slider, right_way));
                out |= u64::from(_mm256_movemask_epi8(m).cast_unsigned()) << (h * 32);
            }
            out
        }
    }

    #[inline(always)]
    pub fn keep(self, mask: u64) -> Self {
        // SAFETY: AVX2 per the gate, AVX-512VL/BW under its own cfg.
        unsafe {
            #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
            {
                Self(_mm256_maskz_mov_epi8(mask as __mmask32, self.0), _mm256_maskz_mov_epi8((mask >> 32) as __mmask32, self.1))
            }

            #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
            {
                Self(_mm256_and_si256(self.0, spread(mask as u32)), _mm256_and_si256(self.1, spread((mask >> 32) as u32)))
            }
        }
    }

    #[inline(always)]
    pub fn lane_spread(self) -> Self {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let pick = _mm256_setr_epi8(
                -128, 0, 0, 0, 0, 0, 0, 0, -128, 8, 8, 8, 8, 8, 8, 8, -128, 0, 0, 0, 0, 0, 0, 0, -128, 8, 8, 8, 8, 8, 8, 8,
            );

            let z = _mm256_setzero_si256();
            Self(_mm256_shuffle_epi8(_mm256_sad_epu8(self.0, z), pick), _mm256_shuffle_epi8(_mm256_sad_epu8(self.1, z), pick))
        }
    }

    #[inline(always)]
    pub fn write(self, mask: u64, b: u8) -> Self {
        // SAFETY: AVX2 per the gate, AVX-512VL/BW under its own cfg.
        unsafe {
            let v = _mm256_set1_epi8(b.cast_signed());

            #[cfg(all(target_feature = "avx512vl", target_feature = "avx512bw"))]
            {
                Self(
                    _mm256_mask_blend_epi8(mask as __mmask32, self.0, v),
                    _mm256_mask_blend_epi8((mask >> 32) as __mmask32, self.1, v),
                )
            }

            #[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
            {
                let (lo, hi) = (spread(mask as u32), spread((mask >> 32) as u32));
                Self(
                    _mm256_or_si256(_mm256_andnot_si256(lo, self.0), _mm256_and_si256(lo, v)),
                    _mm256_or_si256(_mm256_andnot_si256(hi, self.1), _mm256_and_si256(hi, v)),
                )
            }
        }
    }

    #[inline(always)]
    pub fn rotate(self) -> Self {
        Self(self.1, self.0)
    }

    #[inline(always)]
    pub fn to_wordboards(self) -> ([__m256i; 4], [__m256i; 4]) {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let lut_lo = _mm256_setr_epi8(
                1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0,
            );
            let lut_hi = _mm256_setr_epi8(
                0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 8, 16, 32, 64, -128,
            );
            let z = _mm256_setzero_si256();

            let mut white = [z; 4];
            let mut black = [z; 4];

            for (h, src) in [self.0, self.1].into_iter().enumerate() {
                let id = _mm256_and_si256(src, _mm256_set1_epi8(15));
                let live = _mm256_cmpeq_epi8(_mm256_cmpeq_epi8(src, z), z);
                let lo = _mm256_and_si256(_mm256_shuffle_epi8(lut_lo, id), live);
                let hi = _mm256_and_si256(_mm256_shuffle_epi8(lut_hi, id), live);
                let blk = _mm256_cmpgt_epi8(z, src);

                let w0 = _mm256_or_si256(
                    _mm256_cvtepu8_epi16(_mm256_castsi256_si128(lo)),
                    _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(hi))),
                );
                let w1 = _mm256_or_si256(
                    _mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(lo)),
                    _mm256_slli_epi16::<8>(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(hi))),
                );

                let m0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(blk));
                let m1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(blk));

                white[h * 2] = _mm256_andnot_si256(m0, w0);
                black[h * 2] = _mm256_and_si256(m0, w0);
                white[h * 2 + 1] = _mm256_andnot_si256(m1, w1);
                black[h * 2 + 1] = _mm256_and_si256(m1, w1);
            }
            (white, black)
        }
    }
}

#[cfg(not(all(target_feature = "avx512vl", target_feature = "avx512bw")))]
#[inline(always)]
fn spread(bits: u32) -> __m256i {
    // SAFETY: AVX2 per the gate.
    unsafe {
        let v = _mm256_set1_epi32(bits.cast_signed());
        let idx = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        let sel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, idx), sel), sel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const A1: Square = Square(0);
    const D4: Square = Square(27);
    const H8: Square = Square(63);

    fn board() -> ByteBoard {
        let mut bb = ByteBoard::empty();
        bb.put(A1, Place::new(Color::White, PieceType::Rook, PieceId::new(0)));
        bb.put(D4, Place::new(Color::White, PieceType::Queen, PieceId::new(3)));
        bb.put(H8, Place::new(Color::Black, PieceType::Rook, PieceId::new(7)));
        bb
    }

    #[test]
    fn a_place_round_trips_through_its_byte() {
        let pieces = [
            (PieceType::King, PT_KING),
            (PieceType::Pawn, PT_PAWN),
            (PieceType::Knight, PT_KNIGHT),
            (PieceType::Bishop, PT_BISHOP),
            (PieceType::Rook, PT_ROOK),
            (PieceType::Queen, PT_QUEEN),
        ];

        for (pt, code) in pieces {
            for color in [Color::White, Color::Black] {
                let p = Place::new(color, pt, PieceId::new(9));
                assert!(!p.is_empty());
                assert_eq!(p.color(), color);
                assert_eq!(p.id(), PieceId::new(9));
                assert_eq!(p.code(), usize::from(code));
            }
        }
        assert!(Place::new(Color::White, PieceType::None, PieceId::new(0)).is_empty());
    }

    #[test]
    fn the_board_answers_by_color_and_by_type() {
        let bb = board();
        assert_eq!(bb.occupied(), 1 << 0 | 1 << 27 | 1 << 63);
        assert_eq!(bb.pieces(Color::White), 1 << 0 | 1 << 27);
        assert_eq!(bb.pieces(Color::Black), 1 << 63);
        assert_eq!(bb.pieces_of(Color::White, PieceType::Rook), 1 << 0);
        assert_eq!(bb.pieces_of(Color::Black, PieceType::Rook), 1 << 63);
        assert_eq!(bb.pieces_of(Color::White, PieceType::Queen), 1 << 27);
        assert_eq!(bb.pieces_of(Color::Black, PieceType::Queen), 0);
        assert_eq!(bb.at(D4).id(), PieceId::new(3));
        assert!(bb.at(Square(1)).is_empty());
    }

    #[test]
    fn a_mask_iterates_the_slots_it_holds() {
        let mut mask = PieceMask::EMPTY;
        mask.insert(PieceId::new(0));
        mask.insert(PieceId::new(15));
        assert_eq!(mask.count(), 2);
        assert!(mask.contains(PieceId::new(15)));
        assert!(!mask.contains(PieceId::new(1)));
        assert_eq!(mask.collect::<Vec<_>>(), vec![PieceId::new(0), PieceId::new(15)]);
        assert!(PieceMask::EMPTY.is_empty());
    }

    #[test]
    fn a_wordboard_reads_back_what_was_recorded() {
        let mut wb = Wordboard::EMPTY;
        wb.insert(D4, PieceId::new(3));
        wb.insert(H8, PieceId::new(4));
        assert!(wb.read(D4).contains(PieceId::new(3)));
        assert!(wb.read(A1).is_empty());
        assert_eq!(wb.any(), 1 << 27 | 1 << 63);
        assert_eq!(wb.with(PieceId::new(4).mask()), 1 << 63);
    }
}
