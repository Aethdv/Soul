//! Rose's ByteBoard, the square-indexed store the rig measures XorBoard against.
//!
//! A byte per square packs type, colour and slot, and `attack[colour][square]`
//! is the sixteen-bit set of that side's slots hitting the square. Updates run
//! setwise over 64-byte vectors against the ray geometry in [`Geometry`].

use core::arch::x86_64::*;

use crate::core::defs::{Color, PieceType, Square};

const PT_PAWN: u8 = 1;
const PT_KNIGHT: u8 = 2;
const PT_KING: u8 = 3;
const PT_BISHOP: u8 = 5;
const PT_ROOK: u8 = 6;
const PT_QUEEN: u8 = 7;

const SLIDER: u8 = 0b100 << 5;
const DIAG: u8 = 0b001 << 5;
const ORTH: u8 = 0b010 << 5;

const DIRS: [(i32, i32); 8] = [(0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1), (-1, 0), (-1, 1)];
const KNIGHTS: [(i32, i32); 8] = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)];

const KNIGHT_SLOTS: u64 = 0x0101_0101_0101_0101;
const RAY_SLOTS: u64 = !KNIGHT_SLOTS;

pub fn place(color: Color, pt: PieceType, id: u8) -> u8 {
    let code = match pt {
        PieceType::Pawn => PT_PAWN,
        PieceType::Knight => PT_KNIGHT,
        PieceType::King => PT_KING,
        PieceType::Bishop => PT_BISHOP,
        PieceType::Rook => PT_ROOK,
        PieceType::Queen => PT_QUEEN,
        PieceType::None => 0,
    };

    code << 5 | (color as u8) << 4 | id
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

pub struct ByteBoard {
    pub mailbox: V64,
    pub attack: [[u16; 64]; 2],
}

impl ByteBoard {
    pub fn empty() -> Self {
        Self { mailbox: V64::zero(), attack: [[0; 64]; 2] }
    }

    #[inline(always)]
    pub fn put(&mut self, sq: Square, p: u8) {
        self.mailbox = self.mailbox.write(1u64 << usize::from(sq), p);
    }

    #[inline(always)]
    pub fn remove(&mut self, g: &Geometry, sq: Square, color: Color, id: u8) {
        self.toggle(g, sq);
        self.clear(color, id);
        self.put(sq, 0);
    }

    #[inline(always)]
    pub fn add(&mut self, g: &Geometry, sq: Square, p: u8, color: Color, pt: PieceType) {
        self.toggle(g, sq);
        self.land(g, sq, p, color, pt);
    }

    #[inline(always)]
    pub fn land(&mut self, g: &Geometry, sq: Square, p: u8, color: Color, pt: PieceType) {
        self.put(sq, p);

        let s = usize::from(sq);
        let places = self.mailbox.permute(&g.rays[s]);
        let raymask = g.fill(s, places.nonzero());
        let reach = raymask & g.reach[usize::from(place(color, pt, 0) >> 5)][usize::from(color)];

        self.apply(V64::splat(p).keep(reach).permute(&g.inverse[s]));
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

    #[inline(always)]
    pub fn clear(&mut self, color: Color, id: u8) {
        // SAFETY: AVX2 per the gate.
        unsafe {
            let m = _mm256_set1_epi16((!(1u16 << id)).cast_signed());
            let p = self.attack[usize::from(color)].as_mut_ptr();

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
                let p = self.attack[c].as_mut_ptr();

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
        // SAFETY: AVX2 per the gate; two 32-byte reads of a 64-byte array.
        unsafe { Self(_mm256_loadu_si256(b.as_ptr().cast()), _mm256_loadu_si256(b.as_ptr().add(32).cast())) }
    }

    #[inline(always)]
    pub fn bytes(self) -> [u8; 64] {
        let mut out = [0u8; 64];
        // SAFETY: AVX2 per the gate; two 32-byte writes of a 64-byte array.
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
                let blk = _mm256_cmpeq_epi8(_mm256_and_si256(src, _mm256_set1_epi8(16)), _mm256_set1_epi8(16));

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
    // SAFETY: AVX2 per the gate; register-only.
    unsafe {
        let v = _mm256_set1_epi32(bits.cast_signed());
        let idx = _mm256_setr_epi8(0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3);
        let sel = _mm256_setr_epi8(
            1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128, 1, 2, 4, 8, 16, 32, 64, -128,
        );
        _mm256_cmpeq_epi8(_mm256_and_si256(_mm256_shuffle_epi8(v, idx), sel), sel)
    }
}
