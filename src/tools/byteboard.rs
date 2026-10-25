//! 87flowers' ByteBoard.

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
#[repr(transparent)]
pub(super) struct Place(u8);

impl Place {
    pub(super) const EMPTY: Self = Self(0);

    const BLACK: u8 = 0x80;
    const PTYPE: u8 = 0b111 << PTYPE_SHIFT;

    pub(super) fn new(color: Color, pt: PieceType, id: PieceId) -> Self {
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

    pub(super) fn is_empty(self) -> bool { self.0 == 0 }
    pub(super) fn color(self) -> Color { if self.0 & Self::BLACK == 0 { Color::White } else { Color::Black } }

    /// The piece code, which is also the row `Geometry::attack_mask` is indexed by.
    fn code(self) -> usize { usize::from(self.0 & Self::PTYPE) >> PTYPE_SHIFT }

    pub(super) fn id(self) -> PieceId { PieceId(self.0 & 0xF) }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(super) struct PieceId(u8);

impl PieceId {
    pub(super) fn new(slot: u8) -> Self {
        debug_assert!(slot < 16, "a side has sixteen slots");
        Self(slot)
    }

    pub(super) fn raw(self) -> u8 { self.0 }
    pub(super) fn to_piece_mask(self) -> PieceMask { PieceMask(1 << self.0) }
}

#[derive(Clone, Copy, PartialEq, Eq, Default, Debug)]
pub(super) struct PieceMask(u16);

impl PieceMask {
    pub(super) const EMPTY: Self = Self(0);

    pub(super) fn is_empty(self) -> bool { self.0 == 0 }
    pub(super) fn popcount(self) -> u32 { self.0.count_ones() }
    pub(super) fn bits(self) -> u16 { self.0 }
    pub(super) fn is_set(self, id: PieceId) -> bool { self.0 & id.to_piece_mask().0 != 0 }
    pub(super) fn set(&mut self, id: PieceId) { self.0 |= id.to_piece_mask().0; }
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

#[derive(Clone, Copy)]
pub(super) struct Wordboard([u16; 64]);

impl Wordboard {
    pub(super) const EMPTY: Self = Self([0; 64]);

    /// Record that `id` reaches `sq`, for a table built from scratch.
    pub(super) fn insert(&mut self, sq: Square, id: PieceId) { self.0[usize::from(sq)] |= id.to_piece_mask().0; }
    pub(super) fn read(&self, sq: Square) -> PieceMask { PieceMask(self.0[usize::from(sq)]) }
    /// The squares this color attacks at all.
    pub(super) fn bitboard_any(&self) -> u64 { self.probe(u16::MAX) }
    /// The squares any slot in `mask` attacks.
    pub(super) fn bitboard_for(&self, mask: PieceMask) -> u64 { self.probe(mask.0) }

    fn probe(&self, mask: u16) -> u64 {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let m = _mm256_set1_epi16(mask.cast_signed());
            let zero = _mm256_setzero_si256();
            let mut out = 0u64;

            for half in 0..2 {
                let load = |at: usize| _mm256_and_si256(_mm256_loadu_si256(self.0.as_ptr().add(at).cast()), m);
                let lo = _mm256_cmpeq_epi16(load(half * 32), zero);
                let hi = _mm256_cmpeq_epi16(load(half * 32 + 16), zero);

                // packs interleaves the 128-bit halves, so the permute puts
                // the squares back in order before the mask is read off.
                let packed = _mm256_permute4x64_epi64::<0xD8>(_mm256_packs_epi16(lo, hi));
                out |= u64::from(!_mm256_movemask_epi8(packed).cast_unsigned()) << (half * 32);
            }

            out
        }
    }
}

pub(super) struct Geometry {
    ray_coords: [[u8; 64]; 64],
    inverse_rays: [[u8; 64]; 64],
    ray_valid: [u64; 64],
    attack_mask: [[u64; 2]; 8],
    lane_kind: [u8; 64],
}

impl Geometry {
    pub(super) fn new() -> Self {
        let mut g = Self {
            ray_coords: [[0xFF; 64]; 64],
            inverse_rays: [[0xFF; 64]; 64],
            ray_valid: [0; 64],
            attack_mask: [[0; 2]; 8],
            lane_kind: [0; 64],
        };

        for sq in 0..64usize {
            let (f, r) = (sq as i32 % 8, sq as i32 / 8);

            for (lane, &(df, dr)) in DIRS.iter().enumerate() {
                let slot = lane * 8;
                let (kf, kr) = KNIGHTS[lane];

                if (0..8).contains(&(f + kf)) && (0..8).contains(&(r + kr)) {
                    let target = ((r + kr) * 8 + f + kf) as usize;
                    g.ray_coords[sq][slot] = target as u8;
                    g.inverse_rays[sq][target] = slot as u8;
                    g.ray_valid[sq] |= 1 << slot;
                }

                for d in 1..8usize {
                    let (tf, tr) = (f + df * d as i32, r + dr * d as i32);

                    if !(0..8).contains(&tf) || !(0..8).contains(&tr) {
                        break;
                    }

                    let target = (tr * 8 + tf) as usize;
                    g.ray_coords[sq][slot + d] = target as u8;
                    g.inverse_rays[sq][target] = (slot + d) as u8;
                    g.ray_valid[sq] |= 1 << (slot + d);
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

        g.attack_mask[PT_KNIGHT as usize] = [KNIGHT_SLOTS; 2];
        g.attack_mask[PT_KING as usize] = [near; 2];
        g.attack_mask[PT_BISHOP as usize] = [diag; 2];
        g.attack_mask[PT_ROOK as usize] = [orth; 2];
        g.attack_mask[PT_QUEEN as usize] = [orth | diag; 2];
        g.attack_mask[PT_PAWN as usize] = [0x0200_0000_0000_0200, 0x0000_0200_0200_0000];
        g
    }

    #[inline(always)]
    fn attacks(&self, sq: usize, ray_occupancy: u64) -> u64 {
        let o = ray_occupancy | 0x8181_8181_8181_8181;
        (o ^ o.wrapping_sub(0x0303_0303_0303_0303)) & self.ray_valid[sq]
    }
}

#[derive(Clone, Copy)]
pub(super) struct ByteBoard {
    mailbox: [Place; 64],
    attack: [Wordboard; 2],
}

impl ByteBoard {
    pub(super) fn empty() -> Self { Self { mailbox: [Place::EMPTY; 64], attack: [Wordboard::EMPTY; 2] } }
    pub(super) fn at(&self, sq: Square) -> Place { self.mailbox[usize::from(sq)] }
    pub(super) fn attack_table(&self, color: Color) -> &Wordboard { &self.attack[usize::from(color)] }
    pub(super) fn set_attacks(&mut self, attacks: [Wordboard; 2]) { self.attack = attacks; }
    pub(super) fn occupied_bitboard(&self) -> u64 { self.to_vector().nonzero() }
    pub(super) fn color_bitboard(&self, color: Color) -> u64 {
        let black = self.to_vector().signs();
        self.occupied_bitboard() & if color == Color::Black { black } else { !black }
    }

    pub(super) fn bitboard_for(&self, color: Color, pt: PieceType) -> u64 {
        self.to_vector().match_high(Place::new(color, pt, PieceId(0)).0)
    }

    #[inline(always)]
    pub(super) fn put(&mut self, sq: Square, p: Place) { self.mailbox[usize::from(sq)] = p; }

    #[inline(always)]
    pub(super) fn remove_piece(&mut self, g: &Geometry, sq: Square) {
        let p = self.at(sq);
        self.toggle_sliders_single(g, sq);
        self.remove_attacks(p.color(), p.id());
        self.put(sq, Place::EMPTY);
    }

    #[inline(always)]
    pub(super) fn add_piece(&mut self, g: &Geometry, sq: Square, p: Place) {
        self.toggle_sliders_single(g, sq);
        self.replace_piece(g, sq, p);
    }

    #[inline(always)]
    pub(super) fn replace_piece(&mut self, g: &Geometry, sq: Square, p: Place) {
        self.put(sq, p);
        let s = usize::from(sq);
        let (_, raymask) = self.rays_from(g, s);
        let reach = raymask & g.attack_mask[p.code()][usize::from(p.color())];
        let marked = V64::splat(0xFF).keep(reach).permute(&g.inverse_rays[s]);
        self.add_attacks(p.color(), p.id(), marked);
    }

    #[inline(always)]
    fn add_attacks(&mut self, color: Color, id: PieceId, marked: V64) { self.xor_into(color, marked.marked_words(id)); }

    #[inline(always)]
    fn rays_from(&self, g: &Geometry, s: usize) -> (V64, u64) {
        let places = self.to_vector().permute(&g.ray_coords[s]);
        (places, g.attacks(s, places.nonzero()))
    }

    #[inline(always)]
    fn toggle_sliders_single(&mut self, g: &Geometry, sq: Square) {
        let s = usize::from(sq);
        let (places, raymask) = self.rays_from(g, s);

        self.apply_words(places.slider_words(g, s, raymask));
    }

    #[inline(always)]
    fn apply_words(&mut self, words: Option<([__m256i; 4], [__m256i; 4])>) {
        if let Some((white, black)) = words {
            self.xor_into(Color::White, white);
            self.xor_into(Color::Black, black);
        }
    }

    #[inline(always)]
    pub(super) fn move_piece(&mut self, g: &Geometry, from: Square, to: Square, p: Place, captured: Option<(Color, PieceId)>) {
        let s = usize::from(from);
        let t = usize::from(to);

        let board = self.to_vector();
        let src_places = board.permute(&g.ray_coords[s]);
        let src_ray = g.attacks(s, src_places.nonzero());

        let moved = board.write(1u64 << s, 0);
        let dst_places = moved.permute(&g.ray_coords[t]);
        let dst_ray = g.attacks(t, dst_places.nonzero());

        moved.write(1u64 << t, p.0).store(self.mailbox.as_mut_ptr().cast());
        self.apply_words(src_places.slider_words(g, s, src_ray));

        let dst_words = if captured.is_some() { None } else { dst_places.slider_words(g, t, dst_ray) };
        self.apply_words(dst_words);

        self.remove_attacks(p.color(), p.id());
        let reach = dst_ray & g.attack_mask[p.code()][usize::from(p.color())];
        let marked = V64::splat(0xFF).keep(reach).permute(&g.inverse_rays[t]);
        self.add_attacks(p.color(), p.id(), marked);

        if let Some((color, id)) = captured {
            self.remove_attacks(color, id);
        }
    }

    /// The board as the vector the updates run on.
    #[inline(always)]
    #[expect(clippy::wrong_self_convention, reason = "the board is 320 bytes, so the view must not copy it")]
    fn to_vector(&self) -> V64 { V64::load(self.mailbox.as_ptr().cast()) }

    #[inline(always)]
    pub(super) fn remove_attacks(&mut self, color: Color, id: PieceId) {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let m = _mm256_set1_epi16((!id.to_piece_mask().0).cast_signed());
            let p = self.attack[usize::from(color)].0.as_mut_ptr();
            for q in 0..4 {
                let v = _mm256_loadu_si256(p.add(q * 16).cast());
                _mm256_storeu_si256(p.add(q * 16).cast(), _mm256_and_si256(v, m));
            }
        }
    }

    #[inline(always)]
    fn xor_into(&mut self, color: Color, words: [__m256i; 4]) {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let p = self.attack[usize::from(color)].0.as_mut_ptr();
            for (q, x) in words.iter().enumerate() {
                let v = _mm256_loadu_si256(p.add(q * 16).cast());
                _mm256_storeu_si256(p.add(q * 16).cast(), _mm256_xor_si256(v, *x));
            }
        }
    }
}

#[derive(Clone, Copy)]
struct V64(__m256i, __m256i);

impl V64 {
    #[inline(always)]
    fn splat(b: u8) -> Self {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let v = _mm256_set1_epi8(b.cast_signed());
            Self(v, v)
        }
    }

    /// The board's bytes as a vector.
    #[inline(always)]
    fn load(ptr: *const u8) -> Self {
        // SAFETY: AVX2 guaranteed; the pointer reads 64 bytes.
        unsafe { Self(_mm256_loadu_si256(ptr.cast()), _mm256_loadu_si256(ptr.add(32).cast())) }
    }

    /// Write the bytes back.
    #[inline(always)]
    fn store(self, ptr: *mut u8) {
        // SAFETY: AVX2 guaranteed; the pointer writes 64 bytes.
        unsafe {
            _mm256_storeu_si256(ptr.cast(), self.0);
            _mm256_storeu_si256(ptr.add(32).cast(), self.1);
        }
    }

    #[inline(always)]
    fn permute(self, idx: &[u8; 64]) -> Self {
        // SAFETY: AVX2 guaranteed, AVX-512 under its own cfg.
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
                let blocks = [
                    _mm256_permute2x128_si256::<0x00>(self.0, self.0),
                    _mm256_permute2x128_si256::<0x11>(self.0, self.0),
                    _mm256_permute2x128_si256::<0x00>(self.1, self.1),
                    _mm256_permute2x128_si256::<0x11>(self.1, self.1),
                ];

                let sign = _mm256_set1_epi8(-128);
                let mut out = [_mm256_setzero_si256(); 2];

                for (h, o) in out.iter_mut().enumerate() {
                    let want = _mm256_loadu_si256(idx.as_ptr().add(h * 32).cast());
                    // The index's high bit zeroes the byte, so the invalid 0xFF entries come out empty.
                    let ctrl = _mm256_or_si256(_mm256_and_si256(want, _mm256_set1_epi8(15)), _mm256_and_si256(want, sign));
                    let four = _mm256_slli_epi16::<3>(want);
                    let five = _mm256_slli_epi16::<2>(want);

                    let near = _mm256_blendv_epi8(_mm256_shuffle_epi8(blocks[0], ctrl), _mm256_shuffle_epi8(blocks[1], ctrl), four);
                    let far = _mm256_blendv_epi8(_mm256_shuffle_epi8(blocks[2], ctrl), _mm256_shuffle_epi8(blocks[3], ctrl), four);
                    *o = _mm256_blendv_epi8(near, far, five);
                }
                Self(out[0], out[1])
            }
        }
    }

    #[inline(always)]
    fn nonzero(self) -> u64 {
        // SAFETY: AVX2 guaranteed, AVX-512VL/BW under its own cfg.
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

    /// Bytes with the sign bit set, which is every black place.
    #[inline(always)]
    fn signs(self) -> u64 {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let lo = _mm256_movemask_epi8(self.0).cast_unsigned();
            let hi = _mm256_movemask_epi8(self.1).cast_unsigned();
            u64::from(lo) | u64::from(hi) << 32
        }
    }

    /// Bytes whose color and piece nibble equals `pattern`'s, ignoring the slot.
    #[inline(always)]
    fn match_high(self, pattern: u8) -> u64 {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let want = _mm256_set1_epi8(pattern.cast_signed());
            let nibble = _mm256_set1_epi8(0xF0u8.cast_signed());
            let hit = |v| _mm256_movemask_epi8(_mm256_cmpeq_epi8(_mm256_and_si256(v, nibble), want)).cast_unsigned();
            u64::from(hit(self.0)) | u64::from(hit(self.1)) << 32
        }
    }

    #[inline(always)]
    fn sliders(self, g: &Geometry) -> u64 {
        // SAFETY: AVX2 guaranteed.
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
    fn keep(self, mask: u64) -> Self {
        // SAFETY: AVX2 guaranteed, AVX-512VL/BW under its own cfg.
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
    fn lane_spread(self) -> Self {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let pick = _mm256_setr_epi8(
                -128, 0, 0, 0, 0, 0, 0, 0, -128, 8, 8, 8, 8, 8, 8, 8, -128, 0, 0, 0, 0, 0, 0, 0, -128, 8, 8, 8, 8, 8, 8, 8,
            );

            let z = _mm256_setzero_si256();
            Self(_mm256_shuffle_epi8(_mm256_sad_epu8(self.0, z), pick), _mm256_shuffle_epi8(_mm256_sad_epu8(self.1, z), pick))
        }
    }

    #[inline(always)]
    fn write(self, mask: u64, b: u8) -> Self {
        // SAFETY: AVX2 guaranteed, AVX-512VL/BW under its own cfg.
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
                Self(_mm256_blendv_epi8(self.0, v, lo), _mm256_blendv_epi8(self.1, v, hi))
            }
        }
    }

    #[inline(always)]
    fn flip(self) -> Self { Self(self.1, self.0) }

    #[inline(always)]
    fn slider_words(self, g: &Geometry, sq: usize, raymask: u64) -> Option<([__m256i; 4], [__m256i; 4])> {
        let visible = raymask & self.sliders(g);
        if visible == 0 {
            return None;
        }

        let ids = self.keep(visible).lane_spread().flip();
        let ids = ids.keep(raymask & RAY_SLOTS).permute(&g.inverse_rays[sq]);

        Some(ids.words_by_color())
    }

    #[inline(always)]
    fn marked_words(self, id: PieceId) -> [__m256i; 4] {
        // SAFETY: AVX2 guaranteed.
        unsafe {
            let mark = _mm256_set1_epi8(-1);
            let bit = _mm256_set1_epi16(1i16 << id.raw());
            let mut out = [_mm256_setzero_si256(); 4];

            for (h, src) in [self.0, self.1].into_iter().enumerate() {
                let live = _mm256_cmpeq_epi8(src, mark);
                let lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(live));
                let hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256::<1>(live));
                out[h * 2] = _mm256_and_si256(lo, bit);
                out[h * 2 + 1] = _mm256_and_si256(hi, bit);
            }

            out
        }
    }

    #[inline(always)]
    fn words_by_color(self) -> ([__m256i; 4], [__m256i; 4]) {
        // SAFETY: AVX2 guaranteed.
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
    // SAFETY: AVX2 guaranteed.
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
        assert_eq!(bb.occupied_bitboard(), 1 << 0 | 1 << 27 | 1 << 63);
        assert_eq!(bb.color_bitboard(Color::White), 1 << 0 | 1 << 27);
        assert_eq!(bb.color_bitboard(Color::Black), 1 << 63);
        assert_eq!(bb.bitboard_for(Color::White, PieceType::Rook), 1 << 0);
        assert_eq!(bb.bitboard_for(Color::Black, PieceType::Rook), 1 << 63);
        assert_eq!(bb.bitboard_for(Color::White, PieceType::Queen), 1 << 27);
        assert_eq!(bb.bitboard_for(Color::Black, PieceType::Queen), 0);
        assert_eq!(bb.at(D4).id(), PieceId::new(3));
        assert!(bb.at(Square(1)).is_empty());
    }

    #[test]
    fn a_byte_reads_back_from_every_square() {
        let types = [PieceType::King, PieceType::Pawn, PieceType::Knight, PieceType::Queen];
        let place = |sq: u8| Place::new(Color::Black, types[usize::from(sq) / 16], PieceId::new(sq % 16));
        let mut bb = ByteBoard::empty();

        for sq in 0..64u8 {
            bb.put(Square(sq), place(sq));
        }
        for sq in 0..64u8 {
            assert_eq!(bb.at(Square(sq)), place(sq));
        }
    }

    #[test]
    fn the_rays_and_their_inverse_agree() {
        let g = Geometry::new();
        let e4 = 28usize;
        let mut reachable = 0u64;

        for sq in 0..64usize {
            for slot in 0..64usize {
                if g.ray_valid[sq] >> slot & 1 == 0 {
                    continue;
                }

                let target = usize::from(g.ray_coords[sq][slot]);
                assert_eq!(usize::from(g.inverse_rays[sq][target]), slot);

                if sq == e4 {
                    reachable |= 1 << target;
                }
            }
        }

        assert_eq!(reachable, 0b0001_0001_1001_0010_0111_1100_0111_1100_1110_1111_0111_1100_0111_1100_1001_0010);
    }

    #[test]
    fn a_permute_takes_each_byte_from_its_index() {
        let types = [PieceType::King, PieceType::Pawn, PieceType::Knight, PieceType::Queen];
        let mut board = ByteBoard::empty();

        for sq in 0..64u8 {
            board.put(Square(sq), Place::new(Color::White, types[usize::from(sq) / 16], PieceId::new(sq % 16)));
        }

        let mut idx = [0u8; 64];
        for (i, slot) in idx.iter_mut().enumerate() {
            *slot = ((i * 37 + 11) % 64) as u8;
        }

        idx[5] = 0xFF;

        let permuted = board.to_vector().permute(&idx);
        let mut out = [0u8; 64];
        permuted.store(out.as_mut_ptr());

        for (i, &slot) in idx.iter().enumerate() {
            let want = if slot == 0xFF { 0 } else { board.at(Square(slot)).0 };
            assert_eq!(out[i], want, "byte {i}");
        }
    }

    #[test]
    fn a_mask_iterates_the_slots_it_holds() {
        let mut mask = PieceMask::EMPTY;
        mask.set(PieceId::new(0));
        mask.set(PieceId::new(15));
        assert_eq!(mask.popcount(), 2);
        assert!(mask.is_set(PieceId::new(15)));
        assert!(!mask.is_set(PieceId::new(1)));
        assert_eq!(mask.collect::<Vec<_>>(), vec![PieceId::new(0), PieceId::new(15)]);
        assert!(PieceMask::EMPTY.is_empty());
    }

    #[test]
    fn a_wordboard_reads_back_what_was_recorded() {
        let mut wb = Wordboard::EMPTY;
        wb.insert(D4, PieceId::new(3));
        wb.insert(H8, PieceId::new(4));
        assert!(wb.read(D4).is_set(PieceId::new(3)));
        assert!(wb.read(A1).is_empty());
        assert_eq!(wb.bitboard_any(), 1 << 27 | 1 << 63);
        assert_eq!(wb.bitboard_for(PieceId::new(4).to_piece_mask()), 1 << 63);
    }
}
