//! Fixed instruction-level benchmark for the attack-table designs.
//!
//! Usage:
//! ```text
//!     soul atb gen <path> [ply_cap]
//!     soul atb export <path> <out>
//!     soul atb validate <path>
//!     soul atb count <path>
//!     soul atb dump <path> [plies]
//!     perf stat -e instructions,cycles soul atb run <variant> <path> <repeats>
//! ```

use core::{arch::x86_64::*, hint::black_box, marker::ConstParamTy};
use std::{fs, hint};

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_queen, atk_rook, between_bb},
            castling_targets,
            spatial::SliderMaps,
        },
        defs::{Bitboard, Color, PieceType, Square},
        moves::Move,
        zobrist::ConstRng,
    },
    engine::{
        mobility,
        movegen::gen_legal_moves,
        search_params::SearchParams,
        see::{see_ge, see_values},
    },
    tools::byteboard::{self, PieceId, Place, Wordboard},
    weave::I16x8,
};

const FENS: &str = include_str!("../data/bench.fens");
const PLY_CAP: usize = 256;
const SEE_VALUES: [i32; 8] = see_values(&SearchParams::new());

struct Game {
    fen: String,
    moves: Vec<Move>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct PieceIds {
    /// Square -> id + 1; zero is empty.
    at: [u8; 64],
    squares: [u8; 32],
    types: [PieceType; 32],
    type_masks: [u64; 12],
    slider_groups: [u8; 2],
}

struct MovePlan {
    movers: [(usize, Square, Square); 2],
    n_movers: usize,
    changed: [Square; 4],
    n_changed: usize,
    captured: Option<(usize, Square)>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct IncrementalAttackTable {
    attackers: [u32; 64],
    rows: [u64; 32],
    danger: [u64; 2],
    ids: PieceIds,
}

#[derive(Clone, Copy, Debug)]
struct XorBoard {
    rows: [u64; 32],
    xray: [u64; 32],
    vision: u64,
    ids: PieceIds,
}

impl PartialEq for XorBoard {
    fn eq(&self, other: &Self) -> bool { self.rows == other.rows && self.xray == other.xray && self.ids == other.ids }
}

#[derive(Clone, Copy)]
struct PreMove {
    occ: Bitboard,
    rq_squares: Bitboard,
    bq_squares: Bitboard,
}

struct Undo {
    len: usize,
    ids: [u8; 24],
    rows: [u64; 24],
    xray: [u64; 24],
    vision: u64,
    slider_groups: [u8; 2],
}

struct ByteStore {
    board: byteboard::ByteBoard,
    ids: PieceIds,
}

struct Outcome {
    variant: &'static str,
    iterations: u64,
    checksum: u64,
}

impl ByteStore {
    fn from_scratch(pos: &Position) -> Self {
        let mut store = Self { board: byteboard::ByteBoard::empty(), ids: PieceIds::seed(pos) };

        for id in 0..32 {
            let sq = store.ids.squares[id];
            if sq != 0xFF {
                store.board.put(Square(sq), Place::new(color(id), store.ids.types[id], slot(id)));
            }
        }

        store.rebuild(pos);
        store
    }

    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occ;
        let mut attacks = [Wordboard::EMPTY; 2];

        for id in 0..32 {
            let sq = self.ids.squares[id];
            if sq == 0xFF {
                continue;
            }

            for s in attacks_of(self.ids.types[id], Square(sq), side_of(id), occ) {
                attacks[side_of(id)].insert(s, slot(id));
            }
        }

        self.board.set_attacks(attacks);
    }

    #[inline(always)]
    fn make(&mut self, g: &byteboard::Geometry, mv: Move) {
        // Clear the victim and all origins first; DFRC castling can set a destination on an origin.
        let plan = self.ids.plan_move(mv);
        let replaced = plan.captured.is_some() && !mv.is_en_passant();

        if let Some((victim, victim_sq)) = plan.captured {
            if !replaced {
                self.board.remove_piece(g, victim_sq);
            }
            self.ids.at[usize::from(victim_sq)] = 0;
            self.ids.squares[victim] = 0xFF;
        }

        if plan.n_movers == 1 {
            let (id, from, to) = plan.movers[0];
            self.ids.apply(mv, &plan);

            let place = Place::new(color(id), self.ids.types[id], slot(id));
            let captured = plan.captured.filter(|_| replaced).map(|(victim, _)| (color(victim), slot(victim)));
            self.board.move_piece(g, from, to, place, captured);
            return;
        }

        for m in 0..plan.n_movers {
            let (_, from, _) = plan.movers[m];
            self.board.remove_piece(g, from);
        }

        if let Some((victim, _)) = plan.captured.filter(|_| replaced) {
            self.board.remove_attacks(color(victim), slot(victim));
        }

        self.ids.apply(mv, &plan);

        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            let place = Place::new(color(id), self.ids.types[id], slot(id));

            if replaced {
                self.board.replace_piece(g, to, place);
            } else {
                self.board.add_piece(g, to, place);
            }
        }
    }
}

#[inline(always)]
fn color(id: usize) -> Color { if id < 16 { Color::White } else { Color::Black } }
fn slot(id: usize) -> PieceId { PieceId::new((id & 15) as u8) }

fn generate(seed: u64, ply_cap: usize) -> Vec<Game> {
    let mut rng = ConstRng::new(seed);
    let mut games = Vec::new();

    for fen in FENS.lines() {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.psqt_accumulator();
        let mut moves = Vec::new();

        for _ in 0..ply_cap {
            let legal = gen_legal_moves(&pos);
            if legal.is_empty() {
                break;
            }

            let mv = legal[(rng.next() % legal.len() as u64) as usize];
            pos.make_move(mv, &mut acc);
            moves.push(mv);
        }

        games.push(Game { fen: fen.to_string(), moves });
    }

    games
}

fn serialize(stream: &[Game]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(u16::try_from(stream.len()).expect("too many games for the stream header").to_le_bytes());

    for game in stream {
        out.push(u8::try_from(game.fen.len()).expect("fen too long for the stream header"));
        out.extend(game.fen.as_bytes());
        out.extend(
            u16::try_from(game.moves.len())
                .expect("too many plies for the stream header")
                .to_le_bytes(),
        );

        for mv in &game.moves {
            out.extend(mv.inner().to_le_bytes());
        }
    }

    out
}

fn deserialize(bytes: &[u8]) -> Vec<Game> {
    let mut rest = bytes;
    let n_games = take_u16(&mut rest);
    let mut games = Vec::with_capacity(n_games);

    for _ in 0..n_games {
        let (&fen_len, tail) = rest.split_first().expect("stream truncated");
        let (fen, tail) = tail.split_at_checked(usize::from(fen_len)).expect("stream truncated");
        rest = tail;

        let n_moves = take_u16(&mut rest);
        let (packed, tail) = rest.split_at_checked(n_moves * 2).expect("stream truncated");
        rest = tail;

        games.push(Game {
            fen: String::from_utf8(fen.to_vec()).expect("stream fen"),
            moves: packed
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&raw| Move::from_u16(u16::from_le_bytes(raw)))
                .collect(),
        });
    }

    games
}

fn take_u16(rest: &mut &[u8]) -> usize {
    let (head, tail) = rest.split_first_chunk::<2>().expect("stream truncated");
    *rest = tail;
    usize::from(u16::from_le_bytes(*head))
}

impl PieceIds {
    fn empty() -> Self {
        Self {
            at: [0; 64],
            squares: [0xFF; 32],
            types: [PieceType::None; 32],
            type_masks: [0; 12],
            slider_groups: [0; 2],
        }
    }

    fn seed(pos: &Position) -> Self {
        let mut ids = Self::empty();
        let mut white_next = 0usize;
        let mut black_next = 16usize;

        for pass in 0..2 {
            for sq in 0..64 {
                let pt = pos.piece_at(Square(sq as u8));
                if pt == PieceType::None || is_slider(pt) != (pass == 0) {
                    continue;
                }

                let (id, piece_color) = if pos.side_bb[Color::White].0 & (1u64 << sq) != 0 {
                    let id = white_next;
                    white_next += 1;
                    (id, Color::White)
                } else {
                    let id = black_next;
                    black_next += 1;
                    (id, Color::Black)
                };

                ids.at[sq] = (id + 1) as u8;
                ids.squares[id] = sq as u8;
                ids.types[id] = pt;
                ids.type_masks[pt as usize * 2 + piece_color as usize] |= 1u64 << id;

                if pass == 0 {
                    ids.slider_groups[piece_color as usize] = group_count(id);
                }
            }
        }

        ids
    }

    #[inline(always)]
    fn plan_move(&self, mv: Move) -> MovePlan {
        let from = mv.from();
        let to = mv.to();
        let mover = usize::from(self.at[usize::from(from)]) - 1;
        let mut movers = [(0usize, Square(0), Square(0)); 2];
        let mut changed;

        let (n_movers, mut n_changed) = if mv.is_castling() {
            let rook = usize::from(self.at[usize::from(to)]) - 1;
            let (king_to, rook_to) = castling_targets(from, to);
            movers[0] = (mover, from, king_to);
            // The encoded destination of a castle is the rook's own square.
            movers[1] = (rook, to, rook_to);
            changed = [from, king_to, to, rook_to];
            (2, 4)
        } else {
            movers[0] = (mover, from, to);
            changed = [from, to, Square(0), Square(0)];
            (1, 2)
        };

        let captured = if mv.is_castling() {
            None
        } else if mv.is_en_passant() {
            let victim_sq = Square(to.0 ^ 8);
            changed[2] = victim_sq;
            let victim = usize::from(self.at[usize::from(victim_sq)]) - 1;
            Some((victim, victim_sq))
        } else if mv.is_capture() {
            Some((usize::from(self.at[usize::from(to)]) - 1, to))
        } else {
            None
        };

        if mv.is_en_passant() {
            n_changed = 3;
        }

        MovePlan { movers, n_movers, changed, n_changed, captured }
    }

    #[inline(always)]
    fn plan_unmove(&self, mv: Move, victim: Option<usize>) -> MovePlan {
        let from = mv.from();
        let to = mv.to();
        let mut movers = [(0usize, Square(0), Square(0)); 2];
        let n_movers = if mv.is_castling() {
            let (king_to, rook_to) = castling_targets(from, to);
            movers[0] = (usize::from(self.at[usize::from(king_to)]) - 1, from, king_to);
            movers[1] = (usize::from(self.at[usize::from(rook_to)]) - 1, to, rook_to);
            2
        } else {
            movers[0] = (usize::from(self.at[usize::from(to)]) - 1, from, to);
            1
        };

        let captured = victim.map(|victim| (victim, if mv.is_en_passant() { Square(to.0 ^ 8) } else { to }));
        MovePlan { movers, n_movers, changed: [Square(0); 4], n_changed: 0, captured }
    }

    #[inline(always)]
    fn apply(&mut self, mv: Move, plan: &MovePlan) {
        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].1)] = 0;
        }

        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].2)] = (plan.movers[m].0 + 1) as u8;
            self.squares[plan.movers[m].0] = plan.movers[m].2.0;
        }

        if mv.is_promotion() {
            let id = plan.movers[0].0;
            let new_type = mv.promo().unwrap_or(PieceType::Queen);
            let old_type = self.types[id];
            let side = side_of(id);
            self.type_masks[old_type as usize * 2 + side] &= !(1u64 << id);
            self.type_masks[new_type as usize * 2 + side] |= 1u64 << id;
            self.types[id] = new_type;
            self.slider_groups[side] = self.slider_groups[side].max(group_count(id));
        }
    }

    #[inline(always)]
    fn undo(&mut self, mv: Move, plan: &MovePlan, slider_groups: [u8; 2]) {
        self.slider_groups = slider_groups;

        if mv.is_promotion() {
            let id = plan.movers[0].0;
            let new_type = self.types[id];
            let side = side_of(id);
            self.type_masks[new_type as usize * 2 + side] &= !(1u64 << id);
            self.type_masks[PieceType::Pawn as usize * 2 + side] |= 1u64 << id;
            self.types[id] = PieceType::Pawn;
        }

        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].2)] = 0;
        }
        for m in 0..plan.n_movers {
            self.at[usize::from(plan.movers[m].1)] = (plan.movers[m].0 + 1) as u8;
            self.squares[plan.movers[m].0] = plan.movers[m].1.0;
        }

        if let Some((victim, victim_sq)) = plan.captured {
            self.at[usize::from(victim_sq)] = (victim + 1) as u8;
            self.squares[victim] = victim_sq.0;
        }
    }

    #[inline(always)]
    fn slider_mask(&self) -> u64 {
        self.type_mask(PieceType::Bishop) | self.type_mask(PieceType::Rook) | self.type_mask(PieceType::Queen)
    }

    #[inline(always)]
    fn type_mask(&self, pt: PieceType) -> u64 { self.type_masks[pt as usize * 2] | self.type_masks[pt as usize * 2 + 1] }
}

#[inline(always)]
fn side_of(id: usize) -> usize { id >> 4 }

/// The number of 4-slot groups up to and including the one holding `id`.
#[inline(always)]
fn group_count(id: usize) -> u8 { (id % 16 / 4 + 1) as u8 }

#[inline(always)]
fn is_slider(pt: PieceType) -> bool { matches!(pt, PieceType::Bishop | PieceType::Rook | PieceType::Queen) }

#[inline(always)]
fn attacks_of(pt: PieceType, sq: Square, side: usize, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => atk_pawn(sq, if side == 0 { Color::White } else { Color::Black }),
        PieceType::Knight => atk_knight(sq),
        PieceType::King => atk_king(sq),
        PieceType::Bishop => atk_bishop(sq, occ),
        PieceType::Rook => atk_rook(sq, occ),
        PieceType::Queen => atk_queen(sq, occ),
        PieceType::None => Bitboard(0),
    }
}

#[inline(always)]
fn planes_of(pt: PieceType, sq: Square, side: usize, occ: Bitboard) -> (Bitboard, Bitboard) {
    let direct = attacks_of(pt, sq, side, occ);
    if !is_slider(pt) {
        return (direct, Bitboard(0));
    }

    (direct, attacks_of(pt, sq, side, occ & !direct) & !direct)
}

#[inline(always)]
fn ids_of(mut set: u64) -> impl Iterator<Item = usize> {
    core::iter::from_fn(move || {
        (set != 0).then(|| {
            let id = set.trailing_zeros() as usize;
            set &= set - 1;
            id
        })
    })
}

impl IncrementalAttackTable {
    fn from_scratch(pos: &Position) -> Self {
        let mut incremental = Self { attackers: [0; 64], rows: [0; 32], danger: [0; 2], ids: PieceIds::seed(pos) };
        incremental.rebuild(pos);
        incremental
    }

    #[inline(always)]
    fn make(&mut self, pos: &Position, mv: Move) {
        let plan = self.ids.plan_move(mv);
        let occ = pos.occ;
        let mut touched = Bitboard(0);
        let mut candidates = 0u64;

        for i in 0..plan.n_changed {
            candidates |= u64::from(self.attackers[usize::from(plan.changed[i])]);
        }

        candidates &= self.ids.slider_mask();

        for m in 0..plan.n_movers {
            candidates &= !(1u64 << plan.movers[m].0);
        }

        if let Some((victim, _)) = plan.captured {
            candidates &= !(1u64 << victim);
        }

        if let Some((victim, victim_sq)) = plan.captured {
            let row = Bitboard(self.rows[victim]);
            touched |= row;

            for s in row {
                self.attackers[usize::from(s)] &= !((1u64 << victim) as u32);
            }

            self.rows[victim] = 0;
            self.ids.at[usize::from(victim_sq)] = 0;
            self.ids.squares[victim] = 0xFF;
        }

        self.ids.apply(mv, &plan);

        for id in ids_of(candidates) {
            let sq = Square(self.ids.squares[id]);
            let fresh = attacks_of(self.ids.types[id], sq, side_of(id), occ);
            touched |= self.diff_apply(id, fresh);
        }

        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            let fresh = attacks_of(self.ids.types[id], to, side_of(id), occ);
            touched |= self.diff_apply(id, fresh);
        }

        // Two pieces can attack one square, so XOR would drop the other's bit.
        let mut white_danger = 0u64;
        let mut black_danger = 0u64;

        for s in touched {
            let entry = self.attackers[usize::from(s)];
            white_danger |= u64::from((entry & 0x0000FFFF) != 0) << s.0;
            black_danger |= u64::from((entry & 0xFFFF0000) != 0) << s.0;
        }
        self.danger[0] = (self.danger[0] & !touched.0) | white_danger;
        self.danger[1] = (self.danger[1] & !touched.0) | black_danger;
    }

    #[inline(always)]
    fn diff_apply(&mut self, id: usize, fresh: Bitboard) -> Bitboard {
        let diff = Bitboard(self.rows[id]) ^ fresh;
        let bit = (1u64 << id) as u32;

        for s in diff {
            self.attackers[usize::from(s)] ^= bit;
        }

        self.rows[id] = fresh.0;
        diff
    }

    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occ;
        self.attackers = [0; 64];
        self.rows = [0; 32];
        self.danger = [0; 2];

        for id in 0..32 {
            let sq = self.ids.squares[id];
            if sq == 0xFF {
                continue;
            }

            let attacks = attacks_of(self.ids.types[id], Square(sq), side_of(id), occ);
            self.rows[id] = attacks.0;

            for s in attacks {
                self.attackers[usize::from(s)] |= (1u64 << id) as u32;
            }
        }

        for sq in 0..64 {
            let entry = self.attackers[sq];
            self.danger[0] |= u64::from((entry & 0x0000FFFF) != 0) << sq;
            self.danger[1] |= u64::from((entry & 0xFFFF0000) != 0) << sq;
        }
    }
}

impl XorBoard {
    fn from_scratch(pos: &Position) -> Self {
        let mut board = Self { rows: [0; 32], xray: [0; 32], vision: 0, ids: PieceIds::seed(pos) };
        board.rebuild(pos);
        board
    }

    #[inline(always)]
    fn make<const G: Gather, const VISION: bool, const XRAY: bool, const RECORD: bool>(
        &mut self,
        pre: PreMove,
        occ: Bitboard,
        mv: Move,
        undo: &mut Undo,
    ) {
        let plan = self.ids.plan_move(mv);

        let mut changed = Bitboard(0);
        for i in 0..plan.n_changed {
            changed |= plan.changed[i].bitboard();
        }

        let mut candidates = if G == Gather::None || (VISION && (Bitboard(self.vision) & changed).is_empty()) {
            0
        } else if G == Gather::Probe {
            let mut orth = Bitboard(0);
            let mut diag = Bitboard(0);

            for i in 0..plan.n_changed {
                let sq = plan.changed[i];
                let rook_rays = atk_rook(sq, pre.occ);
                let bishop_rays = atk_bishop(sq, pre.occ);
                orth |= rook_rays;
                diag |= bishop_rays;

                // Remove the first blocker to reach the slider behind it.
                if XRAY {
                    orth |= atk_rook(sq, pre.occ & !rook_rays);
                    diag |= atk_bishop(sq, pre.occ & !bishop_rays);
                }
            }

            let mut hits = 0u64;

            for sq in (orth & pre.rq_squares) | (diag & pre.bq_squares) {
                hits |= 1u64 << (self.ids.at[usize::from(sq)] - 1);
            }
            hits
        } else {
            let white_groups = usize::from(self.ids.slider_groups[0]);
            let black_groups = usize::from(self.ids.slider_groups[1]);

            let mut hits = match G {
                Gather::Prefix => u64::from(column_prefix(&self.rows, changed, white_groups, black_groups)),
                Gather::Fixed => u64::from(column_fixed(&self.rows, changed, white_groups, black_groups)),
                _ => u64::from(column(&self.rows, changed)),
            };

            if XRAY {
                hits |= u64::from(match G {
                    Gather::Prefix => column_prefix(&self.xray, changed, white_groups, black_groups),
                    Gather::Fixed => column_fixed(&self.xray, changed, white_groups, black_groups),
                    _ => column(&self.xray, changed),
                });
            }

            hits & self.ids.slider_mask()
        };

        // Movers and victim are updated below; probing them would repeat the work.
        for m in 0..plan.n_movers {
            candidates &= !(1u64 << plan.movers[m].0);
        }

        if let Some((victim, _)) = plan.captured {
            candidates &= !(1u64 << victim);
        }

        if RECORD {
            undo.len = 0;
            undo.vision = self.vision;
            undo.slider_groups = self.ids.slider_groups;
        }

        let mut moved_slider = false;

        if let Some((victim, victim_sq)) = plan.captured {
            if RECORD {
                undo.push(victim, self.rows[victim], self.xray[victim]);
            }

            moved_slider |= is_slider(self.ids.types[victim]);
            self.rows[victim] = 0;
            self.xray[victim] = 0;
            self.ids.at[usize::from(victim_sq)] = 0;
            self.ids.squares[victim] = 0xFF;
        }

        self.ids.apply(mv, &plan);

        for id in ids_of(candidates) {
            self.replace::<XRAY, RECORD>(id, Square(self.ids.squares[id]), occ, undo);
            moved_slider = true;
        }

        for m in 0..plan.n_movers {
            let (id, _, to) = plan.movers[m];
            self.replace::<XRAY, RECORD>(id, to, occ, undo);
            moved_slider |= is_slider(self.ids.types[id]);
        }

        if VISION && moved_slider {
            self.vision = self.slider_vision();
        }
    }

    #[inline(always)]
    fn slider_vision(&self) -> u64 {
        let white_groups = usize::from(self.ids.slider_groups[0]);
        let black_groups = usize::from(self.ids.slider_groups[1]);

        // SAFETY: AVX2 guaranteed. The loads read four u64s from
        // group g of a 32-element array, and g never exceeds 7.
        unsafe {
            let mut acc = _mm256_setzero_si256();

            for g in (0..white_groups).chain(4..4 + black_groups) {
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.rows.as_ptr().add(g * 4).cast()));
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.xray.as_ptr().add(g * 4).cast()));
            }

            let folded = _mm_or_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            (_mm_extract_epi64::<0>(folded) | _mm_extract_epi64::<1>(folded)).cast_unsigned()
        }
    }

    #[inline(always)]
    fn replace<const XRAY: bool, const RECORD: bool>(&mut self, id: usize, sq: Square, occ: Bitboard, undo: &mut Undo) {
        if RECORD {
            undo.push(id, self.rows[id], self.xray[id]);
        }

        if XRAY {
            let (direct, behind) = planes_of(self.ids.types[id], sq, side_of(id), occ);
            self.rows[id] = direct.0;
            self.xray[id] = behind.0;
        } else {
            self.rows[id] = attacks_of(self.ids.types[id], sq, side_of(id), occ).0;
        }
    }

    #[inline(always)]
    fn unmake<const XRAY: bool>(&mut self, mv: Move, undo: &Undo) {
        for i in 0..undo.len {
            let id = usize::from(undo.ids[i]);
            self.rows[id] = undo.rows[i];

            if XRAY {
                self.xray[id] = undo.xray[i];
            }
        }

        // A capture records the victim before any candidate or mover.
        let victim = mv.is_capture().then(|| usize::from(undo.ids[0]));
        let plan = self.ids.plan_unmove(mv, victim);
        self.ids.undo(mv, &plan, undo.slider_groups);
        self.vision = undo.vision;
    }

    #[inline(always)]
    fn danger(&self, side: usize) -> u64 {
        let mut acc = 0u64;
        for i in 0..16 {
            acc |= self.rows[side * 16 + i];
        }
        acc
    }

    #[inline(always)]
    fn pinned(&self, pos: &Position, color: Color) -> Bitboard {
        let ksq = pos.pieces(PieceType::King, color).lsb();
        let us = pos.side_bb[color];
        let enemy = 0xFFFF_0000u64 >> (usize::from(color) * 16);
        // A pinner is blocked by the piece it pins, so it shows up in the x-ray plane only.
        let candidates = u64::from(column(&self.xray, ksq.bitboard())) & self.ids.slider_mask() & enemy;
        let mut pinned = Bitboard(0);

        for id in ids_of(candidates) {
            let sq = Square(self.ids.squares[id]);
            pinned |= Bitboard(self.rows[id]) & between_bb(sq, ksq) & us;
        }

        pinned
    }

    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occ;
        self.rows = [0; 32];
        self.xray = [0; 32];

        for id in 0..32 {
            let sq = self.ids.squares[id];
            if sq == 0xFF {
                continue;
            }

            let (direct, behind) = planes_of(self.ids.types[id], Square(sq), side_of(id), occ);
            self.rows[id] = direct.0;
            self.xray[id] = behind.0;
        }

        self.vision = self.slider_vision();
    }
}

impl Undo {
    fn empty() -> Self { Self { len: 0, ids: [0; 24], rows: [0; 24], xray: [0; 24], vision: 0, slider_groups: [0; 2] } }

    #[inline(always)]
    fn push(&mut self, id: usize, row: u64, xray: u64) {
        debug_assert!(self.len < 24, "affected set outgrew twenty sliders plus two movers plus a victim");
        self.ids[self.len] = id as u8;
        self.rows[self.len] = row;
        self.xray[self.len] = xray;
        self.len += 1;
    }
}

impl PreMove {
    const ZERO: Self = Self { occ: Bitboard(0), rq_squares: Bitboard(0), bq_squares: Bitboard(0) };

    #[inline(always)]
    fn of(pos: &Position) -> Self {
        Self {
            occ: pos.occ,
            rq_squares: pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen],
            bq_squares: pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen],
        }
    }
}

#[inline(always)]
fn column(rows: &[u64; 32], mask: Bitboard) -> u32 {
    // SAFETY: AVX2 is guaranteed, AVX-512 under its own cfg. The AVX-512 path
    // loads groups 0..4 and the AVX2 path groups 0..8, all of a 32-element array.
    unsafe {
        #[cfg(target_feature = "avx512f")]
        {
            let m = _mm512_set1_epi64(mask.0.cast_signed());
            let mut live = 0u32;

            for g in 0..4 {
                let v = _mm512_loadu_si512(rows.as_ptr().add(g * 8).cast());
                live |= u32::from(_mm512_test_epi64_mask(v, m)) << (g * 8);
            }

            live
        }

        #[cfg(not(target_feature = "avx512f"))]
        {
            let m = _mm256_set1_epi64x(mask.0.cast_signed());
            let zero = _mm256_setzero_si256();
            let mut live = 0u32;

            for g in 0..8 {
                let v = _mm256_loadu_si256(rows.as_ptr().add(g * 4).cast());
                let miss = _mm256_cmpeq_epi64(_mm256_and_si256(v, m), zero);
                live |= (!_mm256_movemask_pd(_mm256_castsi256_pd(miss)).cast_unsigned() & 0xF) << (g * 4);
            }
            live
        }
    }
}

#[inline(always)]
fn column_fixed(rows: &[u64; 32], mask: Bitboard, white_groups: usize, black_groups: usize) -> u32 {
    // A promotion can push a color past two groups; fall back to the full sweep.
    if white_groups > 2 || black_groups > 2 {
        return column(rows, mask);
    }

    // SAFETY: as column; the four group indices are constants under 8.
    unsafe {
        let m = _mm256_set1_epi64x(mask.0.cast_signed());
        let zero = _mm256_setzero_si256();
        let mut live = 0u32;

        for g in [0, 1, 4, 5] {
            let v = _mm256_loadu_si256(rows.as_ptr().add(g * 4).cast());
            let miss = _mm256_cmpeq_epi64(_mm256_and_si256(v, m), zero);
            live |= (!_mm256_movemask_pd(_mm256_castsi256_pd(miss)).cast_unsigned() & 0xF) << (g * 4);
        }

        live
    }
}

#[inline(always)]
fn column_prefix(rows: &[u64; 32], mask: Bitboard, white_groups: usize, black_groups: usize) -> u32 {
    // SAFETY: as column; g never exceeds 7.
    unsafe {
        let m = _mm256_set1_epi64x(mask.0.cast_signed());
        let zero = _mm256_setzero_si256();
        let mut live = 0u32;

        for g in (0..white_groups).chain(4..4 + black_groups) {
            let v = _mm256_loadu_si256(rows.as_ptr().add(g * 4).cast());
            let miss = _mm256_cmpeq_epi64(_mm256_and_si256(v, m), zero);
            live |= (!_mm256_movemask_pd(_mm256_castsi256_pd(miss)).cast_unsigned() & 0xF) << (g * 4);
        }

        live
    }
}

struct Stores {
    incremental: IncrementalAttackTable,
    xorboard: XorBoard,
    byteboard: ByteStore,
}

// Build every store so setup cost lands in the baseline and cancels.
fn setup(fen: &str) -> (Position, I16x8, Stores) {
    let pos = Position::from_fen(fen);
    let acc = pos.psqt_accumulator();
    let incremental = black_box(IncrementalAttackTable::from_scratch(&pos));
    let xorboard = black_box(XorBoard::from_scratch(&pos));
    let byte_store = black_box(ByteStore::from_scratch(&pos));
    (pos, acc, Stores { incremental, xorboard, byteboard: byte_store })
}

fn play_stores<F>(stream: &[Game], repeats: usize, variant: &'static str, mut step: F) -> Outcome
where F: FnMut(&mut Stores, &mut Position, &mut I16x8, Move, u64) -> u64 {
    let mut checksum = 0u64;
    let mut iterations = 0u64;

    for _ in 0..repeats {
        for game in stream {
            let (mut pos, mut acc, mut stores) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                checksum ^= step(&mut stores, &mut pos, &mut acc, mv, iterations);
                iterations += 1;
            }
        }
    }

    Outcome { variant, iterations, checksum }
}

fn play_stream<const AFTER_MOVE: bool, F>(stream: &[Game], repeats: usize, mut per_ply: F) -> Outcome
where F: FnMut(&Position, Move, PieceType) -> u64 {
    play_stores(stream, repeats, "", |_, pos, acc, mv, _| {
        let pt = pos.expect_piece_at(mv.from());
        let mut sum = 0;

        if !AFTER_MOVE {
            sum ^= per_ply(pos, mv, pt);
        }

        pos.make_move(mv, acc);

        if AFTER_MOVE {
            sum ^= per_ply(pos, mv, pt);
        }

        sum
    })
}

#[derive(PartialEq, Eq, ConstParamTy, Clone, Copy)]
enum Gather {
    None,
    Column,
    Prefix,
    Fixed,
    Probe,
}

#[derive(PartialEq, Eq, ConstParamTy, Clone, Copy)]
enum View {
    None,
    FusedDanger,
    LaggedDanger,
    LaggedStmDanger,
    LaggedPins,
}

fn variant_baseline(stream: &[Game], repeats: usize) -> Outcome { play_stream::<true, _>(stream, repeats, |_, _, _| 0) }

fn variant_dest(stream: &[Game], repeats: usize) -> Outcome {
    play_stores(stream, repeats, "dest", |stores, pos, acc, mv, _| {
        pos.make_move(mv, acc);
        stores.incremental.make(pos, mv);
        stores.incremental.rows[0] | stores.incremental.rows[31] | stores.incremental.danger[0] | stores.incremental.danger[1]
    })
}

fn variant_hosting(stream: &[Game], repeats: usize) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    let mut incremental = IncrementalAttackTable::from_scratch(&Position::from_fen(&stream[0].fen));

    for i in 0..repeats * stream.iter().map(|g| g.moves.len()).sum::<usize>() {
        let saved = black_box(incremental);
        incremental = black_box(saved);
        checksum ^= incremental.rows[i % 32];
        iterations += 1;
    }

    Outcome { variant: "hosting", iterations, checksum }
}

fn variant_xorboard<const G: Gather, const VISION: bool, const XRAY: bool, const VIEW: View, const UNDO: bool>(
    stream: &[Game],
    repeats: usize,
    name: &'static str,
) -> Outcome {
    let mut undo = Undo::empty();

    play_stores(stream, repeats, name, |stores, pos, acc, mv, _| {
        let pre = if G == Gather::Probe { PreMove::of(pos) } else { PreMove::ZERO };

        let mut sum = match VIEW {
            View::LaggedDanger => stores.xorboard.danger(0) | stores.xorboard.danger(1),
            View::LaggedStmDanger => stores.xorboard.danger(usize::from(pos.stm.opposite())),
            View::LaggedPins => (stores.xorboard.pinned(pos, Color::White) | stores.xorboard.pinned(pos, Color::Black)).0,
            _ => 0,
        };

        pos.make_move(mv, acc);
        stores.xorboard.make::<G, VISION, XRAY, UNDO>(pre, pos.occ, mv, &mut undo);

        sum ^= match VIEW {
            View::FusedDanger => stores.xorboard.danger(0) | stores.xorboard.danger(1),
            _ => stores.xorboard.rows[0] | stores.xorboard.rows[31],
        };

        if UNDO {
            stores.xorboard.unmake::<XRAY>(mv, &undo);
            stores.xorboard.make::<G, VISION, XRAY, false>(pre, pos.occ, mv, &mut undo);
        }

        sum
    })
}

fn variant_xorboard_copy(stream: &[Game], repeats: usize) -> Outcome {
    let mut undo = Undo::empty();

    play_stores(stream, repeats, "xorboard_copy", |stores, pos, acc, mv, _| {
        pos.make_move(mv, acc);

        let parent = stores.xorboard;
        stores
            .xorboard
            .make::<{ Gather::Column }, false, false, false>(PreMove::ZERO, pos.occ, mv, &mut undo);
        hint::black_box(&parent);

        stores.xorboard.rows[0] | stores.xorboard.rows[31]
    })
}

fn variant_byteboard_copy(stream: &[Game], repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();

    play_stores(stream, repeats, "byteboard_copy", |stores, pos, acc, mv, _| {
        pos.make_move(mv, acc);

        let parent = stores.byteboard.board;
        stores.byteboard.make(&g, mv);
        hint::black_box(&parent);

        u64::from(byteboard_attackers(&stores.byteboard.board, 0, 0) | byteboard_attackers(&stores.byteboard.board, 1, 63))
    })
}

fn variant_byteboard_reads(stream: &[Game], repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();

    play_stores(stream, repeats, "byteboard_reads", |stores, pos, acc, mv, iterations| {
        pos.make_move(mv, acc);
        stores.byteboard.make(&g, mv);

        let (sq, slot) = probe_points(iterations);
        let attacks = stores.byteboard.board.attack_table(pos.stm);
        let attackers = attacks.read(sq);
        let mut sum = u64::from(attackers.bits());

        sum ^= attackers.fold(0u64, |walked, id| walked | 1 << id.raw());
        sum ^= attacks.bitboard_any();
        sum ^= attacks.bitboard_for(slot.to_piece_mask());
        sum
    })
}

fn variant_xorboard_reads(stream: &[Game], repeats: usize) -> Outcome {
    let mut undo = Undo::empty();

    play_stores(stream, repeats, "xorboard_reads", |stores, pos, acc, mv, iterations| {
        let pre = PreMove::ZERO;
        pos.make_move(mv, acc);
        stores
            .xorboard
            .make::<{ Gather::Column }, false, false, false>(pre, pos.occ, mv, &mut undo);

        let (sq, slot) = probe_points(iterations);
        let side = usize::from(pos.stm);

        let mut attackers = xorboard_attackers(&stores.xorboard, side, usize::from(sq));
        let mut sum = u64::from(attackers);

        while attackers != 0 {
            sum ^= 1 << attackers.trailing_zeros();
            attackers &= attackers - 1;
        }

        sum ^= stores.xorboard.danger(side);
        sum ^= stores.xorboard.rows[side * 16 + usize::from(slot.raw())];
        sum
    })
}

fn probe_points(iterations: u64) -> (Square, PieceId) { (Square((iterations % 64) as u8), PieceId::new((iterations % 16) as u8)) }

fn variant_byteboard(stream: &[Game], repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();

    play_stores(stream, repeats, "byteboard", |stores, pos, acc, mv, _| {
        pos.make_move(mv, acc);
        stores.byteboard.make(&g, mv);
        u64::from(byteboard_attackers(&stores.byteboard.board, 0, 0) | byteboard_attackers(&stores.byteboard.board, 1, 63))
    })
}

fn run_variant(name: &str, stream: &[Game], repeats: usize) -> Outcome {
    match name {
        "baseline" => variant_baseline(stream, repeats),
        "dest" => variant_dest(stream, repeats),
        "byteboard" => variant_byteboard(stream, repeats),
        "byteboard_reads" => variant_byteboard_reads(stream, repeats),
        "byteboard_copy" => variant_byteboard_copy(stream, repeats),
        "xorboard_copy" => variant_xorboard_copy(stream, repeats),
        "xorboard_reads" => variant_xorboard_reads(stream, repeats),
        "hosting" => variant_hosting(stream, repeats),
        "xorboard" => variant_xorboard::<{ Gather::Column }, false, false, { View::None }, false>(stream, repeats, "xorboard"),
        "xorboard_probe" => {
            variant_xorboard::<{ Gather::Probe }, false, false, { View::None }, false>(stream, repeats, "xorboard_probe")
        },
        "xorboard_fused" => {
            variant_xorboard::<{ Gather::Prefix }, true, false, { View::FusedDanger }, false>(stream, repeats, "xorboard_fused")
        },
        "xorboard_lag" => {
            variant_xorboard::<{ Gather::Prefix }, true, false, { View::LaggedDanger }, false>(stream, repeats, "xorboard_lag")
        },
        "xorboard_lag1" => {
            variant_xorboard::<{ Gather::Column }, false, false, { View::LaggedStmDanger }, false>(stream, repeats, "xorboard_lag1")
        },
        "xorboard_undo" => {
            variant_xorboard::<{ Gather::Column }, false, false, { View::None }, true>(stream, repeats, "xorboard_undo")
        },
        "xray" => variant_xorboard::<{ Gather::Prefix }, true, true, { View::None }, false>(stream, repeats, "xray"),
        "xray_pins" => {
            variant_xorboard::<{ Gather::Prefix }, true, true, { View::LaggedPins }, false>(stream, repeats, "xray_pins")
        },
        "xorboard_vision" => {
            variant_xorboard::<{ Gather::Column }, true, false, { View::None }, false>(stream, repeats, "xorboard_vision")
        },
        // The prefix gather with vision maintained and never read.
        "xorboard_pack" => {
            variant_xorboard::<{ Gather::Prefix }, true, false, { View::None }, false>(stream, repeats, "xorboard_pack")
        },
        "xorboard_prefix" => {
            variant_xorboard::<{ Gather::Prefix }, false, false, { View::None }, false>(stream, repeats, "xorboard_prefix")
        },
        "xorboard_fixed" => {
            variant_xorboard::<{ Gather::Fixed }, false, false, { View::None }, false>(stream, repeats, "xorboard_fixed")
        },
        "xorboard_nogather" => {
            variant_xorboard::<{ Gather::None }, false, false, { View::None }, false>(stream, repeats, "xorboard_nogather")
        },
        "xray_undo" => variant_xorboard::<{ Gather::Prefix }, true, true, { View::None }, true>(stream, repeats, "xray_undo"),
        "threats" => play_stream::<true, _>(stream, repeats, |pos, _, _| pos.threats(pos.stm.opposite()).0),
        "pins" => play_stream::<true, _>(stream, repeats, |pos, _, _| {
            let pins = crate::core::board::attacks::Pins::new(pos);
            pins.pinned(Color::White).0 ^ pins.pinned(Color::Black).0
        }),
        "checkers" => play_stream::<true, _>(stream, repeats, |pos, _, _| pos.checkers().0),
        "new_threats" => play_stream::<false, _>(stream, repeats, |pos, mv, pt| pos.new_threats(pt, mv.from(), mv.to()).0),
        "eval_attack" => play_stream::<true, _>(stream, repeats, |pos, _, _| {
            let pinned_w = pos.pinned_pieces(Color::White);
            let pinned_b = pos.pinned_pieces(Color::Black);
            let maps = SliderMaps::compute(pos, pinned_w, pinned_b);
            let mob = mobility::compute_all(pos, &maps, pinned_w, pinned_b, None);
            (mob.metrics_us.mobility ^ mob.metrics_them.mobility) as u64 ^ (mob.safety_us.weak ^ mob.safety_them.weak) as u64
        }),
        "see" => play_stream::<false, _>(stream, repeats, |pos, mv, _| {
            let pins = crate::core::board::attacks::Pins::new(pos);
            u64::from(see_ge(pos, mv, 0, &pins, &SEE_VALUES))
        }),
        "king_legal" => play_stream::<false, _>(stream, repeats, |pos, mv, _| {
            let opp = pos.stm.opposite();
            u64::from(pos.is_attacked(mv.to(), opp, mv.from().bitboard()))
        }),
        _ => {
            eprintln!("atb: unknown variant '{name}'");
            std::process::exit(1);
        },
    }
}

fn validate(stream: &[Game]) -> Option<u64> {
    let mut checked = 0u64;

    for (game_index, game) in stream.iter().enumerate() {
        let mut pos = Position::from_fen(&game.fen);
        let mut acc = pos.psqt_accumulator();
        let mut incremental = IncrementalAttackTable::from_scratch(&pos);
        let mut xorboard = XorBoard::from_scratch(&pos);
        let mut probe = XorBoard::from_scratch(&pos);
        let mut undo = Undo::empty();
        let mut byte_store = ByteStore::from_scratch(&pos);

        let g = byteboard::Geometry::new();

        for (ply, &mv) in game.moves.iter().enumerate() {
            let incremental_before = incremental;
            let xorboard_before = xorboard;
            let plan = incremental.ids.plan_move(mv);

            let stale = plan.movers[0].0 >= 32
                || (plan.n_movers == 2 && plan.movers[1].0 >= 32)
                || plan.captured.is_some_and(|(victim, _)| victim >= 32);

            if stale {
                eprintln!("atb: stale id map, game {game_index}, ply {ply}, move {}", mv.inner());
                eprintln!("fen before: {}", pos.to_fen());
                for m in 0..plan.n_movers {
                    let (id, from, to) = plan.movers[m];
                    eprintln!("  mover[{m}] id={} from={} to={}", id, from.0, to.0);
                }
                eprintln!("  captured: {:?}", plan.captured);
                eprintln!(
                    "  at[from]={} at[to]={} real at from={}",
                    incremental.ids.at[usize::from(mv.from())],
                    incremental.ids.at[usize::from(mv.to())],
                    pos.piece_at(mv.from()) as u8
                );

                return None;
            }

            let pre = PreMove::of(&pos);
            pos.make_move(mv, &mut acc);
            incremental.make(&pos, mv);
            xorboard.make::<{ Gather::Prefix }, true, true, true>(pre, pos.occ, mv, &mut undo);
            byte_store.make(&g, mv);
            probe.make::<{ Gather::Probe }, true, true, false>(pre, pos.occ, mv, &mut undo);

            let mut want = incremental_before;
            if let Some((victim, victim_sq)) = plan.captured {
                want.ids.at[usize::from(victim_sq)] = 0;
                want.ids.squares[victim] = 0xFF;
            }

            want.ids.apply(mv, &plan);
            want.rebuild(&pos);

            let mut id_map_ok = true;
            for sq in 0..64 {
                let real = pos.piece_at(Square(sq as u8)) != PieceType::None;
                if real != (incremental.ids.at[sq] != 0) {
                    if id_map_ok {
                        eprintln!("atb: id map vs position, game {game_index}, ply {ply}, move {}", mv.inner());
                        eprintln!("fen: {}", pos.to_fen());
                        id_map_ok = false;
                    }
                    eprintln!("  sq {sq}: at={} real={}", incremental.ids.at[sq], pos.piece_at(Square(sq as u8)) as u8);
                }
            }

            if !id_map_ok {
                return None;
            }

            if incremental != want {
                eprintln!("atb: attack table mismatch, game {game_index}, move {}", mv.inner());
                eprintln!("fen: {}", pos.to_fen());

                for (i, (a, b)) in incremental.attackers.iter().zip(want.attackers.iter()).enumerate() {
                    if a != b {
                        eprintln!("  attackers[{i}]: got {a:#010x} want {b:#010x}");
                    }
                }
                for (i, (a, b)) in incremental.rows.iter().zip(want.rows.iter()).enumerate() {
                    if a != b {
                        eprintln!("  rows[{i}]: got {a:#018x} want {b:#018x}");
                    }
                }
                for (i, (a, b)) in incremental.ids.at.iter().zip(want.ids.at.iter()).enumerate() {
                    if a != b {
                        eprintln!("  at[{i}]: got {a:#04x} want {b:#04x}");
                    }
                }
                for (i, (a, b)) in incremental.ids.squares.iter().zip(want.ids.squares.iter()).enumerate() {
                    if a != b {
                        eprintln!("  squares[{i}]: got {a:#04x} want {b:#04x}");
                    }
                }
                for (i, (a, b)) in incremental.ids.types.iter().zip(want.ids.types.iter()).enumerate() {
                    if a != b {
                        eprintln!("  types[{i}]: got {a:?} want {b:?}");
                    }
                }
                for (i, (a, b)) in incremental.ids.type_masks.iter().zip(want.ids.type_masks.iter()).enumerate() {
                    if a != b {
                        eprintln!("  type_masks[{i}]: got {a:#018x} want {b:#018x}");
                    }
                }
                return None;
            }

            let mut xorboard_want = xorboard;
            xorboard_want.rebuild(&pos);
            let mut replayed = xorboard;
            replayed.unmake::<true>(mv, &undo);

            let faults: [(&str, bool); 9] = [
                ("xorboard vs rebuild", xorboard != xorboard_want),
                ("probe gather vs prefix gather", probe != xorboard),
                ("xorboard rows vs attack table rows", xorboard.rows != incremental.rows),
                ("danger(white) vs danger[0]", xorboard.danger(0) != incremental.danger[0]),
                ("danger(black) vs danger[1]", xorboard.danger(1) != incremental.danger[1]),
                ("xray pins(white) vs pinned_pieces", xorboard.pinned(&pos, Color::White) != pos.pinned_pieces(Color::White)),
                ("xray pins(black) vs pinned_pieces", xorboard.pinned(&pos, Color::Black) != pos.pinned_pieces(Color::Black)),
                ("unmake replay", replayed != xorboard_before),
                ("byteboard vs xorboard", !byteboard_agrees(&byte_store, &xorboard)),
            ];

            for (what, failed) in faults {
                if failed {
                    eprintln!("atb: {what}, game {game_index}, ply {ply}, move {}", mv.inner());
                    eprintln!("fen: {}", pos.to_fen());
                    eprintln!("record held {} rows", undo.len);

                    for id in 0..32 {
                        let drift = xorboard.rows[id] != xorboard_want.rows[id]
                            || xorboard.xray[id] != xorboard_want.xray[id]
                            || probe.rows[id] != xorboard.rows[id]
                            || xorboard.rows[id] != incremental.rows[id]
                            || replayed.rows[id] != xorboard_before.rows[id];

                        if drift {
                            eprintln!(
                                "  {id}: row {:#018x} rebuild {:#018x} table {:#018x} replay {:#018x} want {:#018x}",
                                xorboard.rows[id],
                                xorboard_want.rows[id],
                                incremental.rows[id],
                                replayed.rows[id],
                                xorboard_before.rows[id]
                            );
                            eprintln!("      xray {:#018x} rebuild {:#018x}", xorboard.xray[id], xorboard_want.xray[id]);
                        }
                    }

                    for sq in 0..64 {
                        for side in 0..2 {
                            let want = xorboard_attackers(&xorboard, side, sq);
                            let got = byteboard_attackers(&byte_store.board, side, sq);

                            if got != want {
                                eprintln!("  sq {sq} side {side}: byteboard {got:#06x} want {want:#06x}");
                            }
                        }
                    }

                    for color in [Color::White, Color::Black] {
                        eprintln!(
                            "  pins {color:?}: got {:#018x} want {:#018x}",
                            xorboard.pinned(&pos, color).0,
                            pos.pinned_pieces(color).0
                        );
                    }
                    return None;
                }
            }
            checked += 1;
        }
    }
    Some(checked)
}

fn byteboard_agrees(byte_store: &ByteStore, xorboard: &XorBoard) -> bool {
    (0..64).all(|sq| (0..2).all(|side| byteboard_attackers(&byte_store.board, side, sq) == xorboard_attackers(xorboard, side, sq)))
}

fn byteboard_attackers(board: &byteboard::ByteBoard, side: usize, sq: usize) -> u16 {
    let color = if side == 0 { Color::White } else { Color::Black };
    board.attack_table(color).read(Square(sq as u8)).bits()
}

#[inline]
fn xorboard_attackers(xorboard: &XorBoard, side: usize, sq: usize) -> u16 {
    let mut attackers = 0u16;
    for slot in 0..16 {
        attackers |= ((xorboard.rows[side * 16 + slot] >> sq & 1) as u16) << slot;
    }
    attackers
}

fn count_writes(stream: &[Game]) {
    let (mut moves, mut pairs, mut rows, mut danger_flips) = (0u64, 0u64, 0u64, 0u64);
    let mut worst = 0u32;

    for game in stream {
        let mut pos = Position::from_fen(&game.fen);
        let mut acc = pos.psqt_accumulator();
        let mut incremental = IncrementalAttackTable::from_scratch(&pos);
        let mut xorboard = XorBoard::from_scratch(&pos);
        let mut undo = Undo::empty();

        for &mv in &game.moves {
            let incremental_before = incremental;
            let pre = PreMove::of(&pos);
            pos.make_move(mv, &mut acc);
            incremental.make(&pos, mv);
            xorboard.make::<{ Gather::Prefix }, true, false, true>(pre, pos.occ, mv, &mut undo);

            let mut delta = 0u32;
            for id in 0..32 {
                delta += (incremental_before.rows[id] ^ incremental.rows[id]).count_ones();
            }

            moves += 1;
            pairs += u64::from(delta);
            rows += undo.len as u64;
            danger_flips += u64::from(
                (incremental_before.danger[0] ^ incremental.danger[0] | incremental_before.danger[1] ^ incremental.danger[1])
                    .count_ones(),
            );
            worst = worst.max(delta);
        }
    }

    let move_count = moves as f64;
    println!("moves {moves}");
    println!("pairs/move {:.2} (worst {worst})", pairs as f64 / move_count);
    println!("rows/move  {:.2}", rows as f64 / move_count);
    println!("danger bits flipped/move {:.2}", danger_flips as f64 / move_count);
}

pub fn run(args: &[&str]) {
    match args.first().copied() {
        Some("count") => {
            let path = args.get(1).copied().expect("atb count <path>");
            count_writes(&deserialize(&fs::read(path).expect("read stream")));
        },
        Some("gen") => {
            let path = args.get(1).copied().expect("atb gen <path> [ply_cap]");
            let cap = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(PLY_CAP);
            let stream = generate(0xC0FFEE, cap);
            fs::write(path, serialize(&stream)).expect("write stream");
            let plies: usize = stream.iter().map(|g| g.moves.len()).sum();
            println!("atb gen {} games, {plies} plies -> {path}", stream.len());
        },
        Some("export") => {
            let path = args.get(1).copied().expect("atb export <stream> <out>");
            let out = args.get(2).copied().expect("atb export <stream> <out>");
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let mut text = String::new();
            for game in &stream {
                text.push_str(&game.fen);
                text.push('\n');

                for (i, mv) in game.moves.iter().enumerate() {
                    if i > 0 {
                        text.push(' ');
                    }
                    text.push_str(&mv.to_uci(false));
                }
                text.push('\n');
            }
            fs::write(out, text).expect("write export");
            println!("atb export {} games -> {out}", stream.len());
        },
        Some("validate") => {
            let path = args.get(1).copied().expect("atb validate <path>");
            let stream = deserialize(&fs::read(path).expect("read stream"));

            match validate(&stream) {
                Some(plies) => println!("atb validate OK, {plies} plies"),
                None => std::process::exit(1),
            }
        },
        Some("dump") => {
            let path = args.get(1).copied().expect("atb dump <path> [plies]");
            let n = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let game = &stream[0];
            for (ply, &mv) in game.moves.iter().take(n).enumerate() {
                let mut pos = Position::from_fen(&game.fen);
                let mut acc = pos.psqt_accumulator();

                for &prior in game.moves.iter().take(ply) {
                    pos.make_move(prior, &mut acc);
                }

                let pre_stm = pos.stm;
                let pre_fen = pos.to_fen();
                let info = pos.make_move(mv, &mut acc);
                println!("ply {ply}: {} ({}) pre_stm={} pre_fen={}", mv.to_uci(false), mv.inner(), pre_stm as u8, pre_fen);
                println!("  post: {}", pos.to_fen());
                pos.unmake_move(mv, &info);
                let restored = pos.to_fen();
                println!("  restored: {restored}  (match={})", restored == pre_fen);
            }
        },
        Some("run") => {
            let name = args.get(1).copied().expect("atb run <variant> <path> [repeats]");
            let path = args.get(2).copied().expect("atb run <variant> <path> [repeats]");
            let repeats = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let outcome = run_variant(name, &stream, repeats);
            // Label here so the name stays out of the measured loop.
            let label = if outcome.variant.is_empty() { name } else { outcome.variant };
            println!("atb {} {} {}", label, outcome.iterations, outcome.checksum);
        },
        _ => {
            eprintln!("atb: expected 'gen', 'export', 'validate', 'count', 'dump', or 'run'");
            std::process::exit(1);
        },
    }
}

#[cfg(test)]
#[test]
fn move_queries_use_pre_move_position() {
    let fen = "3rk3/8/2r5/8/3P4/8/8/4K3 w - - 0 1";
    let pos = Position::from_fen(fen);
    let stream = [Game { fen: fen.to_owned(), moves: vec![Move::new(Square(27), Square(35), Move::QUIET)] }];

    for (variant, result) in [("new_threats", Square(42).bitboard().0), ("see", 0), ("king_legal", 1)] {
        assert_eq!(run_variant(variant, &stream, 1).checksum, pos.hash ^ result, "{variant}");
    }
}
