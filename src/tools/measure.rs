//! Instruction-level cost measurement for the XorBoard design.
//!
//! Usage:
//!     soul measure gen <path> [ply_cap]
//!     soul measure export <path> <out>
//!     soul measure validate <path>
//!     soul measure count <path>
//!     soul measure dump <path> [plies]
//!     perf stat -e instructions,cycles soul measure run <variant> <path> <repeats>

use core::{arch::x86_64::*, hint::black_box, marker::ConstParamTy};
use std::{fs, hint};

use crate::{
    core::{
        board::{
            Position,
            bitboard::{atk_bishop, atk_king, atk_knight, atk_pawn, atk_rook, between_bb},
            castling_targets,
            spatial::SpatialTensor,
        },
        defs::{Bitboard, Color, PieceType, Square},
        moves::Move,
        zobrist::ConstRng,
    },
    engine::{mobility::Mobility, movegen::gen_legal_moves, see::see_ge},
    tools::byteboard::{self, PieceId, Place, Wordboard},
    weave::Vi16x8,
};

const FENS: &str = include_str!("../data/bench.fens");
const PLY_CAP: usize = 256;

struct Game {
    fen: String,
    moves: Vec<Move>,
}

struct Stream {
    games: Vec<Game>,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Ids {
    mailbox: [u8; 64],
    list: [u8; 32],
    lut: [PieceType; 32],
    masks: [u64; 12],
    /// One past the highest slider id in each color half, in groups of four.
    /// Seeding puts sliders first, so the gather tests two short prefixes
    /// instead of all thirty-two rows, and a promotion only pushes the mark of
    /// its own half out.
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
struct AttackTable {
    dest: [u32; 64],
    rows: [u64; 32],
    threat: [u64; 2],
    ids: Ids,
}

/// The same relation transposed: one u64 per piece, and the square-indexed
/// table computed where it is wanted instead of kept in step.
///
/// The index decides the danger view. A square's u32 ORs a side together and
/// forgets which piece set the bit, so `AttackTable` has to recount it; here the
/// contributor is the index, and the view is a reduce-OR over the record.
#[derive(Clone, Copy, Debug)]
struct XorBoard {
    rows: [u64; 32],
    xray: [u64; 32],
    /// Every square any slider attacks. If a move touches none of them, no
    /// unmoved piece's row can have changed and the gather has nothing to find,
    /// which is the common case: 1.56 rows change per move at bench density and
    /// the mover is one of them. Derived, never maintained, so the OR that
    /// defeats a square-indexed summary cannot bite here either.
    vision: u64,
    ids: Ids,
}

/// `vision` is a conservative hint, so a stale-wide value is correct and only
/// costs a gather that finds nothing. It stays out of the comparison.
impl PartialEq for XorBoard {
    fn eq(&self, other: &Self) -> bool {
        self.rows == other.rows && self.xray == other.xray && self.ids == other.ids
    }
}

#[derive(Clone, Copy)]
struct Pre {
    occ: Bitboard,
    rq: Bitboard,
    bq: Bitboard,
}

struct Undo {
    n: usize,
    ids: [u8; 24],
    rows: [u64; 24],
    xray: [u64; 24],
    vision: u64,
    slider_groups: [u8; 2],
}

struct ByteStore {
    bb: byteboard::ByteBoard,
    ids: Ids,
}

struct Outcome {
    variant: &'static str,
    iterations: u64,
    checksum: u64,
}

impl ByteStore {
    fn from_scratch(pos: &Position) -> Self {
        let mut st = Self { bb: byteboard::ByteBoard::empty(), ids: Ids::seed(pos) };
        for id in 0..32 {
            let sq = st.ids.list[id];
            if sq != 0xFF {
                st.bb.put(Square(sq), Place::new(color(id), st.ids.lut[id], slot(id)));
            }
        }
        st.rebuild(pos);
        st
    }

    /// Attack table from first principles, not from the byteboard pipeline, so
    /// it can serve as that pipeline's oracle.
    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occupancy();
        let mut attacks = [Wordboard::EMPTY; 2];

        for id in 0..32 {
            let sq = self.ids.list[id];
            if sq == 0xFF {
                continue;
            }

            for s in attacks_of(self.ids.lut[id], Square(sq), color_of(id), occ) {
                attacks[color_of(id)].insert(s, slot(id));
            }
        }
        self.bb.set_attacks(attacks);
    }

    /// Victim first, then every origin, then every destination. Splitting the
    /// movers that way is what keeps DFRC castling exact when the king comes to
    /// rest on the rook's own origin square.
    #[inline(always)]
    fn make(&mut self, g: &byteboard::Geometry, mv: Move) {
        let plan = self.ids.read_move(mv);
        // A capture replaces one blocker with another, so the lines through the
        // destination never change and the toggle there is skipped on both the
        // victim and the arriving piece.
        let replaced = plan.captured.is_some() && !mv.is_en_passant();

        if let Some((vid, victim_sq)) = plan.captured {
            if !replaced {
                self.bb.remove(g, victim_sq);
            }
            self.ids.mailbox[usize::from(victim_sq)] = 0;
            self.ids.list[vid] = 0xFF;
        }

        for m in 0..plan.n_movers {
            let (_, from, _) = plan.movers[m];
            self.bb.remove(g, from);
        }

        // Stripped after the mover's toggle: that toggle still sees the victim
        // on the board and would hand its bits straight back.
        if let Some((vid, _)) = plan.captured.filter(|_| replaced) {
            self.bb.clear(color(vid), slot(vid));
        }

        self.ids.apply(mv, &plan);

        for m in 0..plan.n_movers {
            let (id, _, dest) = plan.movers[m];
            let p = Place::new(color(id), self.ids.lut[id], slot(id));

            if replaced {
                self.bb.land(g, dest, p);
            } else {
                self.bb.add(g, dest, p);
            }
        }
    }
}

#[inline(always)]
fn color(id: usize) -> Color {
    if id < 16 { Color::White } else { Color::Black }
}

/// A global id's slot within its own side.
fn slot(id: usize) -> PieceId {
    PieceId::new((id & 15) as u8)
}

fn generate(seed: u64, ply_cap: usize) -> Stream {
    let mut rng = ConstRng::new(seed);
    let mut games = Vec::new();

    for fen in FENS.lines() {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.get_initial_accumulator();
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
    Stream { games }
}

/// Counts go out as `u16` and lengths as `u8`, so a stream that outgrew either
/// has to say so here. Silently truncating one writes a file that parses and
/// replays the wrong plies.
fn serialize(stream: &Stream) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend(
        u16::try_from(stream.games.len())
            .expect("too many games for the stream header")
            .to_le_bytes(),
    );

    for game in &stream.games {
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

/// The cursor is the slice: each read splits its own header off the front, so a
/// length and the bytes it governs cannot drift apart, and a truncated file
/// says so instead of indexing off the end.
fn deserialize(bytes: &[u8]) -> Stream {
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
    Stream { games }
}

fn take_u16(rest: &mut &[u8]) -> usize {
    let (head, tail) = rest.split_first_chunk::<2>().expect("stream truncated");
    *rest = tail;
    usize::from(u16::from_le_bytes(*head))
}

impl Ids {
    fn empty() -> Self {
        Self {
            mailbox: [0; 64],
            list: [0xFF; 32],
            lut: [PieceType::None; 32],
            masks: [0; 12],
            slider_groups: [0; 2],
        }
    }

    /// Both stores seed from this, so slot n is the same piece in each and
    /// their rows compare directly.
    fn seed(pos: &Position) -> Self {
        let mut ids = Self::empty();
        let mut w_next = 0usize;
        let mut b_next = 16usize;

        for pass in 0..2 {
            for sq in 0..64 {
                let pt = pos.piece_at(Square(sq as u8));
                if pt == PieceType::None || is_slider(pt) != (pass == 0) {
                    continue;
                }

                let (id, color) = if pos.side_bb[Color::White].0 & (1u64 << sq) != 0 {
                    let id = w_next;
                    w_next += 1;
                    (id, Color::White)
                } else {
                    let id = b_next;
                    b_next += 1;
                    (id, Color::Black)
                };

                ids.mailbox[sq] = (id + 1) as u8;
                ids.list[id] = sq as u8;
                ids.lut[id] = pt;
                ids.masks[pt as usize * 2 + color as usize] |= 1u64 << id;

                if pass == 0 {
                    ids.slider_groups[color as usize] = (id % 16 / 4 + 1) as u8;
                }
            }
        }
        ids
    }

    #[inline(always)]
    fn read_move(&self, mv: Move) -> MovePlan {
        let from = mv.from();
        let to = mv.to();
        let mid = usize::from(self.mailbox[usize::from(from)]) - 1;
        let mut movers = [(0usize, Square(0), Square(0)); 2];
        let mut changed;

        let (n_movers, mut n_changed) = if mv.is_castling() {
            let rid = usize::from(self.mailbox[usize::from(to)]) - 1;
            let (king_to, rook_to) = castling_targets(from, to);
            movers[0] = (mid, from, king_to);
            movers[1] = (rid, to, rook_to);
            changed = [from, king_to, to, rook_to];
            (2, 4)
        } else {
            movers[0] = (mid, from, to);
            changed = [from, to, Square(0), Square(0)];
            (1, 2)
        };

        let captured = if mv.is_castling() {
            None
        } else if mv.is_en_passant() {
            let victim_sq = Square(to.0 ^ 8);
            changed[2] = victim_sq;
            Some((usize::from(self.mailbox[usize::from(victim_sq)]) - 1, victim_sq))
        } else if mv.is_capture() {
            Some((usize::from(self.mailbox[usize::from(to)]) - 1, to))
        } else {
            None
        };

        if mv.is_en_passant() {
            n_changed = 3;
        }
        MovePlan { movers, n_movers, changed, n_changed, captured }
    }

    /// The victim's slot has to be handed in: once its square is overwritten
    /// nothing on the board points at it any more.
    #[inline(always)]
    fn read_unmove(&self, mv: Move, victim: Option<usize>) -> MovePlan {
        let from = mv.from();
        let to = mv.to();
        let mut movers = [(0usize, Square(0), Square(0)); 2];
        let n_movers = if mv.is_castling() {
            let (king_to, rook_to) = castling_targets(from, to);
            movers[0] = (usize::from(self.mailbox[usize::from(king_to)]) - 1, from, king_to);
            movers[1] = (usize::from(self.mailbox[usize::from(rook_to)]) - 1, to, rook_to);
            2
        } else {
            movers[0] = (usize::from(self.mailbox[usize::from(to)]) - 1, from, to);
            1
        };

        let captured = victim.map(|vid| (vid, if mv.is_en_passant() { Square(to.0 ^ 8) } else { to }));
        MovePlan { movers, n_movers, changed: [Square(0); 4], n_changed: 0, captured }
    }

    /// Origins clear before any destination lands, so DFRC castling stays exact
    /// when the king comes to rest on the rook's origin square.
    #[inline(always)]
    fn apply(&mut self, mv: Move, plan: &MovePlan) {
        for m in 0..plan.n_movers {
            self.mailbox[usize::from(plan.movers[m].1)] = 0;
        }

        for m in 0..plan.n_movers {
            self.mailbox[usize::from(plan.movers[m].2)] = (plan.movers[m].0 + 1) as u8;
            self.list[plan.movers[m].0] = plan.movers[m].2.0;
        }

        if mv.is_promotion() {
            let mid = plan.movers[0].0;
            let new_type = mv.promo().unwrap_or(PieceType::Queen);
            let old_type = self.lut[mid];
            let c = color_of(mid);
            self.masks[old_type as usize * 2 + c] &= !(1u64 << mid);
            self.masks[new_type as usize * 2 + c] |= 1u64 << mid;
            self.lut[mid] = new_type;
            self.slider_groups[c] = self.slider_groups[c].max((mid % 16 / 4 + 1) as u8);
        }
    }

    /// The victim lands last. For a capture its square is the mover's
    /// destination, so the later write has to be the one that wins.
    #[inline(always)]
    fn undo(&mut self, mv: Move, plan: &MovePlan, slider_groups: [u8; 2]) {
        self.slider_groups = slider_groups;

        if mv.is_promotion() {
            let mid = plan.movers[0].0;
            let new_type = self.lut[mid];
            let c = color_of(mid);
            self.masks[new_type as usize * 2 + c] &= !(1u64 << mid);
            self.masks[PieceType::Pawn as usize * 2 + c] |= 1u64 << mid;
            self.lut[mid] = PieceType::Pawn;
        }

        for m in 0..plan.n_movers {
            self.mailbox[usize::from(plan.movers[m].2)] = 0;
        }
        for m in 0..plan.n_movers {
            self.mailbox[usize::from(plan.movers[m].1)] = (plan.movers[m].0 + 1) as u8;
            self.list[plan.movers[m].0] = plan.movers[m].1.0;
        }

        if let Some((vid, victim_sq)) = plan.captured {
            self.mailbox[usize::from(victim_sq)] = (vid + 1) as u8;
            self.list[vid] = victim_sq.0;
        }
    }

    #[inline(always)]
    fn sliders(&self) -> u64 {
        self.class(PieceType::Bishop) | self.class(PieceType::Rook) | self.class(PieceType::Queen)
    }

    #[inline(always)]
    fn class(&self, pt: PieceType) -> u64 {
        self.masks[pt as usize * 2] | self.masks[pt as usize * 2 + 1]
    }
}

#[inline(always)]
fn color_of(id: usize) -> usize {
    id >> 4
}

#[inline(always)]
fn is_slider(pt: PieceType) -> bool {
    matches!(pt, PieceType::Bishop | PieceType::Rook | PieceType::Queen)
}

#[inline(always)]
fn attacks_of(pt: PieceType, sq: Square, color: usize, occ: Bitboard) -> Bitboard {
    match pt {
        PieceType::Pawn => atk_pawn(sq, if color == 0 { Color::White } else { Color::Black }),
        PieceType::Knight => atk_knight(sq),
        PieceType::King => atk_king(sq),
        PieceType::Bishop => atk_bishop(sq, occ),
        PieceType::Rook => atk_rook(sq, occ),
        PieceType::Queen => atk_rook(sq, occ) | atk_bishop(sq, occ),
        PieceType::None => Bitboard(0),
    }
}

/// Attacks, and what the piece would reach if everything stopping it stepped
/// aside: lift the first blocker off each ray and fill again.
#[inline(always)]
fn planes_of(pt: PieceType, sq: Square, color: usize, occ: Bitboard) -> (Bitboard, Bitboard) {
    let direct = attacks_of(pt, sq, color, occ);
    if !is_slider(pt) {
        return (direct, Bitboard(0));
    }
    (direct, attacks_of(pt, sq, color, occ & !direct) & !direct)
}

#[inline(always)]
fn bits(mut set: u64) -> impl Iterator<Item = usize> {
    core::iter::from_fn(move || {
        (set != 0).then(|| {
            let id = set.trailing_zeros() as usize;
            set &= set - 1;
            id
        })
    })
}

impl AttackTable {
    fn from_scratch(pos: &Position) -> Self {
        let mut dst = Self { dest: [0; 64], rows: [0; 32], threat: [0; 2], ids: Ids::seed(pos) };
        dst.rebuild(pos);
        dst
    }

    #[inline(always)]
    fn make(&mut self, pos: &Position, mv: Move) {
        let plan = self.ids.read_move(mv);
        let occ = pos.occupancy();
        let mut touched = Bitboard(0);
        let mut cand = 0u64;
        for i in 0..plan.n_changed {
            cand |= u64::from(self.dest[usize::from(plan.changed[i])]);
        }

        cand &= self.ids.sliders();

        for m in 0..plan.n_movers {
            cand &= !(1u64 << plan.movers[m].0);
        }

        if let Some((vid, _)) = plan.captured {
            cand &= !(1u64 << vid);
        }

        if let Some((vid, victim_sq)) = plan.captured {
            let row = Bitboard(self.rows[vid]);
            touched |= row;

            for s in row {
                self.dest[usize::from(s)] &= !((1u64 << vid) as u32);
            }
            self.rows[vid] = 0;
            self.ids.mailbox[usize::from(victim_sq)] = 0;
            self.ids.list[vid] = 0xFF;
        }
        self.ids.apply(mv, &plan);

        for id in bits(cand) {
            let sq = Square(self.ids.list[id]);
            let fresh = attacks_of(self.ids.lut[id], sq, color_of(id), occ);
            touched |= self.diff_apply(id, fresh);
        }

        for m in 0..plan.n_movers {
            let (id, _, dest) = plan.movers[m];
            let fresh = attacks_of(self.ids.lut[id], dest, color_of(id), occ);
            touched |= self.diff_apply(id, fresh);
        }

        // Two pieces of one side can cover the same square, so a toggle would
        // clear a bit the other still owns. Every touched square is re-derived
        // from its dest entry instead.
        let mut t0 = 0u64;
        let mut t1 = 0u64;

        for s in touched {
            let d = self.dest[usize::from(s)];
            t0 |= u64::from((d & 0x0000FFFF) != 0) << s.0;
            t1 |= u64::from((d & 0xFFFF0000) != 0) << s.0;
        }
        self.threat[0] = (self.threat[0] & !touched.0) | t0;
        self.threat[1] = (self.threat[1] & !touched.0) | t1;
    }

    #[inline(always)]
    fn diff_apply(&mut self, id: usize, fresh: Bitboard) -> Bitboard {
        let diff = Bitboard(self.rows[id]) ^ fresh;
        let mask = (1u64 << id) as u32;
        for s in diff {
            self.dest[usize::from(s)] ^= mask;
        }
        self.rows[id] = fresh.0;
        diff
    }

    /// The oracle: same ids, every row recomputed.
    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occupancy();
        self.dest = [0; 64];
        self.rows = [0; 32];
        self.threat = [0; 2];

        for id in 0..32 {
            let sq = self.ids.list[id];
            if sq == 0xFF {
                continue;
            }

            let atk = attacks_of(self.ids.lut[id], Square(sq), color_of(id), occ);
            self.rows[id] = atk.0;
            for s in atk {
                self.dest[usize::from(s)] |= (1u64 << id) as u32;
            }
        }

        for sq in 0..64 {
            let d = self.dest[sq];
            self.threat[0] |= u64::from((d & 0x0000FFFF) != 0) << sq;
            self.threat[1] |= u64::from((d & 0xFFFF0000) != 0) << sq;
        }
    }
}

impl XorBoard {
    fn from_scratch(pos: &Position) -> Self {
        let mut xb = Self { rows: [0; 32], xray: [0; 32], vision: 0, ids: Ids::seed(pos) };
        xb.rebuild(pos);
        xb
    }

    /// Attacks are a function of square and occupancy, so an unmoved piece can
    /// only change where its first blocker did, and any such piece saw a changed
    /// square before the move. The affected set is a theorem, not a scan.
    ///
    /// [`Gather`] picks the finder. `VISION` is independent of it: it gates the
    /// whole gather behind an OR of the slider rows, which has to be recomputed
    /// after any move that touched one. `XRAY` reaches one blocker further,
    /// since a second segment is fixed by the first and second blockers and both
    /// sit inside `rows | xray`. `REC` writes the undo record, which only a
    /// variant pricing hosting needs.
    #[inline(always)]
    fn make<const G: Gather, const VISION: bool, const XRAY: bool, const REC: bool>(
        &mut self,
        pre: Pre,
        occ: Bitboard,
        mv: Move,
        undo: &mut Undo,
    ) {
        let plan = self.ids.read_move(mv);

        let mut changed = Bitboard(0);
        for i in 0..plan.n_changed {
            changed |= plan.changed[i].bitboard();
        }

        // 0 leaves the store wrong on purpose: it is the ablation that prices
        // the search for affected sliders against the work it finds.
        let mut cand = if G == Gather::None || (VISION && (Bitboard(self.vision) & changed).is_empty()) {
            0
        } else if G == Gather::Probe {
            let mut orth = Bitboard(0);
            let mut diag = Bitboard(0);

            for i in 0..plan.n_changed {
                let sq = plan.changed[i];
                let r = atk_rook(sq, pre.occ);
                let b = atk_bishop(sq, pre.occ);
                orth |= r;
                diag |= b;
                // A slider whose second segment ends here is the second piece
                // out along the ray, so the superpiece searches the plane the
                // row does.
                if XRAY {
                    orth |= atk_rook(sq, pre.occ & !r);
                    diag |= atk_bishop(sq, pre.occ & !b);
                }
            }

            // Attack symmetry: the sliders that see a square are the ones
            // standing where a superpiece on that square would strike.
            let mut set = 0u64;
            for sq in (orth & pre.rq) | (diag & pre.bq) {
                set |= 1u64 << (self.ids.mailbox[usize::from(sq)] - 1);
            }
            set
        } else {
            // A row tested against a multi-bit mask answers every square in it
            // at once, so castling's four squares cost one pass.
            let wg = usize::from(self.ids.slider_groups[0]);
            let bg = usize::from(self.ids.slider_groups[1]);

            let mut set = match G {
                Gather::Prefix => u64::from(column_prefix(&self.rows, changed, wg, bg)),
                Gather::Fixed => u64::from(column_fixed(&self.rows, changed, wg, bg)),
                _ => u64::from(column(&self.rows, changed)),
            };

            if XRAY {
                set |= u64::from(match G {
                    Gather::Prefix => column_prefix(&self.xray, changed, wg, bg),
                    Gather::Fixed => column_fixed(&self.xray, changed, wg, bg),
                    _ => column(&self.xray, changed),
                });
            }

            set & self.ids.sliders()
        };

        // The movers and the victim are written outright below; a candidate
        // slot spent on them is a wasted probe.
        for m in 0..plan.n_movers {
            cand &= !(1u64 << plan.movers[m].0);
        }

        if let Some((vid, _)) = plan.captured {
            cand &= !(1u64 << vid);
        }

        if REC {
            undo.n = 0;
            undo.vision = self.vision;
            undo.slider_groups = self.ids.slider_groups;
        }

        let mut moved_slider = false;

        if let Some((vid, victim_sq)) = plan.captured {
            if REC {
                undo.push(vid, self.rows[vid], self.xray[vid]);
            }
            moved_slider |= is_slider(self.ids.lut[vid]);
            self.rows[vid] = 0;
            self.xray[vid] = 0;
            self.ids.mailbox[usize::from(victim_sq)] = 0;
            self.ids.list[vid] = 0xFF;
        }
        self.ids.apply(mv, &plan);

        for id in bits(cand) {
            self.replace::<XRAY, REC>(id, Square(self.ids.list[id]), occ, undo);
            moved_slider = true;
        }

        for m in 0..plan.n_movers {
            let (id, _, dest) = plan.movers[m];
            self.replace::<XRAY, REC>(id, dest, occ, undo);
            moved_slider |= is_slider(self.ids.lut[id]);
        }

        if VISION && moved_slider {
            self.vision = self.slider_vision();
        }
    }

    /// The OR of the slider rows over the same two prefixes the gather tests,
    /// both planes, since the x-ray plane's affected set is `rows | xray`.
    /// Erring wide only costs a gather that finds nothing.
    #[inline(always)]
    fn slider_vision(&self) -> u64 {
        let wg = usize::from(self.ids.slider_groups[0]);
        let bg = usize::from(self.ids.slider_groups[1]);

        // SAFETY: AVX2 per the weave/mod.rs gate; the loads read four u64s from
        // group g of a 32-element array, and g never exceeds 7.
        unsafe {
            let mut acc = _mm256_setzero_si256();

            for g in (0..wg).chain(4..4 + bg) {
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.rows.as_ptr().add(g * 4).cast()));
                acc = _mm256_or_si256(acc, _mm256_loadu_si256(self.xray.as_ptr().add(g * 4).cast()));
            }

            let folded = _mm_or_si128(_mm256_castsi256_si128(acc), _mm256_extracti128_si256::<1>(acc));
            (_mm_extract_epi64::<0>(folded) | _mm_extract_epi64::<1>(folded)).cast_unsigned()
        }
    }

    #[inline(always)]
    fn replace<const XRAY: bool, const REC: bool>(&mut self, id: usize, sq: Square, occ: Bitboard, undo: &mut Undo) {
        if REC {
            undo.push(id, self.rows[id], self.xray[id]);
        }

        if XRAY {
            let (direct, behind) = planes_of(self.ids.lut[id], sq, color_of(id), occ);
            self.rows[id] = direct.0;
            self.xray[id] = behind.0;
        } else {
            self.rows[id] = attacks_of(self.ids.lut[id], sq, color_of(id), occ).0;
        }
    }

    /// The candidate set has the movers and the victim masked out, so the
    /// record holds each piece once and the replay needs no ordering.
    #[inline(always)]
    fn unmake<const XRAY: bool>(&mut self, mv: Move, undo: &Undo) {
        for i in 0..undo.n {
            let id = usize::from(undo.ids[i]);
            self.rows[id] = undo.rows[i];
            if XRAY {
                self.xray[id] = undo.xray[i];
            }
        }
        // En passant sets the capture bit and castling doesn't, so the flag
        // alone says whether slot zero is a victim.
        let victim = mv.is_capture().then(|| usize::from(undo.ids[0]));
        let plan = self.ids.read_unmove(mv, victim);
        self.ids.undo(mv, &plan, undo.slider_groups);
        self.vision = undo.vision;
    }

    /// What `Position::threats` rebuilds per node with SIMD fills.
    #[inline(always)]
    fn danger(&self, side: usize) -> u64 {
        let mut acc = 0u64;
        for i in 0..16 {
            acc |= self.rows[side * 16 + i];
        }
        acc
    }

    /// A pinner is blocked by the piece it pins, so it never shows up in the
    /// direct plane. It is a slider whose second segment lands on the king, and
    /// the blocker is the one square its direct row keeps on that ray.
    #[inline(always)]
    fn pinned(&self, pos: &Position, color: Color) -> Bitboard {
        let ksq = pos.pieces(PieceType::King, color).lsb();
        let us = pos.side_bb[color];
        let enemy = 0xFFFF_0000u64 >> (usize::from(color) * 16);
        let cand = u64::from(column(&self.xray, ksq.bitboard())) & self.ids.sliders() & enemy;
        let mut pinned = Bitboard(0);
        for id in bits(cand) {
            let sq = Square(self.ids.list[id]);
            pinned |= Bitboard(self.rows[id]) & between_bb(sq, ksq) & us;
        }
        pinned
    }

    fn rebuild(&mut self, pos: &Position) {
        let occ = pos.occupancy();
        self.rows = [0; 32];
        self.xray = [0; 32];

        for id in 0..32 {
            let sq = self.ids.list[id];
            if sq == 0xFF {
                continue;
            }

            let (direct, behind) = planes_of(self.ids.lut[id], Square(sq), color_of(id), occ);
            self.rows[id] = direct.0;
            self.xray[id] = behind.0;
        }
        self.vision = self.slider_vision();
    }
}

impl Undo {
    fn empty() -> Self {
        Self { n: 0, ids: [0; 24], rows: [0; 24], xray: [0; 24], vision: 0, slider_groups: [0; 2] }
    }

    #[inline(always)]
    fn push(&mut self, id: usize, row: u64, xray: u64) {
        debug_assert!(self.n < 24, "affected set outgrew twenty sliders plus two movers plus a victim");
        self.ids[self.n] = id as u8;
        self.rows[self.n] = row;
        self.xray[self.n] = xray;
        self.n += 1;
    }
}

impl Pre {
    const ZERO: Self = Self { occ: Bitboard(0), rq: Bitboard(0), bq: Bitboard(0) };

    #[inline(always)]
    fn of(pos: &Position) -> Self {
        Self {
            occ: pos.occ,
            rq: pos.role_bb[PieceType::Rook] | pos.role_bb[PieceType::Queen],
            bq: pos.role_bb[PieceType::Bishop] | pos.role_bb[PieceType::Queen],
        }
    }
}

/// The transpose, one column at a time. On AVX-512 the same shape is four
/// `vptestmq` with the k-masks joined.
#[inline(always)]
fn column(rows: &[u64; 32], mask: Bitboard) -> u32 {
    // SAFETY: AVX2 per the weave/mod.rs gate, AVX-512 under its own cfg. Each
    // load covers a whole group of a 32-element array, so the last ends exactly
    // at its end.
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

/// The same test over the two groups the seed packs a side's sliders into,
/// unrolled like `column` because the bound is constant. A promotion turns a
/// pawn's id into a slider's and can land it past the bound, so that position
/// falls back to the full sweep.
#[inline(always)]
fn column_fixed(rows: &[u64; 32], mask: Bitboard, wg: usize, bg: usize) -> u32 {
    if wg > 2 || bg > 2 {
        return column(rows, mask);
    }

    // SAFETY: as `column`; the four group indices are constants under 8.
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

/// The same test over the two slider prefixes only. Runtime bounds, so it does
/// not unroll; it wins when the prefixes are short, which is whenever the side
/// has not promoted into the leaper ids.
#[inline(always)]
fn column_prefix(rows: &[u64; 32], mask: Bitboard, wg: usize, bg: usize) -> u32 {
    // SAFETY: as `column`; g never exceeds 7.
    unsafe {
        let m = _mm256_set1_epi64x(mask.0.cast_signed());
        let zero = _mm256_setzero_si256();
        let mut live = 0u32;
        for g in (0..wg).chain(4..4 + bg) {
            let v = _mm256_loadu_si256(rows.as_ptr().add(g * 4).cast());
            let miss = _mm256_cmpeq_epi64(_mm256_and_si256(v, m), zero);
            live |= (!_mm256_movemask_pd(_mm256_castsi256_pd(miss)).cast_unsigned() & 0xF) << (g * 4);
        }
        live
    }
}

/// Every variant builds both stores, read or not, so the from-scratch cost
/// lands in the baseline and cancels out of the deltas.
fn setup(fen: &str) -> (Position, Vi16x8, AttackTable, XorBoard, ByteStore) {
    let pos = Position::from_fen(fen);
    let acc = pos.get_initial_accumulator();
    let dst = black_box(AttackTable::from_scratch(&pos));
    let xb = black_box(XorBoard::from_scratch(&pos));
    let bs = black_box(ByteStore::from_scratch(&pos));
    (pos, acc, dst, xb, bs)
}

fn play_stream<F>(stream: &Stream, repeats: usize, mut per_ply: F) -> Outcome
where F: FnMut(&mut Position, &mut Vi16x8, Move, PieceType) -> u64 {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, ..) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                let pt = pos.expect_piece_at(mv.from());
                pos.make_move(mv, &mut acc);
                checksum ^= per_ply(&mut pos, &mut acc, mv, pt);
                iterations += 1;
            }
        }
    }
    Outcome { variant: "", iterations, checksum }
}

/// How a variant finds the sliders a move disturbed.
#[derive(PartialEq, Eq, ConstParamTy, Clone, Copy)]
enum Gather {
    /// Leaves the store wrong on purpose: prices the search for affected
    /// sliders against the work it finds.
    None,
    /// Every row against the changed squares.
    Column,
    /// The slider prefixes only, on runtime bounds.
    Prefix,
    /// The two groups each color's sliders are seeded into, on const bounds.
    Fixed,
    /// A superpiece cast out from the changed squares.
    Probe,
}

/// Which derived map a variant reads, and on which side of the make.
#[derive(PartialEq, Eq, ConstParamTy, Clone, Copy)]
enum View {
    None,
    /// Both danger maps, read straight after the make.
    FusedDanger,
    /// Both, read before it.
    LaggedDanger,
    /// One side's, read before it, which is what the search does.
    LaggedStmDanger,
    /// Both colors' pins, read before it.
    LaggedPins,
}

fn variant_baseline(stream: &Stream, repeats: usize) -> Outcome {
    play_stream(stream, repeats, |_, _, _, _| 0)
}

fn variant_dest(stream: &Stream, repeats: usize) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, mut dst, ..) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                pos.make_move(mv, &mut acc);
                dst.make(&pos, mv);
                checksum ^= dst.rows[0] | dst.rows[31] | dst.threat[0] | dst.threat[1];
                iterations += 1;
            }
        }
    }
    Outcome { variant: "dest", iterations, checksum }
}

/// The copy pair a save/restore host pays per node, nothing updated between the
/// two. The row read forces the restore to be real; touch nothing in between and
/// the compiler proves the pair dead and deletes it.
fn variant_hosting(stream: &Stream, repeats: usize) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    let mut dst = AttackTable::from_scratch(&Position::from_fen(&stream.games[0].fen));
    for i in 0..repeats * stream.games.iter().map(|g| g.moves.len()).sum::<usize>() {
        let saved = black_box(dst);
        dst = black_box(saved);
        checksum ^= dst.rows[i % 32];
        iterations += 1;
    }
    Outcome { variant: "hosting", iterations, checksum }
}

/// `UNDO` replays and re-makes to keep the stream in step, so one
/// extra make sits in its delta.
///
/// Where a [`View`] is read is not cosmetic. Straight after the make, the
/// reduction gets rows the compiler just stored and folds the loads away, a
/// fusion the search never sees across a function boundary. Read before, it
/// loads cold and pays no store-forwarding stall. The search reads one map per
/// node, which is `LaggedStmDanger`.
fn variant_xorboard<const G: Gather, const VISION: bool, const XRAY: bool, const VIEW: View, const UNDO: bool>(
    stream: &Stream,
    repeats: usize,
    name: &'static str,
) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    let mut undo = Undo::empty();

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, mut xb, _) = setup(&game.fen);

            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                let pre = if G == Gather::Probe { Pre::of(&pos) } else { Pre::ZERO };

                checksum ^= match VIEW {
                    View::LaggedDanger => xb.danger(0) | xb.danger(1),
                    View::LaggedStmDanger => xb.danger(usize::from(pos.stm.opposite())),
                    View::LaggedPins => (xb.pinned(&pos, Color::White) | xb.pinned(&pos, Color::Black)).0,
                    _ => 0,
                };

                pos.make_move(mv, &mut acc);
                xb.make::<G, VISION, XRAY, UNDO>(pre, pos.occ, mv, &mut undo);

                checksum ^= match VIEW {
                    View::FusedDanger => xb.danger(0) | xb.danger(1),
                    _ => xb.rows[0] | xb.rows[31],
                };

                if UNDO {
                    xb.unmake::<XRAY>(mv, &undo);
                    xb.make::<G, VISION, XRAY, false>(pre, pos.occ, mv, &mut undo);
                }
                iterations += 1;
            }
        }
    }
    Outcome { variant: name, iterations, checksum }
}

/// The XorBoard under copy-make: copy the whole store per node instead of
/// snapshotting the rows a move touches. The snapshot is proportional to the move,
/// a handful of rows; the copy is proportional to the board, every row every time.
fn variant_xorboard_copy(stream: &Stream, repeats: usize) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    let mut undo = Undo::empty();

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, mut xb, _) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                pos.make_move(mv, &mut acc);

                let parent = xb;
                xb.make::<{ Gather::Column }, false, false, false>(Pre::ZERO, pos.occ, mv, &mut undo);
                hint::black_box(&parent);

                checksum ^= xb.rows[0] | xb.rows[31];
                iterations += 1;
            }
        }
    }
    Outcome { variant: "xorboard_copy", iterations, checksum }
}

/// Rose is copy-make: `search.cpp` builds a child position per node, so the update
/// lands in a fresh copy of the board and its two attack tables, and nothing is ever
/// undone. This pays that copy per ply, against `xorboard_undo`'s snapshot and unmake.
fn variant_byteboard_copy(stream: &Stream, repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();
    let mut checksum = 0u64;
    let mut iterations = 0u64;

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, _, mut st) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                pos.make_move(mv, &mut acc);

                // The parent has to survive the child's update, or the copy is not one;
                // black_box is what stops it collapsing into the update itself. It also
                // overstates: the delta here runs ~511 per ply where an isolated copy of
                // this size costs ~90, so read the excess as barrier, not as copying.
                let parent = st.bb;
                st.make(&g, mv);
                hint::black_box(&parent);

                checksum ^= attack_bits(&st.bb, 0, 0) | attack_bits(&st.bb, 1, 63);
                iterations += 1;
            }
        }
    }
    Outcome { variant: "byteboard_copy", iterations, checksum }
}

/// The questions search asks of an attack store, on the byteboard's side of the
/// transpose: who hits one square and then who each of them is, which squares a
/// color hits at all, and what one piece hits. The first is a single load here and
/// a gather for the XorBoard; the last is the reverse. Same store work as
/// `byteboard`, so the difference between the two runs is what the reads cost.
///
/// Board queries stay out. Rose's byteboard is the board, so `occupied` and
/// `pieces_of` derive from it, while Soul reads bitboards `Position` maintains
/// inside the shared baseline. Pricing those together would credit one side with
/// work the baseline already subtracted.
fn variant_byteboard_reads(stream: &Stream, repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();
    let mut checksum = 0u64;
    let mut iterations = 0u64;

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, _, mut st) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                pos.make_move(mv, &mut acc);
                st.make(&g, mv);

                let (sq, slot) = probe_points(iterations);
                let attacks = st.bb.attacks(pos.stm);

                let attackers = attacks.read(sq);
                checksum ^= u64::from(attackers.bits());
                checksum ^= attackers.fold(0u64, |walked, id| walked | 1 << id.raw());
                checksum ^= attacks.any();
                checksum ^= attacks.with(slot.mask());
                iterations += 1;
            }
        }
    }
    Outcome { variant: "byteboard_reads", iterations, checksum }
}

/// The same three questions on the XorBoard's side, where a piece's set is the
/// single load and one square's attackers is the gather.
fn variant_xorboard_reads(stream: &Stream, repeats: usize) -> Outcome {
    let mut checksum = 0u64;
    let mut iterations = 0u64;
    let mut undo = Undo::empty();

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, mut xb, _) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                let pre = Pre::ZERO;
                pos.make_move(mv, &mut acc);
                xb.make::<{ Gather::Column }, false, false, false>(pre, pos.occ, mv, &mut undo);

                let (sq, slot) = probe_points(iterations);
                let color = usize::from(pos.stm);

                let mut attackers = wordboard_at(&xb, color, usize::from(sq));
                checksum ^= u64::from(attackers);
                while attackers != 0 {
                    checksum ^= 1 << attackers.trailing_zeros();
                    attackers &= attackers - 1;
                }
                checksum ^= xb.danger(color);
                checksum ^= xb.rows[color * 16 + usize::from(slot.raw())];
                iterations += 1;
            }
        }
    }
    Outcome { variant: "xorboard_reads", iterations, checksum }
}

/// The square and slot each ply interrogates. Walking them keeps a load from being
/// hoisted out of the replay and asks both stores about the same place.
fn probe_points(iterations: u64) -> (Square, PieceId) {
    (Square((iterations % 64) as u8), PieceId::new((iterations % 16) as u8))
}

fn variant_byteboard(stream: &Stream, repeats: usize) -> Outcome {
    let g = byteboard::Geometry::new();
    let mut checksum = 0u64;
    let mut iterations = 0u64;

    for _ in 0..repeats {
        for game in &stream.games {
            let (mut pos, mut acc, _, _, mut st) = setup(&game.fen);
            for &mv in &game.moves {
                checksum ^= pos.hash.rotate_left((iterations % 64) as u32);
                pos.make_move(mv, &mut acc);
                st.make(&g, mv);
                checksum ^= attack_bits(&st.bb, 0, 0) | attack_bits(&st.bb, 1, 63);
                iterations += 1;
            }
        }
    }
    Outcome { variant: "byteboard", iterations, checksum }
}

fn run_variant(name: &str, stream: &Stream, repeats: usize) -> Outcome {
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
        "xorboard_pack" => {
            variant_xorboard::<{ Gather::Prefix }, true, false, { View::None }, false>(stream, repeats, "xorboard_pack")
        },
        // The prefix gather alone; every other variant reaching for it also pays the vision recount.
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
        "threats" => play_stream(stream, repeats, |pos, _, _, _| pos.threats(pos.stm.opposite()).0),
        "pins" => play_stream(stream, repeats, |pos, _, _, _| {
            let pins = crate::core::board::attacks::Pins::new(pos);
            pins.blockers(Color::White).0 ^ pins.blockers(Color::Black).0
        }),
        "checkers" => play_stream(stream, repeats, |pos, _, _, _| pos.checkers().0),
        "new_threats" => play_stream(stream, repeats, |pos, _, mv, pt| pos.new_threats(pt, mv.from(), mv.to()).0),
        "eval_attack" => play_stream(stream, repeats, |pos, _, _, _| {
            let pinned_w = pos.pinned_pieces(Color::White);
            let pinned_b = pos.pinned_pieces(Color::Black);
            let tensor = SpatialTensor::compute(pos, pinned_w.0, pinned_b.0);
            let mob = Mobility::compute_all(pos, &tensor, pinned_w, pinned_b, None);
            (mob.metrics_us.mobility ^ mob.metrics_them.mobility) as u64 ^ (mob.safety_us.weak ^ mob.safety_them.weak) as u64
        }),
        "see" => play_stream(stream, repeats, |pos, _, mv, _| {
            let pins = crate::core::board::attacks::Pins::new(pos);
            u64::from(see_ge(pos, mv, 0, &pins))
        }),
        "king_legal" => play_stream(stream, repeats, |pos, _, mv, _| {
            let opp = pos.stm.opposite();
            u64::from(pos.is_attacked::<true>(mv.to(), opp, mv.from().bitboard()))
        }),
        _ => {
            eprintln!("measure: unknown variant '{name}'");
            std::process::exit(1);
        },
    }
}

/// Replay the stream maintaining both stores, cross-checking each against a
/// from-scratch recompute after every ply.
///
/// A store that is only its own oracle proves nothing, so the XorBoard answers
/// to things outside itself: its own rebuild, the other gather, `AttackTable`'s
/// rows and recounted threat maps, `pinned_pieces`, the byteboard transpose,
/// and its own unmake replay.
///
/// Returns the number of plies checked, so the count can be quoted rather than
/// remembered. `None` means a check failed and the message has already printed.
fn validate(stream: &Stream) -> Option<u64> {
    let mut checked = 0u64;

    for (gi, game) in stream.games.iter().enumerate() {
        let mut pos = Position::from_fen(&game.fen);
        let mut acc = pos.get_initial_accumulator();
        let mut dst = AttackTable::from_scratch(&pos);
        let mut xb = XorBoard::from_scratch(&pos);
        let mut col = XorBoard::from_scratch(&pos);
        let mut undo = Undo::empty();
        let g = byteboard::Geometry::new();
        let mut bs = ByteStore::from_scratch(&pos);

        for (ply, &mv) in game.moves.iter().enumerate() {
            let saved = dst;
            let saved_xb = xb;
            let plan = dst.ids.read_move(mv);

            // A stale mailbox byte would underflow the id read and panic inside
            // make. Abort with the state printed instead.
            let bad = plan.movers[0].0 >= 32
                || (plan.n_movers == 2 && plan.movers[1].0 >= 32)
                || plan.captured.is_some_and(|(vid, _)| vid >= 32);

            if bad {
                eprintln!("measure: stale mailbox, game {gi}, ply {ply}, move {}", mv.inner());
                eprintln!("fen before: {}", pos.as_fen());
                for m in 0..plan.n_movers {
                    eprintln!("  mover[{m}] id={} from={} to={}", plan.movers[m].0, plan.movers[m].1.0, plan.movers[m].2.0);
                }
                eprintln!("  captured: {:?}", plan.captured);
                eprintln!(
                    "  mailbox[from]={} mailbox[to]={} real at from={}",
                    dst.ids.mailbox[usize::from(mv.from())],
                    dst.ids.mailbox[usize::from(mv.to())],
                    pos.piece_at(mv.from()) as u8
                );

                return None;
            }

            let pre = Pre::of(&pos);
            pos.make_move(mv, &mut acc);
            dst.make(&pos, mv);
            xb.make::<{ Gather::Prefix }, true, true, true>(pre, pos.occ, mv, &mut undo);
            bs.make(&g, mv);
            col.make::<{ Gather::Probe }, true, true, false>(pre, pos.occ, mv, &mut undo);

            // The victim's clear precedes the bookkeeping, as make orders it:
            // for a capture the victim's square is the mover's destination.
            let mut want = saved;
            if let Some((vid, victim_sq)) = plan.captured {
                want.ids.mailbox[usize::from(victim_sq)] = 0;
                want.ids.list[vid] = 0xFF;
            }
            want.ids.apply(mv, &plan);
            want.rebuild(&pos);

            // The one check a self-derived comparison cannot make.
            let mut mailbox_ok = true;
            for sq in 0..64 {
                let real = pos.piece_at(Square(sq as u8)) != PieceType::None;
                if real != (dst.ids.mailbox[sq] != 0) {
                    if mailbox_ok {
                        eprintln!("measure: mailbox vs position, game {gi}, ply {ply}, move {}", mv.inner());
                        eprintln!("fen: {}", pos.as_fen());
                        mailbox_ok = false;
                    }
                    eprintln!("  sq {sq}: mailbox={} real={}", dst.ids.mailbox[sq], pos.piece_at(Square(sq as u8)) as u8);
                }
            }

            if !mailbox_ok {
                return None;
            }

            if dst != want {
                eprintln!("measure: dest mismatch, game {gi}, move {}", mv.inner());
                eprintln!("fen: {}", pos.as_fen());

                for (i, (a, b)) in dst.dest.iter().zip(want.dest.iter()).enumerate() {
                    if a != b {
                        eprintln!("  dest[{i}]: got {a:#010x} want {b:#010x}");
                    }
                }
                for (i, (a, b)) in dst.rows.iter().zip(want.rows.iter()).enumerate() {
                    if a != b {
                        eprintln!("  rows[{i}]: got {a:#018x} want {b:#018x}");
                    }
                }
                for (i, (a, b)) in dst.ids.mailbox.iter().zip(want.ids.mailbox.iter()).enumerate() {
                    if a != b {
                        eprintln!("  mailbox[{i}]: got {a:#04x} want {b:#04x}");
                    }
                }
                for (i, (a, b)) in dst.ids.list.iter().zip(want.ids.list.iter()).enumerate() {
                    if a != b {
                        eprintln!("  list[{i}]: got {a:#04x} want {b:#04x}");
                    }
                }
                for (i, (a, b)) in dst.ids.lut.iter().zip(want.ids.lut.iter()).enumerate() {
                    if a != b {
                        eprintln!("  lut[{i}]: got {a:?} want {b:?}");
                    }
                }
                for (i, (a, b)) in dst.ids.masks.iter().zip(want.ids.masks.iter()).enumerate() {
                    if a != b {
                        eprintln!("  masks[{i}]: got {a:#018x} want {b:#018x}");
                    }
                }
                return None;
            }

            let mut xb_want = xb;
            xb_want.rebuild(&pos);
            let mut replayed = xb;
            replayed.unmake::<true>(mv, &undo);

            let faults: [(&str, bool); 9] = [
                ("xorboard vs rebuild", xb != xb_want),
                ("probe gather vs column gather", col != xb),
                ("xorboard rows vs dest rows", xb.rows != dst.rows),
                ("danger(white) vs threat[0]", xb.danger(0) != dst.threat[0]),
                ("danger(black) vs threat[1]", xb.danger(1) != dst.threat[1]),
                ("xray pins(white) vs pinned_pieces", xb.pinned(&pos, Color::White) != pos.pinned_pieces(Color::White)),
                ("xray pins(black) vs pinned_pieces", xb.pinned(&pos, Color::Black) != pos.pinned_pieces(Color::Black)),
                ("unmake replay", replayed != saved_xb),
                ("byteboard vs xorboard", !byteboard_agrees(&bs, &xb)),
            ];

            for (what, failed) in faults {
                if failed {
                    eprintln!("measure: {what}, game {gi}, ply {ply}, move {}", mv.inner());
                    eprintln!("fen: {}", pos.as_fen());
                    eprintln!("record held {} rows", undo.n);

                    for id in 0..32 {
                        let drift = xb.rows[id] != xb_want.rows[id]
                            || xb.xray[id] != xb_want.xray[id]
                            || col.rows[id] != xb.rows[id]
                            || xb.rows[id] != dst.rows[id]
                            || replayed.rows[id] != saved_xb.rows[id];

                        if drift {
                            eprintln!(
                                "  {id}: row {:#018x} rebuild {:#018x} dest {:#018x} replay {:#018x} want {:#018x}",
                                xb.rows[id], xb_want.rows[id], dst.rows[id], replayed.rows[id], saved_xb.rows[id]
                            );
                            eprintln!("      xray {:#018x} rebuild {:#018x}", xb.xray[id], xb_want.xray[id]);
                        }
                    }

                    for sq in 0..64 {
                        for c in 0..2 {
                            let want = wordboard_at(&xb, c, sq);
                            let got = attack_bits(&bs.bb, c, sq) as u16;
                            if got != want {
                                eprintln!("  sq {sq} c{c}: bb {got:#06x} want {want:#06x}");
                            }
                        }
                    }

                    for color in [Color::White, Color::Black] {
                        eprintln!(
                            "  pins {color:?}: got {:#018x} want {:#018x}",
                            xb.pinned(&pos, color).0,
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

/// The byteboard's wordboards are the XorBoard's rows transposed, so one is the
/// other's oracle and neither gets to grade its own homework.
fn byteboard_agrees(bs: &ByteStore, xb: &XorBoard) -> bool {
    (0..64).all(|sq| (0..2).all(|c| attack_bits(&bs.bb, c, sq) as u16 == wordboard_at(xb, c, sq)))
}

/// One square's attack entry as raw bits, for comparison against the transpose.
fn attack_bits(bb: &byteboard::ByteBoard, color: usize, sq: usize) -> u64 {
    let mask = bb.attacks(if color == 0 { Color::White } else { Color::Black }).read(Square(sq as u8));
    u64::from(mask.bits())
}

/// One square's column of the transpose: which of a color's slots reach it.
#[inline]
fn wordboard_at(xb: &XorBoard, color: usize, sq: usize) -> u16 {
    let mut set = 0u16;
    for slot in 0..16 {
        set |= ((xb.rows[color * 16 + slot] >> sq & 1) as u16) << slot;
    }
    set
}

/// The write volume each axis commits to, counted rather than timed.
///
/// Rebel's byte, Rookie's direction bits, Muller's distance fields, KnightCap's
/// id bitset and Rose's color-split pair all store one entry per (piece, square)
/// pair whose visibility moved. The entry differs between them; the pair count
/// does not. Transposed, the same move is a couple of whole rows.
fn count_writes(stream: &Stream) {
    let (mut moves, mut pairs, mut rows, mut touched) = (0u64, 0u64, 0u64, 0u64);
    let mut worst = 0u32;

    for game in &stream.games {
        let mut pos = Position::from_fen(&game.fen);
        let mut acc = pos.get_initial_accumulator();
        let mut dst = AttackTable::from_scratch(&pos);
        let mut xb = XorBoard::from_scratch(&pos);
        let mut undo = Undo::empty();

        for &mv in &game.moves {
            let before = dst;
            let pre = Pre::of(&pos);
            pos.make_move(mv, &mut acc);
            dst.make(&pos, mv);
            xb.make::<{ Gather::Prefix }, true, false, true>(pre, pos.occ, mv, &mut undo);

            let mut delta = 0u32;
            for id in 0..32 {
                delta += (before.rows[id] ^ dst.rows[id]).count_ones();
            }

            moves += 1;
            pairs += u64::from(delta);
            rows += undo.n as u64;
            touched += u64::from((before.threat[0] ^ dst.threat[0] | before.threat[1] ^ dst.threat[1]).count_ones());
            worst = worst.max(delta);
        }
    }

    let m = moves as f64;
    println!("moves {moves}");
    println!("pairs/move {:.2} (worst {worst})", pairs as f64 / m);
    println!("rows/move  {:.2}", rows as f64 / m);
    println!("danger bits flipped/move {:.2}", touched as f64 / m);
}

pub fn run(args: &[&str]) {
    match args.first().copied() {
        Some("count") => {
            let path = args.get(1).copied().expect("measure count <path>");
            count_writes(&deserialize(&fs::read(path).expect("read stream")));
        },
        Some("gen") => {
            let path = args.get(1).copied().expect("measure gen <path> [ply_cap]");
            let cap = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(PLY_CAP);
            let stream = generate(0xC0FFEE, cap);
            fs::write(path, serialize(&stream)).expect("write stream");
            let plies: usize = stream.games.iter().map(|g| g.moves.len()).sum();
            println!("measure gen {} games, {plies} plies -> {path}", stream.games.len());
        },
        // One line of FEN, one of UCI moves, per game, so another engine can
        // replay the identical plies and its numbers land on our axis.
        Some("export") => {
            let path = args.get(1).copied().expect("measure export <stream> <out>");
            let out = args.get(2).copied().expect("measure export <stream> <out>");
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let mut text = String::new();
            for game in &stream.games {
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
            println!("measure export {} games -> {out}", stream.games.len());
        },
        Some("validate") => {
            let path = args.get(1).copied().expect("measure validate <path>");
            let stream = deserialize(&fs::read(path).expect("read stream"));

            match validate(&stream) {
                Some(plies) => println!("measure validate OK, {plies} plies"),
                None => std::process::exit(1),
            }
        },
        Some("dump") => {
            let path = args.get(1).copied().expect("measure dump <path> [plies]");
            let n = args.get(2).and_then(|s| s.parse::<usize>().ok()).unwrap_or(10);
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let game = &stream.games[0];
            for (ply, &mv) in game.moves.iter().take(n).enumerate() {
                let mut pos = Position::from_fen(&game.fen);
                let mut acc = pos.get_initial_accumulator();

                for &prior in game.moves.iter().take(ply) {
                    pos.make_move(prior, &mut acc);
                }

                let pre_fen = pos.as_fen();
                let info = pos.make_move(mv, &mut acc);
                println!("ply {ply}: {} ({}) pre_stm={} pre_fen={}", mv.to_uci(false), mv.inner(), pos.stm as u8, pre_fen);
                println!("  post: {}", pos.as_fen());
                pos.unmake_move(mv, &info);
                println!("  restored: {}  (match={})", pos.as_fen(), pos.as_fen() == pre_fen);
            }
        },
        Some("run") => {
            let name = args.get(1).copied().expect("measure run <variant> <path> [repeats]");
            let path = args.get(2).copied().expect("measure run <variant> <path> [repeats]");
            let repeats = args.get(3).and_then(|s| s.parse::<usize>().ok()).unwrap_or(4);
            let stream = deserialize(&fs::read(path).expect("read stream"));
            let outcome = run_variant(name, &stream, repeats);
            println!("measure {} {} {}", outcome.variant, outcome.iterations, outcome.checksum);
        },
        _ => {
            eprintln!("measure: expected 'gen', 'export', 'validate', 'count', 'dump', or 'run'");
            std::process::exit(1);
        },
    }
}
