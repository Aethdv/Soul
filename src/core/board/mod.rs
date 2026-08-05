//! Board representation: the `Position` type and incremental game state.
//!
//! Hybrid bitboard + mailbox representation:
//! - `side_bb[2]` + `role_bb[6]`: bitboard planes for bulk set operations
//! - `pieces[64]`: mailbox for 𝒪(1) "what piece is on this square?" queries
//!
//! Both views stay perfectly synchronized through `add_piece` / `remove_piece`.
//!
//! Zobrist hash and SIMD accumulator are maintained incrementally, updated
//! diff-style in `make_move`, restored from snapshots in `unmake_move`.

use std::{fmt, iter};

use crate::{
    core::{
        defs::{Bitboard, Color, FILE_A, FILE_H, PieceType, Square},
        error::FenError,
        moves::Move,
        psqt, zobrist,
    },
    weave::{Vi16x8, Vu64x4},
};

pub mod attacks;
pub mod bitboard;
mod fen;
mod make;
pub mod spatial;
pub mod xorboard;

#[cfg(test)] mod tests;

pub const STARTPOS: &str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

pub const ROOK_W_KS: usize = 0;
pub const ROOK_W_QS: usize = 1;
pub const ROOK_B_KS: usize = 2;
pub const ROOK_B_QS: usize = 3;

/// King's home file in the standard starting position (e = 4).
pub const DEFAULT_KING_FILE: u8 = 4;
/// h-file
pub const KINGSIDE_FILE: u8 = 7;
/// a-file
pub const QUEENSIDE_FILE: u8 = 0;
/// f1, g1
pub const W_OO_EMPTY: Bitboard = Bitboard(0x60);
/// b1, c1, d1
pub const W_OOO_EMPTY: Bitboard = Bitboard(0x0E);
/// f8, g8
pub const B_OO_EMPTY: Bitboard = Bitboard(0x6000_0000_0000_0000);
/// b8, c8, d8
pub const B_OOO_EMPTY: Bitboard = Bitboard(0x0E00_0000_0000_0000);

/// Toward a-file (queenside rook)
pub const SEARCH_LEFT: i8 = -1;
/// Toward h-file (kingside rook)
pub const SEARCH_RIGHT: i8 = 1;

pub const WHITE_OO: u8 = 1;
pub const WHITE_OOO: u8 = 2;
pub const BLACK_OO: u8 = 4;
pub const BLACK_OOO: u8 = 8;

/// e1, h1 → g1, f1
pub const CASTLE_W_KS: [u8; 4] = [4, 7, 6, 5];
/// e1, a1 → c1, d1
pub const CASTLE_W_QS: [u8; 4] = [4, 0, 2, 3];
/// e8, h8 → g8, f8
pub const CASTLE_B_KS: [u8; 4] = [60, 63, 62, 61];
/// e8, a8 → c8, d8
pub const CASTLE_B_QS: [u8; 4] = [60, 56, 58, 59];
/// e1, f1, g1
pub const CASTLE_W_KS_CHECK: [u8; 3] = [4, 5, 6];
/// e1, d1, c1
pub const CASTLE_W_QS_CHECK: [u8; 3] = [4, 3, 2];
/// e8, f8, g8
pub const CASTLE_B_KS_CHECK: [u8; 3] = [60, 61, 62];
/// e8, d8, c8
pub const CASTLE_B_QS_CHECK: [u8; 3] = [60, 59, 58];

// Compile-time layout verification: catches silent ABI breakage.
const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<Position>() == 192, "Position must be exactly 192 bytes");
    assert!(align_of::<Position>() == 32, "Position must be 32-byte aligned");

    // Hot fields packed at the front for cache locality during move gen.
    assert!(offset_of!(Position, side_bb) == 0);
    assert!(offset_of!(Position, role_bb) == 16);
    assert!(offset_of!(Position, occ) == 64);
    assert!(offset_of!(Position, hash) == 72);
    assert!(offset_of!(Position, pawn_key) == 80);
    assert!(offset_of!(Position, minor_key) == 88);
    assert!(offset_of!(Position, major_key) == 96);

    assert!(size_of::<StateInfo>() == 40, "StateInfo must be exactly 40 bytes");
    assert!(align_of::<StateInfo>() == 8);
};

pub use fen::Fen;
pub(crate) use make::castling_targets;

//  192 bytes, spans three cache lines.
//
//  ┌──────────────────┬────────┬───────┐
//  │ Field            │ Offset │ Bytes │
//  ├──────────────────┼────────┼───────┤
//  │ side_bb          │      0 │    16 │
//  │ role_bb          │     16 │    48 │
//  │ occ              │     64 │     8 │
//  │ hash             │     72 │     8 │
//  │ pawn_key         │     80 │     8 │
//  │ minor_key        │     88 │     8 │
//  │ major_key        │     96 │     8 │
//  │ pieces           │    104 │    64 │
//  │ castling_rooks   │    168 │     4 │
//  │ castling_rights  │    172 │     1 │
//  │ stm              │    173 │     1 │
//  │ halfmove_clock   │    174 │     1 │
//  │ en_passant       │    175 │     2 │
//  │ is_frc           │    177 │     1 │
//  │ fullmove_number  │    178 │     2 │
//  │ (tail padding)   │    180 │    12 │
//  └──────────────────┴────────┴───────┘
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
pub struct Position {
    /// Per-color occupancy: `side_bb[White]` has all white pieces, etc.
    pub side_bb: [Bitboard; 2],
    /// Per-role occupancy: `role_bb[Pawn]` has all pawns regardless of color.
    pub role_bb: [Bitboard; 6],
    /// Union of all occupied squares: always `side_bb[0] | side_bb[1]`.
    pub occ: Bitboard,
    /// Incrementally maintained Zobrist hash.
    pub hash: u64,
    /// Incremental pawn-only Zobrist key; keys pawn correction history.
    pub pawn_key: u64,
    /// Incremental minor-piece (knight + bishop) Zobrist key; keys minor correction history.
    pub minor_key: u64,
    /// Incremental major-piece (rook + queen) Zobrist key; keys major correction history.
    pub major_key: u64,
    /// Mailbox: `pos.piece_at(sq)` gives the piece type (or `None` if empty).
    pub pieces: [PieceType; 64],
    /// Rook home squares for castling, indexed by rights bit position.
    pub castling_rooks: [Square; 4],
    /// Packed castling rights, bits 0–3: WK, WQ, BK, BQ.
    pub castling_rights: u8,
    /// Side to move.
    pub stm: Color,
    /// Fifty-move rule counter (half-moves since last pawn push or capture).
    pub halfmove_clock: u8,
    /// En passant target square, set after a double pawn push.
    pub en_passant: Option<Square>,
    /// Fischer Random Chess mode. When true, FENs and moves use Chess960 format.
    pub is_frc: bool,
    /// Full-move counter (starts at 1, incremented after Black moves).
    pub fullmove_number: u16,
}

/// The irreversible state needed to undo a move: stack-allocated, snapshotted per move.
///
/// A move destroys castling rights, the fifty-move counter, en passant availability and the
/// captured piece. Snapshotting them before the move makes `unmake_move` a memcpy rather
/// than a recalculation.
#[derive(Clone, Copy, Debug)]
#[repr(C)]
pub struct StateInfo {
    pub hash: u64,
    pub pawn_key: u64,
    pub minor_key: u64,
    pub major_key: u64,
    pub en_passant: Option<Square>,
    pub castling_rights: u8,
    pub captured: PieceType,
    pub halfmove_clock: u8,
}

impl Default for StateInfo {
    fn default() -> Self {
        Self {
            hash: 0,
            pawn_key: 0,
            minor_key: 0,
            major_key: 0,
            en_passant: None,
            castling_rights: 0,
            captured: PieceType::None,
            halfmove_clock: 0,
        }
    }
}

impl fmt::Display for Position {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", Fen(self))
    }
}

impl Position {
    /// An empty board: no pieces, White to move, Zobrist initialized.
    pub fn new() -> Self {
        let mut pos = Self {
            side_bb: [Bitboard(0); 2],
            role_bb: [Bitboard(0); 6],
            occ: Bitboard(0),
            hash: 0,
            pawn_key: 0,
            minor_key: 0,
            major_key: 0,
            pieces: [PieceType::None; 64],
            castling_rooks: [Square(0); 4],
            castling_rights: 0,
            stm: Color::White,
            halfmove_clock: 0,
            en_passant: None,
            is_frc: false,
            fullmove_number: 1,
        };
        pos.hash = pos.calc_zobrist();
        pos
    }

    /// Parse a FEN string, panicking on invalid input.
    /// Use `try_from_fen` for user-supplied FENs where graceful error handling is needed.
    pub fn from_fen(fen: &str) -> Self {
        Self::try_from_fen(fen).unwrap_or_else(|e| panic!("Invalid FEN: {e}"))
    }

    /// Parse a FEN string. Returns an error for malformed input.
    pub fn try_from_fen(fen: &str) -> Result<Self, FenError> {
        Self::try_from_tokens(&mut fen.split_whitespace().peekable())
    }

    /// Parse from pre-split FEN tokens
    /// (useful for UCI `position` commands where the token stream is already available).
    pub fn try_from_tokens<'a, I>(parts: &mut iter::Peekable<I>) -> Result<Self, FenError>
    where I: Iterator<Item = &'a str> {
        fen::try_from_tokens(parts)
    }

    /// Apply `mv` to the position. Returns the undo packet needed by
    /// `unmake_move`.
    #[inline]
    pub fn make_move(&mut self, mv: Move, acc: &mut Vi16x8) -> StateInfo {
        make::make_move(self, mv, acc)
    }

    /// Restore the position to its state before `mv` was played.
    #[inline]
    pub fn unmake_move(&mut self, mv: Move, info: &StateInfo) {
        make::unmake_move(self, mv, info);
    }

    /// Pass the turn without moving.
    /// Returns the undo packet.
    /// Used by null move pruning: the opponent gets a free move.
    #[inline]
    pub fn make_null_move(&mut self) -> StateInfo {
        let info = StateInfo {
            hash: self.hash,
            pawn_key: self.pawn_key,
            minor_key: self.minor_key,
            major_key: self.major_key,
            en_passant: self.en_passant,
            castling_rights: self.castling_rights,
            captured: PieceType::None,
            halfmove_clock: self.halfmove_clock,
        };

        if let Some(ep) = self.en_passant.take() {
            self.hash ^= zobrist::key_ep(ep);
        }

        self.stm = self.stm.opposite();
        self.hash ^= zobrist::key_side();
        self.halfmove_clock += 1;
        info
    }

    /// Undo a null move.
    #[inline]
    pub fn unmake_null_move(&mut self, info: &StateInfo) {
        self.stm = self.stm.opposite();
        self.hash = info.hash;
        self.pawn_key = info.pawn_key;
        self.minor_key = info.minor_key;
        self.major_key = info.major_key;
        self.en_passant = info.en_passant;
        self.halfmove_clock = info.halfmove_clock;
    }

    /// Does the given side have any pieces beyond pawns and king?
    /// Used by NMP to avoid null-moving in pure pawn endings (zugzwang).
    #[inline]
    pub fn has_non_pawn_material(&self, side: Color) -> bool {
        let dominated =
            self.side_bb[side as usize] & !(self.role_bb[PieceType::Pawn as usize] | self.role_bb[PieceType::King as usize]);
        dominated.is_not_empty()
    }

    /// Incrementally update the SIMD accumulator for `mv`.
    ///
    /// Must be called before the move is applied to the board: it
    /// reads the current piece layout to determine what changed.
    #[inline(always)]
    pub fn update_accumulator(&self, acc: &mut Vi16x8, mv: Move, pt: PieceType, captured: PieceType, placed: PieceType) {
        make::update_accumulator(self, acc, mv, pt, captured, placed);
    }

    /// Compute the SIMD accumulator from scratch
    /// by summing PSQT vectors for every piece on the board.
    /// Used once at position setup:
    /// after that, `make_move` / `unmake_move` keep it updated incrementally.
    pub fn get_initial_accumulator(&self) -> Vi16x8 {
        let mut acc = Vi16x8::splat(0);
        for sq in self.occ {
            // MG/EG lanes are i16 and won't overflow at realistic piece values
            acc += psqt::get_vec(self.piece_at(sq), sq, self.color_at(sq));
        }
        acc
    }

    /// Detects threefold repetition within the reversible move horizon.
    ///
    /// Only the last `halfmove_clock + 1` positions matter: anything before
    /// a capture or pawn push can never be the same position again.
    ///
    /// This uses a `step_by(2)` optimization for adjudication (where we
    /// explicitly need 3-fold counts). Contrast with `Worker::is_repetition`
    /// which scans every ply for early draw detection in search.
    ///
    /// - `rev()`: scan backward from the most recent position.
    /// - `skip(2)`: the tail entry is the current position (count starts at 1),
    ///   and the position one ply back has the opponent to move, so both are skipped.
    /// - `step_by(2)`: only compare positions where the same side is to move,
    ///   since the Zobrist hash includes a side-to-move key.
    pub fn is_threefold_repetition(&self, history: &[u64]) -> bool {
        if history.len() < 4 {
            return false;
        }

        let limit = history.len().saturating_sub(self.halfmove_clock as usize + 1);
        let mut count = 1; // Current position
        for &h in history[limit..].iter().rev().skip(2).step_by(2) {
            if h == self.hash {
                count += 1;
                if count >= 3 {
                    return true;
                }
            }
        }
        false
    }

    pub fn is_fifty_move_draw(&self) -> bool {
        self.halfmove_clock >= 100
    }

    /// Detects draw by insufficient mating material.
    ///
    /// Covers K vs K, K+N vs K, and K+B vs K. Exotic cases like same colored
    /// bishops or K+N+N vs K are left to the 50-move rule or adjudication.
    pub fn is_draw_by_material(&self) -> bool {
        // Any pawns, rooks, or queens on the board → someone can still win.
        let force = self.role_bb[PieceType::Pawn as usize]
            | self.role_bb[PieceType::Rook as usize]
            | self.role_bb[PieceType::Queen as usize];

        if force.is_not_empty() {
            return false;
        }

        // Only kings and minor pieces remain. A lone minor can't deliver mate.
        let minors = self.role_bb[PieceType::Knight as usize].popcount() + self.role_bb[PieceType::Bishop as usize].popcount();
        minors <= 1
    }

    /// Full legality of a castling move encoded king→rook (`ksq`→`rsq`) for `color`;
    /// the right must be held, the corridor clear, and the king must neither stand
    /// in nor cross an attacked square.
    ///
    /// The single source of truth shared by move generation and TT-move validation;
    /// a TT collision can hand back a `CASTLE`-flagged move from an unrelated position,
    /// and only the full check (not bare geometry) keeps it from castling through
    /// pieces and corrupting the board. The side is read off the rook's file relative
    /// to the king (FRC-safe); callers first establish that the king sits on `ksq` and
    /// the castling rook on `rsq`.
    #[inline]
    pub fn is_castle_move_legal(&self, color: Color, ksq: Square, rsq: Square) -> bool {
        let queenside = rsq.file() < ksq.file();
        let (mask, data, check_sqs, empty): (u8, &[u8; 4], &[u8], Bitboard) = match (color, queenside) {
            (Color::White, false) => (WHITE_OO, &CASTLE_W_KS, &CASTLE_W_KS_CHECK, W_OO_EMPTY),
            (Color::White, true) => (WHITE_OOO, &CASTLE_W_QS, &CASTLE_W_QS_CHECK, W_OOO_EMPTY),
            (Color::Black, false) => (BLACK_OO, &CASTLE_B_KS, &CASTLE_B_KS_CHECK, B_OO_EMPTY),
            (Color::Black, true) => (BLACK_OOO, &CASTLE_B_QS, &CASTLE_B_QS_CHECK, B_OOO_EMPTY),
        };

        self.castling_rights & mask != 0 && self.is_castle_legal(self.occ, ksq, rsq, data, check_sqs, empty, color.opposite())
    }

    /// Evaluates if a castling maneuver is strictly legal.
    ///
    /// It handles both fast-path standard chess, and the more rigorous
    /// box-checking of Chess960 (FRC) positions.
    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub fn is_castle_legal(
        &self,
        occ: Bitboard,
        ksq: Square,
        rsq: Square,
        data: &[u8; 4],
        check_sqs: &[u8],
        empty_mask: Bitboard,
        opp: Color,
    ) -> bool {
        // ── Fast path: Standard castling with canonical piece placement
        if ksq.0 == data[0] && rsq.0 == data[1] && (occ & empty_mask).is_empty() {
            for &sq in check_sqs {
                if self.is_attacked::<false>(Square(sq), opp, Bitboard(0)) {
                    return false;
                }
            }
            return true;
        }

        // ── Slow path: Chess960 arbitrary placement
        // We use a "1D Bounding Box" simplification. Because all valid castling
        // squares reside exclusively on the 1st or 8th rank, checking the emptiness
        // of all squares between the minimum and maximum of the (king, rook, and
        // destinations) correctly captures the union of all traversed paths for
        // both pieces in a single pass.
        let (king_dst, rook_dst) = (data[2], data[3]);
        let min = ksq.0.min(rsq.0).min(king_dst).min(rook_dst);
        let max = ksq.0.max(rsq.0).max(king_dst).max(rook_dst);

        for sq in min..=max {
            if sq == ksq.0 || sq == rsq.0 {
                continue;
            }
            if occ.check_bit(Square(sq)) {
                return false;
            }
        }

        // The king must not traverse or land on any attacked square.
        let k_lo = ksq.0.min(king_dst);
        let k_hi = ksq.0.max(king_dst);
        let king_bb = Bitboard(1u64 << ksq.0);

        for sq in k_lo..=k_hi {
            // We use VIRTUAL = true, passing the king's current square as mask_out.
            // Otherwise, an enemy slider attacking the traversal path might be
            // incorrectly blocked by the king itself before it moves.
            if self.is_attacked::<true>(Square(sq), opp, king_bb) {
                return false;
            }
        }
        true
    }

    /// Serialize the current position back to a FEN string.
    pub fn as_fen(&self) -> String {
        fen::as_fen(self)
    }

    /// Print a board diagram to stdout.
    pub fn pretty_print(&self) {
        fen::pretty_print(self)
    }

    #[inline(always)]
    pub fn piece_at(&self, sq: Square) -> PieceType {
        self.pieces[sq.0 as usize]
    }

    /// Returns the piece at `sq`, asserting that the square is occupied.
    /// Bypasses debug bounds checking for the hottest inner loops only.
    #[inline(always)]
    pub fn expect_piece_at(&self, sq: Square) -> PieceType {
        debug_assert!(self.occ.check_bit(sq));
        self.piece_at(sq)
    }

    /// Which color owns the piece on `sq`?
    /// Branchless: tests one bit in Black's occupancy.
    /// Only meaningful when the square is actually occupied.
    #[inline(always)]
    pub const fn color_at(&self, sq: Square) -> Color {
        // Black's side_bb bit → 1 (Black), absent → 0 (White).
        // bool as u8 is ABI-guaranteed to produce 0 or 1.
        Color::from(self.side_bb[1].check_bit(sq) as u8)
    }

    /// Bitboard of all pieces of type `pt` belonging to `color`.
    #[inline(always)]
    pub fn pieces(&self, pt: PieceType, color: Color) -> Bitboard {
        self.role_bb[pt as usize] & self.side_bb[color as usize]
    }

    /// All occupied squares.
    #[inline(always)]
    pub fn occupancy(&self) -> Bitboard {
        self.occ
    }

    /// Rough material total (both sides) using standard piece values:
    /// P=1, N=3, B=3, R=5, Q=9.
    #[inline]
    pub const fn material_count(&self) -> u32 {
        let p = self.role_bb[PieceType::Pawn as usize].popcount();
        let n = self.role_bb[PieceType::Knight as usize].popcount();
        let b = self.role_bb[PieceType::Bishop as usize].popcount();
        let r = self.role_bb[PieceType::Rook as usize].popcount();
        let q = self.role_bb[PieceType::Queen as usize].popcount();
        p + 3 * n + 3 * b + 5 * r + 9 * q
    }

    /// Number of pieces of type `pt` on the board (both colors combined).
    #[inline(always)]
    pub const fn piece_count(&self, pt: PieceType) -> i32 {
        if pt.is_none() {
            return 0;
        }
        self.role_bb[pt as usize].popcount() as i32
    }

    /// Is `sq` attacked by any piece of `attacker`'s color?
    ///
    /// `VIRTUAL = true` removes `mask_out` from occupancy,
    /// required for detecting X-ray attacks through the king,
    /// basically sliding check evasion.
    #[inline(always)]
    pub fn is_attacked<const VIRTUAL: bool>(&self, sq: Square, attacker: Color, mask_out: Bitboard) -> bool {
        attacks::is_attacked::<VIRTUAL>(self, sq, attacker, mask_out)
    }

    /// Bitboard of all enemy pieces currently giving check to the side to move.
    #[inline(always)]
    pub fn checkers(&self) -> Bitboard {
        attacks::checkers(self)
    }

    /// All pieces of `attacker`'s color that attack `sq`.
    #[inline(always)]
    pub fn get_attackers_on(&self, sq: Square, attacker: Color) -> Bitboard {
        attacks::attackers_of(self, sq, attacker)
    }

    /// Bulk pawn attack mask: all squares attacked by any pawn of `color`.
    ///
    /// Parallel shift-and-mask:
    /// One operation covers all pawns simultaneously, versus per-pawn lookup.
    /// One of the most-called routines in move gen.
    #[inline]
    pub fn pawn_attacks(&self, color: Color) -> Bitboard {
        let pawns = self.pieces(PieceType::Pawn, color);

        match color {
            Color::White => ((pawns << 9) & !FILE_A) | ((pawns << 7) & !FILE_H),
            Color::Black => ((pawns >> 9) & !FILE_H) | ((pawns >> 7) & !FILE_A),
        }
    }

    /// Every square `color` attacks: the threat map.
    #[inline]
    pub fn threats(&self, color: Color) -> Bitboard {
        let us = self.side_bb[color];
        let empty = Vu64x4::splat(!self.occ.0);

        let knights = (self.role_bb[PieceType::Knight] & us).0;
        let bq = ((self.role_bb[PieceType::Bishop] | self.role_bb[PieceType::Queen]) & us).0;
        let rq = ((self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen]) & us).0;
        let king = (self.role_bb[PieceType::King] & us).0;

        let sliders = spatial::atk_rook(Vu64x4::splat(rq), empty) | spatial::atk_bishop(Vu64x4::splat(bq), empty);
        let leapers = spatial::atk_knight(Vu64x4::splat(knights)) | spatial::atk_king(Vu64x4::splat(king));
        Bitboard((sliders | leapers).extract::<0>()) | self.pawn_attacks(color)
    }

    /// Enemy pieces this quiet move newly threatens: the moved piece on `to`
    /// attacking an enemy worth strictly more than itself: a knight or bishop
    /// onto a rook or queen, a rook onto a queen, a pawn onto anything but a pawn.
    /// Sliders subtract the from-square attacks, so sliding along a line already
    /// raked isn't a fresh threat; a leaper lands on all-new squares and needs no
    /// guard. Queen and king out-rank nothing worth the bonus.
    #[inline]
    pub fn new_threats(&self, pt: PieceType, from: Square, to: Square) -> Bitboard {
        let them = self.side_bb[self.stm.opposite()];
        let rq = (self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen]) & them;
        let q = self.role_bb[PieceType::Queen] & them;

        match pt {
            PieceType::Pawn => {
                let bigger = ((self.role_bb[PieceType::Knight] | self.role_bb[PieceType::Bishop]) & them) | rq;
                bitboard::atk_pawn(to, self.stm) & bigger
            },
            PieceType::Knight => bitboard::atk_knight(to) & rq,
            PieceType::Bishop => Bitboard(bitboard::atk_bishop(to, self.occ).0 & !bitboard::atk_bishop(from, self.occ).0) & rq,
            PieceType::Rook => Bitboard(bitboard::atk_rook(to, self.occ).0 & !bitboard::atk_rook(from, self.occ).0) & q,
            _ => Bitboard(0),
        }
    }

    /// Can any pawn of `color` legally capture en passant on `ep_sq`?
    #[inline]
    pub fn can_capture_ep(&self, ep_sq: Square, color: Color) -> bool {
        attacks::can_capture_ep(self, ep_sq, color)
    }

    /// Friendly pieces of the given color pinned to their king.
    #[inline]
    pub fn pinned_pieces(&self, color: Color) -> Bitboard {
        attacks::pinned_pieces(self, color)
    }

    /// Friendly pieces of the side to move pinned to the king.
    #[inline]
    pub fn king_blockers(&self) -> Bitboard {
        attacks::pinned_pieces(self, self.stm)
    }

    #[inline(always)]
    pub fn set_piece_type(&mut self, sq: Square, pt: PieceType) {
        self.pieces[sq.0 as usize] = pt;
    }

    /// Place a piece on `sq`, updating bitboards, mailbox, and Zobrist hash.
    #[inline(always)]
    pub fn add_piece(&mut self, sq: Square, pt: PieceType, color: Color) {
        make::update_piece::<true>(self, sq, pt, color);
    }

    /// Remove a piece from `sq`, updating bitboards, mailbox, and Zobrist hash.
    #[inline(always)]
    pub fn remove_piece(&mut self, sq: Square, pt: PieceType, color: Color) {
        make::update_piece::<false>(self, sq, pt, color);
    }

    /// Full Zobrist re-computation from scratch, for initialization and
    /// debug verification against the incrementally maintained `self.hash`.
    pub fn calc_zobrist(&self) -> u64 {
        let mut key = 0u64;
        for sq in self.occ {
            key ^= zobrist::key_piece(self.piece_at(sq), self.color_at(sq), sq);
        }

        if self.stm == Color::Black {
            key ^= zobrist::key_side();
        }

        if self.castling_rights != 0 {
            key ^= zobrist::key_castling(self.castling_rights);
        }

        if let Some(ep) = self.en_passant {
            key ^= zobrist::key_ep(ep);
        }
        key
    }

    /// Zobrist hash of all pawns (both colors). Recomputed from scratch, for
    /// initialization and debug verification against the incremental [`Self::pawn_key`].
    pub fn calc_pawn_hash(&self) -> u64 {
        self.calc_subset_hash(self.role_bb[PieceType::Pawn])
    }

    /// Zobrist hash of all minor pieces (knights + bishops, both colors).
    /// Keys minor correction history. Global like `calc_pawn_hash`: the
    /// stm dimension in the table carries perspective.
    pub fn calc_minor_hash(&self) -> u64 {
        self.calc_subset_hash(self.role_bb[PieceType::Knight] | self.role_bb[PieceType::Bishop])
    }

    /// Zobrist hash of all major pieces (rooks + queens, both colors).
    /// Keys major correction history. Global like `calc_pawn_hash`.
    pub fn calc_major_hash(&self) -> u64 {
        self.calc_subset_hash(self.role_bb[PieceType::Rook] | self.role_bb[PieceType::Queen])
    }

    /// XOR-folds each piece's Zobrist key over a subset of the board,
    /// the key schema the correction-history tables index on.
    fn calc_subset_hash(&self, pieces: Bitboard) -> u64 {
        let mut key = 0u64;
        for sq in pieces {
            key ^= zobrist::key_piece(self.piece_at(sq), self.color_at(sq), sq);
        }
        key
    }
}

impl Default for Position {
    fn default() -> Self {
        Self::new()
    }
}
