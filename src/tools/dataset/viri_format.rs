//! Viriformat parser.
//!
//! Each file consists of concatenated games. A game is:
//! - 32-byte PackedBoard header (position + result)
//! - Zero or more (Move, Score) pairs - 4 bytes each (2+2)
//! - Four-byte zero sentinel
//!
//! Scores in the (Move, Score) pairs are white-relative, converted to
//! STM-relative on the way into SoulEntry. The header's score field is a
//! stale placeholder: real evals live in the move pairs.

use std::{
    fs,
    io::{self, Read},
};

use fastrand::Rng;

use super::SoulEntry;
use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
        defs::{Color, PieceType, Square},
        moves::Move,
    },
    engine::wdl::wdl_model,
    weave::Vi16x8,
};

const PACKED_BOARD_SIZE: usize = 32;
const SENTINEL: [u8; 4] = [0, 0, 0, 0];

/// White-relative game outcomes, as the header packs them.
const WDL_DRAW: u8 = 1;
const WDL_WIN: u8 = 2;

/// The filter `viriformat::dataformat::Filter` describes, over the replay this
/// module already walks.
///
/// Field names, defaults and predicate order follow that crate, so a filter file
/// written for it reads here. Three departures: the WDL gate asks Soul's own
/// model rather than carrying a foreign fit, so the four coefficient fields are
/// gone; the sampling draws are seeded rather than thread-local, so a config and
/// a file pin the same set; and the 33-entry table is a `Vec` because serde stops
/// deriving at 32.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct ReplayFilter {
    pub min_ply: u32,
    pub min_pieces: u32,
    pub max_eval: u32,
    pub filter_tactical: bool,
    pub filter_check: bool,
    pub filter_castling: bool,
    pub max_eval_incorrectness: u32,
    pub random_fen_skipping: bool,
    pub random_fen_skip_probability: f64,
    pub wdl_filtered: bool,
    pub material_count_filtered: bool,
    /// Skip probability per piece count, indexed 0..=32. Short tables read as
    /// zero, so a file may stop early.
    pub material_count_probabilities: Vec<f64>,
    /// Seeds every probabilistic gate.
    pub seed: u64,
}

impl Default for ReplayFilter {
    fn default() -> Self {
        Self {
            min_ply: 16,
            min_pieces: 4,
            max_eval: 31339,
            filter_tactical: true,
            filter_check: true,
            filter_castling: false,
            max_eval_incorrectness: u32::MAX,
            random_fen_skipping: false,
            random_fen_skip_probability: 0.0,
            wdl_filtered: false,
            material_count_filtered: false,
            material_count_probabilities: Vec::new(),
            seed: 0,
        }
    }
}

impl ReplayFilter {
    /// Every gate off, the shape a loader wants when no filter file was named.
    pub const UNRESTRICTED: Self = Self {
        min_ply: 0,
        min_pieces: 0,
        max_eval: u32::MAX,
        filter_tactical: false,
        filter_check: false,
        filter_castling: false,
        max_eval_incorrectness: u32::MAX,
        random_fen_skipping: false,
        random_fen_skip_probability: 0.0,
        wdl_filtered: false,
        material_count_filtered: false,
        material_count_probabilities: Vec::new(),
        seed: 0,
    };

    /// `eval` and `wdl` are White-relative, as they sit in the file. The STM flip
    /// happens after a position survives, so it cannot reach the predicates.
    fn should_filter(&self, pos: &Position, mv: Move, eval: i32, wdl: u8, ply: u32, rng: &mut Rng) -> bool {
        if ply < self.min_ply {
            return true;
        }

        if eval.unsigned_abs() >= self.max_eval {
            return true;
        }

        if pos.occupancy().popcount() < self.min_pieces {
            return true;
        }

        if self.filter_tactical && mv.is_tactical() {
            return true;
        }

        if self.filter_check && pos.checkers().is_not_empty() {
            return true;
        }

        if self.filter_castling && mv.is_castling() {
            return true;
        }

        if self.random_fen_skipping && rng.f64() < self.random_fen_skip_probability {
            return true;
        }

        if self.wdl_filtered && rng.f64() < 1.0 - Self::result_chance(eval, material_count(pos), wdl) {
            return true;
        }

        if self.material_count_filtered {
            let index = pos.occupancy().popcount().min(32) as usize;

            if rng.f64() < self.material_count_probabilities.get(index).copied().unwrap_or(0.0) {
                return true;
            }
        }

        self.eval_is_incorrect(eval, wdl)
    }

    /// A draw scored far from a draw, or a winner whose eval went negative by more
    /// than the bound. Clamping to zero is what keeps a merely small winning score
    /// from counting.
    fn eval_is_incorrect(&self, eval: i32, wdl: u8) -> bool {
        if self.max_eval_incorrectness == u32::MAX {
            return false;
        }

        if wdl == WDL_DRAW {
            return eval.unsigned_abs() > self.max_eval_incorrectness;
        }

        let winner_pov_eval = if wdl == WDL_WIN { eval } else { -eval };

        winner_pov_eval.min(0).unsigned_abs() > self.max_eval_incorrectness
    }

    /// How likely the eval said this game's actual result was.
    ///
    /// Both arguments are White-relative, and the model is symmetric in the score's
    /// sign, so reading the White-side probability against the White-side outcome is
    /// consistent without a flip.
    fn result_chance(eval: i32, material: u32, wdl: u8) -> f64 {
        let (win, draw, loss) = wdl_model(eval, material);

        match wdl {
            WDL_WIN => win,
            WDL_DRAW => draw,
            _ => loss,
        }
    }
}

/// Classical 1/3/3/5/9 over both sides, kings excluded: the scale `wdl_model`
/// clamps into 17..=78.
fn material_count(pos: &Position) -> u32 {
    let weighted = pos.piece_count(PieceType::Pawn)
        + 3 * pos.piece_count(PieceType::Knight)
        + 3 * pos.piece_count(PieceType::Bishop)
        + 5 * pos.piece_count(PieceType::Rook)
        + 9 * pos.piece_count(PieceType::Queen);

    weighted.unsigned_abs()
}

pub fn parse_viri_file(path: &str, filter: &ReplayFilter) -> io::Result<Vec<SoulEntry>> {
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut entries = Vec::new();
    let mut pos = 0usize;
    let mut rng = Rng::with_seed(filter.seed);

    while pos + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[pos..pos + PACKED_BOARD_SIZE];
        pos += PACKED_BOARD_SIZE;

        let Some((mut position, game_result, mut ply)) = parse_packed_board(header) else {
            break;
        };

        loop {
            if pos + 4 > data.len() {
                break;
            }

            let candidate = &data[pos..pos + 4];

            if candidate == SENTINEL {
                pos += 4;
                break;
            }

            let viri_move = u16::from_le_bytes([candidate[0], candidate[1]]);
            let viri_score = i16::from_le_bytes([candidate[2], candidate[3]]);
            pos += 4;

            let Some(soul_move) = viri_to_soul_move(viri_move, &position) else {
                break;
            };

            if !filter.should_filter(&position, soul_move, i32::from(viri_score), game_result, ply, &mut rng) {
                entries.push(SoulEntry::from_board(
                    &position,
                    f64::from(stm_result(game_result, position.stm)) / 2.0,
                    Some(relative_score(viri_score, position.stm)),
                ));
            }

            ply += 1;

            let mut acc = Vi16x8::zero();
            position.make_move(soul_move, &mut acc);
        }
    }

    Ok(entries)
}

/// Convert a white-relative viri score to STM-relative `i32`.
fn relative_score(viri_score: i16, stm: Color) -> i32 {
    let s = i32::from(viri_score);
    if stm == Color::Black { -s } else { s }
}

/// Convert a white-relative viri result (0=black win, 1=draw, 2=white win)
/// to an STM-relative result (0=loss, 1=draw, 2=win).
fn stm_result(viri_result: u8, stm: Color) -> u8 {
    if stm == Color::Black { 2 - viri_result } else { viri_result }
}

/// Also returns the opening's ply, since `min_ply` counts from the start of chess
/// rather than the start of the record: an opening minted eight plies deep enters
/// its game already at eight.
fn parse_packed_board(data: &[u8]) -> Option<(Position, u8, u32)> {
    if data.len() < PACKED_BOARD_SIZE {
        return None;
    }

    let occupancy = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let pieces = &data[8..24]; // [u4; 32] packed as [u8; 16]
    let stm_ep = data[24];
    let _halfmove = data[25];
    let fullmove = u16::from_le_bytes([data[26], data[27]]);
    let _score = i16::from_le_bytes([data[28], data[29]]);
    let result = data[30];
    let _extra = data[31];

    if result > 2 {
        return None;
    }

    let stm = if stm_ep & 0x80 != 0 { Color::Black } else { Color::White };
    let ep = stm_ep & 0x7F;
    let en_passant = if ep < 64 { Some(Square(ep)) } else { None };

    let mut pos = Position::new();

    pos.stm = stm;
    pos.en_passant = en_passant;

    // Set castling-rook home squares for standard chess (FRC handled later).
    pos.castling_rooks[ROOK_W_KS] = Square(7); // h1
    pos.castling_rooks[ROOK_W_QS] = Square(0); // a1
    pos.castling_rooks[ROOK_B_KS] = Square(63); // h8
    pos.castling_rooks[ROOK_B_QS] = Square(56); // a8

    let mut white_king = None;
    let mut black_king = None;
    let mut unmoved_rooks: [Vec<u8>; 2] = [Vec::new(), Vec::new()];

    let mut occ = occupancy;
    let mut piece_idx: usize = 0;

    while occ != 0 {
        let sq_idx = occ.trailing_zeros() as usize;
        occ &= occ - 1;

        if piece_idx >= 32 {
            break;
        }

        let nibble = if piece_idx.is_multiple_of(2) { pieces[piece_idx / 2] & 0x0F } else { pieces[piece_idx / 2] >> 4 };
        piece_idx += 1;

        let viri_type = nibble & 0x07;
        let color = if nibble & 0x08 != 0 { Color::Black } else { Color::White };

        let pt = match viri_type {
            0 => PieceType::Pawn,
            1 => PieceType::Knight,
            2 => PieceType::Bishop,
            3 => PieceType::Rook,
            4 => PieceType::Queen,
            5 => PieceType::King,
            6 => PieceType::Rook,
            _ => continue,
        };

        let sq = Square(sq_idx as u8);
        pos.add_piece(sq, pt, color);

        if pt == PieceType::King {
            match color {
                Color::White => white_king = Some(sq),
                Color::Black => black_king = Some(sq),
            }
        }

        if viri_type == 6 {
            // Unmoved rook: record for castling-rights detection.
            unmoved_rooks[color as usize].push(sq_idx as u8);
        }
    }

    // Reconstruct castling rights from unmoved rooks relative to kings.
    let mut set_castling_rights = 0u8;

    if let Some(king_sq) = white_king {
        for &rook_idx in &unmoved_rooks[Color::White as usize] {
            let rook = Square(rook_idx);

            if rook.file() > king_sq.file() {
                set_castling_rights |= WHITE_OO;
                pos.castling_rooks[ROOK_W_KS] = rook;
            } else {
                set_castling_rights |= WHITE_OOO;
                pos.castling_rooks[ROOK_W_QS] = rook;
            }
        }
    }

    if let Some(king_sq) = black_king {
        for &rook_idx in &unmoved_rooks[Color::Black as usize] {
            let rook = Square(rook_idx);

            if rook.file() > king_sq.file() {
                set_castling_rights |= BLACK_OO;
                pos.castling_rooks[ROOK_B_KS] = rook;
            } else {
                set_castling_rights |= BLACK_OOO;
                pos.castling_rooks[ROOK_B_QS] = rook;
            }
        }
    }

    pos.castling_rights = set_castling_rights;

    // Mark as FRC if any castling rook is off its standard home square.
    pos.is_frc = pos.castling_rooks[ROOK_W_KS] != Square(7)
        || pos.castling_rooks[ROOK_W_QS] != Square(0)
        || pos.castling_rooks[ROOK_B_KS] != Square(63)
        || pos.castling_rooks[ROOK_B_QS] != Square(56);

    pos.hash = pos.calc_zobrist();
    pos.pawn_key = pos.calc_pawn_hash();
    pos.minor_key = pos.calc_minor_hash();
    pos.major_key = pos.calc_major_hash();

    let ply = u32::from(fullmove).saturating_sub(1) * 2 + u32::from(stm == Color::Black);

    Some((pos, result, ply))
}

/// Convert a viriformat 16-bit move to a Soul `Move`.
///
/// Viri layout:
/// - bits  0..5  → from square (6 bits)
/// - bits  6..11 → to square (6 bits)
/// - bits 12..13 → promotion piece (0=Knight, 1=Bishop, 2=Rook, 3=Queen)
/// - bits 14..15 → move type (0=normal, 1=en-passant, 2=castling, 3=promotion)
///
/// Castling is king-takes-rook (same as Soul's internal encoding).
fn viri_to_soul_move(viri_move: u16, pos: &Position) -> Option<Move> {
    let from = Square((viri_move & 0x3F) as u8);
    let to = Square(((viri_move >> 6) & 0x3F) as u8);
    let promo_piece = (viri_move >> 12) & 0x3;
    let move_type = (viri_move >> 14) & 0x3;

    if from.0 >= 64 || to.0 >= 64 {
        return None;
    }

    let moving_piece = pos.piece_at(from);

    if moving_piece == PieceType::None {
        return None;
    }

    let capture = pos.piece_at(to) != PieceType::None;

    let flag: u16 = match move_type {
        0 => {
            if capture {
                Move::CAPTURE
            } else if moving_piece == PieceType::Pawn && from.file() == to.file() && from.rank().abs_diff(to.rank()) == 2 {
                Move::DOUBLE_PUSH
            } else {
                Move::QUIET
            }
        },
        1 => Move::EP_CAPTURE,
        2 => Move::CASTLE,
        3 => {
            // Promotion flag values embed the capture bit in bit 0.
            match promo_piece {
                0 => Move::PROM_N | if capture { 1 } else { 0 },
                1 => Move::PROM_B | if capture { 1 } else { 0 },
                2 => Move::PROM_R | if capture { 1 } else { 0 },
                3 => Move::PROM_Q | if capture { 1 } else { 0 },
                _ => return None,
            }
        },
        _ => return None,
    };

    Some(Move::new(from, to, flag))
}

#[cfg(test)]
mod tests {
    use super::{Move, Position, ReplayFilter, Rng, WDL_DRAW, WDL_WIN};
    use crate::engine::movegen::gen_legal_moves;

    /// White to move with Bxf7+ available and the short castle still legal, so one
    /// position offers a capture, a castle and many quiet moves.
    const OPEN: &str = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    /// Black is in check from the bishop.
    const IN_CHECK: &str = "r1bqkbnr/pppp1Bpp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4";

    const WDL_LOSS: u8 = 0;

    fn pick(fen: &str, want: impl Fn(Move) -> bool) -> (Position, Move) {
        let pos = Position::from_fen(fen);
        let mv = gen_legal_moves(&pos)
            .iter()
            .find(|&&m| want(m))
            .copied()
            .expect("a move of the asked shape");

        (pos, mv)
    }

    fn keeps(filter: &ReplayFilter, fen: &str, mv: Move, eval: i32, wdl: u8, ply: u32) -> bool {
        let pos = Position::from_fen(fen);

        !filter.should_filter(&pos, mv, eval, wdl, ply, &mut Rng::with_seed(1))
    }

    fn quiet_move(fen: &str) -> Move {
        pick(fen, |m| !m.is_tactical() && !m.is_castling()).1
    }

    #[test]
    fn unrestricted_keeps_everything() {
        let filter = ReplayFilter::UNRESTRICTED;
        let capture = pick(OPEN, Move::is_tactical).1;

        assert!(keeps(&filter, OPEN, quiet_move(OPEN), 0, WDL_DRAW, 0));
        assert!(keeps(&filter, OPEN, capture, 30_000, WDL_WIN, 0));
        assert!(keeps(&filter, IN_CHECK, quiet_move(IN_CHECK), -30_000, WDL_LOSS, 0));
    }

    #[test]
    fn min_ply_and_min_pieces_gate_separately() {
        let by_ply = ReplayFilter { min_ply: 8, ..ReplayFilter::UNRESTRICTED };

        assert!(!keeps(&by_ply, OPEN, quiet_move(OPEN), 0, WDL_DRAW, 7));
        assert!(keeps(&by_ply, OPEN, quiet_move(OPEN), 0, WDL_DRAW, 8));

        let on_board = Position::from_fen(OPEN).occupancy().popcount();
        let by_pieces = ReplayFilter { min_pieces: on_board + 1, ..ReplayFilter::UNRESTRICTED };

        assert!(!keeps(&by_pieces, OPEN, quiet_move(OPEN), 0, WDL_DRAW, 0));
        assert!(keeps(
            &ReplayFilter { min_pieces: on_board, ..ReplayFilter::UNRESTRICTED },
            OPEN,
            quiet_move(OPEN),
            0,
            WDL_DRAW,
            0
        ));
    }

    #[test]
    fn max_eval_is_exclusive() {
        let filter = ReplayFilter { max_eval: 450, ..ReplayFilter::UNRESTRICTED };

        assert!(keeps(&filter, OPEN, quiet_move(OPEN), 449, WDL_DRAW, 0));
        assert!(!keeps(&filter, OPEN, quiet_move(OPEN), 450, WDL_DRAW, 0), "the bound itself is filtered");
        assert!(!keeps(&filter, OPEN, quiet_move(OPEN), -450, WDL_DRAW, 0), "and it reads the magnitude");
    }

    #[test]
    fn tactical_check_and_castling_gate_independently() {
        let quiet = quiet_move(OPEN);
        let capture = pick(OPEN, Move::is_tactical).1;
        let castle = pick(OPEN, Move::is_castling).1;

        let tactical = ReplayFilter { filter_tactical: true, ..ReplayFilter::UNRESTRICTED };
        assert!(!keeps(&tactical, OPEN, capture, 0, WDL_DRAW, 0));
        assert!(keeps(&tactical, IN_CHECK, quiet_move(IN_CHECK), 0, WDL_DRAW, 0), "in check, but the flag is off");

        let check = ReplayFilter { filter_check: true, ..ReplayFilter::UNRESTRICTED };
        assert!(!keeps(&check, IN_CHECK, quiet_move(IN_CHECK), 0, WDL_DRAW, 0));
        assert!(keeps(&check, OPEN, capture, 0, WDL_DRAW, 0), "a capture, but the flag is off");

        let castling = ReplayFilter { filter_castling: true, ..ReplayFilter::UNRESTRICTED };
        assert!(!keeps(&castling, OPEN, castle, 0, WDL_DRAW, 0));
        assert!(keeps(&castling, OPEN, quiet, 0, WDL_DRAW, 0));
    }

    #[test]
    fn eval_incorrectness_is_read_from_the_winner() {
        let filter = ReplayFilter { max_eval_incorrectness: 400, ..ReplayFilter::UNRESTRICTED };
        let quiet = quiet_move(OPEN);

        assert!(!keeps(&filter, OPEN, quiet, -401, WDL_WIN, 0));
        assert!(keeps(&filter, OPEN, quiet, -400, WDL_WIN, 0));
        assert!(keeps(&filter, OPEN, quiet, 30_000, WDL_WIN, 0), "a winning score for the winner is never incorrect");

        // Eval is White-relative, so a Black win puts the winner at -401 here.
        assert!(!keeps(&filter, OPEN, quiet, 401, WDL_LOSS, 0));
        assert!(keeps(&filter, OPEN, quiet, -30_000, WDL_LOSS, 0));

        assert!(!keeps(&filter, OPEN, quiet, 401, WDL_DRAW, 0));
        assert!(!keeps(&filter, OPEN, quiet, -401, WDL_DRAW, 0));
        assert!(keeps(&filter, OPEN, quiet, 400, WDL_DRAW, 0));
    }

    #[test]
    fn skipping_probabilities_read_as_drop_chances() {
        let quiet = quiet_move(OPEN);

        let always = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 1.0, ..ReplayFilter::UNRESTRICTED };
        let never = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 0.0, ..ReplayFilter::UNRESTRICTED };

        assert!(!keeps(&always, OPEN, quiet, 0, WDL_DRAW, 0), "probability 1.0 drops");
        assert!(keeps(&never, OPEN, quiet, 0, WDL_DRAW, 0), "probability 0.0 keeps");

        let table = ReplayFilter {
            material_count_filtered: true,
            material_count_probabilities: vec![1.0; 33],
            ..ReplayFilter::UNRESTRICTED
        };

        assert!(!keeps(&table, OPEN, quiet, 0, WDL_DRAW, 0));
    }

    /// The gate reads the outcome the game actually had, so a won game scored as
    /// winning is near-certain and survives, while the same score against a loss is
    /// a near-zero chance and does not.
    #[test]
    fn the_wdl_gate_keeps_what_the_eval_predicted() {
        let filter = ReplayFilter { wdl_filtered: true, ..ReplayFilter::UNRESTRICTED };
        let quiet = quiet_move(OPEN);

        assert!(keeps(&filter, OPEN, quiet, 2000, WDL_WIN, 0), "a won game scored as won");
        assert!(!keeps(&filter, OPEN, quiet, 2000, WDL_LOSS, 0), "the same score against a loss");
    }

    #[test]
    fn the_defaults_are_the_documented_ones() {
        let filter = ReplayFilter::default();

        assert_eq!(filter.min_ply, 16);
        assert_eq!(filter.min_pieces, 4);
        assert_eq!(filter.max_eval, 31339);
        assert!(filter.filter_tactical && filter.filter_check && !filter.filter_castling);
        assert_eq!(filter.max_eval_incorrectness, u32::MAX);
        assert!(!filter.wdl_filtered && !filter.material_count_filtered && !filter.random_fen_skipping);
    }
}
