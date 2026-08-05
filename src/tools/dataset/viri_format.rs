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

use super::{SoulEntry, flip_result};
use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
        defs::{Color, MATE_BOUND, PieceType, Square},
        moves::Move,
    },
    engine::wdl::wdl_model,
};

const PACKED_BOARD_SIZE: usize = 32;
const SENTINEL: [u8; 4] = [0, 0, 0, 0];

/// A last score this far from equality means the generator stopped on a won position.
pub const DECISIVE_ENDING: i32 = 1000;
/// A last score this near equality means the generator stopped on a drawn one.
pub const QUIET_ENDING: i32 = 50;

/// White-relative game outcomes, as the header packs them.
const WDL_DRAW: u8 = 1;
const WDL_WIN: u8 = 2;

/// The filter `viriformat::dataformat::Filter` describes, over the replay this
/// module already walks.
///
/// Field names, defaults and predicate order follow that crate, so a filter file
/// written for it reads here. Three departures: the WDL gate asks Soul's own model
/// rather than carrying a foreign fit, so the four coefficient fields are gone;
/// the two gates that reshape the mix weight a position instead of dropping it;
/// and the 33-entry table is a `Vec` because serde stops deriving at 32.
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
    /// Applied per epoch by the tuner's sampler, not during the replay: a drop
    /// decided at load would take that share away for the whole run.
    pub random_fen_skipping: bool,
    pub random_fen_skip_probability: f64,
    pub wdl_filtered: bool,
    pub material_count_filtered: bool,
    /// Skip probability per piece count, indexed 0..=32. Short tables read as
    /// zero, so a file may stop early.
    pub material_count_probabilities: Vec<f64>,
    /// Carried for parity with the reference `Filter`; nothing here draws from it.
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
    fn should_filter(&self, pos: &Position, mv: Move, eval: i32, wdl: u8, ply: u32) -> bool {
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

        self.eval_is_incorrect(eval, wdl)
    }

    /// How much a kept position counts for, from the gates that reshape the mix.
    ///
    /// Keeping it with chance `p` and counting it once is, in expectation, counting
    /// it `p`; the weight is that without the variance, and without the loss.
    fn sample_weight(&self, pos: &Position, eval: i32, wdl: u8) -> f64 {
        let mut weight = 1.0;

        if self.wdl_filtered {
            weight *= Self::result_chance(eval, pos.material_count(), wdl);
        }

        if self.material_count_filtered {
            let index = pos.occupancy().popcount().min(32) as usize;
            weight *= 1.0 - self.material_count_probabilities.get(index).copied().unwrap_or(0.0);
        }
        weight
    }

    pub fn weights_positions(&self) -> bool {
        self.wdl_filtered || self.material_count_filtered
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
    /// `wdl_model` is written for an STM-relative score, and gets a White-relative
    /// one here. It is sign-symmetric, so negating the score swaps win and loss:
    /// a White-relative score yields White-relative probabilities, which is the
    /// perspective `wdl` is already in.
    fn result_chance(eval: i32, material: u32, wdl: u8) -> f64 {
        let (win, draw, loss) = wdl_model(eval, material);
        match wdl {
            WDL_WIN => win,
            WDL_DRAW => draw,
            _ => loss,
        }
    }
}

/// The weights are one per kept position, empty when no gate weighs anything. The third return is
/// one count per game that kept anything, in file order and summing to the entries: a game is the
/// independent unit of a replay, and only this walk knows where one ends.
pub fn parse_viri_file(path: &str, filter: &ReplayFilter) -> io::Result<(Vec<SoulEntry>, Vec<f32>, Vec<u32>)> {
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut entries = Vec::new();
    let mut weights = Vec::new();
    let mut games = Vec::new();
    let mut pos = 0usize;
    let mut seen = 0usize;

    while pos + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[pos..pos + PACKED_BOARD_SIZE];
        pos += PACKED_BOARD_SIZE;

        let Some((mut position, game_result, mut ply)) = parse_packed_board(header) else {
            break;
        };

        // `make_move` updates the accumulator as deltas off its current value, so
        // the base has to be the real accumulator, not a zero.
        let mut acc = position.get_initial_accumulator();
        let kept_before = entries.len();

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

            seen += 1;

            if !filter.should_filter(&position, soul_move, i32::from(viri_score), game_result, ply) {
                entries.push(SoulEntry::from_board(
                    &position,
                    f64::from(flip_result(game_result, position.stm)) / 2.0,
                    Some(relative_score(viri_score, position.stm)),
                ));

                if filter.weights_positions() {
                    weights.push(filter.sample_weight(&position, i32::from(viri_score), game_result) as f32);
                }
            }
            ply += 1;
            position.make_move(soul_move, &mut acc);
        }

        if entries.len() > kept_before {
            games.push((entries.len() - kept_before) as u32);
        }
    }

    let share = if seen == 0 { 0.0 } else { 100.0 * entries.len() as f64 / seen as f64 };
    println!("  Replayed {seen} positions, kept {} ({share:.1}%) from {} games", entries.len(), games.len());
    Ok((entries, weights, games))
}

/// How the games in a replay end, which the stream of positions cannot say.
///
/// A file whose games stop in the middlegame holds no endgames and few material
/// imbalances, so a fit against outcomes has nothing to sit on.
///
/// The three ending counts bucket a game by its last recorded score, and a game in none
/// of them stopped for a reason that was not its score.
pub struct GameScan {
    pub games: usize,
    pub plies: u64,
    /// Pieces standing at each game's last position, summed.
    pub pieces_left: u64,
    /// Outcomes by the header's white-relative WDL code.
    pub results: [usize; 3],
    pub mate_endings: usize,
    pub decisive_endings: usize,
    pub quiet_endings: usize,
}

/// Walk every game for its ending, filter ignored: a scan describes the file as it is.
///
/// # Errors
/// Returns an error if the file cannot be read.
pub fn scan_viri_games(path: &str) -> io::Result<GameScan> {
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut scan = GameScan {
        games: 0,
        plies: 0,
        pieces_left: 0,
        results: [0; 3],
        mate_endings: 0,
        decisive_endings: 0,
        quiet_endings: 0,
    };

    let mut pos = 0usize;
    while pos + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[pos..pos + PACKED_BOARD_SIZE];
        pos += PACKED_BOARD_SIZE;

        let Some((mut position, game_result, _)) = parse_packed_board(header) else {
            break;
        };

        let mut acc = position.get_initial_accumulator();
        let mut last_score = 0i32;
        let mut plies = 0u64;

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
            last_score = i32::from(i16::from_le_bytes([candidate[2], candidate[3]]));
            pos += 4;

            let Some(soul_move) = viri_to_soul_move(viri_move, &position) else {
                break;
            };

            plies += 1;
            position.make_move(soul_move, &mut acc);
        }

        scan.games += 1;
        scan.plies += plies;
        scan.pieces_left += u64::from(position.occupancy().popcount());
        scan.results[usize::from(game_result.min(2))] += 1;

        let magnitude = last_score.abs();
        if magnitude >= MATE_BOUND {
            scan.mate_endings += 1;
        } else if magnitude >= DECISIVE_ENDING {
            scan.decisive_endings += 1;
        } else if magnitude <= QUIET_ENDING {
            scan.quiet_endings += 1;
        }
    }
    Ok(scan)
}

/// Convert a white-relative viri score to STM-relative `i32`.
fn relative_score(viri_score: i16, stm: Color) -> i32 {
    let s = i32::from(viri_score);
    if stm == Color::Black { -s } else { s }
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

    // Slots start unset and only the type-6 markers below fill them, so a slot
    // no marker reaches is a right that died.
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

    // Reconstruct rights from the type-6 markers: the writer only marks a rook
    // while its right is live, so the marker's file against the king's decides
    // the side exactly.
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

    // An unmarked slot holds Square(0), which is not a castling rook, so the
    // off-home test only applies to slots whose right bit is set.
    pos.is_frc = (pos.castling_rights & WHITE_OO != 0 && pos.castling_rooks[ROOK_W_KS] != Square(7))
        || (pos.castling_rights & WHITE_OOO != 0 && pos.castling_rooks[ROOK_W_QS] != Square(0))
        || (pos.castling_rights & BLACK_OO != 0 && pos.castling_rooks[ROOK_B_KS] != Square(63))
        || (pos.castling_rights & BLACK_OOO != 0 && pos.castling_rooks[ROOK_B_QS] != Square(56));

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
            let capture_bit = u16::from(capture);

            match promo_piece {
                0 => Move::PROM_N | capture_bit,
                1 => Move::PROM_B | capture_bit,
                2 => Move::PROM_R | capture_bit,
                3 => Move::PROM_Q | capture_bit,
                _ => return None,
            }
        },
        _ => return None,
    };

    Some(Move::new(from, to, flag))
}

#[cfg(test)]
mod tests {
    use super::{Move, Position, ReplayFilter, WDL_DRAW, WDL_WIN, parse_packed_board, viri_to_soul_move};
    use crate::{
        core::{
            board::{ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
            defs::{Color, PieceType, Square},
        },
        engine::movegen::gen_legal_moves,
        tools::dataset::{SoulEntry, quant},
    };

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
        !filter.should_filter(&pos, mv, eval, wdl, ply)
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
    fn decimation_does_not_reach_the_replay() {
        let quiet = quiet_move(OPEN);
        let certain = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 1.0, ..ReplayFilter::UNRESTRICTED };
        assert!(keeps(&certain, OPEN, quiet, 0, WDL_DRAW, 0), "a drop chance of 1.0 must still load");
    }

    #[test]
    fn the_reshaping_gates_weight_instead_of_dropping() {
        let quiet = quiet_move(OPEN);
        let table = ReplayFilter {
            material_count_filtered: true,
            material_count_probabilities: vec![0.75; 33],
            ..ReplayFilter::UNRESTRICTED
        };

        assert!(keeps(&table, OPEN, quiet, 0, WDL_DRAW, 0), "a reshaping gate must not drop");
        assert!((table.sample_weight(&Position::from_fen(OPEN), 0, WDL_DRAW) - 0.25).abs() < 1e-12);
        assert!((ReplayFilter::UNRESTRICTED.sample_weight(&Position::from_fen(OPEN), 0, WDL_DRAW) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_wdl_gate_weighs_what_the_eval_predicted() {
        let filter = ReplayFilter { wdl_filtered: true, ..ReplayFilter::UNRESTRICTED };
        let pos = Position::from_fen(OPEN);
        let won = filter.sample_weight(&pos, 2000, WDL_WIN);
        let lost = filter.sample_weight(&pos, 2000, WDL_LOSS);
        assert!(won > 0.9, "a won game scored as won weighed {won}");
        assert!(lost < 0.1, "the same score against a loss weighed {lost}");
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

    fn wire_from_entry(entry: &SoulEntry, halfmove: u8, fullmove: u16, extra: u8) -> [u8; 32] {
        let mut wire = [0u8; 32];
        wire[0..8].copy_from_slice(&entry.occupancy.to_le_bytes());
        wire[8..24].copy_from_slice(&entry.pieces);
        wire[24] = entry.stm_and_ep;
        wire[25] = halfmove;
        wire[26..28].copy_from_slice(&fullmove.to_le_bytes());
        wire[28..30].copy_from_slice(&entry.score.to_le_bytes());
        wire[30] = entry.result;
        wire[31] = extra;
        wire
    }

    fn assert_position_equal(a: &Position, b: &Position) {
        assert_eq!(a.stm, b.stm);
        assert_eq!(a.en_passant, b.en_passant);
        assert_eq!(a.castling_rights, b.castling_rights);
        assert_eq!(a.castling_rooks, b.castling_rooks);
        assert_eq!(a.is_frc, b.is_frc);
        assert_eq!(a.occupancy(), b.occupancy());
        for sq_idx in 0..64 {
            let sq = Square(sq_idx as u8);
            assert_eq!(a.piece_at(sq), b.piece_at(sq), "piece on {sq:?}");
            assert_eq!(a.color_at(sq), b.color_at(sq), "color on {sq:?}");
        }
    }

    /// Packs through the production [`quant::from_board`], so the round trip covers
    /// the real writer as well as the decoder.
    fn assert_wire_roundtrip(fen: &str, result: f64, score: i32, halfmove: u8, fullmove: u16) {
        let pos = Position::from_fen(fen);
        let entry = quant::from_board(&pos, result, Some(score));
        let (decoded, game_result, ply) =
            parse_packed_board(&wire_from_entry(&entry, halfmove, fullmove, 0)).expect("a test-written wire decodes");

        assert_position_equal(&pos, &decoded);
        assert_eq!(game_result, (result * 2.0) as u8);
        let expected_ply = (u32::from(fullmove) - 1) * 2 + u32::from(decoded.stm == Color::Black);
        assert_eq!(ply, expected_ply);
    }

    #[test]
    fn packed_board_round_trips_through_the_decoder() {
        // All four rights and an en-passant target.
        assert_wire_roundtrip("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 1.0, 120, 0, 32);
        // Black to move.
        assert_wire_roundtrip("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", 0.5, -40, 7, 9);
        // No rights at all.
        assert_wire_roundtrip("4k3/8/8/8/8/8/P6p/4K3 w - - 0 1", 1.0, 10_000, 3, 44);
    }

    #[test]
    fn unmoved_rooks_rebuild_frc_rights_slots() {
        // King d1 with rooks b1 and f1: neither sits on a home square, so the
        // file-vs-king comparison alone has to assign both slots.
        let mut pos = Position::new();
        pos.add_piece(Square(1), PieceType::Rook, Color::White);
        pos.add_piece(Square(3), PieceType::King, Color::White);
        pos.add_piece(Square(5), PieceType::Rook, Color::White);
        pos.add_piece(Square(60), PieceType::King, Color::Black);
        pos.stm = Color::White;
        pos.castling_rights = WHITE_OO | WHITE_OOO;
        pos.castling_rooks[ROOK_W_QS] = Square(1);
        pos.castling_rooks[ROOK_W_KS] = Square(5);
        pos.is_frc = true;

        let entry = quant::from_board(&pos, 1.0, Some(0));
        let (decoded, ..) = parse_packed_board(&wire_from_entry(&entry, 0, 3, 0)).expect("well-formed wire");
        assert_position_equal(&pos, &decoded);
        assert_eq!(decoded.castling_rights, WHITE_OO | WHITE_OOO);
        assert_eq!(decoded.castling_rooks[ROOK_W_QS], Square(1));
        assert_eq!(decoded.castling_rooks[ROOK_W_KS], Square(5));
    }

    const EN_PASSANT: &str = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
    const PROMOTES: &str = "1n5k/P7/8/8/8/8/8/K7 w - - 0 1";

    /// A move and the position it is legal in.
    fn move_shape(fen: &str, want: impl Fn(Move) -> bool) -> (Position, Move) {
        let pos = Position::from_fen(fen);
        let mv = gen_legal_moves(&pos)
            .iter()
            .find(|&&m| want(m))
            .copied()
            .expect("a move of the asked shape");

        (pos, mv)
    }

    /// Encode a Soul move by the format's bit layout, decode it back, and
    /// require the same move.
    fn assert_move_roundtrip(fen: &str, want: impl Fn(Move) -> bool) {
        let (pos, mv) = move_shape(fen, want);
        let promo = mv.promo().map_or(0, |p| match p {
            PieceType::Knight => 0,
            PieceType::Bishop => 1,
            PieceType::Rook => 2,
            _ => 3,
        });
        let (ty, promo_bits) = if mv.is_castling() {
            (2u16, 0u16)
        } else if mv.is_en_passant() {
            (1, 0)
        } else if mv.is_promotion() {
            (3, promo)
        } else {
            (0, 0)
        };

        let wire = u16::from(mv.from().0) | (u16::from(mv.to().0) << 6) | (promo_bits << 12) | (ty << 14);
        assert_eq!(viri_to_soul_move(wire, &pos), Some(mv), "round trip of {mv:?}");
    }

    #[test]
    fn viri_moves_round_trip_through_every_flag() {
        assert_move_roundtrip(OPEN, |m| !m.is_tactical() && !m.is_castling());
        assert_move_roundtrip(OPEN, Move::is_tactical);
        assert_move_roundtrip(OPEN, Move::is_castling);
        assert_move_roundtrip(EN_PASSANT, |m| m.is_en_passant());
        assert_move_roundtrip("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", |m| m.is_double_push());
        assert_move_roundtrip(PROMOTES, |m| m.is_promotion() && m.promo() == Some(PieceType::Queen));
        assert_move_roundtrip(PROMOTES, |m| m.is_promotion() && m.is_tactical() && m.promo() == Some(PieceType::Bishop));
    }

    /// The capture flag is derived from the board, not the encoded type, so a
    /// Black capture proves the decoder sees the side as well.
    #[test]
    fn capture_flag_derives_from_the_black_board() {
        let (pos, mv) = move_shape("4k3/4P3/8/8/8/8/8/4K3 b - - 0 1", |m| m.is_tactical());
        let wire = u16::from(mv.from().0) | (u16::from(mv.to().0) << 6);
        assert_eq!(viri_to_soul_move(wire, &pos), Some(mv));
    }
}
