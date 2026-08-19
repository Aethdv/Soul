//! Viriformat binary parser and serializer.
//!
//! A viriformat stream consists of concatenated games:
//! - 32-byte header ([`Position`], counters, outcome)
//! - Zero or more 4-byte move/eval pairs: `(u16 move, i16 eval)`
//! - 4-byte null sentinel `[0, 0, 0, 0]`
//!
//! Evaluations in move pairs are White-relative centipawns and are converted to
//! side-to-move relative when exported into [`SoulEntry`].

use std::{
    fs,
    io::{self, Read},
};

use super::{SoulEntry, flip_result, flip_score};
use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_B_KS, ROOK_B_QS, ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
        defs::{Color, MATE_BOUND, PieceType, Square},
        moves::Move,
        util::pct,
    },
    engine::wdl::wdl_model,
    tools::dataset::quant,
};

const PACKED_BOARD_SIZE: usize = 32;
const SENTINEL: [u8; 4] = [0, 0, 0, 0];

/// Minimum centipawn advantage to classify a game as ending in a decisive victory.
pub const DECISIVE_ENDING: i32 = 1000;
/// Maximum centipawn deviation from equality to classify a game as ending in a quiet draw.
pub const QUIET_ENDING: i32 = 50;

/// White-relative WDL outcome encodings stored in viriformat headers.
const WDL_DRAW: u8 = 1;
const WDL_WIN: u8 = 2;

/// Position filtering and sample-weighting configuration for dataset replay.
///
/// Field names and defaults match `viriformat::dataformat::Filter`, so a filter file
/// written for that crate loads here. Its four WDL coefficients are absent: the gate
/// asks the local model instead of a foreign fit.
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
    /// Applied per epoch by the tuner sampler, not here: a position dropped at load is
    /// gone for the whole run.
    pub random_fen_skipping: bool,
    pub random_fen_skip_probability: f64,
    pub wdl_filtered: bool,
    pub material_count_filtered: bool,
    /// Skip probability per piece count, indexed `0..=32`. Missing entries default to 0.0.
    pub material_count_probabilities: Vec<f64>,
    /// Retained for schema parity with the reference filter.
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
    /// Pass-through filter with all gates disabled.
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

    /// Returns `true` if the position should be excluded from extraction.
    /// `eval` and `wdl` are White-relative as packed in the file.
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

    /// Computes the sample weight for surviving positions.
    ///
    /// Dropping a position with probability `p` and keeping it at weight `p` contribute
    /// the same amount on average, and only the first one adds noise.
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

    pub fn weights_positions(&self) -> bool { self.wdl_filtered || self.material_count_filtered }

    /// Returns `true` if the eval contradicts the outcome beyond `max_eval_incorrectness`.
    ///
    /// The winner scoring below zero is the contradiction, so the clamp stops a small
    /// positive score from counting as one.
    fn eval_is_incorrect(&self, eval: i32, wdl: u8) -> bool {
        if self.max_eval_incorrectness == u32::MAX {
            return false;
        }

        if wdl == WDL_DRAW {
            return eval.unsigned_abs() > self.max_eval_incorrectness;
        }

        let winner_eval = if wdl == WDL_WIN { eval } else { -eval };
        winner_eval.min(0).unsigned_abs() > self.max_eval_incorrectness
    }

    /// Returns the model-predicted probability of the actual game outcome.
    ///
    /// `wdl_model` is sign-symmetric; providing a White-relative score directly
    /// yields White-relative probabilities matching the header's WDL perspective.
    fn result_chance(eval: i32, material: u32, wdl: u8) -> f64 {
        let (win, draw, loss) = wdl_model(eval, material);
        match wdl {
            WDL_WIN => win,
            WDL_DRAW => draw,
            _ => loss,
        }
    }
}

/// Replays a viriformat file, filtering and extracting positions into `SoulEntry` records.
///
/// Returns:
/// 1. Extracted `SoulEntry` records.
/// 2. Sample weights per entry (empty if no weighting gates are active).
/// 3. Entry counts per game, so the train/val split can hold a game together.
pub fn parse_viri_file(path: &str, filter: &ReplayFilter) -> io::Result<(Vec<SoulEntry>, Vec<f32>, Vec<u32>)> {
    let mut file = fs::File::open(path)?;
    let mut data = Vec::new();
    file.read_to_end(&mut data)?;

    let mut entries = Vec::new();
    let mut weights = Vec::new();
    let mut entries_per_game = Vec::new();
    let mut offset = 0usize;
    let mut seen = 0usize;

    while offset + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[offset..offset + PACKED_BOARD_SIZE];
        offset += PACKED_BOARD_SIZE;

        let Some((mut position, game_result, mut ply)) = parse_packed_board(header) else {
            break;
        };

        // Incremental updates require initializing from the actual board state.
        let mut acc = position.get_initial_accumulator();
        let kept_before = entries.len();

        loop {
            if offset + 4 > data.len() {
                break;
            }

            let chunk = &data[offset..offset + 4];
            if chunk == SENTINEL {
                offset += 4;
                break;
            }

            let packed_move = u16::from_le_bytes([chunk[0], chunk[1]]);
            let score = i16::from_le_bytes([chunk[2], chunk[3]]);
            offset += 4;

            let Some(mv) = decode_move(packed_move, &position) else {
                break;
            };

            seen += 1;

            if !filter.should_filter(&position, mv, i32::from(score), game_result, ply) {
                entries.push(SoulEntry::from_board(
                    &position,
                    f64::from(flip_result(game_result, position.stm)) / 2.0,
                    Some(flip_score(i32::from(score), position.stm)),
                ));

                if filter.weights_positions() {
                    weights.push(filter.sample_weight(&position, i32::from(score), game_result) as f32);
                }
            }
            ply += 1;
            position.make_move(mv, &mut acc);
        }

        if entries.len() > kept_before {
            entries_per_game.push((entries.len() - kept_before) as u32);
        }
    }

    let share = pct(entries.len() as u64, seen as u64);
    println!("  Replayed {seen} positions, kept {} ({share:.1}%) from {} games", entries.len(), entries_per_game.len());
    Ok((entries, weights, entries_per_game))
}

/// Where a file's games ended, which its positions alone do not say. Games that all
/// stop in the middlegame leave no endgames to fit an eval against.
pub struct GameScan {
    pub games: usize,
    pub plies: u64,
    /// Total pieces remaining across all game terminal positions.
    pub pieces_left: u64,
    /// Outcome counts indexed by White-relative WDL code (`0 = Loss, 1 = Draw, 2 = Win`).
    pub results: [usize; 3],
    pub mate_endings: usize,
    pub decisive_endings: usize,
    pub quiet_endings: usize,
}

/// Scans every game in a viriformat file to gather summary statistics without filtering.
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

    let mut offset = 0usize;
    while offset + PACKED_BOARD_SIZE <= data.len() {
        let header = &data[offset..offset + PACKED_BOARD_SIZE];
        offset += PACKED_BOARD_SIZE;

        let Some((mut position, game_result, _)) = parse_packed_board(header) else {
            break;
        };

        let mut acc = position.get_initial_accumulator();
        let mut last_score = 0i32;
        let mut plies = 0u64;

        loop {
            if offset + 4 > data.len() {
                break;
            }

            let chunk = &data[offset..offset + 4];
            if chunk == SENTINEL {
                offset += 4;
                break;
            }

            let packed_move = u16::from_le_bytes([chunk[0], chunk[1]]);
            last_score = i32::from(i16::from_le_bytes([chunk[2], chunk[3]]));
            offset += 4;

            let Some(mv) = decode_move(packed_move, &position) else {
                break;
            };

            plies += 1;
            position.make_move(mv, &mut acc);
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

/// Unpacks a 32-byte viriformat header into a `Position`, outcome, and starting ply.
///
/// Ply counts from the first move of the game rather than the first move recorded, so
/// an opening cut eight plies deep enters its game already at eight. `min_ply` gates
/// on this number.
fn parse_packed_board(data: &[u8]) -> Option<(Position, u8, u32)> {
    if data.len() < PACKED_BOARD_SIZE {
        return None;
    }

    let occupancy = u64::from_le_bytes(data[0..8].try_into().ok()?);
    // Only 32 nibbles follow, so a fuller board than that is a malformed record. The
    // old shape truncated mid-board and handed back whatever had been placed so far.
    if occupancy.count_ones() > 32 {
        return None;
    }

    let pieces: [u8; 16] = data[8..24].try_into().ok()?;
    let stm_and_ep = data[24];
    let _halfmove = data[25];
    let fullmove = u16::from_le_bytes([data[26], data[27]]);
    let _score = i16::from_le_bytes([data[28], data[29]]);
    let result = data[30];
    let _extra = data[31];

    if result > 2 {
        return None;
    }

    let stm = if stm_and_ep & 0x80 != 0 { Color::Black } else { Color::White };
    let ep = stm_and_ep & 0x7F;
    let en_passant = if ep < 64 { Some(Square(ep)) } else { None };

    let mut pos = Position::new();
    pos.stm = stm;
    pos.en_passant = en_passant;

    // A rook cannot be classified until its king is placed, and it may come first, so
    // the markers wait here until every piece is down.
    let mut kings = [None; 2];
    let mut unmoved_rooks = [(Color::White, Square(0)); 4];
    let mut rook_count = 0usize;

    for (sq, nibble) in quant::packed_pieces(occupancy, &pieces) {
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

        pos.add_piece(sq, pt, color);

        if pt == PieceType::King {
            kings[color as usize] = Some(sq);
        }

        // Viriformat encodes unmoved castling rooks as piece type 6.
        if viri_type == 6 {
            unmoved_rooks[rook_count] = (color, sq);
            rook_count += 1;
        }
    }

    let mut castling_rights = 0u8;

    for &(color, rook) in &unmoved_rooks[..rook_count] {
        let Some(king) = kings[color as usize] else {
            continue;
        };

        let (mask, slot) = Position::castling_side(rook, king, color);
        castling_rights |= mask;
        pos.castling_rooks[slot] = rook;
    }

    pos.castling_rights = castling_rights;

    // Detect Chess960 (FRC) by checking whether live castling rooks deviate from standard squares.
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

/// Appends a game (header, move/score pairs, and sentinel) to `out`.
///
/// `result` and scores in `moves` are White-relative.
pub fn write_game(out: &mut Vec<u8>, opening: &Position, result: u8, score: i16, moves: &[(Move, i16)]) {
    debug_assert!(result <= WDL_WIN, "result {result} is outside 0..=2");
    out.extend_from_slice(&pack_board(opening, score, result));
    for &(mv, mv_score) in moves {
        out.extend_from_slice(&encode_move(mv).to_le_bytes());
        out.extend_from_slice(&mv_score.to_le_bytes());
    }
    out.extend_from_slice(&SENTINEL);
}

/// Serializes a `Position` into a 32-byte header. Inverse of [`parse_packed_board`].
fn pack_board(pos: &Position, score: i16, result: u8) -> [u8; PACKED_BOARD_SIZE] {
    let (occupancy, pieces) = quant::pack_pieces(pos);
    let mut out = [0u8; PACKED_BOARD_SIZE];

    out[0..8].copy_from_slice(&occupancy.to_le_bytes());
    out[8..24].copy_from_slice(&pieces);
    // Square values >= 64 indicate no en passant square.
    out[24] = (u8::from(pos.stm == Color::Black) << 7) | pos.en_passant.map_or(64, |sq| sq.0);
    out[25] = pos.halfmove_clock;
    out[26..28].copy_from_slice(&pos.fullmove_number.to_le_bytes());
    out[28..30].copy_from_slice(&score.to_le_bytes());
    out[30] = result;
    out
}

/// Encodes a move into a 16-bit viriformat integer. Inverse of [`decode_move`].
///
/// Capture state is omitted; readers derive it from board occupancy.
fn encode_move(mv: Move) -> u16 {
    let (move_type, promo) = if mv.is_castling() {
        (2, 0)
    } else if mv.is_en_passant() {
        (1, 0)
    } else if let Some(pt) = mv.promo() {
        (3, pt as u16 - PieceType::Knight as u16)
    } else {
        (0, 0)
    };

    u16::from(mv.from().0) | (u16::from(mv.to().0) << 6) | (promo << 12) | (move_type << 14)
}

/// Decodes a 16-bit viriformat move bitfield in the context of `pos`.
///
/// Bitfield layout:
/// - `0..=5`  : `from` square (6 bits)
/// - `6..=11` : `to` square (6 bits)
/// - `12..=13`: Promotion piece (`0=N, 1=B, 2=R, 3=Q`)
/// - `14..=15`: Move type (`0=Normal, 1=En Passant, 2=Castling, 3=Promotion`)
fn decode_move(viri_move: u16, pos: &Position) -> Option<Move> {
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
    use super::{
        Move, PACKED_BOARD_SIZE, Position, ReplayFilter, SENTINEL, WDL_DRAW, WDL_WIN, decode_move, pack_board, parse_packed_board,
        parse_viri_file, write_game,
    };
    use crate::{
        core::{
            board::{ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
            defs::{Color, PieceType, Square},
        },
        engine::movegen::gen_legal_moves,
        tools::dataset::{SoulEntry, flip_result, flip_score, quant},
    };

    const WDL_LOSS: u8 = 0;

    const OPEN: &str = "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4";
    const IN_CHECK_BLACK: &str = "r1bqkbnr/pppp1Bpp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 0 4";
    const EN_PASSANT: &str = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3";
    const PROMOTES: &str = "1n5k/P7/8/8/8/8/8/K7 w - - 0 1";
    const PROMO_AND_EP: &str = "8/1P6/8/3pP3/8/8/8/k1K5 w - d6 0 1";

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

    fn quiet_move(fen: &str) -> Move { pick(fen, |m| !m.is_tactical() && !m.is_castling()).1 }

    #[test]
    fn unrestricted_keeps_everything() {
        let filter = ReplayFilter::UNRESTRICTED;
        let capture = pick(OPEN, Move::is_tactical).1;
        assert!(keeps(&filter, OPEN, quiet_move(OPEN), 0, WDL_DRAW, 0));
        assert!(keeps(&filter, OPEN, capture, 30_000, WDL_WIN, 0));
        assert!(keeps(&filter, IN_CHECK_BLACK, quiet_move(IN_CHECK_BLACK), -30_000, WDL_LOSS, 0));
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
        assert!(!keeps(&filter, OPEN, quiet_move(OPEN), 450, WDL_DRAW, 0));
        assert!(!keeps(&filter, OPEN, quiet_move(OPEN), -450, WDL_DRAW, 0));
    }

    #[test]
    fn tactical_check_and_castling_gate_independently() {
        let quiet = quiet_move(OPEN);
        let capture = pick(OPEN, Move::is_tactical).1;
        let castle = pick(OPEN, Move::is_castling).1;

        let tactical = ReplayFilter { filter_tactical: true, ..ReplayFilter::UNRESTRICTED };
        assert!(!keeps(&tactical, OPEN, capture, 0, WDL_DRAW, 0));
        assert!(keeps(&tactical, IN_CHECK_BLACK, quiet_move(IN_CHECK_BLACK), 0, WDL_DRAW, 0));

        let check = ReplayFilter { filter_check: true, ..ReplayFilter::UNRESTRICTED };
        assert!(!keeps(&check, IN_CHECK_BLACK, quiet_move(IN_CHECK_BLACK), 0, WDL_DRAW, 0));
        assert!(keeps(&check, OPEN, capture, 0, WDL_DRAW, 0));

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
        assert!(keeps(&filter, OPEN, quiet, 30_000, WDL_WIN, 0));
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
        assert!(keeps(&certain, OPEN, quiet, 0, WDL_DRAW, 0));
    }

    #[test]
    fn the_reshaping_gates_weight_instead_of_dropping() {
        let quiet = quiet_move(OPEN);
        let table = ReplayFilter {
            material_count_filtered: true,
            material_count_probabilities: vec![0.75; 33],
            ..ReplayFilter::UNRESTRICTED
        };

        assert!(keeps(&table, OPEN, quiet, 0, WDL_DRAW, 0));
        assert!((table.sample_weight(&Position::from_fen(OPEN), 0, WDL_DRAW) - 0.25).abs() < 1e-12);
        assert!((ReplayFilter::UNRESTRICTED.sample_weight(&Position::from_fen(OPEN), 0, WDL_DRAW) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_wdl_gate_weighs_what_the_eval_predicted() {
        let filter = ReplayFilter { wdl_filtered: true, ..ReplayFilter::UNRESTRICTED };
        let pos = Position::from_fen(OPEN);
        let won = filter.sample_weight(&pos, 2000, WDL_WIN);
        let lost = filter.sample_weight(&pos, 2000, WDL_LOSS);
        assert!(won > 0.9);
        assert!(lost < 0.1);
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

    fn assert_wire_roundtrip(fen: &str, result: f64, score: i32, halfmove: u8, fullmove: u16) {
        let pos = Position::from_fen(fen);
        let entry = quant::from_board(&pos, result, Some(score));
        let (decoded, game_result, ply) =
            parse_packed_board(&wire_from_entry(&entry, halfmove, fullmove, 0)).expect("valid header wire format");
        assert_position_equal(&pos, &decoded);
        assert_eq!(game_result, (result * 2.0) as u8);
        let expected_ply = (u32::from(fullmove) - 1) * 2 + u32::from(decoded.stm == Color::Black);
        assert_eq!(ply, expected_ply);
    }

    #[test]
    fn packed_board_round_trips_through_the_decoder() {
        assert_wire_roundtrip("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 1.0, 120, 0, 32);
        assert_wire_roundtrip("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", 0.5, -40, 7, 9);
        assert_wire_roundtrip("4k3/8/8/8/8/8/P6p/4K3 w - - 0 1", 1.0, 10_000, 3, 44);
    }

    fn frc_rooks_off_home() -> Position {
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
        pos
    }

    #[test]
    fn unmoved_rooks_rebuild_frc_rights_slots() {
        let pos = frc_rooks_off_home();
        let entry = quant::from_board(&pos, 1.0, Some(0));
        let (decoded, ..) = parse_packed_board(&wire_from_entry(&entry, 0, 3, 0)).expect("well-formed wire");
        assert_position_equal(&pos, &decoded);
        assert_eq!(decoded.castling_rights, WHITE_OO | WHITE_OOO);
        assert_eq!(decoded.castling_rooks[ROOK_W_QS], Square(1));
        assert_eq!(decoded.castling_rooks[ROOK_W_KS], Square(5));
    }

    fn assert_move_roundtrip(fen: &str, want: impl Fn(Move) -> bool) {
        let (pos, mv) = pick(fen, want);
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
        assert_eq!(decode_move(wire, &pos), Some(mv), "round trip of {mv:?}");
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

    #[test]
    fn capture_flag_derives_from_the_black_board() {
        let (pos, mv) = pick("4k3/4P3/8/8/8/8/8/4K3 b - - 0 1", |m| m.is_tactical());
        let wire = u16::from(mv.from().0) | (u16::from(mv.to().0) << 6);
        assert_eq!(decode_move(wire, &pos), Some(mv));
    }

    fn find(pos: &Position, uci: &str) -> Move {
        *gen_legal_moves(pos)
            .iter()
            .find(|mv| mv.to_uci(pos.is_frc) == uci)
            .unwrap_or_else(|| panic!("no {uci} in {}", pos.as_fen()))
    }

    fn assert_round_trips(tag: &str, start_fen: &str, ucis: &[&str], result: u8) {
        let opening = Position::from_fen(start_fen);
        let mut replay = opening;
        let mut acc = replay.get_initial_accumulator();
        let mut moves = Vec::new();
        let mut expected = Vec::new();

        for (i, uci) in ucis.iter().enumerate() {
            let mv = find(&replay, uci);
            let score = (i as i16 + 1) * 37;
            expected.push((quant::from_board(&replay, 0.5, None), score, replay.stm));
            moves.push((mv, score));
            replay.make_move(mv, &mut acc);
        }

        let opening_eval = moves.first().map_or(0, |&(_, score)| score);
        let mut bytes = Vec::new();
        write_game(&mut bytes, &opening, result, opening_eval, &moves);

        let (entries, ..) = read_back(tag, &bytes);
        assert_eq!(entries.len(), expected.len());

        for (entry, (want, score, stm)) in entries.iter().zip(&expected) {
            let fen = want.to_fen();
            assert_eq!(entry.occupancy, want.occupancy, "occupancy at {fen}");
            assert_eq!(entry.pieces, want.pieces, "pieces and castling-rook markers at {fen}");
            assert_eq!(entry.stm_and_ep, want.stm_and_ep, "side and en passant at {fen}");
            assert_eq!(i32::from(entry.score), flip_score(i32::from(*score), *stm), "score perspective at {fen}");
            assert_eq!(entry.result, flip_result(result, *stm), "result perspective at {fen}");
        }
    }

    fn read_back(tag: &str, bytes: &[u8]) -> (Vec<SoulEntry>, Vec<f32>, Vec<u32>) {
        let path = std::env::temp_dir().join(format!("soul_viri_{tag}_{}.vf", std::process::id()));
        std::fs::write(&path, bytes).expect("writing the game");
        let read = parse_viri_file(path.to_str().expect("a utf-8 temp path"), &ReplayFilter::UNRESTRICTED);
        std::fs::remove_file(&path).ok();
        read.expect("reading the game back")
    }

    #[test]
    fn a_game_with_castling_and_captures_round_trips() {
        assert_round_trips("castling", OPEN, &["e1g1", "g8f6", "c4f7", "e8f7"], WDL_WIN);
    }

    #[test]
    fn a_game_with_a_promotion_and_an_en_passant_round_trips() {
        assert_round_trips("promo_ep", PROMO_AND_EP, &["e5d6", "a1a2", "b7b8q"], WDL_WIN);
    }

    #[test]
    fn a_game_that_opens_in_check_round_trips() { assert_round_trips("in_check", IN_CHECK_BLACK, &["e8f7"], WDL_LOSS); }

    #[test]
    fn the_written_header_matches_the_wire_the_decoder_reads() {
        let mut boards: Vec<Position> = [OPEN, IN_CHECK_BLACK, EN_PASSANT, PROMOTES, "4k3/8/8/8/8/8/P6p/4K3 w - - 0 1"]
            .map(Position::from_fen)
            .into();
        boards.push(frc_rooks_off_home());

        for pos in &boards {
            let entry = quant::from_board(pos, 1.0, Some(-40));
            let wire = wire_from_entry(&entry, pos.halfmove_clock, pos.fullmove_number, 0);
            assert_eq!(pack_board(pos, -40, WDL_WIN), wire, "header bytes for {}", pos.as_fen());
        }
    }

    #[test]
    fn a_game_with_no_moves_writes_only_its_header_and_sentinel() {
        let mut bytes = Vec::new();
        write_game(&mut bytes, &Position::from_fen(OPEN), WDL_DRAW, 0, &[]);
        assert_eq!(bytes.len(), PACKED_BOARD_SIZE + SENTINEL.len());
    }

    #[test]
    fn two_games_concatenate() {
        let opening = Position::from_fen(OPEN);
        let mv = find(&opening, "e1g1");
        let mut bytes = Vec::new();
        write_game(&mut bytes, &opening, WDL_WIN, 0, &[(mv, 10)]);
        write_game(&mut bytes, &opening, WDL_WIN, 0, &[(mv, 20)]);
        let (entries, _, entry_counts) = read_back("pair", &bytes);
        assert_eq!(entry_counts, vec![1, 1]);
        assert_eq!(entries[0].to_fen(), entries[1].to_fen());
        assert_eq!(i32::from(entries[1].score) - i32::from(entries[0].score), 10);
    }
}
