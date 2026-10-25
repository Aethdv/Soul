//! Viriformat: concatenated games, each a 32-byte board header, 4-byte move/eval pairs,
//! and a zero sentinel. Move evals are White-relative centipawns; replay converts them
//! into side-to-move labels in [`SoulEntry`].

use std::{fs, io};

use super::{SoulEntry, flip_result, flip_score};
use crate::{
    core::{
        board::Position,
        defs::{Color, MATE_BOUND, PieceType, Square},
        moves::Move,
    },
    engine::wdl::wdl_model,
    tools::dataset::packing,
};

const HEADER_SIZE: usize = 32;
const SENTINEL: [u8; 4] = [0, 0, 0, 0];

pub const MIN_DECISIVE_ENDING: i32 = 1000;
pub const MAX_QUIET_ENDING: i32 = 50;

const WDL_DRAW: u8 = 1;
const WDL_WIN: u8 = 2;

#[derive(Default)]
pub struct GameScan {
    pub games: usize,
    pub plies: u64,
    pub pieces_left_sum: u64,
    pub draws: usize,
    pub mate_endings: usize,
    pub decisive_endings: usize,
    pub quiet_endings: usize,
}

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
    pub material_count_probabilities: Vec<f64>,
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
    };

    fn should_filter(&self, pos: &Position, mv: Move, white_eval: i32, white_result: u8, ply: u32) -> bool {
        if ply < self.min_ply {
            return true;
        }

        if white_eval.unsigned_abs() >= self.max_eval {
            return true;
        }

        if pos.occ.popcount() < self.min_pieces {
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

        self.eval_is_incorrect(white_eval, white_result)
    }

    // Weight by p instead of sampling by p: same expectation, no noise.
    fn sample_weight(&self, pos: &Position, white_eval: i32, white_result: u8) -> f64 {
        let mut weight = 1.0;
        if self.wdl_filtered {
            weight *= Self::result_chance(white_eval, pos.material_count(), white_result);
        }

        if self.material_count_filtered {
            let index = pos.occ.popcount().min(32) as usize;
            weight *= 1.0 - self.material_count_probabilities.get(index).copied().unwrap_or(0.0);
        }
        weight
    }

    fn is_weighted(&self) -> bool { self.wdl_filtered || self.material_count_filtered }

    fn eval_is_incorrect(&self, white_eval: i32, white_result: u8) -> bool {
        if self.max_eval_incorrectness == u32::MAX {
            return false;
        }

        if white_result == WDL_DRAW {
            return white_eval.unsigned_abs() > self.max_eval_incorrectness;
        }

        let winner_eval = if white_result == WDL_WIN { white_eval } else { -white_eval };
        winner_eval.min(0).unsigned_abs() > self.max_eval_incorrectness
    }

    fn result_chance(white_eval: i32, material: u32, white_result: u8) -> f64 {
        let (win, draw, loss) = wdl_model(white_eval, material);
        match white_result {
            WDL_WIN => win,
            WDL_DRAW => draw,
            _ => loss,
        }
    }
}

/// Replays a file into training entries, per-entry weights, and per-game kept counts.
///
/// `weights` is empty when the filter does not weight. A truncated or malformed stream is
/// returned as `InvalidData`.
pub fn parse_viri_file(path: &str, filter: &ReplayFilter) -> io::Result<(Vec<SoulEntry>, Vec<f32>, Vec<u32>)> {
    let data = fs::read(path)?;

    let mut entries = Vec::new();
    let mut weights = Vec::new();
    let mut kept_per_game = Vec::new();
    let mut offset = 0usize;

    while offset < data.len() {
        let (mut position, white_result, mut ply, _) = read_header(&data, &mut offset)?;
        let mut acc = position.psqt_accumulator();
        let kept_before = entries.len();

        while let Some((packed_move, white_score)) = read_move_pair(&data, &mut offset)? {
            let mv = decode_move(packed_move, &position).ok_or_else(|| corrupt("invalid move"))?;

            if !filter.should_filter(&position, mv, i32::from(white_score), white_result, ply) {
                entries.push(SoulEntry::from_board(
                    &position,
                    f64::from(flip_result(white_result, position.stm)) / 2.0,
                    Some(flip_score(i32::from(white_score), position.stm)),
                ));

                if filter.is_weighted() {
                    weights.push(filter.sample_weight(&position, i32::from(white_score), white_result) as f32);
                }
            }

            ply += 1;
            position.make_move(mv, &mut acc);
        }

        if entries.len() > kept_before {
            kept_per_game.push((entries.len() - kept_before) as u32);
        }
    }

    Ok((entries, weights, kept_per_game))
}

/// Scans every game without filtering. Fails on truncated or invalid data.
pub fn scan_viri_games(path: &str) -> io::Result<GameScan> {
    let data = fs::read(path)?;

    let mut scan = GameScan::default();
    let mut offset = 0usize;

    while offset < data.len() {
        let (mut position, game_result, _, header_score) = read_header(&data, &mut offset)?;
        let mut acc = position.psqt_accumulator();
        let mut last_score = i32::from(header_score);
        let mut plies = 0u64;

        while let Some((packed_move, score)) = read_move_pair(&data, &mut offset)? {
            last_score = i32::from(score);
            let mv = decode_move(packed_move, &position).ok_or_else(|| corrupt("invalid move"))?;
            plies += 1;
            position.make_move(mv, &mut acc);
        }

        scan.games += 1;
        scan.plies += plies;
        scan.pieces_left_sum += u64::from(position.occ.popcount());
        if game_result == WDL_DRAW {
            scan.draws += 1;
        }

        let magnitude = last_score.abs();
        if magnitude >= MATE_BOUND {
            scan.mate_endings += 1;
        } else if magnitude >= MIN_DECISIVE_ENDING {
            scan.decisive_endings += 1;
        } else if magnitude <= MAX_QUIET_ENDING {
            scan.quiet_endings += 1;
        }
    }

    Ok(scan)
}

fn corrupt(message: &str) -> io::Error { io::Error::new(io::ErrorKind::InvalidData, message) }

fn read_header(data: &[u8], offset: &mut usize) -> io::Result<(Position, u8, u32, i16)> {
    let end = offset
        .checked_add(HEADER_SIZE)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| corrupt("truncated game header"))?;
    let header = data.get(*offset..end).ok_or_else(|| corrupt("truncated game header"))?;
    *offset = end;

    let (position, result, ply) = parse_header(header).ok_or_else(|| corrupt("invalid game header"))?;
    let score = i16::from_le_bytes([header[28], header[29]]);
    Ok((position, result, ply, score))
}

fn read_move_pair(data: &[u8], offset: &mut usize) -> io::Result<Option<(u16, i16)>> {
    let end = offset
        .checked_add(4)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| corrupt("truncated move pair"))?;
    let chunk = data.get(*offset..end).ok_or_else(|| corrupt("truncated move pair"))?;
    *offset = end;

    if chunk == SENTINEL {
        return Ok(None);
    }

    let mv = u16::from_le_bytes([chunk[0], chunk[1]]);
    let score = i16::from_le_bytes([chunk[2], chunk[3]]);
    Ok(Some((mv, score)))
}

// Starting ply counts from move one of the game, so a cut opening enters at its real ply.
fn parse_header(data: &[u8]) -> Option<(Position, u8, u32)> {
    if data.len() < HEADER_SIZE {
        return None;
    }

    let occupancy = u64::from_le_bytes(data[0..8].try_into().ok()?);
    let pieces: [u8; 16] = data[8..24].try_into().ok()?;
    let result = data[30];
    if result > 2 {
        return None;
    }

    let fullmove = u16::from_le_bytes([data[26], data[27]]);
    let position = packing::unpack_board(occupancy, &pieces, data[24], data[25], fullmove, None)?;
    let ply = u32::from(fullmove).saturating_sub(1) * 2 + u32::from(position.stm == Color::Black);
    Some((position, result, ply))
}

// `result` and scores in `moves` are White-relative.
#[cfg(any(test, feature = "datagen"))]
pub(crate) fn write_game(out: &mut Vec<u8>, opening: &Position, result: u8, score: i16, moves: &[(Move, i16)]) {
    debug_assert!(result <= WDL_WIN, "result {result} is outside 0..=2");
    out.extend_from_slice(&pack_header(opening, score, result));
    for &(mv, mv_score) in moves {
        out.extend_from_slice(&encode_move(mv).to_le_bytes());
        out.extend_from_slice(&mv_score.to_le_bytes());
    }
    out.extend_from_slice(&SENTINEL);
}

#[cfg(any(test, feature = "datagen"))]
fn pack_header(pos: &Position, score: i16, result: u8) -> [u8; HEADER_SIZE] {
    let (occupancy, pieces) = packing::pack_pieces(pos);
    let mut out = [0u8; HEADER_SIZE];
    out[0..8].copy_from_slice(&occupancy.to_le_bytes());
    out[8..24].copy_from_slice(&pieces);
    out[24] = (u8::from(pos.stm == Color::Black) << 7) | pos.en_passant.map_or(64, |sq| sq.0);
    out[25] = pos.halfmove_clock;
    out[26..28].copy_from_slice(&pos.fullmove_number.to_le_bytes());
    out[28..30].copy_from_slice(&score.to_le_bytes());
    out[30] = result;
    out
}

#[cfg(any(test, feature = "datagen"))]
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

// Bitfield: from 0..6, to 6..12, promo 12..14 (N/B/R/Q), move type 14..16.
fn decode_move(viri_move: u16, pos: &Position) -> Option<Move> {
    let from = Square((viri_move & 0x3F) as u8);
    let to = Square(((viri_move >> 6) & 0x3F) as u8);
    let promo_piece = (viri_move >> 12) & 0x3;
    let move_type = (viri_move >> 14) & 0x3;

    let moving_piece = pos.piece_at(from);
    if moving_piece == PieceType::None || from == to || pos.color_at(from) != pos.stm {
        return None;
    }

    let target = pos.piece_at(to);
    if target == PieceType::King {
        return None;
    }
    if move_type != 2 && target != PieceType::None && pos.color_at(to) == pos.stm {
        return None;
    }

    if move_type == 1 && (moving_piece != PieceType::Pawn || pos.en_passant != Some(to) || target != PieceType::None) {
        return None;
    }
    if move_type == 2 && (moving_piece != PieceType::King || target != PieceType::Rook || pos.color_at(to) != pos.stm) {
        return None;
    }
    let promo_rank = if pos.stm == Color::White { 7 } else { 0 };
    if moving_piece == PieceType::Pawn && to.rank() == promo_rank && move_type != 3 {
        return None;
    }
    if move_type == 3 && (moving_piece != PieceType::Pawn || to.rank() != promo_rank) {
        return None;
    }

    let capture = target != PieceType::None;

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
        _ => {
            let capture_bit = u16::from(capture);
            match promo_piece {
                0 => Move::PROM_N | capture_bit,
                1 => Move::PROM_B | capture_bit,
                2 => Move::PROM_R | capture_bit,
                _ => Move::PROM_Q | capture_bit,
            }
        },
    };

    Some(Move::new(from, to, flag))
}

#[cfg(test)]
mod tests {
    use super::{
        HEADER_SIZE, Move, Position, ReplayFilter, SENTINEL, WDL_DRAW, WDL_WIN, decode_move, pack_header, parse_header,
        parse_viri_file, scan_viri_games, write_game,
    };
    use crate::{
        core::{
            board::{ROOK_W_KS, ROOK_W_QS, WHITE_OO, WHITE_OOO},
            defs::{Color, PieceType, Square},
        },
        engine::movegen::gen_legal_moves,
        tools::dataset::{SoulEntry, flip_result, flip_score, packing},
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
        let on_board = Position::from_fen(OPEN).occ.popcount();
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
    fn eval_incorrectness_uses_winner_perspective() {
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
    fn random_skipping_does_not_filter_at_load() {
        let quiet = quiet_move(OPEN);
        let certain = ReplayFilter { random_fen_skipping: true, random_fen_skip_probability: 1.0, ..ReplayFilter::UNRESTRICTED };
        assert!(keeps(&certain, OPEN, quiet, 0, WDL_DRAW, 0));
    }

    #[test]
    fn material_count_gate_weights_instead_of_dropping() {
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
    fn wdl_gate_weights_by_predicted_result() {
        let filter = ReplayFilter { wdl_filtered: true, ..ReplayFilter::UNRESTRICTED };
        let pos = Position::from_fen(OPEN);
        let won = filter.sample_weight(&pos, 2000, WDL_WIN);
        let lost = filter.sample_weight(&pos, 2000, WDL_LOSS);
        assert!(won > 0.9);
        assert!(lost < 0.1);
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
        assert_eq!(a.frc_notation, b.frc_notation);
        assert_eq!(a.occ, b.occ);
        assert_eq!(a.side_bb, b.side_bb);
        for sq_idx in 0..64 {
            let sq = Square(sq_idx as u8);
            assert_eq!(a.piece_at(sq), b.piece_at(sq), "piece on {sq:?}");
        }
    }

    fn assert_wire_roundtrip(fen: &str, result: f64, score: i32, halfmove: u8, fullmove: u16) {
        let pos = Position::from_fen(fen);
        let entry = packing::from_board(&pos, result, Some(score));
        let (decoded, game_result, ply) =
            parse_header(&wire_from_entry(&entry, halfmove, fullmove, 0)).expect("valid header wire format");
        assert_position_equal(&pos, &decoded);
        assert_eq!(game_result, (result * 2.0) as u8);
        let expected_ply = (u32::from(fullmove) - 1) * 2 + u32::from(decoded.stm == Color::Black);
        assert_eq!(ply, expected_ply);
    }

    #[test]
    fn header_decodes_to_original_position() {
        assert_wire_roundtrip("rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3", 1.0, 120, 0, 32);
        assert_wire_roundtrip("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq - 0 1", 0.5, -40, 7, 9);
        assert_wire_roundtrip("4k3/8/8/8/8/8/P6p/4K3 w - - 0 1", 1.0, 10_000, 3, 44);
    }

    fn frc_rooks_off_home() -> Position {
        let mut pos = Position::empty();
        pos.add_piece(Square(1), PieceType::Rook, Color::White);
        pos.add_piece(Square(3), PieceType::King, Color::White);
        pos.add_piece(Square(5), PieceType::Rook, Color::White);
        pos.add_piece(Square(60), PieceType::King, Color::Black);
        pos.stm = Color::White;
        pos.castling_rights = WHITE_OO | WHITE_OOO;
        pos.castling_rooks[ROOK_W_QS] = Square(1);
        pos.castling_rooks[ROOK_W_KS] = Square(5);
        pos.frc_notation = true;
        pos
    }

    #[test]
    fn unmoved_rooks_rebuild_frc_rights_slots() {
        let pos = frc_rooks_off_home();
        let entry = packing::from_board(&pos, 1.0, Some(0));
        let (decoded, ..) = parse_header(&wire_from_entry(&entry, 0, 3, 0)).expect("well-formed wire");
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
    fn capture_flag_derives_from_occupancy() {
        let (pos, mv) = pick("4k3/4P3/8/8/8/8/8/4K3 b - - 0 1", |m| m.is_tactical());
        let wire = u16::from(mv.from().0) | (u16::from(mv.to().0) << 6);
        assert_eq!(decode_move(wire, &pos), Some(mv));
    }

    fn find(pos: &Position, uci: &str) -> Move {
        *gen_legal_moves(pos)
            .iter()
            .find(|mv| mv.to_uci(pos.frc_notation) == uci)
            .unwrap_or_else(|| panic!("no {uci} in {}", pos.to_fen()))
    }

    fn assert_round_trips(tag: &str, start_fen: &str, ucis: &[&str], result: u8) {
        let opening = Position::from_fen(start_fen);
        let mut replay = opening;
        let mut acc = replay.psqt_accumulator();
        let mut moves = Vec::new();
        let mut expected = Vec::new();

        for (i, uci) in ucis.iter().enumerate() {
            let mv = find(&replay, uci);
            let score = (i as i16 + 1) * 37;
            expected.push((packing::from_board(&replay, 0.5, None), score, replay.stm));
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

    #[test]
    fn malformed_headers_are_rejected() {
        let mut five_rooks = [0u8; HEADER_SIZE];
        five_rooks[0] = 0b11111;
        five_rooks[8..11].copy_from_slice(&[0x66, 0x66, 0x06]);
        assert!(parse_header(&five_rooks).is_none());

        let mut bad_piece = [0u8; HEADER_SIZE];
        bad_piece[0] = 1;
        bad_piece[8] = 7;
        assert!(parse_header(&bad_piece).is_none());

        let mut no_king = [0u8; HEADER_SIZE];
        no_king[0] = 1;
        assert!(parse_header(&no_king).is_none());

        let mut bad_ep = pack_header(&Position::from_fen(OPEN), 0, WDL_DRAW);
        bad_ep[24] = Square::from_coords(4, 3).0;
        assert!(parse_header(&bad_ep).is_none());
    }

    #[test]
    fn truncated_files_are_rejected() {
        let path = std::env::temp_dir().join(format!("soul_viri_truncated_{}.vf", std::process::id()));
        let path = path.to_str().unwrap();
        let rejects = |path: &str| {
            assert!(scan_viri_games(path).is_err());
            assert!(parse_viri_file(path, &ReplayFilter::UNRESTRICTED).is_err());
        };

        let mut truncated = pack_header(&Position::from_fen(OPEN), 0, WDL_DRAW).to_vec();
        truncated.extend_from_slice(&[0, 0]);
        std::fs::write(path, truncated).unwrap();
        rejects(path);

        std::fs::write(path, vec![0u8; HEADER_SIZE - 1]).unwrap();
        rejects(path);

        std::fs::write(path, pack_header(&Position::from_fen(OPEN), 0, WDL_LOSS)).unwrap();
        rejects(path);

        std::fs::remove_file(path).ok();
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
    fn pack_header_matches_hand_built_header() {
        let mut boards: Vec<Position> = [OPEN, IN_CHECK_BLACK, EN_PASSANT, PROMOTES, "4k3/8/8/8/8/8/P6p/4K3 w - - 0 1"]
            .map(Position::from_fen)
            .into();
        boards.push(frc_rooks_off_home());

        for pos in &boards {
            let entry = packing::from_board(pos, 1.0, Some(-40));
            let wire = wire_from_entry(&entry, pos.halfmove_clock, pos.fullmove_number, 0);
            assert_eq!(pack_header(pos, -40, WDL_WIN), wire, "header bytes for {}", pos.to_fen());
        }
    }

    #[test]
    fn a_game_with_no_moves_writes_only_its_header_and_sentinel() {
        let mut bytes = Vec::new();
        write_game(&mut bytes, &Position::from_fen(OPEN), WDL_DRAW, 0, &[]);
        assert_eq!(bytes.len(), HEADER_SIZE + SENTINEL.len());
        let (entries, _, kept_per_game) = read_back("empty", &bytes);
        assert!(entries.is_empty());
        assert!(kept_per_game.is_empty());
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
