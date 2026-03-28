//! Position quantization and dataset encoding.
//!
//! Provides the serialization logic to pack chess board states into the compact
//! `SoulEntry` binary layout.

use crate::{
    core::{
        board::Position,
        defs::{Color, PieceType},
    },
    engine::mobility::{Mobility, OPEN_UNITY},
    tools::dataset::{PackedPiece, PackedSafety, STM_WHITE, SoulEntry},
};

/// FEN characters indexed by `[color_index][piece_type]`.
/// In normalised entries: index 0 = White = "Us" (STM), index 1 = Black = "Them".
const PIECE_CHARS: [[u8; 6]; 2] = [*b"PNBRQK", *b"pnbrqk"];

/// Encode a board position into a [`SoulEntry`],
/// normalised to the side-to-move's perspective.
/// When STM is Black, all squares are rank-flipped so that "our"
/// pieces always advance up the board.
pub fn from_board(
    board: &Position,
    result: f64,
    static_score: Option<i32>,
    search_score: Option<i32>,
) -> SoulEntry {
    let stm_is_white = board.stm == Color::White;

    // The only thing that changes between perspectives:
    // which real color maps to "Us" vs "Them",
    // and whether squares need a rank-flip.
    let (us, them) = if stm_is_white {
        (Color::White, Color::Black)
    } else {
        (Color::Black, Color::White)
    };

    let mut entry = SoulEntry {
        result: result as f32,
        static_score: static_score.unwrap_or(0) as i16,
        search_score: search_score.unwrap_or(0) as i16,
        original_stm: u8::from(!stm_is_white),
        ..Default::default()
    };

    // Piece list (perspective-normalised)
    //
    // For each piece type: encode Us pieces (as White), then Them (as Black).
    // The compiler unrolls the 2-element inner iterator — zero overhead.
    let mut idx = 0usize;

    for piece in PieceType::ALL {
        let pt = piece.as_usize();

        for &(board_color, packed_color) in &[(us, Color::White), (them, Color::Black)] {
            let mut bb = board.pieces(piece, board_color);
            while bb.is_not_empty() {
                let mut sq = bb.pop_lsb();
                if !stm_is_white {
                    sq = sq.flip_rank();
                }
                if idx < 32 {
                    entry.pieces[idx] = PackedPiece::new(pt, packed_color, sq);
                    idx += 1;
                }
            }
        }
    }

    entry.piece_count = idx as u8;

    // ── Mobility & king safety ──
    let pinned_w = board.pinned_pieces(crate::core::defs::Color::White);
    let pinned_b = board.pinned_pieces(crate::core::defs::Color::Black);
    let tensor = crate::core::board::spatial::SpatialTensor::compute(board, pinned_w.0, pinned_b.0);
    let mob = Mobility::compute_all(board, board.stm, &tensor, pinned_w, pinned_b);
    entry.safety_us = PackedSafety::from(mob.safety_us);
    entry.safety_them = PackedSafety::from(mob.safety_them);

    // ── X-Ray feature (ortho) ──
    let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
    let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
    let w_ring = crate::core::board::bitboard::atk_king(w_ksq).0;
    let b_ring = crate::core::board::bitboard::atk_king(b_ksq).0;

    let xray_val = (tensor.w_ortho_xray() & b_ring).count_ones() as i32
        - (tensor.b_ortho_xray() & w_ring).count_ones() as i32;

    // Normalize to STM
    entry.xray_ortho = (if stm_is_white { xray_val } else { -xray_val }) as i8;

    entry.mobility[0] = mob.metrics_us.mobility.clamp(-127, 127) as i8;
    entry.mobility[1] = mob.metrics_us.shadow_mobility.clamp(-127, 127) as i8;
    entry.mobility[2] = mob.metrics_us.threats.clamp(-127, 127) as i8;
    entry.mobility[3] = mob.metrics_us.shadow_threats.clamp(-127, 127) as i8;

    entry.mobility[4] = mob.metrics_them.mobility.clamp(-127, 127) as i8;
    entry.mobility[5] = mob.metrics_them.shadow_mobility.clamp(-127, 127) as i8;
    entry.mobility[6] = mob.metrics_them.threats.clamp(-127, 127) as i8;
    entry.mobility[7] = mob.metrics_them.shadow_threats.clamp(-127, 127) as i8;

    // ── Stateful context ──
    let mut castling = board.castling_rights;
    if !stm_is_white {
        castling = (castling >> 2) | ((castling & 0x3) << 2);
    }
    entry.castling = castling;

    entry.ep_square = if let Some(mut sq) = board.en_passant
        && board.can_capture_ep(sq, board.stm)
    {
        if !stm_is_white {
            sq = sq.flip_rank();
        }
        sq.0
    } else {
        64
    };

    entry
}

/// Reconstruct a FEN string from a [`SoulEntry`].
pub fn to_fen(entry: &SoulEntry) -> String {
    // Scatter pieces onto a flat board.
    let mut board = [b'.'; 64];

    for i in 0..entry.piece_count as usize {
        let (pt, mut color, mut sq) = entry.pieces[i].unpack();
        if entry.original_stm == 1 {
            sq = sq.flip_rank();
            color = color.opposite();
        }
        if pt < 6 {
            let ci = if color == Color::White { 0 } else { 1 };
            board[usize::from(sq)] = PIECE_CHARS[ci][pt];
        }
    }

    // Build the FEN directly into a single String — no intermediate Vec<String>.
    let mut fen = String::with_capacity(80);

    for rank in (0..8usize).rev() {
        if rank < 7 {
            fen.push('/');
        }
        let mut empty = 0u8;
        for file in 0..8usize {
            let ch = board[rank * 8 + file];
            if ch == b'.' {
                empty += 1;
            } else {
                if empty > 0 {
                    fen.push((b'0' + empty) as char);
                    empty = 0;
                }
                fen.push(ch as char);
            }
        }
        if empty > 0 {
            fen.push((b'0' + empty) as char);
        }
    }

    fen.push(' ');
    fen.push(
        if entry.original_stm == STM_WHITE {
            'w'
        } else {
            'b'
        },
    );
    fen.push(' ');

    let mut castling = entry.castling;
    if entry.original_stm != STM_WHITE {
        castling = (castling >> 2) | ((castling & 0x3) << 2);
    }

    if castling == 0 {
        fen.push('-');
    } else {
        if castling & 1 != 0 {
            fen.push('K');
        }
        if castling & 2 != 0 {
            fen.push('Q');
        }
        if castling & 4 != 0 {
            fen.push('k');
        }
        if castling & 8 != 0 {
            fen.push('q');
        }
    }

    fen.push(' ');

    // En passant
    if entry.ep_square >= 64 {
        fen.push('-');
    } else {
        let mut sq = crate::core::defs::Square(entry.ep_square);
        if entry.original_stm != STM_WHITE {
            sq = sq.flip_rank();
        }
        fen.push_str(&sq.to_string());
    }

    fen.push_str(" 0 1");
    fen
}

/// Positional openness derived from the pawn structure in a normalised entry.
///
/// Returns `(openness, closedness)` where `closedness = 1.0 - openness`.
pub fn compute_openness_factors(entry: &SoulEntry) -> (f64, f64) {
    let mut us_pawns = 0u64;
    let mut them_pawns = 0u64;

    for i in 0..entry.piece_count as usize {
        let (pt, color, sq) = entry.pieces[i].unpack();
        if pt == 0 {
            let bit = 1u64 << u8::from(sq);
            if color == Color::White {
                us_pawns |= bit;
            } else {
                them_pawns |= bit;
            }
        }
    }

    // Openness computed via the raw bitboard formula from mobility.rs.
    let open_i32 = crate::engine::mobility::compute_openness_raw(us_pawns, them_pawns);
    let openness = f64::from(open_i32) / f64::from(OPEN_UNITY);
    (openness, 1.0 - openness)
}
