//! Regression tests for board state, move application, and game logic.

use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, STARTPOS, WHITE_OO, WHITE_OOO},
        defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
        moves::Move,
    },
    engine::movegen::gen_legal_moves,
};

#[test]
fn fen_roundtrip_startpos() {
    let pos = Position::from_fen(STARTPOS);
    let fen = pos.as_fen();
    let pos2 = Position::from_fen(&fen);
    assert_eq!(pos.hash, pos2.hash, "FEN round-trip changed the Zobrist hash");
    assert_eq!(pos.occ, pos2.occ, "FEN round-trip changed occupancy");
}

#[test]
fn fen_roundtrip_complex() {
    let fens = [
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - -",
    ];

    for fen in fens {
        let pos = Position::from_fen(fen);
        let roundtrip = pos.as_fen();
        let pos2 = Position::from_fen(&roundtrip);
        assert_eq!(pos.hash, pos2.hash, "FEN round-trip hash mismatch for: {fen}");
    }
}

#[test]
fn zobrist_make_unmake_identity() {
    let positions = [
        STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
    ];

    for fen in positions {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.get_initial_accumulator();

        let original_hash = pos.hash;
        let original_fen = pos.as_fen();
        let moves = gen_legal_moves(&pos);
        for mv in &moves {
            let saved_acc = acc;
            let undo = pos.make_move(*mv, &mut acc);
            pos.unmake_move(*mv, &undo);
            acc = saved_acc;
            assert_eq!(pos.hash, original_hash, "Zobrist hash not restored after make/unmake {} in {fen}", mv.to_uci(false));
            assert_eq!(pos.as_fen(), original_fen, "FEN not restored after make/unmake {} in {fen}", mv.to_uci(false));
        }
    }
}

#[test]
fn zobrist_incremental_matches_full() {
    let mut pos = Position::from_fen(STARTPOS);
    let mut acc = pos.get_initial_accumulator();

    let moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"];
    for uci in moves {
        let mv = find_uci_move(&pos, uci);
        pos.make_move(mv, &mut acc);
        let expected = pos.calc_zobrist();
        assert_eq!(pos.hash, expected, "Incremental hash diverged after {uci}: got {:#x}, expected {:#x}", pos.hash, expected);
    }
}

#[test]
fn zobrist_long_sequence() {
    let mut pos = Position::from_fen(STARTPOS);
    let mut acc = pos.get_initial_accumulator();

    // Ruy Lopez: Exchange Variation line
    let game = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "e1g1", "f7f6", "d2d4", "e5d4", "f3d4", "c6c5", "d4b3",
        "d8d1", "f1d1",
    ];

    for uci in game {
        let mv = find_uci_move(&pos, uci);
        pos.make_move(mv, &mut acc);
        assert_eq!(pos.hash, pos.calc_zobrist(), "Hash diverged after move {uci}. Position: {}", pos.as_fen());
        assert_eq!(pos.pawn_key, pos.calc_pawn_hash(), "Pawn key diverged after move {uci}. Position: {}", pos.as_fen());
        assert_eq!(pos.minor_key, pos.calc_minor_hash(), "Minor key diverged after move {uci}. Position: {}", pos.as_fen());
        assert_eq!(pos.major_key, pos.calc_major_hash(), "Major key diverged after move {uci}. Position: {}", pos.as_fen());
        let fresh_acc = pos.get_initial_accumulator();
        assert_eq!(acc.to_array(), fresh_acc.to_array(), "Accumulator diverged after move {uci}");
    }
}

#[test]
fn correction_keys_promotion_and_en_passant() {
    let cases = [
        ("8/P7/8/8/8/8/k7/7K w - - 0 1", "a7a8q"),     // queen promotion → major_key
        ("8/P7/8/8/8/8/k7/7K w - - 0 1", "a7a8n"),     // knight underpromotion → minor_key
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"), // en passant → pawn_key
    ];

    for (fen, uci) in cases {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.get_initial_accumulator();
        let before = (pos.pawn_key, pos.minor_key, pos.major_key);
        let mv = find_uci_move(&pos, uci);
        let undo = pos.make_move(mv, &mut acc);
        assert_eq!(pos.pawn_key, pos.calc_pawn_hash(), "Pawn key diverged after {uci} from {fen}");
        assert_eq!(pos.minor_key, pos.calc_minor_hash(), "Minor key diverged after {uci} from {fen}");
        assert_eq!(pos.major_key, pos.calc_major_hash(), "Major key diverged after {uci} from {fen}");
        pos.unmake_move(mv, &undo);
        assert_eq!((pos.pawn_key, pos.minor_key, pos.major_key), before, "Keys not restored after unmaking {uci} from {fen}");
    }
}

#[test]
fn accumulator_incremental_matches_full() {
    let positions = [STARTPOS, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -"];
    for fen in positions {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.get_initial_accumulator();

        let moves = gen_legal_moves(&pos);
        for mv in &moves {
            let saved_acc = acc;
            let undo = pos.make_move(*mv, &mut acc);
            let fresh_acc = pos.get_initial_accumulator();
            assert_eq!(acc.to_array(), fresh_acc.to_array(), "Accumulator mismatch after {} in {fen}", mv.to_uci(false));
            pos.unmake_move(*mv, &undo);
            acc = saved_acc;
        }
    }
}

#[test]
fn castling_kingside_white() {
    let mut pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -");
    let mut acc = pos.get_initial_accumulator();

    let mv = Move::new(Square(4), Square(7), Move::CASTLE);
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square(6)), PieceType::King, "King should be on g1");
    assert_eq!(pos.piece_at(Square(5)), PieceType::Rook, "Rook should be on f1");
    assert_eq!(pos.piece_at(Square(4)), PieceType::None, "e1 should be empty");
    assert_eq!(pos.piece_at(Square(7)), PieceType::None, "h1 should be empty");
    let expected_hash = pos.calc_zobrist();
    assert_eq!(pos.hash, expected_hash, "Hash mismatch after O-O");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.piece_at(Square(4)), PieceType::King, "King not restored to e1");
    assert_eq!(pos.piece_at(Square(7)), PieceType::Rook, "Rook not restored to h1");
}

#[test]
fn castling_queenside_black() {
    let mut pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R b KQkq -");
    let mut acc = pos.get_initial_accumulator();

    let mv = Move::new(Square(60), Square(56), Move::CASTLE);
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square(58)), PieceType::King, "King should be on c8");
    assert_eq!(pos.piece_at(Square(59)), PieceType::Rook, "Rook should be on d8");
    assert_eq!(pos.piece_at(Square(60)), PieceType::None, "e8 should be empty");
    assert_eq!(pos.piece_at(Square(56)), PieceType::None, "a8 should be empty");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.piece_at(Square(60)), PieceType::King, "King not restored to e8");
    assert_eq!(pos.piece_at(Square(56)), PieceType::Rook, "Rook not restored to a8");
}

#[test]
fn en_passant_capture() {
    let mut pos = Position::from_fen("4k3/8/8/3Pp3/8/8/8/4K3 w - e6 0 1");
    let mut acc = pos.get_initial_accumulator();

    let original_hash = pos.hash;
    let mv = Move::new(Square::from_coords(3, 4), Square::from_coords(4, 5), Move::EP_CAPTURE);
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square::from_coords(4, 4)), PieceType::None, "EP victim should be removed");
    assert_eq!(pos.piece_at(Square::from_coords(4, 5)), PieceType::Pawn, "Pawn should land on e6");
    let expected_hash = pos.calc_zobrist();
    assert_eq!(pos.hash, expected_hash, "Hash mismatch after EP capture");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.hash, original_hash, "Hash not restored after EP unmake");
    assert_eq!(pos.piece_at(Square::from_coords(4, 4)), PieceType::Pawn, "EP victim not restored");
}

#[test]
fn promotion_queen() {
    let mut pos = Position::from_fen("4k3/P7/8/8/8/8/8/4K3 w - - 0 1");
    let mut acc = pos.get_initial_accumulator();

    let mv = Move::new(Square::from_coords(0, 6), Square::from_coords(0, 7), Move::PROM_Q);
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square::from_coords(0, 7)), PieceType::Queen, "Promoted piece should be queen");
    assert_eq!(pos.piece_at(Square::from_coords(0, 6)), PieceType::None, "Old square should be empty");
    let expected_hash = pos.calc_zobrist();
    assert_eq!(pos.hash, expected_hash, "Hash mismatch after promotion");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.piece_at(Square::from_coords(0, 6)), PieceType::Pawn, "Pawn not restored after promotion unmake");
}

#[test]
fn draw_by_material_kvk() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert!(pos.is_draw_by_material(), "K vs K should be a draw");
}

#[test]
fn draw_by_material_kn_vs_k() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/3NK3 w - - 0 1");
    assert!(pos.is_draw_by_material(), "KN vs K should be a draw");
}

#[test]
fn not_draw_knn_vs_k() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1");
    assert!(!pos.is_draw_by_material(), "KNN vs K should NOT be a material draw (50mr handles it)");
}

#[test]
fn not_draw_with_pawns() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/P7/4K3 w - - 0 1");
    assert!(!pos.is_draw_by_material(), "Any pawns → not a material draw");
}

#[test]
fn threefold_repetition() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 10 1");
    let history = vec![pos.hash, 999, pos.hash, 888, pos.hash];
    assert!(pos.is_threefold_repetition(&history));
}

#[test]
fn move_encoding_roundtrip() {
    for from in 0..64u8 {
        for to in 0..64u8 {
            for flag in [Move::QUIET, Move::CAPTURE, Move::PROM_Q, Move::CASTLE, Move::EP_CAPTURE, Move::DOUBLE_PUSH] {
                let mv = Move::new(Square(from), Square(to), flag);
                assert_eq!(mv.from().0, from, "from mismatch");
                assert_eq!(mv.to().0, to, "to mismatch");
                assert_eq!(mv.flag(), flag, "flag mismatch for flag {flag}");
            }
        }
    }
}

#[test]
fn move_flag_predicates() {
    let quiet = Move::new(Square(0), Square(8), Move::QUIET);
    let cap = Move::new(Square(0), Square(8), Move::CAPTURE);
    let ep = Move::new(Square(0), Square(8), Move::EP_CAPTURE);
    let castle = Move::new(Square(4), Square(7), Move::CASTLE);
    let prom = Move::new(Square(0), Square(8), Move::PROM_Q);
    let prom_cap = Move::new(Square(0), Square(8), Move::PROM_Q_CAPTURE);
    let double = Move::new(Square(0), Square(8), Move::DOUBLE_PUSH);

    assert!(quiet.is_quiet());
    assert!(!quiet.is_capture());
    assert!(!quiet.is_promotion());

    assert!(cap.is_capture());
    assert!(!cap.is_quiet());
    assert!(cap.is_tactical());

    assert!(ep.is_en_passant());
    assert!(ep.is_capture());

    assert!(castle.is_castling());
    assert!(!castle.is_capture());

    assert!(prom.is_promotion());
    assert!(prom.is_tactical());
    assert!(!prom.is_capture());

    assert!(prom_cap.is_promotion());
    assert!(prom_cap.is_capture());
    assert!(prom_cap.is_tactical());

    assert!(double.is_double_push());
    assert!(!double.is_capture());
}

#[test]
fn move_promotion_piece_types() {
    let n = Move::new(Square(0), Square(8), Move::PROM_N);
    let b = Move::new(Square(0), Square(8), Move::PROM_B);
    let r = Move::new(Square(0), Square(8), Move::PROM_R);
    let q = Move::new(Square(0), Square(8), Move::PROM_Q);

    assert_eq!(n.promo(), Some(PieceType::Knight));
    assert_eq!(b.promo(), Some(PieceType::Bishop));
    assert_eq!(r.promo(), Some(PieceType::Rook));
    assert_eq!(q.promo(), Some(PieceType::Queen));

    let quiet = Move::new(Square(0), Square(8), Move::QUIET);
    assert_eq!(quiet.promo(), None);
}

#[test]
fn material_count_startpos() {
    let pos = Position::from_fen(STARTPOS);
    // 16P + 4N + 4B + 4R + 2Q = 16·1 + 4·3 + 4·3 + 4·5 + 2·9 = 78
    assert_eq!(pos.material_count(), 78);
}

#[test]
fn material_count_bare_kings() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    assert_eq!(pos.material_count(), 0, "Bare kings should have 0 material");
}

#[test]
fn startpos_piece_layout() {
    let pos = Position::from_fen(STARTPOS);

    assert_eq!(pos.piece_at(Square(0)), PieceType::Rook); // a1
    assert_eq!(pos.piece_at(Square(4)), PieceType::King); // e1
    assert_eq!(pos.piece_at(Square(12)), PieceType::Pawn); // e2
    assert_eq!(pos.piece_at(Square(28)), PieceType::None); // e4 (empty)
    assert_eq!(pos.piece_at(Square(60)), PieceType::King); // e8

    assert_eq!(pos.color_at(Square(0)), Color::White);
    assert_eq!(pos.color_at(Square(56)), Color::Black);

    assert_eq!(pos.stm, Color::White);
}

#[test]
fn eval_startpos_tempo_only() {
    use crate::engine::{eval::evaluate, eval_params::TEMPO};

    let pos = Position::from_fen(STARTPOS);
    let acc = pos.get_initial_accumulator();
    let score = evaluate(&pos, &acc);
    assert_eq!(score, TEMPO[0], "got {score}");
}

#[test]
fn eval_stm_symmetry() {
    use crate::engine::eval::evaluate;

    let w = Position::from_fen("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1");
    let b = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let w_acc = w.get_initial_accumulator();
    let b_acc = b.get_initial_accumulator();
    let w_score = evaluate(&w, &w_acc);
    let b_score = evaluate(&b, &b_acc);
    assert_eq!(w_score, b_score, "Mirror positions should eval identically, got w={w_score}, b={b_score}");
}

#[test]
fn eval_strict_symmetry() {
    use crate::engine::eval::evaluate;

    let w = Position::from_fen(STARTPOS);
    let b = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    let w_acc = w.get_initial_accumulator();
    let b_acc = b.get_initial_accumulator();
    let w_score = evaluate(&w, &w_acc);
    let b_score = evaluate(&b, &b_acc);
    assert_eq!(w_score, b_score);
}

#[test]
fn history_gravity_bounds() {
    use crate::engine::history::{ContContext, History};

    let mut hist = History::new();

    let stm = Color::White;
    let pt = PieceType::Knight;
    let to = Square(28);

    // Slam it with huge bonuses; should converge toward +16384 without overflow.
    for _ in 0..10_000 {
        hist.update(
            stm,
            pt,
            Square(0),
            to,
            Bitboard(0),
            ContContext::default(),
            ContContext::default(),
            ContContext::default(),
            400,
        );
    }
    let score = hist.score_quiet(
        stm,
        pt,
        Square(0),
        to,
        Bitboard(0),
        ContContext::default(),
        ContContext::default(),
        ContContext::default(),
    );

    assert!(score > 0, "After massive positive bonus, score should be positive: {score}");
    assert!(score <= 32768, "Score should not exceed combined gravity bound (16384 * 2): {score}");

    // Now slam it negative; should converge toward -16384.
    for _ in 0..20_000 {
        hist.update(
            stm,
            pt,
            Square(0),
            to,
            Bitboard(0),
            ContContext::default(),
            ContContext::default(),
            ContContext::default(),
            -400,
        );
    }
    let score = hist.score_quiet(
        stm,
        pt,
        Square(0),
        to,
        Bitboard(0),
        ContContext::default(),
        ContContext::default(),
        ContContext::default(),
    );
    assert!(score < 0, "After massive negative bonus, score should be negative: {score}");
    assert!(score >= -32768, "Score should not go below -32768: {score}");
}

#[test]
fn history_clear() {
    use crate::engine::history::{ContContext, History};

    let mut hist = History::new();

    hist.update(
        Color::White,
        PieceType::Knight,
        Square(0),
        Square(28),
        Bitboard(0),
        ContContext::default(),
        ContContext::default(),
        ContContext::default(),
        400,
    );
    assert!(
        hist.score_quiet(
            Color::White,
            PieceType::Knight,
            Square(0),
            Square(28),
            Bitboard(0),
            ContContext::default(),
            ContContext::default(),
            ContContext::default()
        ) > 0
    );

    hist.clear();
    assert_eq!(
        hist.score_quiet(
            Color::White,
            PieceType::Knight,
            Square(0),
            Square(28),
            Bitboard(0),
            ContContext::default(),
            ContContext::default(),
            ContContext::default()
        ),
        0,
        "Clear should zero out"
    );
}

#[test]
fn phase_startpos() {
    use crate::engine::eval::extract_phase;

    let pos = Position::from_fen(STARTPOS);
    let acc = pos.get_initial_accumulator();
    let phase = extract_phase(&acc);
    assert_eq!(phase, TOTAL_PHASE, "Startpos should have full phase (24)");
}

#[test]
fn phase_endgame() {
    use crate::engine::eval::extract_phase;

    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let acc = pos.get_initial_accumulator();
    let phase = extract_phase(&acc);
    assert_eq!(phase, 0, "Bare kings should have phase 0");
}

#[test]
fn wdl_monotonicity() {
    use crate::engine::wdl::wdl_model;

    let material = 78; // Full material
    let (mut last_w, _, mut last_l) = wdl_model(-2000, material);

    // As score increases, win prob should increase, loss prob should decrease.
    for score in (-1500..=2000).step_by(500) {
        let (w, d, l) = wdl_model(score, material);
        assert!(w >= last_w, "Win prob decreased at score {score}");
        assert!(l <= last_l, "Loss prob increased at score {score}");
        assert!((w + d + l - 1.0).abs() < 1e-6, "Probabilities should sum to 1.0");
        last_w = w;
        last_l = l;
    }
}

#[test]
fn square_coordinate_mapping() {
    // Square 0 = A1
    let a1 = Square::from_coords(0, 0);
    assert_eq!(a1.0, 0);
    assert_eq!(a1.rank(), 0);
    assert_eq!(a1.file(), 0);
    // Square 63 = H8
    let h8 = Square::from_coords(7, 7);
    assert_eq!(h8.0, 63);
    assert_eq!(h8.rank(), 7);
    assert_eq!(h8.file(), 7);
    // Mirroring/Flipping
    assert_eq!(a1.flip_rank().0, 56); // a8
    assert_eq!(a1.flip_file().0, 7); // h1
}

#[test]
fn pinned_pieces_detection() {
    // 1. Orthogonal Pin
    let pos = Position::from_fen("k3r3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let pinned = pos.pinned_pieces(Color::White);
    assert!(pinned.check_bit(Square::from_coords(4, 1)), "Pawn on e2 should be pinned orthogonally");
    // 2. Diagonal Pin
    let pos = Position::from_fen("4k3/8/8/b7/8/2P5/8/4K3 w - - 0 1");
    let pinned = pos.pinned_pieces(Color::White);
    assert!(pinned.check_bit(Square::from_coords(2, 2)), "Pawn on c3 should be pinned diagonally");
    // 3. Verify pins don't apply to the other side
    let pinned_b = pos.pinned_pieces(Color::Black);
    assert!(pinned_b.is_empty(), "Black should have no pinned pieces");
}

#[test]
fn draw_by_material_kb_vs_k() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/2BK4 w - - 0 1");
    assert!(pos.is_draw_by_material(), "K+B vs K should be a draw");
}

#[test]
fn not_draw_rook_vs_king() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/2RK4 w - - 0 1");
    assert!(!pos.is_draw_by_material(), "R vs K is NOT a material draw");
}

#[test]
fn fifty_move_draw_detection() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 100 50");
    assert!(pos.is_fifty_move_draw(), "halfmove=100 should be a 50-move draw");
    let pos2 = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 99 50");
    assert!(!pos2.is_fifty_move_draw(), "halfmove=99 is not yet a draw");
}

#[test]
fn double_push_sets_en_passant() {
    let mut pos = Position::from_fen("4k3/8/8/8/3p4/8/4P3/4K3 w - - 0 1");
    let mut acc = pos.get_initial_accumulator();
    let mv = find_uci_move(&pos, "e2e4");
    pos.make_move(mv, &mut acc);
    assert_eq!(pos.en_passant, Some(Square(20)), "e2e4 double push should set EP square e3 when capture is possible");
}

#[test]
fn promotion_capture_and_undo() {
    let mut pos = Position::from_fen("2r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    let mut acc = pos.get_initial_accumulator();
    // b7c8=Q
    let mv = Move::new(Square::from_coords(1, 6), Square::from_coords(2, 7), Move::PROM_Q_CAPTURE);
    let original_hash = pos.hash;
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(
        pos.piece_at(Square::from_coords(2, 7)),
        PieceType::Queen,
        "Piece on c8 should be a Queen after PROM_Q_CAPTURE"
    );
    assert_eq!(pos.color_at(Square::from_coords(2, 7)), Color::White);
    assert!(pos.pieces(PieceType::Rook, Color::Black).is_empty(), "Black rook on c8 should be captured");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.piece_at(Square::from_coords(1, 6)), PieceType::Pawn, "Pawn not restored to b7");
    assert_eq!(pos.piece_at(Square::from_coords(2, 7)), PieceType::Rook, "Rook not restored to c8");
    assert_eq!(pos.hash, original_hash, "Hash not restored after promotion capture unmake");
}

#[test]
fn checkmate_no_legal_moves() {
    let pos = Position::from_fen("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(moves.is_empty(), "Checkmate position should have 0 legal moves");
    assert!(!pos.checkers().is_empty(), "Should be in check in checkmate");
}

#[test]
fn double_check_only_king_moves() {
    let pos = Position::from_fen("4r3/8/8/8/8/8/8/2k1K2q w - - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.is_empty(), "Should have some escaping moves");
    for mv in &moves {
        assert_eq!(
            pos.piece_at(mv.from()),
            PieceType::King,
            "In double check, only king moves should be legal. Found: {:?}",
            mv.to_uci(false)
        );
    }
}

#[test]
fn knight_under_promotion_check() {
    let mut pos = Position::from_fen("8/1kP5/8/8/8/8/8/4K3 w - - 0 1");
    let mut acc = pos.get_initial_accumulator();
    let mv = Move::new(Square::from_coords(2, 6), Square::from_coords(3, 7), Move::PROM_N); // c7d8=N
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square::from_coords(3, 7)), PieceType::Knight);
    assert!(!pos.checkers().is_empty(), "Promoted knight to d8 should check king on b7");
    pos.unmake_move(mv, &undo);
    assert!(pos.checkers().is_empty(), "Unmaking promotion should remove check");
    assert_eq!(pos.piece_at(Square::from_coords(2, 6)), PieceType::Pawn);
}

#[test]
fn stalemate_no_legal_moves() {
    let pos = Position::from_fen("7k/5K2/6Q1/8/8/8/8/8 b - - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(moves.is_empty(), "Stalemate position should have 0 legal moves");
    assert!(pos.checkers().is_empty(), "Should NOT be in check in stalemate");
}

#[test]
fn perft_suite() {
    let cases = [
        (STARTPOS, 3, 8_902),
        ("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -", 3, 97_862),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -", 3, 2_812),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -", 3, 9_467),
        ("rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8", 3, 62_379),
        ("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", 3, 89_890),
    ];

    for (fen, depth, nodes) in cases {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.get_initial_accumulator();
        assert_eq!(perft_recursive(&mut pos, depth, &mut acc), nodes, "Perft mismatch for {fen} at depth {depth}");
    }
}

fn perft_recursive(pos: &mut Position, depth: u32, acc: &mut crate::weave::Vi16x8) -> u64 {
    if depth == 0 {
        return 1;
    }

    let moves = gen_legal_moves(pos);
    if depth == 1 {
        return moves.len() as u64;
    }

    let mut total = 0u64;

    for &mv in &moves {
        let saved_acc = *acc;
        let undo = pos.make_move(mv, acc);
        total += perft_recursive(pos, depth - 1, acc);
        pos.unmake_move(mv, &undo);
        *acc = saved_acc;
    }
    total
}

#[test]
fn king_move_removes_castling_rights() {
    let mut pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let mut acc = pos.get_initial_accumulator();

    let original_rights = pos.castling_rights;
    let mv = find_uci_move(&pos, "e1d1");
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.castling_rights & (WHITE_OO | WHITE_OOO), 0, "White should lose all castling rights after moving the King");
    assert_ne!(pos.castling_rights & (BLACK_OO | BLACK_OOO), 0, "Black should retain castling rights");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.castling_rights, original_rights, "Castling rights should be restored after unmake");
}

#[test]
fn rook_capture_removes_castling_rights() {
    let mut pos = Position::from_fen("r3k2r/6B1/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let mut acc = pos.get_initial_accumulator();

    let original_rights = pos.castling_rights;
    let mv = find_uci_move(&pos, "g7h8");
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.castling_rights & BLACK_OO, 0, "Black should lose kingside rights after rook is captured");
    assert_ne!(pos.castling_rights & BLACK_OOO, 0, "Black should keep queenside rights");
    pos.unmake_move(mv, &undo);
    assert_eq!(pos.castling_rights, original_rights, "Castling rights should be restored after rook-capture unmake");
}

#[test]
fn castling_legality_checks() {
    // 1. Cannot castle out of check
    let pos = Position::from_fen("1k2r3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling()), "Should not be able to castle out of check");
    // 2. Cannot castle through check
    let pos = Position::from_fen("1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling() && m.to().0 == 7), "Should not castle through check on f1");
    // 3. Cannot castle into check
    let pos = Position::from_fen("1k4r1/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling() && m.to().0 == 7), "Should not castle into check on g1");
}

fn find_uci_move(pos: &Position, uci: &str) -> Move {
    let moves = gen_legal_moves(pos);
    for mv in &moves {
        if mv.to_uci(pos.is_frc) == uci {
            return *mv;
        }
    }
    panic!("Move {uci} not found in position {}", pos.as_fen());
}
