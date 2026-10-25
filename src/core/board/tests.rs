use crate::{
    core::{
        board::{BLACK_OO, BLACK_OOO, Position, ROOK_W_KS, ROOK_W_QS, STARTPOS, WHITE_OO, WHITE_OOO},
        defs::{Bitboard, Color, PieceType, Square, TOTAL_PHASE},
        error::FenError,
        moves::Move,
    },
    engine::movegen::gen_legal_moves,
    tools::perft::perft,
};

#[test]
fn fen_roundtrip() {
    let fens = [
        STARTPOS,
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq -",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - -",
        "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq -",
        "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ -",
        "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/3P1N1P/PPP1NPP1/R2Q1RK1 w - - 37 81",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 81",
        "r1k2r2/8/8/8/8/8/8/R1K2R2 w FAfa - 9 12",
    ];

    for fen in fens {
        let pos = Position::from_fen(fen);
        let roundtrip = pos.to_fen();
        let decoded = Position::from_fen(&roundtrip);
        assert_eq!(decoded.side_bb, pos.side_bb, "{fen}");
        assert_eq!(decoded.role_bb, pos.role_bb, "{fen}");
        assert_eq!(decoded.occ, pos.occ, "{fen}");
        assert_eq!(decoded.pieces, pos.pieces, "{fen}");
        assert_eq!(decoded.stm, pos.stm, "{fen}");
        assert_eq!(decoded.castling_rights, pos.castling_rights, "{fen}");
        assert_eq!(decoded.castling_rooks, pos.castling_rooks, "{fen}");
        assert_eq!(decoded.en_passant, pos.en_passant, "{fen}");
        assert_eq!(decoded.halfmove_clock, pos.halfmove_clock, "{fen}");
        assert_eq!(decoded.fullmove_number, pos.fullmove_number, "{fen}");
        assert_eq!(decoded.frc_notation, pos.frc_notation, "{fen}");
        assert_eq!(decoded.hash, pos.hash, "{fen}");
        assert_eq!(decoded.pawn_key, pos.pawn_key, "{fen}");
        assert_eq!(decoded.minor_key, pos.minor_key, "{fen}");
        assert_eq!(decoded.major_key, pos.major_key, "{fen}");
    }
}

#[test]
fn fen_castling_rights_are_strict() {
    for fen in [
        "4k3/8/8/8/8/8/8/4K2r w H - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w K - 0 1",
        "4k3/8/8/8/8/8/8/4K3 w Z - 0 1",
        "4k3/8/8/8/8/8/8/R3K3 w K - 0 1",
    ] {
        assert!(matches!(Position::try_from_fen(fen), Err(FenError::InvalidCastlingRight { .. })), "{fen}");
    }
    assert!(matches!(Position::try_from_fen("4k3/8/8/8/8/8/4K3/7R w H - 0 1"), Err(FenError::InvalidCastlingRight { .. })));

    let frc = Position::from_fen("4k3/8/8/8/8/8/8/R1K2R2 w FA - 0 1");
    assert_eq!(frc.castling_rights, WHITE_OO | WHITE_OOO);
    assert_eq!((frc.castling_rooks[ROOK_W_KS], frc.castling_rooks[ROOK_W_QS]), (Square(5), Square(0)));
}

#[test]
fn fen_en_passant_rank() {
    for fen in ["4k3/8/8/8/p7/8/1P6/4K3 w - a3 0 1", "4k3/1p6/8/P7/8/8/8/4K3 b - a6 0 1"] {
        assert!(matches!(Position::try_from_fen(fen), Err(FenError::InvalidEnPassant { .. })), "{fen}");
    }
}

#[test]
fn fen_en_passant_normalization() {
    for (fen, ep) in [
        ("4k3/8/8/pP6/8/8/8/4K3 w - a6 0 1", Some(Square(40))),
        ("4k3/8/8/PP6/8/8/8/4K3 w - a6 0 1", None),
        ("4k3/8/N7/pP6/8/8/8/4K3 w - a6 0 1", None),
        ("k2r4/8/8/3Pp3/8/8/8/3K4 w - e6 0 1", Some(Square(44))),
    ] {
        assert_eq!(Position::from_fen(fen).en_passant, ep, "{fen}");
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
        let mut acc = pos.psqt_accumulator();

        let original_hash = pos.hash;
        let original_fen = pos.to_fen();
        let moves = gen_legal_moves(&pos);
        for mv in &moves {
            let saved_acc = acc;
            let undo = pos.make_move(*mv, &mut acc);
            pos.unmake_move(*mv, &undo);
            acc = saved_acc;
            assert_eq!(pos.hash, original_hash, "Zobrist hash not restored after make/unmake {} in {fen}", mv.to_uci(false));
            assert_eq!(pos.to_fen(), original_fen, "FEN not restored after make/unmake {} in {fen}", mv.to_uci(false));
        }
    }
}

#[test]
fn zobrist_long_sequence() {
    let mut pos = Position::from_fen(STARTPOS);
    let mut acc = pos.psqt_accumulator();

    let game = [
        "e2e4", "e7e5", "g1f3", "b8c6", "f1b5", "a7a6", "b5c6", "d7c6", "e1g1", "f7f6", "d2d4", "e5d4", "f3d4", "c6c5", "d4b3",
        "d8d1", "f1d1",
    ];

    for uci in game {
        let mv = find_uci_move(&pos, uci);
        pos.make_move(mv, &mut acc);
        assert_eq!(pos.hash, pos.calc_zobrist(), "Hash diverged after move {uci}. Position: {}", pos.to_fen());
        assert_eq!(pos.pawn_key, pos.calc_pawn_hash(), "Pawn key diverged after move {uci}. Position: {}", pos.to_fen());
        assert_eq!(pos.minor_key, pos.calc_minor_hash(), "Minor key diverged after move {uci}. Position: {}", pos.to_fen());
        assert_eq!(pos.major_key, pos.calc_major_hash(), "Major key diverged after move {uci}. Position: {}", pos.to_fen());
        let fresh_acc = pos.psqt_accumulator();
        assert_eq!(acc.to_array(), fresh_acc.to_array(), "Accumulator diverged after move {uci}");
    }
}

#[test]
fn correction_keys_promotion_and_en_passant() {
    let cases = [
        ("8/P7/8/8/8/8/k7/7K w - - 0 1", "a7a8q"),
        ("8/P7/8/8/8/8/k7/7K w - - 0 1", "a7a8n"),
        ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
    ];

    for (fen, uci) in cases {
        let mut pos = Position::from_fen(fen);
        let mut acc = pos.psqt_accumulator();
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
        let mut acc = pos.psqt_accumulator();

        let moves = gen_legal_moves(&pos);
        for mv in &moves {
            let saved_acc = acc;
            let undo = pos.make_move(*mv, &mut acc);
            let fresh_acc = pos.psqt_accumulator();
            assert_eq!(acc.to_array(), fresh_acc.to_array(), "Accumulator mismatch after {} in {fen}", mv.to_uci(false));
            pos.unmake_move(*mv, &undo);
            acc = saved_acc;
        }
    }
}

#[test]
fn castling_kingside_white() {
    let mut pos = Position::from_fen("r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq -");
    let mut acc = pos.psqt_accumulator();

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
    let mut acc = pos.psqt_accumulator();

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
    let mut acc = pos.psqt_accumulator();

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
    let mut acc = pos.psqt_accumulator();

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
fn draw_by_material() {
    for (fen, draw) in [
        ("4k3/8/8/8/8/8/8/4K3 w - - 0 1", true),
        ("4k3/8/8/8/8/8/8/3NK3 w - - 0 1", true),
        ("4k3/8/8/8/8/8/8/2BK4 w - - 0 1", true),
        ("4k3/8/8/8/8/8/8/2NNK3 w - - 0 1", false),
        ("4k3/8/8/8/8/8/8/2RK4 w - - 0 1", false),
        ("4k3/8/8/8/8/8/P7/4K3 w - - 0 1", false),
    ] {
        assert_eq!(Position::from_fen(fen).is_draw_by_material(), draw, "{fen}");
    }
}

#[test]
fn threefold_repetition() {
    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 10 1");
    let history = [pos.hash, 999, pos.hash, 888, pos.hash];
    assert!(pos.is_threefold_repetition(&history));
    let twice = [pos.hash, 999, 777, 888, pos.hash];
    assert!(!pos.is_threefold_repetition(&twice), "two occurrences is not threefold");
    assert!(!pos.is_threefold_repetition(&[pos.hash, 999, pos.hash]), "a history this short cannot hold three");
}

#[test]
fn move_encoding_roundtrip() {
    for from in 0..64u8 {
        for to in 0..64u8 {
            for flag in 0..16 {
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
    for (quiet, capture, piece) in [
        (Move::PROM_N, Move::PROM_N_CAPTURE, PieceType::Knight),
        (Move::PROM_B, Move::PROM_B_CAPTURE, PieceType::Bishop),
        (Move::PROM_R, Move::PROM_R_CAPTURE, PieceType::Rook),
        (Move::PROM_Q, Move::PROM_Q_CAPTURE, PieceType::Queen),
    ] {
        for flag in [quiet, capture] {
            assert_eq!(Move::new(Square(0), Square(8), flag).promo(), Some(piece), "flag {flag}");
        }
    }

    let quiet = Move::new(Square(0), Square(8), Move::QUIET);
    assert_eq!(quiet.promo(), None);
}

#[test]
fn material_count_startpos() {
    let pos = Position::from_fen(STARTPOS);
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

    assert_eq!(pos.piece_count(PieceType::None), 0);
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
    let acc = pos.psqt_accumulator();
    let score = evaluate(&pos, &acc);
    assert_eq!(score, TEMPO[0], "got {score}");
}

#[test]
fn eval_stm_symmetry() {
    use crate::engine::eval::evaluate;

    let w = Position::from_fen("rnbqkbnr/pppp1ppp/8/4p3/8/8/PPPPPPPP/RNBQKBNR w KQkq e6 0 1");
    let b = Position::from_fen("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1");
    let w_acc = w.psqt_accumulator();
    let b_acc = b.psqt_accumulator();
    let w_score = evaluate(&w, &w_acc);
    let b_score = evaluate(&b, &b_acc);
    assert_eq!(w_score, b_score, "Mirror positions should eval identically, got w={w_score}, b={b_score}");
}

#[test]
fn eval_strict_symmetry() {
    use crate::engine::eval::evaluate;

    let w = Position::from_fen(STARTPOS);
    let b = Position::from_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 0 1");
    let w_acc = w.psqt_accumulator();
    let b_acc = b.psqt_accumulator();
    let w_score = evaluate(&w, &w_acc);
    let b_score = evaluate(&b, &b_acc);
    assert_eq!(w_score, b_score);
}

#[test]
fn history_gravity_reaches_caps() {
    use crate::engine::history::{ContContext, History};

    let mut hist = History::new();

    let stm = Color::White;
    let pt = PieceType::Knight;
    let from = Square(0);
    let to = Square(28);
    let threats = Bitboard::EMPTY;
    let cont = ContContext::default();
    let cap = hist.params.quiet_cap + hist.params.butterfly_cap;

    for _ in 0..10_000 {
        hist.update_quiet(stm, pt, from, to, threats, cont, cont, cont, 400);
    }

    let score = hist.score_quiet(stm, pt, from, to, threats, cont, cont, cont);
    assert_eq!(score, cap);

    for _ in 0..20_000 {
        hist.update_quiet(stm, pt, from, to, threats, cont, cont, cont, -400);
    }
    let score = hist.score_quiet(stm, pt, from, to, threats, cont, cont, cont);
    assert_eq!(score, -cap);
}

#[test]
fn history_clear() {
    use crate::engine::history::{ContContext, History};

    let mut hist = History::new();
    let stm = Color::White;
    let pt = PieceType::Knight;
    let from = Square(0);
    let to = Square(28);
    let threats = Bitboard::EMPTY;
    let cont = ContContext::default();

    hist.update_quiet(stm, pt, from, to, threats, cont, cont, cont, 400);
    assert!(hist.score_quiet(stm, pt, from, to, threats, cont, cont, cont) > 0);

    hist.clear();
    assert_eq!(hist.score_quiet(stm, pt, from, to, threats, cont, cont, cont), 0);
}

#[test]
fn phase_startpos() {
    use crate::engine::eval::extract_phase;

    let pos = Position::from_fen(STARTPOS);
    let acc = pos.psqt_accumulator();
    let phase = extract_phase(&acc);
    assert_eq!(phase, TOTAL_PHASE, "Startpos should have full phase (24)");
}

#[test]
fn phase_endgame() {
    use crate::engine::eval::extract_phase;

    let pos = Position::from_fen("4k3/8/8/8/8/8/8/4K3 w - - 0 1");
    let acc = pos.psqt_accumulator();
    let phase = extract_phase(&acc);
    assert_eq!(phase, 0, "Bare kings should have phase 0");
}

#[test]
fn wdl_monotonicity() {
    use crate::engine::wdl::wdl_model;

    let material = 78;
    let (mut last_w, _, mut last_l) = wdl_model(-2000, material);

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
    for rank in 0..8 {
        for file in 0..8 {
            let sq = Square::from_coords(file, rank);
            assert_eq!(sq.0, rank * 8 + file);
            assert_eq!(sq.rank(), rank);
            assert_eq!(sq.file(), file);
            assert_eq!(sq.flip_rank().0, (7 - rank) * 8 + file);
        }
    }
}

#[test]
fn pinned_pieces_detection() {
    let pos = Position::from_fen("k3r3/8/8/8/8/8/4P3/4K3 w - - 0 1");
    let pinned = pos.pinned_pieces(Color::White);
    assert!(pinned.check_bit(Square::from_coords(4, 1)), "Pawn on e2 should be pinned orthogonally");
    let pos = Position::from_fen("4k3/8/8/b7/8/2P5/8/4K3 w - - 0 1");
    let pinned = pos.pinned_pieces(Color::White);
    assert!(pinned.check_bit(Square::from_coords(2, 2)), "Pawn on c3 should be pinned diagonally");
    let pinned_b = pos.pinned_pieces(Color::Black);
    assert!(pinned_b.is_empty(), "Black should have no pinned pieces");
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
    let mut acc = pos.psqt_accumulator();
    let mv = find_uci_move(&pos, "e2e4");
    pos.make_move(mv, &mut acc);
    assert_eq!(pos.en_passant, Some(Square(20)), "e2e4 double push should set EP square e3 when capture is possible");
}

#[test]
fn promotion_capture_and_undo() {
    let mut pos = Position::from_fen("2r1k3/1P6/8/8/8/8/8/4K3 w - - 0 1");
    let mut acc = pos.psqt_accumulator();

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
fn knight_underpromotion_gives_check() {
    let mut pos = Position::from_fen("8/1k1P4/8/8/8/8/8/4K3 w - - 0 1");
    let mut acc = pos.psqt_accumulator();

    let mv = find_uci_move(&pos, "d7d8n");
    let undo = pos.make_move(mv, &mut acc);
    assert_eq!(pos.piece_at(Square::from_coords(3, 7)), PieceType::Knight);
    assert!(!pos.checkers().is_empty(), "Promoted knight to d8 should check king on b7");
    pos.unmake_move(mv, &undo);
    assert!(pos.checkers().is_empty(), "Unmaking promotion should remove check");
    assert_eq!(pos.piece_at(Square::from_coords(3, 6)), PieceType::Pawn);
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
        let mut acc = pos.psqt_accumulator();
        assert_eq!(perft(&mut pos, depth, &mut acc, true), nodes, "Perft mismatch for {fen} at depth {depth}");
    }
}

#[test]
fn king_move_removes_castling_rights() {
    let mut pos = Position::from_fen("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1");
    let mut acc = pos.psqt_accumulator();

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
    let mut acc = pos.psqt_accumulator();

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
    let pos = Position::from_fen("1k2r3/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling()), "Should not be able to castle out of check");

    let pos = Position::from_fen("1k3r2/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling() && m.to().0 == 7), "Should not castle through check on f1");

    let pos = Position::from_fen("1k4r1/8/8/8/8/8/8/R3K2R w KQ - 0 1");
    let moves = gen_legal_moves(&pos);
    assert!(!moves.iter().any(|m| m.is_castling() && m.to().0 == 7), "Should not castle into check on g1");
}

fn find_uci_move(pos: &Position, uci: &str) -> Move {
    let moves = gen_legal_moves(pos);
    for mv in &moves {
        if mv.to_uci(pos.frc_notation) == uci {
            return *mv;
        }
    }
    panic!("Move {uci} not found in position {}", pos.to_fen());
}
