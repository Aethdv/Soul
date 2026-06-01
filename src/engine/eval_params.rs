//! Static evaluation parameters and weight arrays.
//!
//! Contains the constants and tunable structures that map piece-square placement
//! and spatial features to centipawn scores.

#![allow(non_snake_case)]

use crate::weave::Vi32x4;

#[derive(Debug, Clone)]
pub struct Tunable {
    pub name: String,
    pub value: f64,
    pub idx: usize,
    pub is_fixed: bool,
    pub freeze_resistant: bool,
}

#[derive(Clone, Copy)]
pub struct PhaseScore {
    pub mg: i32,
    pub eg: i32,
}

impl PhaseScore {
    pub const fn new(mg: i32, eg: i32) -> Self {
        Self { mg, eg }
    }
}

/// S(mg, eg)  = Tune
/// CS(mg, eg) = Don't tune
/// V(val)     = Tune
/// CV(val)    = Don't tune
pub enum Param {
    S(i32, i32),
    CS(i32, i32),
    Val(i32),
    Const(i32),
}

const fn S(mg: i32, eg: i32) -> Param {
    Param::S(mg, eg)
}
const fn CS(mg: i32, eg: i32) -> Param {
    Param::CS(mg, eg)
}
const fn V(v: i32) -> Param {
    Param::Val(v)
}
const fn CV(v: i32) -> Param {
    Param::Const(v)
}

fn collect_params_from_arrays<const N: usize>(name: &str, arr: &[Param; N]) -> Vec<Tunable> {
    let mut params = Vec::new();
    let freeze_resistant = name.starts_with("ATTACKER_WEIGHTS");

    for (i, param) in arr.iter().enumerate() {
        let (val, is_fixed) = match param {
            Param::Val(v) => (*v, false),
            Param::Const(v) => (*v, true),
            // Phase-score pairs (S/CS) are mg,eg pairs consumed by the SIMD tapered
            // eval, not individual tunable scalars. Map to fixed zero.
            Param::S(..) | Param::CS(..) => (0, true),
        };
        params.push(Tunable { value: val as f64, name: format!("{name}[{i}]"), idx: i, is_fixed, freeze_resistant });
    }
    params
}

macro_rules! define_psqt_params {
    ($($name:ident = [$($val:expr),* $(,)?]),* $(,)?) => {
        paste::paste! {
            $(
                pub const $name: [PhaseScore; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, eg) | Param::CS(mg, eg) => PhaseScore::new(mg, eg),
                            _ => panic!("PSQT params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<MG_ $name>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, _) | Param::CS(mg, _) => mg,
                            _ => panic!("PSQT params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<EG_ $name>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(_, eg) | Param::CS(_, eg) => eg,
                            _ => panic!("PSQT params must be S or CS"),
                        }
                    ),*
                ];
            )*

            fn collect_psqt_params() -> Vec<Tunable> {
                let mut params = Vec::new();

                let names = [$(stringify!($name)),*];
                let expected = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"];
                assert_eq!(names.len(), expected.len(), "Must define exactly 6 PSQT arrays");

                for (actual, expected) in names.iter().zip(expected.iter()) {
                    assert_eq!(actual, expected, "PSQT macro order MUST exactly match PieceType integer values");
                }

                $(
                    for (i, param) in $name.iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.mg as f64,
                            name: format!("MG_{}[{i}]", stringify!($name)),
                            idx: 0,
                            is_fixed,
                            freeze_resistant: false,
                        });
                    }
                    for (i, param) in $name.iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.eg as f64,
                            name: format!("EG_{}[{i}]", stringify!($name)),
                            idx: 0,
                            is_fixed,
                            freeze_resistant: false,
                        });
                    }
                )*
                params
            }
        }
    };
}

macro_rules! define_simple_params {
    ($($name:ident = [$($val:expr),* $(,)?]),* $(,)?) => {
        paste::paste! {
            $(
                pub const $name: [PhaseScore; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, eg) | Param::CS(mg, eg) => PhaseScore::new(mg, eg),
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<MG_ $name>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, _) | Param::CS(mg, _) => mg,
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<EG_ $name>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(_, eg) | Param::CS(_, eg) => eg,
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];
            )*

            fn collect_simple_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $(
                    for (i, param) in $name.iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.mg as f64,
                            name: format!("MG_{}[{i}]", stringify!($name)),
                            idx: 0,
                            is_fixed,
                            freeze_resistant: false,
                        });
                    }
                    for (i, param) in $name.iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.eg as f64,
                            name: format!("EG_{}[{i}]", stringify!($name)),
                            idx: 0,
                            is_fixed,
                            freeze_resistant: false,
                        });
                    }
                )*
                params
            }
        }
    };
}

macro_rules! define_simd_params {
    ($($name:ident = [$($val:expr),*]),* $(,)?) => {
        $(#[allow(clippy::excessive_precision)]
          pub const $name: Vi32x4 = Vi32x4::new([
            $(
                match $val {
                    Param::Val(v) | Param::Const(v) => v,
                    _ => 0,
                }
            ),*
          ]);)*

        fn collect_simd_params() -> Vec<Tunable> {
            let mut params = Vec::new();
            $(
                let arr = [$($val),*];
                params.append(&mut collect_params_from_arrays(stringify!($name), &arr));
            )*
            params
        }
    };
}

macro_rules! define_weight_params {
    ($($name:ident = [$($val:expr),* $(,)?]),* $(,)?) => {
        $(pub const $name: [i32; { [$($val),*].len() }] = [
            $(
                match $val {
                    Param::Val(v) | Param::Const(v) => v,
                    _ => 0,
                }
            ),*
        ];)*

        fn collect_weight_params() -> Vec<Tunable> {
            let mut params = Vec::new();
            $(
                let arr = [$($val),*];
                params.append(&mut collect_params_from_arrays(stringify!($name), &arr));
            )*
            params
        }
    };
}

#[macro_export]
macro_rules! define_tunables {
    ($macro:ident) => {
        $macro! {
            (mg_mob_open,        Vec4,   mobility_open_offset,      0),
            (eg_mob_open,        Vec4,   mobility_open_offset,      4),
            (mg_mob_closed,      Vec4,   mobility_closed_offset,    0),
            (eg_mob_closed,      Vec4,   mobility_closed_offset,    4),
            (w_shield,           Scalar, king_safety_offset,        0),
            (w_ortho,            Scalar, king_safety_offset,        1),
            (w_diag,             Scalar, king_safety_offset,        2),
            (atk_weights,        Array6, attacker_offset,           0),
            (w_xray_ortho,       Scalar, xray_offset,               0),
            (w_bp_mg,            Scalar, bishop_pair_offset,        0),
            (w_bp_eg,            Scalar, bishop_pair_offset,        1),
            (w_rook_open_mg,     Scalar, rook_open_offset,          0),
            (w_rook_open_eg,     Scalar, rook_open_offset,          1),
            (passed_mg,          Array6, passed_mg_offset,          0),
            (passed_eg,          Array6, passed_eg_offset,          0),
            (enemy_king_dist_mg, Array6, enemy_king_dist_mg_offset, 0),
            (enemy_king_dist_eg, Array6, enemy_king_dist_eg_offset, 0)
        }
    };
}

/// Slot count each `EvalParams` field consumes in the dual-AD gradient vector.
#[rustfmt::skip]
macro_rules! slot_width {
    (Scalar) => (1);
    (Vec4)   => (4);
    (Array4) => (4);
    (Array6) => (6);
}

/// Sum the dual-AD footprint over the tunable list; 2 accumulator lanes (MG/EG)
/// plus one slot per scalar weight.
macro_rules! count_dual_slots {
    ($( ($name:ident, $ty:ident, $off:ident, $extra:expr) ),* $(,)?) => {
        2usize $( + slot_width!($ty) )*
    };
}

/// Total dual-AD inputs; the 2 accumulator lanes plus every tunable weight.
/// Drives `DUAL_N`, so the gradient array sizes itself as eval terms are added.
pub const DUAL_SLOTS: usize = crate::define_tunables!(count_dual_slots);

pub struct Layout {
    pub psqt_offset: usize,
    pub psqt_len: usize,
    pub material_offset: usize,
    pub material_len: usize,
    pub mobility_open_offset: usize,
    pub mobility_open_len: usize,
    pub mobility_closed_offset: usize,
    pub mobility_closed_len: usize,
    pub weight_offset: usize,
    pub weight_len: usize,
    pub attacker_offset: usize,
    pub attacker_len: usize,
    pub king_safety_offset: usize,
    pub king_safety_len: usize,
    pub xray_offset: usize,
    pub xray_len: usize,
    pub bishop_pair_offset: usize,
    pub bishop_pair_len: usize,
    pub rook_open_offset: usize,
    pub rook_open_len: usize,
    pub passed_mg_offset: usize,
    pub passed_mg_len: usize,
    pub passed_eg_offset: usize,
    pub passed_eg_len: usize,
    pub enemy_king_dist_mg_offset: usize,
    pub enemy_king_dist_mg_len: usize,
    pub enemy_king_dist_eg_offset: usize,
    pub enemy_king_dist_eg_len: usize,
}

pub const LAYOUT: Layout = calc_layout();

const fn calc_layout() -> Layout {
    let psqt_len = (PAWN.len() + KNIGHT.len() + BISHOP.len() + ROOK.len() + QUEEN.len() + KING.len()) * 2;
    let material_len = MATERIAL.len() * 2;
    let mobility_len = 4 * 2; // MG + EG
    let phase_len = PHASE_WEIGHTS.len();
    let attacker_len = ATTACKER_WEIGHTS.len();
    let safety_len = KING_SAFETY_WEIGHTS.len();
    let xray_len = XRAY_WEIGHTS.len();
    let bishop_pair_len = BISHOP_PAIR_WEIGHTS.len();
    let rook_open_len = ROOK_OPEN_WEIGHTS.len();
    let passed_mg_len = PASSED_PAWN_MG.len();
    let passed_eg_len = PASSED_PAWN_EG.len();
    let enemy_king_dist_mg_len = ENEMY_KING_DIST_MG.len();
    let enemy_king_dist_eg_len = ENEMY_KING_DIST_EG.len();

    let psqt_offset = 0;
    let material_offset = psqt_offset + psqt_len;
    let mobility_open_offset = material_offset + material_len;
    let mobility_closed_offset = mobility_open_offset + mobility_len;
    let weight_offset = mobility_closed_offset + mobility_len;
    let attacker_offset = weight_offset + phase_len;
    let king_safety_offset = attacker_offset + attacker_len;
    let xray_offset = king_safety_offset + safety_len;
    let bishop_pair_offset = xray_offset + xray_len;
    let rook_open_offset = bishop_pair_offset + bishop_pair_len;
    let passed_mg_offset = rook_open_offset + rook_open_len;
    let passed_eg_offset = passed_mg_offset + passed_mg_len;
    let enemy_king_dist_mg_offset = passed_eg_offset + passed_eg_len;
    let enemy_king_dist_eg_offset = enemy_king_dist_mg_offset + enemy_king_dist_mg_len;

    Layout {
        psqt_offset,
        psqt_len,
        material_offset,
        material_len,
        mobility_open_offset,
        mobility_open_len: mobility_len,
        mobility_closed_offset,
        mobility_closed_len: mobility_len,
        weight_offset,
        weight_len: phase_len,
        attacker_offset,
        attacker_len,
        king_safety_offset,
        king_safety_len: safety_len,
        xray_offset,
        xray_len,
        bishop_pair_offset,
        bishop_pair_len,
        rook_open_offset,
        rook_open_len,
        passed_mg_offset,
        passed_mg_len,
        passed_eg_offset,
        passed_eg_len,
        enemy_king_dist_mg_offset,
        enemy_king_dist_mg_len,
        enemy_king_dist_eg_offset,
        enemy_king_dist_eg_len,
    }
}

pub fn collect_parameters() -> Vec<Tunable> {
    let mut all = Vec::new();

    let psqts = collect_psqt_params();
    for mut p in psqts {
        p.idx = all.len();
        all.push(p);
    }

    let simples = collect_simple_params();
    for mut p in simples {
        p.idx = all.len();
        all.push(p);
    }

    let simds = collect_simd_params();
    for mut p in simds {
        p.idx = all.len();
        all.push(p);
    }

    let weights = collect_weight_params();
    for mut p in weights {
        if p.name.starts_with("PHASE_WEIGHTS") {
            assert!(p.is_fixed, "PHASE_WEIGHTS must be constant (CV) — tuning phase is not supported.");
        }
        p.idx = all.len();
        all.push(p);
    }
    all
}

define_psqt_params! {
    // Files A-D (mirrored to E-H) × 8 ranks
    PAWN = [
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
        S(  -4,  209),  S(  16,  203),  S(  11,  196),  S(  35,  159),
        S(  11,   80),  S(  26,   89),  S(  73,   34),  S(  53,    3),
        S( -11,   60),  S(  12,   62),  S(  13,   34),  S(  28,   18),
        S( -20,   44),  S(  -2,   59),  S(   0,   37),  S(  19,   31),
        S( -15,   39),  S(  13,   51),  S(  -7,   36),  S(   6,   42),
        S( -24,   41),  S(   9,   51),  S(  -7,   42),  S( -11,   48),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-132,   20),  S(-106,  122),  S( -28,  130),  S(  48,  132),
        S(  32,  117),  S(  47,  147),  S( 105,  146),  S(  82,  154),
        S(  61,  130),  S(  88,  157),  S( 117,  173),  S( 132,  176),
        S(  67,  151),  S(  59,  183),  S( 105,  195),  S(  98,  205),
        S(  35,  160),  S(  70,  169),  S(  69,  196),  S(  79,  210),
        S(  15,  145),  S(  48,  167),  S(  56,  174),  S(  73,  197),
        S(  24,  149),  S(  12,  151),  S(  36,  167),  S(  49,  175),
        S( -44,  118),  S(   2,  125),  S(  12,  139),  S(   8,  152),
    ],

    BISHOP = [
        S(   8,  134),  S(   5,  144),  S(  -8,  137),  S( -47,  153),
        S(  42,  116),  S(  59,  144),  S(  63,  142),  S(  58,  141),
        S(  79,  143),  S(  88,  145),  S(  89,  158),  S(  91,  147),
        S(  49,  141),  S(  68,  156),  S(  84,  159),  S(  96,  172),
        S(  55,  126),  S(  52,  158),  S(  66,  165),  S(  83,  170),
        S(  61,  131),  S(  68,  144),  S(  65,  162),  S(  70,  165),
        S(  69,  121),  S(  75,  137),  S(  84,  125),  S(  59,  151),
        S(  46,  100),  S(  73,  126),  S(  38,  124),  S(  34,  134),
    ],

    ROOK = [
        S( -21,  216),  S( -49,  228),  S( -44,  232),  S( -52,  227),
        S( -26,  213),  S( -31,  224),  S(   1,  223),  S(   7,  216),
        S( -24,  204),  S(  27,  199),  S(  10,  202),  S(  13,  198),
        S( -24,  205),  S(  -3,  200),  S(  -2,  203),  S(  -8,  202),
        S( -47,  206),  S( -30,  204),  S( -33,  207),  S( -19,  205),
        S( -41,  197),  S( -13,  185),  S( -26,  194),  S( -23,  201),
        S( -57,  203),  S( -32,  195),  S( -27,  197),  S( -26,  194),
        S( -29,  213),  S( -28,  202),  S( -35,  206),  S( -17,  191),
    ],

    QUEEN = [
        S( -13,  543),  S(  -4,  549),  S( -22,  599),  S( -16,  599),
        S(  36,  538),  S( -27,  591),  S(  -7,  617),  S( -36,  653),
        S(  48,  537),  S(  51,  547),  S(   3,  623),  S(  20,  624),
        S(  45,  553),  S(  19,  601),  S(  20,  602),  S(   2,  630),
        S(  35,  565),  S(  36,  587),  S(  18,  597),  S(  18,  612),
        S(  35,  539),  S(  43,  559),  S(  40,  575),  S(  33,  582),
        S(  48,  499),  S(  48,  498),  S(  56,  508),  S(  50,  546),
        S(  23,  513),  S(  17,  513),  S(  19,  512),  S(  42,  522),
    ],

    KING = [
        S(  72, -156),  S(  22,  -49),  S(  -6,  -33),  S(-114,    1),
        S(-126,    4),  S( -20,   44),  S( -50,   58),  S(  57,   28),
        S(-122,   17),  S(   4,   54),  S(  11,   65),  S( -29,   73),
        S(-162,   13),  S( -75,   53),  S( -71,   70),  S(-129,   87),
        S(-168,   -3),  S( -81,   29),  S( -73,   56),  S(-123,   79),
        S( -86,  -19),  S( -13,    3),  S( -52,   33),  S( -59,   48),
        S(  26,  -54),  S(  29,  -23),  S(  -7,    6),  S( -29,   19),
        S(  33, -117),  S(  48,  -69),  S(  22,  -39),  S(  29,  -48),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  95,  119), // Pawn
         S( 350,  384), // Knight
         S( 348,  424), // Bishop
         S( 538,  794), // Rook
         S(1094, 1325), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-6), V(6), V(2)],
    EG_MOBILITY_OPEN = [
        V(-5), V(-8), V(-4), V(-12)],
    MG_MOBILITY_CLOSED = [
        V(3), V(9), V(-22), V(0)],
    EG_MOBILITY_CLOSED = [
        V(22), V(7), V(40), V(-27)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(170), V(280), V(490), V(592), V(653)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(18), V(12), V(8)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(12)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(33), V(85)], // [MG, EG]
    ROOK_OPEN_WEIGHTS   = [V(40), V(2)], // [MG, EG]
    PASSED_PAWN_MG      = [V(-39), V(-58), V(-61), V(-35), V(-29), V(46)], // by relative rank 1-6
    PASSED_PAWN_EG      = [V(-50), V(-31), V(17), V(70), V(170), V(101)], // by relative rank 1-6
    ENEMY_KING_DIST_MG  = [V(-97), V(46), V(37), V(30), V(25), V(14)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG  = [V(-44), V(-6), V(31), V(46), V(56), V(65)], // enemy king→passer dist, 7 clamps to 6
}
