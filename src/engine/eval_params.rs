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
            (mg_mob_open,   Vec4,   mobility_open_offset,   0),
            (eg_mob_open,   Vec4,   mobility_open_offset,   4),
            (mg_mob_closed, Vec4,   mobility_closed_offset, 0),
            (eg_mob_closed, Vec4,   mobility_closed_offset, 4),
            (w_shield,      Scalar, king_safety_offset,     0),
            (w_ortho,       Scalar, king_safety_offset,     1),
            (w_diag,        Scalar, king_safety_offset,     2),
            (atk_weights,   Array6, attacker_offset,        0),
            (w_xray_ortho,  Scalar, xray_offset,            0),
            (w_bp_mg,       Scalar, bishop_pair_offset,     0),
            (w_bp_eg,       Scalar, bishop_pair_offset,     1),
            (w_rook_open_mg, Scalar, rook_open_offset,      0),
            (w_rook_open_eg, Scalar, rook_open_offset,      1),
            (passed_mg,     Array6, passed_mg_offset,       0),
            (passed_eg,     Array6, passed_eg_offset,       0)
        }
    };
}

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
        S( -45,  245),  S( -15,  223),  S(  37,  187),  S(  74,  138),
        S(   3,   96),  S(  26,   95),  S(  77,   31),  S(  66,  -10),
        S( -13,   74),  S(  12,   70),  S(  14,   39),  S(  30,   20),
        S( -21,   56),  S(  -2,   65),  S(   2,   42),  S(  21,   35),
        S( -17,   51),  S(  12,   59),  S(  -5,   41),  S(   7,   47),
        S( -24,   53),  S(   9,   59),  S(  -5,   47),  S(  -8,   51),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-128,   25),  S(-106,  119),  S( -34,  132),  S(  29,  134),
        S(  32,  115),  S(  45,  144),  S( 100,  144),  S(  78,  151),
        S(  60,  125),  S(  87,  152),  S( 111,  171),  S( 127,  170),
        S(  63,  149),  S(  56,  178),  S( 100,  189),  S(  93,  199),
        S(  33,  155),  S(  69,  163),  S(  65,  190),  S(  77,  201),
        S(  14,  139),  S(  45,  161),  S(  53,  167),  S(  70,  190),
        S(  22,  141),  S(  11,  146),  S(  34,  160),  S(  47,  168),
        S( -44,  119),  S(   2,  122),  S(  11,  135),  S(   8,  145),
    ],

    BISHOP = [
        S(  11,  131),  S(  -1,  147),  S( -13,  138),  S( -45,  150),
        S(  43,  116),  S(  53,  143),  S(  62,  139),  S(  55,  138),
        S(  77,  140),  S(  88,  142),  S(  88,  152),  S(  89,  142),
        S(  48,  138),  S(  64,  154),  S(  80,  156),  S(  94,  167),
        S(  53,  124),  S(  52,  155),  S(  63,  161),  S(  80,  165),
        S(  58,  129),  S(  66,  141),  S(  64,  158),  S(  68,  161),
        S(  67,  119),  S(  72,  135),  S(  82,  123),  S(  57,  148),
        S(  43,  100),  S(  71,  122),  S(  37,  122),  S(  32,  132),
    ],

    ROOK = [
        S( -30,  220),  S( -47,  228),  S( -47,  235),  S( -55,  230),
        S( -29,  215),  S( -32,  226),  S(   0,  223),  S(   1,  218),
        S( -26,  208),  S(  16,  205),  S(   5,  204),  S(   8,  201),
        S( -27,  209),  S(  -7,  202),  S(  -8,  206),  S( -11,  203),
        S( -47,  207),  S( -34,  207),  S( -39,  211),  S( -23,  208),
        S( -43,  198),  S( -16,  188),  S( -30,  196),  S( -26,  201),
        S( -58,  202),  S( -34,  196),  S( -29,  196),  S( -30,  196),
        S( -31,  212),  S( -31,  202),  S( -38,  206),  S( -21,  192),
    ],

    QUEEN = [
        S( -11,  547),  S(  -2,  557),  S( -23,  609),  S( -11,  598),
        S(  34,  542),  S( -25,  593),  S(  -7,  618),  S( -41,  659),
        S(  48,  542),  S(  47,  556),  S(   5,  621),  S(  18,  625),
        S(  43,  559),  S(  19,  599),  S(  16,  607),  S(   1,  632),
        S(  32,  569),  S(  33,  590),  S(  15,  601),  S(  16,  614),
        S(  35,  539),  S(  41,  562),  S(  39,  574),  S(  30,  583),
        S(  47,  501),  S(  44,  502),  S(  53,  512),  S(  48,  546),
        S(  20,  520),  S(  16,  516),  S(  19,  510),  S(  40,  523),
    ],

    KING = [
        S( 101, -166),  S(  28,  -60),  S(  38,  -42),  S( -59,   -9),
        S( -96,   -8),  S(  11,   40),  S( -17,   55),  S(  63,   36),
        S(-110,   12),  S(  32,   51),  S(  42,   64),  S(  -4,   77),
        S(-148,   11),  S( -63,   55),  S( -59,   79),  S(-101,   94),
        S(-174,    5),  S( -83,   35),  S( -72,   64),  S(-110,   90),
        S( -85,  -17),  S( -26,    7),  S( -62,   41),  S( -60,   57),
        S(  19,  -54),  S(  22,  -23),  S( -15,    8),  S( -37,   23),
        S(  25, -117),  S(  39,  -70),  S(  15,  -41),  S(  21,  -49),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  91,  104), // Pawn
         S( 335,  365), // Knight
         S( 333,  399), // Bishop
         S( 519,  747), // Rook
         S(1043, 1239), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(5), V(4)],
    EG_MOBILITY_OPEN = [
        V(-5), V(-7), V(-5), V(-13)],
    MG_MOBILITY_CLOSED = [
        V(1), V(8), V(-20), V(-2)],
    EG_MOBILITY_CLOSED = [
        V(22), V(7), V(38), V(-23)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(160), V(260), V(439), V(518), V(568)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(16), V(12), V(8)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(11)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(32), V(84)], // [MG, EG]
    ROOK_OPEN_WEIGHTS   = [V(39), V(3)], // [MG, EG]
    PASSED_PAWN_MG      = [V(-17), V(-26), V(-22), V(7), V(1), V(43)], // by relative rank 1-6
    PASSED_PAWN_EG      = [V(1), V(8), V(41), V(80), V(178), V(112)], // by relative rank 1-6
}
