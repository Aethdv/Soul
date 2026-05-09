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
            (w_bp_eg,       Scalar, bishop_pair_offset,     1)
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

    let psqt_offset = 0;
    let material_offset = psqt_offset + psqt_len;
    let mobility_open_offset = material_offset + material_len;
    let mobility_closed_offset = mobility_open_offset + mobility_len;
    let weight_offset = mobility_closed_offset + mobility_len;
    let attacker_offset = weight_offset + phase_len;
    let king_safety_offset = attacker_offset + attacker_len;
    let xray_offset = king_safety_offset + safety_len;
    let bishop_pair_offset = xray_offset + xray_len;

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
        S(  -7,  277),  S(  14,  263),  S(  61,  231),  S(  94,  189),
        S( -31,  168),  S(  -8,  175),  S(  38,  117),  S(  35,  114),
        S( -32,   73),  S(  -9,   62),  S(  -7,   39),  S(   8,   27),
        S( -39,   42),  S( -21,   47),  S( -15,   25),  S(   1,   22),
        S( -34,   37),  S(  -7,   41),  S( -21,   26),  S(  -8,   33),
        S( -39,   39),  S( -10,   42),  S( -19,   33),  S( -21,   43),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-127,   31),  S( -97,  119),  S( -32,  121),  S(  18,  129),
        S(  27,  104),  S(  36,  138),  S(  85,  132),  S(  68,  141),
        S(  45,  127),  S(  66,  147),  S(  89,  161),  S( 105,  159),
        S(  49,  140),  S(  42,  166),  S(  81,  177),  S(  74,  186),
        S(  22,  145),  S(  53,  154),  S(  49,  178),  S(  61,  186),
        S(   6,  130),  S(  34,  150),  S(  40,  153),  S(  55,  175),
        S(  13,  134),  S(   5,  136),  S(  23,  149),  S(  36,  156),
        S( -47,  115),  S(  -4,  116),  S(   5,  124),  S(   2,  137),
    ],

    BISHOP = [
        S(  13,  144),  S(  10,  151),  S(   9,  149),  S( -38,  163),
        S(  47,  126),  S(  60,  152),  S(  66,  148),  S(  57,  149),
        S(  78,  152),  S(  84,  153),  S(  85,  162),  S(  88,  154),
        S(  52,  149),  S(  66,  163),  S(  78,  165),  S(  91,  173),
        S(  57,  136),  S(  53,  163),  S(  66,  168),  S(  80,  170),
        S(  62,  141),  S(  68,  150),  S(  66,  166),  S(  70,  167),
        S(  71,  130),  S(  75,  148),  S(  82,  136),  S(  61,  160),
        S(  52,  119),  S(  72,  136),  S(  44,  138),  S(  40,  146),
    ],

    ROOK = [
        S( -31,  257),  S( -54,  268),  S( -44,  273),  S( -54,  271),
        S( -31,  253),  S( -36,  266),  S(  -8,  266),  S(   0,  262),
        S( -40,  252),  S(  -1,  252),  S(  -8,  253),  S(   2,  249),
        S( -47,  255),  S( -32,  253),  S( -28,  256),  S( -26,  254),
        S( -72,  254),  S( -59,  255),  S( -58,  258),  S( -40,  257),
        S( -68,  245),  S( -43,  238),  S( -51,  245),  S( -44,  251),
        S( -81,  250),  S( -57,  244),  S( -50,  245),  S( -47,  246),
        S( -59,  258),  S( -56,  251),  S( -59,  252),  S( -40,  244),
    ],

    QUEEN = [
        S( -25,  569),  S( -14,  576),  S( -34,  621),  S( -17,  604),
        S(  15,  560),  S( -35,  606),  S( -22,  634),  S( -47,  665),
        S(  27,  566),  S(  28,  575),  S( -11,  639),  S(   3,  638),
        S(  26,  577),  S(   0,  615),  S(   0,  620),  S( -14,  643),
        S(  16,  585),  S(  18,  602),  S(   0,  617),  S(   2,  627),
        S(  18,  562),  S(  25,  580),  S(  24,  591),  S(  15,  602),
        S(  32,  524),  S(  27,  533),  S(  36,  540),  S(  31,  570),
        S(   7,  546),  S(   2,  545),  S(   6,  538),  S(  26,  546),
    ],

    KING = [
        S(  77, -147),  S(  25,  -50),  S(  11,  -40),  S( -66,   -8),
        S(-109,   -2),  S(  14,   32),  S( -30,   50),  S(  52,   30),
        S( -89,   11),  S(  23,   46),  S(  16,   60),  S(  -4,   68),
        S(-126,   10),  S( -63,   49),  S( -53,   68),  S( -90,   82),
        S(-139,   -1),  S( -78,   30),  S( -59,   52),  S( -92,   73),
        S( -77,  -19),  S( -23,    1),  S( -55,   30),  S( -53,   44),
        S(  17,  -53),  S(  18,  -27),  S( -13,    1),  S( -31,   14),
        S(  23, -106),  S(  34,  -67),  S(  13,  -41),  S(  22,  -50),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  88,  115), // Pawn
         S( 301,  301), // Knight
         S( 306,  320), // Bishop
         S( 494,  570), // Rook
         S( 922,  981), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(4), V(1)],
    EG_MOBILITY_OPEN = [
        V(-4), V(-5), V(-8), V(-10)],
    MG_MOBILITY_CLOSED = [
        V(4), V(10), V(-18), V(0)],
    EG_MOBILITY_CLOSED = [
        V(19), V(3), V(40), V(-32)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(120), V(220), V(390), V(457), V(496)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(14), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(7)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(23), V(68)], // [MG, EG]
}
