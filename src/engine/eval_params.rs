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
            (w_rook_open_eg, Scalar, rook_open_offset,      1)
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
        S(   6,  284),  S(  29,  269),  S(  78,  238),  S( 115,  190),
        S( -20,  168),  S(   2,  176),  S(  55,  113),  S(  50,  110),
        S( -25,   66),  S(   0,   56),  S(   4,   29),  S(  19,   18),
        S( -33,   34),  S( -13,   38),  S(  -8,   15),  S(  10,   12),
        S( -28,   28),  S(   1,   33),  S( -14,   16),  S(  -2,   24),
        S( -35,   30),  S(  -2,   34),  S( -13,   24),  S( -16,   34),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-126,   33),  S( -94,  114),  S( -32,  123),  S(  22,  128),
        S(  35,  102),  S(  43,  140),  S(  97,  138),  S(  74,  145),
        S(  55,  123),  S(  80,  148),  S( 104,  167),  S( 122,  163),
        S(  58,  144),  S(  53,  170),  S(  95,  182),  S(  88,  193),
        S(  31,  149),  S(  64,  157),  S(  61,  184),  S(  72,  194),
        S(  14,  134),  S(  43,  155),  S(  50,  160),  S(  66,  182),
        S(  21,  134),  S(  10,  139),  S(  32,  154),  S(  44,  162),
        S( -39,  104),  S(   1,  116),  S(  10,  129),  S(   7,  142),
    ],

    BISHOP = [
        S(  10,  126),  S(  -3,  135),  S(  -9,  134),  S( -39,  145),
        S(  44,  108),  S(  57,  138),  S(  64,  131),  S(  53,  134),
        S(  77,  136),  S(  84,  137),  S(  87,  147),  S(  88,  139),
        S(  48,  135),  S(  64,  149),  S(  79,  152),  S(  91,  161),
        S(  53,  118),  S(  50,  150),  S(  62,  156),  S(  80,  157),
        S(  58,  124),  S(  65,  136),  S(  63,  152),  S(  67,  153),
        S(  66,  112),  S(  71,  131),  S(  81,  118),  S(  56,  144),
        S(  45,   96),  S(  70,  119),  S(  38,  118),  S(  34,  127),
    ],

    ROOK = [
        S( -34,  214),  S( -57,  223),  S( -47,  228),  S( -54,  223),
        S( -33,  208),  S( -36,  220),  S( -11,  219),  S(  -8,  214),
        S( -32,  203),  S(  10,  202),  S(  -4,  202),  S(  -1,  199),
        S( -33,  204),  S( -14,  198),  S( -15,  204),  S( -19,  200),
        S( -51,  198),  S( -43,  201),  S( -44,  204),  S( -29,  201),
        S( -48,  191),  S( -22,  181),  S( -35,  190),  S( -32,  195),
        S( -63,  196),  S( -39,  190),  S( -36,  192),  S( -35,  191),
        S( -38,  206),  S( -37,  198),  S( -43,  202),  S( -28,  189),
    ],

    QUEEN = [
        S( -15,  535),  S(  -2,  544),  S( -17,  586),  S(   4,  572),
        S(  26,  526),  S( -28,  578),  S( -15,  609),  S( -41,  643),
        S(  39,  530),  S(  39,  542),  S(   1,  609),  S(  12,  615),
        S(  37,  541),  S(  13,  585),  S(  10,  593),  S(  -3,  617),
        S(  24,  554),  S(  28,  572),  S(   9,  588),  S(  10,  601),
        S(  28,  526),  S(  33,  549),  S(  32,  561),  S(  24,  565),
        S(  40,  489),  S(  37,  499),  S(  45,  503),  S(  40,  536),
        S(  14,  511),  S(  10,  505),  S(  12,  503),  S(  33,  514),
    ],

    KING = [
        S(  84, -151),  S(   9,  -53),  S(   9,  -35),  S( -36,  -17),
        S(-114,    2),  S(  -5,   41),  S( -46,   61),  S(  51,   36),
        S(-107,   18),  S(  19,   55),  S(  14,   70),  S( -15,   80),
        S(-138,   14),  S( -63,   56),  S( -59,   77),  S( -94,   91),
        S(-150,    2),  S( -82,   36),  S( -63,   60),  S( -98,   83),
        S( -78,  -19),  S( -20,    6),  S( -56,   36),  S( -55,   51),
        S(  20,  -53),  S(  22,  -25),  S( -11,    5),  S( -33,   20),
        S(  26, -111),  S(  40,  -68),  S(  16,  -42),  S(  22,  -48),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  88,  137), // Pawn
         S( 323,  335), // Knight
         S( 319,  367), // Bishop
         S( 508,  688), // Rook
         S(1013, 1133), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(5), V(3)],
    EG_MOBILITY_OPEN = [
        V(-4), V(-4), V(-7), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(2), V(10), V(-20), V(-1)],
    EG_MOBILITY_CLOSED = [
        V(19), V(3), V(43), V(-29)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(140), V(240), V(416), V(490), V(532)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(15), V(11), V(8)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(29), V(82)], // [MG, EG]
    ROOK_OPEN_WEIGHTS   = [V(36), V(5)], // [MG, EG]
}
