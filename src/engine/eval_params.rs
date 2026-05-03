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

                // ── Material params ──
                for (i, param) in MATERIAL.iter().enumerate() {
                    params.push(Tunable {
                        value: param.mg as f64,
                        name: format!("MG_MATERIAL[{i}]"),
                        idx: 0,
                        is_fixed: true,
                        freeze_resistant: false,
                    });
                }
                for (i, param) in MATERIAL.iter().enumerate() {
                    params.push(Tunable {
                        value: param.eg as f64,
                        name: format!("EG_MATERIAL[{i}]"),
                        idx: 0,
                        is_fixed: true,
                        freeze_resistant: false,
                    });
                }

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
        S( 163,  129),  S( 208,  116),  S( 216,  110),  S( 261,   78),  
        S(  47,   98),  S(  84,   95),  S( 105,   65),  S( 107,   69),  
        S(   6,   49),  S(  38,   38),  S(  35,   16),  S(  72,   10),  
        S(   2,   31),  S(  38,   29),  S(  37,    7),  S(  57,   11),  
        S(  11,   22),  S(  51,   15),  S(  26,   10),  S(  37,   10),  
        S(   3,   27),  S(  42,   24),  S(  30,   17),  S(   6,   30),  
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  
    ],
    KNIGHT = [
        S(  69,  -71),  S( -95,   17),  S( 159,  -15),  S( 255,  -36),  
        S( 252,  -42),  S( 247,  -17),  S( 289,  -19),  S( 299,  -28),  
        S( 221,  -17),  S( 283,  -17),  S( 303,   -9),  S( 363,  -25),  
        S( 255,  -23),  S( 257,    3),  S( 304,    5),  S( 284,   25),  
        S( 225,  -29),  S( 253,   -6),  S( 264,    5),  S( 278,   10),  
        S( 196,  -37),  S( 241,  -22),  S( 249,  -21),  S( 268,   -1),  
        S( 208,  -48),  S( 212,  -25),  S( 237,  -32),  S( 242,  -14),  
        S( 121,  -58),  S( 209,  -64),  S( 196,  -38),  S( 218,  -42),  
    ],
    BISHOP = [
        S( 222,    6),  S( 248,    6),  S( 219,    3),  S( 222,    7),  
        S( 262,   -1),  S( 270,   17),  S( 292,   11),  S( 275,    8),  
        S( 296,    7),  S( 298,   21),  S( 320,   17),  S( 320,    2),  
        S( 284,   10),  S( 285,   23),  S( 306,   15),  S( 339,   23),  
        S( 275,    0),  S( 293,   10),  S( 289,   23),  S( 318,   22),  
        S( 267,    2),  S( 300,   -5),  S( 295,   20),  S( 292,   25),  
        S( 279,  -32),  S( 302,   -5),  S( 307,  -10),  S( 286,   13),  
        S( 258,  -23),  S( 278,   -7),  S( 252,   -3),  S( 263,    5),  
    ],
    ROOK = [
        S( 293,   24),  S( 309,   22),  S( 305,   24),  S( 297,   27),  
        S( 319,   14),  S( 335,   19),  S( 362,   10),  S( 378,    6),  
        S( 286,   22),  S( 334,   11),  S( 335,   10),  S( 371,   -4),  
        S( 277,   28),  S( 291,   17),  S( 305,   18),  S( 341,    6),  
        S( 240,   22),  S( 262,   20),  S( 267,   21),  S( 312,   11),  
        S( 242,   17),  S( 272,    2),  S( 272,   10),  S( 284,    9),  
        S( 228,    7),  S( 268,   -7),  S( 270,    0),  S( 272,    2),  
        S( 270,   10),  S( 268,    3),  S( 272,    3),  S( 297,  -11),  
    ],
    QUEEN = [
        S( 175,  351),  S( 160,  383),  S( 137,  396),  S( 189,  360),  
        S( 203,  345),  S( 148,  386),  S( 185,  368),  S( 180,  383),  
        S( 221,  327),  S( 223,  343),  S( 203,  373),  S( 230,  359),  
        S( 234,  306),  S( 197,  374),  S( 215,  353),  S( 207,  368),  
        S( 220,  338),  S( 235,  322),  S( 215,  353),  S( 219,  363),  
        S( 228,  280),  S( 244,  302),  S( 234,  320),  S( 227,  328),  
        S( 238,  239),  S( 250,  224),  S( 269,  206),  S( 248,  270),  
        S( 226,  213),  S( 223,  224),  S( 215,  241),  S( 249,  240),  
    ],
    KING = [
        S(  74, -180),  S( -28,  -66),  S(  16,  -47),  S( -49,  -28),  
        S(-115,  -41),  S(  35,   -2),  S(  15,   10),  S(  22,   -3),  
        S( -17,  -28),  S(   7,   21),  S( -29,   34),  S( -81,   42),  
        S( -32,  -20),  S( -41,   28),  S( -38,   39),  S( -93,   53),  
        S(-105,  -14),  S( -73,   13),  S( -73,   33),  S( -72,   49),  
        S(-131,   -9),  S( -73,   -2),  S( -79,   21),  S( -88,   40),  
        S( -51,  -44),  S( -49,  -18),  S( -86,   10),  S( -91,   15),  
        S( -48,  -98),  S( -20,  -56),  S( -74,  -21),  S( -59,  -30),  
    ],
}

define_simple_params! {
    MATERIAL = [
         CS( 100,  100), // Pawn
         CS( 300,  300), // Knight
         CS( 300,  300), // Bishop
         CS( 500,  500), // Rook
         CS( 900,  900), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(7), V(0), V(4), V(-5)],
    EG_MOBILITY_OPEN = [
        V(-7), V(-10), V(2), V(-8)],
    MG_MOBILITY_CLOSED = [
        V(-1), V(-1), V(-12), V(8)],
    EG_MOBILITY_CLOSED = [
        V(25), V(18), V(27), V(-26)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(80), V(150), V(155), V(85), V(46)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(13), V(7), V(6)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(9)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(26), V(75)], // [MG, EG]
}
