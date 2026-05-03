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
        S( 155,  121),  S( 198,  111),  S( 193,  103),  S( 234,   80),  
        S(  51,   80),  S(  79,   83),  S(  92,   60),  S( 105,   57),  
        S(   2,   47),  S(  31,   39),  S(  29,   17),  S(  67,   11),  
        S(  -3,   29),  S(  33,   27),  S(  32,    6),  S(  51,    9),  
        S(   7,   19),  S(  45,   12),  S(  20,    7),  S(  32,    7),  
        S(  -1,   25),  S(  36,   23),  S(  24,   16),  S(  -2,   28),  
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  
    ],
    KNIGHT = [
        S(  63,  -93),  S(  91,  -38),  S( 193,  -44),  S( 264,  -65),  
        S( 219,  -59),  S( 207,  -29),  S( 261,  -39),  S( 276,  -45),  
        S( 184,  -40),  S( 253,  -35),  S( 269,  -21),  S( 324,  -38),  
        S( 225,  -44),  S( 225,  -15),  S( 274,  -17),  S( 248,    6),  
        S( 193,  -52),  S( 221,  -28),  S( 232,  -15),  S( 245,  -12),  
        S( 162,  -56),  S( 208,  -39),  S( 215,  -40),  S( 235,  -22),  
        S( 171,  -66),  S( 179,  -46),  S( 200,  -49),  S( 208,  -38),  
        S(  91,  -72),  S( 178,  -87),  S( 162,  -58),  S( 186,  -59),  
    ],
    BISHOP = [
        S( 197,  -11),  S( 225,  -13),  S( 213,  -12),  S( 227,  -15),  
        S( 225,  -14),  S( 251,    2),  S( 268,   -5),  S( 256,   -8),  
        S( 270,  -13),  S( 276,    5),  S( 296,    2),  S( 303,  -15),  
        S( 262,  -12),  S( 259,    6),  S( 279,   -2),  S( 310,    7),  
        S( 247,  -13),  S( 268,   -8),  S( 261,   10),  S( 290,    8),  
        S( 240,  -15),  S( 272,  -21),  S( 267,    6),  S( 263,   10),  
        S( 251,  -47),  S( 275,  -20),  S( 278,  -26),  S( 258,   -3),  
        S( 227,  -34),  S( 248,  -24),  S( 224,  -19),  S( 232,   -8),  
    ],
    ROOK = [
        S( 260,  -14),  S( 278,  -17),  S( 285,  -20),  S( 272,  -15),  
        S( 281,  -20),  S( 296,  -18),  S( 320,  -25),  S( 345,  -32),  
        S( 247,  -12),  S( 283,  -22),  S( 294,  -26),  S( 327,  -39),  
        S( 235,   -9),  S( 250,  -18),  S( 255,  -15),  S( 298,  -30),  
        S( 197,  -13),  S( 219,  -13),  S( 224,  -12),  S( 264,  -22),  
        S( 199,  -20),  S( 228,  -33),  S( 226,  -21),  S( 236,  -24),  
        S( 186,  -27),  S( 220,  -38),  S( 223,  -33),  S( 225,  -31),  
        S( 226,  -32),  S( 225,  -36),  S( 226,  -34),  S( 252,  -48),  
    ],
    QUEEN = [
        S( 184,  225),  S( 173,  261),  S( 152,  266),  S( 189,  246),  
        S( 187,  237),  S( 144,  279),  S( 183,  252),  S( 179,  274),  
        S( 201,  218),  S( 205,  233),  S( 191,  261),  S( 222,  250),  
        S( 216,  194),  S( 179,  264),  S( 201,  242),  S( 192,  258),  
        S( 204,  225),  S( 215,  213),  S( 200,  243),  S( 205,  250),  
        S( 206,  178),  S( 226,  189),  S( 216,  208),  S( 208,  220),  
        S( 222,  121),  S( 232,  111),  S( 252,   99),  S( 230,  157),  
        S( 221,   96),  S( 213,  101),  S( 196,  122),  S( 232,  110),  
    ],
    KING = [
        S(  26, -149),  S( -15,  -53),  S(  31,  -31),  S( -30,  -23),  
        S( 177,  -89),  S(  87,    4),  S(  80,   12),  S(  36,    9),  
        S(   7,  -23),  S(   1,   36),  S(  -9,   44),  S( -48,   46),  
        S(  -6,  -15),  S( -27,   37),  S( -16,   46),  S( -63,   59),  
        S( -86,  -12),  S( -59,   22),  S( -54,   38),  S( -38,   52),  
        S(-121,   -4),  S( -71,    9),  S( -75,   32),  S( -75,   46),  
        S( -50,  -32),  S( -50,   -7),  S( -87,   22),  S( -96,   29),  
        S( -48,  -78),  S( -21,  -41),  S( -81,   -6),  S( -62,  -23),  
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
        V(9), V(-2), V(7), V(-6)],
    EG_MOBILITY_OPEN = [
        V(-8), V(-10), V(3), V(-14)],
    MG_MOBILITY_CLOSED = [
        V(-3), V(2), V(-18), V(7)],
    EG_MOBILITY_CLOSED = [
        V(30), V(15), V(37), V(-24)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(80), V(180), V(227), V(180), V(158)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(16), V(10), V(8)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(11)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(26), V(75)], // [MG, EG]
}
