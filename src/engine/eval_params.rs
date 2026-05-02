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
        S( 160,  124),  S( 198,  117),  S( 204,  108),  S( 248,   80),  
        S(  47,   92),  S(  79,   88),  S(  97,   64),  S( 106,   64),  
        S(   3,   49),  S(  34,   39),  S(  32,   18),  S(  69,   11),  
        S(  -1,   31),  S(  35,   28),  S(  34,    7),  S(  54,   11),  
        S(   9,   22),  S(  48,   14),  S(  22,    9),  S(  34,    9),  
        S(   1,   27),  S(  39,   23),  S(  27,   17),  S(   1,   29),  
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  
    ],
    KNIGHT = [
        S(  71,  -86),  S(  93,  -31),  S( 193,  -39),  S( 281,  -56),  
        S( 233,  -55),  S( 229,  -24),  S( 273,  -31),  S( 288,  -38),  
        S( 202,  -33),  S( 267,  -29),  S( 287,  -18),  S( 345,  -35),  
        S( 242,  -36),  S( 241,   -9),  S( 290,  -13),  S( 266,   12),  
        S( 209,  -43),  S( 238,  -19),  S( 249,  -11),  S( 261,   -4),  
        S( 180,  -52),  S( 224,  -33),  S( 231,  -33),  S( 250,  -14),  
        S( 186,  -62),  S( 191,  -39),  S( 217,  -45),  S( 225,  -30),  
        S( 112,  -69),  S( 193,  -76),  S( 178,  -48),  S( 199,  -50),  
    ],
    BISHOP = [
        S( 216,   -5),  S( 245,   -7),  S( 215,   -5),  S( 226,   -8),  
        S( 240,   -8),  S( 263,    8),  S( 279,   -1),  S( 274,   -4),  
        S( 283,   -6),  S( 290,    9),  S( 306,    5),  S( 314,   -9),  
        S( 274,   -3),  S( 274,    9),  S( 293,    2),  S( 323,   13),  
        S( 262,  -10),  S( 283,   -2),  S( 276,   12),  S( 304,   13),  
        S( 256,  -12),  S( 287,  -16),  S( 283,    7),  S( 278,   14),  
        S( 267,  -41),  S( 289,  -17),  S( 293,  -21),  S( 273,    1),  
        S( 239,  -34),  S( 264,  -20),  S( 239,  -15),  S( 247,   -6),  
    ],
    ROOK = [
        S( 283,    0),  S( 294,   -5),  S( 303,   -4),  S( 290,   -3),  
        S( 298,   -7),  S( 318,   -5),  S( 345,  -13),  S( 363,  -18),  
        S( 265,    0),  S( 310,  -12),  S( 319,  -14),  S( 350,  -27),  
        S( 256,    4),  S( 271,   -5),  S( 283,   -2),  S( 322,  -18),  
        S( 221,   -2),  S( 241,   -2),  S( 248,    0),  S( 287,  -11),  
        S( 222,   -5),  S( 250,  -20),  S( 250,  -10),  S( 261,  -14),  
        S( 204,  -14),  S( 245,  -30),  S( 249,  -24),  S( 249,  -19),  
        S( 249,  -17),  S( 247,  -22),  S( 249,  -21),  S( 275,  -35),  
    ],
    QUEEN = [
        S( 196,  269),  S( 191,  294),  S( 179,  293),  S( 215,  274),  
        S( 213,  268),  S( 161,  306),  S( 202,  287),  S( 200,  300),  
        S( 224,  253),  S( 230,  266),  S( 215,  293),  S( 241,  284),  
        S( 239,  234),  S( 202,  301),  S( 224,  279),  S( 212,  296),  
        S( 227,  260),  S( 241,  247),  S( 221,  282),  S( 226,  287),  
        S( 232,  203),  S( 250,  224),  S( 240,  243),  S( 232,  253),  
        S( 247,  154),  S( 253,  148),  S( 277,  126),  S( 254,  191),  
        S( 243,  123),  S( 230,  142),  S( 222,  157),  S( 256,  147),  
    ],
    KING = [
        S(  33, -161),  S( -22,  -58),  S(  22,  -34),  S( -47,  -18),  
        S( 116,  -82),  S(  88,    1),  S(  65,    8),  S(  27,    5),  
        S(   6,  -29),  S(   4,   30),  S( -14,   40),  S( -55,   45),  
        S( -21,  -17),  S( -33,   34),  S( -24,   43),  S( -72,   56),  
        S( -93,  -13),  S( -65,   18),  S( -61,   36),  S( -51,   48),  
        S(-125,   -7),  S( -72,    1),  S( -76,   27),  S( -80,   43),  
        S( -49,  -37),  S( -47,  -13),  S( -82,   16),  S( -92,   24),  
        S( -46,  -87),  S( -19,  -48),  S( -75,  -14),  S( -59,  -27),  
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
        V(8), V(-2), V(6), V(-5)],
    EG_MOBILITY_OPEN = [
        V(-8), V(-10), V(3), V(-10)],
    MG_MOBILITY_CLOSED = [
        V(-2), V(1), V(-16), V(7)],
    EG_MOBILITY_CLOSED = [
        V(28), V(16), V(32), V(-28)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(80), V(170), V(204), V(145), V(113)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(15), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(26), V(75)], // [MG, EG]
}
