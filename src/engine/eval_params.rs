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
        S( 157,  123),  S( 196,  117),  S( 198,  107),  S( 238,   79),
        S(  45,   92),  S(  80,   87),  S(  98,   59),  S( 106,   62),
        S(   4,   47),  S(  34,   38),  S(  31,   17),  S(  69,   10),
        S(  -1,   29),  S(  35,   28),  S(  35,    6),  S(  54,   10),
        S(   9,   20),  S(  48,   13),  S(  23,    8),  S(  34,    9),
        S(   1,   26),  S(  39,   23),  S(  27,   16),  S(   2,   30),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],
    KNIGHT = [
        S(  55,  -86),  S( -24,   -9),  S( 200,  -37),  S( 271,  -56),
        S( 235,  -58),  S( 228,  -25),  S( 270,  -29),  S( 284,  -37),
        S( 201,  -32),  S( 266,  -29),  S( 282,  -18),  S( 339,  -33),
        S( 240,  -38),  S( 238,  -10),  S( 288,  -13),  S( 263,   12),
        S( 206,  -43),  S( 234,  -17),  S( 246,   -9),  S( 259,   -6),
        S( 176,  -49),  S( 221,  -32),  S( 229,  -34),  S( 248,  -16),
        S( 185,  -58),  S( 189,  -36),  S( 214,  -44),  S( 222,  -30),
        S( 109,  -72),  S( 191,  -79),  S( 178,  -51),  S( 199,  -54),
    ],
    BISHOP = [
        S( 206,   -5),  S( 241,   -9),  S( 213,   -8),  S( 226,   -9),
        S( 237,  -12),  S( 258,    5),  S( 280,    0),  S( 269,   -7),
        S( 281,   -5),  S( 284,   10),  S( 305,    5),  S( 313,  -11),
        S( 270,   -6),  S( 271,   11),  S( 288,    2),  S( 321,   11),
        S( 260,  -13),  S( 280,   -4),  S( 273,   13),  S( 302,   10),
        S( 253,  -13),  S( 284,  -16),  S( 280,    8),  S( 275,   15),
        S( 264,  -41),  S( 287,  -16),  S( 291,  -23),  S( 270,    1),
        S( 240,  -36),  S( 260,  -19),  S( 236,  -17),  S( 245,   -7),
    ],
    ROOK = [
        S( 275,   -1),  S( 287,   -5),  S( 291,   -4),  S( 285,   -3),
        S( 296,   -9),  S( 316,   -6),  S( 339,  -15),  S( 358,  -19),
        S( 261,   -2),  S( 308,  -12),  S( 316,  -16),  S( 347,  -28),
        S( 254,    3),  S( 268,   -6),  S( 276,   -4),  S( 318,  -21),
        S( 216,   -3),  S( 239,   -2),  S( 243,    0),  S( 284,  -12),
        S( 217,   -7),  S( 248,  -21),  S( 246,  -11),  S( 256,  -13),
        S( 205,  -16),  S( 243,  -29),  S( 244,  -22),  S( 245,  -21),
        S( 246,  -18),  S( 244,  -23),  S( 246,  -23),  S( 272,  -37),
    ],
    QUEEN = [
        S( 191,  269),  S( 160,  316),  S( 144,  317),  S( 198,  285),
        S( 196,  276),  S( 147,  317),  S( 184,  289),  S( 181,  307),
        S( 210,  259),  S( 214,  270),  S( 195,  304),  S( 223,  289),
        S( 223,  236),  S( 186,  301),  S( 213,  279),  S( 203,  294),
        S( 211,  265),  S( 223,  255),  S( 206,  284),  S( 212,  292),
        S( 216,  211),  S( 235,  228),  S( 224,  249),  S( 216,  260),
        S( 230,  166),  S( 240,  146),  S( 261,  135),  S( 238,  201),
        S( 220,  143),  S( 218,  143),  S( 206,  161),  S( 240,  155),
    ],
    KING = [
        S(  35, -156),  S( -24,  -56),  S(  18,  -35),  S( -49,  -20),
        S( 114,  -79),  S(  79,    3),  S(  78,    6),  S(  24,    5),
        S(  16,  -25),  S(   0,   33),  S( -11,   41),  S( -51,   44),
        S(  -3,  -19),  S( -29,   33),  S( -24,   44),  S( -69,   56),
        S( -86,  -13),  S( -65,   20),  S( -55,   37),  S( -48,   49),
        S(-124,   -4),  S( -68,    3),  S( -70,   26),  S( -80,   46),
        S( -46,  -36),  S( -45,  -11),  S( -80,   18),  S( -92,   26),
        S( -43,  -84),  S( -16,  -47),  S( -73,  -14),  S( -57,  -25),
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
        V(-8), V(-11), V(3), V(-11)],
    MG_MOBILITY_CLOSED = [
        V(-2), V(1), V(-16), V(7)],
    EG_MOBILITY_CLOSED = [
        V(29), V(17), V(33), V(-27)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(80), V(170), V(200), V(143), V(110)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(14), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(11)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(26), V(75)], // [MG, EG]
}
