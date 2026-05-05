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
        S( -31,  214),  S( -25,  163),  S( -14,  137),  S( -40,  138),  
        S( -42,  122),  S( -40,  107),  S( -31,   76),  S( -20,   83),  
        S( -53,   64),  S( -24,   58),  S( -32,   27),  S( -15,   37),  
        S( -60,   37),  S( -41,   41),  S( -34,   25),  S( -20,   24),  
        S( -55,   31),  S( -28,   42),  S( -39,   27),  S( -32,   34),  
        S( -64,   32),  S( -26,   44),  S( -37,   35),  S( -48,   46),  
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  
    ],

    KNIGHT = [
        S(-166,   63),  S(-203,  136),  S(-189,  136),  S(-137,  138),  
        S( -63,  117),  S( -66,  128),  S( -19,  121),  S( -40,  130),  
        S( -57,  115),  S( -10,  117),  S( -41,  131),  S(   0,  131),  
        S( -28,  116),  S( -34,  134),  S( -25,  145),  S( -14,  136),  
        S( -38,  120),  S( -32,  127),  S( -21,  133),  S( -15,  143),  
        S( -41,  111),  S( -26,  123),  S( -18,  127),  S( -11,  137),  
        S( -41,   98),  S( -37,  111),  S( -26,  119),  S( -23,  123),  
        S( -69,   95),  S( -52,   99),  S( -45,  110),  S( -43,  113),  
    ],

    BISHOP = [
        S( -53,  148),  S( -58,  146),  S( -65,  150),  S( -76,  150),  
        S( -21,  138),  S(  -3,  137),  S(  -5,  136),  S( -36,  143),  
        S(  -6,  143),  S( -18,  142),  S( -18,  142),  S(  -2,  138),  
        S( -20,  137),  S(   0,  141),  S(  -7,  148),  S(   0,  146),  
        S(  -3,  130),  S( -11,  142),  S(   3,  144),  S(  -3,  149),  
        S(   6,  131),  S(  13,  135),  S(   7,  140),  S(  12,  143),  
        S(  18,  127),  S(  26,  128),  S(  29,  131),  S(  10,  140),  
        S(   8,  128),  S(  21,  131),  S(  -4,  134),  S(  -5,  137),  
    ],

    ROOK = [
        S(-194,  252),  S(-227,  265),  S(-222,  264),  S(-216,  260),  
        S(-178,  250),  S(-190,  260),  S(-176,  257),  S(-178,  257),  
        S(-204,  255),  S(-193,  258),  S(-195,  260),  S(-197,  258),  
        S(-195,  253),  S(-182,  248),  S(-196,  259),  S(-194,  258),  
        S(-189,  247),  S(-199,  253),  S(-197,  254),  S(-194,  253),  
        S(-189,  241),  S(-162,  233),  S(-178,  245),  S(-173,  237),  
        S(-191,  237),  S(-167,  229),  S(-170,  231),  S(-164,  230),  
        S(-168,  225),  S(-168,  224),  S(-167,  225),  S(-151,  222),  
    ],

    QUEEN = [
        S(-116,  539),  S(-141,  554),  S( -94,  518),  S(-140,  539),  
        S(-109,  506),  S(-130,  507),  S(-109,  508),  S(-142,  533),  
        S(-106,  508),  S(-123,  521),  S(-130,  528),  S(-124,  530),  
        S(-105,  532),  S(-117,  532),  S(-127,  540),  S(-124,  539),  
        S( -91,  495),  S(-108,  515),  S(-106,  532),  S(-111,  540),  
        S( -98,  519),  S( -80,  496),  S( -92,  533),  S( -91,  521),  
        S( -77,  477),  S( -68,  468),  S( -69,  485),  S( -75,  493),  
        S( -91,  480),  S(-102,  463),  S( -90,  469),  S( -74,  434),  
    ],

    KING = [
        S(  71,  -68),  S(   3,  -20),  S(  13,  -32),  S( -50,  -21),  
        S(-105,   -2),  S(   0,  -16),  S( -33,   -9),  S(  42,  -26),  
        S(-108,   13),  S(   7,   13),  S(  14,    4),  S( -25,   12),  
        S(-131,   16),  S(-114,   23),  S( -91,   24),  S( -77,   19),  
        S(-110,   -8),  S(-105,   11),  S(-111,   16),  S(-127,   23),  
        S( -65,  -28),  S( -72,  -18),  S( -98,    3),  S(-100,    6),  
        S(  -5,  -56),  S( -15,  -41),  S( -33,  -31),  S( -50,  -21),  
        S(  10,  -79),  S(  14,  -68),  S(  -7,  -49),  S(  -3,  -51),  
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
        V(2), V(-3), V(6), V(3)],
    EG_MOBILITY_OPEN = [
        V(0), V(-1), V(1), V(-5)],
    MG_MOBILITY_CLOSED = [
        V(5), V(5), V(-8), V(2)],
    EG_MOBILITY_CLOSED = [
        V(7), V(-4), V(18), V(-19)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(20), V(50), V(60), V(32), V(22)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(13), V(3), V(3)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(4)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(25), V(73)], // [MG, EG]
}
