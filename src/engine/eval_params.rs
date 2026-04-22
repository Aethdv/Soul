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
            // Field Name,  Type,   Offset in values array
            (mg_mob_open,   Vec4,   $crate::engine::eval_params::LAYOUT.mobility_open_offset),
            (eg_mob_open,   Vec4,   $crate::engine::eval_params::LAYOUT.mobility_open_offset + 4),
            (mg_mob_closed, Vec4,   $crate::engine::eval_params::LAYOUT.mobility_closed_offset),
            (eg_mob_closed, Vec4,   $crate::engine::eval_params::LAYOUT.mobility_closed_offset + 4),
            (w_shield,      Scalar, $crate::engine::eval_params::LAYOUT.king_safety_offset),
            (w_ortho,       Scalar, $crate::engine::eval_params::LAYOUT.king_safety_offset + 1),
            (w_diag,        Scalar, $crate::engine::eval_params::LAYOUT.king_safety_offset + 2),
            (atk_weights,   Array6, $crate::engine::eval_params::LAYOUT.attacker_offset),
            (w_xray_ortho,  Scalar, $crate::engine::eval_params::LAYOUT.xray_offset)
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

    let psqt_offset = 0;
    let material_offset = psqt_offset + psqt_len;
    let mobility_open_offset = material_offset + material_len;
    let mobility_closed_offset = mobility_open_offset + mobility_len;
    let weight_offset = mobility_closed_offset + mobility_len;
    let attacker_offset = weight_offset + phase_len;
    let king_safety_offset = attacker_offset + attacker_len;
    let xray_offset = king_safety_offset + safety_len;

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
        S( -22,  286),  S(   3,  271),  S(  42,  241),  S(  78,  199),
        S( -44,  178),  S( -22,  186),  S(  24,  129),  S(  21,  125),
        S( -46,   85),  S( -22,   75),  S( -19,   51),  S(  -5,   40),
        S( -51,   55),  S( -34,   59),  S( -28,   38),  S( -12,   35),
        S( -47,   50),  S( -20,   54),  S( -33,   39),  S( -21,   46),
        S( -52,   52),  S( -23,   55),  S( -32,   46),  S( -34,   56),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],
    KNIGHT = [
        S(-125,   23),  S( -97,  110),  S( -37,  117),  S(  20,  118),
        S(  28,   98),  S(  29,  130),  S(  80,  127),  S(  63,  134),
        S(  40,  117),  S(  64,  137),  S(  87,  153),  S( 101,  152),
        S(  45,  134),  S(  39,  158),  S(  77,  169),  S(  71,  178),
        S(  20,  138),  S(  51,  145),  S(  47,  170),  S(  58,  177),
        S(   3,  122),  S(  31,  142),  S(  38,  146),  S(  52,  167),
        S(  13,  124),  S(   3,  131),  S(  21,  141),  S(  33,  148),
        S( -47,  104),  S(  -5,  107),  S(   2,  118),  S(   1,  131),
    ],
    BISHOP = [
        S(  19,  152),  S(  18,  159),  S(   2,  159),  S( -30,  171),
        S(  49,  137),  S(  61,  164),  S(  66,  160),  S(  55,  163),
        S(  80,  162),  S(  87,  163),  S(  88,  173),  S(  90,  164),
        S(  54,  160),  S(  69,  173),  S(  83,  173),  S(  93,  182),
        S(  61,  146),  S(  56,  173),  S(  68,  178),  S(  83,  179),
        S(  66,  150),  S(  71,  160),  S(  69,  176),  S(  73,  178),
        S(  73,  142),  S(  78,  158),  S(  85,  147),  S(  64,  170),
        S(  53,  129),  S(  75,  146),  S(  47,  148),  S(  42,  157),
    ],
    ROOK = [
        S( -45,  313),  S( -62,  322),  S( -56,  327),  S( -61,  324),
        S( -43,  309),  S( -45,  320),  S( -17,  320),  S( -12,  317),
        S( -48,  307),  S( -13,  309),  S( -20,  308),  S(  -9,  305),
        S( -59,  311),  S( -45,  308),  S( -39,  312),  S( -36,  309),
        S( -81,  309),  S( -68,  308),  S( -69,  314),  S( -50,  312),
        S( -78,  301),  S( -55,  294),  S( -62,  301),  S( -54,  306),
        S( -92,  305),  S( -68,  301),  S( -61,  301),  S( -58,  302),
        S( -70,  313),  S( -66,  307),  S( -70,  308),  S( -51,  299),
    ],
    QUEEN = [
        S(  -4,  615),  S(  -5,  626),  S( -16,  669),  S(   9,  647),
        S(  35,  611),  S( -13,  650),  S(   0,  676),  S( -26,  710),
        S(  48,  611),  S(  46,  623),  S(  10,  680),  S(  24,  683),
        S(  46,  623),  S(  22,  658),  S(  20,  666),  S(   9,  685),
        S(  36,  633),  S(  38,  649),  S(  22,  659),  S(  22,  672),
        S(  38,  607),  S(  45,  626),  S(  44,  637),  S(  36,  645),
        S(  52,  572),  S(  48,  578),  S(  56,  586),  S(  51,  616),
        S(  28,  585),  S(  22,  591),  S(  26,  586),  S(  46,  593),
    ],
    KING = [
        S(  71, -152),  S(  16,  -57),  S(  27,  -45),  S( -58,  -17),
        S(-108,   -8),  S(  10,   27),  S( -35,   44),  S(  55,   22),
        S(-100,    7),  S(  23,   39),  S(  15,   53),  S( -17,   62),
        S(-138,    7),  S( -65,   42),  S( -62,   61),  S( -85,   73),
        S(-157,   -3),  S( -93,   25),  S( -69,   46),  S( -98,   67),
        S( -86,  -24),  S( -31,   -4),  S( -65,   24),  S( -60,   38),
        S(   5,  -57),  S(   7,  -32),  S( -22,   -5),  S( -41,    8),
        S(  12, -109),  S(  24,  -71),  S(   2,  -45),  S(  12,  -54),
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
        V(5), V(-7), V(4), V(1)],
    EG_MOBILITY_OPEN = [
        V(-4), V(-4), V(-8), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(2), V(10), V(-18), V(0)],
    EG_MOBILITY_CLOSED = [
        V(19), V(3), V(39), V(-31)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(120), V(220), V(387), V(450), V(487)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(14), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(9)], // [Ortho King]
}
