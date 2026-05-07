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
        S( -23,  276),  S(   0,  263),  S(  40,  235),  S(  78,  192),
        S( -45,  172),  S( -23,  180),  S(  22,  124),  S(  19,  120),
        S( -47,   82),  S( -24,   72),  S( -22,   48),  S(  -7,   37),
        S( -53,   52),  S( -36,   56),  S( -30,   35),  S( -15,   33),
        S( -48,   47),  S( -23,   51),  S( -35,   36),  S( -23,   43),
        S( -54,   49),  S( -25,   52),  S( -34,   43),  S( -36,   53),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-130,   24),  S(-103,  109),  S( -46,  111),  S(   7,  113),
        S(  15,   93),  S(  23,  124),  S(  70,  121),  S(  54,  128),
        S(  33,  110),  S(  56,  128),  S(  76,  145),  S(  93,  143),
        S(  37,  126),  S(  31,  150),  S(  67,  161),  S(  62,  169),
        S(  12,  129),  S(  43,  137),  S(  39,  161),  S(  49,  168),
        S(  -3,  114),  S(  24,  134),  S(  29,  138),  S(  44,  158),
        S(   4,  115),  S(  -5,  122),  S(  14,  133),  S(  25,  140),
        S( -56,  100),  S( -13,  101),  S(  -5,  111),  S(  -8,  123),
    ],

    BISHOP = [
        S(  18,  140),  S(   7,  150),  S(  -2,  151),  S( -31,  162),
        S(  41,  129),  S(  53,  154),  S(  59,  149),  S(  51,  151),
        S(  72,  153),  S(  77,  155),  S(  79,  162),  S(  81,  157),
        S(  46,  151),  S(  60,  164),  S(  71,  167),  S(  84,  173),
        S(  52,  137),  S(  47,  164),  S(  60,  170),  S(  74,  172),
        S(  56,  143),  S(  62,  152),  S(  60,  167),  S(  64,  168),
        S(  64,  134),  S(  68,  150),  S(  76,  138),  S(  55,  161),
        S(  45,  121),  S(  66,  139),  S(  39,  141),  S(  34,  148),
    ],

    ROOK = [
        S( -56,  299),  S( -69,  306),  S( -64,  311),  S( -73,  309),
        S( -52,  293),  S( -55,  305),  S( -28,  305),  S( -23,  302),
        S( -61,  292),  S( -24,  292),  S( -31,  293),  S( -21,  290),
        S( -69,  296),  S( -54,  292),  S( -49,  297),  S( -46,  294),
        S( -91,  294),  S( -80,  295),  S( -80,  298),  S( -61,  297),
        S( -89,  285),  S( -64,  278),  S( -72,  286),  S( -65,  292),
        S(-102,  291),  S( -79,  286),  S( -71,  285),  S( -69,  286),
        S( -80,  297),  S( -77,  291),  S( -79,  292),  S( -62,  284),
    ],

    QUEEN = [
        S( -32,  589),  S( -23,  597),  S( -44,  644),  S( -21,  624),
        S(   7,  583),  S( -41,  624),  S( -31,  654),  S( -54,  683),
        S(  17,  588),  S(  19,  596),  S( -17,  656),  S(  -4,  655),
        S(  17,  596),  S(  -8,  636),  S(  -9,  639),  S( -21,  660),
        S(   7,  606),  S(   9,  621),  S(  -8,  636),  S(  -6,  647),
        S(   9,  582),  S(  15,  602),  S(  14,  612),  S(   6,  623),
        S(  22,  548),  S(  19,  555),  S(  26,  562),  S(  21,  592),
        S(  -1,  565),  S(  -6,  568),  S(  -3,  562),  S(  17,  567),
    ],

    KING = [
        S(  95, -149),  S(  22,  -54),  S(  21,  -39),  S( -54,  -12),
        S( -99,   -1),  S(   0,   31),  S( -21,   44),  S(  67,   22),
        S( -85,    8),  S(  24,   42),  S(  13,   57),  S( -11,   64),
        S(-130,    9),  S( -53,   44),  S( -52,   63),  S( -79,   76),
        S(-143,   -1),  S( -79,   26),  S( -56,   47),  S( -90,   68),
        S( -74,  -22),  S( -24,   -1),  S( -51,   26),  S( -52,   40),
        S(  15,  -54),  S(  16,  -29),  S( -13,   -2),  S( -31,   11),
        S(  21, -105),  S(  32,  -67),  S(  12,  -43),  S(  22,  -51),
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
        V(-4), V(-4), V(-7), V(-10)],
    MG_MOBILITY_CLOSED = [
        V(2), V(10), V(-17), V(0)],
    EG_MOBILITY_CLOSED = [
        V(18), V(3), V(38), V(-30)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(120), V(215), V(375), V(443), V(478)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(13), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(7)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(25), V(71)], // [MG, EG]
}
