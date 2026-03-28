//! Static evaluation parameters and weight arrays.
//!
//! Contains the constants and tunable structures that map piece-square placement
//! and spatial features to centipawn scores.

#![allow(non_snake_case)]

use crate::weave::Vi32x4;

#[derive(Debug, Clone)]
pub struct Tunable {
    pub name:             String,
    pub value:            f64,
    pub idx:              usize,
    pub is_fixed:         bool,
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
        params.push(Tunable {
            value: val as f64,
            name: format!("{name}[{i}]"),
            idx: i,
            is_fixed,
            freeze_resistant,
        });
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
    pub psqt_offset:            usize,
    pub psqt_len:               usize,
    pub material_offset:        usize,
    pub material_len:           usize,
    pub mobility_open_offset:   usize,
    pub mobility_open_len:      usize,
    pub mobility_closed_offset: usize,
    pub mobility_closed_len:    usize,
    pub weight_offset:          usize,
    pub weight_len:             usize,
    pub attacker_offset:        usize,
    pub attacker_len:           usize,
    pub king_safety_offset:     usize,
    pub king_safety_len:        usize,
    pub xray_offset:            usize,
    pub xray_len:               usize,
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
            assert!(
                p.is_fixed,
                "PHASE_WEIGHTS must be constant (CV) — tuning phase is not supported."
            );
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
        S( -15,  277),  S(   1,  267),  S(  49,  236),  S(  79,  197),
        S( -44,  174),  S( -21,  182),  S(  24,  126),  S(  21,  122),
        S( -44,   83),  S( -22,   73),  S( -19,   49),  S(  -5,   39),
        S( -51,   54),  S( -33,   57),  S( -28,   36),  S( -12,   34),
        S( -46,   48),  S( -20,   52),  S( -34,   37),  S( -21,   44),
        S( -52,   50),  S( -22,   53),  S( -32,   44),  S( -34,   54),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],
    KNIGHT = [
        S(-123,   21),  S(-103,  108),  S( -40,  115),  S(  17,  118),
        S(  24,   95),  S(  36,  125),  S(  81,  121),  S(  65,  128),
        S(  43,  110),  S(  66,  132),  S(  88,  146),  S( 102,  145),
        S(  47,  127),  S(  41,  152),  S(  78,  162),  S(  73,  171),
        S(  20,  131),  S(  48,  138),  S(  48,  163),  S(  58,  170),
        S(   3,  116),  S(  29,  136),  S(  36,  139),  S(  49,  160),
        S(  11,  118),  S(   5,  125),  S(  23,  134),  S(  32,  142),
        S( -49,   97),  S(  -4,  103),  S(   4,  113),  S(  -1,  125),
    ],
    BISHOP = [
        S(  19,  147),  S(  10,  154),  S(   2,  155),  S( -23,  164),
        S(  46,  130),  S(  61,  155),  S(  67,  151),  S(  59,  153),
        S(  81,  156),  S(  85,  156),  S(  87,  165),  S(  89,  157),
        S(  57,  153),  S(  70,  166),  S(  79,  166),  S(  90,  175),
        S(  60,  141),  S(  58,  166),  S(  67,  172),  S(  79,  172),
        S(  64,  145),  S(  69,  154),  S(  67,  169),  S(  70,  170),
        S(  71,  135),  S(  75,  151),  S(  85,  139),  S(  64,  163),
        S(  52,  124),  S(  75,  140),  S(  47,  142),  S(  44,  150),
    ],
    ROOK = [
        S( -35,  297),  S( -60,  308),  S( -51,  313),  S( -57,  311),
        S( -39,  295),  S( -41,  307),  S( -14,  306),  S(  -6,  303),
        S( -47,  294),  S(  -9,  294),  S( -17,  294),  S(  -4,  289),
        S( -57,  297),  S( -41,  295),  S( -38,  298),  S( -30,  296),
        S( -79,  295),  S( -69,  297),  S( -69,  302),  S( -45,  298),
        S( -78,  288),  S( -52,  281),  S( -61,  288),  S( -49,  293),
        S( -99,  292),  S( -75,  287),  S( -67,  287),  S( -63,  289),
        S( -76,  299),  S( -72,  293),  S( -76,  296),  S( -56,  286),
    ],
    QUEEN = [
        S(  24,  577),  S(  33,  578),  S(  15,  625),  S(  35,  612),
        S(  59,  569),  S(  10,  612),  S(  28,  636),  S(   3,  671),
        S(  75,  571),  S(  74,  581),  S(  37,  639),  S(  55,  643),
        S(  75,  580),  S(  48,  621),  S(  48,  620),  S(  35,  645),
        S(  66,  588),  S(  67,  605),  S(  47,  619),  S(  50,  627),
        S(  65,  565),  S(  74,  579),  S(  69,  594),  S(  65,  602),
        S(  74,  528),  S(  68,  536),  S(  79,  540),  S(  73,  573),
        S(  52,  538),  S(  46,  543),  S(  48,  541),  S(  69,  549),
    ],
    KING = [
        S(  48, -141),  S(   4,  -48),  S(  45,  -47),  S( -65,  -13),
        S( -66,  -14),  S(   5,   28),  S( -24,   44),  S(  44,   23),
        S( -83,    6),  S(  27,   39),  S(  34,   51),  S(  -7,   61),
        S(-117,    4),  S( -60,   43),  S( -52,   62),  S( -86,   73),
        S(-142,   -4),  S( -76,   23),  S( -60,   47),  S( -93,   67),
        S( -80,  -22),  S( -32,   -5),  S( -64,   25),  S( -58,   38),
        S(   5,  -56),  S(   0,  -33),  S( -25,   -5),  S( -37,    8),
        S(  12, -108),  S(  17,  -71),  S(   2,  -45),  S(  17,  -54),
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
        V(5), V(-6), V(4), V(1)],
    EG_MOBILITY_OPEN = [
        V(-4), V(-4), V(-7), V(-8)],
    MG_MOBILITY_CLOSED = [
        V(3), V(9), V(-18), V(0)],
    EG_MOBILITY_CLOSED = [
        V(19), V(4), V(38), V(-33)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(108), V(204), V(369), V(434), V(478)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(15), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(9)], // [Ortho King]
}
