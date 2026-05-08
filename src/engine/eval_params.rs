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
        S( -42,  157),  S( -23,  134),  S(   0,  131),  S(  18,  110),
        S( -59,   93),  S( -44,   97),  S( -14,   61),  S( -17,   59),
        S( -62,   28),  S( -44,   22),  S( -43,    4),  S( -30,   -4),
        S( -68,    8),  S( -56,   12),  S( -49,   -4),  S( -36,   -7),
        S( -63,    4),  S( -44,    8),  S( -55,   -1),  S( -45,   -2),
        S( -69,    7),  S( -44,    8),  S( -52,    4),  S( -59,   13),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-196,  -66),  S(-172,  -10),  S(-147,  -10),  S(-117,   -3),
        S( -88,  -24),  S( -88,   -6),  S( -56,   -7),  S( -73,    1),
        S( -80,  -16),  S( -63,   -5),  S( -53,    7),  S( -38,    4),
        S( -77,   -7),  S( -85,   13),  S( -57,   17),  S( -61,   24),
        S( -95,   -2),  S( -75,    3),  S( -75,   14),  S( -68,   22),
        S(-105,  -12),  S( -85,   -2),  S( -80,   -2),  S( -72,   14),
        S(-100,  -12),  S(-106,   -7),  S( -92,    4),  S( -85,    1),
        S(-140,  -14),  S(-113,  -21),  S(-109,  -10),  S(-109,   -4),
    ],

    BISHOP = [
        S(-105,   14),  S( -91,   10),  S(-109,    6),  S(-135,   17),
        S( -72,   -3),  S( -64,    9),  S( -60,    6),  S( -70,    7),
        S( -54,    7),  S( -53,   10),  S( -49,   14),  S( -46,   10),
        S( -70,    6),  S( -59,   15),  S( -53,   15),  S( -44,   22),
        S( -63,    1),  S( -68,   12),  S( -59,   19),  S( -51,   21),
        S( -61,    2),  S( -55,    9),  S( -59,   17),  S( -54,   17),
        S( -56,   -2),  S( -48,    2),  S( -46,    0),  S( -61,   13),
        S( -68,   -3),  S( -54,    3),  S( -74,    0),  S( -78,    9),
    ],

    ROOK = [
        S(-219,   48),  S(-238,   58),  S(-237,   62),  S(-234,   57),
        S(-218,   47),  S(-220,   56),  S(-203,   55),  S(-201,   53),
        S(-226,   49),  S(-200,   48),  S(-208,   49),  S(-200,   45),
        S(-233,   52),  S(-222,   50),  S(-220,   53),  S(-218,   50),
        S(-248,   53),  S(-242,   55),  S(-240,   56),  S(-229,   52),
        S(-249,   49),  S(-228,   42),  S(-237,   49),  S(-232,   51),
        S(-257,   50),  S(-238,   44),  S(-234,   44),  S(-232,   44),
        S(-237,   45),  S(-236,   46),  S(-238,   49),  S(-224,   41),
    ],

    QUEEN = [
        S(-490,  141),  S(-489,  159),  S(-495,  181),  S(-489,  183),
        S(-469,  166),  S(-502,  192),  S(-492,  209),  S(-507,  228),
        S(-463,  179),  S(-467,  184),  S(-483,  214),  S(-482,  223),
        S(-466,  188),  S(-482,  212),  S(-485,  210),  S(-489,  223),
        S(-467,  182),  S(-473,  202),  S(-480,  205),  S(-480,  213),
        S(-468,  168),  S(-460,  173),  S(-466,  189),  S(-469,  187),
        S(-457,  139),  S(-458,  140),  S(-453,  143),  S(-459,  167),
        S(-476,  150),  S(-480,  155),  S(-476,  148),  S(-459,  144),
    ],

    KING = [
        S(  63,  -96),  S(  15,  -34),  S(  25,  -25),  S( -25,   -8),
        S( -57,   -4),  S(  27,   17),  S(  -2,   26),  S(  44,   14),
        S( -67,    9),  S(  25,   26),  S(  20,   34),  S(  -4,   40),
        S( -78,    4),  S( -34,   26),  S( -30,   40),  S( -61,   49),
        S( -89,   -2),  S( -47,   16),  S( -42,   31),  S( -66,   45),
        S( -49,  -16),  S( -17,   -4),  S( -37,   16),  S( -38,   26),
        S(  12,  -35),  S(  11,  -21),  S(  -8,   -4),  S( -21,    5),
        S(  17,  -66),  S(  26,  -44),  S(   7,  -28),  S(  14,  -33),
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
        V(2), V(-5), V(3), V(1)],
    EG_MOBILITY_OPEN = [
        V(-2), V(-3), V(-5), V(-6)],
    MG_MOBILITY_CLOSED = [
        V(2), V(8), V(-11), V(1)],
    EG_MOBILITY_CLOSED = [
        V(12), V(2), V(25), V(-21)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(60), V(130), V(235), V(275), V(290)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(9), V(5), V(3)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(4)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(24), V(67)], // [MG, EG]
}
