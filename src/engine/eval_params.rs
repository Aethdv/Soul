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
        S( -21,  287),  S(   3,  274),  S(  50,  244),  S(  82,  205),
        S( -43,  181),  S( -21,  188),  S(  26,  133),  S(  22,  128),
        S( -45,   87),  S( -22,   77),  S( -18,   53),  S(  -4,   42),
        S( -51,   57),  S( -33,   61),  S( -28,   39),  S( -11,   37),
        S( -46,   51),  S( -20,   56),  S( -33,   40),  S( -20,   48),
        S( -52,   53),  S( -22,   57),  S( -31,   47),  S( -33,   58),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-120,   35),  S(-102,  122),  S( -28,  121),  S(  27,  123),
        S(  31,  102),  S(  35,  136),  S(  83,  134),  S(  66,  141),
        S(  47,  121),  S(  66,  145),  S(  89,  158),  S( 104,  159),
        S(  48,  140),  S(  42,  166),  S(  79,  175),  S(  73,  185),
        S(  23,  143),  S(  54,  150),  S(  50,  175),  S(  61,  185),
        S(   7,  128),  S(  34,  149),  S(  40,  153),  S(  54,  174),
        S(  15,  131),  S(   4,  135),  S(  25,  147),  S(  36,  155),
        S( -45,  109),  S(  -3,  114),  S(   6,  125),  S(   2,  137),
    ],

    BISHOP = [
        S(   4,  151),  S(  -6,  159),  S( -16,  158),  S( -43,  169),
        S(  28,  136),  S(  41,  162),  S(  49,  158),  S(  39,  160),
        S(  60,  161),  S(  67,  163),  S(  65,  174),  S(  68,  165),
        S(  34,  159),  S(  48,  172),  S(  61,  176),  S(  72,  184),
        S(  39,  144),  S(  36,  173),  S(  47,  179),  S(  63,  181),
        S(  43,  151),  S(  50,  161),  S(  47,  176),  S(  52,  177),
        S(  51,  141),  S(  56,  158),  S(  64,  146),  S(  42,  169),
        S(  31,  126),  S(  54,  145),  S(  25,  146),  S(  22,  153),
    ],

    ROOK = [
        S( -38,  325),  S( -58,  335),  S( -50,  340),  S( -62,  339),
        S( -39,  321),  S( -43,  333),  S( -14,  333),  S( -10,  330),
        S( -48,  320),  S(  -9,  320),  S( -19,  322),  S(  -7,  317),
        S( -56,  323),  S( -39,  319),  S( -37,  324),  S( -34,  322),
        S( -79,  321),  S( -66,  322),  S( -66,  325),  S( -48,  323),
        S( -76,  313),  S( -53,  305),  S( -59,  312),  S( -53,  318),
        S( -89,  318),  S( -67,  312),  S( -59,  312),  S( -56,  314),
        S( -67,  324),  S( -64,  318),  S( -67,  320),  S( -49,  311),
    ],

    QUEEN = [
        S(  14,  625),  S(   5,  646),  S(  -3,  685),  S(  16,  670),
        S(  44,  629),  S(  -3,  669),  S(   9,  699),  S( -14,  730),
        S(  58,  627),  S(  57,  642),  S(  21,  702),  S(  34,  704),
        S(  57,  638),  S(  32,  679),  S(  30,  685),  S(  18,  704),
        S(  46,  651),  S(  47,  669),  S(  31,  680),  S(  33,  691),
        S(  50,  625),  S(  55,  644),  S(  53,  656),  S(  46,  663),
        S(  60,  596),  S(  57,  599),  S(  67,  603),  S(  62,  633),
        S(  35,  615),  S(  32,  610),  S(  35,  606),  S(  56,  611),
    ],

    KING = [
        S(  87, -152),  S(  19,  -53),  S(  21,  -44),  S( -61,  -17),
        S( -95,  -10),  S(   3,   31),  S( -35,   47),  S(  64,   22),
        S( -92,    8),  S(  20,   41),  S(  29,   54),  S(  -3,   62),
        S(-124,    5),  S( -56,   44),  S( -54,   63),  S( -88,   76),
        S(-139,   -5),  S( -74,   24),  S( -58,   47),  S( -86,   67),
        S( -72,  -24),  S( -19,   -4),  S( -53,   25),  S( -52,   39),
        S(  19,  -57),  S(  19,  -31),  S( -12,   -4),  S( -30,   10),
        S(  24, -109),  S(  36,  -71),  S(  15,  -46),  S(  24,  -53),
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
        V(-4), V(-4), V(-9), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(2), V(10), V(-19), V(0)],
    EG_MOBILITY_CLOSED = [
        V(18), V(3), V(39), V(-34)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(120), V(220), V(385), V(450), V(493)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(14), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(9)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(26), V(74)], // [MG, EG]
}
