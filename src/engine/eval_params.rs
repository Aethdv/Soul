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

            fn collect_simple_params() -> Vec<Tunable> {
                let mut params = Vec::new();
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
                params
            }
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

    let simples = collect_simple_params();
    for mut p in simples {
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
        S(  -4,  270),  S(  22,  256),  S(  61,  225),  S(  98,  183),
        S( -29,  161),  S(  -8,  169),  S(  39,  113),  S(  36,  107),
        S( -32,   66),  S(  -8,   57),  S(  -5,   32),  S(   9,   22),
        S( -38,   36),  S( -20,   41),  S( -14,   19),  S(   2,   17),
        S( -33,   31),  S(  -6,   35),  S( -20,   20),  S(  -7,   28),
        S( -38,   33),  S(  -9,   36),  S( -18,   27),  S( -20,   38),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-127,   40),  S(-110,  121),  S( -33,  124),  S(  23,  131),
        S(  30,  107),  S(  36,  140),  S(  86,  140),  S(  66,  143),
        S(  47,  125),  S(  70,  147),  S(  91,  164),  S( 107,  160),
        S(  50,  144),  S(  44,  168),  S(  81,  179),  S(  76,  187),
        S(  24,  148),  S(  55,  154),  S(  51,  180),  S(  63,  187),
        S(   8,  131),  S(  35,  153),  S(  42,  156),  S(  57,  176),
        S(  16,  133),  S(   7,  137),  S(  26,  150),  S(  38,  159),
        S( -42,  111),  S(  -1,  117),  S(   7,  128),  S(   4,  140),
    ],

    BISHOP = [
        S(   7,  132),  S(  -1,  141),  S(  -5,  139),  S( -44,  151),
        S(  39,  115),  S(  51,  142),  S(  58,  136),  S(  50,  138),
        S(  72,  139),  S(  78,  142),  S(  78,  152),  S(  80,  143),
        S(  46,  136),  S(  59,  150),  S(  72,  154),  S(  84,  161),
        S(  50,  122),  S(  47,  151),  S(  59,  157),  S(  73,  159),
        S(  55,  128),  S(  61,  138),  S(  59,  154),  S(  63,  155),
        S(  63,  119),  S(  68,  135),  S(  76,  123),  S(  54,  146),
        S(  43,  104),  S(  66,  123),  S(  36,  123),  S(  33,  132),
    ],

    ROOK = [
        S( -22,  231),  S( -43,  239),  S( -36,  246),  S( -41,  241),
        S( -22,  226),  S( -25,  238),  S(   4,  239),  S(   9,  234),
        S( -29,  225),  S(  10,  225),  S(   1,  226),  S(  10,  222),
        S( -40,  228),  S( -21,  223),  S( -18,  230),  S( -15,  226),
        S( -61,  225),  S( -48,  225),  S( -48,  229),  S( -30,  228),
        S( -59,  217),  S( -33,  209),  S( -41,  216),  S( -35,  223),
        S( -73,  222),  S( -47,  215),  S( -41,  217),  S( -37,  218),
        S( -50,  229),  S( -47,  223),  S( -50,  225),  S( -31,  216),
    ],

    QUEEN = [
        S( -19,  544),  S( -25,  561),  S( -35,  601),  S( -19,  587),
        S(  13,  545),  S( -35,  585),  S( -20,  613),  S( -47,  645),
        S(  29,  541),  S(  28,  555),  S(  -8,  614),  S(   5,  613),
        S(  28,  552),  S(   2,  596),  S(   2,  599),  S( -12,  622),
        S(  19,  562),  S(  18,  583),  S(   1,  594),  S(   5,  605),
        S(  21,  539),  S(  26,  560),  S(  24,  571),  S(  17,  578),
        S(  35,  501),  S(  30,  511),  S(  38,  518),  S(  33,  547),
        S(  11,  521),  S(   3,  525),  S(   7,  518),  S(  27,  525),
    ],

    KING = [
        S( 109, -160),  S(  16,  -51),  S(   9,  -40),  S( -53,  -13),
        S( -83,   -4),  S(  -4,   36),  S( -28,   50),  S(  64,   27),
        S(-108,   13),  S(  17,   47),  S(  20,   60),  S( -20,   70),
        S(-139,   12),  S( -68,   50),  S( -64,   69),  S( -84,   80),
        S(-147,    0),  S( -83,   30),  S( -66,   52),  S( -97,   74),
        S( -79,  -20),  S( -28,    2),  S( -61,   30),  S( -58,   45),
        S(  13,  -53),  S(  13,  -27),  S( -18,    2),  S( -36,   15),
        S(  17, -105),  S(  30,  -66),  S(   9,  -41),  S(  17,  -48),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  87,  122), // Pawn
         S( 301,  301), // Knight
         S( 292,  327), // Bishop
         S( 487,  603), // Rook
         S( 938, 1000), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(4), V(1)],
    EG_MOBILITY_OPEN = [
        V(-4), V(-4), V(-9), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(4), V(10), V(-19), V(0)],
    EG_MOBILITY_CLOSED = [
        V(19), V(3), V(39), V(-35)],
}

define_weight_params! {
    PHASE_WEIGHTS       = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS    = [CV(0), V(120), V(220), V(385), V(454), V(494)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS = [V(14), V(9), V(7)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS        = [V(9)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS = [V(27), V(75)], // [MG, EG]
}
