//! Static evaluation parameters and weight arrays.
//!
//! Contains the constants and tunable structures that map piece-square placement
//! and spatial features to centipawn scores.

#![allow(non_snake_case)]

use crate::weave::Vi32x4;

/// Slot count each `EvalParams` field consumes in the dual-AD gradient vector.
#[rustfmt::skip]
macro_rules! slot_width {
    (Scalar) => (1);
    (Vec4)   => (4);
    (Array4) => (4);
    (Array6) => (6);
}

/// Sum the dual-AD footprint over the tunable list; 2 accumulator lanes (MG/EG)
/// plus one slot per scalar weight.
macro_rules! count_dual_slots {
    ($( ($name:ident, $ty:ident, $off:ident, $extra:expr) ),* $(,)?) => {
        2usize $( + slot_width!($ty) )*
    };
}

/// Total dual-AD inputs; the 2 accumulator lanes plus every tunable weight.
/// Drives `DUAL_N`, so the gradient array sizes itself as eval terms are added.
pub const DUAL_SLOTS: usize = crate::define_tunables!(count_dual_slots);

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
            (mg_mob_open,        Vec4,   mobility_open_offset,      0),
            (eg_mob_open,        Vec4,   mobility_open_offset,      4),
            (mg_mob_closed,      Vec4,   mobility_closed_offset,    0),
            (eg_mob_closed,      Vec4,   mobility_closed_offset,    4),
            (w_shield,           Scalar, king_safety_offset,        0),
            (w_ortho,            Scalar, king_safety_offset,        1),
            (w_diag,             Scalar, king_safety_offset,        2),
            (atk_weights,        Array6, attacker_offset,           0),
            (w_xray_ortho,       Scalar, xray_offset,               0),
            (w_bp_mg,            Scalar, bishop_pair_offset,        0),
            (w_bp_eg,            Scalar, bishop_pair_offset,        1),
            (w_rook_open_mg,     Scalar, rook_open_offset,          0),
            (w_rook_open_eg,     Scalar, rook_open_offset,          1),
            (passed_pawn_mg,     Array6, passed_pawn_mg_offset,     0),
            (passed_pawn_eg,     Array6, passed_pawn_eg_offset,     0),
            (enemy_king_dist_mg, Array6, enemy_king_dist_mg_offset, 0),
            (enemy_king_dist_eg, Array6, enemy_king_dist_eg_offset, 0),
            (w_doubled_pawn_mg,  Scalar, doubled_pawn_offset,       0),
            (w_doubled_pawn_eg,  Scalar, doubled_pawn_offset,       1),
            (w_isolated_pawn_mg, Scalar, isolated_pawn_offset,      0),
            (w_isolated_pawn_eg, Scalar, isolated_pawn_offset,      1)
        }
    };
}

/// One ordered row per parameter block: name and slot width.
/// Generates the `Layout` struct (`<name>_offset` / `<name>_len`) and the `LAYOUT` prefix-sum.
/// The order *is* the slot map — it must match `collect_parameters`'s collection
/// order, or every gradient indexes the wrong slot.
macro_rules! define_layout {
    ($( $name:ident = $len:expr ),* $(,)?) => {
        paste::paste! {
            pub struct Layout {
                $(
                    pub [<$name _offset>]: usize,
                    pub [<$name _len>]: usize,
                )*
            }

            pub const LAYOUT: Layout = {
                $( let [<$name _len>]: usize = $len; )*
                let mut acc = 0usize;
                $(
                    let [<$name _offset>] = acc;
                    acc += [<$name _len>];
                )*
                let _total = acc;
                Layout { $( [<$name _offset>], [<$name _len>], )* }
            };
        }
    };
}

define_layout! {
    psqt               = (PAWN.len() + KNIGHT.len() + BISHOP.len() + ROOK.len() + QUEEN.len() + KING.len()) * 2,
    material           = MATERIAL.len() * 2,
    mobility_open      = 4 * 2, // MG + EG
    mobility_closed    = 4 * 2,
    weight             = PHASE_WEIGHTS.len(),
    attacker           = ATTACKER_WEIGHTS.len(),
    king_safety        = KING_SAFETY_WEIGHTS.len(),
    xray               = XRAY_WEIGHTS.len(),
    bishop_pair        = BISHOP_PAIR_WEIGHTS.len(),
    rook_open          = ROOK_OPEN_WEIGHTS.len(),
    passed_pawn_mg     = PASSED_PAWN_MG.len(),
    passed_pawn_eg     = PASSED_PAWN_EG.len(),
    enemy_king_dist_mg = ENEMY_KING_DIST_MG.len(),
    enemy_king_dist_eg = ENEMY_KING_DIST_EG.len(),
    doubled_pawn       = DOUBLED_PAWN_WEIGHTS.len(),
    isolated_pawn      = ISOLATED_PAWN_WEIGHTS.len(),
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
        S(  10,  224),  S(  12,  209),  S(  25,  203),  S(  55,  162),
        S(  21,   86),  S(  25,   90),  S(  88,   35),  S(  65,    0),
        S(  -4,   62),  S(   4,   56),  S(  23,   38),  S(  39,   20),
        S( -16,   42),  S( -13,   46),  S(   8,   38),  S(  29,   31),
        S( -12,   37),  S(   5,   38),  S(   3,   42),  S(  14,   42),
        S( -22,   38),  S(  -2,   36),  S(   0,   44),  S(  -5,   44),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-151,   14),  S(-122,  116),  S( -38,  132),  S(  25,  142),
        S(  36,  113),  S(  48,  149),  S( 111,  146),  S(  89,  158),
        S(  72,  127),  S(  98,  158),  S( 121,  178),  S( 141,  179),
        S(  70,  153),  S(  62,  186),  S( 111,  199),  S( 106,  210),
        S(  37,  160),  S(  76,  171),  S(  73,  200),  S(  85,  214),
        S(  15,  145),  S(  51,  169),  S(  59,  176),  S(  77,  201),
        S(  25,  146),  S(  14,  153),  S(  38,  169),  S(  52,  176),
        S( -53,  118),  S(   1,  121),  S(  15,  139),  S(  11,  151),
    ],

    BISHOP = [
        S(   3,  132),  S( -12,  150),  S( -12,  137),  S( -47,  154),
        S(  46,  116),  S(  55,  146),  S(  69,  141),  S(  63,  141),
        S(  86,  143),  S(  97,  145),  S(  96,  160),  S(  97,  150),
        S(  53,  141),  S(  72,  158),  S(  89,  162),  S( 101,  176),
        S(  58,  129),  S(  57,  159),  S(  70,  168),  S(  88,  175),
        S(  63,  131),  S(  74,  148),  S(  69,  165),  S(  75,  170),
        S(  73,  122),  S(  79,  138),  S(  90,  129),  S(  63,  154),
        S(  48,   98),  S(  80,  126),  S(  40,  127),  S(  35,  136),
    ],

    ROOK = [
        S( -27,  214),  S( -49,  224),  S( -43,  228),  S( -51,  222),
        S( -18,  207),  S( -27,  218),  S(   5,  217),  S(   7,  213),
        S( -14,  196),  S(  34,  192),  S(  17,  192),  S(  18,  190),
        S( -19,  200),  S(   4,  192),  S(   6,  196),  S(  -3,  194),
        S( -44,  202),  S( -24,  197),  S( -31,  203),  S( -16,  200),
        S( -40,  190),  S(  -8,  179),  S( -22,  188),  S( -18,  193),
        S( -56,  196),  S( -27,  188),  S( -22,  189),  S( -22,  186),
        S( -26,  206),  S( -23,  196),  S( -31,  198),  S( -13,  183),
    ],

    QUEEN = [
        S( -21,  538),  S(  -4,  539),  S( -27,  589),  S(  -9,  578),
        S(  38,  526),  S( -32,  581),  S( -13,  610),  S( -46,  656),
        S(  53,  523),  S(  55,  535),  S(   5,  613),  S(  20,  618),
        S(  47,  541),  S(  22,  589),  S(  20,  592),  S(   2,  626),
        S(  36,  555),  S(  41,  573),  S(  20,  589),  S(  19,  608),
        S(  39,  520),  S(  48,  545),  S(  43,  564),  S(  36,  572),
        S(  52,  477),  S(  52,  482),  S(  61,  494),  S(  54,  532),
        S(  22,  501),  S(  19,  499),  S(  22,  495),  S(  45,  507),
    ],

    KING = [
        S( 104, -167),  S(  -3,  -44),  S( -32,  -29),  S(-102,   -1),
        S(-149,   12),  S(  -4,   49),  S( -50,   66),  S(  49,   34),
        S(-133,   21),  S(  17,   57),  S(  11,   70),  S( -39,   80),
        S(-164,   15),  S( -73,   56),  S( -81,   77),  S(-143,   94),
        S(-179,   -1),  S( -86,   33),  S( -77,   59),  S(-130,   84),
        S( -87,  -21),  S( -17,    6),  S( -56,   36),  S( -68,   52),
        S(  29,  -56),  S(  31,  -23),  S(  -7,    6),  S( -31,   21),
        S(  39, -123),  S(  53,  -74),  S(  26,  -43),  S(  31,  -52),
    ],
}

define_simple_params! {
    MATERIAL = [
         S(  92,  124), // Pawn
         S( 373,  419), // Knight
         S( 372,  462), // Bishop
         S( 568,  867), // Rook
         S(1160, 1468), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(6), V(2)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-8), V(-6), V(-11)],
    MG_MOBILITY_CLOSED = [
        V(2), V(9), V(-20), V(1)],
    EG_MOBILITY_CLOSED = [
        V(23), V(7), V(48), V(-36)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(190), V(305), V(520), V(627), V(698)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(19), V(12), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(12)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(35), V(93)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(41), V(4)], // [MG, EG]
    PASSED_PAWN_MG        = [V(-35), V(-53), V(-54), V(-28), V(-23), V(53)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(-46), V(-29), V(22), V(75), V(182), V(113)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-107), V(46), V(34), V(27), V(29), V(24)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-36), V(6), V(45), V(61), V(73), V(82)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(-9), V(-42)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-21), V(-20)], // [MG, EG]
}
