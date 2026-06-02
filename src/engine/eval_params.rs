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
    (Vec4)   => (4);inconceivable
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
        S(  -7,  212),  S(  14,  204),  S(  21,  194),  S(  41,  157),
        S(  10,   79),  S(  28,   89),  S(  81,   33),  S(  59,   -1),
        S( -13,   58),  S(  13,   62),  S(  15,   36),  S(  32,   18),
        S( -24,   40),  S(  -3,   57),  S(   3,   37),  S(  23,   30),
        S( -19,   35),  S(  16,   51),  S(  -2,   41),  S(   9,   42),
        S( -29,   35),  S(  10,   48),  S(  -5,   44),  S( -11,   44),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-144,   19),  S(-111,  121),  S( -34,  133),  S(  32,  138),
        S(  43,  113),  S(  51,  148),  S( 112,  148),  S(  88,  157),
        S(  68,  128),  S(  97,  158),  S( 124,  175),  S( 143,  177),
        S(  69,  153),  S(  62,  187),  S( 111,  199),  S( 105,  210),
        S(  36,  161),  S(  76,  171),  S(  72,  201),  S(  86,  213),
        S(  13,  145),  S(  50,  169),  S(  59,  175),  S(  78,  201),
        S(  25,  144),  S(  11,  153),  S(  38,  170),  S(  52,  176),
        S( -50,  124),  S(   1,  121),  S(  14,  137),  S(   9,  153),
    ],

    BISHOP = [
        S(   9,  135),  S(   2,  145),  S( -13,  139),  S( -51,  154),
        S(  45,  117),  S(  61,  145),  S(  67,  144),  S(  59,  143),
        S(  87,  145),  S(  97,  147),  S(  97,  158),  S(  96,  150),
        S(  53,  142),  S(  72,  158),  S(  88,  164),  S( 102,  176),
        S(  57,  129),  S(  56,  161),  S(  70,  169),  S(  87,  175),
        S(  62,  134),  S(  73,  147),  S(  69,  166),  S(  76,  170),
        S(  71,  125),  S(  79,  138),  S(  90,  127),  S(  63,  155),
        S(  46,  101),  S(  79,  128),  S(  40,  126),  S(  36,  137),
    ],

    ROOK = [
        S( -26,  216),  S( -54,  228),  S( -41,  232),  S( -55,  227),
        S( -19,  210),  S( -27,  222),  S(   3,  221),  S(  10,  215),
        S( -19,  202),  S(  31,  197),  S(  17,  197),  S(  15,  195),
        S( -23,  204),  S(   4,  196),  S(   3,  201),  S(  -4,  198),
        S( -46,  206),  S( -27,  203),  S( -34,  207),  S( -19,  205),
        S( -41,  194),  S( -10,  184),  S( -23,  192),  S( -21,  199),
        S( -59,  204),  S( -28,  193),  S( -23,  193),  S( -25,  193),
        S( -28,  211),  S( -26,  201),  S( -33,  203),  S( -15,  188),
    ],

    QUEEN = [
        S( -16,  538),  S(   5,  538),  S( -27,  598),  S( -13,  589),
        S(  37,  535),  S( -32,  589),  S( -14,  620),  S( -42,  656),
        S(  51,  532),  S(  55,  541),  S(   6,  619),  S(  21,  620),
        S(  47,  547),  S(  20,  597),  S(  20,  598),  S(   3,  624),
        S(  35,  563),  S(  39,  581),  S(  20,  593),  S(  19,  612),
        S(  38,  528),  S(  46,  554),  S(  42,  571),  S(  35,  580),
        S(  52,  484),  S(  49,  494),  S(  60,  500),  S(  53,  538),
        S(  22,  509),  S(  17,  509),  S(  20,  505),  S(  44,  513),
    ],

    KING = [
        S(  79, -158),  S(   7,  -48),  S( -22,  -31),  S( -83,   -8),
        S(-137,    9),  S( -23,   49),  S( -52,   60),  S(  41,   31),
        S(-126,   17),  S(  15,   55),  S(   6,   66),  S( -40,   76),
        S(-172,   14),  S( -74,   54),  S( -80,   76),  S(-143,   93),
        S(-178,   -2),  S( -82,   30),  S( -82,   58),  S(-127,   83),
        S( -87,  -23),  S( -17,    3),  S( -55,   34),  S( -64,   50),
        S(  29,  -57),  S(  31,  -25),  S(  -6,    5),  S( -29,   19),
        S(  39, -125),  S(  53,  -76),  S(  27,  -44),  S(  33,  -53),
    ],
}

define_simple_params! {
    MATERIAL = [
         S( 103,  137), // Pawn
         S( 375,  423), // Knight
         S( 375,  464), // Bishop
         S( 574,  870), // Rook
         S(1167, 1471), // Queen
         S(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(5), V(-7), V(7), V(4)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-9), V(-5), V(-13)],
    MG_MOBILITY_CLOSED = [
        V(4), V(9), V(-23), V(-2)],
    EG_MOBILITY_CLOSED = [
        V(24), V(8), V(45), V(-34)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(180), V(305), V(525), V(630), V(698)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(19), V(12), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(12)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(35), V(93)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(42), V(2)], // [MG, EG]
    PASSED_PAWN_MG        = [V(-39), V(-57), V(-60), V(-32), V(-24), V(57)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(-58), V(-40), V(12), V(69), V(176), V(109)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-107), V(43), V(32), V(25), V(21), V(14)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-46), V(-5), V(36), V(51), V(63), V(72)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(-26), V(-73)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-7), V(-15)], // [MG, EG]
}
