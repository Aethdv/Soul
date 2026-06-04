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
            (w_isolated_pawn_eg, Scalar, isolated_pawn_offset,      1),
            (phalanx_mg,         Array6, phalanx_mg_offset,         0),
            (phalanx_eg,         Array6, phalanx_eg_offset,         0),
            (defended_pawn_mg,   Array6, defended_pawn_mg_offset,   0),
            (defended_pawn_eg,   Array6, defended_pawn_eg_offset,   0),
            (w_backward_pawn_mg, Scalar, backward_pawn_offset,      0),
            (w_backward_pawn_eg, Scalar, backward_pawn_offset,      1),
            (w_tempo_mg,         Scalar, tempo_offset,              0),
            (w_tempo_eg,         Scalar, tempo_offset,              1)
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
    phalanx_mg         = PHALANX_MG.len(),
    phalanx_eg         = PHALANX_EG.len(),
    defended_pawn_mg   = DEFENDED_PAWN_MG.len(),
    defended_pawn_eg   = DEFENDED_PAWN_EG.len(),
    backward_pawn      = BACKWARD_PAWN_WEIGHTS.len(),
    tempo              = TEMPO_WEIGHTS.len(),
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
        S(  62,  189),  S(  88,  151),  S(  72,  184),  S(  87,  159),
        S(  12,   55),  S(  19,   56),  S(  73,    9),  S(  54,  -18),
        S( -19,   34),  S( -10,   26),  S(   0,    9),  S(  18,   -7),
        S( -32,   23),  S( -28,   27),  S( -19,   19),  S(   0,   12),
        S( -40,   19),  S( -32,   21),  S( -34,   26),  S( -25,   26),
        S( -37,   25),  S( -24,   27),  S( -25,   33),  S( -30,   33),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-184,  -37),  S(-143,   53),  S(-114,   78),  S( -56,   83),
        S( -21,   60),  S( -10,   91),  S(  42,   88),  S(  11,   98),
        S(  -2,   77),  S(  32,   99),  S(  54,  117),  S(  70,  115),
        S(   2,   95),  S(  -6,  124),  S(  40,  138),  S(  35,  147),
        S( -32,  101),  S(   6,  116),  S(   2,  138),  S(  13,  153),
        S( -55,   87),  S( -19,  108),  S( -15,  118),  S(   4,  142),
        S( -47,   92),  S( -55,  101),  S( -32,  113),  S( -23,  116),
        S(-106,   70),  S( -71,   62),  S( -54,   84),  S( -56,   98),
    ],

    BISHOP = [
        S( -64,   82),  S( -76,   95),  S( -94,   87),  S(-142,  104),
        S( -15,   70),  S(   5,   92),  S(   2,   89),  S(  -5,   88),
        S(  19,   90),  S(  22,   94),  S(  26,  105),  S(  30,   91),
        S( -10,   87),  S(   2,  101),  S(  17,  106),  S(  34,  114),
        S( -10,   74),  S(  -9,  100),  S(  -3,  109),  S(  19,  114),
        S( -13,   79),  S(   2,   89),  S(  -3,  107),  S(   0,  109),
        S(  -4,   71),  S(   6,   77),  S(  14,   72),  S( -11,   95),
        S( -20,   52),  S(   3,   83),  S( -32,   69),  S( -30,   82),
    ],

    ROOK = [
        S(-147,  112),  S(-176,  124),  S(-171,  130),  S(-173,  121),
        S(-128,  106),  S(-134,  115),  S(-108,  110),  S(-105,  105),
        S(-123,  101),  S( -81,   94),  S( -94,   94),  S( -94,   90),
        S(-129,  104),  S(-111,   97),  S(-110,  100),  S(-114,   96),
        S(-153,  106),  S(-134,  101),  S(-139,  103),  S(-127,  102),
        S(-147,   92),  S(-119,   79),  S(-134,   90),  S(-131,   95),
        S(-162,   93),  S(-140,   88),  S(-136,   88),  S(-135,   86),
        S(-143,   96),  S(-136,   91),  S(-147,   95),  S(-128,   81),
    ],

    QUEEN = [
        S(-490,  369),  S(-477,  384),  S(-497,  431),  S(-481,  429),
        S(-432,  407),  S(-483,  446),  S(-470,  467),  S(-501,  502),
        S(-423,  421),  S(-416,  418),  S(-451,  475),  S(-449,  483),
        S(-432,  436),  S(-454,  473),  S(-450,  464),  S(-467,  498),
        S(-448,  444),  S(-441,  455),  S(-459,  472),  S(-458,  480),
        S(-441,  404),  S(-440,  434),  S(-444,  449),  S(-449,  453),
        S(-429,  359),  S(-431,  367),  S(-428,  383),  S(-434,  418),
        S(-467,  388),  S(-460,  379),  S(-461,  379),  S(-446,  397),
    ],

    KING = [
        S(  16, -142),  S(  -7,  -47),  S( -26,  -31),  S( -93,  -16),
        S(-151,   11),  S(   2,   38),  S( -47,   47),  S(  36,   22),
        S(-161,   13),  S(   2,   46),  S( -12,   55),  S( -37,   60),
        S(-179,    3),  S( -62,   34),  S( -82,   59),  S(-129,   70),
        S(-166,  -16),  S( -86,   15),  S( -84,   39),  S(-129,   62),
        S( -86,  -35),  S( -29,   -9),  S( -59,   17),  S( -61,   30),
        S(  16,  -63),  S(  14,  -31),  S( -18,   -6),  S( -37,    6),
        S(  19, -120),  S(  32,  -77),  S(   5,  -47),  S(   9,  -57),
    ],
}

define_simple_params! {
    MATERIAL = [
         CS(  92,  124), // Pawn
         CS( 373,  419), // Knight
         CS( 372,  462), // Bishop
         CS( 568,  867), // Rook
         CS(1160, 1468), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    MG_MOBILITY_OPEN = [
        V(4), V(-7), V(5), V(5)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-9), V(5), V(-8)],
    MG_MOBILITY_CLOSED = [
        V(0), V(7), V(-18), V(-7)],
    EG_MOBILITY_CLOSED = [
        V(26), V(10), V(42), V(-30)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(170), V(270), V(457), V(543), V(560)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(20), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(35), V(75)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(41), V(2)], // [MG, EG]
    PASSED_PAWN_MG        = [V(-18), V(-32), V(-36), V(-12), V(-19), V(-4)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(-42), V(-19), V(29), V(80), V(174), V(107)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-91), V(35), V(20), V(16), V(14), V(9)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-53), V(-4), V(36), V(50), V(62), V(68)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(1), V(-43)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-9), V(-13)], // [MG, EG]
    PHALANX_MG            = [V(7), V(17), V(28), V(60), V(152), V(-364)], // by relative rank 2-7
    PHALANX_EG            = [V(-7), V(2), V(23), V(89), V(201), V(607)], // by relative rank 2-7
    DEFENDED_PAWN_MG      = [CV(0), V(30), V(20), V(18), V(27), V(227)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG      = [CV(0), V(15), V(12), V(26), V(62), V(5)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS = [V(-8), V(-17)], // [MG, EG]
    TEMPO_WEIGHTS         = [V(30), V(37)], // [MG, EG] — side-to-move initiative
}
