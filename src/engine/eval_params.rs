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
        S(  64,  181),  S(  88,  145),  S(  73,  178),  S(  87,  152),
        S(   9,   49),  S(  16,   51),  S(  67,    4),  S(  49,  -21),
        S( -20,   29),  S( -12,   21),  S(  -2,    5),  S(  15,  -10),
        S( -33,   18),  S( -30,   23),  S( -20,   14),  S(  -3,    9),
        S( -41,   15),  S( -33,   17),  S( -36,   22),  S( -26,   21),
        S( -39,   21),  S( -26,   22),  S( -27,   28),  S( -32,   28),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-190,  -48),  S(-143,   39),  S(-120,   64),  S( -65,   68),
        S( -29,   46),  S( -22,   77),  S(  30,   75),  S(   0,   84),
        S( -11,   64),  S(  21,   84),  S(  41,  101),  S(  57,   99),
        S(  -9,   80),  S( -16,  109),  S(  29,  122),  S(  24,  131),
        S( -41,   87),  S(  -4,  101),  S(  -9,  123),  S(   2,  137),
        S( -64,   72),  S( -29,   94),  S( -25,  102),  S(  -6,  126),
        S( -57,   78),  S( -64,   85),  S( -42,   99),  S( -32,  100),
        S(-113,   55),  S( -80,   49),  S( -64,   69),  S( -65,   83),
    ],

    BISHOP = [
        S( -72,   67),  S( -85,   80),  S(-102,   71),  S(-148,   87),
        S( -27,   54),  S(  -5,   76),  S(  -9,   72),  S( -16,   73),
        S(   9,   74),  S(  10,   78),  S(  15,   90),  S(  19,   75),
        S( -20,   72),  S(  -8,   84),  S(   7,   91),  S(  23,   99),
        S( -21,   57),  S( -19,   84),  S( -13,   93),  S(   8,   97),
        S( -23,   64),  S(  -9,   75),  S( -13,   91),  S( -10,   93),
        S( -15,   56),  S(  -5,   62),  S(   3,   57),  S( -21,   78),
        S( -30,   37),  S(  -8,   68),  S( -42,   54),  S( -40,   67),
    ],

    ROOK = [
        S(-160,   84),  S(-187,   97),  S(-183,  102),  S(-184,   93),
        S(-140,   78),  S(-147,   88),  S(-121,   83),  S(-118,   77),
        S(-136,   74),  S( -96,   67),  S(-107,   67),  S(-107,   63),
        S(-141,   76),  S(-124,   69),  S(-124,   72),  S(-126,   69),
        S(-165,   78),  S(-145,   73),  S(-151,   75),  S(-139,   73),
        S(-159,   64),  S(-132,   51),  S(-146,   62),  S(-143,   67),
        S(-173,   66),  S(-151,   60),  S(-148,   61),  S(-148,   59),
        S(-155,   69),  S(-149,   63),  S(-159,   68),  S(-140,   54),
    ],

    QUEEN = [
        S(-509,  316),  S(-497,  331),  S(-516,  377),  S(-500,  374),
        S(-453,  354),  S(-502,  393),  S(-489,  413),  S(-520,  446),
        S(-445,  367),  S(-439,  365),  S(-472,  420),  S(-470,  427),
        S(-454,  381),  S(-475,  420),  S(-471,  409),  S(-487,  441),
        S(-468,  390),  S(-462,  401),  S(-479,  416),  S(-477,  425),
        S(-461,  351),  S(-460,  378),  S(-465,  395),  S(-470,  399),
        S(-450,  307),  S(-453,  314),  S(-449,  330),  S(-455,  365),
        S(-487,  335),  S(-480,  328),  S(-481,  327),  S(-466,  344),
    ],

    KING = [
        S(  16, -142),  S(  -7,  -50),  S( -26,  -35),  S( -93,  -21),
        S(-151,    6),  S(   2,   32),  S( -47,   42),  S(  36,   16),
        S(-161,    9),  S(   2,   40),  S( -12,   49),  S( -37,   54),
        S(-179,   -1),  S( -62,   29),  S( -82,   53),  S(-126,   63),
        S(-162,  -20),  S( -84,   12),  S( -82,   34),  S(-126,   55),
        S( -84,  -38),  S( -29,  -13),  S( -58,   14),  S( -61,   25),
        S(  13,  -66),  S(  12,  -35),  S( -18,  -11),  S( -38,    2),
        S(  18, -121),  S(  30,  -80),  S(   4,  -50),  S(   8,  -60),
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
        V(5), V(-6), V(6), V(5)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-8), V(5), V(-7)],
    MG_MOBILITY_CLOSED = [
        V(0), V(7), V(-18), V(-7)],
    EG_MOBILITY_CLOSED = [
        V(25), V(11), V(40), V(-30)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(170), V(260), V(444), V(526), V(543)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(19), V(10), V(8)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(33), V(72)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(39), V(2)], // [MG, EG]
    PASSED_PAWN_MG        = [V(-19), V(-33), V(-36), V(-14), V(-20), V(-10)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(-41), V(-18), V(28), V(77), V(169), V(101)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-87), V(36), V(20), V(16), V(14), V(9)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-52), V(-3), V(35), V(47), V(60), V(67)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(1), V(-42)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-8), V(-13)], // [MG, EG]
    PHALANX_MG            = [V(6), V(17), V(27), V(60), V(152), V(-364)], // by relative rank 2-7
    PHALANX_EG            = [V(-6), V(3), V(22), V(87), V(193), V(607)], // by relative rank 2-7
    DEFENDED_PAWN_MG      = [CV(0), V(29), V(19), V(18), V(27), V(227)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG      = [CV(0), V(15), V(12), V(26), V(59), V(2)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS = [V(-8), V(-17)], // [MG, EG]
    TEMPO_WEIGHTS         = [V(30), V(36)], // [MG, EG] — side-to-move initiative
}
