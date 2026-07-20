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
        $(pub const $name: Vi32x4 = Vi32x4::new([
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
            (mg_mob_open,            Vec4,   mobility_open_offset,      0),
            (eg_mob_open,            Vec4,   mobility_open_offset,      4),
            (mg_mob_closed,          Vec4,   mobility_closed_offset,    0),
            (eg_mob_closed,          Vec4,   mobility_closed_offset,    4),
            (w_shield,               Scalar, king_safety_offset,        0),
            (w_ortho,                Scalar, king_safety_offset,        1),
            (w_diag,                 Scalar, king_safety_offset,        2),
            (atk_weights,            Array6, attacker_offset,           0),
            (w_xray_ortho,           Scalar, xray_offset,               0),
            (w_bp_mg,                Scalar, bishop_pair_offset,        0),
            (w_bp_eg,                Scalar, bishop_pair_offset,        1),
            (w_rook_open_mg,         Scalar, rook_open_offset,          0),
            (w_rook_open_eg,         Scalar, rook_open_offset,          1),
            (passed_pawn_mg,         Array6, passed_pawn_mg_offset,     0),
            (passed_pawn_eg,         Array6, passed_pawn_eg_offset,     0),
            (enemy_king_dist_mg,     Array6, enemy_king_dist_mg_offset, 0),
            (enemy_king_dist_eg,     Array6, enemy_king_dist_eg_offset, 0),
            (w_doubled_pawn_mg,      Scalar, doubled_pawn_offset,       0),
            (w_doubled_pawn_eg,      Scalar, doubled_pawn_offset,       1),
            (w_isolated_pawn_mg,     Scalar, isolated_pawn_offset,      0),
            (w_isolated_pawn_eg,     Scalar, isolated_pawn_offset,      1),
            (phalanx_mg,             Array6, phalanx_mg_offset,         0),
            (phalanx_eg,             Array6, phalanx_eg_offset,         0),
            (defended_pawn_mg,       Array6, defended_pawn_mg_offset,   0),
            (defended_pawn_eg,       Array6, defended_pawn_eg_offset,   0),
            (w_backward_pawn_mg,     Scalar, backward_pawn_offset,      0),
            (w_backward_pawn_eg,     Scalar, backward_pawn_offset,      1),
            (w_tempo_mg,             Scalar, tempo_offset,              0),
            (w_tempo_eg,             Scalar, tempo_offset,              1),
            (w_minor_behind_pawn_mg, Scalar, minor_behind_pawn_offset,  0),
            (w_minor_behind_pawn_eg, Scalar, minor_behind_pawn_offset,  1)
        }
    };
}

/// One ordered row per parameter block: name and slot width.
/// Generates the `Layout` struct (`<name>_offset` / `<name>_len`) and the `LAYOUT` prefix-sum.
/// The order is the slot map; it must match `collect_parameters`'s collection
/// order, or every gradient indexes the wrong slot.
macro_rules! define_layout {
    ($( $name:ident = $len:expr ),* $(,)?) => {
        paste::paste! {
            pub struct Layout {
                $(
                    pub [<$name _offset>]: usize,
                    pub [<$name _len>]: usize,
                )*
                /// One past the last slot: the full tunable-region length.
                pub total: usize,
            }

            pub const LAYOUT: Layout = {
                $( let [<$name _len>]: usize = $len; )*
                let mut acc = 0usize;

                $(
                    let [<$name _offset>] = acc;
                    acc += [<$name _len>];
                )*

                Layout { $( [<$name _offset>], [<$name _len>], )* total: acc }
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
    minor_behind_pawn  = MINOR_BEHIND_PAWN_WEIGHTS.len(),
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
            assert!(p.is_fixed, "PHASE_WEIGHTS must be constant (CV); tuning phase is not supported.");
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
        S( 107,  177),  S( 119,  157),  S( 118,  157),  S( 148,  117),
        S(  20,   63),  S(  25,   63),  S(  81,    9),  S(  67,  -25),
        S(  -9,   42),  S(   1,   32),  S(  14,   15),  S(  30,   -2),
        S( -20,   31),  S( -18,   35),  S(  -8,   26),  S(  11,   21),
        S( -29,   26),  S( -24,   29),  S( -23,   32),  S( -16,   33),
        S( -27,   31),  S( -16,   33),  S( -15,   38),  S( -21,   38),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-155,    0),  S(-120,  102),  S( -46,  122),  S(  14,  118),
        S(  19,  105),  S(  33,  139),  S(  90,  134),  S(  70,  142),
        S(  38,  119),  S(  71,  146),  S( 100,  161),  S( 121,  164),
        S(  45,  140),  S(  39,  167),  S(  87,  180),  S(  76,  192),
        S(  18,  144),  S(  57,  154),  S(  49,  182),  S(  57,  194),
        S(  -5,  131),  S(  30,  155),  S(  34,  165),  S(  49,  183),
        S(  -3,  126),  S( -12,  133),  S(  15,  157),  S(  26,  162),
        S( -66,   99),  S( -29,   98),  S(  -9,  127),  S(  -5,  145),
    ],

    BISHOP = [
        S( -19,  132),  S( -15,  137),  S( -41,  130),  S( -69,  144),
        S(  30,  115),  S(  51,  138),  S(  53,  135),  S(  49,  133),
        S(  59,  129),  S(  67,  134),  S(  75,  147),  S(  80,  139),
        S(  33,  127),  S(  48,  142),  S(  67,  148),  S(  77,  162),
        S(  42,  113),  S(  40,  143),  S(  46,  153),  S(  62,  156),
        S(  38,  116),  S(  51,  132),  S(  47,  148),  S(  44,  153),
        S(  39,   99),  S(  50,  111),  S(  61,  112),  S(  36,  138),
        S(  29,   87),  S(  50,  116),  S(  13,  111),  S(  23,  127),
    ],

    ROOK = [
        S( -63,  195),  S( -87,  205),  S( -78,  208),  S( -82,  200),
        S( -48,  185),  S( -52,  194),  S( -25,  191),  S( -20,  184),
        S( -41,  178),  S(   3,  171),  S( -10,  170),  S( -10,  167),
        S( -45,  182),  S( -28,  174),  S( -24,  175),  S( -30,  173),
        S( -70,  181),  S( -50,  176),  S( -58,  180),  S( -43,  177),
        S( -67,  172),  S( -37,  157),  S( -53,  167),  S( -48,  172),
        S( -81,  175),  S( -57,  167),  S( -52,  167),  S( -51,  165),
        S( -61,  176),  S( -53,  171),  S( -64,  175),  S( -43,  160),
    ],

    QUEEN = [
        S(-171,  531),  S(-154,  528),  S(-185,  581),  S(-155,  561),
        S( -98,  527),  S(-157,  572),  S(-148,  605),  S(-175,  638),
        S( -87,  525),  S( -89,  542),  S(-128,  606),  S(-121,  608),
        S( -95,  546),  S(-118,  588),  S(-121,  590),  S(-137,  619),
        S(-107,  552),  S(-103,  572),  S(-119,  580),  S(-124,  603),
        S(-104,  523),  S( -99,  545),  S(-103,  564),  S(-109,  568),
        S( -91,  475),  S( -93,  486),  S( -86,  496),  S( -91,  532),
        S(-121,  504),  S(-119,  495),  S(-118,  497),  S(-104,  513),
    ],

    KING = [
        S(  61, -154),  S(  -6,  -49),  S(  -7,  -37),  S( -92,  -13),
        S(-151,    8),  S( -36,   49),  S( -61,   55),  S(  28,   29),
        S(-148,   16),  S( -24,   56),  S(   4,   60),  S( -44,   69),
        S(-182,   10),  S( -84,   49),  S( -85,   67),  S(-136,   79),
        S(-178,  -11),  S( -95,   23),  S( -83,   46),  S(-123,   67),
        S( -90,  -32),  S( -28,   -7),  S( -62,   23),  S( -67,   37),
        S(  26,  -68),  S(  21,  -35),  S( -12,   -5),  S( -36,   10),
        S(  33, -130),  S(  42,  -84),  S(  15,  -50),  S(  17,  -59),
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
        V(6), V(-5), V(6), V(5)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-8), V(7), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(0), V(7), V(-17), V(-4)],
    EG_MOBILITY_CLOSED = [
        V(23), V(6), V(37), V(-31)],
}

define_weight_params! {
    PHASE_WEIGHTS             = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS          = [CV(0), V(170), V(270), V(460), V(553), V(620)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS       = [V(22), V(10), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS              = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS       = [V(34), V(86)], // [MG, EG]
    ROOK_OPEN_WEIGHTS         = [V(39), V(4)], // [MG, EG]
    PASSED_PAWN_MG            = [V(-10), V(-25), V(-28), V(-4), V(-6), V(-31)], // by relative rank 1-6
    PASSED_PAWN_EG            = [V(-50), V(-30), V(19), V(75), V(177), V(147)], // by relative rank 1-6
    ENEMY_KING_DIST_MG        = [V(-118), V(27), V(11), V(8), V(8), V(1)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG        = [V(-39), V(7), V(47), V(62), V(75), V(83)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS      = [V(1), V(-41)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS     = [V(-11), V(-14)], // [MG, EG]
    PHALANX_MG                = [V(6), V(16), V(27), V(60), V(167), V(-330)], // by relative rank 2-7
    PHALANX_EG                = [V(-5), V(2), V(23), V(95), V(275), V(667)], // by relative rank 2-7
    DEFENDED_PAWN_MG          = [CV(0), V(31), V(20), V(20), V(26), V(195)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG          = [CV(0), V(17), V(13), V(29), V(73), V(62)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS     = [V(-8), V(-18)], // [MG, EG]
    TEMPO_WEIGHTS             = [V(30), V(37)], // [MG, EG], side-to-move initiative
    MINOR_BEHIND_PAWN_WEIGHTS = [V(14), V(31)], // [MG, EG]
}
