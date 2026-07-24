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
        S( 136,  168),  S( 170,  129),  S( 152,  160),  S( 168,  136),
        S(  20,   63),  S(  30,   64),  S(  84,   15),  S(  67,  -13),
        S( -10,   41),  S(   1,   32),  S(  14,   14),  S(  30,   -3),
        S( -25,   30),  S( -21,   34),  S( -13,   25),  S(   9,   19),
        S( -34,   26),  S( -28,   28),  S( -29,   33),  S( -21,   33),
        S( -32,   33),  S( -19,   34),  S( -20,   41),  S( -26,   41),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-169,  -27),  S(-132,   69),  S( -88,   90),  S( -35,   97),
        S(   8,   72),  S(  22,  105),  S(  76,  102),  S(  46,  110),
        S(  26,   87),  S(  56,  108),  S(  87,  124),  S( 111,  125),
        S(  32,  100),  S(  25,  129),  S(  76,  144),  S(  68,  157),
        S(  -1,  110),  S(  43,  121),  S(  33,  147),  S(  44,  160),
        S( -25,   95),  S(  14,  117),  S(  18,  128),  S(  33,  149),
        S( -23,   97),  S( -32,  105),  S(  -5,  121),  S(   7,  126),
        S( -87,   73),  S( -51,   62),  S( -31,   92),  S( -25,  109),
    ],

    BISHOP = [
        S( -39,   98),  S( -55,  111),  S( -73,  102),  S(-118,  121),
        S(  16,   86),  S(  38,  108),  S(  33,  105),  S(  29,  102),
        S(  44,   98),  S(  50,  103),  S(  54,  115),  S(  63,  106),
        S(  17,   96),  S(  32,  110),  S(  53,  114),  S(  65,  127),
        S(  21,   84),  S(  22,  111),  S(  30,  119),  S(  49,  122),
        S(  18,   89),  S(  35,   99),  S(  30,  116),  S(  28,  119),
        S(  18,   71),  S(  32,   78),  S(  43,   80),  S(  18,  105),
        S(   4,   59),  S(  31,   90),  S(  -7,   77),  S(   2,   97),
    ],

    ROOK = [
        S(-104,  136),  S(-137,  150),  S(-131,  155),  S(-136,  147),
        S( -80,  129),  S( -89,  139),  S( -60,  135),  S( -56,  127),
        S( -78,  125),  S( -35,  119),  S( -40,  115),  S( -44,  111),
        S( -84,  129),  S( -66,  122),  S( -61,  124),  S( -66,  118),
        S(-107,  128),  S( -88,  127),  S( -95,  128),  S( -81,  125),
        S(-102,  115),  S( -75,  103),  S( -89,  113),  S( -86,  118),
        S(-119,  118),  S( -95,  110),  S( -89,  110),  S( -90,  109),
        S( -97,  119),  S( -90,  114),  S(-102,  118),  S( -80,  103),
    ],

    QUEEN = [
        S(-419,  393),  S(-410,  412),  S(-431,  464),  S(-415,  461),
        S(-362,  443),  S(-418,  484),  S(-407,  509),  S(-435,  540),
        S(-356,  459),  S(-350,  452),  S(-383,  511),  S(-384,  523),
        S(-365,  473),  S(-392,  520),  S(-386,  506),  S(-403,  538),
        S(-384,  485),  S(-375,  495),  S(-396,  513),  S(-393,  522),
        S(-377,  443),  S(-375,  473),  S(-379,  489),  S(-384,  490),
        S(-365,  396),  S(-369,  405),  S(-362,  417),  S(-368,  456),
        S(-406,  426),  S(-399,  421),  S(-399,  419),  S(-381,  434),
    ],

    KING = [
        S(   9, -144),  S( -38,  -44),  S( -40,  -30),  S(-105,  -12),
        S(-145,    4),  S(  -2,   40),  S( -58,   53),  S(  29,   24),
        S(-155,   12),  S(   7,   48),  S(  -8,   56),  S( -34,   62),
        S(-167,   -1),  S( -72,   38),  S( -79,   61),  S(-134,   75),
        S(-166,  -18),  S( -92,   17),  S( -91,   43),  S(-137,   66),
        S( -91,  -36),  S( -30,  -11),  S( -60,   18),  S( -61,   32),
        S(  20,  -67),  S(  18,  -35),  S( -14,   -8),  S( -36,    6),
        S(  25, -127),  S(  39,  -83),  S(  10,  -50),  S(  13,  -60),
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
        V(5), V(-7), V(5), V(6)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-9), V(8), V(-8)],
    MG_MOBILITY_CLOSED = [
        V(0), V(9), V(-18), V(-6)],
    EG_MOBILITY_CLOSED = [
        V(28), V(12), V(43), V(-31)],
}

define_weight_params! {
    PHASE_WEIGHTS             = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS          = [CV(0), V(190), V(290), V(485), V(569), V(588)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS       = [V(23), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS              = [V(11)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS       = [V(40), V(75)], // [MG, EG]
    ROOK_OPEN_WEIGHTS         = [V(45), V(2)], // [MG, EG]
    PASSED_PAWN_MG            = [V(-15), V(-30), V(-33), V(-10), V(-13), V(-60)], // by relative rank 1-6
    PASSED_PAWN_EG            = [V(-47), V(-23), V(27), V(82), V(179), V(145)], // by relative rank 1-6
    ENEMY_KING_DIST_MG        = [V(-109), V(33), V(17), V(12), V(11), V(5)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG        = [V(-52), V(0), V(40), V(55), V(67), V(75)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS      = [V(2), V(-46)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS     = [V(-9), V(-13)], // [MG, EG]
    PHALANX_MG                = [V(7), V(16), V(28), V(61), V(164), V(-354)], // by relative rank 2-7
    PHALANX_EG                = [V(-8), V(2), V(24), V(93), V(204), V(583)], // by relative rank 2-7
    DEFENDED_PAWN_MG          = [CV(0), V(33), V(21), V(19), V(31), V(239)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG          = [CV(0), V(16), V(14), V(29), V(64), V(1)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS     = [V(-8), V(-17)], // [MG, EG]
    TEMPO_WEIGHTS             = [V(34), V(38)], // [MG, EG], side-to-move initiative
    MINOR_BEHIND_PAWN_WEIGHTS = [V(15), V(33)], // [MG, EG]
}
