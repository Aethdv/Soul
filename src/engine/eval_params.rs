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
        S( 118,  174),  S( 151,  136),  S( 135,  167),  S( 151,  141),
        S(  13,   54),  S(  25,   53),  S(  75,    8),  S(  58,  -19),
        S( -16,   32),  S(  -5,   23),  S(   7,    5),  S(  23,  -10),
        S( -29,   21),  S( -25,   24),  S( -18,   16),  S(   3,   10),
        S( -37,   17),  S( -32,   19),  S( -32,   24),  S( -25,   24),
        S( -36,   24),  S( -24,   25),  S( -25,   31),  S( -31,   31),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-179,  -49),  S(-132,   37),  S(-103,   61),  S( -47,   62),
        S( -11,   45),  S(  -1,   75),  S(  48,   71),  S(  25,   79),
        S(   3,   55),  S(  30,   76),  S(  60,   92),  S(  83,   94),
        S(   8,   70),  S(   1,   96),  S(  50,  111),  S(  41,  123),
        S( -23,   80),  S(  16,   91),  S(  10,  113),  S(  20,  125),
        S( -46,   66),  S(  -9,   87),  S(  -5,   95),  S(   8,  116),
        S( -45,   66),  S( -53,   72),  S( -26,   88),  S( -15,   93),
        S(-105,   45),  S( -70,   34),  S( -52,   60),  S( -47,   79),
    ],

    BISHOP = [
        S( -60,   67),  S( -74,   78),  S( -81,   68),  S(-135,   85),
        S(  -7,   51),  S(  18,   70),  S(   9,   71),  S(   6,   68),
        S(  19,   64),  S(  24,   70),  S(  28,   81),  S(  37,   71),
        S(  -6,   63),  S(   8,   76),  S(  28,   80),  S(  40,   91),
        S(  -3,   50),  S(  -1,   76),  S(   6,   85),  S(  24,   88),
        S(  -6,   55),  S(  10,   67),  S(   6,   82),  S(   4,   84),
        S(  -6,   40),  S(   8,   46),  S(  18,   47),  S(  -6,   72),
        S( -18,   29),  S(   6,   57),  S( -29,   45),  S( -21,   62),
    ],

    ROOK = [
        S(-138,   80),  S(-163,   93),  S(-162,   97),  S(-159,   87),
        S(-109,   70),  S(-116,   80),  S( -87,   75),  S( -89,   70),
        S(-106,   67),  S( -66,   60),  S( -75,   59),  S( -74,   54),
        S(-110,   71),  S( -95,   63),  S( -93,   66),  S( -96,   62),
        S(-136,   72),  S(-118,   68),  S(-124,   70),  S(-111,   67),
        S(-132,   59),  S(-104,   45),  S(-118,   55),  S(-118,   62),
        S(-145,   60),  S(-124,   52),  S(-118,   53),  S(-119,   52),
        S(-126,   62),  S(-119,   57),  S(-130,   61),  S(-110,   47),
    ],

    QUEEN = [
        S(-471,  287),  S(-455,  302),  S(-469,  343),  S(-466,  352),
        S(-414,  333),  S(-467,  373),  S(-454,  394),  S(-481,  422),
        S(-407,  350),  S(-400,  341),  S(-433,  400),  S(-431,  405),
        S(-415,  360),  S(-438,  401),  S(-434,  391),  S(-454,  425),
        S(-433,  373),  S(-425,  380),  S(-444,  396),  S(-442,  406),
        S(-426,  332),  S(-425,  360),  S(-429,  377),  S(-434,  378),
        S(-415,  288),  S(-418,  296),  S(-413,  310),  S(-419,  345),
        S(-454,  314),  S(-448,  308),  S(-448,  310),  S(-431,  324),
    ],

    KING = [
        S(  37, -142),  S( -29,  -42),  S( -35,  -29),  S(-109,  -12),
        S(-170,   11),  S(  -5,   37),  S( -59,   44),  S(  26,   18),
        S(-134,    6),  S(  -1,   43),  S(  -5,   50),  S( -30,   57),
        S(-169,    0),  S( -74,   35),  S( -76,   56),  S(-135,   68),
        S(-163,  -19),  S( -88,   13),  S( -87,   38),  S(-127,   59),
        S( -87,  -37),  S( -29,  -12),  S( -60,   15),  S( -63,   28),
        S(  16,  -65),  S(  14,  -35),  S( -18,   -9),  S( -38,    4),
        S(  20, -122),  S(  33,  -80),  S(   5,  -49),  S(   8,  -58),
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
        V(-6), V(-8), V(7), V(-7)],
    MG_MOBILITY_CLOSED = [
        V(1), V(8), V(-18), V(-5)],
    EG_MOBILITY_CLOSED = [
        V(26), V(11), V(41), V(-29)],
}

define_weight_params! {
    PHASE_WEIGHTS             = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS          = [CV(0), V(180), V(270), V(455), V(535), V(560)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS       = [V(22), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS              = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS       = [V(38), V(70)], // [MG, EG]
    ROOK_OPEN_WEIGHTS         = [V(42), V(1)], // [MG, EG]
    PASSED_PAWN_MG            = [V(-19), V(-32), V(-36), V(-14), V(-15), V(-60)], // by relative rank 1-6
    PASSED_PAWN_EG            = [V(-48), V(-25), V(22), V(74), V(165), V(110)], // by relative rank 1-6
    ENEMY_KING_DIST_MG        = [V(-96), V(35), V(21), V(17), V(14), V(10)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG        = [V(-46), V(3), V(42), V(54), V(66), V(73)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS      = [V(2), V(-44)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS     = [V(-9), V(-12)], // [MG, EG]
    PHALANX_MG                = [V(7), V(15), V(27), V(60), V(144), V(-322)], // by relative rank 2-7
    PHALANX_EG                = [V(-7), V(1), V(23), V(86), V(193), V(561)], // by relative rank 2-7
    DEFENDED_PAWN_MG          = [CV(0), V(31), V(20), V(18), V(26), V(232)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG          = [CV(0), V(15), V(13), V(28), V(61), V(-7)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS     = [V(-8), V(-16)], // [MG, EG]
    TEMPO_WEIGHTS             = [V(32), V(36)], // [MG, EG], side-to-move initiative
    MINOR_BEHIND_PAWN_WEIGHTS = [V(14), V(31)], // [MG, EG]
}
