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
        S(  58,  151),  S(  56,  152),  S(  81,  138),  S( 109,  104),
        S( -18,   86),  S(  -5,   91),  S(  23,   55),  S(  26,   43),
        S( -33,   32),  S( -17,   29),  S( -13,    9),  S(   2,   -1),
        S( -41,   14),  S( -26,   19),  S( -22,    3),  S(  -8,    1),
        S( -38,    8),  S( -19,   15),  S( -31,    8),  S( -22,   13),
        S( -40,   10),  S( -17,   15),  S( -26,   12),  S( -37,   20),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-214,  -95),  S( -87,  -42),  S( -89,  -15),  S( -39,  -21),
        S( -67,  -41),  S( -63,  -13),  S( -27,  -14),  S( -23,   -4),
        S( -58,  -29),  S( -33,   -9),  S( -21,    3),  S(  -5,    2),
        S( -50,  -19),  S( -58,    6),  S( -30,   14),  S( -39,   24),
        S( -72,  -17),  S( -52,   -2),  S( -53,   13),  S( -48,   19),
        S( -91,  -29),  S( -66,   -9),  S( -62,   -6),  S( -50,   11),
        S( -82,  -36),  S( -90,  -18),  S( -77,   -9),  S( -68,   -7),
        S(-137,  -45),  S( -95,  -40),  S( -87,  -29),  S( -91,  -20),
    ],

    BISHOP = [
        S( -87,  -43),  S( -70,  -43),  S( -73,  -46),  S( -75,  -45),
        S( -69,  -56),  S( -55,  -39),  S( -48,  -40),  S( -47,  -42),
        S( -41,  -50),  S( -31,  -41),  S( -27,  -35),  S( -25,  -39),
        S( -58,  -50),  S( -54,  -30),  S( -38,  -32),  S( -31,  -22),
        S( -55,  -54),  S( -55,  -35),  S( -53,  -24),  S( -37,  -23),
        S( -55,  -54),  S( -48,  -41),  S( -52,  -28),  S( -51,  -27),
        S( -48,  -62),  S( -44,  -51),  S( -41,  -53),  S( -59,  -37),
        S( -71,  -61),  S( -50,  -60),  S( -74,  -55),  S( -70,  -50),
    ],

    ROOK = [
        S(-137, -123),  S(-135, -123),  S(-134, -120),  S(-146, -117),
        S(-144, -117),  S(-144, -115),  S(-126, -117),  S(-121, -119),
        S(-153, -118),  S(-128, -122),  S(-125, -124),  S(-120, -127),
        S(-157, -120),  S(-148, -120),  S(-143, -119),  S(-139, -123),
        S(-174, -122),  S(-164, -120),  S(-165, -118),  S(-154, -121),
        S(-176, -129),  S(-159, -134),  S(-165, -126),  S(-160, -126),
        S(-186, -129),  S(-168, -135),  S(-162, -134),  S(-162, -133),
        S(-166, -137),  S(-165, -137),  S(-167, -134),  S(-154, -142),
    ],

    QUEEN = [
        S(-496,   68),  S(-491,   77),  S(-492,   74),  S(-495,   76),
        S(-479,   55),  S(-504,   83),  S(-489,   81),  S(-495,   97),
        S(-472,   48),  S(-471,   62),  S(-484,   88),  S(-477,   88),
        S(-468,   43),  S(-482,   75),  S(-481,   83),  S(-488,   98),
        S(-472,   54),  S(-472,   63),  S(-479,   76),  S(-480,   87),
        S(-471,   31),  S(-464,   44),  S(-469,   62),  S(-473,   66),
        S(-461,    3),  S(-463,   11),  S(-455,   12),  S(-462,   35),
        S(-469,    7),  S(-478,   13),  S(-479,    9),  S(-467,    8),
    ],

    KING = [
        S( 110, -127),  S( 101,  -25),  S(  19,    2),  S(  -7,    0),
        S(  99,  -44),  S( -14,   36),  S( -37,   39),  S( -43,   34),
        S( -28,   -2),  S( -25,   37),  S( -19,   38),  S( -45,   40),
        S( -52,   -9),  S( -37,   26),  S( -41,   35),  S( -52,   38),
        S( -97,  -16),  S( -72,   10),  S( -64,   20),  S( -68,   27),
        S( -78,  -31),  S( -60,  -10),  S( -73,    5),  S( -70,   10),
        S( -17,  -53),  S( -23,  -34),  S( -47,  -17),  S( -60,   -9),
        S( -12,  -90),  S(  -6,  -60),  S( -28,  -47),  S( -21,  -54),
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
        V(3), V(-3), V(5), V(1)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-3), V(-3), V(3), V(-12)],
    MG_MOBILITY_CLOSED = [
        V(2), V(5), V(-12), V(-2)],
    EG_MOBILITY_CLOSED = [
        V(14), V(-1), V(26), V(-15)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(40), V(70), V(143), V(160), V(181)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(15), V(5), V(4)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(7)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(23), V(66)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(13), V(-5)], // [MG, EG]
    PASSED_PAWN_MG        = [V(4), V(-1), V(-10), V(-9), V(-2), V(-47)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(9), V(12), V(33), V(57), V(69), V(77)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-19), V(12), V(1), V(0), V(-3), V(-4)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-61), V(-30), V(-16), V(-12), V(-9), V(-9)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(2), V(-9)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-4), V(-3)], // [MG, EG]
    PHALANX_MG            = [V(-1), V(6), V(11), V(15), V(33), V(-293)], // by relative rank 2-7
    PHALANX_EG            = [V(3), V(0), V(10), V(43), V(64), V(206)], // by relative rank 2-7
    DEFENDED_PAWN_MG      = [CV(0), V(5), V(6), V(9), V(11), V(102)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG      = [CV(0), V(7), V(4), V(7), V(20), V(-18)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS = [V(-2), V(-6)], // [MG, EG]
    TEMPO_WEIGHTS         = [V(6), V(1)], // [MG, EG] — side-to-move initiative
}
