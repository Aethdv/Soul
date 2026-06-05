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
        S(  65,  143),  S(  56,  154),  S(  76,  145),  S(  88,  126),
        S(  -9,   63),  S(  -6,   77),  S(  16,   50),  S(  22,   36),
        S( -32,   20),  S( -23,   22),  S( -18,    4),  S(  -3,   -5),
        S( -42,    6),  S( -32,   14),  S( -27,   -1),  S( -15,   -4),
        S( -38,   -1),  S( -23,    9),  S( -35,    2),  S( -26,    6),
        S( -40,    0),  S( -21,    8),  S( -30,    5),  S( -40,   15),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-206, -145),  S(-118,  -80),  S(-175,  -25),  S( -47,  -48),
        S( -70,  -63),  S( -74,  -30),  S( -25,  -30),  S( -25,  -21),
        S( -57,  -49),  S( -44,  -18),  S( -29,   -1),  S( -21,    2),
        S( -56,  -33),  S( -66,   -2),  S( -39,   10),  S( -49,   20),
        S( -80,  -30),  S( -56,  -15),  S( -62,    9),  S( -60,   16),
        S( -95,  -47),  S( -76,  -15),  S( -72,   -9),  S( -58,    4),
        S( -87,  -61),  S( -93,  -45),  S( -86,  -21),  S( -77,  -14),
        S(-130,  -92),  S(-103,  -55),  S( -88,  -58),  S( -95,  -39),
    ],

    BISHOP = [
        S( -88,  -54),  S(-114,  -46),  S(-106,  -47),  S(-104,  -49),
        S( -83,  -59),  S( -69,  -36),  S( -53,  -42),  S( -64,  -42),
        S( -53,  -56),  S( -45,  -40),  S( -53,  -25),  S( -41,  -34),
        S( -67,  -55),  S( -65,  -30),  S( -53,  -25),  S( -45,  -13),
        S( -68,  -55),  S( -67,  -34),  S( -66,  -19),  S( -55,  -13),
        S( -67,  -58),  S( -58,  -41),  S( -64,  -25),  S( -63,  -24),
        S( -56,  -75),  S( -55,  -49),  S( -52,  -53),  S( -70,  -38),
        S( -82,  -71),  S( -59,  -67),  S( -84,  -57),  S( -78,  -56),
    ],

    ROOK = [
        S(-169, -110),  S(-160, -111),  S(-167, -106),  S(-176, -104),
        S(-166, -105),  S(-165, -101),  S(-149, -104),  S(-144, -107),
        S(-170, -111),  S(-149, -111),  S(-146, -113),  S(-142, -116),
        S(-172, -116),  S(-163, -116),  S(-161, -112),  S(-154, -118),
        S(-190, -118),  S(-176, -118),  S(-181, -112),  S(-172, -116),
        S(-189, -126),  S(-173, -130),  S(-178, -123),  S(-174, -123),
        S(-200, -125),  S(-184, -130),  S(-179, -128),  S(-176, -131),
        S(-180, -130),  S(-178, -132),  S(-182, -129),  S(-168, -138),
    ],

    QUEEN = [
        S(-479,   60),  S(-460,   57),  S(-480,   70),  S(-483,   71),
        S(-461,   52),  S(-492,   85),  S(-478,   85),  S(-477,   96),
        S(-449,   36),  S(-450,   56),  S(-467,   84),  S(-462,   88),
        S(-449,   39),  S(-467,   76),  S(-465,   85),  S(-471,  100),
        S(-455,   55),  S(-453,   58),  S(-465,   79),  S(-466,   90),
        S(-457,   35),  S(-446,   40),  S(-450,   58),  S(-456,   65),
        S(-446,    9),  S(-445,   11),  S(-437,   11),  S(-442,   30),
        S(-453,   12),  S(-459,    7),  S(-460,    3),  S(-446,    4),
    ],

    KING = [
        S(  83, -104),  S(  67,  -25),  S(  23,  -10),  S( -18,   -3),
        S( 110,  -54),  S(  50,   23),  S(  16,   32),  S( -28,   38),
        S(  18,  -22),  S(  24,   31),  S(  14,   38),  S( -19,   46),
        S(  -7,  -30),  S(   5,   21),  S(  -5,   33),  S( -25,   40),
        S( -81,  -28),  S( -54,   10),  S( -40,   20),  S( -38,   27),
        S( -71,  -34),  S( -60,   -2),  S( -68,   10),  S( -61,   15),
        S( -25,  -43),  S( -32,  -18),  S( -50,   -5),  S( -60,   -2),
        S( -26,  -67),  S( -18,  -42),  S( -40,  -30),  S( -33,  -35),
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
        V(6), V(-1), V(9), V(3)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-7), V(-6), V(-3), V(-14)],
    MG_MOBILITY_CLOSED = [
        V(-3), V(2), V(-15), V(-5)],
    EG_MOBILITY_CLOSED = [
        V(21), V(4), V(32), V(-13)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(40), V(70), V(155), V(200), V(227)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(16), V(6), V(5)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(8)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(20), V(68)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(12), V(-2)], // [MG, EG]
    PASSED_PAWN_MG        = [V(7), V(-2), V(-15), V(-15), V(-5), V(-44)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(8), V(15), V(37), V(61), V(73), V(64)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-20), V(20), V(5), V(1), V(-5), V(-8)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-57), V(-34), V(-16), V(-10), V(-5), V(-5)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(6), V(-17)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-4), V(-2)], // [MG, EG]
    PHALANX_MG            = [V(-2), V(4), V(11), V(19), V(75), V(-186)], // by relative rank 2-7
    PHALANX_EG            = [V(3), V(0), V(10), V(42), V(45), V(211)], // by relative rank 2-7
    DEFENDED_PAWN_MG      = [CV(0), V(5), V(7), V(10), V(15), V(143)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG      = [CV(0), V(6), V(1), V(3), V(12), V(-64)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS = [V(-2), V(-5)], // [MG, EG]
    TEMPO_WEIGHTS         = [V(3), V(1)], // [MG, EG] — side-to-move initiative
}
