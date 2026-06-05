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
        S(  95,   60),  S(  58,   79),  S(  72,   73),  S( 118,   58),
        S( -15,  -52),  S(  -7,  -55),  S(   4,  -58),  S(  20,  -66),
        S( -18,  -58),  S(  15,  -72),  S(  10,  -70),  S(  21,  -74),
        S( -28,  -61),  S(  -5,  -63),  S(   1,  -71),  S(  10,  -79),
        S( -32,  -58),  S(   1,  -70),  S( -16,  -63),  S( -27,  -67),
        S( -42,  -49),  S(  -4,  -60),  S( -13,  -53),  S( -52,  -44),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-185, -370),  S(-219, -299),  S(-246, -292),  S(-202, -299),
        S(-215, -311),  S(-238, -293),  S(-192, -299),  S(-240, -271),
        S(-209, -309),  S(-176, -298),  S(-220, -277),  S(-192, -284),
        S(-196, -310),  S(-206, -287),  S(-200, -281),  S(-200, -279),
        S(-219, -301),  S(-221, -289),  S(-209, -283),  S(-213, -279),
        S(-251, -311),  S(-218, -303),  S(-215, -300),  S(-208, -288),
        S(-251, -312),  S(-234, -301),  S(-232, -307),  S(-231, -298),
        S(-278, -310),  S(-266, -317),  S(-254, -309),  S(-242, -309),
    ],

    BISHOP = [
        S(-104, -382),  S(-242, -326),  S(-321, -317),  S(-229, -332),
        S(-228, -329),  S(-268, -312),  S(-217, -324),  S(-282, -313),
        S(-236, -317),  S(-195, -325),  S(-254, -306),  S(-216, -318),
        S(-211, -336),  S(-207, -318),  S(-218, -315),  S(-199, -316),
        S(-207, -336),  S(-226, -322),  S(-209, -317),  S(-207, -316),
        S(-225, -326),  S(-205, -329),  S(-210, -328),  S(-210, -319),
        S(-204, -336),  S(-201, -341),  S(-216, -332),  S(-220, -329),
        S(-225, -342),  S(-213, -332),  S(-243, -326),  S(-244, -334),
    ],

    ROOK = [
        S(-413, -578),  S(-408, -579),  S(-477, -559),  S(-481, -560),
        S(-340, -601),  S(-355, -593),  S(-342, -599),  S(-340, -597),
        S(-352, -603),  S(-335, -604),  S(-345, -598),  S(-327, -609),
        S(-350, -604),  S(-339, -605),  S(-344, -602),  S(-338, -606),
        S(-359, -610),  S(-361, -605),  S(-355, -606),  S(-352, -607),
        S(-379, -610),  S(-361, -612),  S(-364, -613),  S(-355, -616),
        S(-393, -610),  S(-372, -620),  S(-360, -623),  S(-356, -622),
        S(-369, -615),  S(-361, -612),  S(-354, -620),  S(-336, -629),
    ],

    QUEEN = [
        S(-911, -1022), S(-983, -947),  S(-1021, -921), S(-1148, -822),
        S(-969, -970),  S(-1017, -921), S(-999, -931),  S(-1068, -873),
        S(-974, -966),  S(-964, -981),  S(-991, -946),  S(-988, -945),
        S(-949, -1009), S(-978, -958),  S(-979, -956),  S(-983, -951),
        S(-969, -973),  S(-974, -966),  S(-974, -960),  S(-982, -949),
        S(-965, -989),  S(-958, -990),  S(-963, -987),  S(-964, -982),
        S(-952, -1025), S(-952, -1024), S(-949, -1030), S(-959, -1009),
        S(-966, -1011), S(-966, -1023), S(-976, -1010), S(-978, -998),
    ],

    KING = [
        S( -62,  -38),  S( -77,    6),  S( -91,    9),  S(   1,    6),
        S( -55,   -7),  S( -65,   35),  S( -23,   36),  S(  -6,   32),
        S( -43,   13),  S( -22,   23),  S( -16,   29),  S(  10,   21),
        S( -37,   -3),  S( -11,   10),  S( -35,   19),  S( -24,   19),
        S( -51,  -11),  S( -50,    4),  S( -49,   13),  S( -58,   20),
        S( -49,  -21),  S( -54,   -8),  S( -67,    3),  S( -69,    9),
        S( -12,  -34),  S( -18,  -27),  S( -52,   -8),  S( -57,   -6),
        S(  13,  -64),  S(  21,  -47),  S( -11,  -33),  S( -11,  -45),
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
        V(-1), V(-6), V(11), V(-1)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(0), V(3), V(12), V(7)],
    MG_MOBILITY_CLOSED = [
        V(8), V(5), V(-15), V(0)],
    EG_MOBILITY_CLOSED = [
        V(1), V(-13), V(-3), V(-53)],
}

define_weight_params! {
    PHASE_WEIGHTS         = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS      = [CV(0), V(8), V(18), V(65), V(62), V(108)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS   = [V(18), V(3), V(4)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS          = [V(7)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS   = [V(27), V(28)], // [MG, EG]
    ROOK_OPEN_WEIGHTS     = [V(17), V(-3)], // [MG, EG]
    PASSED_PAWN_MG        = [V(-47), V(-44), V(-27), V(-11), V(26), V(-33)], // by relative rank 1-6
    PASSED_PAWN_EG        = [V(16), V(25), V(38), V(47), V(51), V(-52)], // by relative rank 1-6
    ENEMY_KING_DIST_MG    = [V(-69), V(-16), V(-2), V(4), V(15), V(26)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG    = [V(-34), V(-13), V(-11), V(-10), V(-13), V(-11)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS  = [V(-23), V(-17)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS = [V(-3), V(-6)], // [MG, EG]
    PHALANX_MG            = [V(6), V(9), V(21), V(22), V(49), V(7)], // by relative rank 2-7
    PHALANX_EG            = [V(10), V(8), V(8), V(25), V(22), V(174)], // by relative rank 2-7
    DEFENDED_PAWN_MG      = [CV(0), V(22), V(13), V(21), V(33), V(69)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG      = [CV(0), V(7), V(6), V(12), V(22), V(-15)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS = [V(-9), V(-2)], // [MG, EG]
    TEMPO_WEIGHTS         = [V(27), V(6)], // [MG, EG] — side-to-move initiative
}
