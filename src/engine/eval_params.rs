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
        S( 109,  183),  S( 132,  146),  S( 122,  178),  S( 133,  155),
        S(  16,   66),  S(  24,   66),  S(  80,   17),  S(  62,  -11),
        S( -14,   43),  S(  -3,   34),  S(   9,   16),  S(  26,    1),
        S( -28,   31),  S( -23,   35),  S( -16,   27),  S(   4,   21),
        S( -36,   27),  S( -30,   30),  S( -31,   35),  S( -23,   35),
        S( -35,   34),  S( -22,   35),  S( -23,   42),  S( -30,   42),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-180,  -16),  S(-147,   87),  S( -94,  101),  S( -33,  104),
        S(   3,   83),  S(  11,  118),  S(  63,  117),  S(  37,  123),
        S(  13,  103),  S(  44,  123),  S(  71,  141),  S(  93,  143),
        S(  19,  118),  S(  11,  147),  S(  62,  160),  S(  51,  175),
        S( -13,  125),  S(  28,  138),  S(  20,  163),  S(  30,  177),
        S( -38,  113),  S(   0,  134),  S(   4,  144),  S(  19,  166),
        S( -35,  111),  S( -44,  119),  S( -17,  138),  S(  -5,  142),
        S( -94,   85),  S( -61,   78),  S( -43,  108),  S( -39,  127),
    ],

    BISHOP = [
        S( -50,  112),  S( -69,  127),  S( -77,  117),  S(-128,  134),
        S(   5,  100),  S(  28,  122),  S(  21,  118),  S(  19,  116),
        S(  31,  114),  S(  38,  120),  S(  39,  131),  S(  50,  121),
        S(   3,  111),  S(  19,  126),  S(  39,  130),  S(  51,  142),
        S(   9,  100),  S(  10,  126),  S(  17,  136),  S(  34,  138),
        S(   5,  105),  S(  21,  115),  S(  17,  132),  S(  15,  135),
        S(   6,   85),  S(  19,   94),  S(  29,   96),  S(   5,  122),
        S(  -5,   74),  S(  17,  106),  S( -19,   92),  S( -10,  113),
    ],

    ROOK = [
        S(-125,  165),  S(-154,  179),  S(-146,  184),  S(-149,  175),
        S(-100,  157),  S(-108,  167),  S( -78,  162),  S( -78,  157),
        S( -96,  153),  S( -54,  146),  S( -64,  144),  S( -65,  140),
        S(-104,  158),  S( -88,  151),  S( -83,  152),  S( -87,  147),
        S(-127,  157),  S(-110,  155),  S(-114,  156),  S(-101,  152),
        S(-122,  144),  S( -94,  131),  S(-108,  141),  S(-107,  148),
        S(-138,  147),  S(-114,  138),  S(-109,  138),  S(-110,  138),
        S(-117,  147),  S(-111,  143),  S(-122,  148),  S(-101,  133),
    ],

    QUEEN = [
        S(-454,  475),  S(-440,  485),  S(-455,  535),  S(-445,  534),
        S(-387,  510),  S(-444,  551),  S(-428,  574),  S(-465,  612),
        S(-381,  526),  S(-375,  522),  S(-409,  583),  S(-408,  588),
        S(-390,  539),  S(-414,  582),  S(-411,  570),  S(-430,  606),
        S(-408,  551),  S(-400,  562),  S(-418,  575),  S(-417,  587),
        S(-401,  507),  S(-399,  536),  S(-403,  556),  S(-408,  557),
        S(-388,  460),  S(-393,  474),  S(-387,  485),  S(-393,  523),
        S(-424,  485),  S(-420,  481),  S(-421,  481),  S(-405,  500),
    ],

    KING = [
        S(  12, -142),  S( -25,  -44),  S( -20,  -34),  S( -87,  -12),
        S(-149,   10),  S(  -9,   42),  S( -46,   51),  S(  41,   23),
        S(-141,   11),  S(   4,   50),  S(  -2,   58),  S( -43,   66),
        S(-179,    5),  S( -65,   40),  S( -78,   63),  S(-136,   77),
        S(-167,  -17),  S( -89,   19),  S( -90,   45),  S(-133,   67),
        S( -90,  -33),  S( -26,   -8),  S( -57,   21),  S( -59,   34),
        S(  21,  -63),  S(  21,  -32),  S( -13,   -4),  S( -34,    9),
        S(  26, -123),  S(  39,  -79),  S(  11,  -47),  S(  15,  -57),
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
        V(5), V(-7), V(4), V(6)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-9), V(7), V(-9)],
    MG_MOBILITY_CLOSED = [
        V(0), V(8), V(-17), V(-7)],
    EG_MOBILITY_CLOSED = [
        V(27), V(12), V(42), V(-29)],
}

define_weight_params! {
    PHASE_WEIGHTS             = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS          = [CV(0), V(180), V(280), V(475), V(565), V(587)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS       = [V(22), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS              = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS       = [V(37), V(77)], // [MG, EG]
    ROOK_OPEN_WEIGHTS         = [V(43), V(2)], // [MG, EG]
    PASSED_PAWN_MG            = [V(-12), V(-26), V(-31), V(-8), V(-13), V(-33)], // by relative rank 1-6
    PASSED_PAWN_EG            = [V(-47), V(-24), V(26), V(81), V(181), V(131)], // by relative rank 1-6
    ENEMY_KING_DIST_MG        = [V(-106), V(32), V(14), V(11), V(8), V(2)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG        = [V(-53), V(0), V(42), V(56), V(69), V(77)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS      = [V(2), V(-46)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS     = [V(-9), V(-13)], // [MG, EG]
    PHALANX_MG                = [V(7), V(16), V(27), V(61), V(153), V(-332)], // by relative rank 2-7
    PHALANX_EG                = [V(-7), V(1), V(25), V(92), V(215), V(573)], // by relative rank 2-7
    DEFENDED_PAWN_MG          = [CV(0), V(31), V(21), V(19), V(29), V(220)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG          = [CV(0), V(17), V(14), V(29), V(66), V(8)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS     = [V(-8), V(-17)], // [MG, EG]
    TEMPO_WEIGHTS             = [V(32), V(39)], // [MG, EG], side-to-move initiative
    MINOR_BEHIND_PAWN_WEIGHTS = [V(14), V(33)], // [MG, EG]
}
