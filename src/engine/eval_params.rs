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
        S( 133,  173),  S( 167,  132),  S( 151,  165),  S( 168,  138),
        S(  17,   62),  S(  30,   60),  S(  82,   13),  S(  64,  -14),
        S( -13,   38),  S(  -1,   29),  S(  11,   11),  S(  28,   -5),
        S( -27,   27),  S( -22,   31),  S( -15,   22),  S(   7,   16),
        S( -35,   23),  S( -29,   25),  S( -29,   30),  S( -22,   30),
        S( -34,   30),  S( -21,   31),  S( -22,   38),  S( -28,   37),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-172,  -34),  S(-129,   58),  S(-100,   84),  S( -42,   85),
        S(   5,   64),  S(  14,   95),  S(  66,   92),  S(  41,  100),
        S(  19,   75),  S(  46,   97),  S(  78,  113),  S( 102,  115),
        S(  24,   90),  S(  17,  118),  S(  67,  133),  S(  58,  146),
        S(  -8,  100),  S(  33,  112),  S(  26,  135),  S(  36,  148),
        S( -33,   85),  S(   6,  107),  S(  10,  117),  S(  24,  139),
        S( -31,   86),  S( -40,   93),  S( -12,  110),  S(   0,  114),
        S( -94,   65),  S( -57,   52),  S( -39,   80),  S( -33,  100),
    ],

    BISHOP = [
        S( -47,   89),  S( -72,  103),  S( -69,   90),  S(-125,  108),
        S(   8,   73),  S(  34,   92),  S(  25,   93),  S(  22,   90),
        S(  35,   86),  S(  41,   92),  S(  45,  104),  S(  54,   94),
        S(  10,   85),  S(  23,   99),  S(  44,  103),  S(  57,  114),
        S(  12,   72),  S(  15,   98),  S(  21,  109),  S(  40,  111),
        S(  10,   77),  S(  26,   89),  S(  22,  105),  S(  20,  106),
        S(  10,   61),  S(  24,   67),  S(  35,   68),  S(  10,   94),
        S(  -3,   50),  S(  22,   79),  S( -15,   66),  S(  -6,   84),
    ],

    ROOK = [
        S(-120,  119),  S(-147,  133),  S(-144,  137),  S(-143,  127),
        S( -90,  109),  S( -97,  119),  S( -67,  115),  S( -69,  109),
        S( -87,  106),  S( -45,   99),  S( -55,   98),  S( -54,   92),
        S( -91,  110),  S( -75,  102),  S( -73,  105),  S( -76,  101),
        S(-118,  112),  S( -99,  107),  S(-105,  110),  S( -92,  106),
        S(-114,   98),  S( -85,   83),  S( -99,   94),  S( -99,  101),
        S(-127,   98),  S(-105,   90),  S( -99,   91),  S(-101,   91),
        S(-107,  100),  S(-101,   96),  S(-112,  100),  S( -91,   85),
    ],

    QUEEN = [
        S(-442,  360),  S(-425,  375),  S(-440,  418),  S(-436,  427),
        S(-383,  408),  S(-438,  449),  S(-423,  471),  S(-452,  500),
        S(-375,  426),  S(-368,  415),  S(-402,  477),  S(-400,  482),
        S(-384,  436),  S(-407,  479),  S(-403,  468),  S(-423,  503),
        S(-403,  449),  S(-394,  456),  S(-414,  473),  S(-411,  484),
        S(-395,  407),  S(-394,  435),  S(-398,  453),  S(-403,  455),
        S(-383,  360),  S(-387,  369),  S(-381,  383),  S(-388,  421),
        S(-424,  388),  S(-417,  382),  S(-418,  384),  S(-400,  398),
    ],

    KING = [
        S(  37, -145),  S( -29,  -42),  S( -34,  -29),  S(-109,  -11),
        S(-170,   13),  S(  -2,   40),  S( -59,   48),  S(  28,   21),
        S(-133,    7),  S(   0,   47),  S(  -4,   53),  S( -28,   61),
        S(-169,    1),  S( -73,   37),  S( -76,   59),  S(-135,   72),
        S(-166,  -18),  S( -88,   15),  S( -87,   41),  S(-129,   63),
        S( -87,  -37),  S( -27,  -12),  S( -59,   17),  S( -62,   31),
        S(  20,  -66),  S(  18,  -35),  S( -14,   -8),  S( -36,    6),
        S(  25, -126),  S(  39,  -82),  S(   9,  -49),  S(  12,  -59),
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
        V(5), V(-7), V(5), V(5)], // [mobility, battery, threats, xray threats]
    EG_MOBILITY_OPEN = [
        V(-6), V(-9), V(7), V(-8)],
    MG_MOBILITY_CLOSED = [
        V(0), V(8), V(-20), V(-5)],
    EG_MOBILITY_CLOSED = [
        V(27), V(12), V(43), V(-29)],
}

define_weight_params! {
    PHASE_WEIGHTS             = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    ATTACKER_WEIGHTS          = [CV(0), V(190), V(280), V(474), V(558), V(582)], // [0, 1, 2, 3, 4, 5] attackers × weak
    KING_SAFETY_WEIGHTS       = [V(23), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    XRAY_WEIGHTS              = [V(10)], // [Ortho King]
    BISHOP_PAIR_WEIGHTS       = [V(39), V(74)], // [MG, EG]
    ROOK_OPEN_WEIGHTS         = [V(44), V(1)], // [MG, EG]
    PASSED_PAWN_MG            = [V(-19), V(-32), V(-36), V(-13), V(-15), V(-68)], // by relative rank 1-6
    PASSED_PAWN_EG            = [V(-49), V(-26), V(24), V(78), V(173), V(130)], // by relative rank 1-6
    ENEMY_KING_DIST_MG        = [V(-101), V(35), V(20), V(17), V(14), V(9)], // enemy king→passer dist, 7 clamps to 6
    ENEMY_KING_DIST_EG        = [V(-49), V(2), V(43), V(55), V(68), V(75)], // enemy king→passer dist, 7 clamps to 6
    DOUBLED_PAWN_WEIGHTS      = [V(2), V(-46)], // [MG, EG]
    ISOLATED_PAWN_WEIGHTS     = [V(-9), V(-12)], // [MG, EG]
    PHALANX_MG                = [V(7), V(16), V(27), V(62), V(146), V(-319)], // by relative rank 2-7
    PHALANX_EG                = [V(-7), V(1), V(24), V(90), V(202), V(563)], // by relative rank 2-7
    DEFENDED_PAWN_MG          = [CV(0), V(32), V(21), V(19), V(28), V(234)], // by relative rank 2-7 (rank 2 unreachable)
    DEFENDED_PAWN_EG          = [CV(0), V(16), V(14), V(29), V(63), V(-6)], // by relative rank 2-7 (rank 2 unreachable)
    BACKWARD_PAWN_WEIGHTS     = [V(-8), V(-17)], // [MG, EG]
    TEMPO_WEIGHTS             = [V(33), V(37)], // [MG, EG], side-to-move initiative
    MINOR_BEHIND_PAWN_WEIGHTS = [V(14), V(32)], // [MG, EG]
}
