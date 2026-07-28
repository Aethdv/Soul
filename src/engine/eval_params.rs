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
    let freeze_resistant = name.starts_with("ATTACKER");

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

            // The six tables are one block; every square carries an MG and an EG slot.
            const PSQT_BLOCKS: &[(&str, usize)] = &[("psqt", (0 $( + $name.len() )*) * 2)];

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
    ($($block:ident = [$($val:expr),* $(,)?]),* $(,)?) => {
        paste::paste! {
            $(
                pub const [<$block:upper>]: [PhaseScore; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, eg) | Param::CS(mg, eg) => PhaseScore::new(mg, eg),
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<MG_ $block:upper>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(mg, _) | Param::CS(mg, _) => mg,
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];

                pub const [<EG_ $block:upper>]: [i32; { [$($val),*].len() }] = [
                    $(
                        match $val {
                            Param::S(_, eg) | Param::CS(_, eg) => eg,
                            _ => panic!("Simple params must be S or CS"),
                        }
                    ),*
                ];
            )*

            const SIMPLE_BLOCKS: &[(&str, usize)] = &[$( (stringify!($block), [<$block:upper>].len() * 2) ),*];

            fn collect_simple_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $(
                    for (i, param) in [<$block:upper>].iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.mg as f64,
                            name: format!("MG_{}[{i}]", stringify!([<$block:upper>])),
                            idx: 0,
                            is_fixed,
                            freeze_resistant: false,
                        });
                    }

                    for (i, param) in [<$block:upper>].iter().enumerate() {
                        let is_fixed = matches!([$($val),*][i], Param::CS(_, _));

                        params.push(Tunable {
                            value: param.eg as f64,
                            name: format!("EG_{}[{i}]", stringify!([<$block:upper>])),
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
    ($($block:ident { mg = [$($mg:expr),* $(,)?], eg = [$($eg:expr),* $(,)?] $(,)? }),* $(,)?) => {
        paste::paste! {
            $(
                pub const [<MG_ $block:upper>]: Vi32x4 = Vi32x4::new([
                    $(
                        match $mg {
                            Param::Val(v) | Param::Const(v) => v,
                            _ => 0,
                        }
                    ),*
                ]);

                pub const [<EG_ $block:upper>]: Vi32x4 = Vi32x4::new([
                    $(
                        match $eg {
                            Param::Val(v) | Param::Const(v) => v,
                            _ => 0,
                        }
                    ),*
                ]);
            )*

            const SIMD_BLOCKS: &[(&str, usize)] = &[
                $( (stringify!($block), [$($mg),*].len() + [$($eg),*].len()) ),*
            ];

            fn collect_simd_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $(
                    let mg = [$($mg),*];
                    let eg = [$($eg),*];

                    params.append(&mut collect_params_from_arrays(stringify!([<MG_ $block:upper>]), &mg));
                    params.append(&mut collect_params_from_arrays(stringify!([<EG_ $block:upper>]), &eg));
                )*

                params
            }
        }
    };
}

macro_rules! define_weight_params {
    ($($block:ident = [$($val:expr),* $(,)?]),* $(,)?) => {
        paste::paste! {
            $(pub const [<$block:upper>]: [i32; { [$($val),*].len() }] = [
                $(
                    match $val {
                        Param::Val(v) | Param::Const(v) => v,
                        _ => 0,
                    }
                ),*
            ];)*

            const WEIGHT_BLOCKS: &[(&str, usize)] = &[$( (stringify!($block), [<$block:upper>].len()) ),*];

            fn collect_weight_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $(
                    let arr = [$($val),*];
                    params.append(&mut collect_params_from_arrays(stringify!([<$block:upper>]), &arr));
                )*

                params
            }
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
            (w_king_danger,          Scalar, king_danger_offset,        0),
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

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Psqt,
    Simple,
    Simd,
    Weight,
}

/// One parameter block: a named, contiguous run of slots in the tunable vector.
#[derive(Clone, Copy)]
pub struct Block {
    pub name: &'static str,
    pub group: Group,
    pub offset: usize,
    pub len: usize,
}

/// The declaration groups in slot order, each with the collector that emits it.
/// `BLOCKS`, the layout accessors and the parameter vector all derive from the
/// four tables named here.
macro_rules! param_groups {
    ($macro:ident) => {
        $macro! {
            (Psqt,   PSQT_BLOCKS,   collect_psqt_params),
            (Simple, SIMPLE_BLOCKS, collect_simple_params),
            (Simd,   SIMD_BLOCKS,   collect_simd_params),
            (Weight, WEIGHT_BLOCKS, collect_weight_params),
        }
    };
}

macro_rules! block_sources {
    ($( ($group:ident, $blocks:ident, $collect:ident) ),* $(,)?) => { &[$( (Group::$group, $blocks) ),*] };
}

const BLOCK_SOURCES: &[(Group, &[(&str, usize)])] = param_groups!(block_sources);

const BLOCK_COUNT: usize = {
    let mut count = 0;
    let mut i = 0;

    while i < BLOCK_SOURCES.len() {
        count += BLOCK_SOURCES[i].1.len();

        i += 1;
    }

    count
};

const BLOCK_TABLE: [Block; BLOCK_COUNT] = {
    let mut table = [Block { name: "", group: Group::Psqt, offset: 0, len: 0 }; BLOCK_COUNT];
    let mut offset = 0;
    let mut next = 0;
    let mut s = 0;

    while s < BLOCK_SOURCES.len() {
        let (group, source) = BLOCK_SOURCES[s];
        let mut i = 0;

        while i < source.len() {
            let (name, len) = source[i];
            table[next] = Block { name, group, offset, len };
            offset += len;

            next += 1;
            i += 1;
        }

        s += 1;
    }

    table
};

/// Every parameter block in slot order, offsets prefix-summed over the groups.
/// `LAYOUT` is the same table under named accessors.
pub const BLOCKS: &[Block] = &BLOCK_TABLE;

const fn block_slots(blocks: &[(&str, usize)]) -> usize {
    let mut slots = 0;
    let mut i = 0;

    while i < blocks.len() {
        slots += blocks[i].1;

        i += 1;
    }

    slots
}

const fn str_eq(a: &str, b: &str) -> bool {
    let (a, b) = (a.as_bytes(), b.as_bytes());

    if a.len() != b.len() {
        return false;
    }

    let mut i = 0;

    while i < a.len() {
        if a[i] != b[i] {
            return false;
        }

        i += 1;
    }

    true
}

/// The named view of `BLOCKS`: generates the `Layout` struct
/// (`<name>_offset` / `<name>_len`) and takes each block's extent by position.
/// A name out of order against the declarations fails the build at the first
/// one that disagrees.
macro_rules! define_layout {
    ($( $name:ident ),* $(,)?) => {
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
                let mut acc = 0usize;
                let mut idx = 0usize;

                $(
                    assert!(
                        idx < BLOCKS.len(),
                        concat!("define_layout! runs past the declared blocks at `", stringify!($name), "`")
                    );
                    assert!(
                        str_eq(BLOCKS[idx].name, stringify!($name)),
                        concat!("define_layout! disagrees with the parameter declarations at `", stringify!($name), "`")
                    );

                    let [<$name _offset>] = BLOCKS[idx].offset;
                    let [<$name _len>] = BLOCKS[idx].len;

                    acc += [<$name _len>];
                    idx += 1;
                )*

                assert!(idx == BLOCKS.len(), "define_layout! is missing blocks that the parameter declarations emit");

                Layout { $( [<$name _offset>], [<$name _len>], )* total: acc }
            };
        }
    };
}

define_layout! {
    psqt,
    material,
    mobility_open,
    mobility_closed,
    phase,
    attacker,
    king_safety,
    xray,
    king_danger,
    bishop_pair,
    rook_open,
    passed_pawn_mg,
    passed_pawn_eg,
    enemy_king_dist_mg,
    enemy_king_dist_eg,
    doubled_pawn,
    isolated_pawn,
    phalanx_mg,
    phalanx_eg,
    defended_pawn_mg,
    defended_pawn_eg,
    backward_pawn,
    tempo,
    minor_behind_pawn,
}

/// Concatenates the groups into the parameter vector `LAYOUT` describes. A
/// collector that emits a different count than its own block table declares
/// would shift every slot after it, so each group is checked as it lands.
macro_rules! collect_groups {
    ($( ($group:ident, $blocks:ident, $collect:ident) ),* $(,)?) => {{
        let mut all = Vec::new();

        $(
            let group = $collect();
            assert_eq!(
                group.len(),
                block_slots($blocks),
                concat!(stringify!($collect), " and ", stringify!($blocks), " disagree on slot count")
            );

            for mut p in group {
                if p.name.starts_with("PHASE") {
                    assert!(p.is_fixed, "PHASE must be constant (CV); tuning phase is not supported.");
                }

                p.idx = all.len();
                all.push(p);
            }
        )*

        all
    }};
}

pub fn collect_parameters() -> Vec<Tunable> {
    param_groups!(collect_groups)
}

define_psqt_params! {
    // Files A-D (mirrored to E-H) × 8 ranks
    PAWN = [
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
        S( 143,  179),  S( 179,  136),  S( 162,  169),  S( 180,  141),
        S(  22,   71),  S(  33,   71),  S(  92,   19),  S(  73,   -9),
        S(  -9,   47),  S(   4,   37),  S(  16,   17),  S(  34,    1),
        S( -24,   35),  S( -18,   38),  S( -11,   29),  S(  12,   23),
        S( -33,   31),  S( -26,   33),  S( -27,   38),  S( -19,   38),
        S( -31,   38),  S( -17,   39),  S( -18,   46),  S( -25,   46),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-161,  -17),  S(-131,   87),  S( -84,  107),  S( -26,  108),
        S(  26,   83),  S(  36,  118),  S(  88,  117),  S(  64,  124),
        S(  39,  101),  S(  69,  121),  S(  99,  141),  S( 124,  142),
        S(  44,  115),  S(  36,  147),  S(  91,  159),  S(  80,  173),
        S(  11,  126),  S(  53,  137),  S(  46,  162),  S(  57,  176),
        S( -16,  112),  S(  25,  133),  S(  29,  143),  S(  46,  166),
        S( -12,  109),  S( -23,  118),  S(   7,  135),  S(  19,  142),
        S( -76,   80),  S( -41,   76),  S( -22,  106),  S( -16,  126),
    ],

    BISHOP = [
        S( -32,  116),  S( -54,  130),  S( -60,  120),  S(-112,  139),
        S(  31,  100),  S(  54,  123),  S(  46,  120),  S(  42,  118),
        S(  57,  115),  S(  61,  120),  S(  67,  132),  S(  77,  122),
        S(  29,  112),  S(  44,  127),  S(  66,  130),  S(  78,  143),
        S(  33,   99),  S(  35,  126),  S(  42,  136),  S(  60,  140),
        S(  29,  104),  S(  47,  115),  S(  42,  133),  S(  39,  135),
        S(  29,   88),  S(  45,   93),  S(  56,   94),  S(  29,  122),
        S(  17,   73),  S(  40,  107),  S(   4,   92),  S(  12,  113),
    ],

    ROOK = [
        S( -89,  166),  S(-128,  182),  S(-118,  186),  S(-126,  178),
        S( -67,  158),  S( -72,  168),  S( -44,  165),  S( -41,  157),
        S( -62,  155),  S( -18,  147),  S( -26,  145),  S( -27,  140),
        S( -67,  158),  S( -51,  151),  S( -48,  155),  S( -50,  147),
        S( -94,  159),  S( -76,  156),  S( -82,  157),  S( -67,  155),
        S( -88,  144),  S( -61,  131),  S( -75,  141),  S( -74,  148),
        S(-106,  146),  S( -79,  139),  S( -76,  139),  S( -76,  138),
        S( -83,  148),  S( -76,  143),  S( -88,  148),  S( -66,  133),
    ],

    QUEEN = [
        S(-402,  452),  S(-385,  462),  S(-404,  516),  S(-393,  520),
        S(-339,  501),  S(-399,  548),  S(-382,  564),  S(-413,  599),
        S(-334,  520),  S(-324,  510),  S(-361,  574),  S(-362,  582),
        S(-344,  534),  S(-369,  580),  S(-364,  564),  S(-384,  600),
        S(-363,  542),  S(-354,  556),  S(-375,  571),  S(-371,  582),
        S(-355,  498),  S(-354,  531),  S(-358,  550),  S(-364,  552),
        S(-343,  456),  S(-348,  461),  S(-340,  476),  S(-347,  515),
        S(-381,  476),  S(-379,  479),  S(-377,  474),  S(-360,  493),
    ],

    KING = [
        S(  12, -143),  S( -53,  -35),  S( -20,  -29),  S( -83,  -12),
        S(-169,   16),  S( -18,   49),  S( -58,   58),  S(  39,   29),
        S(-152,   16),  S(  23,   50),  S(  -1,   63),  S( -24,   69),
        S(-179,    6),  S( -70,   45),  S( -86,   69),  S(-143,   84),
        S(-172,  -13),  S( -93,   23),  S( -92,   49),  S(-141,   74),
        S( -91,  -31),  S( -27,   -6),  S( -61,   25),  S( -65,   39),
        S(  22,  -64),  S(  21,  -31),  S( -14,   -2),  S( -36,   12),
        S(  28, -126),  S(  42,  -80),  S(  12,  -47),  S(  15,  -56),
    ],
}

define_simple_params! {
    material = [
         CS(  92,  124), // Pawn
         CS( 373,  419), // Knight
         CS( 372,  462), // Bishop
         CS( 568,  867), // Rook
         CS(1160, 1468), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(5), V(-7), V(5), V(5)], // [mobility, battery, threats, xray threats]
        eg = [V(-6), V(-9), V(8), V(-7)],
    },
    mobility_closed {
        mg = [V(0), V(9), V(-18), V(-5)],
        eg = [V(28), V(13), V(44), V(-33)],
    },
}

define_weight_params! {
    phase              = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    attacker           = [CV(0), V(190), V(295), V(497), V(585), V(603)], // [0, 1, 2, 3, 4, 5] attackers × weak
    king_safety        = [V(24), V(12), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    xray               = [V(11)], // [Ortho King]
    king_danger        = [V(32)], // pressure curvature, over DANGER_SCALE
    bishop_pair        = [V(41), V(76)], // [MG, EG]
    rook_open          = [V(46), V(1)], // [MG, EG]
    passed_pawn_mg     = [V(-11), V(-25), V(-28), V(-5), V(-9), V(-61)], // by relative rank 1-6
    passed_pawn_eg     = [V(-49), V(-24), V(27), V(82), V(183), V(148)], // by relative rank 1-6
    enemy_king_dist_mg = [V(-114), V(27), V(12), V(8), V(6), V(0)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-53), V(0), V(42), V(56), V(69), V(77)], // enemy king→passer dist, 7 clamps to 6
    doubled_pawn       = [V(3), V(-48)], // [MG, EG]
    isolated_pawn      = [V(-9), V(-13)], // [MG, EG]
    phalanx_mg         = [V(7), V(17), V(29), V(65), V(174), V(-307)], // by relative rank 2-7
    phalanx_eg         = [V(-7), V(1), V(25), V(93), V(206), V(591)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(33), V(22), V(20), V(31), V(258)], // by relative rank 2-7 (rank 2 unreachable)
    defended_pawn_eg   = [CV(0), V(16), V(14), V(30), V(65), V(-7)], // by relative rank 2-7 (rank 2 unreachable)
    backward_pawn      = [V(-8), V(-17)], // [MG, EG]
    tempo              = [V(34), V(39)], // [MG, EG], side-to-move initiative
    minor_behind_pawn  = [V(14), V(34)], // [MG, EG]
}
