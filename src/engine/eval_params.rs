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
        S(  73,  155),  S( 120,   98),  S(  98,  141),  S( 120,  105),
        S(  56,   67),  S(  70,   67),  S( 147,    0),  S( 122,  -36),
        S(  16,   36),  S(  33,   23),  S(  49,   -2),  S(  71,  -23),
        S(  -3,   20),  S(   4,   25),  S(  14,   13),  S(  43,    5),
        S( -15,   15),  S(  -7,   17),  S(  -8,   24),  S(   3,   25),
        S( -13,   24),  S(   5,   26),  S(   3,   34),  S(  -5,   34),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-120,  -70),  S( -58,   60),  S(  -4,   87),  S(  69,   90),
        S( 122,   61),  S( 136,  106),  S( 204,  105),  S( 172,  115),
        S( 140,   84),  S( 179,  110),  S( 218,  136),  S( 250,  139),
        S( 146,  103),  S( 135,  144),  S( 207,  161),  S( 194,  178),
        S( 103,  117),  S( 158,  132),  S( 149,  164),  S( 163,  183),
        S(  68,   98),  S( 122,  126),  S( 127,  140),  S( 149,  169),
        S(  73,   94),  S(  59,  107),  S(  98,  129),  S( 113,  138),
        S( -12,   64),  S(  36,   52),  S(  61,   91),  S(  68,  117),
    ],

    BISHOP = [
        S(  18,  118),  S(   1,  134),  S( -19,  123),  S( -83,  146),
        S(  99,   97),  S( 129,  127),  S( 119,  124),  S( 113,  121),
        S( 132,  117),  S( 138,  123),  S( 146,  138),  S( 159,  126),
        S(  97,  112),  S( 116,  133),  S( 145,  137),  S( 161,  153),
        S( 102,   96),  S( 104,  132),  S( 113,  145),  S( 137,  149),
        S(  97,  103),  S( 120,  117),  S( 113,  141),  S( 110,  143),
        S(  97,   82),  S( 117,   88),  S( 132,   89),  S(  96,  126),
        S(  80,   62),  S( 110,  106),  S(  63,   87),  S(  75,  114),
    ],

    ROOK = [
        S(  93,  196),  S(  43,  216),  S(  55,  222),  S(  45,  211),
        S( 122,  186),  S( 115,  198),  S( 152,  195),  S( 156,  184),
        S( 129,  181),  S( 186,  171),  S( 175,  168),  S( 174,  161),
        S( 122,  185),  S( 142,  177),  S( 147,  181),  S( 144,  171),
        S(  86,  187),  S( 111,  183),  S( 103,  184),  S( 122,  181),
        S(  95,  166),  S( 130,  150),  S( 112,  163),  S( 113,  172),
        S(  71,  170),  S( 106,  160),  S( 110,  160),  S( 110,  159),
        S( 101,  172),  S( 110,  165),  S(  94,  173),  S( 123,  152),
    ],

    QUEEN = [
        S( 168,  425),  S( 191,  438),  S( 166,  508),  S( 180,  514),
        S( 251,  488),  S( 171,  550),  S( 194,  571),  S( 153,  616),
        S( 257,  513),  S( 269,  501),  S( 221,  584),  S( 220,  594),
        S( 245,  531),  S( 211,  592),  S( 217,  571),  S( 191,  619),
        S( 220,  542),  S( 231,  560),  S( 204,  580),  S( 208,  594),
        S( 230,  485),  S( 232,  528),  S( 226,  553),  S( 218,  555),
        S( 246,  430),  S( 239,  437),  S( 249,  456),  S( 240,  507),
        S( 196,  455),  S( 198,  459),  S( 201,  453),  S( 223,  478),
    ],

    KING = [
        S(  61, -135),  S( -25,   -8),  S(  17,   -3),  S( -58,   18),
        S(-157,   55),  S(  19,  101),  S( -30,  113),  S(  88,   76),
        S(-139,   55),  S(  66,  104),  S(  39,  119),  S(  11,  126),
        S(-173,   42),  S( -42,   95),  S( -61,  126),  S(-127,  143),
        S(-168,   17),  S( -71,   66),  S( -68,  100),  S(-133,  133),
        S( -67,   -5),  S(  16,   27),  S( -28,   68),  S( -33,   86),
        S(  80,  -47),  S(  79,   -4),  S(  34,   33),  S(   4,   51),
        S(  88, -129),  S( 106,  -69),  S(  67,  -25),  S(  71,  -38),
    ],
}

define_simple_params! {
    material = [
         S(  92,  187), // Pawn
         S( 397,  594), // Knight
         S( 426,  636), // Bishop
         S( 530, 1152), // Rook
         S( 817, 2080), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(7), V(-9), V(7), V(7)], // [mobility, battery, threats, xray threats]
        eg = [V(-8), V(-12), V(10), V(-10)],
    },
    mobility_closed {
        mg = [V(1), V(11), V(-25), V(-7)],
        eg = [V(37), V(17), V(57), V(-43)],
    },
}

define_weight_params! {
    phase              = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    attacker           = [CV(0), V(250), V(385), V(645), V(760), V(784)], // [0, 1, 2, 3, 4, 5] attackers × weak
    king_safety        = [V(31), V(15), V(12)], // [Pawn Shield, Ortho Exp, Diag Exp]
    xray               = [V(15)], // [Ortho King]
    king_danger        = [V(0)], // pressure curvature, over DANGER_SCALE
    bishop_pair        = [V(54), V(99)], // [MG, EG]
    rook_open          = [V(60), V(1)], // [MG, EG]
    passed_pawn_mg     = [V(-8), V(-27), V(-31), V(0), V(-5), V(68)], // by relative rank 1-6
    passed_pawn_eg     = [V(-74), V(-41), V(24), V(97), V(229), V(236)], // by relative rank 1-6
    enemy_king_dist_mg = [V(-154), V(29), V(9), V(4), V(2), V(-7)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-59), V(11), V(66), V(84), V(101), V(111)], // enemy king→passer dist, 7 clamps to 6
    doubled_pawn       = [V(4), V(-62)], // [MG, EG]
    isolated_pawn      = [V(-12), V(-17)], // [MG, EG]
    phalanx_mg         = [V(9), V(22), V(38), V(85), V(204), V(-331)], // by relative rank 2-7
    phalanx_eg         = [V(-10), V(2), V(32), V(121), V(249), V(693)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(44), V(28), V(25), V(40), V(306)], // by relative rank 2-7 (rank 2 unreachable)
    defended_pawn_eg   = [CV(0), V(21), V(18), V(39), V(85), V(-2)], // by relative rank 2-7 (rank 2 unreachable)
    backward_pawn      = [V(-11), V(-22)], // [MG, EG]
    tempo              = [V(45), V(51)], // [MG, EG], side-to-move initiative
    minor_behind_pawn  = [V(19), V(44)], // [MG, EG]
}
