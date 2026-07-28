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
        S(  58,  107),  S(  97,   61),  S(  79,   96),  S(  98,   66),
        S(  48,   54),  S(  60,   53),  S( 122,   -2),  S( 102,  -31),
        S(  15,   28),  S(  29,   18),  S(  42,   -3),  S(  60,  -20),
        S(   0,   15),  S(   5,   19),  S(  13,    9),  S(  37,    3),
        S( -10,   11),  S(  -3,   13),  S(  -4,   18),  S(   5,   19),
        S(  -8,   19),  S(   7,   20),  S(   5,   27),  S(  -2,   27),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-103,  -56),  S( -39,   45),  S( -13,   73),  S(  50,   74),
        S(  95,   50),  S( 106,   87),  S( 161,   86),  S( 135,   94),
        S( 109,   69),  S( 141,   90),  S( 173,  111),  S( 198,  113),
        S( 114,   84),  S( 105,  118),  S( 164,  131),  S( 153,  146),
        S(  79,   96),  S( 124,  108),  S( 117,  134),  S( 128,  149),
        S(  51,   80),  S(  94,  104),  S(  98,  114),  S( 116,  138),
        S(  55,   77),  S(  43,   87),  S(  75,  105),  S(  87,  113),
        S( -15,   53),  S(  24,   43),  S(  45,   75),  S(  50,   96),
    ],

    BISHOP = [
        S(  26,   94),  S(  17,  105),  S(  -3,   98),  S( -59,  117),
        S(  92,   76),  S( 117,  101),  S( 108,   98),  S( 104,   96),
        S( 120,   93),  S( 124,   98),  S( 130,  110),  S( 141,  100),
        S(  91,   89),  S( 106,  105),  S( 130,  109),  S( 143,  122),
        S(  95,   75),  S(  96,  105),  S( 104,  115),  S( 123,  119),
        S(  91,   81),  S( 109,   93),  S( 104,  112),  S( 101,  114),
        S(  91,   64),  S( 107,   69),  S( 119,   70),  S(  90,  100),
        S(  77,   48),  S( 102,   84),  S(  63,   68),  S(  72,   90),
    ],

    ROOK = [
        S(  99,  191),  S(  57,  208),  S(  67,  213),  S(  60,  204),
        S( 122,  183),  S( 116,  193),  S( 146,  191),  S( 150,  182),
        S( 127,  180),  S( 174,  172),  S( 165,  169),  S( 164,  163),
        S( 122,  183),  S( 139,  176),  S( 143,  179),  S( 140,  171),
        S(  93,  184),  S( 113,  181),  S( 107,  182),  S( 122,  179),
        S( 100,  167),  S( 128,  154),  S( 114,  165),  S( 115,  172),
        S(  81,  170),  S( 109,  162),  S( 113,  162),  S( 113,  161),
        S( 105,  172),  S( 112,  167),  S(  99,  173),  S( 123,  156),
    ],

    QUEEN = [
        S( 191,  432),  S( 209,  443),  S( 190,  500),  S( 201,  504),
        S( 259,  484),  S( 194,  534),  S( 212,  551),  S( 179,  588),
        S( 264,  504),  S( 273,  494),  S( 234,  562),  S( 233,  570),
        S( 253,  519),  S( 226,  568),  S( 231,  551),  S( 210,  590),
        S( 233,  528),  S( 242,  542),  S( 220,  559),  S( 224,  570),
        S( 241,  481),  S( 243,  516),  S( 238,  536),  S( 231,  538),
        S( 254,  436),  S( 249,  442),  S( 257,  458),  S( 250,  499),
        S( 214,  457),  S( 215,  460),  S( 218,  456),  S( 236,  476),
    ],

    KING = [
        S(  17, -107),  S( -35,   -1),  S( -10,    8),  S( -56,   22),
        S(-113,   45),  S(  -6,   90),  S( -34,   98),  S(  34,   73),
        S(-102,   47),  S(  26,   93),  S(   8,  105),  S( -10,  110),
        S(-145,   39),  S( -50,   84),  S( -65,  109),  S(-124,  124),
        S(-158,   22),  S( -74,   61),  S( -72,   88),  S(-125,  115),
        S( -71,    3),  S(  -4,   29),  S( -39,   62),  S( -43,   77),
        S(  49,  -32),  S(  48,    4),  S(  11,   33),  S( -13,   48),
        S(  55,  -98),  S(  70,  -49),  S(  38,  -13),  S(  42,  -24),
    ],
}

define_simple_params! {
    material = [
         S(  73,  153), // Pawn
         S( 328,  483), // Knight
         S( 335,  520), // Bishop
         S( 409,  906), // Rook
         S( 611, 1607), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(6), V(-7), V(5), V(6)], // [mobility, battery, threats, xray threats]
        eg = [V(-7), V(-10), V(8), V(-8)],
    },
    mobility_closed {
        mg = [V(1), V(9), V(-20), V(-6)],
        eg = [V(30), V(14), V(47), V(-36)],
    },
}

define_weight_params! {
    phase              = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    attacker           = [CV(0), V(200), V(315), V(527), V(620), V(638)], // [0, 1, 2, 3, 4, 5] attackers × weak
    king_safety        = [V(25), V(12), V(10)], // [Pawn Shield, Ortho Exp, Diag Exp]
    xray               = [V(12)], // [Ortho King]
    king_danger        = [V(0)], // pressure curvature, over DANGER_SCALE
    bishop_pair        = [V(44), V(80)], // [MG, EG]
    rook_open          = [V(49), V(1)], // [MG, EG]
    passed_pawn_mg     = [V(-5), V(-20), V(-23), V(2), V(-2), V(61)], // by relative rank 1-6
    passed_pawn_eg     = [V(-73), V(-46), V(7), V(66), V(173), V(198)], // by relative rank 1-6
    enemy_king_dist_mg = [V(-127), V(22), V(6), V(1), V(0), V(-7)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-36), V(21), V(66), V(81), V(95), V(103)], // enemy king→passer dist, 7 clamps to 6
    doubled_pawn       = [V(3), V(-50)], // [MG, EG]
    isolated_pawn      = [V(-10), V(-14)], // [MG, EG]
    phalanx_mg         = [V(7), V(18), V(31), V(69), V(122), V(-209)], // by relative rank 2-7
    phalanx_eg         = [V(-8), V(1), V(26), V(98), V(235), V(420)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(35), V(23), V(20), V(32), V(183)], // by relative rank 2-7 (rank 2 unreachable)
    defended_pawn_eg   = [CV(0), V(17), V(15), V(32), V(69), V(17)], // by relative rank 2-7 (rank 2 unreachable)
    backward_pawn      = [V(-9), V(-18)], // [MG, EG]
    tempo              = [V(36), V(41)], // [MG, EG], side-to-move initiative
    minor_behind_pawn  = [V(15), V(36)], // [MG, EG]
}
