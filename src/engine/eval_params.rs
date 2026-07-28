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
        S(  41,  122),  S(  79,   76),  S(  62,  111),  S(  81,   81),
        S(  39,   56),  S(  51,   56),  S( 113,    1),  S(  92,  -28),
        S(   6,   31),  S(  20,   21),  S(  33,    0),  S(  51,  -17),
        S( -10,   18),  S(  -4,   22),  S(   4,   12),  S(  28,    6),
        S( -19,   14),  S( -12,   16),  S( -13,   21),  S(  -5,   22),
        S( -18,   21),  S(  -3,   23),  S(  -4,   29),  S( -11,   30),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S( -91,  -51),  S( -24,   49),  S(   2,   77),  S(  62,   79),
        S( 106,   55),  S( 117,   91),  S( 172,   91),  S( 147,   98),
        S( 121,   74),  S( 152,   95),  S( 184,  116),  S( 210,  118),
        S( 126,   89),  S( 117,  122),  S( 175,  136),  S( 164,  150),
        S(  91,  100),  S( 136,  112),  S( 128,  138),  S( 139,  154),
        S(  63,   85),  S( 106,  108),  S( 110,  119),  S( 128,  143),
        S(  67,   82),  S(  55,   92),  S(  86,  110),  S(  99,  117),
        S(  -3,   59),  S(  36,   48),  S(  57,   79),  S(  62,  100),
    ],

    BISHOP = [
        S(  17,   97),  S(   8,  108),  S( -13,  101),  S( -69,  120),
        S(  83,   79),  S( 107,  104),  S(  98,  102),  S(  94,   99),
        S( 110,   96),  S( 114,  101),  S( 120,  113),  S( 131,  103),
        S(  81,   92),  S(  96,  109),  S( 120,  112),  S( 133,  125),
        S(  85,   78),  S(  87,  108),  S(  94,  118),  S( 113,  122),
        S(  81,   84),  S(  99,   96),  S(  94,  115),  S(  91,  117),
        S(  81,   67),  S(  97,   72),  S( 109,   74),  S(  80,  103),
        S(  67,   51),  S(  92,   87),  S(  54,   71),  S(  62,   93),
    ],

    ROOK = [
        S(  92,  175),  S(  51,  192),  S(  61,  197),  S(  53,  188),
        S( 115,  167),  S( 110,  178),  S( 139,  175),  S( 143,  166),
        S( 121,  163),  S( 167,  156),  S( 159,  153),  S( 158,  147),
        S( 115,  167),  S( 132,  160),  S( 136,  163),  S( 133,  155),
        S(  86,  168),  S( 106,  165),  S( 100,  166),  S( 115,  163),
        S(  94,  152),  S( 122,  138),  S( 108,  149),  S( 108,  156),
        S(  74,  154),  S( 102,  146),  S( 106,  146),  S( 106,  145),
        S(  99,  156),  S( 106,  151),  S(  93,  157),  S( 117,  140),
    ],

    QUEEN = [
        S( 190,  428),  S( 209,  438),  S( 189,  495),  S( 199,  499),
        S( 257,  479),  S( 192,  529),  S( 211,  546),  S( 178,  583),
        S( 263,  499),  S( 272,  489),  S( 233,  557),  S( 232,  565),
        S( 252,  514),  S( 225,  563),  S( 230,  546),  S( 209,  585),
        S( 232,  523),  S( 241,  538),  S( 219,  554),  S( 223,  565),
        S( 240,  476),  S( 241,  511),  S( 237,  531),  S( 230,  534),
        S( 253,  432),  S( 247,  437),  S( 255,  453),  S( 248,  494),
        S( 212,  452),  S( 214,  456),  S( 216,  451),  S( 234,  471),
    ],

    KING = [
        S(  28, -109),  S( -20,   -9),  S(   2,    3),  S( -42,   17),
        S(-102,   41),  S(   5,   85),  S( -22,   92),  S(  47,   68),
        S( -92,   43),  S(  35,   89),  S(  19,  100),  S(   1,  105),
        S(-125,   32),  S( -37,   79),  S( -52,  104),  S(-111,  119),
        S(-145,   17),  S( -61,   56),  S( -59,   83),  S(-112,  110),
        S( -58,   -2),  S(   9,   24),  S( -27,   57),  S( -31,   72),
        S(  61,  -36),  S(  60,   -1),  S(  23,   29),  S(  -1,   43),
        S(  67, -103),  S(  82,  -54),  S(  50,  -18),  S(  54,  -28),
    ],
}

define_simple_params! {
    material = [
         S(  82,  150), // Pawn
         S( 315,  477), // Knight
         S( 344,  515), // Bishop
         S( 414,  919), // Rook
         S( 610, 1606), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(6), V(-7), V(6), V(5)], // [mobility, battery, threats, xray threats]
        eg = [V(-7), V(-10), V(8), V(-8)],
    },
    mobility_closed {
        mg = [V(1), V(9), V(-20), V(-6)],
        eg = [V(30), V(14), V(47), V(-36)],
    },
}

define_weight_params! {
    phase              = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    attacker           = [CV(0), V(200), V(311), V(525), V(620), V(637)], // [0, 1, 2, 3, 4, 5] attackers × weak
    king_safety        = [V(25), V(12), V(10)], // [Pawn Shield, Ortho Exp, Diag Exp]
    xray               = [V(12)], // [Ortho King]
    king_danger        = [V(0)], // pressure curvature, over DANGER_SCALE
    bishop_pair        = [V(44), V(80)], // [MG, EG]
    rook_open          = [V(49), V(1)], // [MG, EG]
    passed_pawn_mg     = [V(-7), V(-23), V(-26), V(-1), V(-4), V(66)], // by relative rank 1-6
    passed_pawn_eg     = [V(-65), V(-38), V(15), V(74), V(180), V(193)], // by relative rank 1-6
    enemy_king_dist_mg = [V(-124), V(25), V(9), V(4), V(2), V(-4)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-43), V(13), V(58), V(73), V(87), V(95)], // enemy king→passer dist, 7 clamps to 6
    doubled_pawn       = [V(3), V(-50)], // [MG, EG]
    isolated_pawn      = [V(-10), V(-14)], // [MG, EG]
    phalanx_mg         = [V(7), V(18), V(31), V(69), V(123), V(-210)], // by relative rank 2-7
    phalanx_eg         = [V(-8), V(1), V(26), V(98), V(235), V(419)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(35), V(23), V(20), V(32), V(184)], // by relative rank 2-7 (rank 2 unreachable)
    defended_pawn_eg   = [CV(0), V(17), V(15), V(32), V(69), V(17)], // by relative rank 2-7 (rank 2 unreachable)
    backward_pawn      = [V(-9), V(-18)], // [MG, EG]
    tempo              = [V(36), V(41)], // [MG, EG], side-to-move initiative
    minor_behind_pawn  = [V(15), V(36)], // [MG, EG]
}
