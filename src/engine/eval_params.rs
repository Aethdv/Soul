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
        S(  53,   97),  S(  88,   55),  S(  72,   87),  S(  89,   60),
        S(  44,   49),  S(  55,   48),  S( 111,   -2),  S(  93,  -28),
        S(  14,   25),  S(  26,   16),  S(  38,   -3),  S(  55,  -18),
        S(   0,   14),  S(   5,   17),  S(  12,    8),  S(  34,    3),
        S(  -9,   10),  S(  -3,   12),  S(  -4,   16),  S(   5,   17),
        S(  -7,   17),  S(   6,   18),  S(   5,   25),  S(  -2,   25),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S( -94,  -51),  S( -35,   41),  S( -12,   66),  S(  45,   67),
        S(  86,   45),  S(  96,   79),  S( 146,   78),  S( 123,   85),
        S(  99,   63),  S( 128,   82),  S( 157,  101),  S( 180,  103),
        S( 104,   76),  S(  95,  107),  S( 149,  119),  S( 139,  133),
        S(  72,   87),  S( 113,   98),  S( 106,  122),  S( 116,  135),
        S(  46,   73),  S(  85,   95),  S(  89,  104),  S( 105,  125),
        S(  50,   70),  S(  39,   79),  S(  68,   95),  S(  79,  103),
        S( -14,   48),  S(  22,   39),  S(  41,   68),  S(  45,   87),
    ],

    BISHOP = [
        S(  24,   85),  S(  15,   95),  S(  -3,   89),  S( -54,  106),
        S(  84,   69),  S( 106,   92),  S(  98,   89),  S(  95,   87),
        S( 109,   85),  S( 113,   89),  S( 118,  100),  S( 128,   91),
        S(  83,   81),  S(  96,   95),  S( 118,   99),  S( 130,  111),
        S(  86,   68),  S(  87,   95),  S(  95,  105),  S( 112,  108),
        S(  83,   74),  S(  99,   85),  S(  95,  102),  S(  92,  104),
        S(  83,   58),  S(  97,   63),  S( 108,   64),  S(  82,   91),
        S(  70,   44),  S(  93,   76),  S(  57,   62),  S(  65,   82),
    ],

    ROOK = [
        S(  90,  174),  S(  52,  189),  S(  61,  194),  S(  55,  185),
        S( 111,  166),  S( 105,  175),  S( 133,  174),  S( 136,  165),
        S( 115,  164),  S( 158,  156),  S( 150,  154),  S( 149,  148),
        S( 111,  166),  S( 126,  160),  S( 130,  163),  S( 127,  155),
        S(  85,  167),  S( 103,  165),  S(  97,  165),  S( 111,  163),
        S(  91,  152),  S( 116,  140),  S( 104,  150),  S( 105,  156),
        S(  74,  155),  S(  99,  147),  S( 103,  147),  S( 103,  146),
        S(  95,  156),  S( 102,  152),  S(  90,  157),  S( 112,  142),
    ],

    QUEEN = [
        S( 174,  393),  S( 190,  403),  S( 173,  455),  S( 183,  458),
        S( 235,  440),  S( 176,  485),  S( 193,  501),  S( 163,  535),
        S( 240,  458),  S( 248,  449),  S( 213,  511),  S( 212,  518),
        S( 230,  472),  S( 205,  516),  S( 210,  501),  S( 191,  536),
        S( 212,  480),  S( 220,  493),  S( 200,  508),  S( 204,  518),
        S( 219,  437),  S( 221,  469),  S( 216,  487),  S( 210,  489),
        S( 231,  396),  S( 226,  402),  S( 234,  416),  S( 227,  454),
        S( 195,  415),  S( 195,  418),  S( 198,  415),  S( 215,  433),
    ],

    KING = [
        S(  15,  -97),  S( -32,   -1),  S(  -9,    7),  S( -51,   20),
        S(-103,   41),  S(  -5,   82),  S( -31,   89),  S(  31,   66),
        S( -93,   43),  S(  24,   85),  S(   7,   95),  S(  -9,  100),
        S(-132,   35),  S( -45,   76),  S( -59,   99),  S(-113,  113),
        S(-144,   20),  S( -67,   55),  S( -65,   80),  S(-114,  105),
        S( -65,    3),  S(  -4,   26),  S( -35,   56),  S( -39,   70),
        S(  45,  -29),  S(  44,    4),  S(  10,   30),  S( -12,   44),
        S(  50,  -89),  S(  64,  -45),  S(  35,  -12),  S(  38,  -22),
    ],
}

define_simple_params! {
    material = [
         S(  66,  139), // Pawn
         S( 298,  439), // Knight
         S( 305,  473), // Bishop
         S( 372,  824), // Rook
         S( 555, 1461), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(5), V(-6), V(5), V(5)], // [mobility, battery, threats, xray threats]
        eg = [V(-6), V(-9), V(7), V(-7)],
    },
    mobility_closed {
        mg = [V(1), V(8), V(-18), V(-5)],
        eg = [V(27), V(13), V(43), V(-33)],
    },
}

define_weight_params! {
    phase              = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)], // [P, N, B, R, Q, K]
    attacker           = [CV(0), V(182), V(286), V(479), V(564), V(580)], // [0, 1, 2, 3, 4, 5] attackers × weak
    king_safety        = [V(23), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    xray               = [V(11)], // [Ortho King]
    king_danger        = [V(0)], // pressure curvature, over DANGER_SCALE
    bishop_pair        = [V(40), V(73)], // [MG, EG]
    rook_open          = [V(45), V(1)], // [MG, EG]
    passed_pawn_mg     = [V(-5), V(-18), V(-21), V(2), V(-2), V(55)], // by relative rank 1-6
    passed_pawn_eg     = [V(-66), V(-42), V(6), V(60), V(157), V(180)], // by relative rank 1-6
    enemy_king_dist_mg = [V(-115), V(20), V(5), V(1), V(0), V(-6)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-33), V(19), V(60), V(74), V(86), V(94)], // enemy king→passer dist, 7 clamps to 6
    doubled_pawn       = [V(3), V(-45)], // [MG, EG]
    isolated_pawn      = [V(-9), V(-13)], // [MG, EG]
    phalanx_mg         = [V(6), V(16), V(28), V(63), V(111), V(-190)], // by relative rank 2-7
    phalanx_eg         = [V(-7), V(1), V(24), V(89), V(214), V(382)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(32), V(21), V(18), V(29), V(166)], // by relative rank 2-7 (rank 2 unreachable)
    defended_pawn_eg   = [CV(0), V(15), V(14), V(29), V(63), V(15)], // by relative rank 2-7 (rank 2 unreachable)
    backward_pawn      = [V(-8), V(-16)], // [MG, EG]
    tempo              = [V(33), V(37)], // [MG, EG], side-to-move initiative
    minor_behind_pawn  = [V(14), V(33)], // [MG, EG]
}
