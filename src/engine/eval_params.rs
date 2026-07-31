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
    ($( ($name:ident, $ty:ident, $off:ident, $extra:expr, $konst:expr) ),* $(,)?) => {
        2usize $( + slot_width!($ty) )*
    };
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

/// Piece tables in the PSQT block, read off the material block's MG and EG halves.
pub const PIECE_TABLES: usize = LAYOUT.material_len / 2;

/// Half-board squares in one phase of one piece's table; also the MG-to-EG stride within it.
pub const TABLE_SQUARES: usize = LAYOUT.psqt_len / (2 * PIECE_TABLES);

// The record's PSQT gather and the tape's lane accumulation index the raw vector
// at pt · 64 + sq; off zero, those reads land in whatever took its place.
const _: () = assert!(LAYOUT.psqt_offset == 0, "PSQT must be the first block");

/// Total dual-AD inputs; the 2 accumulator lanes plus every tunable weight.
/// Drives `DUAL_N`, so the gradient array sizes itself as eval terms are added.
pub const DUAL_SLOTS: usize = crate::define_tunables!(count_dual_slots);

/// Every parameter block in slot order, offsets prefix-summed over the groups.
/// `LAYOUT` is the same table under named accessors.
pub const BLOCKS: &[Block] = &BLOCK_TABLE;

const BLOCK_SOURCES: &[(Group, SectionDecls)] = param_groups!(block_sources);

const BLOCK_COUNT: usize = {
    let mut count = 0;
    let mut g = 0;

    while g < BLOCK_SOURCES.len() {
        let sections = BLOCK_SOURCES[g].1;
        let mut s = 0;

        while s < sections.len() {
            count += sections[s].len();
            s += 1;
        }
        g += 1;
    }
    count
};

const BLOCK_TABLE: [Block; BLOCK_COUNT] = {
    let mut table = [Block { name: "", group: Group::Psqt, section: 0, offset: 0, len: 0 }; BLOCK_COUNT];
    let mut offset = 0;
    let mut next = 0;
    let mut g = 0;

    while g < BLOCK_SOURCES.len() {
        let (group, sections) = BLOCK_SOURCES[g];
        let mut s = 0;

        while s < sections.len() {
            let mut i = 0;

            while i < sections[s].len() {
                let (name, len) = sections[s][i];
                table[next] = Block { name, group, section: s as u8, offset, len };
                offset += len;

                next += 1;
                i += 1;
            }
            s += 1;
        }
        g += 1;
    }
    table
};

const fn block_slots(sections: SectionDecls) -> usize {
    let mut slots = 0;
    let mut s = 0;

    while s < sections.len() {
        let mut i = 0;

        while i < sections[s].len() {
            slots += sections[s][i].1;
            i += 1;
        }
        s += 1;
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

/// A declared block: its name and the slots it takes.
type BlockDecl = (&'static str, usize);

/// A group's blocks, in the sections its declaration breaks them into.
type SectionDecls = &'static [&'static [BlockDecl]];

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Group {
    Psqt,
    Simple,
    Simd,
    Weight,
}

/// One parameter block: a named, contiguous run of slots in the tunable vector.
/// `section` is which run of its group's declaration it sits in, so a paste-back
/// reprints the blank lines between them.
#[derive(Clone, Copy)]
pub struct Block {
    pub name: &'static str,
    pub group: Group,
    pub section: u8,
    pub offset: usize,
    pub len: usize,
}

#[derive(Debug, Clone)]
pub struct Tunable {
    pub name: String,
    pub value: f64,
    pub idx: usize,
    pub is_fixed: bool,
    pub freeze_resistant: bool,
}

/// S(mg, eg)  tapered score
/// V(v)       flat weight
///
/// The `C` prefix pins a slot; the tuner reads it as `is_fixed` and never steps it.
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

/// A tapered table's slots, every MG then every EG, named after the const they
/// came from. `shape` is the `S`/`CS` list the values were split out of and is all
/// that still distinguishes a tuned slot from a pinned one.
fn collect_phase_arrays<const N: usize>(name: &str, mg: &[i32; N], eg: &[i32; N], shape: &[Param; N]) -> Vec<Tunable> {
    let mut params = Vec::with_capacity(2 * N);

    for (prefix, values) in [("MG", mg), ("EG", eg)] {
        for (i, &value) in values.iter().enumerate() {
            let is_fixed = matches!(shape[i], Param::CS(..));

            params.push(Tunable {
                value: f64::from(value),
                name: format!("{prefix}_{name}[{i}]"),
                idx: 0,
                is_fixed,
                freeze_resistant: false,
            });
        }
    }
    params
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
            const PSQT_BLOCKS: SectionDecls = &[&[("psqt", (0 $( + [<MG_ $name>].len() )*) * 2)]];

            fn collect_psqt_params() -> Vec<Tunable> {
                let mut params = Vec::new();

                let names = [$(stringify!($name)),*];
                let expected = ["PAWN", "KNIGHT", "BISHOP", "ROOK", "QUEEN", "KING"];
                assert_eq!(names.len(), expected.len(), "Must define exactly 6 PSQT arrays");

                for (actual, expected) in names.iter().zip(expected.iter()) {
                    assert_eq!(actual, expected, "PSQT macro order MUST exactly match PieceType integer values");
                }

                $(
                    params.append(&mut collect_phase_arrays(
                        stringify!($name),
                        &[<MG_ $name>],
                        &[<EG_ $name>],
                        &[$($val),*],
                    ));
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

            const SIMPLE_BLOCKS: SectionDecls = &[&[$( (stringify!($block), [<MG_ $block:upper>].len() * 2) ),*]];

            fn collect_simple_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $(
                    params.append(&mut collect_phase_arrays(
                        stringify!([<$block:upper>]),
                        &[<MG_ $block:upper>],
                        &[<EG_ $block:upper>],
                        &[$($val),*],
                    ));
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

            const SIMD_BLOCKS: SectionDecls = &[&[
                $( (stringify!($block), [$($mg),*].len() + [$($eg),*].len()) ),*
            ]];

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

/// Rows are comma-separated; `;` closes a section. The paste block prints a blank
/// line between sections and gives each its own column width.
macro_rules! define_weight_params {
    ($( $($block:ident = [$($val:expr),* $(,)?]),* $(,)? ; )*) => {
        paste::paste! {
            $($(pub const [<$block:upper>]: [i32; { [$($val),*].len() }] = [
                $(
                    match $val {
                        Param::Val(v) | Param::Const(v) => v,
                        _ => 0,
                    }
                ),*
            ];)*)*

            const WEIGHT_BLOCKS: SectionDecls = &[
                $( &[$( (stringify!($block), [<$block:upper>].len()) ),*] ),*
            ];

            fn collect_weight_params() -> Vec<Tunable> {
                let mut params = Vec::new();
                $($(
                    let arr = [$($val),*];
                    params.append(&mut collect_params_from_arrays(stringify!([<$block:upper>]), &arr));
                )*)*
                params
            }
        }
    };
}

/// The core weights, then every bonus term the roster declares. Consumers see
/// one flat list in slot order and never learn that two declarations made it.
#[macro_export]
macro_rules! define_tunables {
    ($macro:ident) => {
        $crate::bonus_terms! { @tunables $macro,
            (mg_mob_open,   Vec4,   mobility_open_offset,   0, MG_MOBILITY_OPEN),
            (eg_mob_open,   Vec4,   mobility_open_offset,   4, EG_MOBILITY_OPEN),
            (mg_mob_closed, Vec4,   mobility_closed_offset, 0, MG_MOBILITY_CLOSED),
            (eg_mob_closed, Vec4,   mobility_closed_offset, 4, EG_MOBILITY_CLOSED),
            (w_shield,      Scalar, king_safety_offset,     0, KING_SAFETY[0]),
            (w_ortho,       Scalar, king_safety_offset,     1, KING_SAFETY[1]),
            (w_diag,        Scalar, king_safety_offset,     2, KING_SAFETY[2]),
            (atk_weights,   Array6, attacker_offset,        0, ATTACKER),
            (w_xray_ortho,  Scalar, xray_offset,            0, XRAY[0]),
            (w_king_danger, Scalar, king_danger_offset,     0, KING_DANGER[0]),
        }
    };
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

// The core blocks, then one per bonus term: a scalar owns a block under its own
// name, an array owns an MG and an EG block.
crate::bonus_terms! { @blocks define_layout,
    psqt,
    material,
    mobility_open,
    mobility_closed,
    phase,
    king_safety,
    attacker,
    xray,
    king_danger,
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

/// The values the declarations ship, in slot order: what a tuner starts from and what
/// every diagnostic measures the shipped eval at.
#[must_use]
pub fn default_values(params: &[Tunable]) -> Vec<f64> {
    params.iter().map(|p| p.value).collect()
}

define_psqt_params! {
    // Files A-D (mirrored to E-H) × 8 ranks
    PAWN = [
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
        S( 144,   20),  S(  69,   97),  S(  61,  120),  S(  70,   93),
        S(  58,  -15),  S( -14,    4),  S( -10,    3),  S(  28,  -51),
        S( -19,    1),  S( -10,   -8),  S( -12,  -27),  S(   6,  -34),
        S( -27,  -14),  S( -35,  -10),  S( -10,  -23),  S(   5,  -43),
        S( -41,  -34),  S( -29,  -26),  S( -30,  -24),  S( -37,  -23),
        S( -40,  -21),  S( -23,   -8),  S( -26,    3),  S( -72,   19),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-120, -193),  S(  62,  -81),  S(-106,    2),  S( -23,   -9),
        S( -34,  -19),  S( -63,   16),  S(   8,    8),  S(  16,   29),
        S(  44,  -40),  S(  34,   10),  S(  55,   44),  S(  71,   43),
        S(  20,   -7),  S(  10,   11),  S(  56,   21),  S(  61,    6),
        S( -18,  -19),  S(  21,   45),  S(  25,   44),  S(  18,   50),
        S( -47,   -9),  S(   6,   24),  S(   8,   43),  S(  20,   61),
        S( -45,   -1),  S( -21,  -31),  S( -19,   30),  S( -14,   46),
        S(  31,  -79),  S( -50,  -18),  S( -20,  -13),  S( -31,   -1),
    ],

    BISHOP = [
        S( -17,    7),  S(   9,  -84),  S(  -9,  -19),  S( -73,  -50),
        S( -29,  -35),  S( -17,   39),  S(   5,   -2),  S( -47,   20),
        S(  22,  -41),  S(  24,    8),  S(  48,   31),  S(  42,   42),
        S(  28,  -38),  S(  14,   19),  S(   8,   38),  S(  23,   67),
        S(  10,  -16),  S(  -3,    2),  S(   4,   45),  S(  19,   46),
        S( -16,  -19),  S(  30,    4),  S(  31,   26),  S(   2,   36),
        S( -19,  -41),  S(  16,  -15),  S(   9,   -1),  S(  -8,    5),
        S(  10,    6),  S(  -4,  -43),  S( -52,  -10),  S( -42,   -5),
    ],

    ROOK = [
        S( -49,   40),  S(   5,   20),  S(  48,   -4),  S(  51,  -13),
        S(  -1,   14),  S(   7,    3),  S(  33,  -13),  S( 148,  -64),
        S( -13,    6),  S(  20,    4),  S(  17,   16),  S(  42,   -2),
        S( -41,   18),  S( -23,    9),  S( -13,    8),  S(  11,   -9),
        S( -22,   11),  S(  -3,    6),  S(   7,    5),  S(  -4,    8),
        S( -32,    5),  S(  -4,   -8),  S(  -9,   -2),  S(  17,  -28),
        S( -75,   13),  S( -21,  -12),  S(  -2,  -11),  S( -18,  -15),
        S( -22,    0),  S( -19,    2),  S(  -5,   -2),  S(   4,  -25),
    ],

    QUEEN = [
        S( -56,   19),  S( -46,   27),  S(-108,   98),  S( -59,   63),
        S(  23,  -22),  S( -45,   35),  S( -11,   21),  S(  15,    6),
        S( -12,   42),  S(  11,   26),  S(  -1,   79),  S( -16,   87),
        S(  30,   -4),  S(  17,  -50),  S(   7,   55),  S(  -1,   64),
        S(   3,   20),  S(   5,   35),  S( -14,   50),  S(  -9,   82),
        S(   1,  -72),  S(  13,  -42),  S(  19,   19),  S(   6,   34),
        S(  45, -131),  S(  24, -110),  S(  38,  -90),  S(  13,  -19),
        S(  29,  -77),  S(  28,  -44),  S(  11, -127),  S(   9,  -84),
    ],

    KING = [
        S(-532,  217),  S( -80,   93),  S( 289,  -25),  S(  51,   19),
        S(-182,   79),  S( 136,    9),  S(  91,   20),  S(  23,   44),
        S( -88,  -19),  S( -47,   34),  S( -42,   -3),  S(-268,   19),
        S( -48,  -46),  S(   6,   18),  S( -59,   29),  S( -48,   20),
        S(   5,  -62),  S( -16,    4),  S(  24,   22),  S( -34,   42),
        S(  26,  -46),  S(  51,   -8),  S(  28,   15),  S(   0,   31),
        S( 131,  -74),  S( 105,  -40),  S(  76,  -19),  S(  29,    0),
        S( 106, -118),  S( 135, -102),  S(  89,  -89),  S(  84, -105),
    ],
}

define_simple_params! {
    material = [
         S( 109,  193), // Pawn
         S( 435,  436), // Knight
         S( 436,  499), // Bishop
         S( 571,  903), // Rook
         S(1404, 1505), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(14), V(-2), V(4), V(-10)], // [mobility, battery, threats, xray threats]
        eg = [V(-21), V(-25), V(3), V(-36)],
    },
    mobility_closed {
        mg = [V(-21), V(-2), V(-19), V(15)],
        eg = [V(92), V(73), V(111), V(37)],
    },
}

define_weight_params! {
    phase = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)]; // [P, N, B, R, Q, K]

    king_safety = [V(24), V(12), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    attacker    = [CV(0), V(154), V(154), V(361), V(387), V(501)], // [0, 1, 2, 3, 4, 5] attackers × weak
    xray        = [V(9)], // [Ortho King]
    king_danger = [V(0)]; // pressure curvature, over DANGER_SCALE; floored at 0, the data pulls under

    tempo             = [V(15), V(6)], // [MG, EG], side-to-move initiative
    bishop_pair       = [V(53), V(50)], // [MG, EG]
    rook_open         = [V(56), V(-22)], // [MG, EG]
    minor_behind_pawn = [V(7), V(52)]; // [MG, EG]

    doubled_pawn       = [V(-14), V(-43)], // [MG, EG]
    isolated_pawn      = [V(-11), V(-7)], // [MG, EG]
    backward_pawn      = [V(-10), V(-8)], // [MG, EG]
    phalanx_mg         = [V(-4), V(10), V(11), V(58), V(61), V(-9)], // by relative rank 2-7
    phalanx_eg         = [V(-9), V(10), V(22), V(78), V(252), V(477)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(25), V(11), V(27), V(90), V(386)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    defended_pawn_eg   = [CV(0), V(22), V(15), V(15), V(40), V(-64)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    passed_pawn_mg     = [V(5), V(-32), V(-49), V(-3), V(39), V(81)], // by relative rank 2-7
    passed_pawn_eg     = [V(-67), V(-27), V(32), V(76), V(150), V(96)], // by relative rank 2-7
    enemy_king_dist_mg = [V(-133), V(-29), V(-22), V(-1), V(-9), V(-1)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-46), V(51), V(89), V(94), V(106), V(109)]; // enemy king→passer dist, 7 clamps to 6
}
