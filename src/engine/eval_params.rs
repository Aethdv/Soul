//! The tunable parameter vector: its layout, and the weights it ships with.
//!
//! Weights are declared in groups, and one const-evaluated table assigns every block its
//! offset and length. `LAYOUT` is that table under named accessors, and the eval, the
//! gradient scatter and the tuner all index through it, so moving a block updates every
//! consumer at once.

#![allow(non_snake_case)]

use crate::weave::I32x4;

/// Slot count each `EvalParams` field consumes in the dual-AD gradient vector.
#[rustfmt::skip]
macro_rules! slot_width {
    (Scalar) => (1);
    (Vec4)   => (4);
    (Array4) => (4);
    (Array6) => (6);
    (Array14) => (14);
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
enum Param {
    S(i32, i32),
    CS(i32, i32),
    Val(i32),
    Const(i32),
}

const fn S(mg: i32, eg: i32) -> Param { Param::S(mg, eg) }
const fn CS(mg: i32, eg: i32) -> Param { Param::CS(mg, eg) }
const fn V(v: i32) -> Param { Param::Val(v) }
const fn CV(v: i32) -> Param { Param::Const(v) }

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
                pub const [<MG_ $block:upper>]: I32x4 = I32x4::new([
                    $(
                        match $mg {
                            Param::Val(v) | Param::Const(v) => v,
                            _ => 0,
                        }
                    ),*
                ]);

                pub const [<EG_ $block:upper>]: I32x4 = I32x4::new([
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

pub fn collect_parameters() -> Vec<Tunable> { param_groups!(collect_groups) }

#[must_use]
pub fn default_values(params: &[Tunable]) -> Vec<f64> { params.iter().map(|p| p.value).collect() }

define_psqt_params! {
    // Files A-D (mirrored to E-H) × 8 ranks
    PAWN = [
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
        S(  29,   96),  S(  56,   54),  S(  34,   85),  S(  43,   60),
        S(  21,   25),  S(  23,   24),  S(  71,  -23),  S(  50,  -49),
        S( -13,    0),  S( -11,  -11),  S(  -1,  -30),  S(  12,  -45),
        S( -25,  -13),  S( -28,  -13),  S( -21,  -22),  S(  -3,  -28),
        S( -32,  -17),  S( -32,  -19),  S( -33,  -15),  S( -24,  -15),
        S( -29,  -11),  S( -23,  -12),  S( -22,   -8),  S( -28,   -7),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-144, -121),  S(-120,  -30),  S( -86,  -13),  S( -29,  -13),
        S(  18,  -31),  S(  21,   -1),  S(  71,   -7),  S(  45,    1),
        S(  27,  -17),  S(  52,   -2),  S(  75,   10),  S(  93,   13),
        S(  32,   -3),  S(  15,   19),  S(  63,   29),  S(  59,   41),
        S(  -1,    9),  S(  29,   13),  S(  20,   33),  S(  30,   46),
        S( -30,   -3),  S(   0,    8),  S(  -5,   11),  S(  10,   32),
        S( -25,   -4),  S( -36,    1),  S( -16,    9),  S(  -5,   12),
        S( -83,  -23),  S( -41,  -30),  S( -33,  -11),  S( -22,    6),
    ],

    BISHOP = [
        S( -47,   -1),  S( -74,   14),  S( -83,    5),  S(-136,   24),
        S(  -1,  -11),  S(  17,    4),  S(   9,    2),  S(  11,   -2),
        S(  25,    1),  S(  29,    0),  S(  31,    6),  S(  34,   -2),
        S(  -2,   -1),  S(  10,    8),  S(  28,    5),  S(  39,   12),
        S(   4,  -13),  S(   0,    7),  S(   1,   11),  S(  25,    9),
        S(   4,   -8),  S(  13,    1),  S(   8,   10),  S(   8,   13),
        S(   7,  -21),  S(  14,  -21),  S(  24,  -20),  S(  -1,    2),
        S(  -3,  -33),  S(  21,   -2),  S(  -8,  -11),  S(  -4,   -1),
    ],

    ROOK = [
        S( -13,   12),  S( -54,   27),  S( -53,   34),  S( -68,   29),
        S(  13,    2),  S(   4,   13),  S(  23,   12),  S(  26,    3),
        S(  20,    1),  S(  56,   -5),  S(  46,   -6),  S(  47,  -10),
        S(   9,    8),  S(  21,    2),  S(  21,    4),  S(  21,   -2),
        S( -19,    9),  S(  -5,    6),  S( -14,    6),  S(  -3,    5),
        S( -16,   -6),  S(   7,  -21),  S( -10,  -12),  S( -10,   -6),
        S( -33,   -5),  S( -10,  -14),  S(  -6,  -14),  S(  -4,  -15),
        S( -15,   -6),  S(  -3,  -12),  S( -11,   -7),  S(  11,  -21),
    ],

    QUEEN = [
        S(  -9,  -78),  S( -10,  -60),  S( -30,   -9),  S( -28,    0),
        S(  47,  -27),  S( -20,   13),  S( -10,   29),  S( -40,   62),
        S(  45,   -2),  S(  47,  -17),  S(  14,   37),  S(   0,   50),
        S(  25,   16),  S(   0,   52),  S(  -2,   31),  S( -27,   64),
        S(   0,   27),  S(   3,   31),  S( -21,   39),  S( -17,   44),
        S(  11,  -23),  S(   1,    6),  S( -12,   18),  S( -17,   22),
        S(  28,  -64),  S(   3,  -52),  S(   8,  -38),  S(   3,   -9),
        S(  -9,  -43),  S(   5,  -47),  S(   4,  -52),  S(  13,  -22),
    ],

    KING = [
        S(  35, -122),  S(  11,  -32),  S(  21,  -23),  S( -21,  -11),
        S( -68,    5),  S(  33,   41),  S(  -4,   48),  S(  73,   24),
        S( -84,   12),  S(  72,   42),  S(  50,   54),  S(  30,   58),
        S(-129,    6),  S( -15,   37),  S( -30,   59),  S( -79,   72),
        S(-126,  -13),  S( -37,   17),  S( -35,   40),  S( -77,   63),
        S( -48,  -31),  S(  25,   -8),  S(  -9,   20),  S( -11,   32),
        S(  54,  -63),  S(  64,  -27),  S(  29,   -2),  S(   5,   11),
        S(  58, -126),  S(  82,  -79),  S(  53,  -47),  S(  63,  -56),
    ],
}

define_simple_params! {
    material = [
         S(  85,  159), // Pawn
         S( 348,  495), // Knight
         S( 357,  524), // Bishop
         S( 441,  927), // Rook
         S( 698, 1820), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(10), V(-5), V(5), V(3)], // [mobility, battery, threats, xray threats]
        eg = [V(-8), V(-7), V(4), V(-4)],
    },
    mobility_closed {
        mg = [V(1), V(9), V(-8), V(-8)],
        eg = [V(52), V(15), V(78), V(-36)],
    },
}

define_weight_params! {
    phase = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)]; // [P, N, B, R, Q, K]

    king_safety = [V(23), V(9), V(10)], // [Pawn Shield, Ortho Exp, Diag Exp]
    attacker    = [CV(0), V(147), V(250), V(441), V(530), V(546)], // [0..5] attacked king-zone squares × weak
    xray        = [V(10)], // [Ortho King]
    king_danger = [V(0)]; // pressure curvature, over DANGER_SCALE; the floor at 0 holds the curvature where the data pulls negative

    tempo             = [V(32), V(37)], // [MG, EG], side-to-move initiative
    bishop_pair       = [V(40), V(74)], // [MG, EG]
    rook_open         = [V(40), V(-2)], // [MG, EG]
    minor_behind_pawn = [V(11), V(32)]; // [MG, EG]

    doubled_pawn       = [V(1), V(-44)], // [MG, EG]
    isolated_pawn      = [V(-10), V(-12)], // [MG, EG]
    backward_pawn      = [V(-8), V(-18)], // [MG, EG]
    phalanx_mg         = [V(7), V(14), V(28), V(62), V(154), V(-153)], // by relative rank 2-7
    phalanx_eg         = [V(-6), V(3), V(24), V(87), V(194), V(325)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(31), V(19), V(18), V(31), V(242)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    defended_pawn_eg   = [CV(0), V(16), V(15), V(30), V(63), V(-10)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    passed_pawn_mg     = [V(-2), V(-15), V(-17), V(6), V(-2), V(43)], // by relative rank 2-7
    passed_pawn_eg     = [V(-52), V(-29), V(18), V(68), V(159), V(159)], // by relative rank 2-7
    enemy_king_dist_mg = [V(-112), V(17), V(-2), V(-5), V(-8), V(-11)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-41), V(9), V(49), V(62), V(73), V(79)], // enemy king→passer dist, 7 clamps to 6
    bishop_mobility_mg = [V(0), V(8), V(16), V(23), V(31), V(39), V(47), V(55), V(63), V(70), V(78), V(86), V(94), V(102)], // by mobility count 0-13
    bishop_mobility_eg = [V(0), V(7), V(13), V(20), V(26), V(33), V(39), V(46), V(52), V(59), V(65), V(72), V(78), V(85)]; // by mobility count 0-13
}
