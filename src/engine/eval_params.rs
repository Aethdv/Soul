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
        S(  20,   98),  S(  51,   56),  S(  33,   87),  S(  46,   61),
        S(  14,   20),  S(  21,   19),  S(  74,  -29),  S(  55,  -55),
        S( -17,   -2),  S(  -8,  -13),  S(   3,  -31),  S(  17,  -46),
        S( -29,  -14),  S( -26,  -12),  S( -18,  -21),  S(   1,  -27),
        S( -36,  -18),  S( -31,  -18),  S( -30,  -14),  S( -22,  -13),
        S( -33,  -11),  S( -25,  -12),  S( -23,   -6),  S( -28,   -6),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-158, -124),  S(-104,  -38),  S( -88,  -14),  S( -41,  -11),
        S(  11,  -32),  S(  19,   -1),  S(  68,   -5),  S(  41,    2),
        S(  24,  -17),  S(  49,    0),  S(  75,   14),  S(  97,   15),
        S(  27,   -3),  S(  16,   22),  S(  66,   32),  S(  58,   45),
        S(  -4,    8),  S(  34,   16),  S(  24,   36),  S(  34,   49),
        S( -30,   -5),  S(   7,   11),  S(   9,   19),  S(  23,   38),
        S( -28,   -6),  S( -38,    0),  S( -12,   13),  S(   2,   15),
        S( -86,  -28),  S( -47,  -36),  S( -31,  -11),  S( -22,    7),
    ],

    BISHOP = [
        S( -52,   -1),  S( -73,   13),  S( -84,    4),  S(-138,   23),
        S(  -2,  -13),  S(  19,    6),  S(  10,    3),  S(  10,   -1),
        S(  23,    1),  S(  29,    2),  S(  32,   11),  S(  39,    1),
        S(  -4,   -2),  S(   8,   11),  S(  31,   10),  S(  44,   18),
        S(   3,  -14),  S(   1,    9),  S(   5,   15),  S(  29,   16),
        S(   1,   -9),  S(  18,    1),  S(  14,   13),  S(   9,   16),
        S(   2,  -24),  S(  16,  -21),  S(  24,  -21),  S(   1,    1),
        S(  -8,  -37),  S(  16,   -6),  S( -14,  -17),  S(  -9,   -2),
    ],

    ROOK = [
        S( -15,   14),  S( -56,   29),  S( -51,   34),  S( -63,   28),
        S(   9,    4),  S(   2,   14),  S(  24,   12),  S(  27,    3),
        S(  14,    1),  S(  53,   -6),  S(  42,   -8),  S(  42,  -12),
        S(   5,    5),  S(  19,   -2),  S(  21,    1),  S(  18,   -6),
        S( -20,    6),  S(  -3,    2),  S( -11,    3),  S(   0,    1),
        S( -15,   -8),  S(  11,  -22),  S(  -3,  -13),  S(  -3,   -7),
        S( -33,   -3),  S(  -7,  -12),  S(  -3,  -12),  S(   0,  -13),
        S( -14,   -2),  S(  -2,   -9),  S(  -8,   -5),  S(  13,  -20),
    ],

    QUEEN = [
        S( -20,  -74),  S( -14,  -61),  S( -32,  -11),  S( -30,   -1),
        S(  38,  -25),  S( -23,   17),  S( -12,   33),  S( -43,   66),
        S(  38,   -6),  S(  42,  -17),  S(   8,   41),  S(  -1,   51),
        S(  21,   10),  S(  -4,   51),  S(  -1,   32),  S( -22,   66),
        S(   1,   21),  S(   5,   30),  S( -14,   40),  S( -12,   48),
        S(  12,  -26),  S(   7,    8),  S(   1,   20),  S(  -7,   22),
        S(  24,  -65),  S(  10,  -53),  S(  15,  -39),  S(  12,  -11),
        S( -13,  -43),  S(  -2,  -48),  S(   2,  -54),  S(  13,  -28),
    ],

    KING = [
        S(  35, -125),  S(   4,  -37),  S(  17,  -26),  S( -26,  -13),
        S( -70,    4),  S(  25,   43),  S(  -6,   49),  S(  51,   29),
        S( -67,    9),  S(  46,   47),  S(  28,   59),  S(  15,   63),
        S(-101,    1),  S( -19,   39),  S( -32,   62),  S( -81,   74),
        S(-118,  -13),  S( -38,   18),  S( -36,   43),  S( -82,   66),
        S( -39,  -31),  S(  23,  -10),  S(  -8,   19),  S( -10,   32),
        S(  66,  -62),  S(  67,  -33),  S(  35,   -7),  S(  15,    6),
        S(  72, -121),  S(  87,  -80),  S(  63,  -49),  S(  72,  -56),
    ],
}

define_simple_params! {
    material = [
         S(  92,  163), // Pawn
         S( 360,  502), // Knight
         S( 371,  534), // Bishop
         S( 458,  947), // Rook
         S( 728, 1860), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(4), V(-5), V(4), V(4)], // [mobility, battery, threats, xray threats]
        eg = [V(-5), V(-8), V(9), V(-5)],
    },
    mobility_closed {
        mg = [V(1), V(9), V(-16), V(-6)],
        eg = [V(27), V(14), V(45), V(-28)],
    },
}

define_weight_params! {
    phase = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)]; // [P, N, B, R, Q, K]

    king_safety = [V(23), V(10), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    attacker    = [CV(0), V(185), V(288), V(476), V(562), V(581)], // [0, 1, 2, 3, 4, 5] attackers × weak
    xray        = [V(10)], // [Ortho King]
    king_danger = [V(0)]; // pressure curvature, over DANGER_SCALE; the floor at 0 holds the curvature where the data pulls negative

    tempo             = [V(32), V(37)], // [MG, EG], side-to-move initiative
    bishop_pair       = [V(39), V(74)], // [MG, EG]
    rook_open         = [V(40), V(0)], // [MG, EG]
    minor_behind_pawn = [V(12), V(32)], // [MG, EG]
    piece_mobility    = [V(5), V(6)]; // [MG, EG], mobility summed per piece rather than over the union

    doubled_pawn       = [V(1), V(-45)], // [MG, EG]
    isolated_pawn      = [V(-9), V(-12)], // [MG, EG]
    backward_pawn      = [V(-7), V(-16)], // [MG, EG]
    phalanx_mg         = [V(8), V(14), V(28), V(62), V(105), V(-155)], // by relative rank 2-7
    phalanx_eg         = [V(-7), V(2), V(23), V(88), V(211), V(343)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(33), V(22), V(20), V(32), V(156)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    defended_pawn_eg   = [CV(0), V(17), V(14), V(29), V(62), V(17)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    passed_pawn_mg     = [V(-6), V(-20), V(-22), V(1), V(-3), V(52)], // by relative rank 2-7
    passed_pawn_eg     = [V(-56), V(-33), V(14), V(67), V(161), V(154)], // by relative rank 2-7
    enemy_king_dist_mg = [V(-111), V(20), V(4), V(1), V(-1), V(-6)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-40), V(11), V(51), V(64), V(75), V(83)]; // enemy king→passer dist, 7 clamps to 6
}
