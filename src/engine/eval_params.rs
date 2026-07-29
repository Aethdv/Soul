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
            const PSQT_BLOCKS: SectionDecls = &[&[("psqt", (0 $( + $name.len() )*) * 2)]];

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

            const SIMPLE_BLOCKS: SectionDecls = &[&[$( (stringify!($block), [<$block:upper>].len() * 2) ),*]];

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

/// Rows are comma-separated and every section closes with `;`, which the paste
/// block renders as a blank line and aligns its own column width against.
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
            (w_tempo_mg,             Scalar, tempo_offset,              0),
            (w_tempo_eg,             Scalar, tempo_offset,              1),
            (w_bp_mg,                Scalar, bishop_pair_offset,        0),
            (w_bp_eg,                Scalar, bishop_pair_offset,        1),
            (w_rook_open_mg,         Scalar, rook_open_offset,          0),
            (w_rook_open_eg,         Scalar, rook_open_offset,          1),
            (w_minor_behind_pawn_mg, Scalar, minor_behind_pawn_offset,  0),
            (w_minor_behind_pawn_eg, Scalar, minor_behind_pawn_offset,  1),
            (w_doubled_pawn_mg,      Scalar, doubled_pawn_offset,       0),
            (w_doubled_pawn_eg,      Scalar, doubled_pawn_offset,       1),
            (w_isolated_pawn_mg,     Scalar, isolated_pawn_offset,      0),
            (w_isolated_pawn_eg,     Scalar, isolated_pawn_offset,      1),
            (w_backward_pawn_mg,     Scalar, backward_pawn_offset,      0),
            (w_backward_pawn_eg,     Scalar, backward_pawn_offset,      1),
            (phalanx_mg,             Array6, phalanx_mg_offset,         0),
            (phalanx_eg,             Array6, phalanx_eg_offset,         0),
            (defended_pawn_mg,       Array6, defended_pawn_mg_offset,   0),
            (defended_pawn_eg,       Array6, defended_pawn_eg_offset,   0),
            (passed_pawn_mg,         Array6, passed_pawn_mg_offset,     0),
            (passed_pawn_eg,         Array6, passed_pawn_eg_offset,     0),
            (enemy_king_dist_mg,     Array6, enemy_king_dist_mg_offset, 0),
            (enemy_king_dist_eg,     Array6, enemy_king_dist_eg_offset, 0)
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
///
/// `section` numbers the visual runs a group's declaration is broken into, so the
/// paste block reproduces the grouping instead of flattening it on the next tune.
#[derive(Clone, Copy)]
pub struct Block {
    pub name: &'static str,
    pub group: Group,
    pub section: u8,
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

/// A declared block: the name it goes by and the slots it takes.
type BlockDecl = (&'static str, usize);

/// A group's blocks, in the sections its declaration breaks them into.
type SectionDecls = &'static [&'static [BlockDecl]];

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

/// Every parameter block in slot order, offsets prefix-summed over the groups.
/// `LAYOUT` is the same table under named accessors.
pub const BLOCKS: &[Block] = &BLOCK_TABLE;

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
    king_safety,
    attacker,
    xray,
    king_danger,
    tempo,
    bishop_pair,
    rook_open,
    minor_behind_pawn,
    doubled_pawn,
    isolated_pawn,
    backward_pawn,
    phalanx_mg,
    phalanx_eg,
    defended_pawn_mg,
    defended_pawn_eg,
    passed_pawn_mg,
    passed_pawn_eg,
    enemy_king_dist_mg,
    enemy_king_dist_eg,
}

// `FeatureRecord`'s PSQT gather and `tape.rs`'s lane accumulation both address the
// table as `pt · 64 + sq` against the raw parameter vector. Move the block off zero
// and every one of those reads lands in whatever took its place.
const _: () = assert!(LAYOUT.psqt_offset == 0, "PSQT must be the first block");

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
        S(  20,   78),  S(  55,   37),  S(  39,   68),  S(  56,   42),
        S(  11,   23),  S(  21,   23),  S(  77,  -26),  S(  59,  -52),
        S( -19,    1),  S(  -6,   -9),  S(   5,  -27),  S(  22,  -43),
        S( -33,  -11),  S( -28,   -7),  S( -20,  -16),  S(   1,  -22),
        S( -41,  -14),  S( -35,  -13),  S( -36,   -8),  S( -28,   -7),
        S( -40,   -8),  S( -26,   -7),  S( -28,   -1),  S( -34,    0),
        CS(  0,    0),  CS(  0,    0),  CS(  0,    0),  CS(  0,    0),
    ],

    KNIGHT = [
        S(-168, -132),  S(-111,  -41),  S( -86,  -16),  S( -35,  -14),
        S(   9,  -36),  S(  19,   -3),  S(  68,   -4),  S(  45,    3),
        S(  22,  -19),  S(  50,    0),  S(  79,   19),  S( 102,   20),
        S(  26,   -6),  S(  19,   24),  S(  71,   36),  S(  61,   49),
        S(  -5,    4),  S(  35,   15),  S(  29,   39),  S(  38,   52),
        S( -30,   -9),  S(   8,   11),  S(  12,   21),  S(  28,   42),
        S( -27,  -12),  S( -37,   -3),  S(  -9,   13),  S(   2,   19),
        S( -89,  -33),  S( -54,  -43),  S( -36,  -15),  S( -31,    5),
    ],

    BISHOP = [
        S( -58,    0),  S( -69,   11),  S( -85,    3),  S(-135,   21),
        S(   1,  -16),  S(  23,    6),  S(  15,    4),  S(  11,    2),
        S(  25,   -1),  S(  29,    3),  S(  35,   14),  S(  44,    5),
        S(  -1,   -5),  S(  13,   10),  S(  34,   13),  S(  46,   25),
        S(   3,  -17),  S(   4,   10),  S(  11,   19),  S(  29,   22),
        S(  -1,  -12),  S(  16,   -1),  S(  11,   16),  S(   9,   18),
        S(  -1,  -27),  S(  14,  -22),  S(  25,  -21),  S(  -1,    5),
        S( -13,  -41),  S(   9,   -9),  S( -25,  -23),  S( -17,   -4),
    ],

    ROOK = [
        S( -16,   12),  S( -53,   27),  S( -44,   32),  S( -51,   24),
        S(   5,    5),  S(   0,   14),  S(  26,   12),  S(  30,    4),
        S(  10,    2),  S(  51,   -5),  S(  44,   -7),  S(  43,  -13),
        S(   4,    5),  S(  20,   -1),  S(  23,    2),  S(  21,   -5),
        S( -21,    6),  S(  -3,    3),  S(  -9,    4),  S(   5,    2),
        S( -15,   -9),  S(  10,  -21),  S(  -2,  -11),  S(  -1,   -5),
        S( -32,   -6),  S(  -7,  -14),  S(  -4,  -14),  S(  -4,  -14),
        S( -10,   -5),  S(  -4,  -10),  S( -15,   -4),  S(   6,  -20),
    ],

    QUEEN = [
        S( -34,  -70),  S( -17,  -61),  S( -35,  -10),  S( -25,   -6),
        S(  27,  -24),  S( -31,   21),  S( -15,   36),  S( -44,   69),
        S(  31,   -6),  S(  40,  -15),  S(   5,   45),  S(   4,   53),
        S(  22,    7),  S(  -2,   51),  S(   2,   36),  S( -17,   70),
        S(   4,   15),  S(  12,   28),  S(  -8,   43),  S(  -4,   53),
        S(  11,  -27),  S(  13,    4),  S(   8,   23),  S(   2,   25),
        S(  23,  -67),  S(  18,  -62),  S(  25,  -47),  S(  19,  -11),
        S( -14,  -48),  S( -12,  -45),  S( -10,  -49),  S(   6,  -31),
    ],

    KING = [
        S(  39, -131),  S(  -3,  -37),  S(  18,  -28),  S( -22,  -16),
        S( -75,    5),  S(  23,   45),  S(  -3,   52),  S(  59,   29),
        S( -65,    6),  S(  50,   48),  S(  34,   59),  S(  18,   63),
        S(-109,    0),  S( -18,   40),  S( -32,   62),  S( -78,   75),
        S(-115,  -16),  S( -40,   19),  S( -38,   44),  S( -85,   67),
        S( -37,  -33),  S(  23,   -9),  S(  -9,   20),  S( -12,   34),
        S(  70,  -64),  S(  69,  -32),  S(  36,   -6),  S(  15,    8),
        S(  76, -123),  S(  89,  -80),  S(  61,  -48),  S(  64,  -57),
    ],
}

define_simple_params! {
    material = [
         S(  98,  162), // Pawn
         S( 370,  513), // Knight
         S( 382,  550), // Bishop
         S( 471,  970), // Rook
         S( 752, 1897), // Queen
         CS(   0,    0), // King
    ],
}

define_simd_params! {
    mobility_open {
        mg = [V(5), V(-7), V(5), V(5)], // [mobility, battery, threats, xray threats]
        eg = [V(-6), V(-9), V(7), V(-7)],
    },
    mobility_closed {
        mg = [V(0), V(8), V(-18), V(-5)],
        eg = [V(27), V(12), V(42), V(-32)],
    },
}

define_weight_params! {
    phase = [CV(0), CV(1), CV(1), CV(2), CV(4), CV(0)]; // [P, N, B, R, Q, K]

    king_safety = [V(23), V(11), V(9)], // [Pawn Shield, Ortho Exp, Diag Exp]
    attacker    = [CV(0), V(180), V(279), V(472), V(555), V(572)], // [0, 1, 2, 3, 4, 5] attackers × weak
    xray        = [V(11)], // [Ortho King]
    king_danger = [V(0)]; // pressure curvature, over DANGER_SCALE; floored at 0, the data pulls under

    tempo             = [V(33), V(37)], // [MG, EG], side-to-move initiative
    bishop_pair       = [V(39), V(72)], // [MG, EG]
    rook_open         = [V(44), V(1)], // [MG, EG]
    minor_behind_pawn = [V(14), V(32)]; // [MG, EG]

    doubled_pawn       = [V(3), V(-45)], // [MG, EG]
    isolated_pawn      = [V(-9), V(-12)], // [MG, EG]
    backward_pawn      = [V(-8), V(-16)], // [MG, EG]
    phalanx_mg         = [V(6), V(16), V(28), V(62), V(109), V(-189)], // by relative rank 2-7
    phalanx_eg         = [V(-7), V(1), V(24), V(88), V(211), V(378)], // by relative rank 2-7
    defended_pawn_mg   = [CV(0), V(32), V(21), V(18), V(28), V(165)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    defended_pawn_eg   = [CV(0), V(16), V(13), V(29), V(62), V(15)], // by relative rank 2-7; rank 2 needs a defender on rank 1
    passed_pawn_mg     = [V(-3), V(-17), V(-20), V(3), V(0), V(56)], // by relative rank 2-7
    passed_pawn_eg     = [V(-64), V(-40), V(8), V(61), V(156), V(172)], // by relative rank 2-7
    enemy_king_dist_mg = [V(-115), V(19), V(4), V(0), V(-2), V(-8)], // enemy king→passer dist, 7 clamps to 6
    enemy_king_dist_eg = [V(-33), V(18), V(58), V(71), V(83), V(91)]; // enemy king→passer dist, 7 clamps to 6
}
