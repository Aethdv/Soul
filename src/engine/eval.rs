//! Hand-Crafted Evaluation.
//!
//! Computes the heuristic value of a leaf node. The evaluation relies on an
//! incrementally updated SIMD accumulator for piece-square material (PSQT),
//! combined with dynamically computed spatial features (mobility, king safety, threats).
//!
//! The function is generic over `EvalMath` (either `i32` or `DualNode`).
//! During search, `i32` monomorphizes into direct compiler-optimized arithmetic.
//! During tuning, `DualNode` carries each value's partials alongside it: one
//! evaluation pass yields the exact gradient for every parameter, no backward pass.

use crate::{
    core::{
        board::{Position, bitboard, spatial::SpatialTensor},
        defs::{Bitboard, Color, Direction, LANE_PHASE, PieceType, Square, TOTAL_PHASE},
    },
    engine::{
        autograd::EvalMath,
        combiner::{Accumulators, Combiner, CombinerParams, LinearCombiner, taper},
        eval_params::{
            self, ATTACKER, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_DANGER, KING_SAFETY, MG_MOBILITY_CLOSED, MG_MOBILITY_OPEN,
            XRAY,
        },
        mobility::{self, Mobility, MobilityData},
        search_params::SearchParams,
        term::{self},
    },
    weave::Vi16x8,
};

/// Non-PSQT evaluation weights, generic over the math type.
///
/// - For the search hot path: `EvalParams::<i32>::from_const()`
///   inlines to direct constant loads.
/// - For the tuner: `EvalParams::<DualNode>::load_tunable()` seeds each weight as a
///   dual variable (`grad[slot] = 1`), so a gradient flows back to its slot.
macro_rules! impl_eval_params {
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr, $konst:expr) ),* $(,)?) => {
        pub struct EvalParams<T: EvalMath> {
            $( pub $name: <T as EvalMath>::$ty, )*
        }

        impl EvalParams<i32> {
            /// Load from compile-time const arrays. The compiler inlines this entirely.
            #[inline(always)]
            pub fn from_const() -> Self {
                Self { $( $name: $konst, )* }
            }
        }

        impl<T: EvalMath<Scalar = T>> EvalParams<T> {
            // Tuner-only: the engine build seeds params via from_const, never this.
            #[allow(dead_code)]
            pub fn load_tunable(values: &[f64]) -> Self {
                // PSQT gradients occupy slots 0 (MG) and 1 (EG); tunable params start at 2.
                let mut slot = 2;
                paste::paste! {
                    Self {
                        $(
                            $name: T::[<load_ $ty:lower>](
                                values,
                                $crate::engine::eval_params::LAYOUT.$offset_field + $extra,
                                &mut slot,
                            ),
                        )*
                    }
                }
            }
        }
    };
}

crate::define_tunables! {impl_eval_params}

/// Every consumer works from this list. A term implemented in one place and
/// registered in another builds clean and stops being evaluated.
///
/// Rows reach a consumer behind a bracketed carry slot, so a list declared
/// elsewhere can chain in through `@tunables` or `@blocks` and have its own rows
/// arrive in front of them. Both rewrites live here rather than beside the
/// shapes they produce, because a consumer travels as a bare ident and resolves
/// in the scope that named it, while `$crate::` resolves from anywhere.
///
/// Every array term is six slots wide, so the width is matched as a literal
/// rather than captured: one declared wider has no arm and names itself at the
/// build, where a captured width would scatter past its own block.
#[macro_export]
macro_rules! bonus_terms {
    ($macro:ident $($carried:tt)*) => {
        $crate::bonus_terms! { @emit direct [$macro] [$($carried)*] }
    };

    (@tunables $macro:ident, $($carried:tt)*) => {
        $crate::bonus_terms! { @emit rewrite [$macro] [$($carried)*] }
    };

    (@blocks $macro:ident, $($carried:tt)*) => {
        $crate::bonus_terms! { @emit blocks [$macro] [$($carried)*] }
    };

    (@emit $mode:ident [$macro:ident] [$($carried:tt)*]) => {
        $crate::bonus_terms! { @$mode [$macro] [$($carried)*]
            tempo             = scalar(TempoTerm, tempo, tempo_mg, tempo_eg); // ~9 Elo
            bishop_pair       = scalar(BishopPairTerm, bishop_pair_diff, bishop_pair_mg, bishop_pair_eg); // ~9 Elo
            rook_open         = scalar(RookOpenTerm, rook_open_diff, rook_open_mg, rook_open_eg); // ~5 Elo
            minor_behind_pawn = scalar(MinorBehindPawnTerm, minor_behind_pawn_diff, minor_behind_pawn_mg, minor_behind_pawn_eg); // ~3 Elo
            piece_mobility    = scalar(PieceMobilityTerm, piece_mobility_diff, piece_mobility_mg, piece_mobility_eg);
            doubled_pawn      = scalar(DoubledPawnTerm, doubled_pawn_diff, doubled_pawn_mg, doubled_pawn_eg); // ~10 Elo
            isolated_pawn     = scalar(IsolatedPawnTerm, isolated_pawn_diff, isolated_pawn_mg, isolated_pawn_eg); // ~8 Elo
            backward_pawn     = scalar(BackwardPawnTerm, backward_pawn_diff, backward_pawn_mg, backward_pawn_eg); // ~13 Elo
            phalanx           = array(PhalanxTerm, phalanx, phalanx_mg, phalanx_eg, 6); // ~5 Elo
            defended_pawn     = array(DefendedPawnTerm, defended_pawn, defended_pawn_mg, defended_pawn_eg, 6); // ~10 Elo
            passed_pawn       = array(PassedPawnTerm, passed_pawn, passed_pawn_mg, passed_pawn_eg, 6); // ~15 Elo
            enemy_king_dist   = array(EnemyKingDistTerm, enemy_king_dist, enemy_king_dist_mg, enemy_king_dist_eg, 6); // ~12 Elo
        }
    };

    (@direct [$macro:ident] [$($carried:tt)*] $($rows:tt)*) => {
        $macro! { [$($carried)*] $($rows)* }
    };

    (@rewrite [$macro:ident] [$($out:tt)*]
        $block:ident = scalar($term:ident, $feature:ident, $mg:ident, $eg:ident); $($rest:tt)*
    ) => {
        $crate::bonus_terms! {
            @rewrite [$macro] [$($out)*
                ($mg, Scalar, [<$block _offset>], 0, $crate::engine::eval_params::[<$block:upper>][0]),
                ($eg, Scalar, [<$block _offset>], 1, $crate::engine::eval_params::[<$block:upper>][1]),
            ] $($rest)*
        }
    };

    (@rewrite [$macro:ident] [$($out:tt)*]
        $block:ident = array($term:ident, $feature:ident, $mg:ident, $eg:ident, 6); $($rest:tt)*
    ) => {
        $crate::bonus_terms! {
            @rewrite [$macro] [$($out)*
                ($mg, Array6, [<$block _mg_offset>], 0, $crate::engine::eval_params::[<$block:upper _MG>]),
                ($eg, Array6, [<$block _eg_offset>], 0, $crate::engine::eval_params::[<$block:upper _EG>]),
            ] $($rest)*
        }
    };

    (@rewrite [$macro:ident] [$($out:tt)*]) => { paste::paste! { $macro! { $($out)* } } };

    (@blocks [$macro:ident] [$($out:tt)*]
        $block:ident = scalar($term:ident, $feature:ident, $mg:ident, $eg:ident); $($rest:tt)*
    ) => {
        $crate::bonus_terms! { @blocks [$macro] [$($out)* $block,] $($rest)* }
    };

    (@blocks [$macro:ident] [$($out:tt)*]
        $block:ident = array($term:ident, $feature:ident, $mg:ident, $eg:ident, 6); $($rest:tt)*
    ) => {
        $crate::bonus_terms! { @blocks [$macro] [$($out)* [<$block _mg>], [<$block _eg>],] $($rest)* }
    };

    (@blocks [$macro:ident] [$($out:tt)*]) => { paste::paste! { $macro! { $($out)* } } };
}

macro_rules! register_bonus {
    ([] $( $block:ident = $kind:ident ( $term:ident, $($spec:tt)* ) ; )*) => {
        $( pub struct $term; )*

        crate::register_terms! {
            mobility::MobilityTerm   => mobility,
            mobility::KingSafetyTerm => king_safety,
            XrayTerm                 => xray,
            $( $term                 => bonus, )*
        }
    };
}

bonus_terms!(register_bonus);

pub struct XrayTerm;

pub struct DetailedEval {
    pub psqt: i32,
    pub mobility: i32,
    pub bonus: i32,
    pub safety: i32,
    pub total: i32,
}

/// Macroscopic features extracted once per position, consumed by both the
/// engine's score pass and every registered `LinearTerm::scatter` in the tuner.
/// New eval terms that depend on fresh board state should extend this struct,
/// not duplicate extraction.
pub struct SharedFeatures {
    pub openness: i32,
    pub data: MobilityData,
    pub xray_ortho: i32,
    /// Per-piece mobility minus the union it collapses to.
    pub piece_mobility_diff: i32,
    /// +1/0/−1 per side's `more_than_one()`.
    pub bishop_pair_diff: i32,
    /// Rooks on fully open files (no pawns of either color).
    pub rook_open_diff: i32,
    /// Bucketed by relative rank (rank 2 → index 0).
    pub passed_pawn: [i32; 6],
    /// Chebyshev distance (dist 1 → index 0, dist 6+ → index 5).
    pub enemy_king_dist: [i32; 6],
    /// Adjacent pairs only (gapped stacks go uncounted).
    pub doubled_pawn_diff: i32,
    pub isolated_pawn_diff: i32,
    /// Bucketed by relative rank.
    pub phalanx: [i32; 6],
    /// Bucketed by relative rank; index 0 (rank 2) unreachable (defender would sit on rank 1).
    pub defended_pawn: [i32; 6],
    pub backward_pawn_diff: i32,
    /// White-relative; the combiner flips it to STM-positive.
    pub tempo: i32,
    /// Minors with a pawn (either color) directly ahead.
    pub minor_behind_pawn_diff: i32,
}

/// The standard integer evaluation used in the alpha-beta search.
#[inline]
pub fn evaluate(board: &Position, acc: &Vi16x8) -> i32 {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    evaluate_generic::<i32>(board, acc, phase, &params, None)
}

/// A version of evaluate that returns individual components for debugging and visualization.
pub fn detailed_eval(board: &Position, acc: &Vi16x8) -> DetailedEval {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    let features = SharedFeatures::compute(board);
    let buckets = fill_accumulators::<i32>(acc, phase, &features, &params);
    let psqt = buckets.mg_eg;
    let mobility = buckets.mobility;
    let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
    let total = LinearCombiner::forward(&buckets, phase, &CombinerParams::from_eval(&params));
    let safety = total - psqt - mobility - bonus;

    let (p, m, b, s, t) = if board.stm == Color::White {
        (psqt, mobility, bonus, safety, total)
    } else {
        (-psqt, -mobility, -bonus, -safety, -total)
    };

    DetailedEval { psqt: p, mobility: m, bonus: b, safety: s, total: t }
}

/// Stripped-down eval for volatility filtering: accumulator-only, no spatial features.
#[inline(always)]
pub fn evaluate_fast(board: &Position, acc: &Vi16x8, phase: i32) -> i32 {
    let score = i32::tapered(acc, phase);
    if board.stm == Color::White { score } else { -score }
}

/// Volatility-scaled margin for lazy eval pruning; scales with game phase.
#[inline]
pub fn lazy_eval_margin(board: &Position, phase: i32, params: &SearchParams) -> i32 {
    let volatility = board.piece_count(PieceType::Pawn) * params.vol_pawn
        + board.piece_count(PieceType::Knight) * params.vol_knight
        + board.piece_count(PieceType::Bishop) * params.vol_bishop
        + board.piece_count(PieceType::Rook) * params.vol_rook
        + board.piece_count(PieceType::Queen) * params.vol_queen
        + board.piece_count(PieceType::King) * params.vol_king;

    let div = std::cmp::max(params.lazy_eval_divisor, 1);
    let scaled = (volatility * phase) / div;
    params.lazy_eval_margin + scaled
}

/// Monomorphized to `i32` for search, `DualNode` for tuning.
///
/// [`LinearTerm::scatter`] assumes `y = w · x`. A non-linear shape like
/// `feature · feature · weight` or `max(feature, 0)` produces invalid gradients;
/// put it in the [`Combiner`] layer. Run `make oracle` to verify.
///
/// TODO: `NonLinearTerm` trait.
#[inline(always)]
pub fn evaluate_generic<T: EvalMath<Scalar = T>>(
    board: &Position,
    acc: &T::Vec8,
    phase: T,
    params: &EvalParams<T>,
    features: Option<&SharedFeatures>,
) -> T {
    let features_store;
    let features = if let Some(f) = features {
        f
    } else {
        features_store = SharedFeatures::compute(board);
        &features_store
    };

    let score = compute_macro_eval(acc, phase, features, params);
    if board.stm == Color::White { score } else { -score }
}

#[inline(always)]
pub fn compute_macro_eval<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> T {
    let buckets = fill_accumulators::<T>(acc, phase, features, params);
    LinearCombiner::forward(&buckets, phase, &CombinerParams::from_eval(params))
}

/// Extracts the game phase directly from the accumulator's dedicated lane.
#[inline(always)]
pub fn extract_phase(acc: &Vi16x8) -> i32 {
    i32::from(acc.extract::<{ LANE_PHASE as i32 }>()).clamp(0, TOTAL_PHASE)
}

/// Cached on `pawn_key`; the passed-span scan is the hot part of
/// `SharedFeatures::compute`. Passer squares retained so
/// `enemy_king_dist` rebuilds without re-running the scan.
#[derive(Clone, Copy, Default)]
pub struct PawnFeatures {
    openness: i32,
    passed_pawn: [i32; 6],
    doubled_pawn_diff: i32,
    isolated_pawn_diff: i32,
    phalanx: [i32; 6],
    defended_pawn: [i32; 6],
    backward_pawn_diff: i32,
    w_passers: Bitboard,
    b_passers: Bitboard,
}

/// Per-search pawn-structure hash, keyed on `pawn_key`. Pawn structure barely
/// shifts walking the tree, so the hit rate is high and the scan collapses to
/// a probe.
pub struct PawnCache {
    entries: Box<[PawnEntry]>,
}

#[derive(Clone, Copy)]
struct PawnEntry {
    key: u64,
    pawn: PawnFeatures,
}

impl PawnFeatures {
    fn compute(board: &Position) -> Self {
        let openness = Mobility::compute_openness(board);

        let wp = board.pieces(PieceType::Pawn, Color::White);
        let bp = board.pieces(PieceType::Pawn, Color::Black);

        // Passed pawns; no enemy pawn on the file or adjacent files ahead. Passer squares
        // retained for the enemy-king distance bucket in SharedFeatures::with_pawn.
        let mut passed_pawn = [0i32; 6];
        let mut w_passers = Bitboard::default();
        let mut b_passers = Bitboard::default();

        for sq in wp {
            if (bitboard::passed_span(sq, Color::White) & bp).is_empty() {
                passed_pawn[(sq.rank() - 1) as usize] += 1;
                w_passers |= sq.bitboard();
            }
        }
        for sq in bp {
            if (bitboard::passed_span(sq, Color::Black) & wp).is_empty() {
                passed_pawn[(6 - sq.rank()) as usize] -= 1;
                b_passers |= sq.bitboard();
            }
        }

        // Doubled pawns; a pawn directly ahead of a friendly pawn on the same file.
        // & shift(North) marks the front pawn of each adjacent vertical pair;
        // gapped stacks go uncounted.
        let w_doubled = (wp & wp.shift(Direction::North)).popcount() as i32;
        let b_doubled = (bp & bp.shift(Direction::North)).popcount() as i32;
        let doubled_pawn_diff = w_doubled - b_doubled;

        // Isolated pawns; no friendly pawn on either adjacent file. file_fill smears
        // each pawn across its file; shifting east/west gives the neighbor-file mask.
        // Adjacency masks shared with backward-pawn detection.
        let w_adj = wp.file_fill().shift(Direction::East) | wp.file_fill().shift(Direction::West);
        let b_adj = bp.file_fill().shift(Direction::East) | bp.file_fill().shift(Direction::West);
        let w_isolated = (wp & !w_adj).popcount() as i32;
        let b_isolated = (bp & !b_adj).popcount() as i32;
        let isolated_pawn_diff = w_isolated - b_isolated;

        // Phalanx; side-by-side friendly pawns. & shift(East) marks the east pawn of each pair.
        let phalanx = by_relative_rank(wp & wp.shift(Direction::East), bp & bp.shift(Direction::East));

        // Defended; a pawn on a square its own side's pawns attack.
        // Pawn-attack maps shared with backward-pawn detection.
        let w_pawn_atk = board.pawn_attacks(Color::White);
        let b_pawn_atk = board.pawn_attacks(Color::Black);
        let defended_pawn = by_relative_rank(wp & w_pawn_atk, bp & b_pawn_atk);

        // Backward pawns; behind all neighbors with a stop square the enemy controls.
        // Isolated pawns are excluded by the adjacency mask (they score as isolated).
        // north_fill/south_fill then shift to adjacent files marks every square with a
        // friendly pawn at or behind its rank.
        let w_fill = wp.north_fill();
        let b_fill = bp.south_fill();
        let w_rear = w_fill.shift(Direction::East) | w_fill.shift(Direction::West);
        let b_rear = b_fill.shift(Direction::East) | b_fill.shift(Direction::West);
        let w_stop_bad = (bp | b_pawn_atk) >> 8; // stop square blocked or pawn-attacked
        let b_stop_bad = (wp | w_pawn_atk) << 8;
        let w_backward = (wp & w_adj & !w_rear & w_stop_bad).popcount() as i32;
        let b_backward = (bp & b_adj & !b_rear & b_stop_bad).popcount() as i32;
        let backward_pawn_diff = w_backward - b_backward;

        Self {
            openness,
            passed_pawn,
            doubled_pawn_diff,
            isolated_pawn_diff,
            phalanx,
            defended_pawn,
            backward_pawn_diff,
            w_passers,
            b_passers,
        }
    }
}

impl Default for PawnCache {
    fn default() -> Self {
        Self::new()
    }
}

impl PawnCache {
    const SIZE: usize = 1 << 14;

    pub fn new() -> Self {
        // u64::MAX sentinel; a real pawnless position hashes to 0, and must not
        // false-hit a fresh slot's default zero key.
        let empty = PawnEntry { key: u64::MAX, pawn: PawnFeatures::default() };
        Self { entries: vec![empty; Self::SIZE].into_boxed_slice() }
    }

    /// Probe by `pawn_key`, recomputing pawn structure on a miss.
    #[inline]
    pub fn probe(&mut self, board: &Position) -> PawnFeatures {
        let key = board.pawn_key;
        let slot = &mut self.entries[key as usize & (Self::SIZE - 1)];

        if slot.key == key {
            slot.pawn
        } else {
            let pawn = PawnFeatures::compute(board);
            *slot = PawnEntry { key, pawn };
            pawn
        }
    }
}

impl SharedFeatures {
    #[inline]
    pub fn compute(board: &Position) -> Self {
        Self::with_pawn(board, &PawnFeatures::compute(board))
    }

    /// The piece-dependent terms computed fresh, the pawn terms taken from
    /// `pawn`. `enemy_king_dist` is the one pawn-derived bucket rebuilt here;
    /// it moves with the enemy king, not the pawns.
    pub fn with_pawn(board: &Position, pawn: &PawnFeatures) -> Self {
        let pinned_w = board.pinned_pieces(Color::White);
        let pinned_b = board.pinned_pieces(Color::Black);
        let tensor = SpatialTensor::compute(board, pinned_w.0, pinned_b.0);

        let counts = [Color::White, Color::Black].map(|color| {
            let pinned = if color == Color::White { pinned_w } else { pinned_b };
            let ksq = board.pieces(PieceType::King, color).lsb();
            let area = !board.side_bb[color] & !board.pawn_attacks(color.opposite());
            doubled_control(board, color, pinned, ksq, area)
        });
        // Held inside the tuner's byte record, far outside what a legal position
        // reaches. Clamping a feature keeps the eval linear in its parameters.
        let piece_mobility_diff = (counts[0] - counts[1]).clamp(-127, 127);

        let data = Mobility::compute_all(board, &tensor, pinned_w, pinned_b);

        let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
        let w_king_ring = bitboard::atk_king(w_ksq).0;
        let b_king_ring = bitboard::atk_king(b_ksq).0;

        let xray_ortho =
            (tensor.w_ortho_xray() & b_king_ring).count_ones() as i32 - (tensor.b_ortho_xray() & w_king_ring).count_ones() as i32;

        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        let open = !board.role_bb[PieceType::Pawn].file_fill();
        let rooks_open = board.role_bb[PieceType::Rook] & open;
        let w_open = (rooks_open & board.side_bb[Color::White]).popcount() as i32;
        let b_open = (rooks_open & board.side_bb[Color::Black]).popcount() as i32;
        let rook_open_diff = w_open - b_open;

        // Enemy-king Chebyshev distance to each passer, bucketed (1..=6, 7 clamps
        // to 6). Walks the cached passer sets rather than re-detecting them.
        let mut enemy_king_dist = [0i32; 6];

        for sq in pawn.w_passers {
            enemy_king_dist[(b_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] += 1;
        }
        for sq in pawn.b_passers {
            enemy_king_dist[(w_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] -= 1;
        }

        // Minor behind pawn; a knight or bishop shielded by a pawn (either color)
        // directly ahead. Shifting all pawns toward us drops a front pawn onto the
        // minor's own square, so the AND counts shielded minors.
        let minors = board.role_bb[PieceType::Knight] | board.role_bb[PieceType::Bishop];
        let w_minors = minors & board.side_bb[Color::White];
        let b_minors = minors & board.side_bb[Color::Black];
        let all_pawns = board.role_bb[PieceType::Pawn];
        let w_minor_behind = (w_minors & all_pawns.shift(Direction::South)).popcount() as i32;
        let b_minor_behind = (b_minors & all_pawns.shift(Direction::North)).popcount() as i32;
        let minor_behind_pawn_diff = w_minor_behind - b_minor_behind;

        let tempo = if board.stm == Color::White { 1 } else { -1 };

        Self {
            openness: pawn.openness,
            data,
            xray_ortho,
            piece_mobility_diff,
            bishop_pair_diff,
            rook_open_diff,
            passed_pawn: pawn.passed_pawn,
            enemy_king_dist,
            doubled_pawn_diff: pawn.doubled_pawn_diff,
            isolated_pawn_diff: pawn.isolated_pawn_diff,
            phalanx: pawn.phalanx,
            defended_pawn: pawn.defended_pawn,
            backward_pawn_diff: pawn.backward_pawn_diff,
            tempo,
            minor_behind_pawn_diff,
        }
    }
}

/// Zero all buckets, seed PSQT from the SIMD accumulator, then `apply_all_terms`.
/// Isolated so both `compute_macro_eval` and `detailed_eval` produce identical values.
#[inline]
pub fn fill_accumulators<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> Accumulators<T> {
    let mut buckets = Accumulators::<T> {
        mg_eg: T::tapered(acc, phase),
        mobility: T::zero(),
        bonus_mg: T::zero(),
        bonus_eg: T::zero(),
        safety_us: T::zero(),
        safety_them: T::zero(),
        danger_us: T::zero(),
        danger_them: T::zero(),
        xray: T::zero(),
    };

    apply_all_terms::<T>(features, params, phase, &mut buckets);
    buckets
}

impl term::LinearTerm for XrayTerm {
    /// Scalar; x-ray is tapered MG-only inside the combiner's king-safety block.
    type Upstream = f64;
    type Input = f64;

    #[inline(always)]
    fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
        acc.xray = params.w_xray_ortho * T::from_i32(features.xray_ortho);
    }

    #[inline(always)]
    fn apply_input(feature: f64, values: &[f64], _phase: f64, acc: &mut Accumulators<f64>) {
        acc.xray = values[eval_params::LAYOUT.xray_offset] * feature;
    }

    #[inline(always)]
    fn scatter(feature: f64, upstream: f64, grads: &mut [f64]) {
        let off = eval_params::LAYOUT.xray_offset;
        grads[off] += upstream * feature;
    }
}

impl term::TermSource<XrayTerm> for SharedFeatures {
    type Input = f64;

    #[inline(always)]
    fn extract(&self) -> f64 {
        self.xray_ortho as f64
    }
}

/// How much of a side's control is doubled: the per-piece mobility sum minus the
/// union it collapses to.
///
/// The mobility term scores the union, which is all a setwise fill can give:
/// ORing a side's attacks together loses which piece reached where, so a square
/// two pieces both cover counts once. This is the remainder, orthogonal to the
/// term beside it instead of collinear with it.
///
/// Rebuilt per call rather than read off a maintained attack store, because the
/// full eval runs on barely half of nodes and only sliders need a probe at all.
///
/// Pins bind as they do in the tensor: a pinned slider may use only its pin ray,
/// a pinned knight nothing.
fn doubled_control(board: &Position, color: Color, pinned: Bitboard, ksq: Square, area: Bitboard) -> i32 {
    let us = board.side_bb[color];
    let occ = board.occupancy();
    let mut summed = 0;
    let mut union = board.pawn_attacks(color) & area;

    let count = |atk: Bitboard, summed: &mut i32, union: &mut Bitboard| {
        let reach = atk & area;
        *summed += reach.popcount() as i32;
        *union |= reach;
    };

    for sq in (board.role_bb[PieceType::Bishop] | board.role_bb[PieceType::Queen]) & us {
        let atk = bitboard::atk_bishop(sq, occ);
        count(if pinned.check_bit(sq) { atk & bitboard::line_bb(ksq, sq) } else { atk }, &mut summed, &mut union);
    }

    for sq in (board.role_bb[PieceType::Rook] | board.role_bb[PieceType::Queen]) & us {
        let atk = bitboard::atk_rook(sq, occ);
        count(if pinned.check_bit(sq) { atk & bitboard::line_bb(ksq, sq) } else { atk }, &mut summed, &mut union);
    }

    for sq in (board.role_bb[PieceType::Knight] & us) & !pinned {
        count(bitboard::atk_knight(sq), &mut summed, &mut union);
    }

    summed + (board.pawn_attacks(color) & area).popcount() as i32 - union.popcount() as i32
}

/// Generates `LinearTerm` + `TermSource for SharedFeatures` for a tapered bonus.
/// `scalar` writes one `(mg, eg)` slot pair; `array` writes MG/EG blocks of `$n` slots.
macro_rules! tapered_bonus_term {
    ( [] $( $block:ident = $kind:ident ( $($spec:tt)* ) ; )* ) => {
        $( tapered_bonus_term!(@$kind $block, $($spec)*); )*
    };

    (@scalar $block:ident, $term:ident, $sf_field:ident, $mg:ident, $eg:ident) => {
        impl term::LinearTerm for $term {
            type Upstream = term::TaperPair;
            type Input = f64;

            #[inline(always)]
            fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
                let feature = T::from_i32(features.$sf_field);
                acc.bonus_mg += params.$mg * feature;
                acc.bonus_eg += params.$eg * feature;
            }

            #[inline(always)]
            fn apply_input(feature: f64, values: &[f64], _phase: f64, acc: &mut Accumulators<f64>) {
                let off = paste::paste!(eval_params::LAYOUT.[<$block _offset>]);
                acc.bonus_mg += values[off] * feature;
                acc.bonus_eg += values[off + 1] * feature;
            }

            #[inline(always)]
            fn scatter(feature: f64, upstream: term::TaperPair, grads: &mut [f64]) {
                let off = paste::paste!(eval_params::LAYOUT.[<$block _offset>]);
                grads[off] += upstream.d_mg * feature;
                grads[off + 1] += upstream.d_eg * feature;
            }
        }

        impl term::TermSource<$term> for SharedFeatures {
            type Input = f64;

            #[inline(always)]
            fn extract(&self) -> f64 { self.$sf_field as f64 }
        }
    };

    (@array $block:ident, $term:ident, $sf_field:ident, $mg:ident, $eg:ident, $n:literal) => {
        impl term::LinearTerm for $term {
            type Upstream = term::TaperPair;
            type Input = [f64; $n];

            #[inline(always)]
            fn apply<T: EvalMath<Scalar = T>>(features: &SharedFeatures, params: &EvalParams<T>, _phase: T, acc: &mut Accumulators<T>) {
                for i in 0..$n {
                    let feature = T::from_i32(features.$sf_field[i]);
                    acc.bonus_mg += params.$mg[i] * feature;
                    acc.bonus_eg += params.$eg[i] * feature;
                }
            }

            #[inline(always)]
            fn apply_input(features: [f64; $n], values: &[f64], _phase: f64, acc: &mut Accumulators<f64>) {
                if features.iter().all(|f| *f == 0.0) {
                    return;
                }

                let mg = paste::paste!(eval_params::LAYOUT.[<$block _mg_offset>]);
                let eg = paste::paste!(eval_params::LAYOUT.[<$block _eg_offset>]);
                for i in 0..$n {
                    acc.bonus_mg += values[mg + i] * features[i];
                    acc.bonus_eg += values[eg + i] * features[i];
                }
            }

            #[inline(always)]
            fn scatter(features: [f64; $n], upstream: term::TaperPair, grads: &mut [f64]) {
                if features.iter().all(|f| *f == 0.0) {
                    return;
                }

                let mg = paste::paste!(eval_params::LAYOUT.[<$block _mg_offset>]);
                let eg = paste::paste!(eval_params::LAYOUT.[<$block _eg_offset>]);
                for i in 0..$n {
                    grads[mg + i] += upstream.d_mg * features[i];
                    grads[eg + i] += upstream.d_eg * features[i];
                }
            }
        }

        impl term::TermSource<$term> for SharedFeatures {
            type Input = [f64; $n];

            #[inline(always)]
            fn extract(&self) -> [f64; $n] {
                std::array::from_fn(|i| self.$sf_field[i] as f64)
            }
        }
    };
}

bonus_terms!(tapered_bonus_term);

/// White adds, Black subtracts, bucketed by the mover's own rank (rank 2 → index 0).
fn by_relative_rank(white: Bitboard, black: Bitboard) -> [i32; 6] {
    let mut buckets = [0i32; 6];
    for sq in white {
        buckets[(sq.rank() - 1) as usize] += 1;
    }
    for sq in black {
        buckets[(6 - sq.rank()) as usize] -= 1;
    }
    buckets
}

#[cfg(test)]
mod doubled_control_tests {
    use super::*;
    use crate::core::board::STARTPOS;

    /// A column of zeros or a column of clamps teaches the tuner nothing.
    #[test]
    fn counts_what_the_union_collapses() {
        for fen in [STARTPOS, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"] {
            let board = Position::from_fen(fen);

            for color in [Color::White, Color::Black] {
                let pinned = board.pinned_pieces(color);
                let ksq = board.pieces(PieceType::King, color).lsb();
                let area = !board.side_bb[color] & !board.pawn_attacks(color.opposite());
                let doubled = doubled_control(&board, color, pinned, ksq, area);

                assert!(doubled > 0, "no doubled control for {color:?} in {fen}");
                assert!(doubled < 127, "feature would clamp: {doubled} for {color:?} in {fen}");
            }
        }
    }
}
