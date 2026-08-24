//! Hand-Crafted Evaluation (HCE).
//!
//! Computes the heuristic score of a leaf position by combining an incrementally
//! maintained SIMD accumulator for piece-square material (PSQT) with dynamically
//! extracted spatial features (mobility, king safety, pawn structure).
//!
//! Evaluation is generic over [`EvalMath`] (`i32` or `DualNode`):
//! - search monomorphizes to `i32`, which compiles to plain arithmetic.
//! - tuning runs `DualNode`, carrying each value's partials so one forward pass yields the
//!   gradient with no reverse tape.

use crate::{
    core::{
        board::{Position, bitboard, spatial::SpatialMaps, xorboard::XorBoard},
        defs::{Bitboard, Color, Direction, LANE_PHASE, PieceType, TOTAL_PHASE},
    },
    engine::{
        autograd::EvalMath,
        combiner::{Accumulators, Combiner, CombinerParams, LinearCombiner, safety_block, taper},
        eval_params::{
            self, ATTACKER, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_DANGER, KING_SAFETY, MG_MOBILITY_CLOSED, MG_MOBILITY_OPEN,
            XRAY,
        },
        mobility::{self, Mobility, SpatialMetrics},
        search_params::SearchParams,
        term::{self},
    },
    weave::Vi16x8,
};

/// Non-PSQT evaluation weights, generic over the computation type `T`.
///
/// - In search: `EvalParams::<i32>::from_const()` loads compile-time constants.
/// - In tuning: `EvalParams::<DualNode>::load_tunable()` seeds each parameter as a
///   dual variable (`grad[slot] = 1.0`) for derivative tracking.
macro_rules! impl_eval_params {
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr, $konst:expr) ),* $(,)?) => {
        pub struct EvalParams<T: EvalMath> {
            $( pub $name: <T as EvalMath>::$ty, )*
        }

        impl EvalParams<i32> {
            /// Loads weights from the compile-time const tables.
            #[inline(always)]
            pub fn from_const() -> Self {
                Self { $( $name: $konst, )* }
            }
        }

        impl<T: EvalMath<Scalar = T>> EvalParams<T> {
            // Used exclusively during tuning.
            #[allow(dead_code)]
            pub fn load_tunable(values: &[f64]) -> Self {
                // Parameter gradient offsets: slots 0 (MG) and 1 (EG) are reserved for PSQT.
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

/// The one list a bonus term generates from: its struct, params, layout and scatter.
///
/// The `@tunables` and `@blocks` arms rewrite the rows for other consumers and live here
/// rather than beside them, because a consumer arrives as a bare ident that resolves where
/// it was named while `$crate::` resolves anywhere. Array widths match the literal `6`, so
/// a row declared wider fails at the build instead of scattering past its block.
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

/// Terms outside the bonus list are named here by hand; one left out builds clean and
/// never evaluates.
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

/// The score split per bucket, for the UCI `eval` command.
pub struct DetailedEval {
    pub psqt: i32,
    pub mobility: i32,
    pub bonus: i32,
    pub safety: i32,
    pub total: i32,
}

/// Positional features extracted once per position.
///
/// Shared between the search evaluation pass and tuner feature scattering. Fields are
/// documented only where their encoding or bucketing is not in the name; bare fields
/// mean exactly what they say.
pub struct SharedFeatures {
    pub openness: i32,
    pub spatial: SpatialMetrics,
    /// Orthogonal x-rays landing in the enemy king ring, White minus Black.
    pub xray_ortho: i32,
    pub bishop_pair_diff: i32,
    /// Rooks on fully open files (no pawns of either color), White minus Black.
    pub rook_open_diff: i32,
    /// Passed pawns bucketed by relative rank (rank 2 → index 0).
    pub passed_pawn: [i32; 6],
    /// Chebyshev distance from enemy king to passer (dist 1 → index 0, dist 6+ → index 5).
    pub enemy_king_dist: [i32; 6],
    /// Adjacent vertical doubled pawn difference (gapped stacks uncounted).
    pub doubled_pawn_diff: i32,
    pub isolated_pawn_diff: i32,
    /// Horizontally adjacent friendly pawn pairs bucketed by relative rank.
    pub phalanx: [i32; 6],
    /// Friendly pawn defended by another pawn bucketed by relative rank (index 0 / rank 2 unreachable).
    pub defended_pawn: [i32; 6],
    pub backward_pawn_diff: i32,
    /// White-relative tempo (+1 for White, -1 for Black); normalized to STM in combination.
    pub tempo: i32,
    /// Minor piece with a friendly or enemy pawn directly in front.
    pub minor_behind_pawn_diff: i32,
}

/// The integer evaluation the search calls.
#[inline]
pub fn evaluate(board: &Position, acc: &Vi16x8) -> i32 {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    evaluate_generic::<i32>(board, acc, phase, &params, None)
}

/// The same score, with each bucket kept separate.
pub fn detailed_eval(board: &Position, acc: &Vi16x8) -> DetailedEval {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    let features = SharedFeatures::compute(board);
    let buckets = fill_accumulators::<i32>(acc, phase, &features, &params);
    let psqt = buckets.mg_eg;
    let mobility = buckets.mobility;
    let combiner_params = CombinerParams::from_eval(&params);
    let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
    let safety = safety_block(&buckets, phase, &combiner_params);
    let total = LinearCombiner::forward(&buckets, phase, &combiner_params);
    let (p, m, b, s, t) = if board.stm == Color::White {
        (psqt, mobility, bonus, safety, total)
    } else {
        (-psqt, -mobility, -bonus, -safety, -total)
    };

    DetailedEval { psqt: p, mobility: m, bonus: b, safety: s, total: t }
}

/// Accumulator-only evaluation, no spatial features. Datagen's volatility filter.
#[inline(always)]
pub fn evaluate_psqt(board: &Position, acc: &Vi16x8, phase: i32) -> i32 {
    let score = i32::taper_acc(acc, phase);
    if board.stm == Color::White { score } else { -score }
}

/// Computes a phase- and piece-scaled safety margin for lazy evaluation pruning.
#[inline]
pub fn lazy_eval_margin(board: &Position, phase: i32, params: &SearchParams) -> i32 {
    let volatility = board.piece_count(PieceType::Pawn) * params.vol_pawn
        + board.piece_count(PieceType::Knight) * params.vol_knight
        + board.piece_count(PieceType::Bishop) * params.vol_bishop
        + board.piece_count(PieceType::Rook) * params.vol_rook
        + board.piece_count(PieceType::Queen) * params.vol_queen
        + board.piece_count(PieceType::King) * params.vol_king;

    let div = std::cmp::max(params.qs_lazy_divisor, 1);
    let scaled = (volatility * phase) / div;
    params.qs_lazy_margin + scaled
}

/// Generic evaluation core. Monomorphized to `i32` for search and `DualNode` for tuning.
///
/// Invariant: [`LinearTerm::scatter`] assumes linear feature scaling, `y = w · x`. A shape
/// like `feature · feature · weight` or `max(feature, 0)` produces invalid gradients and
/// belongs in [`Combiner`] instead. Run `make oracle` to verify.
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

    let score = combine_buckets(acc, phase, features, params);
    if board.stm == Color::White { score } else { -score }
}

#[inline(always)]
fn combine_buckets<T: EvalMath<Scalar = T>>(acc: &T::Vec8, phase: T, features: &SharedFeatures, params: &EvalParams<T>) -> T {
    let buckets = fill_accumulators::<T>(acc, phase, features, params);
    LinearCombiner::forward(&buckets, phase, &CombinerParams::from_eval(params))
}

/// Extracts the game phase, clamped to `0..=TOTAL_PHASE`, from its accumulator lane.
#[inline(always)]
pub fn extract_phase(acc: &Vi16x8) -> i32 { i32::from(acc.extract::<{ LANE_PHASE as i32 }>()).clamp(0, TOTAL_PHASE) }

/// Cached pawn structure features keyed on `pawn_key`.
/// Retains passer bitboards to compute `enemy_king_dist` without re-running passed-span scans.
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

/// Fixed-size direct-mapped cache for pawn features, keyed on `pawn_key`. Pawn structure
/// barely shifts walking the tree, so the hit rate is high and the scan collapses to a probe.
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

        // Passed pawns: no enemy pawns ahead on the same or adjacent files.
        // Bucketed inline rather than calling `by_relative_rank` (+0.8 instrs/node saved).
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

        // Doubled pawns: `& shift(North)` marks the front pawn of each adjacent vertical
        // pair, so gapped stacks go uncounted.
        let w_doubled = (wp & wp.shift(Direction::North)).popcount() as i32;
        let b_doubled = (bp & bp.shift(Direction::North)).popcount() as i32;
        let doubled_pawn_diff = w_doubled - b_doubled;

        // Isolated pawns: no friendly pawn on either adjacent file. `file_fill` smears each
        // pawn down its file, and shifting east/west gives the neighbour-file mask. Shared
        // with backward-pawn detection below.
        let (w_files, b_files) = (wp.file_fill(), bp.file_fill());
        let w_adj = w_files.shift(Direction::East) | w_files.shift(Direction::West);
        let b_adj = b_files.shift(Direction::East) | b_files.shift(Direction::West);
        let w_isolated = (wp & !w_adj).popcount() as i32;
        let b_isolated = (bp & !b_adj).popcount() as i32;
        let isolated_pawn_diff = w_isolated - b_isolated;

        // Phalanx: `& shift(East)` marks the east pawn of each side-by-side pair.
        let phalanx = by_relative_rank(wp & wp.shift(Direction::East), bp & bp.shift(Direction::East));

        // Defended pawns: a pawn standing where its own side's pawns attack. The attack
        // maps are shared with backward detection below.
        let w_pawn_atk = board.pawn_attacks(Color::White);
        let b_pawn_atk = board.pawn_attacks(Color::Black);
        let defended_pawn = by_relative_rank(wp & w_pawn_atk, bp & b_pawn_atk);

        // Backward pawns: behind every friendly neighbour, with a stop square the enemy
        // controls. The adjacency mask excludes isolated pawns, which score as isolated
        // instead. `north_fill`/`south_fill` shifted sideways marks every square with a
        // friendly pawn at or behind its rank.
        let w_fill = wp.north_fill();
        let b_fill = bp.south_fill();
        let w_rear = w_fill.shift(Direction::East) | w_fill.shift(Direction::West);
        let b_rear = b_fill.shift(Direction::East) | b_fill.shift(Direction::West);
        let w_stop_bad = (bp | b_pawn_atk) >> 8; // Stop square blocked or attacked by enemy pawns
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
    fn default() -> Self { Self::new() }
}

impl PawnCache {
    const SIZE: usize = 1 << 14;

    pub fn new() -> Self {
        // `u64::MAX` sentinel prevents false hits on empty positions (pawnless Zobrist key is 0).
        let empty = PawnEntry { key: u64::MAX, pawn: PawnFeatures::default() };
        Self { entries: vec![empty; Self::SIZE].into_boxed_slice() }
    }

    /// Probes the cache by `pawn_key`, computing and caching on a miss.
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
    pub fn compute(board: &Position) -> Self { Self::with_pawn(board, &PawnFeatures::compute(board), None) }

    /// Piece features fresh, pawn features from `pawn`. `enemy_king_dist` is the one
    /// pawn-derived bucket rebuilt here, because it moves with the enemy king rather than
    /// with the pawns.
    pub fn with_pawn(board: &Position, pawn: &PawnFeatures, rows: Option<&XorBoard>) -> Self {
        let pinned_w = board.pinned_pieces(Color::White);
        let pinned_b = board.pinned_pieces(Color::Black);
        let maps = SpatialMaps::compute(board, pinned_w.0, pinned_b.0);

        let spatial = Mobility::compute_all(board, &maps, pinned_w, pinned_b, rows);

        let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
        let w_king_ring = bitboard::atk_king(w_ksq).0;
        let b_king_ring = bitboard::atk_king(b_ksq).0;

        let xray_ortho =
            (maps.w_ortho_xray() & b_king_ring).count_ones() as i32 - (maps.b_ortho_xray() & w_king_ring).count_ones() as i32;

        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        let open = !board.role_bb[PieceType::Pawn].file_fill();
        let rooks_open = board.role_bb[PieceType::Rook] & open;
        let w_open = (rooks_open & board.side_bb[Color::White]).popcount() as i32;
        let b_open = (rooks_open & board.side_bb[Color::Black]).popcount() as i32;
        let rook_open_diff = w_open - b_open;

        // Enemy-king Chebyshev distance to each passed pawn (clamped to 1..=6).
        let mut enemy_king_dist = [0i32; 6];
        for sq in pawn.w_passers {
            enemy_king_dist[(b_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] += 1;
        }
        for sq in pawn.b_passers {
            enemy_king_dist[(w_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] -= 1;
        }

        // Shielded minors: shifting all pawns toward us drops a front pawn onto the minor's
        // own square, so the AND counts knights and bishops with a pawn of either color ahead.
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
            spatial,
            xray_ortho,
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

/// Seeds the buckets from the SIMD accumulator, then applies every term. Its own function
/// so `combine_buckets` and `detailed_eval` cannot drift apart.
#[inline]
pub fn fill_accumulators<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> Accumulators<T> {
    let mut buckets = Accumulators::<T> {
        mg_eg: T::taper_acc(acc, phase),
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
    fn extract(&self) -> f64 { self.xray_ortho as f64 }
}

/// Generates `LinearTerm` and `TermSource` implementations for tapered positional bonus terms.
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

/// Populates 6 relative-rank buckets (White adds, Black subtracts, rank 2 → index 0).
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
