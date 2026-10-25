//! Hand-crafted evaluation, shared by search and tuning through [`EvalMath`].

use crate::{
    core::{
        board::{Position, bitboard, spatial::SliderMaps, xorboard::XorBoard},
        defs::{Bitboard, Color, Direction, LANE_PHASE, PieceType, TOTAL_PHASE},
    },
    engine::{
        autograd::EvalMath,
        combiner::{self, Accumulators, CombinerParams, safety_block, taper},
        eval_params::{
            self, ATTACKER, EG_MOBILITY_CLOSED, EG_MOBILITY_OPEN, KING_DANGER, KING_SAFETY, MG_MOBILITY_CLOSED, MG_MOBILITY_OPEN,
            XRAY,
        },
        mobility::{self, SpatialMetrics},
        search_params::SearchParams,
        term,
    },
    weave::I16x8,
};

macro_rules! impl_eval_params {
    ($( ($name:ident, $ty:ident, $offset_field:ident, $extra:expr, $konst:expr) ),* $(,)?) => {
        pub(crate) struct EvalParams<T: EvalMath> {
            $( pub(crate) $name: <T as EvalMath>::$ty, )*
        }

        impl EvalParams<i32> {
            #[inline(always)]
            pub(super) fn from_const() -> Self {
                Self { $( $name: $konst, )* }
            }
        }

        #[cfg(feature = "dataset")]
        impl<T: EvalMath<Scalar = T>> EvalParams<T> {
            pub(crate) fn load_tunable(values: &[f64]) -> Self {
                // Slots 0 (MG) and 1 (EG) are reserved for the material/PSQT accumulator.
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

/// Bonus terms and their parameter blocks.
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
        $( pub(crate) struct $term; )*

        crate::register_terms! {
            mobility::MobilityTerm   => mobility,
            mobility::KingSafetyTerm => king_safety,
            XrayTerm                 => xray,
            $( $term                 => bonus, )*
        }
    };
}

bonus_terms!(register_bonus);

pub(crate) struct XrayTerm;

/// Score components from the side-to-move perspective.
pub(crate) struct DetailedEval {
    pub(crate) material_psqt: i32,
    pub(crate) mobility: i32,
    pub(crate) bonus: i32,
    pub(crate) safety: i32,
    pub(crate) total: i32,
}

/// White-relative evaluation features.
pub(crate) struct SharedFeatures {
    pub(crate) openness: i32,
    pub(crate) spatial: SpatialMetrics,
    /// Orthogonal x-rays landing in the enemy king ring, White minus Black.
    pub(crate) xray_ortho: i32,
    pub(crate) bishop_pair_diff: i32,
    pub(crate) rook_open_diff: i32,
    /// Passed pawns bucketed by relative rank (rank 2 → index 0).
    pub(crate) passed_pawn: [i32; 6],
    /// Chebyshev distance from enemy king to passer (dist 1 → index 0, dist 6+ → index 5).
    pub(crate) enemy_king_dist: [i32; 6],
    /// Adjacent vertical doubled pawn difference (gapped stacks uncounted).
    pub(crate) doubled_pawn_diff: i32,
    pub(crate) isolated_pawn_diff: i32,
    /// Horizontally adjacent friendly pawn pairs bucketed by relative rank.
    pub(crate) phalanx: [i32; 6],
    /// Friendly pawn defended by another pawn bucketed by relative rank (index 0 / rank 2 unreachable).
    pub(crate) defended_pawn: [i32; 6],
    pub(crate) backward_pawn_diff: i32,
    /// +1 for White to move, -1 for Black.
    pub(crate) tempo: i32,
    /// Minor piece with a friendly or enemy pawn directly in front.
    pub(crate) minor_behind_pawn_diff: i32,
}

#[cfg(test)]
#[inline]
pub(crate) fn evaluate(board: &Position, acc: &I16x8) -> i32 {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    let features = SharedFeatures::compute(board);
    evaluate_generic::<i32>(board, acc, phase, &params, &features)
}

pub(crate) fn detailed_eval(board: &Position, acc: &I16x8) -> DetailedEval {
    let phase = extract_phase(acc);
    let params = EvalParams::<i32>::from_const();
    let features = SharedFeatures::compute(board);
    let buckets = evaluate_buckets::<i32>(acc, phase, &features, &params);
    let material_psqt = buckets.material_psqt;
    let mobility = buckets.mobility;
    let combiner_params = CombinerParams::from_eval(&params);
    let bonus = taper(buckets.bonus_mg, buckets.bonus_eg, phase);
    let safety = safety_block(&buckets, phase, &combiner_params);
    let total = combiner::combine(&buckets, phase, &combiner_params);
    let (material_psqt, mobility, bonus, safety, total) = if board.stm == Color::White {
        (material_psqt, mobility, bonus, safety, total)
    } else {
        (-material_psqt, -mobility, -bonus, -safety, -total)
    };

    DetailedEval { material_psqt, mobility, bonus, safety, total }
}

/// Accumulator-only evaluation, no spatial features.
#[inline(always)]
pub(crate) fn evaluate_psqt(board: &Position, acc: &I16x8, phase: i32) -> i32 {
    let score = i32::taper_acc(acc, phase);
    if board.stm == Color::White { score } else { -score }
}

/// Heuristic margin between accumulator-only and full evaluation.
#[inline]
pub(super) fn lazy_eval_margin(board: &Position, phase: i32, params: &SearchParams) -> i32 {
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

/// [`term::LinearTerm::scatter`] assumes every term is linear in its weights, `y = w · x`.
/// Nonlinearities over aggregate scores belong in [`combiner::combine`].
#[inline(always)]
pub(crate) fn evaluate_generic<T: EvalMath<Scalar = T>>(
    board: &Position,
    acc: &T::Vec8,
    phase: T,
    params: &EvalParams<T>,
    features: &SharedFeatures,
) -> T {
    let buckets = evaluate_buckets::<T>(acc, phase, features, params);
    let score = combiner::combine(&buckets, phase, &CombinerParams::from_eval(params));
    if board.stm == Color::White { score } else { -score }
}

#[inline(always)]
pub(crate) fn extract_phase(acc: &I16x8) -> i32 { i32::from(acc.extract::<{ LANE_PHASE as i32 }>()).clamp(0, TOTAL_PHASE) }

#[derive(Clone, Copy, Default)]
pub(super) struct PawnFeatures {
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

/// Direct-mapped pawn-feature cache.
pub(super) struct PawnCache {
    entries: Box<[PawnEntry]>,
}

#[derive(Clone, Copy)]
struct PawnEntry {
    key: u64,
    pawn: PawnFeatures,
}

impl PawnFeatures {
    fn compute(board: &Position) -> Self {
        let openness = mobility::compute_openness(board);
        let wp = board.pieces(PieceType::Pawn, Color::White);
        let bp = board.pieces(PieceType::Pawn, Color::Black);

        // Bucketed inline rather than calling by_relative_rank (+0.8 instrs/node saved).
        let mut passed_pawn = [0i32; 6];
        let mut w_passers = Bitboard::default();
        let mut b_passers = Bitboard::default();

        for sq in wp {
            if bitboard::is_passed_pawn(sq, Color::White, bp) {
                passed_pawn[(sq.rank() - 1) as usize] += 1;
                w_passers |= sq.bitboard();
            }
        }
        for sq in bp {
            if bitboard::is_passed_pawn(sq, Color::Black, wp) {
                passed_pawn[(6 - sq.rank()) as usize] -= 1;
                b_passers |= sq.bitboard();
            }
        }

        let w_doubled = (wp & wp.shift(Direction::North)).popcount() as i32;
        let b_doubled = (bp & bp.shift(Direction::North)).popcount() as i32;
        let doubled_pawn_diff = w_doubled - b_doubled;

        let (w_files, b_files) = (wp.file_fill(), bp.file_fill());
        let w_adj = w_files.shift(Direction::East) | w_files.shift(Direction::West);
        let b_adj = b_files.shift(Direction::East) | b_files.shift(Direction::West);
        let w_isolated = (wp & !w_adj).popcount() as i32;
        let b_isolated = (bp & !b_adj).popcount() as i32;
        let isolated_pawn_diff = w_isolated - b_isolated;

        let phalanx = by_relative_rank(wp & wp.shift(Direction::East), bp & bp.shift(Direction::East));

        let w_pawn_atk = board.pawn_attacks(Color::White);
        let b_pawn_atk = board.pawn_attacks(Color::Black);
        let defended_pawn = by_relative_rank(wp & w_pawn_atk, bp & b_pawn_atk);

        // Backward pawns: behind every friendly neighbor, with a stop square blocked or
        // attacked by enemy pawns. The adjacency mask excludes isolated pawns, which score
        // as isolated instead. north_fill/south_fill shifted sideways marks every square
        // with a friendly pawn at or behind its rank.
        let w_fill = wp.north_fill();
        let b_fill = bp.south_fill();
        let w_rear = w_fill.shift(Direction::East) | w_fill.shift(Direction::West);
        let b_rear = b_fill.shift(Direction::East) | b_fill.shift(Direction::West);
        let w_stop_bad = (bp | b_pawn_atk) >> 8;
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

    pub(super) fn new() -> Self {
        // u64::MAX sentinel prevents false hits on empty positions (pawnless Zobrist key is 0).
        let empty = PawnEntry { key: u64::MAX, pawn: PawnFeatures::default() };
        Self { entries: vec![empty; Self::SIZE].into_boxed_slice() }
    }

    #[inline]
    pub(super) fn probe(&mut self, board: &Position) -> PawnFeatures {
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
    pub(crate) fn compute(board: &Position) -> Self { Self::with_pawn(board, &PawnFeatures::compute(board), None) }

    pub(super) fn with_pawn(board: &Position, pawn: &PawnFeatures, rows: Option<&XorBoard>) -> Self {
        let pinned_w = board.pinned_pieces(Color::White);
        let pinned_b = board.pinned_pieces(Color::Black);
        let maps = SliderMaps::compute(board, pinned_w, pinned_b);
        let spatial = mobility::compute_all(board, &maps, pinned_w, pinned_b, rows);

        let w_ksq = board.pieces(PieceType::King, Color::White).lsb();
        let b_ksq = board.pieces(PieceType::King, Color::Black).lsb();
        let w_king_ring = bitboard::atk_king(w_ksq);
        let b_king_ring = bitboard::atk_king(b_ksq);

        let xray_ortho =
            (maps.w_ortho_xray() & b_king_ring).popcount() as i32 - (maps.b_ortho_xray() & w_king_ring).popcount() as i32;

        let w_pair = i32::from(board.pieces(PieceType::Bishop, Color::White).more_than_one());
        let b_pair = i32::from(board.pieces(PieceType::Bishop, Color::Black).more_than_one());
        let bishop_pair_diff = w_pair - b_pair;

        let open = !board.role_bb[PieceType::Pawn].file_fill();
        let rooks_open = board.role_bb[PieceType::Rook] & open;
        let w_open = (rooks_open & board.side_bb[Color::White]).popcount() as i32;
        let b_open = (rooks_open & board.side_bb[Color::Black]).popcount() as i32;
        let rook_open_diff = w_open - b_open;

        // King distances must be recomputed even when the pawn structure is unchanged.
        let mut enemy_king_dist = [0i32; 6];
        for sq in pawn.w_passers {
            enemy_king_dist[(b_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] += 1;
        }
        for sq in pawn.b_passers {
            enemy_king_dist[(w_ksq.chebyshev_distance(sq).clamp(1, 6) - 1) as usize] -= 1;
        }

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

#[inline]
pub(crate) fn evaluate_buckets<T: EvalMath<Scalar = T>>(
    acc: &T::Vec8,
    phase: T,
    features: &SharedFeatures,
    params: &EvalParams<T>,
) -> Accumulators<T> {
    let mut buckets = Accumulators::<T> {
        material_psqt: T::taper_acc(acc, phase),
        mobility: T::zero(),
        bonus_mg: T::zero(),
        bonus_eg: T::zero(),
        safety_us: T::zero(),
        safety_them: T::zero(),
        pressure_us: T::zero(),
        pressure_them: T::zero(),
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

macro_rules! impl_bonus_terms {
    ( [] $( $block:ident = $kind:ident ( $($spec:tt)* ) ; )* ) => {
        $( impl_bonus_terms!(@$kind $block, $($spec)*); )*
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

bonus_terms!(impl_bonus_terms);

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
