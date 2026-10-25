//! History heuristics for move ordering and static evaluation correction.

use crate::{
    core::defs::{Bitboard, Color, PieceType, Square},
    engine::search_params::SearchParams,
};

const CORRECTION_SIZE: usize = 16384;
pub(super) const CORRECTION_SCALE: i32 = 256;
pub(super) const CORRECTION_LIMIT: i32 = 32 * CORRECTION_SCALE;

const _: () = assert!(CORRECTION_SIZE.is_power_of_two());

/// Continuation table indices for [n-1, n-2, n-4] plies back.
const CONT_SLOTS: [usize; 3] = [0, 1, 1];

/// Gravity caps must be positive and fit in `i16`.
#[derive(Clone, Copy)]
pub(crate) struct HistoryParams {
    pub(crate) quiet_cap: i32,
    pub(crate) butterfly_cap: i32,
    pub(crate) cont_cap: i32,
    pub(crate) capt_cap: i32,
    pub(crate) corr_weight_div: i32,
    pub(crate) corr_weight_max: i32,
    #[cfg(not(feature = "nocorr"))]
    pub(crate) minor_corr_weight: i32,
    #[cfg(not(feature = "nocorr"))]
    pub(crate) major_corr_weight: i32,
}

#[derive(Clone)]
pub(crate) struct History {
    /// `[side][piece][to_square]`.
    quiet: [[[i16; 64]; 6]; 2],
    /// `[side][from_atk][to_atk][from · 64 + to]` (~35 Elo).
    butterfly: [[[[i16; 4096]; 2]; 2]; 2],
    /// Slot 0 is n-1 (~13 Elo); slot 1 is shared by n-2 (~3 Elo) and n-4 (~3 Elo).
    cont: [ContinuationHistory; 2],
    /// Pawn structure correction, keyed on `pawn_key` (~53 Elo).
    pawn_correction: CorrectionHistory,
    /// Minor placement (knights + bishops, both colors), keyed on `minor_key`.
    /// Minor and major together ~23 Elo.
    minor_correction: CorrectionHistory,
    /// Major placement (rooks + queens, both colors), keyed on `major_key`.
    major_correction: CorrectionHistory,
    /// `[side][attacker][to][victim]` (~8 Elo).
    capt: CaptureHistory,
    pub(crate) params: HistoryParams,
}

#[derive(Clone, Copy)]
pub(crate) struct ContContext {
    pub(crate) pt: PieceType,
    pub(crate) to: Square,
}

#[derive(Clone)]
struct ContinuationHistory {
    data: Box<[i16]>,
}

#[derive(Clone)]
struct CaptureHistory {
    data: Box<[i16]>,
}

/// Moving average of search score minus uncorrected static evaluation.
/// One centipawn is [`CORRECTION_SCALE`] units.
#[derive(Clone)]
struct CorrectionHistory {
    data: Box<[i32]>,
}

impl From<&SearchParams> for HistoryParams {
    fn from(sp: &SearchParams) -> Self {
        Self {
            quiet_cap: sp.quiet_hist_cap,
            butterfly_cap: sp.butterfly_hist_cap,
            cont_cap: sp.cont_hist_cap,
            capt_cap: sp.capt_hist_cap,
            corr_weight_div: sp.corr_weight_div,
            corr_weight_max: sp.corr_weight_max,
            #[cfg(not(feature = "nocorr"))]
            minor_corr_weight: sp.minor_corr_weight,
            #[cfg(not(feature = "nocorr"))]
            major_corr_weight: sp.major_corr_weight,
        }
    }
}

impl Default for HistoryParams {
    fn default() -> Self { Self::from(&SearchParams::default()) }
}

impl Default for ContContext {
    fn default() -> Self { Self { pt: PieceType::None, to: Square(0) } }
}

impl ContinuationHistory {
    fn new() -> Self { Self { data: vec![0; 2 * 6 * 64 * 6 * 64].into_boxed_slice() } }
    fn clear(&mut self) { self.data.fill(0); }

    #[inline(always)]
    fn get(&self, stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> i16 {
        self.data[Self::idx(stm, prev_pt, prev_to, pt, to)]
    }

    #[inline(always)]
    fn get_mut(&mut self, stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> &mut i16 {
        let idx = Self::idx(stm, prev_pt, prev_to, pt, to);
        &mut self.data[idx]
    }

    #[inline(always)]
    fn idx(stm: Color, prev_pt: PieceType, prev_to: Square, pt: PieceType, to: Square) -> usize {
        let mut i = stm as usize;
        i = i * 6 + prev_pt as usize;
        i = i * 64 + prev_to.0 as usize;
        i = i * 6 + pt as usize;
        i = i * 64 + to.0 as usize;
        i
    }
}

impl Default for ContinuationHistory {
    fn default() -> Self { Self::new() }
}

impl CaptureHistory {
    fn new() -> Self { Self { data: vec![0; 2 * 6 * 64 * 6].into_boxed_slice() } }
    fn clear(&mut self) { self.data.fill(0); }

    #[inline(always)]
    fn get(&self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> i16 {
        self.data[Self::idx(stm, attacker, to, victim)]
    }

    #[inline(always)]
    fn get_mut(&mut self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> &mut i16 {
        let idx = Self::idx(stm, attacker, to, victim);
        &mut self.data[idx]
    }

    #[inline(always)]
    fn idx(stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> usize {
        debug_assert!((attacker as usize) < 6, "attacker out of range");
        debug_assert!((victim as usize) < 6, "victim out of range");
        let mut i = stm as usize;
        i = i * 6 + attacker as usize;
        i = i * 64 + to.0 as usize;
        i = i * 6 + victim as usize;
        i
    }
}

impl Default for CaptureHistory {
    fn default() -> Self { Self::new() }
}

impl CorrectionHistory {
    fn new() -> Self { Self { data: vec![0; 2 * CORRECTION_SIZE].into_boxed_slice() } }
    fn clear(&mut self) { self.data.fill(0); }

    #[inline(always)]
    fn idx(stm: Color, hash: u64) -> usize { stm as usize * CORRECTION_SIZE + (hash as usize & (CORRECTION_SIZE - 1)) }

    #[cfg(not(feature = "nocorr"))]
    #[inline(always)]
    fn get(&self, stm: Color, hash: u64) -> i32 { self.data[Self::idx(stm, hash)] }

    /// The new observation's weight is out of 256.
    #[inline(always)]
    fn update(&mut self, stm: Color, hash: u64, raw_diff: i32, depth: i32, params: &HistoryParams) {
        let entry = &mut self.data[Self::idx(stm, hash)];
        let weight = ((1 + depth) * (1 + depth) / params.corr_weight_div).min(params.corr_weight_max);
        let scaled = i64::from(raw_diff) * i64::from(CORRECTION_SCALE);
        let blended = (i64::from(*entry) * i64::from(256 - weight) + scaled * i64::from(weight)) / 256;
        *entry = blended.clamp(-i64::from(CORRECTION_LIMIT), i64::from(CORRECTION_LIMIT)) as i32;
    }
}

impl Default for CorrectionHistory {
    fn default() -> Self { Self::new() }
}

impl History {
    pub(crate) fn new() -> Self {
        Self {
            quiet: [[[0; 64]; 6]; 2],
            butterfly: [[[[0; 4096]; 2]; 2]; 2],
            cont: [ContinuationHistory::new(), ContinuationHistory::new()],
            pawn_correction: CorrectionHistory::new(),
            minor_correction: CorrectionHistory::new(),
            major_correction: CorrectionHistory::new(),
            capt: CaptureHistory::new(),
            params: HistoryParams::default(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.quiet = [[[0; 64]; 6]; 2];
        self.butterfly = [[[[0; 4096]; 2]; 2]; 2];
        self.cont[0].clear();
        self.cont[1].clear();
        self.pawn_correction.clear();
        self.minor_correction.clear();
        self.major_correction.clear();
        self.capt.clear();
    }

    #[inline(always)]
    pub(crate) fn score_quiet(
        &self,
        stm: Color,
        pt: PieceType,
        from: Square,
        to: Square,
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
    ) -> i32 {
        let from_atk = threats.check_bit(from) as usize;
        let to_atk = threats.check_bit(to) as usize;
        let mut score =
            i32::from(self.quiet[stm][pt][to]) + i32::from(self.butterfly[stm][from_atk][to_atk][butterfly_idx(from, to)]);

        for (slot, ctx) in CONT_SLOTS.into_iter().zip([cont1, cont2, cont4]) {
            if ctx.pt != PieceType::None {
                score += i32::from(self.cont[slot].get(stm, ctx.pt, ctx.to, pt, to));
            }
        }
        score
    }

    #[inline(always)]
    pub(crate) fn update_quiet(
        &mut self,
        stm: Color,
        pt: PieceType,
        from: Square,
        to: Square,
        threats: Bitboard,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
        bonus: i32,
    ) {
        let from_atk = threats.check_bit(from) as usize;
        let to_atk = threats.check_bit(to) as usize;
        Self::update_entry(&mut self.quiet[stm][pt][to], bonus, self.params.quiet_cap);
        Self::update_entry(&mut self.butterfly[stm][from_atk][to_atk][butterfly_idx(from, to)], bonus, self.params.butterfly_cap);
        self.update_conthist(stm, pt, to, cont1, cont2, cont4, bonus);
    }

    #[inline(always)]
    pub(super) fn update_conthist(
        &mut self,
        stm: Color,
        pt: PieceType,
        to: Square,
        cont1: ContContext,
        cont2: ContContext,
        cont4: ContContext,
        bonus: i32,
    ) {
        for (slot, ctx) in CONT_SLOTS.into_iter().zip([cont1, cont2, cont4]) {
            if ctx.pt != PieceType::None {
                Self::update_entry(self.cont[slot].get_mut(stm, ctx.pt, ctx.to, pt, to), bonus, self.params.cont_cap);
            }
        }
    }

    // Gravity damps bonuses that reinforce an entry and amplifies those that oppose it.
    #[inline(always)]
    fn update_entry(entry: &mut i16, bonus: i32, cap: i32) {
        debug_assert!(cap > 0, "cap must be positive");
        let e = i32::from(*entry);
        *entry = (e + bonus - e * bonus.abs() / cap).clamp(-cap, cap) as i16;
    }

    /// In fixed-point centipawns. Pawn correction counts in full; minor and major are weighted
    /// estimates of the same placement error, so average rather than sum them.
    #[cfg(not(feature = "nocorr"))]
    #[inline(always)]
    pub(super) fn correction(&self, stm: Color, pawn_key: u64, minor_key: u64, major_key: u64) -> i32 {
        let pawn = self.pawn_correction.get(stm, pawn_key);
        let minor = self.minor_correction.get(stm, minor_key);
        let major = self.major_correction.get(stm, major_key);

        #[cfg(feature = "corrstats")]
        {
            use crate::engine::corrstats::{Table, record_read};
            record_read(Table::Pawn, pawn);
            record_read(Table::Minor, minor);
            record_read(Table::Major, major);
        }

        let placement = (minor * self.params.minor_corr_weight + major * self.params.major_corr_weight)
            / (self.params.minor_corr_weight + self.params.major_corr_weight);
        pawn + placement
    }

    #[inline(always)]
    pub(super) fn update_correction(&mut self, stm: Color, pawn_key: u64, minor_key: u64, major_key: u64, diff: i32, depth: i32) {
        self.pawn_correction.update(stm, pawn_key, diff, depth, &self.params);
        self.minor_correction.update(stm, minor_key, diff, depth, &self.params);
        self.major_correction.update(stm, major_key, diff, depth, &self.params);

        #[cfg(feature = "corrstats")]
        {
            use crate::engine::corrstats::{Table, record_update};
            record_update(Table::Pawn);
            record_update(Table::Minor);
            record_update(Table::Major);
        }
    }

    #[inline(always)]
    pub(super) fn score_capture(&self, stm: Color, attacker: PieceType, to: Square, victim: PieceType) -> i32 {
        i32::from(self.capt.get(stm, attacker, to, victim))
    }

    #[inline(always)]
    pub(super) fn update_capture(&mut self, stm: Color, attacker: PieceType, to: Square, victim: PieceType, bonus: i32) {
        Self::update_entry(self.capt.get_mut(stm, attacker, to, victim), bonus, self.params.capt_cap);
    }
}

impl Default for History {
    fn default() -> Self { Self::new() }
}

#[inline(always)]
fn butterfly_idx(from: Square, to: Square) -> usize { from.0 as usize * 64 + to.0 as usize }
