use std::collections::HashMap;

use parking_lot::Mutex;
use soul::engine::search_params::{PARAM_DEFS, SearchParams};

use super::{pentanomial::Pentanomial, selfplay};

/// Zero-allocation, continuous-to-discrete FNV-1a cache hash.
///
/// Converts the continuous `normalized` floats into the EXACT discrete `i32` values
/// the engine will use, and hashes them. This completely eliminates plateau noise
/// (where CMA-ES tests 0.500 and 0.501, producing the exact same engine config,
/// but generating fictitious gradient noise from random game outcomes).
fn hash_discrete_params(candidate_norm: &[f64], opponent_norm: &[f64], openings: &[String]) -> u64 {
    let mut fnv = crate::core::fnv::Fnv1a::new();

    // Hash candidate
    for (i, &v) in candidate_norm.iter().enumerate() {
        let param = &PARAM_DEFS[i];
        let discrete_val = param.denormalize(v);
        fnv.write_bytes(&discrete_val.to_bits().to_le_bytes());
    }

    // Hash opponent
    for (i, &v) in opponent_norm.iter().enumerate() {
        let param = &PARAM_DEFS[i];
        let discrete_val = param.denormalize(v);
        fnv.write_bytes(&discrete_val.to_bits().to_le_bytes());
    }

    // Incorporate opening hashes
    for op in openings {
        fnv.write_bytes(op.as_bytes());
    }

    fnv.digest()
}

#[derive(Clone)]
struct CachedEntry {
    penta:   Pentanomial,
    c_nodes: u64,
    b_nodes: u64,
    pairs:   usize,
}

/// Thread-safe memoization layer for self-play matches.
///
/// Running millions of engine nodes is the absolute bottleneck of parameter tuning.
/// The `MatchCache` ensures we never play the exact same matched parameters twice.
///
/// Note on Hit Rate & Scope:
/// Because the opening subset is randomly sliced every epoch, the hash key purposefully
/// never collides across epochs. This cache functions strictly as an intra-epoch deduplicator
/// mapping CMA-ES parameter samples that quantize to the identical discrete integers
/// within the same generation. It does not provide historical inter-epoch memoization.
pub struct MatchCache {
    inner: Mutex<CacheInner>,
    tc:    String,
}

struct CacheInner {
    entries: HashMap<u64, CachedEntry>,
    hits:    usize,
    misses:  usize,
}

pub struct MatchRequest<'a> {
    pub params:              SearchParams,
    pub normalized:          &'a [f64],
    pub opponent_params:     SearchParams,
    pub opponent_normalized: &'a [f64],
    pub openings:            &'a [String],
    pub min_pairs:           usize,
}

impl MatchCache {
    #[must_use]
    pub fn new(tc: &str) -> Self {
        Self {
            inner: Mutex::new(CacheInner {
                entries: HashMap::new(),
                hits:    0,
                misses:  0,
            }),
            tc:    tc.to_string(),
        }
    }

    /// Peeks into the cache for a given key.
    ///
    /// Locks the mutex precisely enough to copy the payload.
    fn try_get(&self, key: u64) -> Option<(Pentanomial, u64, u64, usize)> {
        let inner = self.inner.lock();
        let entry = inner.entries.get(&key)?;
        Some((entry.penta, entry.c_nodes, entry.b_nodes, entry.pairs))
    }

    /// Insert or update cache entry.
    fn insert(&self, key: u64, penta: Pentanomial, c_nodes: u64, b_nodes: u64, pairs: usize) {
        let mut inner = self.inner.lock();
        inner.entries.insert(key, CachedEntry {
            penta,
            c_nodes,
            b_nodes,
            pairs,
        });
    }

    /// Fetches a match result from the cache or executes it.
    /// Drops the lock during game generation to minimize contention.
    /// We require full cache hits (cached pair count ≥ requested pairs) to prevent partial
    /// result sets from mixing opening subsets, which would corrupt the Pentanomial statistics.
    /// If the cache doesn't have the required number of pairs, we play the full match set
    /// and insert a fresh entry.
    ///
    /// NOTE: there is an intentional TOCTOU race between try_get and insert.
    /// If two threads simultaneously miss on the same key (common when σ is small
    /// and several candidates collapse to the same integer vector), both will play
    /// and the last writer wins. This causes duplicate work and slightly inflated
    /// miss counts, but both samples are unbiased so correctness is preserved.
    pub fn get_or_run<F>(&self, req: MatchRequest<'_>, on_pair: F) -> (Pentanomial, u64, u64)
    where F: Fn() + Sync + Send {
        let subset = &req.openings[..req.min_pairs];
        let key = hash_discrete_params(req.normalized, req.opponent_normalized, subset);

        if let Some((cached_penta, cached_c, cached_b, cached_pairs)) = self.try_get(key)
            && cached_pairs >= req.min_pairs
        {
            self.inner.lock().hits += 1;
            return (cached_penta, cached_c, cached_b);
        }

        self.inner.lock().misses += 1;

        let (penta, c_nodes, b_nodes) =
            selfplay::run_matches(req.params, req.opponent_params, subset, &self.tc, on_pair);

        // Update cache (quick lock)
        self.insert(key, penta, c_nodes, b_nodes, req.min_pairs);

        (penta, c_nodes, b_nodes)
    }

    /// Returns (`total_entries`, `total_pairs_cached`, `hits`, `misses`).
    ///
    /// # Panics
    /// if the lock is poisoned.
    #[must_use]
    pub fn stats(&self) -> (usize, usize, usize, usize) {
        let inner = self.inner.lock();
        let total_pairs = inner.entries.values().map(|e| e.pairs).sum();
        (inner.entries.len(), total_pairs, inner.hits, inner.misses)
    }
}
