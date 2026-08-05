//! Uniform random permutation of `0..n`, parallel and cache-blocked.
//!
//! A serial Fisher-Yates spends one DRAM access per swap once the array outgrows L3, and
//! leaves every core but one idle. Bucketing first fixes both: the swaps happen inside buckets
//! whose concurrent working set stays in cache, and every phase runs across the pool.
//!
//! Drawing an iid uniform bucket for each element and then ordering by bucket, ties broken
//! uniformly at random, is the assign-keys-and-sort construction of a random permutation, so
//! every one of the `n!` orderings stays equally likely whatever the bucket count.
//!
//! Bucket and task counts follow from `n`, so the permutation depends on the seed alone: a
//! checkpoint resumed on another host draws the same batches from it.

use fastrand::Rng;
use rayon::prelude::*;

/// Elements per bucket. The sweep below finds a wide flat plateau, so this only has to land
/// inside it; anything from a few million positions up clamps to `MAX_BUCKETS`.
const TARGET_BUCKET: usize = 1 << 15;

/// Enough buckets to keep the per-bucket phase parallel on a small dataset.
const MIN_BUCKETS: usize = 16;

/// Bucket ids are `u8`, and measurement says keep it that way: `u16` ids buy bucket counts
/// past 256, which time no better, and the doubled side table costs 20% on its own.
const MAX_BUCKETS: usize = 256;

/// Elements per counting and scattering task.
const TASK: usize = 1 << 16;

/// Stream separator for the two generators, so the per-bucket draws cannot echo the
/// per-task draws that decided the buckets in the first place. Any odd constant distinct from
/// the gamma in `mix` does the job.
const BUCKET_DOMAIN: u64 = 0xD1B5_4A32_D192_ED03;

/// Scratch for one dataset's permutations, sized once and reused every epoch.
pub struct Shuffler {
    /// The bucket each position landed in. Materialized rather than replayed, so the counting
    /// pass and the scattering pass cannot disagree about where an element goes.
    ids: Vec<u8>,
    /// The block permutation [`Shuffler::fill_blocked`] draws before expanding it.
    order: Vec<u32>,
}

impl Shuffler {
    pub fn new(len: usize) -> Self {
        Self { ids: vec![0; len], order: Vec::new() }
    }

    /// Fills `out` with a uniform random permutation of `0..out.len()`.
    ///
    /// # Panics
    /// If `out` is longer than the length this was constructed for.
    pub fn fill(&mut self, out: &mut [u32], seed: u64) {
        self.fill_into(out, seed, bucket_count(out.len()));
    }

    /// Permutes blocks of `block` consecutive indices instead of the indices themselves.
    ///
    /// The entry list is shuffled once at load, before features are extracted, so a run of
    /// consecutive records is already a random sample of games and a batch drawn from whole
    /// blocks is still unbiased. What it gives up is the fresh partition a full permutation
    /// buys every epoch: positions inside a block travel together for the whole run, and the
    /// blocks themselves are the same on every seed, since the load-time shuffle is seeded by
    /// the fixed split.
    ///
    /// What it buys is sequential reads. The gather over `FeatureRecord`s is DRAM latency per
    /// record, and a block turns that back into a stream the prefetcher can follow.
    ///
    /// `take` stops the expansion there, leaving the rest of `out` alone. The order is still
    /// drawn over every block, so the result is a uniform sample and not the front of a shorter
    /// draw; the last block written is cut to length.
    pub fn fill_blocked(&mut self, out: &mut [u32], seed: u64, block: usize, take: usize) {
        let n = out.len();
        let take = take.min(n);
        let blocks = n.div_ceil(block.max(1));

        // Lifted out so the block permutation can borrow the same `ids` scratch, and put
        // back for the next epoch: at a block of 4 this vector is most of the dataset.
        let mut order = std::mem::take(&mut self.order);
        order.resize(blocks, 0);

        self.fill(&mut order[..blocks], seed);

        let mut w = 0;

        'expand: for &b in &order[..blocks] {
            let start = b as usize * block;

            for i in start..(start + block).min(n) {
                if w == take {
                    break 'expand;
                }

                out[w] = i as u32;
                w += 1;
            }
        }

        self.order = order;
    }

    fn fill_into(&mut self, out: &mut [u32], seed: u64, buckets: usize) {
        let n = out.len();
        assert!(self.ids.len() >= n, "shuffler built for a shorter length");
        assert!(u32::try_from(n).is_ok(), "more elements than a u32 index can name");

        if n < 2 {
            for (j, slot) in out.iter_mut().enumerate() {
                *slot = j as u32;
            }

            return;
        }

        let tasks = n.div_ceil(TASK);
        let mask = (buckets - 1) as u32;

        // Draw a bucket per element, counting as we go. The counts are what let the scatter
        // write straight into `out` instead of staging the permutation somewhere first.
        let mut counts = vec![0u32; tasks * buckets];

        self.ids[..n]
            .par_chunks_mut(TASK)
            .zip(counts.par_chunks_mut(buckets))
            .enumerate()
            .for_each(|(t, (ids, row))| {
                let mut rng = Rng::with_seed(mix(seed, t as u64));

                for id in ids.iter_mut() {
                    // Power-of-two bucket count, so masking is exactly uniform.
                    let b = rng.u32(..) & mask;
                    *id = b as u8;
                    row[b as usize] += 1;
                }
            });

        // Hand every (bucket, task) pair its own piece of `out`, laid out bucket-major so a
        // bucket ends up contiguous and can be permuted in place below. The pieces are sized
        // from the counts, so they tile `out` exactly and no two tasks can reach the same slot.
        let mut slots: Vec<Vec<&mut [u32]>> = (0..tasks).map(|_| Vec::with_capacity(buckets)).collect();
        let mut rest = &mut out[..];

        for b in 0..buckets {
            for (t, task_slots) in slots.iter_mut().enumerate() {
                let (head, tail) = std::mem::take(&mut rest).split_at_mut(counts[t * buckets + b] as usize);
                task_slots.push(head);
                rest = tail;
            }
        }

        self.ids[..n]
            .par_chunks(TASK)
            .zip(slots.par_iter_mut())
            .enumerate()
            .for_each(|(t, (ids, task_slots))| {
                let base = (t * TASK) as u32;
                let mut cursor = vec![0usize; buckets];

                for (k, &id) in ids.iter().enumerate() {
                    let b = id as usize;
                    task_slots[b][cursor[b]] = base + k as u32;
                    cursor[b] += 1;
                }
            });

        drop(slots);

        let mut parts: Vec<&mut [u32]> = Vec::with_capacity(buckets);
        let mut rest = &mut out[..];

        for b in 0..buckets {
            let size = (0..tasks).map(|t| counts[t * buckets + b] as usize).sum();
            let (head, tail) = std::mem::take(&mut rest).split_at_mut(size);
            parts.push(head);
            rest = tail;
        }

        parts.par_iter_mut().enumerate().for_each(|(b, part)| {
            let mut rng = Rng::with_seed(mix(seed ^ BUCKET_DOMAIN, b as u64));
            rng.shuffle(&mut part[..]);
        });
    }
}

fn bucket_count(n: usize) -> usize {
    n.div_ceil(TARGET_BUCKET).next_power_of_two().clamp(MIN_BUCKETS, MAX_BUCKETS)
}

/// SplitMix64 (Steele, Lea and Flood 2014), so neighboring task indices give unrelated
/// generator states. The gamma is the golden-ratio constant, 2^64/φ forced odd.
fn mix(seed: u64, stream: u64) -> u64 {
    let mut z = seed.wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Not a multiple of `TASK` and not a power of two, so the tail task and a ragged final
    /// bucket both get exercised.
    const RAGGED: usize = 1_000_003;

    #[test]
    fn fills_a_complete_permutation() {
        let mut shuffler = Shuffler::new(RAGGED);
        let mut out = vec![0u32; RAGGED];
        shuffler.fill(&mut out, 0x1234_5678);

        let mut sorted = out.clone();
        sorted.sort_unstable();

        assert!(sorted.iter().copied().eq(0..RAGGED as u32), "output is not a permutation of 0..n");
        assert!(out.iter().copied().ne(0..RAGGED as u32), "output is the identity");
    }

    #[test]
    fn blocked_fill_is_still_a_permutation() {
        for (n, block) in [(1000usize, 64usize), (1000, 7), (64, 64), (5, 8), (1, 4)] {
            let mut out = vec![0u32; n];
            Shuffler::new(n).fill_blocked(&mut out, 0x5EED, block, n);

            let mut seen = out.clone();
            seen.sort_unstable();

            assert!(seen.iter().copied().eq(0..n as u32), "n={n} block={block} is not a permutation");
        }
    }

    #[test]
    fn a_take_is_the_prefix_of_the_whole_draw() {
        const N: usize = 4096;
        const TAKE: usize = 300;

        let mut whole = vec![0u32; N];
        let mut part = vec![0u32; N];

        Shuffler::new(N).fill_blocked(&mut whole, 0xC0FF_EE00, 8, N);
        Shuffler::new(N).fill_blocked(&mut part, 0xC0FF_EE00, 8, TAKE);

        assert_eq!(part[..TAKE], whole[..TAKE], "the take diverged from the draw it is a prefix of");
        assert!(part[TAKE..].iter().all(|&i| i == 0), "the expansion wrote past its take");

        let mut sorted = part[..TAKE].to_vec();
        sorted.sort_unstable();
        sorted.dedup();

        assert_eq!(sorted.len(), TAKE, "the sample repeats a position");
    }

    #[test]
    fn blocked_fill_keeps_each_block_in_order() {
        let mut out = vec![0u32; 512];
        Shuffler::new(512).fill_blocked(&mut out, 0x5EED, 64, 512);

        for run in out.chunks(64) {
            assert!(run.windows(2).all(|w| w[1] == w[0] + 1), "a block came apart: {run:?}");
        }
    }

    #[test]
    fn a_seed_reproduces_its_permutation() {
        let mut shuffler = Shuffler::new(RAGGED);
        let mut first = vec![0u32; RAGGED];
        let mut second = vec![0u32; RAGGED];
        let mut other = vec![0u32; RAGGED];

        shuffler.fill(&mut first, 7);
        shuffler.fill(&mut second, 7);
        shuffler.fill(&mut other, 8);

        assert_eq!(first, second, "same seed gave a different permutation");
        assert_ne!(first, other, "different seeds gave the same permutation");

        // The blocked path reuses one scratch buffer across calls, so its determinism is a
        // property of that reuse rather than of the algorithm alone. A resume draws its
        // batches from the seed and nothing else.
        shuffler.fill_blocked(&mut first, 7, 4, RAGGED);
        shuffler.fill_blocked(&mut second, 7, 4, RAGGED);

        assert_eq!(first, second, "same seed gave a different blocked permutation");
    }

    #[test]
    fn every_ordering_is_equally_likely() {
        // Bucketing is only worth trusting if it did not bias the distribution, and n = 4 is
        // small enough to check all 24 orderings against their expected count directly.
        const TRIALS: usize = 120_000;
        const ORDERINGS: usize = 24;

        let mut shuffler = Shuffler::new(4);
        let mut out = [0u32; 4];
        let mut seen = [0u32; ORDERINGS];

        for trial in 0..TRIALS {
            shuffler.fill(&mut out, trial as u64);
            seen[rank(&out)] += 1;
        }

        let expected = TRIALS as f64 / ORDERINGS as f64;
        let chi2: f64 = seen.iter().map(|&c| (f64::from(c) - expected).powi(2) / expected).sum();

        // 23 degrees of freedom puts the 0.999 quantile at 49.7.
        // The seeds are fixed, so a failure here is a real skew and never a flake.
        assert!(chi2 < 49.7, "chi-square {chi2:.1} over {ORDERINGS} orderings, distribution is skewed");
    }

    /// Lehmer code of a permutation of 0..4, giving each ordering a distinct index.
    fn rank(perm: &[u32; 4]) -> usize {
        const FACTORIAL: [usize; 4] = [6, 2, 1, 1];

        (0..4)
            .map(|i| perm[i + 1..].iter().filter(|&&v| v < perm[i]).count() * FACTORIAL[i])
            .sum()
    }
}

/// Where `TARGET_BUCKET` comes from, and how to move it on other hardware.
///
/// `cargo test -p tuner --release --lib sweep -- --ignored --nocapture`, on an idle box: a run
/// sharing the cores moves the tail by more than the tail's own spread. Too few buckets and the
/// swaps fall out of cache, too many and the scatter keeps more write streams open than the
/// store buffers can track.
///
/// Ryzen 7 7735HS, 8 cores, 16 MB L3, ms per shuffle:
///
/// ```text
///                16     32     64    128    256
/// n = 6.39M    6.20   5.80   4.87   4.36   4.25
/// n = 32.8M   94.65  72.16  40.44  26.73  26.67
/// ```
#[cfg(test)]
mod sweep {
    use std::time::Instant;

    use super::*;

    #[test]
    #[ignore]
    fn time_bucket_counts() {
        for n in [6_390_000, 32_815_897] {
            let mut shuffler = Shuffler::new(n);
            let mut out = vec![0u32; n];

            println!("n = {n}, default {} buckets", bucket_count(n));

            for buckets in [16, 32, 64, 128, 256] {
                shuffler.fill_into(&mut out, 1, buckets);

                let t = Instant::now();

                for r in 0..5 {
                    shuffler.fill_into(&mut out, r, buckets);
                }
                println!("  buckets {buckets:>4}: {:.2} ms/shuffle", t.elapsed().as_secs_f64() * 1000.0 / 5.0);
            }
        }
    }
}
