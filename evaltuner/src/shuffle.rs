//! Uniform random permutation of `0..n`, parallelized and cache-blocked.
//!
//! Standard serial Fisher-Yates incurs an un-cached DRAM access per swap once
//! the target slice exceeds L3 cache capacity. This implementation partitions elements
//! across uniform buckets in parallel before executing in-cache Fisher-Yates shuffles
//! within each bucket.
//!
//! Assigning independent uniform bucket keys followed by uniform intra-bucket shuffles
//! preserves an exact uniform distribution over all `n!` permutations. Task and bucket
//! partitions are deterministically derived from slice length, ensuring cross-host
//! reproducibility given identical seeds.

use fastrand::Rng;
use rayon::prelude::*;

const TARGET_BUCKET: usize = 1 << 15;
const MIN_BUCKETS: usize = 16;
const MAX_BUCKETS: usize = 256;
const BUCKET_DOMAIN: u64 = 0xD1B5_4A32_D192_ED03;

/// Element chunk size processed per parallel task during classification and scatter passes.
const TASK_SIZE: usize = 1 << 16;

/// Reusable scratch buffers for generating cache-blocked random permutations.
pub struct Shuffler {
    /// Materialized bucket assignments per element.
    bucket_ids: Vec<u8>,
    /// Scratch buffer for coarse-grained block permutations.
    block_order: Vec<u32>,
}

impl Shuffler {
    #[must_use]
    pub fn new(len: usize) -> Self { Self { bucket_ids: vec![0; len], block_order: Vec::new() } }

    /// Fills `out` with a uniform random permutation of `0..out.len()`.
    ///
    /// # Panics
    /// Panics if `out` exceeds the capacity allocated during construction, or exceeds `u32::MAX`.
    pub fn fill(&mut self, out: &mut [u32], seed: u64) { self.fill_into(out, seed, bucket_count(out.len())); }

    /// Permutes consecutive contiguous blocks of `block_size` elements.
    ///
    /// `take` limits output expansion to a prefix length while drawing the block
    /// order across all blocks, yielding a uniform subsample without requiring full array expansion.
    pub fn fill_blocked(&mut self, out: &mut [u32], seed: u64, block_size: usize, take: usize) {
        let total_elements = out.len();
        let target_len = take.min(total_elements);
        let effective_block_size = block_size.max(1);
        let num_blocks = total_elements.div_ceil(effective_block_size);

        // Temporarily extract the order buffer to reuse self for permutation generation.
        let mut block_order = std::mem::take(&mut self.block_order);
        block_order.resize(num_blocks, 0);

        self.fill(&mut block_order[..num_blocks], seed);

        let mut written = 0;
        'expand: for &block_idx in &block_order[..num_blocks] {
            let start = block_idx as usize * effective_block_size;
            let end = (start + effective_block_size).min(total_elements);

            for idx in start..end {
                if written == target_len {
                    break 'expand;
                }
                out[written] = idx as u32;
                written += 1;
            }
        }

        self.block_order = block_order;
    }

    fn fill_into(&mut self, out: &mut [u32], seed: u64, buckets: usize) {
        let total_len = out.len();
        assert!(self.bucket_ids.len() >= total_len, "Shuffler capacity smaller than target slice");
        assert!(u32::try_from(total_len).is_ok(), "Element count exceeds u32 indexing limit");
        assert!(
            buckets.is_power_of_two() && buckets <= MAX_BUCKETS,
            "Bucket count ({buckets}) must be a power of two <= {MAX_BUCKETS}"
        );

        if total_len < 2 {
            for (idx, slot) in out.iter_mut().enumerate() {
                *slot = idx as u32;
            }
            return;
        }

        let num_tasks = total_len.div_ceil(TASK_SIZE);
        let bucket_mask = (buckets - 1) as u32;

        // Compute per-task bucket histograms to determine exact disjoint partition boundaries.
        let mut task_bucket_counts = vec![0u32; num_tasks * buckets];

        self.bucket_ids[..total_len]
            .par_chunks_mut(TASK_SIZE)
            .zip(task_bucket_counts.par_chunks_mut(buckets))
            .enumerate()
            .for_each(|(task_idx, (task_bucket_ids, task_histogram))| {
                let mut rng = Rng::with_seed(splitmix64(seed, task_idx as u64));
                for slot in task_bucket_ids.iter_mut() {
                    let bucket = rng.u32(..) & bucket_mask;
                    *slot = bucket as u8;
                    task_histogram[bucket as usize] += 1;
                }
            });

        // Partition the output buffer into disjoint mutable slices per (task, bucket) pair.
        let mut task_destinations: Vec<Vec<&mut [u32]>> = (0..num_tasks).map(|_| Vec::with_capacity(buckets)).collect();
        let mut remaining_out = &mut out[..];

        for bucket in 0..buckets {
            for (task_idx, task_slices) in task_destinations.iter_mut().enumerate() {
                let count = task_bucket_counts[task_idx * buckets + bucket] as usize;
                let (head, tail) = std::mem::take(&mut remaining_out).split_at_mut(count);
                task_slices.push(head);
                remaining_out = tail;
            }
        }

        // Scatter input indices into bucket-major output segments.
        self.bucket_ids[..total_len]
            .par_chunks(TASK_SIZE)
            .zip(task_destinations.par_iter_mut())
            .enumerate()
            .for_each(|(task_idx, (task_bucket_ids, task_slices))| {
                let base_index = (task_idx * TASK_SIZE) as u32;
                let mut cursors = [0usize; MAX_BUCKETS];

                for (offset, &bucket_id) in task_bucket_ids.iter().enumerate() {
                    let bucket = bucket_id as usize;
                    task_slices[bucket][cursors[bucket]] = base_index + offset as u32;
                    cursors[bucket] += 1;
                }
            });

        drop(task_destinations);

        // Re-slice output contiguously per bucket and perform in-place Fisher-Yates shuffles in parallel.
        let mut bucket_slices: Vec<&mut [u32]> = Vec::with_capacity(buckets);
        let mut remaining_out = &mut out[..];

        for bucket in 0..buckets {
            let bucket_len: usize = (0..num_tasks).map(|t| task_bucket_counts[t * buckets + bucket] as usize).sum();
            let (head, tail) = std::mem::take(&mut remaining_out).split_at_mut(bucket_len);
            bucket_slices.push(head);
            remaining_out = tail;
        }

        bucket_slices.par_iter_mut().enumerate().for_each(|(bucket_idx, slice)| {
            let mut rng = Rng::with_seed(splitmix64(seed ^ BUCKET_DOMAIN, bucket_idx as u64));
            rng.shuffle(&mut slice[..]);
        });
    }
}

fn bucket_count(len: usize) -> usize { len.div_ceil(TARGET_BUCKET).next_power_of_two().clamp(MIN_BUCKETS, MAX_BUCKETS) }

/// The Weyl constant `0x9E37_79B9_7F4A_7C15` is floor(2^64/φ), odd as the additive step needs,
/// so sequential stream indices land on unrelated generator states.
fn splitmix64(seed: u64, stream: u64) -> u64 {
    let mut z = seed.wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RAGGED_LEN: usize = 1_000_003;

    #[test]
    fn fills_a_complete_permutation() {
        let mut shuffler = Shuffler::new(RAGGED_LEN);
        let mut out = vec![0u32; RAGGED_LEN];
        shuffler.fill(&mut out, 0x1234_5678);
        let mut sorted = out.clone();
        sorted.sort_unstable();
        assert!(sorted.iter().copied().eq(0..RAGGED_LEN as u32), "output is not a valid permutation of 0..n");
        assert!(out.iter().copied().ne(0..RAGGED_LEN as u32), "output remained identity");
    }

    #[test]
    fn blocked_fill_is_still_a_permutation() {
        for (n, block) in [(1000usize, 64usize), (1000, 7), (64, 64), (5, 8), (1, 4)] {
            let mut out = vec![0u32; n];
            Shuffler::new(n).fill_blocked(&mut out, 0x5EED, block, n);
            let mut seen = out.clone();
            seen.sort_unstable();
            assert!(seen.iter().copied().eq(0..n as u32), "blocked permutation invalid for n={n}, block={block}");
        }
    }

    #[test]
    fn a_take_is_the_prefix_of_the_whole_draw() {
        const N: usize = 4096;
        const TAKE: usize = 300;

        let mut whole = vec![0u32; N];
        let mut part = vec![u32::MAX; N];
        Shuffler::new(N).fill_blocked(&mut whole, 0xC0FF_EE00, 8, N);
        Shuffler::new(N).fill_blocked(&mut part, 0xC0FF_EE00, 8, TAKE);
        assert_eq!(part[..TAKE], whole[..TAKE], "prefix sub-sample deviated from full blocked draw");
        assert!(part[TAKE..].iter().all(|&i| i == u32::MAX), "expansion wrote beyond requested take length");
        let mut sorted = part[..TAKE].to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), TAKE, "sub-sample contains duplicate indices");
    }

    #[test]
    fn blocked_fill_keeps_each_block_in_order() {
        let mut out = vec![0u32; 512];
        Shuffler::new(512).fill_blocked(&mut out, 0x5EED, 64, 512);
        for run in out.chunks(64) {
            assert!(run.windows(2).all(|w| w[1] == w[0] + 1), "internal block order disrupted: {run:?}");
        }
    }

    #[test]
    fn a_seed_reproduces_its_permutation() {
        let mut shuffler = Shuffler::new(RAGGED_LEN);
        let mut first = vec![0u32; RAGGED_LEN];
        let mut second = vec![0u32; RAGGED_LEN];
        let mut other = vec![0u32; RAGGED_LEN];
        shuffler.fill(&mut first, 7);
        shuffler.fill(&mut second, 7);
        shuffler.fill(&mut other, 8);
        assert_eq!(first, second, "same seed gave a different permutation");
        assert_ne!(first, other, "different seeds gave the same permutation");
        shuffler.fill_blocked(&mut first, 7, 4, RAGGED_LEN);
        shuffler.fill_blocked(&mut second, 7, 4, RAGGED_LEN);
        assert_eq!(first, second, "same seed gave a different blocked permutation");
    }

    #[test]
    fn every_ordering_is_equally_likely() {
        const TRIALS: usize = 120_000;
        const ORDERINGS: usize = 24;

        let mut shuffler = Shuffler::new(4);
        let mut out = [0u32; 4];
        let mut seen = [0u32; ORDERINGS];

        for trial in 0..TRIALS {
            shuffler.fill(&mut out, trial as u64);
            seen[lehmer_code(&out)] += 1;
        }

        let expected = TRIALS as f64 / ORDERINGS as f64;
        let chi2: f64 = seen.iter().map(|&c| (f64::from(c) - expected).powi(2) / expected).sum();
        // 23 degrees of freedom sets the 0.999 quantile at 49.7.
        assert!(chi2 < 49.7, "chi-square {chi2:.1} over {ORDERINGS} orderings indicates skewed distribution");
    }

    /// Computes the Lehmer code index for a permutation of 0..4.
    fn lehmer_code(perm: &[u32; 4]) -> usize {
        const FACTORIAL: [usize; 4] = [6, 2, 1, 1];

        (0..4)
            .map(|i| perm[i + 1..].iter().filter(|&&v| v < perm[i]).count() * FACTORIAL[i])
            .sum()
    }
}

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

                let start = Instant::now();
                for r in 0..5 {
                    shuffler.fill_into(&mut out, r, buckets);
                }
                println!("  buckets {buckets:>4}: {:.2} ms/shuffle", start.elapsed().as_secs_f64() * 1000.0 / 5.0);
            }
        }
    }
}
