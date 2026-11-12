//! Cache-blocked parallel uniform permutation of `0..n`.
//!
//! Independent uniform bucket keys followed by parallel in-cache Fisher-Yates shuffles yield an
//! exact uniform draw over all `n!` orderings; fixed task and bucket geometry makes the permutation
//! reproducible across hosts for a given seed.

use fastrand::Rng;
use rayon::prelude::*;

const TARGET_BUCKET: usize = 1 << 15;
const MIN_BUCKETS: usize = 16;
const MAX_BUCKETS: usize = 256;
const BUCKET_DOMAIN: u64 = 0xD1B5_4A32_D192_ED03;
const TASK_SIZE: usize = 1 << 16;

pub(crate) struct Shuffler {
    bucket_ids:  Vec<u8>,
    block_order: Vec<u32>,
}

impl Shuffler {
    pub(crate) fn new(len: usize) -> Self { Self { bucket_ids: vec![0; len], block_order: Vec::new() } }
    pub(crate) fn fill(&mut self, out: &mut [u32], seed: u64) { self.fill_into(out, seed, bucket_count(out.len())); }

    /// Permutes contiguous blocks of `block_size`, expanding only the first `take` indices.
    pub(crate) fn fill_blocked(&mut self, out: &mut [u32], seed: u64, block_size: usize, take: usize) {
        let (len, target, block) = (out.len(), take.min(out.len()), block_size.max(1));
        let num_blocks = len.div_ceil(block);

        let mut order = std::mem::take(&mut self.block_order);
        order.resize(num_blocks, 0);
        self.fill(&mut order[..num_blocks], seed);

        let expanded = order[..num_blocks].iter().flat_map(|&b| {
            let start = b as usize * block;
            start..(start + block).min(len)
        });

        for (slot, idx) in out[..target].iter_mut().zip(expanded) {
            *slot = idx as u32;
        }

        self.block_order = order;
    }

    fn fill_into(&mut self, out: &mut [u32], seed: u64, buckets: usize) {
        let len = out.len();
        assert!(self.bucket_ids.len() >= len && u32::try_from(len).is_ok() && buckets.is_power_of_two() && buckets <= MAX_BUCKETS);

        if len < 2 {
            for (idx, slot) in out.iter_mut().enumerate() {
                *slot = idx as u32;
            }
            return;
        }

        let num_tasks = len.div_ceil(TASK_SIZE);
        let mask = (buckets - 1) as u32;
        let mut counts = vec![0u32; num_tasks * buckets];

        self.bucket_ids[..len]
            .par_chunks_mut(TASK_SIZE)
            .zip(counts.par_chunks_mut(buckets))
            .enumerate()
            .for_each(|(task, (ids, hist))| {
                let mut rng = Rng::with_seed(splitmix64(seed, task as u64));
                for slot in ids.iter_mut() {
                    let b = rng.u32(..) & mask;
                    *slot = b as u8;
                    hist[b as usize] += 1;
                }
            });

        let mut dests: Vec<Vec<&mut [u32]>> = (0..num_tasks).map(|_| Vec::with_capacity(buckets)).collect();
        let mut rest = &mut out[..];

        for b in 0..buckets {
            for (task, slices) in dests.iter_mut().enumerate() {
                slices.push(take_slice(&mut rest, counts[task * buckets + b] as usize));
            }
        }

        self.bucket_ids[..len]
            .par_chunks(TASK_SIZE)
            .zip(dests.par_iter_mut())
            .enumerate()
            .for_each(|(task, (ids, slices))| {
                let base = (task * TASK_SIZE) as u32;
                let mut cursors = [0usize; MAX_BUCKETS];
                for (off, &id) in ids.iter().enumerate() {
                    let b = id as usize;
                    slices[b][cursors[b]] = base + off as u32;
                    cursors[b] += 1;
                }
            });
        drop(dests);

        let mut bucket_slices: Vec<&mut [u32]> = Vec::with_capacity(buckets);
        let mut rest = &mut out[..];

        for b in 0..buckets {
            let bucket_len: usize = (0..num_tasks).map(|t| counts[t * buckets + b] as usize).sum();
            bucket_slices.push(take_slice(&mut rest, bucket_len));
        }

        bucket_slices.par_iter_mut().enumerate().for_each(|(b, slice)| {
            Rng::with_seed(splitmix64(seed ^ BUCKET_DOMAIN, b as u64)).shuffle(&mut slice[..]);
        });
    }
}

#[inline]
fn take_slice<'a, T>(rest: &mut &'a mut [T], n: usize) -> &'a mut [T] {
    let (head, tail) = std::mem::take(rest).split_at_mut(n);
    *rest = tail;
    head
}

fn bucket_count(len: usize) -> usize { len.div_ceil(TARGET_BUCKET).next_power_of_two().clamp(MIN_BUCKETS, MAX_BUCKETS) }

fn splitmix64(seed: u64, stream: u64) -> u64 {
    let mut z = seed.wrapping_add(stream.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn is_bijection(slice: &[u32]) -> bool {
        let mut s = slice.to_vec();
        s.sort_unstable();
        s.into_iter().eq(0..slice.len() as u32)
    }

    #[test]
    fn full_and_blocked_permutations_are_bijective_reproducible_and_prefix_consistent() {
        const N: usize = 1_000_003;
        let mut shuffler = Shuffler::new(N);
        let (mut first, mut second, mut other) = (vec![0u32; N], vec![0u32; N], vec![0u32; N]);

        shuffler.fill(&mut first, 7);
        shuffler.fill(&mut second, 7);
        shuffler.fill(&mut other, 8);
        assert!(first == second && first != other && is_bijection(&first));

        shuffler.fill_blocked(&mut first, 7, 4, N);
        shuffler.fill_blocked(&mut second, 7, 4, N);
        assert_eq!(first, second);

        for (n, block) in [(1000usize, 64usize), (1000, 7), (64, 64), (5, 8), (1, 4)] {
            let mut out = vec![0u32; n];
            Shuffler::new(n).fill_blocked(&mut out, 0x5EED, block, n);
            assert!(is_bijection(&out), "n={n}, block={block}");
        }

        let mut out = vec![0u32; 512];
        Shuffler::new(512).fill_blocked(&mut out, 0x5EED, 64, 512);
        assert!(out.chunks(64).all(|run| run.windows(2).all(|w| w[1] == w[0] + 1)));

        let (mut whole, mut part) = (vec![0u32; 4096], vec![u32::MAX; 4096]);
        Shuffler::new(4096).fill_blocked(&mut whole, 0xC0FF_EE00, 8, 4096);
        Shuffler::new(4096).fill_blocked(&mut part, 0xC0FF_EE00, 8, 300);
        assert!(part[..300] == whole[..300] && part[300..].iter().all(|&i| i == u32::MAX));
    }

    #[test]
    fn uniform_over_all_orderings() {
        const TRIALS: usize = 120_000;
        const FACTORIAL: [usize; 4] = [6, 2, 1, 1];

        let (mut shuffler, mut out, mut seen) = (Shuffler::new(4), [0u32; 4], [0u32; 24]);
        for trial in 0..TRIALS {
            shuffler.fill(&mut out, trial as u64);
            let code: usize = (0..4).map(|i| out[i + 1..].iter().filter(|&&v| v < out[i]).count() * FACTORIAL[i]).sum();
            seen[code] += 1;
        }
        let expected = TRIALS as f64 / 24.0;
        let chi2: f64 = seen.iter().map(|&c| (f64::from(c) - expected).powi(2) / expected).sum();
        // 23 df sets the 0.999 quantile at 49.7.
        assert!(chi2 < 49.7, "chi-square {chi2:.1}");
    }

    #[test]
    #[ignore]
    fn time_bucket_counts() {
        for n in [6_390_000, 32_815_897] {
            let (mut shuffler, mut out) = (Shuffler::new(n), vec![0u32; n]);
            println!("n = {n}, default {} buckets", bucket_count(n));
            for buckets in [16, 32, 64, 128, 256] {
                shuffler.fill_into(&mut out, 1, buckets);
                let start = std::time::Instant::now();
                for r in 0..5 {
                    shuffler.fill_into(&mut out, r, buckets);
                }
                println!("  buckets {buckets:>4}: {:.2} ms/shuffle", start.elapsed().as_secs_f64() * 1000.0 / 5.0);
            }
        }
    }
}
