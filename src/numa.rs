//! NUMA topology detection and thread binding.
//!
//! A many-threaded search on a multi-socket box pays for memory it doesn't place
//! and caches it lets go cold. Two localities matter, at two granularities:
//!
//! - The NUMA node is the memory domain, where a memory controller lives. The
//!   transposition table's pages want to spread across these so no one controller
//!   serves every thread. This is the unit `/sys/devices/system/node/` reports.
//! - The L3 domain (a CCX on AMD) is the cache domain, finer, sitting inside a
//!   node. A thread pinned to one keeps its own hot state in a warm L3. On an EPYC
//!   under NPS1 the kernel folds a whole socket into one node and hides these, so
//!   they're read from the cache topology instead.
//!
//! Bind a thread to its L3 domain and it lands cache-local and memory-local at
//! once, since the L3 is inside the node. Detection is plain `/sys` file reads, so
//! it works on any Linux and needs no `libc`; only the binding itself is a syscall,
//! and that path is x86-64 Linux. Everywhere else the whole module degrades to a
//! single domain and binding becomes a no-op.
//!
//! Two refinements stay out until a box asks for them, and they're noted here so
//! they don't get lost: the SLIT distance matrix (`node{N}/distance`), worth reading
//! only once placement is distance-aware across far-apart nodes, and L3 bundling
//! (SF's `BundledL3Policy`), worth adding only when a machine splinters into so many
//! tiny domains that binding to each stops helping. Detecting either ahead of a
//! consumer is structure nothing reads, so each waits for the placement that needs it.
//!
//! Further out, the L3 domains are also the natural unit for a per-domain copy of a
//! shared correction history, the moment corrhist becomes shared at all, which is an
//! experiment in its own right. The domains are detected and ready when it lands.

use std::{fmt, fs, thread};

/// A logical CPU index, as the kernel numbers it.
type Cpu = usize;

/// The machine's locality map: memory domains and the finer cache domains that
/// subdivide them. Both partition the CPUs the process is actually allowed on.
pub struct NumaTopology {
    /// NUMA nodes, the memory domains. The TT spreads its pages across these.
    nodes: Vec<Vec<Cpu>>,
    /// L3 domains, the cache domains. Search threads bind to these. Falls back to
    /// `nodes` when the cache topology can't be read.
    domains: Vec<Vec<Cpu>>,
}

impl NumaTopology {
    /// Read the topology from the running system. It collapses to a single domain
    /// only when `/sys` is unreadable, as off Linux. On Linux it reads the nodes and
    /// L3 domains, including the NPS1 EPYC that shows one node over many caches.
    pub fn detect() -> Self {
        let allowed = allowed_cpus();
        let nodes = read_numa_nodes(&allowed).unwrap_or_else(|| vec![allowed.clone()]);
        // ♪ numa numa iei ♪
        let domains = read_l3_domains(&allowed).unwrap_or_else(|| nodes.clone());
        Self { nodes, domains }
    }

    /// Number of memory domains. The TT clear spreads its slices across this many.
    pub fn num_nodes(&self) -> usize { self.nodes.len() }

    /// Number of cache domains. Threads distribute across this many.
    pub fn num_domains(&self) -> usize { self.domains.len() }

    /// Whether binding earns its keep: more than one domain to spread over, and
    /// more than one thread to spread.
    pub fn should_bind(&self, threads: usize) -> bool { self.domains.len() > 1 && threads > 1 }

    /// Whether spreading the TT across nodes pays: more than one memory domain to
    /// spread over, and more than one thread to share the bandwidth. A lone thread
    /// wants its table local, not striped across remote controllers.
    pub fn should_distribute(&self, threads: usize) -> bool { self.nodes.len() > 1 && threads > 1 }

    /// Assign `threads` workers to L3 domains, one index per worker, balanced by
    /// fill so a domain with more CPUs takes proportionally more threads and no
    /// node is favored (which would fight other instances sharing the box).
    pub fn distribute(&self, threads: usize) -> Vec<usize> {
        let mut assignment = Vec::with_capacity(threads);
        let mut occupied = vec![0usize; self.domains.len().max(1)];

        for _ in 0..threads {
            let pick = (0..self.domains.len())
                .min_by(|&a, &b| fill(occupied[a], &self.domains[a]).total_cmp(&fill(occupied[b], &self.domains[b])))
                .unwrap_or(0);

            occupied[pick] += 1;
            assignment.push(pick);
        }
        assignment
    }

    /// Pin the calling thread to a cache domain. Returns whether the bind took.
    pub fn bind_to_domain(&self, domain: usize) -> bool { self.domains.get(domain).is_some_and(|cpus| sys::bind_thread(cpus)) }

    /// Pin the calling thread to a memory domain, for the threads that first-touch
    /// the TT's slices into place. Returns whether the bind took.
    pub fn bind_to_node(&self, node: usize) -> bool { self.nodes.get(node).is_some_and(|cpus| sys::bind_thread(cpus)) }
}

impl fmt::Display for NumaTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} NUMA node(s), {} L3 domain(s)", self.nodes.len(), self.domains.len())
    }
}

/// Fill ratio of a domain once one more thread joins it. Lower means more room.
fn fill(occupied: usize, domain: &[Cpu]) -> f64 { (occupied + 1) as f64 / domain.len().max(1) as f64 }

/// The CPUs the process may run on: its affinity mask if the kernel will tell us,
/// else every online CPU. Respecting the mask keeps an instance pinned by `taskset`
/// from binding onto CPUs it was fenced off from, which is how several instances
/// share a box without fighting for the same cores.
fn allowed_cpus() -> Vec<Cpu> {
    if let Some(mask) = sys::process_affinity() {
        return mask;
    }

    if let Ok(online) = fs::read_to_string("/sys/devices/system/cpu/online") {
        let cpus = parse_cpu_list(online.trim());
        if !cpus.is_empty() {
            return cpus;
        }
    }

    let n = thread::available_parallelism().map_or(1, |n| n.get());
    (0..n).collect()
}

/// The NUMA nodes, each the allowed CPUs the kernel places on that node. `None`
/// when `/sys` has nothing to say, leaving the caller to assume one node.
fn read_numa_nodes(allowed: &[Cpu]) -> Option<Vec<Vec<Cpu>>> {
    let online = fs::read_to_string("/sys/devices/system/node/online").ok()?;
    let mut nodes = Vec::new();
    for node in parse_cpu_list(online.trim()) {
        let list = fs::read_to_string(format!("/sys/devices/system/node/node{node}/cpulist")).ok()?;
        let cpus: Vec<Cpu> = parse_cpu_list(list.trim()).into_iter().filter(|c| allowed.contains(c)).collect();
        if !cpus.is_empty() {
            nodes.push(cpus);
        }
    }
    (!nodes.is_empty()).then_some(nodes)
}

/// The L3 domains: each allowed CPU grouped with the CPUs it shares an L3 with.
/// `None` when the cache topology is unreadable, so binding falls back to nodes.
fn read_l3_domains(allowed: &[Cpu]) -> Option<Vec<Vec<Cpu>>> {
    let ceiling = allowed.iter().copied().max().map_or(0, |m| m + 1);
    let mut grouped = vec![false; ceiling];
    let mut domains = Vec::new();

    for &cpu in allowed {
        if grouped[cpu] {
            continue;
        }

        let siblings = read_l3_siblings(cpu)?;
        let group: Vec<Cpu> = parse_cpu_list(&siblings).into_iter().filter(|c| allowed.contains(c)).collect();

        for &c in &group {
            grouped[c] = true;
        }

        if !group.is_empty() {
            domains.push(group);
        }
    }
    (!domains.is_empty()).then_some(domains)
}

/// The `shared_cpu_list` of a CPU's level-3 cache. The cache index for L3 isn't
/// fixed (it's whichever index reports `level` 3), so we scan for it.
fn read_l3_siblings(cpu: Cpu) -> Option<String> {
    for index in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{index}");
        let Ok(level) = fs::read_to_string(format!("{base}/level")) else {
            break;
        };

        if level.trim() == "3" {
            return fs::read_to_string(format!("{base}/shared_cpu_list")).ok().map(|s| s.trim().to_string());
        }
    }
    None
}

/// Parse a kernel cpulist (`"0-15,128-143"`, `"0,2,4"`) into its CPU indices.
fn parse_cpu_list(s: &str) -> Vec<Cpu> {
    let mut cpus = Vec::new();

    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }

        match part.split_once('-') {
            Some((lo, hi)) => {
                if let (Ok(lo), Ok(hi)) = (lo.parse::<Cpu>(), hi.parse::<Cpu>()) {
                    cpus.extend(lo..=hi);
                }
            },
            None => {
                if let Ok(c) = part.parse::<Cpu>() {
                    cpus.push(c);
                }
            },
        }
    }
    cpus
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod sys {
    use super::Cpu;

    const SYS_SCHED_YIELD: usize = 24;
    const SYS_SCHED_SETAFFINITY: usize = 203;
    const SYS_SCHED_GETAFFINITY: usize = 204;

    /// The calling thread's allowed CPUs, decoded from its affinity bitmask, or
    /// `None` if the call fails. A 512-byte buffer covers 4096 CPUs, past any real
    /// machine, so the kernel never truncates the mask into it.
    pub fn process_affinity() -> Option<Vec<Cpu>> {
        let mut mask = [0u64; 64];

        // SAFETY: sched_getaffinity(0, bytes, ptr) fills the calling thread's mask
        // into `mask` for `bytes`; the buffer is `mask.len() * 8` and lives here.
        let ret = unsafe { syscall3(SYS_SCHED_GETAFFINITY, 0, mask.len() * 8, mask.as_mut_ptr() as usize) };
        if ret < 0 {
            return None;
        }

        let mut cpus = Vec::new();

        for (word, &bits) in mask.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                cpus.push(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1; // clear the lowest set bit
            }
        }
        Some(cpus)
    }

    /// Pin the calling thread to `cpus`, then yield so the scheduler acts on the
    /// new mask now rather than at the next natural preemption. Returns the bind's
    /// success; a failure is best-effort-ignored, leaving the thread unpinned.
    pub fn bind_thread(cpus: &[Cpu]) -> bool {
        let Some(&highest) = cpus.iter().max() else {
            return false;
        };

        let words = highest / 64 + 1;
        let mut mask = vec![0u64; words];

        for &cpu in cpus {
            mask[cpu / 64] |= 1u64 << (cpu % 64);
        }

        // SAFETY: sched_setaffinity(0, bytes, ptr) reads bytes of mask for the
        // calling thread (pid 0); bytes is words * 8, exactly the buffer.
        let ret = unsafe { syscall3(SYS_SCHED_SETAFFINITY, 0, words * 8, mask.as_ptr() as usize) };
        if ret != 0 {
            return false;
        }
        // SAFETY: sched_yield takes no arguments and only reschedules.
        unsafe { syscall3(SYS_SCHED_YIELD, 0, 0, 0) };
        true
    }

    /// # Safety
    /// A valid x86-64 Linux syscall: `n` and its arguments must form a sound call,
    /// and any pointer argument must satisfy that syscall's requirements.
    unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> isize {
        let ret: isize;
        // SAFETY: x86-64 Linux ABI: number in rax, args in rdi/rsi/rdx, result in
        // rax; the instruction clobbers rcx and r11.
        unsafe {
            std::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        ret
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod sys {
    use super::Cpu;

    pub fn process_affinity() -> Option<Vec<Cpu>> { None }

    pub fn bind_thread(_cpus: &[Cpu]) -> bool { false }
}

#[cfg(test)]
mod tests {
    use super::{NumaTopology, parse_cpu_list};

    #[test]
    fn parses_ranges_and_singles() {
        assert_eq!(parse_cpu_list("0-3"), vec![0, 1, 2, 3]);
        assert_eq!(parse_cpu_list("0,2,4"), vec![0, 2, 4]);
        assert_eq!(parse_cpu_list(""), Vec::<usize>::new());
    }

    #[test]
    fn parses_an_epyc_node_cpulist() {
        // EPYC 9654 node 0: physical cores plus their SMT siblings, two blocks.
        let cpus = parse_cpu_list("0-95,192-287");
        assert_eq!(cpus.len(), 192);
        assert_eq!(cpus.first(), Some(&0));
        assert_eq!(cpus.last(), Some(&287));
        assert!(!cpus.contains(&96));
    }

    #[test]
    fn distribute_balances_by_fill() {
        let topo = NumaTopology { nodes: vec![(0..32).collect()], domains: vec![(0..16).collect(), (16..32).collect()] };
        let assignment = topo.distribute(4);
        assert_eq!(assignment.iter().filter(|&&d| d == 0).count(), 2);
        assert_eq!(assignment.iter().filter(|&&d| d == 1).count(), 2);
    }

    #[test]
    fn single_domain_never_binds() {
        let topo = NumaTopology { nodes: vec![(0..8).collect()], domains: vec![(0..8).collect()] };
        assert!(!topo.should_bind(8));
        assert_eq!(topo.distribute(4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn detect_is_sane_on_this_box() {
        // Exercises the live `/sys` reads and the affinity syscall end to end.
        // Whatever this machine is, detection always lands at least one domain.
        let topo = NumaTopology::detect();
        assert!(topo.num_nodes() >= 1);
        assert!(topo.num_domains() >= 1);
    }
}
