//! NUMA topology detection and thread binding.

use std::{fmt, fs, thread};

type Cpu = usize;

/// NUMA memory nodes and L3 cache domains available to this process.
pub(crate) struct NumaTopology {
    nodes: Vec<Vec<Cpu>>,
    l3_domains: Vec<Vec<Cpu>>,
}

impl NumaTopology {
    pub(crate) fn detect() -> Self {
        let allowed = allowed_cpus();
        let nodes = read_numa_nodes(&allowed).unwrap_or_else(|| vec![allowed.clone()]);
        // ♪ numa numa iei ♪
        let l3_domains = read_l3_domains(&allowed).unwrap_or_else(|| nodes.clone());
        Self { nodes, l3_domains }
    }

    pub(crate) fn num_nodes(&self) -> usize { self.nodes.len() }
    #[cfg(test)]
    fn num_l3_domains(&self) -> usize { self.l3_domains.len() }
    pub(crate) fn should_bind_l3(&self, threads: usize) -> bool { self.l3_domains.len() > 1 && threads > 1 }

    /// Don't stripe a single thread's table across NUMA nodes.
    pub(crate) fn should_split_across_nodes(&self, threads: usize) -> bool { self.nodes.len() > 1 && threads > 1 }

    /// Assigns workers to L3 domains proportional to their CPU counts.
    pub(crate) fn assign_l3_domains(&self, threads: usize) -> Vec<usize> {
        let mut assignment = Vec::with_capacity(threads);
        let mut occupied = vec![0usize; self.l3_domains.len().max(1)];

        for _ in 0..threads {
            let pick = (0..self.l3_domains.len())
                .min_by(|&a, &b| {
                    fill_ratio(occupied[a], &self.l3_domains[a]).total_cmp(&fill_ratio(occupied[b], &self.l3_domains[b]))
                })
                .unwrap_or(0);

            occupied[pick] += 1;
            assignment.push(pick);
        }
        assignment
    }

    pub(crate) fn bind_to_l3_domain(&self, domain: usize) -> bool {
        self.l3_domains.get(domain).is_some_and(|cpus| sys::bind_thread(cpus))
    }
    /// Pins the calling thread to a NUMA node for first-touch placement.
    pub(crate) fn bind_to_node(&self, node: usize) -> bool { self.nodes.get(node).is_some_and(|cpus| sys::bind_thread(cpus)) }
}

impl fmt::Display for NumaTopology {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} NUMA node(s), {} L3 domain(s)", self.nodes.len(), self.l3_domains.len())
    }
}

fn fill_ratio(occupied: usize, domain: &[Cpu]) -> f64 { (occupied + 1) as f64 / domain.len().max(1) as f64 }

/// CPUs available to this process, preferring its affinity mask.
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

fn read_l3_domains(allowed: &[Cpu]) -> Option<Vec<Vec<Cpu>>> {
    let ceiling = allowed.iter().copied().max().map_or(0, |m| m + 1);
    let mut grouped = vec![false; ceiling];
    let mut l3_domains = Vec::new();

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
            l3_domains.push(group);
        }
    }
    (!l3_domains.is_empty()).then_some(l3_domains)
}

fn read_l3_siblings(cpu: Cpu) -> Option<String> {
    // Cache indices are not fixed; find the entry whose level is 3.
    for index in 0..8 {
        let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{index}");
        let Ok(level) = fs::read_to_string(format!("{base}/level")) else {
            break;
        };

        if level.trim() == "3" {
            return fs::read_to_string(format!("{base}/shared_cpu_list")).ok();
        }
    }
    None
}

/// Parse a kernel cpulist such as `0-15,128-143`.
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

    /// Calling thread's allowed CPUs, or `None` if the query fails.
    pub(super) fn process_affinity() -> Option<Vec<Cpu>> {
        let mut mask = [0u64; 64]; // 4096 CPUs

        // SAFETY: mask is a writable mask.len() * 8 byte buffer.
        let ret = unsafe { syscall3(SYS_SCHED_GETAFFINITY, 0, mask.len() * 8, mask.as_mut_ptr() as usize) };
        if ret < 0 {
            return None;
        }

        let mut cpus = Vec::new();

        for (word, &bits) in mask.iter().enumerate() {
            let mut bits = bits;
            while bits != 0 {
                cpus.push(word * 64 + bits.trailing_zeros() as usize);
                bits &= bits - 1;
            }
        }
        Some(cpus)
    }

    pub(super) fn bind_thread(cpus: &[Cpu]) -> bool {
        let Some(&highest) = cpus.iter().max() else {
            return false;
        };

        let words = highest / 64 + 1;
        let mut mask = vec![0u64; words];

        for &cpu in cpus {
            mask[cpu / 64] |= 1u64 << (cpu % 64);
        }

        // SAFETY: mask contains words * 8 readable bytes.
        let ret = unsafe { syscall3(SYS_SCHED_SETAFFINITY, 0, words * 8, mask.as_ptr() as usize) };
        if ret != 0 {
            return false;
        }

        // Let the scheduler move the thread onto its new affinity immediately.
        // SAFETY: sched_yield takes no arguments.
        unsafe { syscall3(SYS_SCHED_YIELD, 0, 0, 0) };
        true
    }

    /// # Safety
    /// `n` and its arguments must form a valid x86-64 Linux syscall.
    unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> isize {
        let ret: isize;

        // SAFETY: x86-64 Linux syscall ABI; syscall clobbers rcx and r11.
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

    pub(super) fn process_affinity() -> Option<Vec<Cpu>> { None }
    pub(super) fn bind_thread(_cpus: &[Cpu]) -> bool { false }
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
        let cpus = parse_cpu_list("0-95,192-287"); // EPYC 9654 node 0
        assert_eq!(cpus.len(), 192);
        assert_eq!(cpus.first(), Some(&0));
        assert_eq!(cpus.last(), Some(&287));
        assert!(!cpus.contains(&96));
    }

    #[test]
    fn distribute_balances_by_fill() {
        let topo = NumaTopology { nodes: vec![(0..32).collect()], l3_domains: vec![(0..16).collect(), (16..32).collect()] };
        let assignment = topo.assign_l3_domains(4);
        assert_eq!(assignment.iter().filter(|&&d| d == 0).count(), 2);
        assert_eq!(assignment.iter().filter(|&&d| d == 1).count(), 2);
    }

    #[test]
    fn single_domain_never_binds() {
        let topo = NumaTopology { nodes: vec![(0..8).collect()], l3_domains: vec![(0..8).collect()] };
        assert!(!topo.should_bind_l3(8));
        assert_eq!(topo.assign_l3_domains(4), vec![0, 0, 0, 0]);
    }

    #[test]
    fn detect_is_sane_on_this_box() {
        let topo = NumaTopology::detect();
        assert!(topo.num_nodes() >= 1);
        assert!(topo.num_l3_domains() >= 1);
    }
}
