//! Huge-page-backed allocation for the transposition table.
//!
//! A transposition table is the pathological case for the TLB: megabytes to
//! hundreds of gigabytes, touched at random. On 4KB pages a 256GB table needs
//! 67 million page-table entries and the CPU caches a couple thousand, so almost
//! every probe pays a multi-level page walk before it reaches the entry. Huge
//! pages collapse that walk. A 2MB page covers 512× the ground of a 4KB one, a
//! 1GB page another 512×, and at 1GB the page table for a 256GB region is 2KB
//! that simply stays in cache.
//!
//! So the allocator asks for the largest page it can, falls back when the pool
//! isn't there, and reports which size it got. Every tier returns zeroed memory
//! and faults it in up front, so the first search runs hot; `clear` rewrites the
//! pages in place rather than discarding them, keeping them resident.
//!
//! Huge pages are x86-64 Linux only here; other targets get a plain zeroed allocation.
//! That path issues `mmap`/`madvise`/`munmap` as raw syscalls, so the build needs
//! no `libc` at all.

use std::{fmt, marker::PhantomData, mem, ops::Deref, ptr::NonNull, slice};

/// The page size the OS actually backed an allocation with.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PageKind {
    /// 1GB explicit huge pages (`hugetlb`); needs a reserved pool.
    Huge1G,
    /// 2MB explicit huge pages (`hugetlb`); needs a reserved pool.
    Huge2M,
    /// Transparent huge pages: a 2MB hint the kernel may or may not honor.
    Thp,
    /// Ordinary 4KB pages.
    Base,
}

impl fmt::Display for PageKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            PageKind::Huge1G => "1G",
            PageKind::Huge2M => "2M",
            PageKind::Thp => "THP",
            PageKind::Base => "4K",
        })
    }
}

/// An owned region of `len` `T`s, backed by the largest huge page the OS will
/// give. Derefs to `[T]`, so it stands in for a `Box<[T]>` at the call site.
pub struct HugePages<T> {
    ptr: NonNull<T>,
    /// Addressable `T`s. The mapping rounds up to a page boundary, so this can
    /// exceed the requested count.
    len: usize,
    /// Mapped byte length, kept for the matching free.
    bytes: usize,
    kind: PageKind,
    _marker: PhantomData<T>,
}

// SAFETY: HugePages owns a unique mapping for its whole lifetime; whether the
// elements may be shared across threads is then governed by T's own Send/Sync,
// exactly as for Box<[T]>. The TT's Cluster is atomic, hence Sync.
unsafe impl<T: Send> Send for HugePages<T> {}
unsafe impl<T: Sync> Sync for HugePages<T> {}

impl<T> HugePages<T> {
    /// Allocate at least `min_bytes`, viewed as `[T]`, mapped but not pre-faulted.
    ///
    /// The pages come lazily, so the first write to each one faults it in.
    /// The TT uses this to drive first-touch from its own NUMA-bound threads,
    /// which places each page on the node that touched it. `zeroed` is this
    /// same map with the pre-fault folded in, for the single-domain path.
    ///
    /// # Safety
    /// As [`zeroed`]: `T` must be valid when zero-initialized.
    /// The OS zeroes each page at fault time and no constructor runs.
    pub unsafe fn mapped(min_bytes: usize) -> Self {
        let (ptr, bytes, kind) = map(min_bytes.max(mem::size_of::<T>()));

        Self { ptr: ptr.cast(), len: bytes / mem::size_of::<T>(), bytes, kind, _marker: PhantomData }
    }

    /// Allocate at least `min_bytes` of zeroed memory, viewed as `[T]`, every page
    /// pre-faulted so the search runs hot from its first node.
    ///
    /// # Safety
    /// `T` must be valid when zero-initialized: the mapping comes back zeroed and
    /// is reinterpreted as `T` without running any constructor.
    pub unsafe fn zeroed(min_bytes: usize) -> Self {
        // SAFETY: caller's contract on T.
        let pages = unsafe { Self::mapped(min_bytes) };

        // Pre-fault the whole table now so the search runs hot from its first node.
        // mmap hands the pages out lazily, and faulting them mid-search would stall
        // on the clock with no ucinewgame to absorb it. That is the regression we
        // refuse.
        //
        // SAFETY: a fresh exclusive mapping of bytes, nothing aliases it.
        unsafe { zero_region(pages.ptr.cast::<u8>().as_ptr(), pages.bytes) };

        pages
    }

    /// The page size the OS actually gave us.
    pub fn kind(&self) -> PageKind {
        self.kind
    }

    /// Reset to an empty table, leaving the pages resident.
    ///
    /// A zeroing memset, not a page discard. Discarding empties the table for one
    /// syscall, but then the next search re-faults every page it touches, on the
    /// clock; the cost of emptying belongs here, off it, with the pages kept hot.
    ///
    /// The caller guarantees no searcher is concurrently probing (this runs on
    /// `ucinewgame`, when the engine is idle).
    pub fn clear(&self) {
        // SAFETY: idle precondition: no searcher reads the region during the clear.
        unsafe { zero_region(self.ptr.cast::<u8>().as_ptr(), self.bytes) };
    }
}

/// Zero `bytes` from `base`, shared by the allocation pre-fault and the
/// `ucinewgame` clear. The write is memory-bandwidth-bound, so one pass is the floor.
///
/// # Safety
/// `base` must be writable for `bytes`, and nothing may read the region concurrently.
unsafe fn zero_region(base: *mut u8, bytes: usize) {
    // SAFETY: caller's contract.
    unsafe { base.write_bytes(0, bytes) };
}

impl<T> Deref for HugePages<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: ptr maps len elements, zeroed then valid as T (zeroed() contract).
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for HugePages<T> {
    fn drop(&mut self) {
        // SAFETY: ptr/bytes name the mapping map handed back.
        unsafe { unmap(self.ptr.cast(), self.bytes) }
    }
}

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
use linux::{map, unmap};
#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
use portable::{map, unmap};

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod linux {
    use std::ptr::NonNull;

    use super::PageKind;

    const SYS_MMAP: usize = 9;
    const SYS_MUNMAP: usize = 11;
    const SYS_MADVISE: usize = 28;

    const PROT_RW: usize = 0x1 | 0x2; // PROT_READ | PROT_WRITE
    const MAP_PRIVATE_ANON: usize = 0x02 | 0x20; // MAP_PRIVATE | MAP_ANONYMOUS
    const MAP_HUGETLB: usize = 0x40000;
    // The huge page's log2 size sits at bit 26; 2^21 = 2MB, 2^30 = 1GB.
    const MAP_HUGE_2MB: usize = 21 << 26;
    const MAP_HUGE_1GB: usize = 30 << 26;
    const MADV_HUGEPAGE: usize = 14;

    const GB: usize = 1 << 30;
    const MB2: usize = 2 << 20;

    /// The tier ladder: 1GB hugetlb, 2MB hugetlb, then a plain mapping hinted
    /// toward transparent huge pages. Each tier rounds the length up to its page
    /// size: a `hugetlb` mapping is rejected outright unless the length is a
    /// whole multiple, the bug that silently drops a careless caller to 4KB.
    pub fn map(min_bytes: usize) -> (NonNull<u8>, usize, PageKind) {
        // 1GB only past a gigabyte, or rounding up to the boundary is mostly waste.
        if min_bytes >= GB {
            let len = round_up(min_bytes, GB);

            if let Some(p) = mmap(len, MAP_HUGETLB | MAP_HUGE_1GB) {
                return (p, len, PageKind::Huge1G);
            }
        }

        let len_2m = round_up(min_bytes, MB2);

        if let Some(p) = mmap(len_2m, MAP_HUGETLB | MAP_HUGE_2MB) {
            return (p, len_2m, PageKind::Huge2M);
        }

        // Round to 2MB, and align the base to 2MB. THP only fills a 2MB-aligned
        // span; a mapping that starts mid-page loses both ends back to 4KB, and
        // under madvise policy the kernel can't pre-align, since the hint lands
        // after it has already picked the address.
        let len = round_up(min_bytes, MB2);
        let p = mmap_aligned(len, MB2);
        // SAFETY: p/len name the mapping just returned. madvise reports whether the
        // hint was accepted at all (it is rejected only when THP is compiled out).
        let kind = if unsafe { madvise(p, len, MADV_HUGEPAGE) } { PageKind::Thp } else { PageKind::Base };

        (p, len, kind)
    }

    /// # Safety
    /// `ptr`/`bytes` must name a mapping returned by [`map`].
    pub unsafe fn unmap(ptr: NonNull<u8>, bytes: usize) {
        // SAFETY: caller contract.
        unsafe { syscall6(SYS_MUNMAP, ptr.as_ptr() as usize, bytes, 0, 0, 0, 0) };
    }

    fn mmap(len: usize, extra_flags: usize) -> Option<NonNull<u8>> {
        // SAFETY: anonymous mapping (fd = -1, offset = 0); all arguments are kernel-checked.
        let ret = unsafe { syscall6(SYS_MMAP, 0, len, PROT_RW, MAP_PRIVATE_ANON | extra_flags, usize::MAX, 0) };

        // Failure returns -errno in -4095..=-1; success returns a user address.
        if (-4095..0).contains(&ret) { None } else { NonNull::new(ret as *mut u8) }
    }

    /// `mmap` `len` bytes at a base aligned to `align`, a power-of-two multiple of
    /// the page size. Over-maps by one `align` and returns the slack at each end,
    /// so the kept span is exactly `len` and a later `munmap(base, len)` frees it
    /// whole. Bare `mmap` only aligns to 4KB; THP needs the 2MB boundary.
    fn mmap_aligned(len: usize, align: usize) -> NonNull<u8> {
        let over = mmap(len + align, 0).expect("mmap failed for the transposition table");
        let base = over.as_ptr() as usize;
        let head = round_up(base, align) - base;

        // SAFETY: over maps len + align bytes. head and head + len stay inside it,
        // both are page multiples, so each partial munmap splits the VMA on a page
        // boundary and frees only the slack; the [aligned, aligned + len) span stays.
        unsafe {
            let aligned = over.as_ptr().add(head);

            if head > 0 {
                unmap(over, head);
            }

            let tail = align - head;

            if tail > 0 {
                unmap(NonNull::new_unchecked(aligned.add(len)), tail);
            }
            NonNull::new_unchecked(aligned)
        }
    }

    /// # Safety
    /// `ptr`/`len` must name a live mapping.
    unsafe fn madvise(ptr: NonNull<u8>, len: usize, advice: usize) -> bool {
        // SAFETY: caller contract.
        unsafe { syscall6(SYS_MADVISE, ptr.as_ptr() as usize, len, advice, 0, 0, 0) == 0 }
    }

    const fn round_up(x: usize, align: usize) -> usize {
        (x + align - 1) & !(align - 1)
    }

    /// # Safety
    /// A valid x86_64 Linux syscall: `n` and its arguments must form a sound call,
    /// and any pointer arguments must satisfy that syscall's requirements.
    #[inline]
    unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize) -> isize {
        let ret: isize;
        // SAFETY: x86_64 Linux ABI: number in rax, args in rdi/rsi/rdx/r10/r8/r9,
        // result in rax; the instruction clobbers rcx and r11.
        unsafe {
            std::arch::asm!(
                "syscall",
                inlateout("rax") n as isize => ret,
                in("rdi") a1,
                in("rsi") a2,
                in("rdx") a3,
                in("r10") a4,
                in("r8") a5,
                in("r9") a6,
                out("rcx") _,
                out("r11") _,
                options(nostack),
            );
        }
        ret
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod portable {
    use std::{
        alloc::{self, Layout},
        ptr::NonNull,
    };

    use super::PageKind;

    const PAGE: usize = 4096;

    pub fn map(min_bytes: usize) -> (NonNull<u8>, usize, PageKind) {
        let bytes = round_up(min_bytes, PAGE);
        let layout = Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: layout is non-zero (min_bytes >= size_of::<T>() >= 1).
        let ptr = unsafe { alloc::alloc_zeroed(layout) };

        (NonNull::new(ptr).expect("allocation failed for the transposition table"), bytes, PageKind::Base)
    }

    /// # Safety
    /// `ptr`/`bytes` must name an allocation returned by [`map`].
    pub unsafe fn unmap(ptr: NonNull<u8>, bytes: usize) {
        let layout = Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: layout matches the one map allocated with.
        unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
    }

    const fn round_up(x: usize, align: usize) -> usize {
        (x + align - 1) & !(align - 1)
    }
}
