//! Huge-page-backed allocation for the transposition table.
//!
//! x86-64 Linux tries explicit huge pages and THP; other targets use a page-aligned allocation.

use std::{fmt, marker::PhantomData, mem, ops::Deref, ptr::NonNull, slice};

/// Allocation backing; `Thp` records a hint, not a confirmed page size.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum PageKind {
    /// 1GB explicit huge pages (`hugetlb`); needs a reserved pool.
    Huge1G,
    /// 2MB explicit huge pages (`hugetlb`); needs a reserved pool.
    Huge2M,
    /// Transparent huge pages: a 2MB hint the kernel may or may not honor.
    Thp,
    /// Ordinary 4KB pages.
    Base,
}

pub(crate) struct HugePages<T> {
    ptr: NonNull<T>,
    len: usize,
    bytes: usize,
    kind: PageKind,
    _marker: PhantomData<T>,
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

// SAFETY: HugePages uniquely owns its region; sharing elements is therefore
// governed by T's Send/Sync, as with Box<[T]>.
unsafe impl<T: Send> Send for HugePages<T> {}
unsafe impl<T: Sync> Sync for HugePages<T> {}

impl<T> HugePages<T> {
    pub(crate) fn kind(&self) -> PageKind { self.kind }

    /// Allocate at least `min_bytes` without pre-faulting.
    ///
    /// # Safety
    /// `T` must be valid when zero-initialized and require no alignment above a page.
    pub(crate) unsafe fn mapped(min_bytes: usize) -> Self {
        assert!(mem::size_of::<T>() != 0);
        let (ptr, bytes, kind) = map(min_bytes.max(mem::size_of::<T>()));
        Self { ptr: ptr.cast(), len: bytes / mem::size_of::<T>(), bytes, kind, _marker: PhantomData }
    }

    /// Allocate at least `min_bytes` and pre-fault every page.
    ///
    /// # Safety
    /// Same requirements as [`Self::mapped`].
    pub(crate) unsafe fn prefaulted(min_bytes: usize) -> Self {
        // SAFETY: caller satisfies mapped's contract.
        let pages = unsafe { Self::mapped(min_bytes) };
        // SAFETY: pages uniquely owns the entire region.
        unsafe { pages.clear() };
        pages
    }

    /// Zero the region without discarding its resident pages.
    ///
    /// # Safety
    /// No references into the region may be used while it is cleared.
    pub(crate) unsafe fn clear(&self) {
        // SAFETY: caller excludes all aliases during the write.
        unsafe { self.ptr.cast::<u8>().as_ptr().write_bytes(0, self.bytes) };
    }
}

impl<T> Deref for HugePages<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &[T] {
        // SAFETY: construction guarantees ptr covers len valid T values.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T> Drop for HugePages<T> {
    fn drop(&mut self) {
        // SAFETY: ptr/bytes are exactly the region returned by map.
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
    // The huge page's log2 size sits at bit 26.
    const MAP_HUGE_2MB: usize = 21 << 26;
    const MAP_HUGE_1GB: usize = 30 << 26;
    const MADV_HUGEPAGE: usize = 14;

    const GB: usize = 1 << 30;
    const MB2: usize = 2 << 20;

    pub(super) fn map(min_bytes: usize) -> (NonNull<u8>, usize, PageKind) {
        // Avoid rounding a sub-gigabyte table up to 1GB.
        if min_bytes >= GB {
            let len = min_bytes.next_multiple_of(GB);
            if let Some(p) = mmap(len, MAP_HUGETLB | MAP_HUGE_1GB) {
                return (p, len, PageKind::Huge1G);
            }
        }

        let len = min_bytes.next_multiple_of(MB2);
        if let Some(p) = mmap(len, MAP_HUGETLB | MAP_HUGE_2MB) {
            return (p, len, PageKind::Huge2M);
        }

        // THP needs a 2MB-aligned span; mmap itself guarantees only page alignment.
        let p = mmap_aligned(len, MB2);
        // SAFETY: p/len name the live mapping above.
        let kind = if unsafe { madvise(p, len, MADV_HUGEPAGE) } { PageKind::Thp } else { PageKind::Base };
        (p, len, kind)
    }

    /// # Safety
    /// `ptr`/`bytes` must name a mapping returned by [`map`].
    pub(super) unsafe fn unmap(ptr: NonNull<u8>, bytes: usize) {
        // SAFETY: caller guarantees ptr/bytes name the live mapping.
        unsafe { syscall6(SYS_MUNMAP, ptr.as_ptr() as usize, bytes, 0, 0, 0, 0) };
    }

    fn mmap(len: usize, extra_flags: usize) -> Option<NonNull<u8>> {
        // SAFETY: valid anonymous mmap arguments; the kernel validates the request.
        let ret = unsafe { syscall6(SYS_MMAP, 0, len, PROT_RW, MAP_PRIVATE_ANON | extra_flags, usize::MAX, 0) };
        // Raw syscalls return -errno in -4095..=-1.
        if (-4095..0).contains(&ret) { None } else { NonNull::new(ret as *mut u8) }
    }

    /// Map `len` bytes at an `align` boundary by overmapping and freeing the slack.
    fn mmap_aligned(len: usize, align: usize) -> NonNull<u8> {
        let over = mmap(len + align, 0).expect("mmap failed for the transposition table");
        let base = over.as_ptr() as usize;
        let head = base.next_multiple_of(align) - base;

        // SAFETY: the retained len-byte span and both slack regions are disjoint,
        // page-aligned parts of the original len + align mapping.
        unsafe {
            let aligned = over.as_ptr().add(head);
            if head > 0 {
                unmap(over, head);
            }
            unmap(NonNull::new_unchecked(aligned.add(len)), align - head);
            NonNull::new_unchecked(aligned)
        }
    }

    /// # Safety
    /// `ptr`/`len` must name a live mapping.
    unsafe fn madvise(ptr: NonNull<u8>, len: usize, advice: usize) -> bool {
        // SAFETY: caller guarantees ptr/len name a live mapping.
        unsafe { syscall6(SYS_MADVISE, ptr.as_ptr() as usize, len, advice, 0, 0, 0) == 0 }
    }

    /// # Safety
    /// `n` and its arguments must form a valid x86-64 Linux syscall.
    #[inline]
    unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize) -> isize {
        let ret: isize;
        // SAFETY: x86-64 Linux syscall ABI; syscall clobbers rcx and r11.
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

    pub(super) fn map(min_bytes: usize) -> (NonNull<u8>, usize, PageKind) {
        let bytes = min_bytes.next_multiple_of(PAGE);
        let layout = Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: layout is valid and non-zero.
        let ptr = unsafe { alloc::alloc_zeroed(layout) };
        (NonNull::new(ptr).expect("allocation failed for the transposition table"), bytes, PageKind::Base)
    }

    /// # Safety
    /// `ptr`/`bytes` must name an allocation returned by [`map`].
    pub(super) unsafe fn unmap(ptr: NonNull<u8>, bytes: usize) {
        let layout = Layout::from_size_align(bytes, PAGE).unwrap();
        // SAFETY: layout matches the allocation.
        unsafe { alloc::dealloc(ptr.as_ptr(), layout) };
    }
}
