//! Per-lane comparison results.

use core::arch::x86_64::*;

#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Mask(pub(super) __m256i);

impl Mask {
    #[inline(always)]
    pub(crate) fn any(self) -> bool {
        // SAFETY: AVX2 available per the mod.rs compile_error gate.
        unsafe { _mm256_testz_si256(self.0, self.0) == 0 }
    }
}
