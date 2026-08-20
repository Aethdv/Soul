//! FNV-1a rather than std's, because these digests go on disk and `DefaultHasher` makes no promise across releases.

pub struct Fnv1a(u64);

impl Default for Fnv1a {
    #[inline]
    fn default() -> Self { Self::new() }
}

impl Fnv1a {
    const OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;

    #[inline]
    pub const fn new() -> Self { Self(Self::OFFSET) }
    #[inline]
    pub const fn digest(self) -> u64 { self.0 }

    #[inline]
    pub fn write_bytes(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0 ^ u64::from(b)).wrapping_mul(Self::PRIME);
        }
    }
}
