/// FNV-1a hash state. Feed it bytes, ask for the digest.
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
            self.0 ^= u64::from(b);
            self.0 = self.0.wrapping_mul(Self::PRIME);
        }
    }

    // Convenience for the common "hash a slice of little-endian u32s" pattern
    #[inline]
    pub fn write_u32s(&mut self, vals: &[u32]) {
        for v in vals {
            self.write_bytes(&v.to_le_bytes());
        }
    }
}
