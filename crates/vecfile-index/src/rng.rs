//! A tiny deterministic RNG (SplitMix64), so index builds are reproducible
//! from a seed without depending on `rand`.

#[derive(Clone, Debug)]
pub struct SplitMix64(u64);

impl SplitMix64 {
    pub fn new(seed: u64) -> Self {
        SplitMix64(seed)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in (0, 1]; never zero, so `ln` is always finite.
    pub fn next_f64_open(&mut self) -> f64 {
        ((self.next_u64() >> 11) + 1) as f64 / (1u64 << 53) as f64
    }
}
