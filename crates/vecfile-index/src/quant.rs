//! int8 scalar quantization.
//!
//! Each component is stored as `x_i ≈ o_i + s · (c_i + 128)`, with a
//! per-dimension offset `o_i` and one scale `s` shared by all dimensions.
//! Sharing the scale is what lets the int8 kernels give real distances:
//!
//! - L2:  `|x - y|² ≈ s² · Σ (c_i - d_i)²` (the offsets cancel).
//! - Dot: `x · y ≈ s² · Σ c_i d_i + corr(c) + corr(d) + K`, where `corr` is
//!   one f32 stored per vector and `K` is a constant of the quantizer.
//!
//! So int8 distances come out in the same units as f32 ones, and candidates
//! from segments with different quantizers can be merged directly.
//!
//! Offsets and the scale come from the 0.1% and 99.9% quantiles of each
//! dimension, so a few outliers don't spread the codes thin; values outside
//! the range are clamped.

use crate::distance::{Kernels, Metric};

/// Rows sampled when training.
const TRAIN_SAMPLE: usize = 20_000;
const LOW_Q: f64 = 0.001;
const HIGH_Q: f64 = 0.999;

#[derive(Clone, Debug, PartialEq)]
pub struct ScalarQuantizer {
    metric: Metric,
    offset: Vec<f32>,
    scale: f32,
    /// `Σ o_i² + s² · 128² · dim`, the constant term of the dot expansion.
    k: f32,
}

impl ScalarQuantizer {
    /// Fits a quantizer to row-major `data`.
    pub fn train(data: &[f32], dim: usize, metric: Metric) -> Self {
        assert!(
            dim > 0 && data.len().is_multiple_of(dim),
            "data must be n x dim"
        );
        let n = data.len() / dim;
        let step = (n / TRAIN_SAMPLE).max(1);
        let rows: Vec<usize> = (0..n).step_by(step).collect();

        let mut lo = vec![0f32; dim];
        let mut hi = vec![0f32; dim];
        let mut col = Vec::with_capacity(rows.len());
        for d in 0..dim {
            col.clear();
            col.extend(rows.iter().map(|&r| data[r * dim + d]));
            if col.is_empty() {
                continue;
            }
            lo[d] = quantile(&mut col, LOW_Q);
            hi[d] = quantile(&mut col, HIGH_Q);
        }
        let range = lo.iter().zip(&hi).map(|(l, h)| h - l).fold(0f32, f32::max);
        let scale = if range > 0.0 { range / 255.0 } else { 1.0 };
        let offset: Vec<f32> = lo
            .iter()
            .zip(&hi)
            .map(|(l, h)| (l + h) / 2.0 - 127.5 * scale)
            .collect();
        Self::from_parts(metric, offset, scale)
    }

    pub fn from_parts(metric: Metric, offset: Vec<f32>, scale: f32) -> Self {
        let k = offset.iter().map(|o| o * o).sum::<f32>()
            + scale * scale * 16384.0 * offset.len() as f32;
        ScalarQuantizer {
            metric,
            offset,
            scale,
            k,
        }
    }

    pub fn dim(&self) -> usize {
        self.offset.len()
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn offset(&self) -> &[f32] {
        &self.offset
    }

    pub fn scale(&self) -> f32 {
        self.scale
    }

    /// Encodes `v` into `out` and returns its correction term (zero for L2).
    pub fn encode(&self, v: &[f32], out: &mut [i8]) -> f32 {
        assert_eq!(v.len(), self.dim());
        assert_eq!(out.len(), self.dim());
        let inv = 1.0 / self.scale;
        for ((x, o), c) in v.iter().zip(&self.offset).zip(out.iter_mut()) {
            let u = ((x - o) * inv).round().clamp(0.0, 255.0);
            *c = (u as i32 - 128) as i8;
        }
        self.correction(out)
    }

    /// The per-vector term of the dot expansion for an encoded vector.
    pub fn correction(&self, code: &[i8]) -> f32 {
        if self.metric == Metric::L2 {
            return 0.0;
        }
        let s = self.scale;
        let mut weighted = 0f32;
        let mut sum = 0i32;
        for (&c, o) in code.iter().zip(&self.offset) {
            weighted += o * (i32::from(c) + 128) as f32;
            sum += i32::from(c);
        }
        s * weighted + s * s * 128.0 * sum as f32
    }

    pub fn decode(&self, code: &[i8], out: &mut [f32]) {
        for ((c, o), x) in code.iter().zip(&self.offset).zip(out.iter_mut()) {
            *x = o + self.scale * (i32::from(*c) + 128) as f32;
        }
    }

    /// Distance between two encoded vectors, in the metric's real units.
    #[inline]
    pub fn distance(&self, k: &Kernels, a: &[i8], a_corr: f32, b: &[i8], b_corr: f32) -> f32 {
        let s2 = self.scale * self.scale;
        match self.metric {
            Metric::L2 => s2 * (k.l2_sq_i8)(a, b) as f32,
            Metric::Dot => -(s2 * (k.dot_i8)(a, b) as f32 + a_corr + b_corr + self.k),
            Metric::Cosine => 1.0 - (s2 * (k.dot_i8)(a, b) as f32 + a_corr + b_corr + self.k),
        }
    }
}

fn quantile(values: &mut [f32], q: f64) -> f32 {
    let i = ((values.len() - 1) as f64 * q).round() as usize;
    *values.select_nth_unstable_by(i, f32::total_cmp).1
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::kernels;
    use crate::rng::SplitMix64;

    fn data(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dim)
            .map(|i| (r.next_f64_open() as f32 - 0.5) * (1.0 + (i % dim) as f32 * 0.1))
            .collect()
    }

    #[test]
    fn int8_distance_tracks_f32_distance() {
        let dim = 64;
        let v = data(2000, dim, 1);
        for metric in [Metric::L2, Metric::Dot] {
            let q = ScalarQuantizer::train(&v, dim, metric);
            let (mut ca, mut cb) = (vec![0i8; dim], vec![0i8; dim]);
            let mut max_rel = 0f32;
            for i in 0..200 {
                let (a, b) = (&v[i * dim..(i + 1) * dim], &v[(i + 1) * dim..(i + 2) * dim]);
                let (ka, kb) = (q.encode(a, &mut ca), q.encode(b, &mut cb));
                let approx = q.distance(kernels(), &ca, ka, &cb, kb);
                let exact = metric.distance(kernels(), a, b);
                let mag = match metric {
                    Metric::L2 => exact.abs(),
                    _ => a.iter().zip(b).map(|(x, y)| (x * y).abs()).sum(),
                };
                max_rel = max_rel.max((approx - exact).abs() / mag.max(1e-6));
            }
            assert!(max_rel < 0.05, "{metric:?}: max relative error {max_rel}");
        }
    }

    #[test]
    fn decode_inverts_encode_within_half_a_step() {
        let dim = 16;
        let v = data(500, dim, 2);
        let q = ScalarQuantizer::train(&v, dim, Metric::L2);
        let mut code = vec![0i8; dim];
        let mut back = vec![0f32; dim];
        let row = &v[..dim];
        q.encode(row, &mut code);
        q.decode(&code, &mut back);
        for (x, y) in row.iter().zip(&back) {
            assert!((x - y).abs() <= q.scale() * 0.5 + 1e-6, "{x} vs {y}");
        }
    }

    #[test]
    fn constant_data_does_not_divide_by_zero() {
        let q = ScalarQuantizer::train(&[3.0; 40], 4, Metric::L2);
        let mut code = [0i8; 4];
        q.encode(&[3.0; 4], &mut code);
        assert!(q.scale().is_finite());
    }
}
