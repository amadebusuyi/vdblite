//! 1-bit codes (RaBitQ), for walking a graph with an eighth of int8's
//! memory traffic.
//!
//! A vector `x` is centred on the segment's centroid `c` and rotated by a
//! random orthogonal matrix `P`: `r = P(x - c)`. Its code is the sign of
//! each component of `r`, one bit per dimension. Read as a unit vector
//! `ō = sign(r)/√D`, the code is a good direction estimate, and RaBitQ's
//! estimator corrects for the angle between `ō` and `r` itself:
//!
//! `(x - c)·(q - c) ≈ |r| · ⟨ō, r_q⟩ / ⟨ō, r/|r|⟩`, with `r_q = P(q - c)`.
//!
//! The rotation spreads every vector's energy evenly over the dimensions,
//! which is what keeps the error small (about `1/√D` of the norms) and
//! unbiased.
//!
//! The query side is quantized to 4 bits per dimension, uniformly between
//! its smallest and largest rotated component, and stored as four bit
//! planes, so `⟨ō, r_q⟩` comes from popcounts:
//!
//! `√D · ⟨ō, r_q⟩ = 2Δ·S + 2·lo·B - Δ·Σu - D·lo`
//!
//! where `u_i` are the 4-bit query values, `S = Σ_j 2^j · popcount(code &
//! plane_j)`, `B = popcount(code)`, and `lo`, `Δ` the query's grid.
//!
//! Each vector stores three f32 factors beside its bits: `a = |r|² / Σ|r_i|`
//! (which folds `|r|`, `⟨ō, r/|r|⟩` and the `√D`s into one), `B`, and a
//! metric term: `(x - c)·c` for dot and cosine, `|x - c|²` for L2.

use crate::distance::{Kernels, Metric};
use crate::rng::SplitMix64;

/// f32 factors stored per vector.
pub const FACTORS: usize = 3;
/// Bits per query component.
const QUERY_BITS: usize = 4;
/// Seed of every segment's rotation. Fixed, so the rotation of a given
/// dimension is always the same matrix; it is stored with each segment
/// anyway, so this could change without breaking files.
const ROTATION_SEED: u64 = 0x5241_4249_5451_0001;

#[derive(Clone, Debug, PartialEq)]
pub struct BinaryQuantizer {
    metric: Metric,
    dim: usize,
    /// The centroid (`dim`), then the rotation: a row-major `dim × dim`
    /// orthogonal matrix. One array, as segments store it.
    state: Vec<f32>,
}

/// A query prepared for [`BinaryQuantizer::distance`].
#[derive(Clone, Debug, Default)]
pub struct BinaryQuery {
    /// `QUERY_BITS` planes of `words` u64 each, lowest bit first.
    planes: Vec<u64>,
    lo: f32,
    delta: f32,
    sum_u: f32,
    /// `c·q` (dot, cosine) or `|q - c|²` (L2).
    term: f32,
    rotated: Vec<f32>,
    centred: Vec<f32>,
}

impl BinaryQuantizer {
    /// Fits a quantizer to row-major, prepared `data`: its centroid, and
    /// the rotation for its dimension.
    pub fn train(data: &[f32], dim: usize, metric: Metric) -> Self {
        assert!(
            dim > 0 && data.len().is_multiple_of(dim),
            "data must be n x dim"
        );
        let n = data.len() / dim;
        let mut centroid = vec![0f64; dim];
        for row in data.chunks_exact(dim) {
            for (c, &x) in centroid.iter_mut().zip(row) {
                *c += f64::from(x);
            }
        }
        let mut state: Vec<f32> = centroid
            .iter()
            .map(|&c| if n > 0 { (c / n as f64) as f32 } else { 0.0 })
            .collect();
        state.extend(random_rotation(dim));
        BinaryQuantizer { metric, dim, state }
    }

    /// The quantizer [`Self::state`] describes, if it has the right length
    /// for `dim`.
    pub fn from_state(metric: Metric, dim: usize, state: Vec<f32>) -> Option<Self> {
        (dim > 0 && state.len() == dim + dim * dim).then_some(BinaryQuantizer {
            metric,
            dim,
            state,
        })
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The centroid, then the rotation, as one array.
    pub fn state(&self) -> &[f32] {
        &self.state
    }

    fn centroid(&self) -> &[f32] {
        &self.state[..self.dim]
    }

    fn rotation(&self) -> &[f32] {
        &self.state[self.dim..]
    }

    /// u64 words per code.
    pub fn words(&self) -> usize {
        self.dim.div_ceil(64)
    }

    fn rotate(&self, k: &Kernels, v: &[f32], out: &mut [f32]) {
        for (o, row) in out.iter_mut().zip(self.rotation().chunks_exact(self.dim)) {
            *o = (k.dot_f32)(row, v);
        }
    }

    /// Encodes prepared `v` into `code` (`words()` long) and `factors`.
    pub fn encode(&self, k: &Kernels, v: &[f32], code: &mut [u64], factors: &mut [f32]) {
        assert_eq!(v.len(), self.dim);
        let centred: Vec<f32> = v.iter().zip(self.centroid()).map(|(x, c)| x - c).collect();
        let mut r = vec![0f32; self.dim];
        self.rotate(k, &centred, &mut r);
        code.fill(0);
        let (mut norm2, mut abs) = (0f64, 0f64);
        for (i, &x) in r.iter().enumerate() {
            if x > 0.0 {
                code[i / 64] |= 1 << (i % 64);
            }
            norm2 += f64::from(x) * f64::from(x);
            abs += f64::from(x.abs());
        }
        let ones: u32 = code.iter().map(|w| w.count_ones()).sum();
        factors[0] = if abs > 0.0 { (norm2 / abs) as f32 } else { 0.0 };
        factors[1] = ones as f32;
        factors[2] = match self.metric {
            Metric::L2 => norm2 as f32,
            Metric::Dot | Metric::Cosine => (k.dot_f32)(&centred, self.centroid()),
        };
    }

    /// Prepares prepared query `q` for [`Self::distance`].
    pub fn prepare_query(&self, k: &Kernels, q: &[f32], out: &mut BinaryQuery) {
        assert_eq!(q.len(), self.dim);
        let dim = self.dim;
        out.centred.clear();
        out.centred
            .extend(q.iter().zip(self.centroid()).map(|(x, c)| x - c));
        out.rotated.resize(dim, 0.0);
        let mut rotated = std::mem::take(&mut out.rotated);
        self.rotate(k, &out.centred, &mut rotated);
        let (lo, hi) = rotated
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(l, h), &x| {
                (l.min(x), h.max(x))
            });
        let levels = ((1 << QUERY_BITS) - 1) as f32;
        let delta = if hi > lo { (hi - lo) / levels } else { 1.0 };
        let words = self.words();
        out.planes.clear();
        out.planes.resize(QUERY_BITS * words, 0);
        let mut sum_u = 0u32;
        for (i, &x) in rotated.iter().enumerate() {
            let u = ((x - lo) / delta).round().clamp(0.0, levels) as u32;
            sum_u += u;
            for j in 0..QUERY_BITS {
                if u >> j & 1 == 1 {
                    out.planes[j * words + i / 64] |= 1 << (i % 64);
                }
            }
        }
        out.lo = lo;
        out.delta = delta;
        out.sum_u = sum_u as f32;
        out.term = match self.metric {
            Metric::L2 => (k.dot_f32)(&out.centred, &out.centred),
            Metric::Dot | Metric::Cosine => (k.dot_f32)(self.centroid(), q),
        };
        out.rotated = rotated;
    }

    /// Estimated distance from a prepared query to a code, in the same
    /// units as f32 distances.
    #[inline]
    pub fn distance(&self, k: &Kernels, q: &BinaryQuery, code: &[u64], factors: &[f32]) -> f32 {
        let s = (k.binary_ip4)(code, &q.planes) as f32;
        let (a, ones, term) = (factors[0], factors[1], factors[2]);
        // ≈ (x - c)·(q - c)
        let ip = a
            * (2.0 * q.delta * s + 2.0 * q.lo * ones - q.delta * q.sum_u - self.dim as f32 * q.lo);
        match self.metric {
            Metric::L2 => term + q.term - 2.0 * ip,
            // x·q = (x - c)·(q - c) + (x - c)·c + c·q
            Metric::Dot => -(ip + term + q.term),
            Metric::Cosine => 1.0 - (ip + term + q.term),
        }
    }
}

/// A random `dim × dim` orthogonal matrix (row-major): Gram–Schmidt on
/// Gaussian rows, from a fixed seed.
pub fn random_rotation(dim: usize) -> Vec<f32> {
    let mut rng = SplitMix64::new(ROTATION_SEED ^ dim as u64);
    let mut gauss = || {
        // Box–Muller.
        let (u, v) = (rng.next_f64_open(), rng.next_f64_open());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    };
    let mut m: Vec<f64> = (0..dim * dim).map(|_| gauss()).collect();
    for i in 0..dim {
        // Twice, for orthogonality in floating point.
        for _ in 0..2 {
            for j in 0..i {
                let (done, rest) = m.split_at_mut(i * dim);
                let prev = &done[j * dim..(j + 1) * dim];
                let row = &mut rest[..dim];
                let d: f64 = prev.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                for (r, p) in row.iter_mut().zip(prev) {
                    *r -= d * p;
                }
            }
        }
        let row = &mut m[i * dim..(i + 1) * dim];
        let norm = row.iter().map(|x| x * x).sum::<f64>().sqrt();
        for x in row.iter_mut() {
            *x /= norm;
        }
    }
    m.into_iter().map(|x| x as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distance::kernels;
    use crate::segment::prepare;

    fn data(n: usize, dim: usize, seed: u64, metric: Metric) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        let mut v: Vec<f32> = (0..n * dim)
            .map(|i| (r.next_f64_open() as f32 - 0.5) + 0.3 * ((i % dim) as f32 / dim as f32))
            .collect();
        for row in v.chunks_exact_mut(dim) {
            prepare(metric, row);
        }
        v
    }

    #[test]
    fn rotations_are_orthogonal() {
        let dim = 40;
        let p = random_rotation(dim);
        for i in 0..dim {
            for j in 0..dim {
                let d: f32 = (0..dim).map(|t| p[i * dim + t] * p[j * dim + t]).sum();
                let want = if i == j { 1.0 } else { 0.0 };
                assert!((d - want).abs() < 1e-5, "({i},{j}) = {d}");
            }
        }
    }

    #[test]
    fn estimates_track_exact_distances() {
        let k = kernels();
        let dim = 256;
        for metric in [Metric::Cosine, Metric::Dot, Metric::L2] {
            let v = data(500, dim, 3, metric);
            let quant = BinaryQuantizer::train(&v, dim, metric);
            let words = quant.words();
            let mut codes = vec![0u64; 500 * words];
            let mut factors = vec![0f32; 500 * FACTORS];
            for (i, row) in v.chunks_exact(dim).enumerate() {
                quant.encode(
                    k,
                    row,
                    &mut codes[i * words..(i + 1) * words],
                    &mut factors[i * FACTORS..(i + 1) * FACTORS],
                );
            }
            let q = &data(1, dim, 99, metric);
            let mut prepared = BinaryQuery::default();
            quant.prepare_query(k, q, &mut prepared);
            let (mut err, mut bias, mut scale) = (0f64, 0f64, 0f64);
            let qc: Vec<f32> = q.iter().zip(quant.centroid()).map(|(a, b)| a - b).collect();
            let qn = qc.iter().map(|x| x * x).sum::<f32>().sqrt();
            let exact: Vec<f32> = v
                .chunks_exact(dim)
                .map(|x| metric.distance(k, q, x))
                .collect();
            for (i, &e) in exact.iter().enumerate() {
                let est = quant.distance(
                    k,
                    &prepared,
                    &codes[i * words..(i + 1) * words],
                    &factors[i * FACTORS..(i + 1) * FACTORS],
                );
                err += f64::from((est - e).powi(2));
                bias += f64::from(est - e);
                let xn = v[i * dim..(i + 1) * dim]
                    .iter()
                    .zip(quant.centroid())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f32>()
                    .sqrt();
                scale += f64::from(xn * qn / (dim as f32).sqrt());
            }
            // RaBitQ's error is about |x - c|·|q - c|/√D times a constant
            // near 0.8 for vectors like these, and unbiased.
            // (L2 has the inner product twice.)
            let twice = if metric == Metric::L2 { 2.0 } else { 1.0 };
            let (rms, bias, scale) = ((err / 500.0).sqrt(), bias / 500.0, twice * scale / 500.0);
            assert!(
                rms < 1.0 * scale,
                "{metric:?}: rms error {rms} vs scale {scale}"
            );
            assert!(
                bias.abs() < 0.2 * scale,
                "{metric:?}: bias {bias} vs scale {scale}"
            );
        }
    }
}
