//! Distance kernels, scalar against every SIMD set this CPU supports.

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use vecfile_index::distance::Kernels;

/// Deterministic pseudo-random values in [-1, 1) without pulling in `rand`.
fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x as f32 / u32::MAX as f32) * 2.0 - 1.0
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    for dim in [128usize, 768] {
        let a = values(dim, 1);
        let b = values(dim, 2);
        let ai: Vec<i8> = a.iter().map(|x| (x * 127.0) as i8).collect();
        let bi: Vec<i8> = b.iter().map(|x| (x * 127.0) as i8).collect();

        let mut g = c.benchmark_group(format!("distance/{dim}"));
        g.throughput(Throughput::Elements(dim as u64));
        for k in Kernels::available() {
            g.bench_function(BenchmarkId::new("dot_f32", k.name), |bn| {
                bn.iter(|| (k.dot_f32)(black_box(&a), black_box(&b)))
            });
            g.bench_function(BenchmarkId::new("l2_sq_f32", k.name), |bn| {
                bn.iter(|| (k.l2_sq_f32)(black_box(&a), black_box(&b)))
            });
            g.bench_function(BenchmarkId::new("dot_i8", k.name), |bn| {
                bn.iter(|| (k.dot_i8)(black_box(&ai), black_box(&bi)))
            });
            g.bench_function(BenchmarkId::new("l2_sq_i8", k.name), |bn| {
                bn.iter(|| (k.l2_sq_i8)(black_box(&ai), black_box(&bi)))
            });
            g.bench_function(BenchmarkId::new("dot_f32_i8", k.name), |bn| {
                bn.iter(|| (k.dot_f32_i8)(black_box(&a), black_box(&bi)))
            });
            g.bench_function(BenchmarkId::new("l2_sq_f32_i8", k.name), |bn| {
                bn.iter(|| (k.l2_sq_f32_i8)(black_box(&a), black_box(&bi)))
            });
        }
        g.finish();
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
