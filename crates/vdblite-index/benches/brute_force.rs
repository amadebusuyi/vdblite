//! Exact top-10 search over 100K random 128-dim vectors.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use vdblite_index::{Metric, brute_force};

const N: usize = 100_000;
const DIM: usize = 128;
const QUERIES: usize = 64;

fn values(n: usize, seed: u32) -> Vec<f32> {
    let mut x = seed.wrapping_mul(2_654_435_761).max(1);
    (0..n)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            x as f32 / u32::MAX as f32
        })
        .collect()
}

fn bench(c: &mut Criterion) {
    let data = values(N * DIM, 7);
    let queries = values(QUERIES * DIM, 11);

    let mut g = c.benchmark_group("brute_force/100k x 128");
    g.sample_size(20);
    g.throughput(Throughput::Elements(1));
    g.bench_function("search (1 query)", |b| {
        b.iter(|| brute_force::search(black_box(&data), DIM, Metric::L2, &queries[..DIM], 10))
    });
    g.throughput(Throughput::Elements(QUERIES as u64));
    g.bench_function("search_batch (64 queries)", |b| {
        b.iter(|| brute_force::search_batch(black_box(&data), DIM, Metric::L2, &queries, 10))
    });
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
