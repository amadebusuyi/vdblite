//! The headline target at full size: 1M vectors of 768 dimensions, cosine.
//!
//! ```text
//! cargo run --release -p vdblite-index --example synthetic768 -- [--n 1000000] [--segment 262144]
//! ```
//!
//! Vectors are synthetic but shaped like text embeddings (see
//! [`SyntheticEmbeddings`]). For real embeddings, see the `steady_state`
//! example in the `vdblite` crate with `scripts/fetch-arxiv768.sh`.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use vdblite_index::dataset::SyntheticEmbeddings;
use vdblite_index::index::{IndexConfig, SealedEntry, VectorIndex};
use vdblite_index::recall::recall_at_k_with_ties;
use vdblite_index::segment::SealedSegment;
use vdblite_index::{Metric, brute_force};

const DIM: usize = 768;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut n = 1_000_000usize;
    let mut segment = 262_144usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--n" => n = args.next().and_then(|v| v.parse().ok()).expect("--n N"),
            "--segment" => {
                segment = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--segment N")
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let g = SyntheticEmbeddings::new(DIM, 7);
    let t = Instant::now();
    let mut data = vec![0f32; n * DIM];
    data.par_chunks_mut(DIM)
        .enumerate()
        .for_each(|(i, row)| g.vector(i as u64, row));
    let queries_n = 1000;
    let mut queries = vec![0f32; queries_n * DIM];
    queries
        .par_chunks_mut(DIM)
        .enumerate()
        .for_each(|(i, row)| g.vector(10_000_000_000 + i as u64, row));
    println!("generated {n} x {DIM} in {:.1?}", t.elapsed());

    let t = Instant::now();
    let truth: Vec<f32> = brute_force::search_batch(&data, DIM, Metric::Cosine, &queries, 10)
        .iter()
        .map(|r| r[9].dist)
        .collect();
    println!(
        "exact ground truth for {queries_n} queries in {:.1?}",
        t.elapsed()
    );

    let cfg = IndexConfig::new(DIM, Metric::Cosine);
    let t = Instant::now();
    let mut index = VectorIndex::new(cfg);
    for (s, start) in (0..n).step_by(segment).enumerate() {
        let end = (start + segment).min(n);
        let ids: Vec<u32> = (start as u32..end as u32).collect();
        let seg = SealedSegment::build(
            &ids,
            &data[start * DIM..end * DIM],
            DIM,
            Metric::Cosine,
            &cfg.hnsw,
        );
        index.restore_sealed(SealedEntry::new(s as u64 + 1, Arc::new(seg)));
    }
    println!(
        "built {} segment(s) in {:.1?}; resident index {:.0} MB (f32 vectors, kept on disk, would add {:.0} MB)",
        index.sealed().len(),
        t.elapsed(),
        index.memory_bytes() as f64 / 1e6,
        (n * DIM * 4) as f64 / 1e6
    );

    println!(
        "\n{:>5} {:>10} {:>9} {:>9}",
        "ef", "recall@10", "p50 µs", "p99 µs"
    );
    for ef in [32, 64, 128] {
        let mut lat = Vec::new();
        let mut results = Vec::new();
        for q in queries.as_chunks::<DIM>().0 {
            let t = Instant::now();
            results.push(index.search(q, 10, ef, None));
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        lat.sort_by(f64::total_cmp);
        let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
        println!(
            "{ef:>5} {:>10.4} {:>9.0} {:>9.0}",
            recall_at_k_with_ties(&results, &truth, 10),
            pct(0.5),
            pct(0.99)
        );
    }
}
