//! Phase 2 and 3 gates: int8-quantized segments with f32 re-ranking on SIFT1M.
//!
//! ```text
//! cargo run --release -p vdblite-index --example segments_sift -- data/sift \
//!     [--segment N] [--flat] [--filters]
//! ```
//!
//! `--segment 1000000` gives one segment (the Phase 2 setup); the default,
//! 65536, splits SIFT1M into 16 sealed segments searched in parallel.
//! `--filters` also measures filtered recall at several selectivities.

use std::sync::Arc;
use std::time::Instant;

use rayon::prelude::*;
use roaring::RoaringBitmap;
use vdblite_index::dataset::Sift1M;
use vdblite_index::distance::kernels;
use vdblite_index::hnsw::HnswParams;
use vdblite_index::index::{IndexConfig, SealedEntry, VectorIndex};
use vdblite_index::recall::recall_at_k_with_ties;
use vdblite_index::segment::SealedSegment;
use vdblite_index::{Metric, Neighbor, brute_force};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = String::from("data/sift");
    let mut segment_size = 65_536usize;
    let mut flat = false;
    let mut filters = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--segment" => {
                segment_size = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--segment N")
            }
            "--flat" => flat = true,
            "--filters" => filters = true,
            _ => dir = a,
        }
    }
    let ds = Sift1M::load(&dir, Some(1000)).expect("load SIFT1M (run scripts/fetch-sift1m.sh)");
    let dim = ds.base.dim;
    let base = &ds.base.data;
    let n = ds.base.len();
    let k = 10;

    let mut cfg = IndexConfig::new(dim, Metric::L2);
    cfg.hnsw = HnswParams {
        layered: !flat,
        ..cfg.hnsw
    };
    cfg.seal_threshold = segment_size;
    println!(
        "building {} segment(s) of up to {segment_size} ({} graph, M={}, efC={})",
        n.div_ceil(segment_size),
        if flat { "flat" } else { "layered" },
        cfg.hnsw.m,
        cfg.hnsw.ef_construction
    );
    let t = Instant::now();
    let mut index = VectorIndex::new(cfg);
    for (i, start) in (0..n).step_by(segment_size).enumerate() {
        let end = (start + segment_size).min(n);
        let ids: Vec<u32> = (start as u32..end as u32).collect();
        let seg = SealedSegment::build(
            &ids,
            &base[start * dim..end * dim],
            dim,
            Metric::L2,
            &cfg.hnsw,
        );
        index.restore_sealed(SealedEntry::new(i as u64 + 1, Arc::new(seg)));
    }
    println!(
        "built in {:.1?}; resident index {:.1} MB (f32 vectors would add {:.1} MB)",
        t.elapsed(),
        index.memory_bytes() as f64 / 1e6,
        (n * dim * 4) as f64 / 1e6
    );

    let truth_kth: Vec<f32> = ds
        .queries
        .rows()
        .zip(&ds.groundtruth)
        .map(|(q, gt)| {
            let id = gt[k - 1] as usize;
            Metric::L2.distance(kernels(), q, &base[id * dim..(id + 1) * dim])
        })
        .collect();

    println!(
        "\n{:>5} {:>10} {:>9} {:>9} {:>9}",
        "ef", "recall@10", "QPS", "p50 µs", "p99 µs"
    );
    for ef in [16, 32, 64, 128, 256] {
        let (results, qps, p50, p99) = run(&index, &ds.queries.data, dim, k, ef, None);
        println!(
            "{ef:>5} {:>10.4} {qps:>9.0} {p50:>9.0} {p99:>9.0}",
            recall_at_k_with_ties(&results, &truth_kth, k)
        );
    }

    if filters {
        println!("\nfiltered search, ef=64:");
        println!(
            "{:>11} {:>10} {:>9} {:>9}",
            "selectivity", "recall@10", "p50 µs", "p99 µs"
        );
        for sel in [0.5, 0.1, 0.03, 0.01, 0.001] {
            // Deterministic pseudo-random subset of ids.
            let allowed: RoaringBitmap = (0..n as u32)
                .filter(|&i| {
                    (i.wrapping_mul(2_654_435_761) >> 8) as f64 / (1u64 << 24) as f64 <= sel
                })
                .collect();
            let subset_ids: Vec<u32> = allowed.iter().collect();
            let subset: Vec<f32> = subset_ids
                .iter()
                .flat_map(|&i| {
                    base[i as usize * dim..(i as usize + 1) * dim]
                        .iter()
                        .copied()
                })
                .collect();
            let truth: Vec<f32> =
                brute_force::search_batch(&subset, dim, Metric::L2, &ds.queries.data, k)
                    .iter()
                    .map(|r| r.get(k - 1).map_or(f32::INFINITY, |n| n.dist))
                    .collect();
            let (results, _, p50, p99) = run(&index, &ds.queries.data, dim, k, 64, Some(&allowed));
            assert!(results.iter().flatten().all(|h| allowed.contains(h.id)));
            println!(
                "{:>10.1}% {:>10.4} {p50:>9.0} {p99:>9.0}",
                sel * 100.0,
                recall_at_k_with_ties(&results, &truth, k)
            );
        }
    }

    // Throughput with many concurrent queries (one query per rayon task).
    let t = Instant::now();
    let _: Vec<_> = ds
        .queries
        .data
        .par_chunks(dim)
        .map(|q| index.search(q, k, 64, None))
        .collect();
    println!(
        "\nconcurrent throughput at ef=64: {:.0} QPS",
        ds.queries.len() as f64 / t.elapsed().as_secs_f64()
    );
}

fn run(
    index: &VectorIndex,
    queries: &[f32],
    dim: usize,
    k: usize,
    ef: usize,
    filter: Option<&RoaringBitmap>,
) -> (Vec<Vec<Neighbor>>, f64, f64, f64) {
    let mut lat = Vec::new();
    let mut results = Vec::new();
    let total = Instant::now();
    for q in queries.chunks_exact(dim) {
        let t = Instant::now();
        results.push(index.search(q, k, ef, filter));
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let qps = results.len() as f64 / total.elapsed().as_secs_f64();
    lat.sort_by(f64::total_cmp);
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    (results, qps, pct(0.5), pct(0.99))
}
