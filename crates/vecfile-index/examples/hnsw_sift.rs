//! Phase 1 gate: in-memory f32 HNSW on SIFT1M.
//!
//! ```text
//! cargo run --release -p vecfile-index --example hnsw_sift -- data/sift [--flat] [--base N]
//! ```
//!
//! Builds the graph in parallel, then sweeps `ef` and reports recall@10
//! (tie-aware) and single-thread latency percentiles.

use std::time::Instant;

use vecfile_index::dataset::Sift1M;
use vecfile_index::distance::kernels;
use vecfile_index::hnsw::{HnswIndex, HnswParams, SearchScratch};
use vecfile_index::recall::recall_at_k_with_ties;
use vecfile_index::{Metric, brute_force};

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = String::from("data/sift");
    let mut flat = false;
    let mut base_limit = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--flat" => flat = true,
            "--base" => base_limit = args.next().and_then(|v| v.parse::<usize>().ok()),
            _ => dir = a,
        }
    }
    let ds = Sift1M::load(&dir, Some(1000)).expect("load SIFT1M (run scripts/fetch-sift1m.sh)");
    let dim = ds.base.dim;
    let mut base = ds.base.data;
    if let Some(n) = base_limit {
        base.truncate(n * dim);
    }
    let n = base.len() / dim;
    let k = 10;

    // Ground truth: the published one for the full set, brute force for a prefix.
    let truth_kth: Vec<f32> = if base_limit.is_none() {
        ds.queries
            .rows()
            .zip(&ds.groundtruth)
            .map(|(q, gt)| {
                let id = gt[k - 1] as usize;
                Metric::L2.distance(kernels(), q, &base[id * dim..(id + 1) * dim])
            })
            .collect()
    } else {
        brute_force::search_batch(&base, dim, Metric::L2, &ds.queries.data, k)
            .iter()
            .map(|r| r[k - 1].dist)
            .collect()
    };

    let params = HnswParams {
        layered: !flat,
        ..HnswParams::default()
    };
    println!(
        "building {} graph over {n} x {dim} (M={}, efConstruction={}, {} threads)",
        if flat { "flat" } else { "layered" },
        params.m,
        params.ef_construction,
        rayon::current_num_threads()
    );
    let t = Instant::now();
    let index = HnswIndex::build(base, dim, Metric::L2, &params);
    let build = t.elapsed();
    println!(
        "built in {build:.1?} ({:.0} inserts/s); graph {:.1} MB, max level {}",
        n as f64 / build.as_secs_f64(),
        index.graph.memory_bytes() as f64 / 1e6,
        index.graph.max_level()
    );

    println!(
        "\n{:>5} {:>10} {:>9} {:>9} {:>9}",
        "ef", "recall@10", "QPS", "p50 µs", "p99 µs"
    );
    let mut scratch = SearchScratch::new();
    for ef in [16, 32, 64, 128, 256] {
        let mut lat = Vec::with_capacity(ds.queries.len());
        let mut results = Vec::with_capacity(ds.queries.len());
        let total = Instant::now();
        for q in ds.queries.rows() {
            let t = Instant::now();
            results.push(index.search(q, k, ef, &mut scratch));
            lat.push(t.elapsed().as_secs_f64() * 1e6);
        }
        let total = total.elapsed().as_secs_f64();
        lat.sort_by(f64::total_cmp);
        let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
        println!(
            "{ef:>5} {:>10.4} {:>9.0} {:>9.0} {:>9.0}",
            recall_at_k_with_ties(&results, &truth_kth, k),
            ds.queries.len() as f64 / total,
            pct(0.50),
            pct(0.99)
        );
    }
}
