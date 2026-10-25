//! Graph parameters against recall, speed and memory, on real vectors.
//!
//! ```text
//! cargo run --release -p vecfile-index --example graph_sweep -- \
//!     --data data/arxiv768 [--n 200000] [--m 16,24,32] [--efc 200,400] [--ef 32,64,128,256]
//! ```
//!
//! Builds one int8 segment per (M, efConstruction) over the first `n` rows
//! of `base.fvecs` (cosine), and searches it with the rows after `n` as
//! queries: recall@10 against exact search, and 16-thread QPS. Re-ranking
//! reads full vectors from memory, so this measures the graph, not storage.
//!
//! Every graph is built first; throughput is then measured in `--rounds`
//! rounds that alternate between them, and the median is reported, so a
//! burst of other load on the machine can't favour one configuration.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use vecfile_index::dataset::read_fvecs;
use vecfile_index::hnsw::HnswParams;
use vecfile_index::index::{IndexConfig, SealedEntry, VectorIndex};
use vecfile_index::recall::recall_at_k;
use vecfile_index::segment::{DenseStore, SealedSegment, prepare};
use vecfile_index::{Metric, brute_force};

const K: usize = 10;

fn main() {
    let mut args = std::env::args().skip(1);
    let mut dir = String::from("data/arxiv768");
    let mut n = 200_000usize;
    let mut nq = 1000usize;
    let list =
        |s: String| -> Vec<usize> { s.split(',').map(|v| v.trim().parse().unwrap()).collect() };
    let (mut ms, mut efcs, mut efs) = (vec![16, 24, 32], vec![200], vec![32, 64, 128, 256]);
    let mut rounds = 3usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--data" => dir = args.next().expect("--data DIR"),
            "--n" => n = args.next().and_then(|v| v.parse().ok()).expect("--n N"),
            "--queries" => {
                nq = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--queries N")
            }
            "--m" => ms = list(args.next().expect("--m 16,32")),
            "--efc" => efcs = list(args.next().expect("--efc 200,400")),
            "--ef" => efs = list(args.next().expect("--ef 32,64")),
            "--rounds" => {
                rounds = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--rounds R")
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let rows = read_fvecs(format!("{dir}/base.fvecs"), Some(n + nq)).expect("read base.fvecs");
    let dim = rows.dim;
    let mut data = rows.data;
    let mut queries = data.split_off(n * dim);
    queries.truncate(nq * dim);
    data.par_chunks_mut(dim)
        .for_each(|v| prepare(Metric::Cosine, v));
    queries
        .par_chunks_mut(dim)
        .for_each(|v| prepare(Metric::Cosine, v));
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = brute_force::search_batch(&data, dim, Metric::Cosine, &queries, K)
        .into_iter()
        .map(|r| r.into_iter().map(|nb| nb.id).collect())
        .collect();
    println!(
        "{n} x {dim}, {nq} queries; exact ground truth in {:.1?}",
        t.elapsed()
    );
    let ids: Vec<u32> = (0..n as u32).collect();
    let store = DenseStore { data: &data, dim };

    let mut configs = Vec::new();
    for &m in &ms {
        for &efc in &efcs {
            let hnsw = HnswParams {
                m,
                ef_construction: efc,
                ..HnswParams::default()
            };
            let t = Instant::now();
            let seg = SealedSegment::build(&ids, &data, dim, Metric::Cosine, &hnsw);
            let built = t.elapsed();
            let cfg = IndexConfig {
                hnsw,
                ..IndexConfig::new(dim, Metric::Cosine)
            };
            let mut index = VectorIndex::new(cfg);
            index.restore_sealed(SealedEntry::new(1, Arc::new(seg)));
            println!(
                "M={m} efC={efc}: built in {built:.1?}, resident index {:.0} MB",
                index.memory_bytes() as f64 / 1e6
            );
            configs.push((format!("M={m} efC={efc}"), index));
        }
    }

    let mut qps = vec![vec![Vec::new(); efs.len()]; configs.len()];
    for _ in 0..rounds {
        for (c, (_, index)) in configs.iter().enumerate() {
            for (e, &ef) in efs.iter().enumerate() {
                qps[c][e].push(throughput(
                    index,
                    &store,
                    &queries,
                    dim,
                    ef,
                    16,
                    Duration::from_secs(3),
                ));
            }
        }
    }
    println!(
        "\n{:<16} {:>5} {:>10} {:>14}",
        "graph", "ef", "recall@10", "median QPS/16"
    );
    for (c, (name, index)) in configs.iter().enumerate() {
        for (e, &ef) in efs.iter().enumerate() {
            let found: Vec<Vec<u32>> = queries
                .chunks_exact(dim)
                .map(|q| {
                    index
                        .search(q, K, ef, None, &store)
                        .iter()
                        .map(|h| h.id)
                        .collect()
                })
                .collect();
            let recall = recall_at_k(&found, &truth, K);
            let mut runs = qps[c][e].clone();
            runs.sort_by(f64::total_cmp);
            println!(
                "{name:<16} {ef:>5} {recall:>10.4} {:>14.0}",
                runs[runs.len() / 2]
            );
        }
    }
}

fn throughput(
    index: &VectorIndex,
    store: &DenseStore,
    queries: &[f32],
    dim: usize,
    ef: usize,
    threads: usize,
    secs: Duration,
) -> f64 {
    let nq = queries.len() / dim;
    let (done, stop) = (AtomicUsize::new(0), AtomicBool::new(false));
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            let (done, stop) = (&done, &stop);
            s.spawn(move || {
                let mut q = th * 97;
                while !stop.load(Ordering::Relaxed) {
                    let i = q % nq;
                    index.search(&queries[i * dim..(i + 1) * dim], K, ef, None, store);
                    done.fetch_add(1, Ordering::Relaxed);
                    q += 1;
                }
            });
        }
        std::thread::sleep(secs);
        stop.store(true, Ordering::Relaxed);
    });
    done.load(Ordering::Relaxed) as f64 / t.elapsed().as_secs_f64()
}
