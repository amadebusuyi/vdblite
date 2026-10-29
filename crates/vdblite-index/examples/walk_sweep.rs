//! Walk codes and re-rank depth against recall@100 and speed, on one graph.
//!
//! ```text
//! cargo run --release -p vdblite-index --example walk_sweep -- \
//!     --base base.f32 --queries queries.f32 --gt gt.u64 [--dim 768] \
//!     [--ef 100,200,400] [--rerank 2,4,8] [--threads 8]
//! ```
//!
//! `base.f32` holds the vectors row by row (row `i` has id `i`),
//! `queries.f32` the queries, and `gt.u64` the 100 true neighbours of each
//! query (cosine). Builds one int8 graph (M=16, efConstruction=200, as
//! VectorDBBench runs), derives int8 and 1-bit segments from it, and for
//! each (codes, re-rank factor, ef) reports recall@100, mean single-thread
//! latency and multi-thread QPS.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;
use vdblite_index::Metric;
use vdblite_index::hnsw::HnswParams;
use vdblite_index::index::{IndexConfig, SealedEntry, VectorIndex};
use vdblite_index::segment::{Quantization, SealedSegment, prepare};

const K: usize = 100;

fn read<T: bytemuck::Pod>(path: &str) -> Vec<T> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{path}: {e}"));
    let n = bytes.len() / size_of::<T>();
    let mut out = vec![T::zeroed(); n];
    bytemuck::cast_slice_mut::<T, u8>(&mut out).copy_from_slice(&bytes[..n * size_of::<T>()]);
    out
}

fn main() {
    let mut args = std::env::args().skip(1);
    let (mut base, mut queries, mut gt) = (String::new(), String::new(), String::new());
    let mut dim = 768usize;
    let list =
        |s: String| -> Vec<usize> { s.split(',').map(|v| v.trim().parse().unwrap()).collect() };
    let (mut efs, mut reranks) = (vec![100, 200, 400], vec![2, 4, 8]);
    let mut threads = 8usize;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--base" => base = args.next().expect("--base FILE"),
            "--queries" => queries = args.next().expect("--queries FILE"),
            "--gt" => gt = args.next().expect("--gt FILE"),
            "--dim" => dim = args.next().and_then(|v| v.parse().ok()).expect("--dim D"),
            "--ef" => efs = list(args.next().expect("--ef 100,200")),
            "--rerank" => reranks = list(args.next().expect("--rerank 2,4")),
            "--threads" => {
                threads = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--threads N")
            }
            other => panic!("unknown argument {other}"),
        }
    }
    let mut data: Vec<f32> = read(&base);
    let mut queries: Vec<f32> = read(&queries);
    let truth: Vec<u64> = read(&gt);
    let n = data.len() / dim;
    let nq = queries.len() / dim;
    let gt_k = truth.len() / nq;
    data.par_chunks_mut(dim)
        .for_each(|v| prepare(Metric::Cosine, v));
    queries
        .par_chunks_mut(dim)
        .for_each(|v| prepare(Metric::Cosine, v));
    let ids: Vec<u32> = (0..n as u32).collect();

    let t = Instant::now();
    let int8 = SealedSegment::build(&ids, &data, dim, Metric::Cosine, &HnswParams::default());
    println!("{n} x {dim}: int8 graph built in {:.1?}", t.elapsed());
    drop(data);
    let t = Instant::now();
    let binary = int8.requantize(Quantization::Binary);
    println!("1-bit codes in {:.1?}", t.elapsed());
    let segments = [("int8", Arc::new(int8)), ("1-bit", Arc::new(binary))];

    println!(
        "\n{:<6} {:>6} {:>5} {:>10} {:>9} {:>9}",
        "codes", "rerank", "ef", "recall@100", "mean ms", "QPS"
    );
    for (name, seg) in &segments {
        println!(
            "{name}: walked bytes {:.0} MB",
            seg.memory_bytes() as f64 / 1e6
        );
        for &rerank in &reranks {
            let cfg = IndexConfig {
                rerank,
                ..IndexConfig::new(dim, Metric::Cosine)
            };
            let mut index = VectorIndex::new(cfg);
            index.restore_sealed(SealedEntry::new(1, seg.clone()));
            for &ef in &efs {
                let t = Instant::now();
                let mut hits = 0usize;
                for (qi, q) in queries.chunks_exact(dim).enumerate() {
                    let want = &truth[qi * gt_k..qi * gt_k + K];
                    hits += index
                        .search(q, K, ef, None)
                        .iter()
                        .filter(|h| want.contains(&u64::from(h.id)))
                        .count();
                }
                let mean = t.elapsed().as_secs_f64() * 1e3 / nq as f64;
                let recall = hits as f64 / (nq * K) as f64;
                let qps = throughput(&index, &queries, dim, ef, threads, Duration::from_secs(3));
                println!("{name:<6} {rerank:>6} {ef:>5} {recall:>10.4} {mean:>9.3} {qps:>9.0}");
            }
        }
    }
}

fn throughput(
    index: &VectorIndex,
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
                    index.search(&queries[i * dim..(i + 1) * dim], K, ef, None);
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
