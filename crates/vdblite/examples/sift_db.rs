//! End-to-end benchmark through the public API: SIFT1M as one million
//! single-chunk documents in a `.vdb` file.
//!
//! ```text
//! cargo run --release -p vdblite --example sift_db -- data/sift [--out /tmp/sift.vdb] [--base N]
//! ```
//!
//! Measures ingest, sealing, optimize, file size, reopen time, resident index
//! size, and search latency/recall including hydration from the file.

use std::time::Instant;

use vdblite::advanced::index::dataset::Sift1M;
use vdblite::advanced::index::recall::recall_at_k;
use vdblite::{Chunking, Db, Doc, IndexOptions, Metric, Quant, Schema};

fn main() -> vdblite::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut dir = String::from("data/sift");
    let mut out = std::env::temp_dir().join("vdblite-sift.vdb");
    let mut base_limit = None;
    let mut reuse = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--reuse" => reuse = true,
            "--out" => out = args.next().expect("--out PATH").into(),
            "--base" => base_limit = args.next().and_then(|v| v.parse::<usize>().ok()),
            _ => dir = a,
        }
    }
    let ds = Sift1M::load(&dir, Some(1000)).expect("load SIFT1M (run scripts/fetch-sift1m.sh)");
    let _dim = ds.base.dim;
    let n = base_limit.unwrap_or(ds.base.len()).min(ds.base.len());
    if !(reuse && out.exists()) {
        let _ = std::fs::remove_file(&out);
        build(&out, &ds, n)?;
    }
    search(&out, &ds, n)
}

fn build(out: &std::path::Path, ds: &Sift1M, n: usize) -> vdblite::Result<()> {
    let dim = ds.base.dim;

    let db = Db::open(out)?;
    let schema = Schema {
        dim,
        metric: Metric::L2,
        quant: Quant::Int8,
        chunking: Chunking::None,
    };
    let coll = db.create_collection_with("sift", schema, IndexOptions::default())?;

    let batch = 10_000;
    let t = Instant::now();
    let mut seal_time = 0.0;
    for start in (0..n).step_by(batch) {
        let mut tx = db.begin_write()?;
        for i in start..(start + batch).min(n) {
            tx.upsert_with_vectors(
                &coll,
                Doc::new(i.to_string()).content(format!("v{i}")),
                vec![ds.base.row(i).to_vec()],
            )?;
        }
        let c = Instant::now();
        tx.commit()?;
        seal_time += c.elapsed().as_secs_f64();
    }
    let ingest = t.elapsed();
    println!(
        "ingested {n} docs in {ingest:.1?} ({:.0} docs/s; {seal_time:.1}s of it in commit + sealing)",
        n as f64 / ingest.as_secs_f64()
    );
    println!("after ingest: {:?}", coll.index_stats());

    let t = Instant::now();
    db.optimize()?;
    println!(
        "optimize (seal + merge) in {:.1?}: {:?}",
        t.elapsed(),
        coll.index_stats()
    );
    drop(coll);
    drop(db);
    let size = || std::fs::metadata(out).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
    println!("file size: {:.1} MB", size());
    let t = Instant::now();
    Db::compact_file(out)?;
    println!(
        "file size after compact_file: {:.1} MB ({:.1?})",
        size(),
        t.elapsed()
    );

    Ok(())
}

fn search(out: &std::path::Path, ds: &Sift1M, n: usize) -> vdblite::Result<()> {
    let t = Instant::now();
    let db = Db::open(out)?;
    let coll = db.collection("sift")?;
    let stats = coll.index_stats();
    println!(
        "reopened in {:.2?}; resident index {:.1} MB, {} segments",
        t.elapsed(),
        stats.memory_bytes as f64 / 1e6,
        stats.sealed_segments
    );

    let k = 10;
    // The first pass after opening reads vectors and records from disk into
    // the page cache; report it separately from steady state.
    for (label, ef) in [("cold", 64), ("warm", 32), ("warm", 64), ("warm", 128)] {
        let mut lat = Vec::new();
        let mut found = Vec::new();
        for q in ds.queries.rows() {
            let t = Instant::now();
            let hits = coll.search(q).k(k).ef(ef).chunks()?;
            lat.push(t.elapsed().as_secs_f64() * 1e6);
            found.push(
                hits.iter()
                    .map(|h| h.key.parse::<u32>().unwrap())
                    .collect::<Vec<u32>>(),
            );
        }
        lat.sort_by(f64::total_cmp);
        let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
        let recall = if n == ds.base.len() {
            recall_at_k(&found, &ds.groundtruth, k)
        } else {
            f64::NAN
        };
        println!(
            "{label} ef={ef:<4} recall@10 (by id) {recall:.4}  p50 {:.0} µs  p99 {:.0} µs  (search + hydrate)",
            pct(0.5),
            pct(0.99)
        );
    }

    // Throughput with one searching thread per core, each walking the query
    // set from a different offset; several passes so the run lasts long
    // enough to measure.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let passes = 10;
    let queries: Vec<&[f32]> = ds.queries.rows().collect();
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            let (coll, queries) = (&coll, &queries);
            s.spawn(move || {
                for i in 0..queries.len() * passes / threads {
                    let q = queries[(th * 97 + i) % queries.len()];
                    coll.search(q).k(k).ef(64).chunks().unwrap();
                }
            });
        }
    });
    let total = queries.len() * passes / threads * threads;
    println!(
        "concurrent ef=64, {threads} threads: {:.0} QPS (search + hydrate)",
        total as f64 / t.elapsed().as_secs_f64()
    );
    Ok(())
}
