//! Search throughput and recall on an existing database, for tuning and
//! profiling.
//!
//! ```text
//! qps --db data/arxiv768-opt.vdb --data data/arxiv768 [--n 999000] \
//!     [--threads 1,16] [--ef 64] [--secs 10] [--mode chunks|ids] [--optimize]
//! ```
//!
//! The database is one built by `steady_state` (document `i` is row `i` of
//! `base.fvecs`, one chunk each); queries are the rows after the first `n`.
//! Exact ground truth is computed once and cached next to the data.
//! `--optimize` runs `Db::optimize` on the file first (and keeps it).
//!
//! `--mode ids` stops at chunk ids (`Search::chunk_ids`); `chunks` also
//! hydrates text and metadata, as an application would.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use vecfile::advanced::index::brute_force;
use vecfile::advanced::index::dataset::{read_fvecs, read_ivecs};
use vecfile::advanced::index::recall::recall_at_k;
use vecfile::{Collection, Db, Metric};

const K: usize = 10;

fn main() -> vecfile::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut db_path = PathBuf::new();
    let mut data = PathBuf::from("data/arxiv768");
    let mut n = 999_000usize;
    let mut nq = 1000usize;
    let mut threads = vec![1usize, 16];
    let mut efs = vec![64usize];
    let mut secs = 10u64;
    let mut ids = false;
    let mut optimize = false;
    let list =
        |s: String| -> Vec<usize> { s.split(',').map(|v| v.trim().parse().unwrap()).collect() };
    while let Some(a) = args.next() {
        match a.as_str() {
            "--db" => db_path = args.next().expect("--db PATH").into(),
            "--data" => data = args.next().expect("--data DIR").into(),
            "--n" => n = args.next().and_then(|v| v.parse().ok()).expect("--n N"),
            "--queries" => {
                nq = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--queries N")
            }
            "--threads" => threads = list(args.next().expect("--threads 1,16")),
            "--ef" => efs = list(args.next().expect("--ef 64,128")),
            "--secs" => secs = args.next().and_then(|v| v.parse().ok()).expect("--secs S"),
            "--mode" => ids = args.next().expect("--mode chunks|ids") == "ids",
            "--optimize" => optimize = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let rows = read_fvecs(data.join("base.fvecs"), Some(n + nq)).expect("read base.fvecs");
    let dim = rows.dim;
    let queries = rows.data[n * dim..(n + nq) * dim].to_vec();
    let truth = ground_truth(&data, &rows.data[..n * dim], &queries, dim, n);
    drop(rows);

    if optimize {
        let t = Instant::now();
        let db = Db::open(&db_path)?;
        db.optimize()?;
        println!("optimize in {:.1?}", t.elapsed());
    }
    let t = Instant::now();
    let db = Db::open(&db_path)?;
    let coll = db.collection("docs")?;
    println!(
        "opened in {:.2?}: {:?}; mode {}",
        t.elapsed(),
        coll.index_stats(),
        if ids { "ids" } else { "chunks" }
    );

    for &ef in &efs {
        // Recall and a warm-up pass in one.
        let found: Vec<Vec<u32>> = (0..nq)
            .map(|q| search(&coll, &queries[q * dim..(q + 1) * dim], ef, ids))
            .collect();
        let recall = recall_at_k(&found, &truth, K);
        for &t in &threads {
            let qps = throughput(&coll, &queries, dim, ef, ids, t, Duration::from_secs(secs));
            println!("ef={ef:<4} recall@10 {recall:.4}  threads {t:>2}: {qps:>8.0} QPS");
            std::io::stdout().flush().ok();
        }
    }
    Ok(())
}

/// Row ids of the `K` nearest neighbours.
fn search(coll: &Collection, q: &[f32], ef: usize, ids: bool) -> Vec<u32> {
    let s = coll.search(q).k(K).ef(ef);
    if ids {
        s.chunk_ids()
            .unwrap()
            .into_iter()
            .map(|(id, _)| id)
            .collect()
    } else {
        s.chunks()
            .unwrap()
            .into_iter()
            .map(|h| h.key.parse().unwrap())
            .collect()
    }
}

fn throughput(
    coll: &Collection,
    queries: &[f32],
    dim: usize,
    ef: usize,
    ids: bool,
    threads: usize,
    secs: Duration,
) -> f64 {
    let nq = queries.len() / dim;
    let done = AtomicUsize::new(0);
    let stop = AtomicBool::new(false);
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            let (done, stop) = (&done, &stop);
            s.spawn(move || {
                let mut q = th * 97;
                while !stop.load(Ordering::Relaxed) {
                    let i = q % nq;
                    search(coll, &queries[i * dim..(i + 1) * dim], ef, ids);
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

fn ground_truth(dir: &Path, base: &[f32], queries: &[f32], dim: usize, n: usize) -> Vec<Vec<u32>> {
    let nq = queries.len() / dim;
    let path = dir.join(format!("truth-{n}-{nq}.ivecs"));
    if let Ok(v) = read_ivecs(&path, None)
        && v.data.len() == nq * K
    {
        return v
            .data
            .chunks(K)
            .map(|r| r.iter().map(|&i| i as u32).collect())
            .collect();
    }
    let t = Instant::now();
    let truth: Vec<Vec<u32>> = brute_force::search_batch(base, dim, Metric::Cosine, queries, K)
        .into_iter()
        .map(|r| r.into_iter().map(|nb| nb.id).collect())
        .collect();
    println!(
        "exact ground truth in {:.1?} (cached in {})",
        t.elapsed(),
        path.display()
    );
    let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("write truth"));
    for row in &truth {
        f.write_all(&(K as i32).to_le_bytes()).unwrap();
        for &id in row {
            f.write_all(&(id as i32).to_le_bytes()).unwrap();
        }
    }
    truth
}
