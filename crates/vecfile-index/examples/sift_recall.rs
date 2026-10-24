//! Recall harness: runs exact search over SIFT1M and scores it against the
//! published ground truth.
//!
//! ```text
//! scripts/fetch-sift1m.sh
//! cargo run --release -p vecfile-index --example sift_recall -- data/sift [--queries N]
//! ```
//!
//! Brute force should reach recall 1.0 (ties at equal distance aside); this
//! proves the loader, kernels and recall maths before any index is measured.

use std::process::ExitCode;
use std::time::Instant;

use vecfile_index::dataset::Sift1M;
use vecfile_index::distance::kernels;
use vecfile_index::recall::{recall_at_k, recall_at_k_with_ties};
use vecfile_index::{Metric, brute_force};

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let mut dir = None;
    let mut queries = None;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--queries" => queries = args.next().and_then(|v| v.parse().ok()),
            "-h" | "--help" => {
                eprintln!("usage: sift_recall <sift-dir> [--queries N]");
                return ExitCode::SUCCESS;
            }
            _ => dir = Some(a),
        }
    }
    let Some(dir) = dir else {
        eprintln!("usage: sift_recall <sift-dir> [--queries N]");
        return ExitCode::FAILURE;
    };

    let t = Instant::now();
    let ds = match Sift1M::load(&dir, queries) {
        Ok(ds) => ds,
        Err(e) => {
            eprintln!("failed to load SIFT1M from {dir}: {e}");
            return ExitCode::FAILURE;
        }
    };
    println!(
        "loaded {} base x {} dim, {} queries in {:.2?}",
        ds.base.len(),
        ds.base.dim,
        ds.queries.len(),
        t.elapsed()
    );
    println!(
        "kernels: {}, threads: {}",
        kernels().name,
        rayon::current_num_threads()
    );

    let k = 100;
    let t = Instant::now();
    let results =
        brute_force::search_batch(&ds.base.data, ds.base.dim, Metric::L2, &ds.queries.data, k);
    let elapsed = t.elapsed();
    println!(
        "brute force top-{k}: {:.2?} total, {:.1} QPS, {:.2} ms/query",
        elapsed,
        ds.queries.len() as f64 / elapsed.as_secs_f64(),
        elapsed.as_secs_f64() * 1e3 / ds.queries.len() as f64
    );

    let found: Vec<Vec<u32>> = results
        .iter()
        .map(|r| r.iter().map(|n| n.id).collect())
        .collect();
    let mut ok = true;
    for at in [1, 10, 100] {
        let kth_dist: Vec<f32> = ds
            .queries
            .rows()
            .zip(&ds.groundtruth)
            .map(|(q, gt)| Metric::L2.distance(kernels(), q, ds.base.row(gt[at - 1] as usize)))
            .collect();
        let with_ties = recall_at_k_with_ties(&results, &kth_dist, at);
        ok &= with_ties == 1.0;
        println!(
            "recall@{at}: {:.4} by id, {:.4} counting ties",
            recall_at_k(&found, &ds.groundtruth, at),
            with_ties
        );
    }
    if ok {
        println!("PASS: exact search matches ground truth");
        ExitCode::SUCCESS
    } else {
        println!("FAIL: exact search missed true neighbours");
        ExitCode::FAILURE
    }
}
