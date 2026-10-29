//! What an application sees while it writes and searches at once, measured
//! through the public API at 768 dimensions.
//!
//! ```text
//! scripts/fetch-arxiv768.sh            # real embeddings: 1M arXiv titles (InstructorXL)
//! cargo run --release -p vdblite --example steady_state -- --data data/arxiv768 [--n 1000000]
//! cargo run --release -p vdblite --example steady_state -- --synthetic [--n 1000000]
//! ```
//!
//! Phases, each printed as it finishes:
//!
//! 1. Bulk ingest in 1,000-document commits: commit latency while sealing
//!    runs in the background.
//! 2. Reopen time and resident index, for a writer and a reading process.
//! 3. Search (with hydration) on the index as background maintenance left it,
//!    then after `optimize`: recall@10 against exact search, p50/p99.
//! 4. Filters through the API, including numeric ranges over a unique field.
//! 5. Hybrid search (BM25 + vectors) with each query's own text.
//! 6. Search with the write buffer a quarter, half and nearly all full.
//! 7. Search and small commits while a seal builds in the background.
//! 8. A reading process following the writer.
//! 9. Single-document commits: by collection size, and shared vs exclusive.
//!
//! The built file is kept at `--out` (default in the temp dir) and reused
//! with `--reuse`; phases 2–9 run on a copy, so reruns start from the same
//! state. `--seal N` sets the seal threshold (default 65,536) for quick runs.

use std::path::{Path, PathBuf};
use std::time::Instant;

use vdblite::advanced::index::Neighbor;
use vdblite::advanced::index::brute_force;
use vdblite::advanced::index::dataset::{SyntheticEmbeddings, read_fvecs};
use vdblite::advanced::index::recall::recall_at_k_with_ties;
use vdblite::advanced::index::rng::SplitMix64;
use vdblite::{Chunking, Collection, Db, Doc, Filter, IndexOptions, Metric, Quant, Schema, meta};

const BATCH: usize = 1000;
const K: usize = 10;

struct Data {
    dim: usize,
    n: usize,
    base: Vec<f32>,
    titles: Vec<String>,
    queries: Vec<f32>,
    query_titles: Vec<String>,
}

impl Data {
    fn row(&self, i: usize) -> &[f32] {
        &self.base[i * self.dim..(i + 1) * self.dim]
    }

    fn query(&self, q: usize) -> &[f32] {
        &self.queries[q * self.dim..(q + 1) * self.dim]
    }

    fn nq(&self) -> usize {
        self.queries.len() / self.dim
    }
}

/// Latencies in microseconds.
#[derive(Default)]
struct Lat(Vec<f64>);

impl Lat {
    fn time<R>(&mut self, f: impl FnOnce() -> R) -> R {
        let t = Instant::now();
        let r = f();
        self.0.push(t.elapsed().as_secs_f64() * 1e6);
        r
    }

    fn pct(&self, p: f64) -> f64 {
        let mut v = self.0.clone();
        v.sort_by(f64::total_cmp);
        v[((v.len() as f64 * p) as usize).min(v.len() - 1)]
    }

    fn summary(&self) -> String {
        format!(
            "p50 {}  p99 {}  max {}  (n={})",
            fmt_us(self.pct(0.5)),
            fmt_us(self.pct(0.99)),
            fmt_us(self.pct(1.0)),
            self.0.len()
        )
    }
}

fn fmt_us(us: f64) -> String {
    if us >= 1e6 {
        format!("{:.2} s", us / 1e6)
    } else if us >= 1e3 {
        format!("{:.2} ms", us / 1e3)
    } else {
        format!("{us:.0} µs")
    }
}

fn main() -> vdblite::Result<()> {
    let mut args = std::env::args().skip(1);
    let mut data_dir: Option<PathBuf> = None;
    let mut n = 1_000_000usize;
    let mut nq = 1000usize;
    let mut out = std::env::temp_dir().join("vdblite-steady.vdb");
    let mut reuse = false;
    let mut seal = IndexOptions::default().seal_threshold;
    let mut phases: Vec<u32> = (3..=9).collect();
    let mut ingest_opts = IngestOptions::default();
    let mut ingest_only = false;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--data" => data_dir = Some(args.next().expect("--data DIR").into()),
            "--synthetic" => data_dir = None,
            "--n" => n = args.next().and_then(|v| v.parse().ok()).expect("--n N"),
            "--queries" => {
                nq = args
                    .next()
                    .and_then(|v| v.parse().ok())
                    .expect("--queries N")
            }
            "--out" => out = args.next().expect("--out PATH").into(),
            "--reuse" => reuse = true,
            // Smaller for quick runs; the default is the library's.
            "--seal" => seal = args.next().and_then(|v| v.parse().ok()).expect("--seal N"),
            // Measurement phases to run, e.g. `--phases 7,9` (2 always runs).
            "--phases" => {
                phases = args
                    .next()
                    .expect("--phases 3,4,...")
                    .split(',')
                    .map(|p| p.trim().parse().expect("a phase number"))
                    .collect()
            }
            // Ingest variants, to see where commit time goes.
            "--plain" => ingest_opts.plain = true,
            "--exclusive" => ingest_opts.exclusive = true,
            "--ingest-only" => ingest_only = true,
            other => panic!("unknown argument {other}"),
        }
    }
    let t = Instant::now();
    let data = match &data_dir {
        Some(dir) => load_real(dir, n, nq),
        None => synthetic(n, nq),
    };
    println!(
        "data: {} x {} ({}), {} queries, loaded in {:.1?}",
        data.n,
        data.dim,
        data_dir
            .as_ref()
            .map_or("synthetic".into(), |d| d.display().to_string()),
        data.nq(),
        t.elapsed()
    );

    if !(reuse && out.exists()) {
        let _ = std::fs::remove_file(&out);
        ingest(&out, &data, seal, ingest_opts)?;
    }
    if ingest_only {
        return Ok(());
    }
    let work = out.with_extension("work.vdb");
    std::fs::copy(&out, &work).expect("copy the built file");
    let truth = ground_truth(&data);
    let r = measure(&work, &data, &truth, &phases);
    let _ = std::fs::remove_file(&work);
    r
}

fn load_real(dir: &Path, n: usize, nq: usize) -> Data {
    let v = read_fvecs(dir.join("base.fvecs"), Some(n + nq))
        .expect("read base.fvecs (run scripts/fetch-arxiv768.sh)");
    let dim = v.dim;
    let rows = v.data.len() / dim;
    assert!(rows > nq, "need more than {nq} vectors");
    let n = n.min(rows - nq);
    let text = std::fs::read_to_string(dir.join("titles.txt")).expect("read titles.txt");
    let mut lines: Vec<String> = text.lines().take(n + nq).map(str::to_owned).collect();
    lines.resize(n + nq, String::new());
    let query_titles = lines.split_off(n);
    let mut base = v.data;
    let mut queries = base.split_off(n * dim);
    queries.truncate(nq * dim);
    Data {
        dim,
        n,
        base,
        titles: lines,
        queries,
        query_titles,
    }
}

fn synthetic(n: usize, nq: usize) -> Data {
    use rayon::prelude::*;
    let dim = 768;
    let g = SyntheticEmbeddings::new(dim, 7);
    let mut base = vec![0f32; n * dim];
    base.par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, row)| g.vector(i as u64, row));
    let mut queries = vec![0f32; nq * dim];
    queries
        .par_chunks_mut(dim)
        .enumerate()
        .for_each(|(i, row)| g.vector(10_000_000_000 + i as u64, row));
    Data {
        dim,
        n,
        base,
        titles: (0..n).map(|i| words(i as u64)).collect(),
        queries,
        query_titles: (0..nq).map(|i| words(10_000_000_000 + i as u64)).collect(),
    }
}

/// Synthetic title text: a few words from a skewed 20K-word vocabulary.
fn words(seed: u64) -> String {
    let mut r = SplitMix64::new(seed ^ 0x5EED);
    let n = 6 + r.next_u64() % 7;
    (0..n)
        .map(|_| format!("w{}", (r.next_f64_open().powi(3) * 20_000.0) as u32))
        .collect::<Vec<_>>()
        .join(" ")
}

fn schema(dim: usize) -> Schema {
    Schema {
        dim,
        metric: Metric::Cosine,
        quant: Quant::Int8,
        chunking: Chunking::None,
    }
}

fn doc(data: &Data, i: usize) -> Doc {
    Doc::new(i.to_string())
        .content(data.titles[i].clone())
        .meta("cat", (i % 100) as i64)
        .meta("lang", if i.is_multiple_of(2) { "en" } else { "fr" })
        .meta("ts", i as i64)
}

#[derive(Clone, Copy, Default)]
struct IngestOptions {
    /// No keyword index and no metadata: vectors and text only.
    plain: bool,
    /// Open with `Db::open_exclusive` (one flush per commit).
    exclusive: bool,
}

fn ingest(out: &Path, data: &Data, seal: usize, o: IngestOptions) -> vdblite::Result<()> {
    println!(
        "\n## 1. Bulk ingest ({BATCH}-document commits, seal threshold {seal}{}{})",
        if o.plain {
            ", no keyword index or metadata"
        } else {
            ""
        },
        if o.exclusive { ", exclusive" } else { "" },
    );
    let db = if o.exclusive {
        Db::open_exclusive(out)?
    } else {
        Db::open(out)?
    };
    let opts = IndexOptions {
        seal_threshold: seal,
        lexical: !o.plain,
        ..IndexOptions::default()
    };
    let coll = db.create_collection_with("docs", schema(data.dim), opts)?;
    let (mut commits, mut upserts) = (Lat::default(), Lat::default());
    let t = Instant::now();
    for start in (0..data.n).step_by(BATCH) {
        let mut tx = db.begin_write()?;
        upserts.time(|| -> vdblite::Result<()> {
            for i in start..(start + BATCH).min(data.n) {
                let d = if o.plain {
                    Doc::new(i.to_string()).content(data.titles[i].clone())
                } else {
                    doc(data, i)
                };
                tx.upsert_with_vectors(&coll, d, vec![data.row(i).to_vec()])?;
            }
            Ok(())
        })?;
        commits.time(|| tx.commit())?;
    }
    let ingest = t.elapsed();
    println!(
        "ingested {} docs in {:.1?} ({:.0} docs/s)",
        data.n,
        ingest,
        data.n as f64 / ingest.as_secs_f64(),
    );
    println!("  upserts per batch: {}", upserts.summary());
    println!("  commits:           {}", commits.summary());
    println!("at end of ingest: {:?}", coll.index_stats());
    let t = Instant::now();
    db.wait_for_maintenance()?;
    println!(
        "background maintenance caught up {:.1?} later: {:?}",
        t.elapsed(),
        coll.index_stats()
    );
    Ok(())
}

fn ground_truth(data: &Data) -> Vec<Vec<Neighbor>> {
    let t = Instant::now();
    let truth = brute_force::search_batch(&data.base, data.dim, Metric::Cosine, &data.queries, K);
    println!(
        "exact ground truth for {} queries in {:.1?}",
        data.nq(),
        t.elapsed()
    );
    truth
}

/// Runs every query through `search`, returning recall@10 (tie-aware) and
/// latencies.
fn run_queries(
    coll: &Collection,
    truth: &[Vec<Neighbor>],
    nq: usize,
    search: impl Fn(&Collection, usize) -> Vec<vdblite::ChunkHit>,
) -> (f64, Lat) {
    let mut lat = Lat::default();
    let mut found = Vec::with_capacity(nq);
    for q in 0..nq {
        let hits = lat.time(|| search(coll, q));
        found.push(
            hits.iter()
                .filter_map(|h| {
                    Some(Neighbor {
                        id: h.key.parse().ok()?,
                        dist: 1.0 - h.vector_score.unwrap_or(h.score),
                    })
                })
                .collect::<Vec<_>>(),
        );
    }
    let kth: Vec<f32> = truth[..nq].iter().map(|t| t[K - 1].dist).collect();
    (recall_at_k_with_ties(&found, &kth, K), lat)
}

fn search_table(label: &str, coll: &Collection, data: &Data, truth: &[Vec<Neighbor>]) {
    println!("\n{label}: {:?}", coll.index_stats());
    for ef in [32, 64, 128] {
        let (recall, lat) = run_queries(coll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(ef).chunks().unwrap()
        });
        println!("  ef={ef:<4} recall@10 {recall:.4}  {}", lat.summary());
    }
}

fn add_docs(db: &Db, coll: &Collection, data: &Data, prefix: &str, range: std::ops::Range<usize>) {
    for start in range.clone().step_by(4096) {
        let mut tx = db.begin_write().unwrap();
        for j in start..(start + 4096).min(range.end) {
            let i = j % data.n;
            tx.upsert_with_vectors(
                coll,
                Doc::new(format!("{prefix}-{j}")).content(data.titles[i].clone()),
                vec![data.row(i).to_vec()],
            )
            .unwrap();
        }
        tx.commit().unwrap();
    }
}

fn measure(
    work: &Path,
    data: &Data,
    truth: &[Vec<Neighbor>],
    phases: &[u32],
) -> vdblite::Result<()> {
    let run = |p: u32| phases.contains(&p);
    println!("\n## 2. Reopen");
    let size = std::fs::metadata(work).map(|m| m.len()).unwrap_or(0) as f64 / 1e9;
    let t = Instant::now();
    let db = Db::open(work)?;
    let coll = db.collection("docs")?;
    let open = t.elapsed();
    let stats = coll.index_stats();
    println!(
        "file {size:.2} GB; writer open {open:.2?}; resident index {:.0} MB in {} segments, {} buffered",
        stats.memory_bytes as f64 / 1e6,
        stats.sealed_segments,
        stats.buffered
    );
    let t = Instant::now();
    let reader = Db::open_read_only(work)?;
    let rcoll = reader.collection("docs")?;
    println!("reader open (alongside the writer) {:.2?}", t.elapsed());

    if run(3) {
        println!("\n## 3. Search with hydration, k={K}");
        // The first pass reads vectors and records from disk into the page cache.
        let (recall, lat) = run_queries(&coll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
        });
        println!(
            "cold first pass, ef=64: recall@10 {recall:.4}  {}",
            lat.summary()
        );
        search_table("as background maintenance left it", &coll, data, truth);
        let t = Instant::now();
        db.optimize()?;
        println!("\noptimize (seal + merge) in {:.1?}", t.elapsed());
        search_table("after optimize", &coll, data, truth);
    }

    if run(4) {
        println!("\n## 4. Filters (ef=64, 200 queries each)");
        let n = data.n as f64;
        let filters: Vec<(&str, Filter)> = vec![
            ("cat = 7 (1%)", meta("cat").eq(7)),
            ("lang = en (50%)", meta("lang").eq("en")),
            ("ts in first 1% (range)", meta("ts").between(0.0, n * 0.01)),
            ("ts in first 10% (range)", meta("ts").between(0.0, n * 0.1)),
            ("ts in first 50% (range)", meta("ts").between(0.0, n * 0.5)),
            (
                "cat = 7 and ts in first 50%",
                meta("cat").eq(7).and(meta("ts").between(0.0, n * 0.5)),
            ),
        ];
        for (label, f) in filters {
            let mut lat = Lat::default();
            for q in 0..200.min(data.nq()) {
                lat.time(|| {
                    coll.search(data.query(q))
                        .k(K)
                        .ef(64)
                        .filter(f.clone())
                        .chunks()
                        .unwrap()
                });
            }
            println!("  {label:<30} {}", lat.summary());
        }
    }

    if run(5) {
        println!("\n## 5. Hybrid search (each query's own title as text)");
        let (recall, plain) = run_queries(&coll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
        });
        let mut hybrid = Lat::default();
        for q in 0..data.nq() {
            hybrid.time(|| {
                coll.search(data.query(q))
                    .k(K)
                    .ef(64)
                    .text(data.query_titles[q].clone())
                    .chunks()
                    .unwrap()
            });
        }
        println!(
            "  vector only, ef=64: recall@10 {recall:.4}  {}",
            plain.summary()
        );
        println!("  hybrid,      ef=64: {}", hybrid.summary());
    }

    let threshold = coll.index_options().seal_threshold;
    let near_full = threshold - threshold / 64;
    if run(6) {
        println!("\n## 6. Search with a write buffer (ef=64)");
        for target in [threshold / 4, threshold / 2, near_full] {
            let buffered = coll.index_stats().buffered;
            add_docs(&db, &coll, data, "buf", buffered..target.max(buffered));
            let (_, lat) = run_queries(&coll, truth, data.nq(), |c, q| {
                c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
            });
            println!(
                "  buffered {:>6}: {}",
                coll.index_stats().buffered,
                lat.summary()
            );
        }
    }

    if run(7) {
        println!("\n## 7. While a seal builds in the background (ef=64)");
        let buffered = coll.index_stats().buffered;
        add_docs(&db, &coll, data, "seal", buffered..near_full.max(buffered));
        let (_, before) = run_queries(&coll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
        });
        println!(
            "  just before, {} buffered: {}",
            coll.index_stats().buffered,
            before.summary()
        );
        let over = threshold + threshold / 64;
        add_docs(&db, &coll, data, "seal", near_full.max(buffered)..over);
        let t = Instant::now();
        let (mut searches, mut commits) = (Lat::default(), Lat::default());
        let mut q = 0;
        while coll.index_stats().buffered >= threshold {
            searches.time(|| {
                coll.search(data.query(q % data.nq()))
                    .k(K)
                    .ef(64)
                    .chunks()
                    .unwrap()
            });
            if q % 20 == 0 {
                let mut tx = db.begin_write()?;
                let i = q % data.n;
                tx.upsert_with_vectors(
                    &coll,
                    Doc::new(format!("during-{q}")).content(data.titles[i].clone()),
                    vec![data.row(i).to_vec()],
                )?;
                commits.time(|| tx.commit())?;
            }
            q += 1;
        }
        println!("  seal of {over} chunks took {:.1?}", t.elapsed());
        println!("  searches meanwhile: {}", searches.summary());
        if !commits.0.is_empty() {
            println!("  1-doc commits meanwhile: {}", commits.summary());
        }
        db.wait_for_maintenance()?;
        db.optimize()?;
    }

    if run(8) {
        println!("\n## 8. A reading process following the writer");
        // The reader has been idle since phase 2: catching up replays every
        // commit, seal and merge since then.
        let t = Instant::now();
        rcoll.search(data.query(0)).k(K).chunks()?;
        println!("  reader catching up since phase 2: {:.2?}", t.elapsed());
        let mut tx = db.begin_write()?;
        for i in 0..BATCH {
            tx.upsert_with_vectors(
                &coll,
                Doc::new(format!("follow-{i}")).content(data.titles[i].clone()),
                vec![data.row(i).to_vec()],
            )?;
        }
        tx.commit()?;
        let t = Instant::now();
        rcoll.search(data.query(0)).k(K).chunks()?;
        println!(
            "  first reader search after a {BATCH}-doc commit (replays it): {:.2?}",
            t.elapsed()
        );
        for q in 0..data.nq() {
            rcoll.search(data.query(q)).k(K).ef(64).chunks()?; // warm the reader's cache
        }
        let (_, lat) = run_queries(&rcoll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
        });
        println!("  reader searches, ef=64: {}", lat.summary());
        let (_, lat) = run_queries(&coll, truth, data.nq(), |c, q| {
            c.search(data.query(q)).k(K).ef(64).chunks().unwrap()
        });
        println!("  writer searches, ef=64: {}", lat.summary());
    }
    drop((rcoll, reader));

    if run(9) {
        println!("\n## 9. Single-document commits (200 each)");
        let small = |db: &Db, coll: &Collection, label: &str| {
            let mut lat = Lat::default();
            for j in 0..200 {
                let i = j % data.n;
                let mut tx = db.begin_write().unwrap();
                tx.upsert_with_vectors(
                    coll,
                    doc(data, i).content(format!("{label} {j}")),
                    vec![data.row(i).to_vec()],
                )
                .unwrap();
                lat.time(|| tx.commit()).unwrap();
            }
            lat
        };
        println!(
            "  shared, {} docs: {}",
            coll.len()?,
            small(&db, &coll, "edit").summary()
        );
        drop(coll);
        db.wait_for_maintenance()?;
        drop(db);
        // Exclusive mode, same file; and both modes on a tiny collection.
        let db = Db::open_exclusive(work)?;
        let coll = db.collection("docs")?;
        println!(
            "  exclusive, {} docs: {}",
            coll.len()?,
            small(&db, &coll, "edit").summary()
        );
        drop((coll, db));
        let tiny = work.with_extension("tiny.vdb");
        for exclusive in [false, true] {
            let _ = std::fs::remove_file(&tiny);
            let db = if exclusive {
                Db::open_exclusive(&tiny)?
            } else {
                Db::open(&tiny)?
            };
            let coll = db.create_collection("docs", schema(data.dim))?;
            add_docs(&db, &coll, data, "tiny", 0..1_000);
            let label = if exclusive { "exclusive" } else { "shared" };
            println!(
                "  {label}, 1000 docs: {}",
                small(&db, &coll, "edit").summary()
            );
        }
        let _ = std::fs::remove_file(&tiny);
    }
    Ok(())
}
