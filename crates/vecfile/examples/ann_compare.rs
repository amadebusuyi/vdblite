//! vecfile's side of the embedded comparison (`scripts/compare/embedded`):
//! the same phases, dataset files and JSON as `bench.py` there.
//!
//! ```text
//! ann_compare build  --data DIR --dataset NAME --db PATH --out build.json
//! ann_compare search --data DIR --dataset NAME --db PATH --out search.json
//! ```
//!
//! `NAME` is a directory made by `prep.py` under `DIR/bench`; its
//! `meta.json` names the base file, row count and metric. Vector-only (no
//! keyword index), like the other engines in the comparison.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vecfile::advanced::index::dataset::{read_fvecs, read_ivecs};
use vecfile::{Chunking, Collection, Db, Doc, IndexOptions, Metric, Quant, Schema};

const K: usize = 10;
const BATCH: usize = 10_000;
// vecfile raises ef to at least k x 4 = 40, so the sweep starts there.
const EFS: [usize; 5] = [40, 64, 128, 256, 512];
const REF_EF: usize = 64;

struct Meta {
    name: String,
    base: String,
    n: usize,
    metric: String,
}

/// Reads the four fields we need from prep.py's flat `meta.json`.
fn read_meta(path: &Path) -> Meta {
    let text = std::fs::read_to_string(path).expect("read meta.json (run prep.py)");
    let field = |key: &str| -> String {
        let at = text
            .find(&format!("\"{key}\":"))
            .unwrap_or_else(|| panic!("meta.json: no {key}"));
        let rest = text[at + key.len() + 3..].trim_start();
        let end = rest.find([',', '\n', '}']).unwrap();
        rest[..end].trim().trim_matches('"').to_string()
    };
    Meta {
        name: field("name"),
        base: field("base"),
        n: field("n").parse().unwrap(),
        metric: field("metric"),
    }
}

fn rss_mb(field: &str) -> f64 {
    let status = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    status
        .lines()
        .find_map(|l| l.strip_prefix(field)?.strip_prefix(':'))
        .and_then(|v| v.split_whitespace().next()?.parse::<f64>().ok())
        .map_or(f64::NAN, |kb| kb / 1024.0)
}

fn file_mb(path: &Path) -> f64 {
    std::fs::metadata(path).map_or(0.0, |m| m.len() as f64 / 1e6)
}

fn pct(lat: &[f64], p: f64) -> f64 {
    lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)]
}

fn recall(found: &[Vec<u32>], gt: &[&[i32]]) -> f64 {
    let hits: usize = found
        .iter()
        .zip(gt)
        .map(|(f, t)| {
            f.iter()
                .filter(|id| t[..K].contains(&(**id as i32)))
                .count()
        })
        .sum();
    hits as f64 / (found.len() * K) as f64
}

fn search_once(coll: &Collection, q: &[f32], ef: usize) -> Vec<u32> {
    let hits = coll.search(q).k(K).ef(ef).chunks().unwrap();
    hits.iter().map(|h| h.key.parse().unwrap()).collect()
}

/// One timed pass over the queries: sorted latencies (µs) and the ids found.
fn timed_pass(coll: &Collection, queries: &[&[f32]], ef: usize) -> (Vec<f64>, Vec<Vec<u32>>) {
    let mut lat = Vec::with_capacity(queries.len());
    let mut found = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        found.push(search_once(coll, q, ef));
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    lat.sort_by(f64::total_cmp);
    (lat, found)
}

fn summary(json: &mut String, knob: usize, lat: &[f64], found: &[Vec<u32>], gt: &[&[i32]]) {
    let mean = lat.iter().sum::<f64>() / lat.len() as f64;
    write!(
        json,
        r#"{{"knob": {{"ef": {knob}}}, "recall": {}, "p50_us": {}, "p99_us": {}, "mean_us": {mean}}}"#,
        recall(found, gt),
        pct(lat, 0.5),
        pct(lat, 0.99)
    )
    .unwrap();
}

fn build(db_path: &Path, data: &Path, meta: &Meta) -> vecfile::Result<String> {
    let base = read_fvecs(data.join(&meta.base), Some(meta.n)).expect("read base vectors");
    let metric = if meta.metric == "l2" {
        Metric::L2
    } else {
        Metric::Cosine
    };
    let rss0 = rss_mb("VmRSS");
    let t = Instant::now();
    let db = Db::open(db_path)?;
    let schema = Schema {
        dim: base.dim,
        metric,
        quant: Quant::Int8,
        chunking: Chunking::None,
    };
    let opts = IndexOptions {
        lexical: false,
        ..IndexOptions::default()
    };
    let coll = db.create_collection_with("bench", schema, opts)?;
    for start in (0..base.len()).step_by(BATCH) {
        let mut tx = db.begin_write()?;
        for i in start..(start + BATCH).min(base.len()) {
            // A document needs content to have a chunk; keep it tiny, as sift_db does.
            let doc = Doc::new(i.to_string()).content(format!("v{i}"));
            tx.upsert_with_vectors(&coll, doc, vec![base.row(i).to_vec()])?;
        }
        tx.commit()?;
    }
    let loaded = t.elapsed().as_secs_f64();
    // Commits seal in the background; settle that, then seal and merge the rest.
    db.wait_for_maintenance()?;
    db.optimize()?;
    db.wait_for_maintenance()?;
    let total = t.elapsed().as_secs_f64();
    drop(coll);
    drop(db);
    let disk = file_mb(db_path);
    let c = Instant::now();
    Db::compact_file(db_path)?;
    let compact_s = c.elapsed().as_secs_f64();
    let compacted = file_mb(db_path);
    println!(
        "ingest {} in {total:.1} s (load {loaded:.1} s, optimize {:.1} s), disk {disk:.0} MB \
         ({compacted:.0} MB after compact_file, {compact_s:.1} s), peak RSS {:.0} MB",
        base.len(),
        total - loaded,
        rss_mb("VmHWM")
    );
    Ok(format!(
        r#""phase": "build", "load_s": {loaded}, "index_s": {}, "ingest_s": {total}, "disk_mb": {disk}, "disk_compacted_mb": {compacted}, "compact_s": {compact_s}, "peak_rss_mb": {}, "rss_before_mb": {rss0}"#,
        total - loaded,
        rss_mb("VmHWM")
    ))
}

fn search(db_path: &Path, data: &Path, meta: &Meta) -> vecfile::Result<String> {
    let dir = data.join("bench").join(&meta.name);
    let queries = read_fvecs(dir.join("query.fvecs"), None).expect("read queries");
    let gt = read_ivecs(dir.join("gt.ivecs"), None).expect("read ground truth");
    let queries: Vec<&[f32]> = queries.rows().collect();
    let gt: Vec<&[i32]> = gt.rows().collect();

    let rss0 = rss_mb("VmRSS");
    let t = Instant::now();
    let db = Db::open(db_path)?;
    let coll = db.collection("bench")?;
    db.wait_for_maintenance()?;
    let open_s = t.elapsed().as_secs_f64();
    let rss_open = rss_mb("VmRSS") - rss0;
    let mut json = format!(
        r#""phase": "search", "open_s": {open_s}, "rss_open_mb": {rss_open}, "index_mb": {}, "#,
        coll.index_stats().memory_bytes as f64 / 1e6
    );

    let t = Instant::now();
    search_once(&coll, queries[0], REF_EF);
    let first = t.elapsed().as_secs_f64() * 1e6;
    let (lat, found) = timed_pass(&coll, &queries, REF_EF);
    write!(json, r#""first_query_us": {first}, "cold": "#).unwrap();
    summary(&mut json, REF_EF, &lat, &found, &gt);
    println!(
        "open {:.0} ms, first query {:.1} ms, cold pass p50 {:.0} µs p99 {:.0} µs",
        open_s * 1e3,
        first / 1e3,
        pct(&lat, 0.5),
        pct(&lat, 0.99)
    );

    json.push_str(r#", "warm": ["#);
    for (i, ef) in EFS.into_iter().enumerate() {
        timed_pass(&coll, &queries, ef); // warm-up
        let (lat, found) = timed_pass(&coll, &queries, ef);
        if i > 0 {
            json.push_str(", ");
        }
        summary(&mut json, ef, &lat, &found, &gt);
        println!(
            "warm ef={ef}: recall@10 {:.4}  p50 {:.0} µs  p99 {:.0} µs",
            recall(&found, &gt),
            pct(&lat, 0.5),
            pct(&lat, 0.99)
        );
    }
    let rss = rss_mb("VmRSS") - rss0;
    write!(json, r#"], "rss_mb": {rss}, "#).unwrap();

    // Throughput: one thread per core, each from its own offset, as bench.py.
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = 600;
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            let (coll, queries) = (&coll, &queries);
            s.spawn(move || {
                for i in 0..per {
                    search_once(coll, queries[(th * 97 + i) % queries.len()], REF_EF);
                }
            });
        }
    });
    let qps = (threads * per) as f64 / t.elapsed().as_secs_f64();
    println!("concurrent ef={REF_EF}, {threads} threads: {qps:.0} QPS");
    write!(
        json,
        r#""qps": {{"threads": {threads}, "knob": {{"ef": {REF_EF}}}, "qps": {qps}}}, "#
    )
    .unwrap();
    drop(coll);
    drop(db);

    // Per-query floor: the same path on a 10-vector collection.
    let tiny = std::env::temp_dir().join("ann-compare-tiny.vdb");
    let _ = std::fs::remove_file(&tiny);
    let floor = {
        let db = Db::open(&tiny)?;
        let metric = if meta.metric == "l2" {
            Metric::L2
        } else {
            Metric::Cosine
        };
        let schema = Schema {
            dim: queries[0].len(),
            metric,
            quant: Quant::Int8,
            chunking: Chunking::None,
        };
        let opts = IndexOptions {
            lexical: false,
            ..IndexOptions::default()
        };
        let coll = db.create_collection_with("bench", schema, opts)?;
        let mut tx = db.begin_write()?;
        for (i, q) in queries[..10].iter().enumerate() {
            tx.upsert_with_vectors(
                &coll,
                Doc::new(i.to_string()).content("q"),
                vec![q.to_vec()],
            )?;
        }
        tx.commit()?;
        db.wait_for_maintenance()?;
        timed_pass(&coll, &queries[..300], REF_EF);
        let (lat, _) = timed_pass(&coll, &queries, REF_EF);
        pct(&lat, 0.5)
    };
    let _ = std::fs::remove_file(&tiny);
    println!("per-query floor (10-vector collection): p50 {floor:.0} µs; resident {rss:.0} MB");
    write!(json, r#""floor_p50_us": {floor}"#).unwrap();
    Ok(json)
}

fn main() -> vecfile::Result<()> {
    let mut args = std::env::args().skip(1);
    let phase = args.next().expect("build | search");
    let (mut data, mut dataset, mut db, mut out) = (PathBuf::from("/data"), None, None, None);
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--data" => data = val().into(),
            "--dataset" => dataset = Some(val()),
            "--db" => db = Some(PathBuf::from(val())),
            "--out" => out = Some(PathBuf::from(val())),
            _ => panic!("unknown argument {a}"),
        }
    }
    let dataset = dataset.expect("--dataset NAME");
    let (db, out) = (db.expect("--db PATH"), out.expect("--out PATH"));
    let meta = read_meta(&data.join("bench").join(&dataset).join("meta.json"));
    let body = match phase.as_str() {
        "build" => {
            let _ = std::fs::remove_file(&db);
            build(&db, &data, &meta)?
        }
        "search" => search(&db, &data, &meta)?,
        _ => panic!("phase must be build or search"),
    };
    let json = format!(
        "{{{body}, \"engine\": \"vecfile\", \"version\": \"{}\", \"dataset\": {}}}\n",
        env!("CARGO_PKG_VERSION"),
        std::fs::read_to_string(data.join("bench").join(&dataset).join("meta.json"))
            .unwrap()
            .trim()
    );
    std::fs::write(out, json).expect("write --out");
    Ok(())
}
