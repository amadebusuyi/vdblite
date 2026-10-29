//! vdblite's side of the embedded comparison (`scripts/compare/embedded`):
//! the same phases, dataset files and JSON as `bench.py` there.
//!
//! ```text
//! ann_compare build  --data DIR --dataset NAME --db PATH --out build.json [--bulk]
//! ann_compare search --data DIR --dataset NAME --db PATH --out search.json
//! ```
//!
//! `NAME` is a directory made by `prep.py` under `DIR/bench`; its
//! `meta.json` names the base file, row count and metric. Vector-only (no
//! keyword index), like the other engines in the comparison.
//!
//! Searches return chunk ids (`Search::chunk_ids`), as the other engines
//! return ids; build writes `PATH.rows` (chunk id to dataset row) so search
//! never reads a document. One extra pass at ef=64 hydrates text and
//! metadata (`Search::chunks`), as an application would.

use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::Instant;

use vdblite::advanced::index::dataset::{read_fvecs, read_ivecs};
use vdblite::{Chunking, Collection, Db, Doc, IndexOptions, Metric, Quant, Schema};

const K: usize = 10;
const BATCH: usize = 10_000;
// vdblite raises ef to at least k x 4 = 40, so the sweep starts there.
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

/// What a search returns: chunk ids, mapped to dataset rows by this table
/// (what the other engines return), or hits hydrated with text and metadata.
#[derive(Clone, Copy)]
enum Shape<'a> {
    Ids(&'a [u32]),
    Hydrated,
}

/// Dataset rows of the `K` nearest neighbours.
fn search_once(coll: &Collection, q: &[f32], ef: usize, shape: Shape<'_>) -> Vec<u32> {
    let s = coll.search(q).k(K).ef(ef);
    match shape {
        Shape::Ids(rows) => s
            .chunk_ids()
            .unwrap()
            .into_iter()
            .map(|(id, _)| rows[id as usize])
            .collect(),
        Shape::Hydrated => s
            .chunks()
            .unwrap()
            .into_iter()
            .map(|h| h.key.parse().unwrap())
            .collect(),
    }
}

/// One timed pass over the queries: sorted latencies (µs) and the rows found.
fn timed_pass(
    coll: &Collection,
    queries: &[&[f32]],
    ef: usize,
    shape: Shape<'_>,
) -> (Vec<f64>, Vec<Vec<u32>>) {
    let mut lat = Vec::with_capacity(queries.len());
    let mut found = Vec::with_capacity(queries.len());
    for q in queries {
        let t = Instant::now();
        found.push(search_once(coll, q, ef, shape));
        lat.push(t.elapsed().as_secs_f64() * 1e6);
    }
    lat.sort_by(f64::total_cmp);
    (lat, found)
}

/// Throughput: one thread per core, each from its own offset, as bench.py.
fn concurrent_qps(coll: &Collection, queries: &[&[f32]], shape: Shape<'_>) -> (usize, f64) {
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    let per = 600;
    let t = Instant::now();
    std::thread::scope(|s| {
        for th in 0..threads {
            s.spawn(move || {
                for i in 0..per {
                    search_once(coll, queries[(th * 97 + i) % queries.len()], REF_EF, shape);
                }
            });
        }
    });
    (threads, (threads * per) as f64 / t.elapsed().as_secs_f64())
}

/// Chunk id to dataset row for documents keyed `0..n`, one chunk each.
fn row_map(coll: &Collection, n: usize) -> vdblite::Result<Vec<u32>> {
    let mut rows = Vec::new();
    for row in 0..n {
        let chunks = coll.chunks(&row.to_string())?;
        assert_eq!(chunks.len(), 1, "document {row} should have one chunk");
        let id = chunks[0].chunk_id as usize;
        if rows.len() <= id {
            rows.resize(id + 1, u32::MAX);
        }
        rows[id] = row as u32;
    }
    Ok(rows)
}

fn rows_path(db: &Path) -> PathBuf {
    db.with_extension("rows")
}

/// Appends one operating point as a JSON object; `extra` adds fields to it.
fn summary(
    json: &mut String,
    knob: usize,
    lat: &[f64],
    found: &[Vec<u32>],
    gt: &[&[i32]],
    extra: &str,
) {
    let mean = lat.iter().sum::<f64>() / lat.len() as f64;
    write!(
        json,
        r#"{{"knob": {{"ef": {knob}}}, "recall": {}, "p50_us": {}, "p99_us": {}, "mean_us": {mean}{extra}}}"#,
        recall(found, gt),
        pct(lat, 0.5),
        pct(lat, 0.99)
    )
    .unwrap();
}

/// `bulk`: load through [`Db::bulk_load`] (indexed once at the end) instead
/// of ordinary transactions sealed and merged as they go.
fn build(db_path: &Path, data: &Path, meta: &Meta, bulk: bool) -> vdblite::Result<String> {
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
    // A document needs content to have a chunk; keep it tiny, as sift_db does.
    let doc = |i: usize| {
        (
            Doc::new(i.to_string()).content(format!("v{i}")),
            vec![base.row(i).to_vec()],
        )
    };
    let mut load = if bulk {
        Some(db.bulk_load(&coll)?)
    } else {
        None
    };
    for start in (0..base.len()).step_by(BATCH) {
        let rows = start..(start + BATCH).min(base.len());
        match &mut load {
            Some(load) => {
                load.add(rows.map(doc))?;
            }
            None => {
                let mut tx = db.begin_write()?;
                for (d, v) in rows.map(doc) {
                    tx.upsert_with_vectors(&coll, d, v)?;
                }
                tx.commit()?;
            }
        }
    }
    let loaded = t.elapsed().as_secs_f64();
    if let Some(load) = load {
        load.finish()?;
    }
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

    // Not timed: the chunk id of every row, for id-only searches.
    let db = Db::open(db_path)?;
    let rows = row_map(&db.collection("bench")?, base.len())?;
    let bytes: Vec<u8> = rows.iter().flat_map(|r| r.to_le_bytes()).collect();
    std::fs::write(rows_path(db_path), bytes).expect("write the row map");
    drop(db);
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

fn search(db_path: &Path, data: &Path, meta: &Meta) -> vdblite::Result<String> {
    let dir = data.join("bench").join(&meta.name);
    let queries = read_fvecs(dir.join("query.fvecs"), None).expect("read queries");
    let gt = read_ivecs(dir.join("gt.ivecs"), None).expect("read ground truth");
    let queries: Vec<&[f32]> = queries.rows().collect();
    let gt: Vec<&[i32]> = gt.rows().collect();
    let rows: Vec<u32> = std::fs::read(rows_path(db_path))
        .expect("read the row map (run build first)")
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let ids = Shape::Ids(&rows);

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
    search_once(&coll, queries[0], REF_EF, ids);
    let first = t.elapsed().as_secs_f64() * 1e6;
    let (lat, found) = timed_pass(&coll, &queries, REF_EF, ids);
    write!(json, r#""first_query_us": {first}, "cold": "#).unwrap();
    summary(&mut json, REF_EF, &lat, &found, &gt, "");
    println!(
        "open {:.0} ms, first query {:.1} ms, cold pass p50 {:.0} µs p99 {:.0} µs",
        open_s * 1e3,
        first / 1e3,
        pct(&lat, 0.5),
        pct(&lat, 0.99)
    );

    json.push_str(r#", "warm": ["#);
    for (i, ef) in EFS.into_iter().enumerate() {
        timed_pass(&coll, &queries, ef, ids); // warm-up
        let (lat, found) = timed_pass(&coll, &queries, ef, ids);
        if i > 0 {
            json.push_str(", ");
        }
        summary(&mut json, ef, &lat, &found, &gt, "");
        println!(
            "warm ef={ef}: recall@10 {:.4}  p50 {:.0} µs  p99 {:.0} µs",
            recall(&found, &gt),
            pct(&lat, 0.5),
            pct(&lat, 0.99)
        );
    }
    let rss = rss_mb("VmRSS") - rss0;
    write!(json, r#"], "rss_mb": {rss}, "#).unwrap();

    let (threads, qps) = concurrent_qps(&coll, &queries, ids);
    println!("concurrent ef={REF_EF}, {threads} threads: {qps:.0} QPS");
    write!(
        json,
        r#""qps": {{"threads": {threads}, "knob": {{"ef": {REF_EF}}}, "qps": {qps}}}, "#
    )
    .unwrap();

    // The same searches, reading each hit's text and metadata from the file.
    timed_pass(&coll, &queries, REF_EF, Shape::Hydrated); // warm-up
    let (lat, found) = timed_pass(&coll, &queries, REF_EF, Shape::Hydrated);
    let (_, hydrated_qps) = concurrent_qps(&coll, &queries, Shape::Hydrated);
    json.push_str(r#""hydrated": "#);
    let extra = format!(r#", "qps": {hydrated_qps}"#);
    summary(&mut json, REF_EF, &lat, &found, &gt, &extra);
    json.push_str(", ");
    println!(
        "hydrated ef={REF_EF}: recall@10 {:.4}  p50 {:.0} µs  p99 {:.0} µs; {threads} threads: {hydrated_qps:.0} QPS",
        recall(&found, &gt),
        pct(&lat, 0.5),
        pct(&lat, 0.99)
    );
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
        let rows = row_map(&coll, 10)?;
        timed_pass(&coll, &queries[..300], REF_EF, Shape::Ids(&rows));
        let (lat, _) = timed_pass(&coll, &queries, REF_EF, Shape::Ids(&rows));
        pct(&lat, 0.5)
    };
    let _ = std::fs::remove_file(&tiny);
    println!("per-query floor (10-vector collection): p50 {floor:.0} µs; resident {rss:.0} MB");
    write!(json, r#""floor_p50_us": {floor}"#).unwrap();
    Ok(json)
}

fn main() -> vdblite::Result<()> {
    let mut args = std::env::args().skip(1);
    let phase = args.next().expect("build | search");
    let (mut data, mut dataset, mut db, mut out) = (PathBuf::from("/data"), None, None, None);
    let mut bulk = false;
    while let Some(a) = args.next() {
        let mut val = || args.next().unwrap_or_else(|| panic!("{a} needs a value"));
        match a.as_str() {
            "--data" => data = val().into(),
            "--dataset" => dataset = Some(val()),
            "--db" => db = Some(PathBuf::from(val())),
            "--out" => out = Some(PathBuf::from(val())),
            "--bulk" => bulk = true,
            _ => panic!("unknown argument {a}"),
        }
    }
    let dataset = dataset.expect("--dataset NAME");
    let (db, out) = (db.expect("--db PATH"), out.expect("--out PATH"));
    let meta = read_meta(&data.join("bench").join(&dataset).join("meta.json"));
    let body = match phase.as_str() {
        "build" => {
            let _ = std::fs::remove_file(&db);
            let _ = std::fs::remove_file(rows_path(&db));
            build(&db, &data, &meta, bulk)?
        }
        "search" => search(&db, &data, &meta)?,
        _ => panic!("phase must be build or search"),
    };
    let json = format!(
        "{{{body}, \"engine\": \"vdblite\", \"version\": \"{}\", \"dataset\": {}}}\n",
        env!("CARGO_PKG_VERSION"),
        std::fs::read_to_string(data.join("bench").join(&dataset).join("meta.json"))
            .unwrap()
            .trim()
    );
    std::fs::write(out, json).expect("write --out");
    Ok(())
}
