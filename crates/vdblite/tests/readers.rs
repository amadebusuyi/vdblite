//! Reading processes: a file opened read-only alongside its writer follows
//! every commit, seal and compaction, in the same process or another one.
//!
//! The cross-process test re-runs this binary as the writer, selected by an
//! env var (as `crash.rs` does).

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use common::{DIM, WordEmbedder, article, embed_one};
use vdblite::{Chunking, Collection, Db, Doc, Error, IndexOptions, Metric, Quant, Schema};

fn schema() -> Schema {
    Schema {
        dim: DIM,
        metric: Metric::Cosine,
        quant: Quant::Int8,
        chunking: Chunking::Structural {
            target_tokens: 20,
            overlap_tokens: 4,
        },
    }
}

fn small_segments() -> IndexOptions {
    IndexOptions {
        seal_threshold: 48,
        ..IndexOptions::default()
    }
}

const TOPICS: [&str; 6] = [
    "rust",
    "python",
    "gardening",
    "astronomy",
    "cooking",
    "chess",
];

fn write_docs(db: &Db, coll: &Collection, range: std::ops::Range<usize>) {
    let e = WordEmbedder::default();
    let mut tx = db.begin_write().unwrap();
    for n in range {
        let topic = TOPICS[n % TOPICS.len()];
        tx.upsert(
            coll,
            Doc::new(format!("doc-{n}"))
                .content(article(topic, n))
                .meta("topic", topic)
                .meta("n", n as i64),
            &e,
        )
        .unwrap();
    }
    tx.commit().unwrap();
}

fn delete_docs(db: &Db, coll: &Collection, keys: impl Iterator<Item = usize>) {
    let mut tx = db.begin_write().unwrap();
    for n in keys {
        tx.delete(coll, &format!("doc-{n}")).unwrap();
    }
    tx.commit().unwrap();
}

/// The reader's index holds exactly what the writer's does.
fn assert_in_step(writer: &Collection, reader: &Collection) {
    let (w, r) = (writer.index_stats(), reader.index_stats());
    assert_eq!(
        (w.chunks, w.sealed_segments, w.buffered),
        (r.chunks, r.sealed_segments, r.buffered),
        "writer {w:?}, reader {r:?}"
    );
    let stored: usize = reader
        .scan(None)
        .unwrap()
        .iter()
        .map(|d| d.chunk_count)
        .sum();
    assert_eq!(r.chunks, stored, "the reader's index matches the records");
    for topic in TOPICS {
        let q = embed_one(&format!("{topic} detail"));
        let key = |c: &Collection| -> Vec<String> {
            c.search(&q)
                .k(3)
                .chunks()
                .unwrap()
                .into_iter()
                .map(|h| h.key)
                .collect()
        };
        assert_eq!(key(writer)[0], key(reader)[0], "top hit for {topic}");
    }
}

#[test]
fn readers_follow_the_writer() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let w = Db::open(&path).unwrap();
    let wc = w
        .create_collection_with("docs", schema(), small_segments())
        .unwrap();
    let r = Db::open_read_only(&path).unwrap();
    assert!(r.is_read_only() && !w.is_read_only());
    let rc = r.collection("docs").unwrap();

    // Writes through the reader are refused.
    assert!(matches!(r.begin_write(), Err(Error::ReadOnly)));
    assert!(matches!(
        r.create_collection("x", schema()),
        Err(Error::ReadOnly)
    ));
    assert!(matches!(r.drop_collection("docs"), Err(Error::ReadOnly)));
    assert!(matches!(r.optimize(), Err(Error::ReadOnly)));
    assert!(matches!(rc.enable_lexical(), Err(Error::ReadOnly)));

    // Each commit is visible as soon as it lands, to reads and searches.
    for round in 0..6 {
        let range = round * 10..round * 10 + 10;
        write_docs(&w, &wc, range.clone());
        let last = format!("doc-{}", range.end - 1);
        assert!(rc.get(&last).unwrap().is_some());
        let hits = rc
            .search(&embed_one(&article(
                TOPICS[(range.end - 1) % 6],
                range.end - 1,
            )))
            .k(20)
            .chunks()
            .unwrap();
        assert!(hits.iter().any(|h| h.key == last), "round {round}");
        assert_in_step(&wc, &rc);
    }

    // Seals the writer makes in the background reach the reader.
    w.wait_for_maintenance().unwrap();
    // (How many remain depends on how background merges were timed.)
    assert!(wc.index_stats().sealed_segments >= 1);
    assert_in_step(&wc, &rc);

    // Deletes, and the compactions they trigger.
    delete_docs(&w, &wc, (0..60).filter(|n| n % 3 != 0));
    w.wait_for_maintenance().unwrap();
    assert_in_step(&wc, &rc);
    assert!(rc.get("doc-1").unwrap().is_none());
    let hits = rc.search(&embed_one("python")).k(50).chunks().unwrap();
    assert!(hits.iter().all(|h| h.key != "doc-1"));

    // Merges.
    w.optimize().unwrap();
    assert_eq!(wc.index_stats().buffered, 0);
    assert_in_step(&wc, &rc);

    // Hybrid search reads the keyword index from the file.
    let hits = rc
        .search(&embed_one("anything"))
        .text("astronomy")
        .k(3)
        .chunks()
        .unwrap();
    assert!(hits.iter().any(|h| h.lexical_score.is_some()));

    // Collections created and dropped by the writer.
    let other = w.create_collection("other", schema()).unwrap();
    write_docs(&w, &other, 0..3);
    assert_eq!(r.collections().unwrap(), ["docs", "other"]);
    assert_eq!(r.collection("other").unwrap().len().unwrap(), 3);
    w.drop_collection("other").unwrap();
    assert_eq!(r.collections().unwrap(), ["docs"]);
    assert!(matches!(
        r.collection("other"),
        Err(Error::NoSuchCollection(_))
    ));

    // A second reader joins late and starts from the current state.
    let r2 = Db::open_read_only(&path).unwrap();
    assert_in_step(&wc, &r2.collection("docs").unwrap());
}

#[test]
fn readers_reload_when_they_fall_behind() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let w = Db::open(&path).unwrap();
    let wc = w
        .create_collection_with("docs", schema(), small_segments())
        .unwrap();
    write_docs(&w, &wc, 0..20);
    w.wait_for_maintenance().unwrap();
    let r = Db::open_read_only(&path).unwrap();
    let rc = r.collection("docs").unwrap();
    assert_in_step(&wc, &rc);

    // While the reader looks away, a segment is sealed and then rebuilt
    // (removed) before the reader ever loads it: replaying the log can't
    // install it, so the reader reloads.
    let before = wc.index_stats().sealed_segments;
    write_docs(&w, &wc, 20..60);
    w.wait_for_maintenance().unwrap();
    assert!(wc.index_stats().sealed_segments > before);
    delete_docs(&w, &wc, 20..50);
    w.wait_for_maintenance().unwrap();

    assert_in_step(&wc, &rc);
    assert!(rc.get("doc-30").unwrap().is_none());
    let hits = rc.search(&embed_one("rust")).k(50).chunks().unwrap();
    assert!(hits.iter().all(
        |h| h.key.trim_start_matches("doc-").parse::<usize>().unwrap() < 20
            || h.key.trim_start_matches("doc-").parse::<usize>().unwrap() >= 50
    ));
}

#[test]
fn exclusive_writers_shut_readers_out() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let w = Db::open_exclusive(&path).unwrap();
        let c = w.create_collection("docs", schema()).unwrap();
        write_docs(&w, &c, 0..3);
        assert!(Db::open_read_only(&path).is_err());
        assert!(Db::open(&path).is_err());
    }
    // Shared files and exclusive ones are the same format.
    let r = Db::open_read_only(&path).unwrap();
    assert_eq!(r.collection("docs").unwrap().len().unwrap(), 3);
    assert!(Db::open_exclusive(&path).is_err(), "a reader is still open");
    let w = Db::open(&path).unwrap();
    assert!(Db::open(&path).is_err(), "one writer at a time");
    drop(w);
}

/// The vector the fixture gave `doc-i` (see `tests/data/README.md`).
fn fixture_vector(i: usize) -> Vec<f32> {
    (0..8)
        .map(|d| {
            let x = ((i * 8 + d) as f64 * 12.9898).sin() * 43758.5453;
            (x - x.floor() - 0.5) as f32
        })
        .collect()
}

#[test]
fn phase_7_files_are_upgraded() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("old.vdb");
    std::fs::copy("tests/data/v2-phase7.vdb", &path).unwrap();
    // A reader can't use it until a writer has upgraded it.
    assert!(matches!(
        Db::open_read_only(&path),
        Err(vdblite::Error::Version { found: 2, .. })
    ));
    let live: Vec<usize> = (0..320).filter(|i| *i >= 300 || i % 7 != 0).collect();
    let check = |db: &Db| {
        let c = db.collection("docs").unwrap();
        assert_eq!(c.len().unwrap(), live.len());
        let stats = c.index_stats();
        assert_eq!(
            (stats.chunks, stats.sealed_segments, stats.buffered),
            (277, 3, 20)
        );
        assert!(c.get("doc-7").unwrap().is_none(), "deletes survive");
        for &i in &[1usize, 150, 299, 300, 319] {
            let hits = c.search(&fixture_vector(i)).k(1).chunks().unwrap();
            assert_eq!(hits[0].key, format!("doc-{i}"));
            assert_eq!(hits[0].text, format!("document number {i}"));
        }
        let deleted = c.search(&fixture_vector(14)).k(5).keys().unwrap();
        assert!(deleted.iter().all(|(k, _)| k != "doc-14"));
        let filtered = c
            .search(&fixture_vector(10))
            .k(3)
            .filter(vdblite::meta("n").gte(300.0))
            .keys()
            .unwrap();
        assert!(filtered.iter().all(|(k, _)| k.as_str() >= "doc-300"));
    };
    {
        let w = Db::open(&path).unwrap();
        check(&w);
        let r = Db::open_read_only(&path).unwrap();
        check(&r);
        // The upgraded file keeps working: seal the buffer, merge, write more.
        let c = w.collection("docs").unwrap();
        let mut tx = w.begin_write().unwrap();
        for i in 320..400 {
            tx.upsert_with_vectors(
                &c,
                vdblite::Doc::new(format!("doc-{i}")).content(format!("document number {i}")),
                vec![fixture_vector(i)],
            )
            .unwrap();
        }
        tx.commit().unwrap();
        w.optimize().unwrap();
        w.wait_for_maintenance().unwrap();
        let rc = r.collection("docs").unwrap();
        assert_eq!(rc.len().unwrap(), live.len() + 80);
        let hit = rc.search(&fixture_vector(350)).k(1).keys().unwrap();
        assert_eq!(hit[0].0, "doc-350");
    }
    // And reopens.
    let w = Db::open(&path).unwrap();
    assert_eq!(
        w.collection("docs").unwrap().len().unwrap(),
        live.len() + 80
    );
}

const CHILD_DB: &str = "VDBLITE_READERS_WRITER_DB";

/// The writer process for `readers_in_other_processes`: writes `doc-i` for
/// i = 0, 1, … and reports each commit on stdout.
#[test]
fn writer_child() {
    let Ok(path) = std::env::var(CHILD_DB) else {
        return; // Not running as the child.
    };
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    let mut out = std::io::stdout();
    // Runs until the parent kills it.
    let mut i = 0;
    loop {
        write_docs(&db, &coll, i..i + 1);
        writeln!(out, "committed {i}").unwrap();
        out.flush().unwrap();
        i += 1;
    }
}

#[test]
fn readers_in_other_processes() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let db = Db::open(&path).unwrap();
        db.create_collection_with("docs", schema(), small_segments())
            .unwrap();
    }
    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(&exe)
        .args(["--exact", "writer_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_DB, &path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    let mut next_commit = || -> usize {
        loop {
            let line = lines.next().expect("the writer exited").unwrap();
            if let Some(n) = line.strip_prefix("committed ") {
                return n.parse().unwrap();
            }
        }
    };

    // Open while the other process is writing, and follow it.
    next_commit();
    let r = Db::open_read_only(&path).unwrap();
    let rc = r.collection("docs").unwrap();
    let start = Instant::now();
    let mut seen = 0;
    while seen < 60 && start.elapsed() < Duration::from_secs(60) {
        let i = next_commit();
        // Acknowledged by the writer, so it must be visible.
        assert!(rc.get(&format!("doc-{i}")).unwrap().is_some(), "doc-{i}");
        let hits = rc
            .search(&embed_one(&article(TOPICS[i % 6], i)))
            .k(20)
            .chunks()
            .unwrap();
        assert!(hits.iter().any(|h| h.key == format!("doc-{i}")), "doc-{i}");
        seen += 1;
    }
    assert_eq!(seen, 60);
    assert!(rc.index_stats().sealed_segments >= 1, "seals were followed");

    // The writer dies mid-write; the reader carries on with what was
    // committed.
    child.kill().unwrap();
    child.wait().unwrap();
    let mut last = 0;
    while let Some(Ok(line)) = lines.next() {
        if let Some(n) = line.strip_prefix("committed ") {
            last = n.parse().unwrap();
        }
    }
    let len = rc.len().unwrap();
    assert!(len > last, "{len} docs, last commit doc-{last}");
    let stored: usize = rc.scan(None).unwrap().iter().map(|d| d.chunk_count).sum();
    assert_eq!(rc.index_stats().chunks, stored);

    // A new writer recovers the file while the reader stays open, and the
    // reader follows it too.
    let w = Db::open(&path).unwrap();
    let wc = w.collection("docs").unwrap();
    write_docs(&w, &wc, 1000..1003);
    assert!(rc.get("doc-1002").unwrap().is_some());
    w.wait_for_maintenance().unwrap();
    assert_in_step(&wc, &rc);
}

const REGION_CHILD_DB: &str = "VDBLITE_READERS_REGION_DB";

/// The reading process for `regions_in_use_elsewhere_are_never_reused`:
/// maps every segment the way a reader does (leasing an epoch first), keeps
/// a copy of their vectors, and on request checks the mapped bytes still
/// match. It never syncs, so it keeps the old segments for its whole life.
#[test]
fn region_reader_child() {
    use redb::{ReadableDatabase, ReadableTable};
    use vdblite_core::container::Container;
    use vdblite_core::records::{SegmentRecord, decode};
    use vdblite_core::segments::map_segment;
    use vdblite_core::tables::SEGMENTS;
    let Ok(path) = std::env::var(REGION_CHILD_DB) else {
        return; // Not running as the child.
    };
    let c = Container::open(std::path::Path::new(&path), false).unwrap();
    let _lease = c.lease(0).unwrap();
    let mut b = redb::Builder::new();
    b.set_concurrency_mode(redb::ConcurrencyMode::SingleWriter);
    let store = b.open_read_only_with_backend(c.redb_backend()).unwrap();
    let rx = store.begin_read().unwrap();
    let mut held = Vec::new();
    for row in rx.open_table(SEGMENTS).unwrap().iter().unwrap() {
        let rec: SegmentRecord = decode(row.unwrap().1.value(), "segment").unwrap();
        let seg = map_segment(&c, &rec, 0).unwrap();
        let copy = seg.vectors().bytes().to_vec();
        held.push((seg, copy));
    }
    drop(rx);
    let mut out = std::io::stdout();
    writeln!(out, "ready {}", held.len()).unwrap();
    out.flush().unwrap();
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        if line.unwrap() == "check" {
            let same = held
                .iter()
                .all(|(seg, copy)| seg.vectors().bytes() == &copy[..]);
            writeln!(out, "{}", if same { "same" } else { "changed" }).unwrap();
            out.flush().unwrap();
        }
    }
}

#[test]
fn regions_in_use_elsewhere_are_never_reused() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let w = Db::open(&path).unwrap();
    let wc = w
        .create_collection_with("docs", schema(), small_segments())
        .unwrap();
    write_docs(&w, &wc, 0..60);
    w.optimize().unwrap();
    w.wait_for_maintenance().unwrap();

    let exe = std::env::current_exe().unwrap();
    let mut child = Command::new(&exe)
        .args([
            "--exact",
            "region_reader_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(REGION_CHILD_DB, &path)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    let mut lines = BufReader::new(child.stdout.take().unwrap()).lines();
    // libtest may print `test region_reader_child ... ` without a newline
    // just before the child's first line, so match anywhere in a line.
    let mut next = || loop {
        let line = lines.next().expect("the reader exited").unwrap();
        for word in ["ready", "same", "changed"] {
            if line.contains(word) {
                return word;
            }
        }
    };
    assert_eq!(next(), "ready");

    // Churn: everything rewritten and merged, so the regions the reader
    // maps are freed, and new ones are needed.
    let churn = |w: &Db| {
        delete_docs(w, &wc, 0..60);
        write_docs(w, &wc, 0..60);
        w.optimize().unwrap();
        w.wait_for_maintenance().unwrap();
    };
    churn(&w);
    churn(&w);
    writeln!(stdin, "check").unwrap();
    stdin.flush().unwrap();
    assert_eq!(next(), "same", "a region another process maps was reused");
    let grown = std::fs::metadata(&path).unwrap().len();

    // Once the reader is gone, freed space is reused instead of growing
    // the file.
    drop(stdin);
    child.wait().unwrap();
    churn(&w);
    churn(&w);
    let after = std::fs::metadata(&path).unwrap().len();
    assert!(
        after <= grown + grown / 4,
        "the file grew from {grown} to {after} bytes with nothing holding its free space"
    );
}
