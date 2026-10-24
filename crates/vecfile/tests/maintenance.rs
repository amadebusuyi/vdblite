//! Background maintenance: seals and compactions run on their own thread
//! while writes and searches go on, and the result is the same as if
//! nothing had overlapped.

mod common;

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use common::{DIM, WordEmbedder, article, embed_one};
use vecfile::{Chunking, Collection, Db, Doc, IndexOptions, Metric, Quant, Schema};

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

fn opts(seal_threshold: usize) -> IndexOptions {
    IndexOptions {
        seal_threshold,
        ..IndexOptions::default()
    }
}

/// The index holds exactly the stored chunks, and no deleted document is
/// ever returned.
fn assert_consistent(coll: &Collection, deleted: &HashSet<String>) {
    let stored: usize = coll.scan(None).unwrap().iter().map(|d| d.chunk_count).sum();
    let stats = coll.index_stats();
    assert_eq!(stats.chunks, stored, "{stats:?}");
    for topic in ["rust", "python", "chess", "cooking"] {
        let hits = coll.search(&embed_one(topic)).k(100).chunks().unwrap();
        for h in hits {
            assert!(!deleted.contains(&h.key), "deleted {} returned", h.key);
        }
    }
}

#[test]
fn writes_during_background_maintenance_stay_consistent() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let topics = ["rust", "python", "chess", "cooking"];
    let mut deleted = HashSet::new();
    {
        let db = Db::open(&path).unwrap();
        let coll = db
            .create_collection_with("docs", schema(), opts(200))
            .unwrap();
        let stop = Arc::new(AtomicBool::new(false));
        let searcher = {
            let (coll, stop) = (coll.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut n = 0;
                while !stop.load(Ordering::Relaxed) {
                    coll.search(&embed_one("rust")).k(10).chunks().unwrap();
                    n += 1;
                }
                n
            })
        };
        let e = WordEmbedder::default();
        // Every commit adds documents and deletes some written a little
        // earlier, so deletes keep landing on chunks that a seal or a
        // compaction is building at that moment.
        for round in 0..40usize {
            let mut tx = db.begin_write().unwrap();
            for i in 0..25 {
                let n = round * 25 + i;
                tx.upsert(
                    &coll,
                    Doc::new(format!("doc-{n}")).content(article(topics[n % 4], n)),
                    &e,
                )
                .unwrap();
            }
            if round >= 2 {
                for i in (0..25).filter(|i| i % 2 == 0) {
                    let key = format!("doc-{}", (round - 2) * 25 + i);
                    assert!(tx.delete(&coll, &key).unwrap());
                    deleted.insert(key);
                }
            }
            tx.commit().unwrap();
        }
        stop.store(true, Ordering::Relaxed);
        assert!(searcher.join().unwrap() > 0);
        db.wait_for_maintenance().unwrap();
        let stats = coll.index_stats();
        assert!(stats.sealed_segments >= 2, "{stats:?}");
        assert_consistent(&coll, &deleted);
    }
    // What was persisted (segments, tombstones, buffered chunks) agrees too.
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert_consistent(&coll, &deleted);
    db.optimize().unwrap();
    assert_consistent(&coll, &deleted);
}

#[test]
fn dropping_the_last_handle_closes_the_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let e = WordEmbedder::default();
    for round in 0..8usize {
        let db = Db::open(&path).unwrap();
        let coll = match db.collection("docs") {
            Ok(c) => c,
            Err(_) => db
                .create_collection_with("docs", schema(), opts(30))
                .unwrap(),
        };
        let mut tx = db.begin_write().unwrap();
        for i in 0..40 {
            let n = round * 40 + i;
            tx.upsert(
                &coll,
                Doc::new(format!("doc-{n}")).content(article("chess", n)),
                &e,
            )
            .unwrap();
        }
        tx.commit().unwrap();
        // Maintenance is likely still running; the file must close anyway,
        // here, so it can be reopened at once.
        drop(coll);
        drop(db);
    }
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.len().unwrap(), 8 * 40);
    assert_consistent(&coll, &HashSet::new());
}

#[test]
fn oversized_write_buffers_seal_before_commit_returns() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db
        .create_collection_with("docs", schema(), opts(10))
        .unwrap();
    let e = WordEmbedder::default();
    let mut tx = db.begin_write().unwrap();
    for n in 0..60 {
        tx.upsert(
            &coll,
            Doc::new(format!("doc-{n}")).content(article("rust", n)),
            &e,
        )
        .unwrap();
    }
    tx.commit().unwrap();
    // Far more than a full segment (4 x the seal threshold) was buffered,
    // so the commit sealed it before returning.
    let stats = coll.index_stats();
    assert!(stats.buffered < 40, "{stats:?}");
    assert!(stats.sealed_segments >= 1, "{stats:?}");
    db.wait_for_maintenance().unwrap();
    assert_consistent(&coll, &HashSet::new());
}
