//! Corrupted files must fail cleanly: opening, reading or searching a file
//! with flipped bytes returns errors (or still-valid data), never a panic.

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use common::{WordEmbedder, article, embed_one};
use vecfile::{Db, Doc, IndexOptions, Metric, Schema};
use vecfile_index::rng::SplitMix64;

#[test]
fn flipped_bytes_never_panic() {
    let dir = tempfile::tempdir().unwrap();
    let pristine = dir.path().join("pristine.vdb");
    {
        let db = Db::open(&pristine).unwrap();
        let coll = db
            .create_collection_with(
                "docs",
                Schema::new(common::DIM, Metric::Cosine),
                IndexOptions {
                    seal_threshold: 16,
                    ..IndexOptions::default()
                },
            )
            .unwrap();
        let e = WordEmbedder::default();
        for batch in 0..4 {
            let mut tx = db.begin_write().unwrap();
            for n in 0..8 {
                let i = batch * 8 + n;
                tx.upsert(
                    &coll,
                    Doc::new(format!("d{i}")).content(article("corrupt", i)),
                    &e,
                )
                .unwrap();
            }
            tx.commit().unwrap();
        }
    }
    let bytes = std::fs::read(&pristine).unwrap();
    let target = dir.path().join("broken.vdb");
    let mut rng = SplitMix64::new(42);
    let mut opened = 0;
    let mut panics = Vec::new();
    for round in 0..120 {
        let mut broken = bytes.clone();
        for _ in 0..1 + rng.next_u64() % 8 {
            let i = (rng.next_u64() as usize) % broken.len();
            broken[i] ^= 1 << (rng.next_u64() % 8);
        }
        std::fs::write(&target, &broken).unwrap();
        let result = catch_unwind(AssertUnwindSafe(|| {
            let Ok(db) = Db::open(&target) else {
                return false;
            };
            if let Ok(coll) = db.collection("docs") {
                let _ = coll.search(&embed_one("corrupt detail")).k(5).chunks();
                let _ = coll.get("d3");
                let _ = coll.scan(None);
            }
            true
        }));
        match result {
            Ok(true) => opened += 1,
            Ok(false) => {}
            Err(_) => panics.push(round),
        }
    }
    assert!(panics.is_empty(), "panicked in rounds {panics:?}");
    eprintln!("{opened}/120 corrupted files still opened");
}

/// A corrupted file on which redb panics, then panics again inside a
/// destructor while unwinding, which aborts the process: no `catch_unwind`
/// can contain it. Found by the bit-flip test above (1 in ~400 rounds).
/// Kept as a reproduction for an upstream redb report; run with `--ignored`
/// and expect SIGABRT.
#[test]
#[ignore = "aborts the process (redb double panic on a corrupted file)"]
fn redb_double_panic_reproduction() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("broken.vdb");
    std::fs::copy(
        concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/redb-abort.vdb"),
        &path,
    )
    .unwrap();
    if let Ok(db) = Db::open(&path)
        && let Ok(coll) = db.collection("docs")
    {
        let _ = coll.search(&embed_one("corrupt detail")).k(5).chunks();
        let _ = coll.get("d3");
        let _ = coll.scan(None);
    }
}
