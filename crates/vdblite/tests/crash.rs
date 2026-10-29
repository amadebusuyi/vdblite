//! Crash safety: a writer process is killed (SIGKILL) at arbitrary points,
//! including mid-commit and mid-seal. Every write it reported as committed
//! must survive, and the reopened file must be internally consistent.
//!
//! The test binary re-runs itself as the child, selected by an env var.

mod common;

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::time::Duration;

use common::{DIM, WordEmbedder, article, embed_one};
use vdblite::{Chunking, Db, Doc, IndexOptions, Metric, Quant, Schema};

const CHILD_DB: &str = "VDBLITE_CRASH_DB";
const CHILD_START: &str = "VDBLITE_CRASH_START";

/// Doc `i` is written at iteration `i`; iteration `i` with `i % 4 == 0` also
/// deletes doc `i - 3`.
fn deleted_by(j: usize) -> usize {
    j + 3
}

fn is_deleted(j: usize, last_committed: usize) -> bool {
    let i = deleted_by(j);
    i <= last_committed && i.is_multiple_of(4)
}

#[test]
fn crash_child() {
    let Ok(path) = std::env::var(CHILD_DB) else {
        return; // Not running as the child.
    };
    let start: usize = std::env::var(CHILD_START).unwrap().parse().unwrap();
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    let e = WordEmbedder::default();
    let mut out = std::io::stdout();
    for i in start.. {
        let mut tx = db.begin_write().unwrap();
        tx.upsert(
            &coll,
            Doc::new(format!("doc-{i}"))
                .content(article("crash", i))
                .meta("i", i as i64),
            &e,
        )
        .unwrap();
        if i % 4 == 0 && i >= 3 {
            tx.delete(&coll, &format!("doc-{}", i - 3)).unwrap();
        }
        match tx.commit() {
            Ok(()) | Err(vdblite::Error::Maintenance(_)) => {}
            Err(e) => panic!("commit failed: {e}"),
        }
        writeln!(out, "committed {i}").unwrap();
        out.flush().unwrap();
    }
}

#[test]
fn killed_writers_lose_nothing_committed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("crash.vdb");
    {
        let db = Db::open(&path).unwrap();
        let schema = Schema {
            dim: DIM,
            metric: Metric::Cosine,
            quant: Quant::Int8,
            chunking: Chunking::Structural {
                target_tokens: 20,
                overlap_tokens: 4,
            },
        };
        db.create_collection_with(
            "docs",
            schema,
            IndexOptions {
                seal_threshold: 24,
                ..IndexOptions::default()
            },
        )
        .unwrap();
    }

    let exe = std::env::current_exe().unwrap();
    let mut next = 0usize;
    let mut kills_mid_run = 0;
    for round in 0..10u64 {
        let mut child = Command::new(&exe)
            .args(["--exact", "crash_child", "--nocapture", "--test-threads=1"])
            .env(CHILD_DB, &path)
            .env(CHILD_START, next.to_string())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let last = Arc::new(AtomicI64::new(-1));
        let stdout = child.stdout.take().unwrap();
        let reader = {
            let last = last.clone();
            std::thread::spawn(move || {
                for line in BufReader::new(stdout).lines() {
                    let Ok(line) = line else { break };
                    if let Some(n) = line.strip_prefix("committed ") {
                        last.store(n.parse().unwrap(), Ordering::SeqCst);
                    }
                }
            })
        };
        // Vary the kill point so it lands in different phases of a commit.
        // The child needs ~150 ms to start, open and make its first commit
        // (two disk flushes, so reading processes stay safe).
        let wait = 300 + (round * 7919 % 600);
        std::thread::sleep(Duration::from_millis(wait));
        if child.try_wait().unwrap().is_none() {
            kills_mid_run += 1;
        }
        child.kill().unwrap();
        child.wait().unwrap();
        reader.join().unwrap();

        let last = last.load(Ordering::SeqCst);
        assert!(
            last >= 0,
            "round {round}: child committed nothing in {wait} ms"
        );
        let last = last as usize;

        let db = Db::open(&path).unwrap();
        let coll = db.collection("docs").unwrap();
        for j in 0..=last {
            let doc = coll.get(&format!("doc-{j}")).unwrap();
            let deleter = deleted_by(j);
            if deleter > last && deleter.is_multiple_of(4) {
                // Its delete may have committed after the last line we saw.
                continue;
            }
            if is_deleted(j, last) {
                assert!(
                    doc.is_none(),
                    "round {round}: doc-{j} should have been deleted"
                );
            } else {
                let doc = doc.unwrap_or_else(|| {
                    panic!("round {round}: committed doc-{j} is missing (last {last})")
                });
                assert_eq!(doc.content, article("crash", j));
            }
        }
        // The index and the records agree: every stored chunk is searchable
        // and nothing else is.
        let stored: usize = coll.scan(None).unwrap().iter().map(|d| d.chunk_count).sum();
        assert_eq!(
            coll.index_stats().chunks,
            stored,
            "round {round}: index and records disagree"
        );
        let hits = coll
            .search(&embed_one(&article("crash", last)))
            .k(50)
            .chunks()
            .unwrap();
        assert!(
            hits.iter().any(|h| h.key == format!("doc-{last}")),
            "round {round}: doc-{last} not searchable"
        );
        next = last + 1;
    }
    assert!(
        kills_mid_run >= 8,
        "the child should be killed while writing"
    );
}
