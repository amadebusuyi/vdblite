mod common;

use std::collections::HashSet;

use common::{DIM, WordEmbedder, article, embed_one};
use vdblite::{
    Agg, Chunking, Db, Doc, Error, IndexOptions, MetaValue, Metric, Quant, Schema, meta,
};

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
        seal_threshold: 64,
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

fn fill(db: &Db, name: &str, docs: usize, opts: IndexOptions) -> vdblite::Collection {
    let coll = db.create_collection_with(name, schema(), opts).unwrap();
    let e = WordEmbedder::default();
    let mut tx = db.begin_write().unwrap();
    for n in 0..docs {
        // Commit in batches and let each one's seal finish, to get several
        // segments.
        if n > 0 && n % 10 == 0 {
            tx.commit().unwrap();
            db.wait_for_maintenance().unwrap();
            tx = db.begin_write().unwrap();
        }
        let topic = TOPICS[n % TOPICS.len()];
        let doc = Doc::new(format!("doc-{n}"))
            .content(article(topic, n))
            .meta("topic", topic)
            .meta("n", n as i64)
            .meta("tags", vec![topic, if n % 2 == 0 { "even" } else { "odd" }]);
        tx.upsert(&coll, doc, &e).unwrap();
    }
    tx.commit().unwrap();
    db.wait_for_maintenance().unwrap();
    coll
}

#[test]
fn write_search_and_reopen() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let db = Db::open(&path).unwrap();
        let coll = fill(&db, "docs", 30, small_segments());
        let stats = coll.index_stats();
        assert!(stats.sealed_segments >= 1, "{stats:?}");
        let hits = coll
            .search(&embed_one("astronomy detail"))
            .k(5)
            .chunks()
            .unwrap();
        assert_eq!(hits.len(), 5);
        assert!(
            hits.iter().all(|h| h.text.contains("astronomy")),
            "{hits:#?}"
        );
        assert!(hits[0].score >= hits[4].score);
    }
    let db = Db::open(&path).unwrap();
    assert_eq!(db.collections().unwrap(), ["docs"]);
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.len().unwrap(), 30);
    let hits = coll
        .search(&embed_one("chess detail"))
        .k(5)
        .chunks()
        .unwrap();
    assert!(hits.iter().all(|h| h.text.contains("chess")));
    let doc = coll.get("doc-3").unwrap().unwrap();
    assert_eq!(doc.content, article("astronomy", 3));
    assert_eq!(doc.meta["topic"], MetaValue::Str("astronomy".into()));
    assert_eq!(doc.version, 1);
    assert_eq!(doc.format, "markdown");
}

#[test]
fn chunks_rebuild_the_document_exactly() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = fill(&db, "docs", 3, IndexOptions::default());
    let doc = coll.get("doc-1").unwrap().unwrap();
    let chunks = coll.chunks("doc-1").unwrap();
    assert!(chunks.len() > 2);
    let rebuilt: String = chunks
        .iter()
        .map(|c| &doc.content[c.range.clone()])
        .collect();
    assert_eq!(rebuilt, doc.content);
    assert_eq!(chunks[1].heading_path, ["python guide", "Part 0"]);
}

#[test]
fn small_edits_reembed_only_changed_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db.create_collection("docs", schema()).unwrap();
    let e = WordEmbedder::default();
    let text = article("rust", 1);

    let mut tx = db.begin_write().unwrap();
    let first = tx
        .upsert(&coll, Doc::new("a").content(text.clone()), &e)
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(first.embedded, first.chunks);

    // Identical content: nothing to do.
    let mut tx = db.begin_write().unwrap();
    let same = tx
        .upsert(&coll, Doc::new("a").content(text.clone()), &e)
        .unwrap();
    tx.commit().unwrap();
    assert!(!same.changed);
    assert_eq!(same.version, 1);

    // Edit the last section only.
    let edited = text.replace("detail 2 for article 1", "detail 2 for article 1, revised");
    let before = e.texts_embedded();
    let mut tx = db.begin_write().unwrap();
    let up = tx
        .upsert(&coll, Doc::new("a").content(edited.clone()), &e)
        .unwrap();
    tx.commit().unwrap();
    assert!(up.changed);
    assert_eq!(up.version, 2);
    assert!(up.embedded >= 1 && up.embedded < up.chunks, "{up:?}");
    assert_eq!(e.texts_embedded() - before, up.embedded);

    let hits = coll.search(&embed_one("revised")).k(1).chunks().unwrap();
    assert!(hits[0].text.contains("revised"));
    assert_eq!(coll.get("a").unwrap().unwrap().content, edited);

    // Metadata-only change keeps every vector.
    let before = e.texts_embedded();
    let mut tx = db.begin_write().unwrap();
    let m = tx
        .upsert(&coll, Doc::new("a").content(edited).meta("lang", "en"), &e)
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(m.embedded, 0);
    assert_eq!(e.texts_embedded(), before);
    assert_eq!(
        coll.search(&embed_one("rust"))
            .filter(meta("lang").eq("en"))
            .chunks()
            .unwrap()
            .len(),
        m.chunks
    );
}

#[test]
fn deletes_disappear_everywhere_and_persist() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let db = Db::open(&path).unwrap();
        let coll = fill(&db, "docs", 40, small_segments());
        assert!(coll.index_stats().sealed_segments >= 1);
        let mut tx = db.begin_write().unwrap();
        for n in (0..40).filter(|n| n % 6 == 0) {
            assert!(tx.delete(&coll, &format!("doc-{n}")).unwrap());
        }
        assert!(!tx.delete(&coll, "nope").unwrap());
        tx.commit().unwrap();
    }
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.len().unwrap(), 40 - 7);
    assert!(coll.get("doc-0").unwrap().is_none());
    // Every "rust" doc (n % 6 == 0) is gone, from the index too.
    let hits = coll
        .search(&embed_one("rust guide rust"))
        .k(50)
        .chunks()
        .unwrap();
    assert!(
        hits.iter().all(|h| !h.text.contains("rust")),
        "{:?}",
        hits.iter().map(|h| &h.key).collect::<Vec<_>>()
    );
    assert!(
        coll.scan(Some(&meta("topic").eq("rust")))
            .unwrap()
            .is_empty()
    );
}

#[test]
fn tombstones_trigger_rebuilds_and_optimize_merges() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let db = Db::open(&path).unwrap();
    let coll = fill(&db, "docs", 60, small_segments());
    let before = coll.index_stats();
    assert!(before.sealed_segments >= 3, "{before:?}");

    // Delete half the documents: segments pass the 20% tombstone threshold.
    let mut tx = db.begin_write().unwrap();
    for n in (0..60).filter(|n| n % 2 == 0) {
        tx.delete(&coll, &format!("doc-{n}")).unwrap();
    }
    tx.commit().unwrap();
    let after = coll.index_stats();
    let expected: usize = (0..60)
        .filter(|n| n % 2 == 1)
        .map(|n| coll.chunks(&format!("doc-{n}")).unwrap().len())
        .sum();
    assert_eq!(after.chunks, expected, "{before:?} -> {after:?}");

    db.optimize().unwrap();
    let optimized = coll.index_stats();
    assert_eq!(optimized.buffered, 0);
    assert!(
        optimized.sealed_segments < before.sealed_segments,
        "{before:?} -> {optimized:?}"
    );
    assert_eq!(optimized.chunks, after.chunks);
    drop(coll);
    drop(db);

    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.index_stats(), optimized);
    let hits = coll.search(&embed_one("python")).k(100).chunks().unwrap();
    let keys: HashSet<String> = hits.iter().map(|h| h.key.clone()).collect();
    assert!(
        keys.iter()
            .all(|k| k.trim_start_matches("doc-").parse::<usize>().unwrap() % 2 == 1)
    );
}

#[test]
fn filters() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = fill(&db, "docs", 24, small_segments());
    let q = embed_one("guide part detail");
    let keys = |f: vdblite::Filter| -> HashSet<usize> {
        let by_search: HashSet<usize> = coll
            .search(&q)
            .k(1000)
            .filter(f.clone())
            .chunks()
            .unwrap()
            .iter()
            .map(|h| h.key[4..].parse().unwrap())
            .collect();
        let by_scan: HashSet<usize> = coll
            .scan(Some(&f))
            .unwrap()
            .iter()
            .map(|d| d.key[4..].parse().unwrap())
            .collect();
        assert_eq!(by_search, by_scan, "index and scan disagree for {f:?}");
        by_scan
    };
    let range = |r: std::ops::Range<usize>| -> HashSet<usize> { r.collect() };

    assert_eq!(keys(meta("topic").eq("chess")), [5, 11, 17, 23].into());
    assert_eq!(keys(meta("tags").eq("chess")), [5, 11, 17, 23].into());
    assert_eq!(
        keys(meta("topic").is_in(["rust", "chess"])),
        [0, 5, 6, 11, 12, 17, 18, 23].into()
    );
    assert_eq!(keys(meta("n").between(3.0, 6.0)), [3, 4, 5, 6].into());
    assert_eq!(keys(meta("n").gt(20.0)), [21, 22, 23].into());
    assert_eq!(keys(meta("n").lt(2.0)), [0, 1].into());
    assert_eq!(keys(meta("n").eq(7)), [7].into());
    assert_eq!(keys(meta("n").eq(7.0)), [7].into());
    assert_eq!(keys(meta("n").between(5.0, 2.0)), HashSet::new());
    assert_eq!(keys(meta("topic").exists()), range(0..24));
    assert_eq!(keys(meta("missing").exists()), HashSet::new());
    assert_eq!(
        keys(meta("n").lt(12.0).and(meta("tags").eq("odd"))),
        [1, 3, 5, 7, 9, 11].into()
    );
    assert_eq!(
        keys(meta("n").lt(2.0).or(meta("n").gte(22.0))),
        [0, 1, 22, 23].into()
    );
    assert_eq!(
        keys(meta("topic").ne("rust").and(meta("n").lt(7.0))),
        [1, 2, 3, 4, 5].into()
    );
    assert_eq!(keys(!meta("n").gte(3.0)), [0, 1, 2].into());
}

#[test]
fn retrieval_shapes() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = fill(&db, "docs", 12, IndexOptions::default());
    let q = embed_one("gardening detail 1 article 2");

    let hits = coll.search(&q).k(1).chunks().unwrap();
    let best = &hits[0];
    assert_eq!(best.key, "doc-2");

    // Keys and ids name the same hits as the hydrated shape, in order.
    let full = coll.search(&q).k(8).chunks().unwrap();
    let keys = coll.search(&q).k(8).keys().unwrap();
    let ids = coll.search(&q).k(8).chunk_ids().unwrap();
    assert_eq!(
        keys,
        full.iter()
            .map(|h| (h.key.clone(), h.score))
            .collect::<Vec<_>>()
    );
    assert_eq!(
        ids,
        full.iter()
            .map(|h| (h.chunk_id, h.score))
            .collect::<Vec<_>>()
    );

    let passages = coll.search(&q).k(1).expand(1).unwrap();
    assert_eq!(passages.len(), 1);
    let p = &passages[0];
    assert!(p.range.start <= best.range.start && best.range.end <= p.range.end);
    assert_eq!(p.ordinals.len(), 3.min(p.ordinals.end as usize));
    let doc = coll.get("doc-2").unwrap().unwrap();
    assert_eq!(p.text, doc.content[p.range.clone()]);

    // Two adjacent hits in one document merge into one passage.
    let merged = coll.search(&q).k(4).expand(2).unwrap();
    let per_doc: HashSet<u64> = merged.iter().map(|p| p.doc_id).collect();
    assert_eq!(per_doc.len(), merged.len(), "touching passages must merge");

    let sections = coll.search(&q).k(1).section().unwrap();
    assert!(
        sections[0].text.starts_with("## Part 1"),
        "{:?}",
        sections[0].text
    );
    assert_eq!(sections[0].heading_path, ["gardening guide", "Part 1"]);

    let docs = coll
        .search(&embed_one("cooking"))
        .k(3)
        .documents(Agg::MaxSim)
        .unwrap();
    assert_eq!(docs.len(), 3);
    // The two cooking documents rank first; the third is the best of the rest.
    assert!(
        docs[..2]
            .iter()
            .all(|d| d.meta["topic"] == MetaValue::Str("cooking".into()))
    );
    assert_ne!(docs[2].meta["topic"], MetaValue::Str("cooking".into()));
    assert!(docs[0].score >= docs[1].score && docs[1].score >= docs[2].score);
    assert!(docs[0].chunks.windows(2).all(|w| w[0].score >= w[1].score));
    let summed = coll
        .search(&embed_one("cooking"))
        .k(2)
        .documents(Agg::SumTop(3))
        .unwrap();
    assert!(summed[0].score > docs[0].score);
}

#[test]
fn caller_supplied_vectors_and_unchunked_collections() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let whole = db
        .create_collection(
            "whole",
            Schema {
                chunking: Chunking::None,
                ..Schema::new(3, Metric::L2)
            },
        )
        .unwrap();
    let mut tx = db.begin_write().unwrap();
    tx.upsert_with_vectors(
        &whole,
        Doc::new("x").content("long text ".repeat(500)),
        vec![vec![1.0, 0.0, 0.0]],
    )
    .unwrap();
    tx.upsert_with_vectors(
        &whole,
        Doc::new("y").content("other"),
        vec![vec![0.0, 5.0, 0.0]],
    )
    .unwrap();
    let err = tx.upsert_with_vectors(
        &whole,
        Doc::new("z").content("bad"),
        vec![vec![0.0; 3], vec![0.0; 3]],
    );
    assert!(matches!(err, Err(Error::InvalidArgument(_))));
    tx.commit().unwrap();
    let hits = whole.search(&[0.9, 0.0, 0.0]).k(2).chunks().unwrap();
    assert_eq!(hits[0].key, "x");
    assert!(
        (hits[0].score + 0.01).abs() < 1e-4,
        "negative squared L2: {}",
        hits[0].score
    );
    assert_eq!(whole.chunks("x").unwrap().len(), 1);

    // Same content with new vectors (say, a new model) replaces the vectors.
    let mut tx = db.begin_write().unwrap();
    let again = tx
        .upsert_with_vectors(
            &whole,
            Doc::new("x").content("long text ".repeat(500)),
            vec![vec![0.0, 0.0, 9.0]],
        )
        .unwrap();
    tx.commit().unwrap();
    assert!(again.changed);
    assert_eq!(
        whole.search(&[0.0, 0.0, 9.0]).k(1).chunks().unwrap()[0].key,
        "x"
    );

    let structured = db.create_collection("s", schema()).unwrap();
    let text = article("chess", 9);
    let inputs = structured.embed_inputs(&text, Some("md"));
    let vectors: Vec<Vec<f32>> = inputs.iter().map(|t| embed_one(t)).collect();
    let mut tx = db.begin_write().unwrap();
    let up = tx
        .upsert_with_vectors(
            &structured,
            Doc::new("c").content(text).format("md"),
            vectors,
        )
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(up.chunks, inputs.len());
}

#[test]
fn f32_collections() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db
        .create_collection_with(
            "f",
            Schema {
                quant: Quant::None,
                ..schema()
            },
            small_segments(),
        )
        .unwrap();
    let e = WordEmbedder::default();
    let mut tx = db.begin_write().unwrap();
    for n in 0..30 {
        tx.upsert(
            &coll,
            Doc::new(format!("d{n}")).content(article(TOPICS[n % 6], n)),
            &e,
        )
        .unwrap();
    }
    tx.commit().unwrap();
    db.wait_for_maintenance().unwrap();
    assert!(coll.index_stats().sealed_segments >= 1);
    let hits = coll.search(&embed_one("cooking")).k(3).chunks().unwrap();
    assert!(hits.iter().all(|h| h.text.contains("cooking")));
}

#[test]
fn rollback_and_transaction_scoping() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db.create_collection("docs", schema()).unwrap();
    let e = WordEmbedder::default();

    let mut tx = db.begin_write().unwrap();
    tx.upsert(&coll, Doc::new("gone").content("never committed"), &e)
        .unwrap();
    drop(tx);
    let mut tx = db.begin_write().unwrap();
    tx.upsert(&coll, Doc::new("gone2").content("rolled back"), &e)
        .unwrap();
    tx.rollback().unwrap();
    assert!(coll.is_empty().unwrap());
    assert_eq!(coll.index_stats().chunks, 0);

    // Upsert twice and delete within one transaction.
    let mut tx = db.begin_write().unwrap();
    tx.upsert(&coll, Doc::new("a").content("first version"), &e)
        .unwrap();
    tx.upsert(&coll, Doc::new("a").content("second version entirely"), &e)
        .unwrap();
    tx.upsert(&coll, Doc::new("b").content("short lived"), &e)
        .unwrap();
    tx.delete(&coll, "b").unwrap();
    tx.commit().unwrap();
    assert_eq!(coll.len().unwrap(), 1);
    assert_eq!(coll.get("a").unwrap().unwrap().version, 2);
    assert_eq!(coll.index_stats().chunks, coll.chunks("a").unwrap().len());
}

#[test]
fn errors() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let db = Db::open(&path).unwrap();
    let coll = db.create_collection("docs", schema()).unwrap();
    assert!(matches!(
        db.create_collection("docs", schema()),
        Err(Error::CollectionExists(_))
    ));
    assert!(matches!(
        db.collection("nope"),
        Err(Error::NoSuchCollection(_))
    ));
    assert!(matches!(
        db.create_collection("bad", Schema::new(0, Metric::L2)),
        Err(Error::InvalidArgument(_))
    ));
    assert!(matches!(
        coll.search(&[1.0; 3]).chunks(),
        Err(Error::Dimension {
            expected: DIM,
            got: 3
        })
    ));
    let mut nan = vec![0.0; DIM];
    nan[0] = f32::NAN;
    assert!(matches!(
        coll.search(&nan).chunks(),
        Err(Error::InvalidArgument(_))
    ));

    let mut tx = db.begin_write().unwrap();
    let wrong_dim = |_: &[&str]| vec![vec![0.0f32; 5]];
    assert!(matches!(
        tx.upsert(&coll, Doc::new("a").content("x"), &wrong_dim),
        Err(Error::Dimension { .. })
    ));
    let bad_field = tx.upsert(
        &coll,
        Doc::new("a").content("x").meta("a\0b", 1),
        &WordEmbedder::default(),
    );
    assert!(matches!(bad_field, Err(Error::InvalidArgument(_))));
    assert!(matches!(
        tx.upsert(&coll, Doc::new("").content("x"), &WordEmbedder::default()),
        Err(Error::InvalidArgument(_))
    ));
    drop(tx);

    // The file is locked while open.
    assert!(Db::open(&path).is_err());
}

#[test]
fn drop_collection_removes_everything() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let db = Db::open(&path).unwrap();
    fill(&db, "a", 20, small_segments());
    let b = fill(&db, "b", 5, IndexOptions::default());
    db.drop_collection("a").unwrap();
    assert_eq!(db.collections().unwrap(), ["b"]);
    assert!(matches!(
        db.drop_collection("a"),
        Err(Error::NoSuchCollection(_))
    ));
    assert_eq!(b.len().unwrap(), 5);
    drop(b);
    drop(db);
    let db = Db::open(&path).unwrap();
    assert_eq!(db.collections().unwrap(), ["b"]);
    let a = db.create_collection("a", schema()).unwrap();
    assert!(a.is_empty().unwrap());
    assert!(a.scan(None).unwrap().is_empty());
}

#[test]
fn readers_run_alongside_the_writer() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = fill(&db, "docs", 20, small_segments());
    let stop = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let readers: Vec<_> = (0..4)
        .map(|i| {
            let (coll, stop) = (coll.clone(), stop.clone());
            std::thread::spawn(move || {
                let mut searches = 0;
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    let q = embed_one(TOPICS[i % TOPICS.len()]);
                    for h in coll.search(&q).k(10).chunks().unwrap() {
                        // Text always matches the document it came from.
                        let doc = coll.get(&h.key).unwrap();
                        if let Some(d) = doc
                            && d.version == 1
                        {
                            assert_eq!(&d.content[h.range.clone()], h.text);
                        }
                    }
                    searches += 1;
                }
                searches
            })
        })
        .collect();
    let e = WordEmbedder::default();
    for round in 0..15 {
        let mut tx = db.begin_write().unwrap();
        for n in 0..10 {
            let key = format!("w-{round}-{n}");
            tx.upsert(
                &coll,
                Doc::new(key).content(article(TOPICS[n % 6], round * 10 + n)),
                &e,
            )
            .unwrap();
        }
        if round > 0 {
            tx.delete(&coll, &format!("w-{}-0", round - 1)).unwrap();
        }
        tx.commit().unwrap();
    }
    stop.store(true, std::sync::atomic::Ordering::Relaxed);
    let total: usize = readers.into_iter().map(|r| r.join().unwrap()).sum();
    assert!(total > 0);
    assert_eq!(coll.len().unwrap(), 20 + 150 - 14);
}

#[test]
fn planned_embeddings_reuse_unchanged_chunks() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db.create_collection("docs", schema()).unwrap();
    let embed_plan = |plan: Vec<(u64, String)>| -> std::collections::HashMap<u64, Vec<f32>> {
        plan.into_iter().map(|(h, t)| (h, embed_one(&t))).collect()
    };

    let text = article("rust", 1);
    let doc = Doc::new("a").content(text.clone());
    let plan = coll.plan_embeddings(&doc).unwrap();
    let n = plan.len();
    assert!(n > 2);
    let mut tx = db.begin_write().unwrap();
    let up = tx
        .upsert_with_embeddings(&coll, doc.clone(), embed_plan(plan))
        .unwrap();
    tx.commit().unwrap();
    assert_eq!((up.chunks, up.embedded), (n, n));

    // Unchanged: nothing to embed.
    assert!(coll.plan_embeddings(&doc).unwrap().is_empty());

    // One section edited: only the affected chunks are planned.
    let edited =
        Doc::new("a").content(text.replace("detail 2 for article 1", "detail 2, rewritten"));
    let plan = coll.plan_embeddings(&edited).unwrap();
    assert!(!plan.is_empty() && plan.len() < n, "{} of {n}", plan.len());
    let stale = embed_plan(plan.clone());

    // The document changes again before the planned write lands: stale.
    let mut tx = db.begin_write().unwrap();
    let other = Doc::new("a").content("completely different text");
    tx.upsert(&coll, other, &WordEmbedder::default()).unwrap();
    tx.commit().unwrap();
    let mut tx = db.begin_write().unwrap();
    let err = tx
        .upsert_with_embeddings(&coll, edited.clone(), stale)
        .unwrap_err();
    assert!(matches!(err, Error::StaleEmbeddings { .. }), "{err}");
    drop(tx);

    // Re-plan and it goes through.
    let plan = coll.plan_embeddings(&edited).unwrap();
    let mut tx = db.begin_write().unwrap();
    tx.upsert_with_embeddings(&coll, edited, embed_plan(plan))
        .unwrap();
    tx.commit().unwrap();
    assert!(
        coll.get("a")
            .unwrap()
            .unwrap()
            .content
            .contains("rewritten")
    );
    let hits = coll.search(&embed_one("rewritten")).k(3).chunks().unwrap();
    assert!(hits.iter().any(|h| h.text.contains("rewritten")));
}

#[test]
fn hybrid_search_finds_exact_terms() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let db = Db::open(&path).unwrap();
    let coll = fill(&db, "docs", 12, IndexOptions::default());
    assert!(coll.has_lexical());
    let e = WordEmbedder::default();
    let mut tx = db.begin_write().unwrap();
    let rare = "# Config\n\nSet ZX_QUUX_LIMIT to raise the ceiling.\n\n## Other\n\nUnrelated gardening notes about soil and seeds.\n";
    tx.upsert(
        &coll,
        Doc::new("rare").content(rare).meta("topic", "config"),
        &e,
    )
    .unwrap();
    tx.commit().unwrap();

    // The vector points at gardening; only the keyword index knows ZX_QUUX_LIMIT.
    let q = embed_one("gardening soil seeds");
    let vector_only = coll.search(&q).k(3).chunks().unwrap();
    assert!(vector_only.iter().all(|h| !h.text.contains("ZX_QUUX")));
    let hybrid = coll.search(&q).text("ZX_QUUX_LIMIT").k(3).chunks().unwrap();
    let hit = hybrid
        .iter()
        .find(|h| h.text.contains("ZX_QUUX_LIMIT"))
        .expect("exact term found");
    assert!(hit.lexical_score.unwrap() > 0.0);
    // Identifier parts match too.
    let parts = coll.search(&q).text("quux limit").k(3).chunks().unwrap();
    assert!(parts.iter().any(|h| h.text.contains("ZX_QUUX_LIMIT")));
    // Filters apply to keyword hits.
    let filtered = coll
        .search(&q)
        .text("ZX_QUUX_LIMIT")
        .filter(meta("topic").eq("rust"))
        .k(20)
        .chunks()
        .unwrap();
    assert!(filtered.iter().all(|h| !h.text.contains("ZX_QUUX")));
    // Stopwords only: plain vector search.
    let stop = coll.search(&q).text("the a of").k(3).chunks().unwrap();
    assert!(
        stop.iter()
            .all(|h| h.lexical_score.is_none() && h.vector_score == Some(h.score))
    );

    // Edits and deletes update the keyword index.
    let mut tx = db.begin_write().unwrap();
    tx.upsert(
        &coll,
        Doc::new("rare").content(rare.replace("ZX_QUUX_LIMIT", "YY_NEW_KNOB")),
        &e,
    )
    .unwrap();
    tx.commit().unwrap();
    let old = coll.search(&q).text("ZX_QUUX_LIMIT").k(5).chunks().unwrap();
    assert!(old.iter().all(|h| h.lexical_score.is_none()));
    assert!(
        coll.search(&q)
            .text("YY_NEW_KNOB")
            .k(5)
            .chunks()
            .unwrap()
            .iter()
            .any(|h| h.text.contains("YY_NEW_KNOB"))
    );
    let mut tx = db.begin_write().unwrap();
    tx.delete(&coll, "rare").unwrap();
    tx.commit().unwrap();
    let gone = coll.search(&q).text("YY_NEW_KNOB").k(5).chunks().unwrap();
    assert!(gone.iter().all(|h| h.lexical_score.is_none()));

    // Hybrid works for every result shape.
    assert!(
        !coll
            .search(&q)
            .text("python")
            .k(2)
            .documents(Agg::MaxSim)
            .unwrap()
            .is_empty()
    );
    assert!(
        !coll
            .search(&q)
            .text("python")
            .k(2)
            .expand(1)
            .unwrap()
            .is_empty()
    );
    assert!(
        !coll
            .search(&q)
            .text("python")
            .k(2)
            .section()
            .unwrap()
            .is_empty()
    );
}

#[test]
fn keyword_index_backfills_existing_collections() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let db = Db::open(&path).unwrap();
        let opts = IndexOptions {
            lexical: false,
            ..IndexOptions::default()
        };
        let coll = fill(&db, "docs", 12, opts);
        assert!(!coll.has_lexical());
        let q = embed_one("anything");
        let hits = coll.search(&q).text("astronomy").k(3).chunks().unwrap();
        assert!(
            hits.iter().all(|h| h.lexical_score.is_none()),
            "no index yet: vector only"
        );

        let e = WordEmbedder::default();
        let before = e.texts_embedded();
        assert!(coll.enable_lexical().unwrap());
        assert!(!coll.enable_lexical().unwrap());
        assert_eq!(e.texts_embedded(), before);
        assert!(coll.has_lexical());
    }
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert!(coll.has_lexical());
    let hits = coll
        .search(&embed_one("chess"))
        .text("astronomy")
        .k(5)
        .chunks()
        .unwrap();
    assert!(
        hits.iter()
            .any(|h| h.lexical_score.is_some() && h.text.contains("astronomy"))
    );
    // Writes after the backfill keep the index current.
    let mut tx = db.begin_write().unwrap();
    tx.upsert(
        &coll,
        Doc::new("late").content("A note on quasars."),
        &WordEmbedder::default(),
    )
    .unwrap();
    tx.commit().unwrap();
    // RRF weighs both lists equally: the top keyword hit ties the top vector hit.
    let hits = coll
        .search(&embed_one("chess"))
        .text("quasars")
        .k(2)
        .chunks()
        .unwrap();
    assert!(hits.iter().any(|h| h.key == "late"));
}

#[test]
fn whole_words_outrank_identifier_parts() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db
        .create_collection(
            "c",
            Schema {
                chunking: Chunking::None,
                ..schema()
            },
        )
        .unwrap();
    let mut tx = db.begin_write().unwrap();
    tx.upsert_with_vectors(
        &coll,
        Doc::new("part").content("struct ChunkInfo { id: u32 }"),
        vec![vec![1.0; DIM]],
    )
    .unwrap();
    tx.upsert_with_vectors(
        &coll,
        Doc::new("whole").content("Call info() to read the schema."),
        vec![vec![1.0; DIM]],
    )
    .unwrap();
    tx.commit().unwrap();
    let hits = coll.search(&[1.0; DIM]).text("info").k(2).chunks().unwrap();
    assert_eq!(hits.len(), 2);
    assert_eq!(hits[0].key, "whole", "{hits:#?}");
    assert!(hits[0].lexical_score > hits[1].lexical_score);
    // The part still matches, and searching the identifier finds it first.
    assert_eq!(
        coll.search(&[1.0; DIM])
            .text("ChunkInfo")
            .k(1)
            .chunks()
            .unwrap()[0]
            .key,
        "part"
    );
}

#[test]
fn outdated_keyword_indexes_are_rebuilt() {
    use vdblite_core::records::{LexicalStats, encode};
    use vdblite_core::tables::LEXICAL;
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let id = {
        let db = Db::open(&path).unwrap();
        fill(&db, "docs", 6, IndexOptions::default());
        db.collections().unwrap().len() as u32 // the first collection gets id 1
    };
    // Rewrite the stats as an older index version would have left them.
    {
        let store = vdblite_core::container::open_store(&path).unwrap();
        let tx = store.begin_write().unwrap();
        let old = LexicalStats {
            chunks: 1,
            total_len: 1,
            version: 1,
        };
        tx.open_table(LEXICAL)
            .unwrap()
            .insert(id, encode(&old).as_slice())
            .unwrap();
        tx.commit().unwrap();
    }
    let db = Db::open(&path).unwrap();
    let coll = db.collection("docs").unwrap();
    assert!(!coll.has_lexical(), "an outdated index is not used");
    let q = embed_one("anything");
    assert!(
        coll.search(&q)
            .text("astronomy")
            .k(3)
            .chunks()
            .unwrap()
            .iter()
            .all(|h| h.lexical_score.is_none())
    );
    assert!(coll.enable_lexical().unwrap());
    assert!(coll.has_lexical());
    let hits = coll.search(&q).text("astronomy").k(3).chunks().unwrap();
    assert!(
        hits.iter()
            .any(|h| h.lexical_score.is_some() && h.text.contains("astronomy"))
    );
}

#[test]
fn compact_file_shrinks_and_keeps_everything() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    let queries = [
        "rust detail 3",
        "gardening guide",
        "chess opening 7",
        "cooking",
    ];
    let before = {
        let db = Db::open(&path).unwrap();
        let coll = fill(&db, "docs", 40, small_segments());
        let mut tx = db.begin_write().unwrap();
        for n in (0..40).filter(|n| n % 4 != 0) {
            tx.delete(&coll, &format!("doc-{n}")).unwrap();
        }
        tx.commit().unwrap();
        db.optimize().unwrap();
        db.wait_for_maintenance().unwrap();
        queries
            .iter()
            .map(|q| coll.search(&embed_one(q)).k(5).keys().unwrap())
            .collect::<Vec<_>>()
    };
    let len = |p: &std::path::Path| std::fs::metadata(p).unwrap().len();
    let big = len(&path);
    assert!(
        Db::compact_file(&path).unwrap(),
        "freed space was reclaimed"
    );
    assert!(len(&path) < big);

    for db in [Db::open(&path).unwrap(), Db::open_read_only(&path).unwrap()] {
        let coll = db.collection("docs").unwrap();
        assert_eq!(coll.len().unwrap(), 10);
        let after: Vec<_> = queries
            .iter()
            .map(|q| coll.search(&embed_one(q)).k(5).keys().unwrap())
            .collect();
        assert_eq!(after, before);
    }
    // The compacted file keeps working.
    let db = Db::open(&path).unwrap();
    let coll = fill(&db, "more", 20, small_segments());
    db.optimize().unwrap();
    assert_eq!(coll.len().unwrap(), 20);
}

fn unchunked() -> Schema {
    Schema {
        chunking: Chunking::None,
        ..schema()
    }
}

/// The text of note `n` in the bulk-load tests.
fn note(n: usize) -> String {
    format!("{} note {n}", TOPICS[n % TOPICS.len()])
}

#[test]
fn bulk_loads_index_once_at_the_end() {
    let dir = tempfile::tempdir().unwrap();
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db
        .create_collection_with("docs", unchunked(), small_segments())
        .unwrap();
    let doc = |n: usize| {
        let text = note(n);
        (
            Doc::new(format!("doc-{n}")).content(text.clone()),
            vec![embed_one(&text)],
        )
    };
    let mut load = db.bulk_load(&coll).unwrap();
    assert!(db.bulk_load(&coll).is_err(), "one bulk load at a time");
    load.add((0..150).map(doc)).unwrap();
    // A delete and a replacement, from another transaction, mid-load.
    let mut tx = db.begin_write().unwrap();
    tx.delete(&coll, "doc-3").unwrap();
    tx.commit().unwrap();
    load.add([doc(4)]).unwrap();
    load.add((150..300).map(doc)).unwrap();

    // Readable, not yet searchable, and nothing sealed along the way.
    assert!(coll.get("doc-200").unwrap().is_some());
    let q = embed_one(&note(200));
    assert!(coll.search(&q).k(5).keys().unwrap().is_empty());
    assert_eq!(coll.index_stats().sealed_segments, 0);

    load.finish().unwrap();
    let stats = coll.index_stats();
    assert_eq!(
        (stats.chunks, stats.sealed_segments, stats.buffered),
        (299, 1, 0)
    );
    assert_eq!(coll.search(&q).k(1).keys().unwrap()[0].0, "doc-200");
    assert!(coll.get("doc-3").unwrap().is_none());
    let hits = coll.search(&embed_one(&note(4))).k(3).keys().unwrap();
    assert_eq!(hits[0].0, "doc-4");
    assert_eq!(hits.iter().filter(|(k, _)| k == "doc-4").count(), 1);
    // Keyword search covers bulk-loaded documents too.
    let hybrid = coll.search(&q).text("note 200").k(1).keys().unwrap();
    assert_eq!(hybrid[0].0, "doc-200");
    // And the collection takes ordinary writes after.
    db.bulk_load(&coll).unwrap().finish().unwrap();
    let mut tx = db.begin_write().unwrap();
    tx.upsert_with_vectors(&coll, doc(300).0, doc(300).1)
        .unwrap();
    tx.commit().unwrap();
    assert_eq!(coll.len().unwrap(), 300);

    // The rows deleted and replaced mid-load stay gone after reopening.
    drop((coll, db));
    let db = Db::open(dir.path().join("t.vdb")).unwrap();
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.index_stats().chunks, 300);
    let hits = coll.search(&embed_one(&note(3))).k(5).keys().unwrap();
    assert!(hits.iter().all(|(k, _)| k != "doc-3"));
    let hits = coll.search(&embed_one(&note(4))).k(3).keys().unwrap();
    assert_eq!(hits.iter().filter(|(k, _)| k == "doc-4").count(), 1);
}

#[test]
fn unfinished_bulk_loads_are_finished_after_reopening() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("t.vdb");
    {
        let db = Db::open(&path).unwrap();
        let coll = db
            .create_collection_with("docs", unchunked(), small_segments())
            .unwrap();
        let mut load = db.bulk_load(&coll).unwrap().checkpoint_every(64);
        for start in (0..200).step_by(50) {
            load.add((start..start + 50).map(|n| {
                let text = note(n);
                (
                    Doc::new(format!("doc-{n}")).content(text.clone()),
                    vec![embed_one(&text)],
                )
            }))
            .unwrap();
        }
        // Dropped without finishing: checkpointed, finished by maintenance.
    }
    let db = Db::open(&path).unwrap();
    db.wait_for_maintenance().unwrap();
    let coll = db.collection("docs").unwrap();
    assert_eq!(coll.index_stats().chunks, 200);
    let hit = coll.search(&embed_one(&note(129))).k(1).keys().unwrap();
    assert_eq!(hit[0].0, "doc-129");
    // A reader sees it too.
    let r = Db::open_read_only(&path).unwrap();
    assert_eq!(r.collection("docs").unwrap().index_stats().chunks, 200);
}
