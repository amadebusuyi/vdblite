//! Writes the format-version-2 fixture for vdblite's upgrade test.
use vecfile::{Chunking, Db, Doc, HnswParams, IndexOptions, Metric, Quant, Schema};

fn vector(i: usize) -> Vec<f32> {
    (0..8)
        .map(|d| {
            let x = ((i * 8 + d) as f64 * 12.9898).sin() * 43758.5453;
            (x - x.floor() - 0.5) as f32
        })
        .collect()
}

fn main() -> vecfile::Result<()> {
    let path = std::env::args().nth(1).expect("output path");
    let _ = std::fs::remove_file(&path);
    let db = Db::open_exclusive(&path)?;
    let schema = Schema { dim: 8, metric: Metric::Cosine, quant: Quant::Int8, chunking: Chunking::None };
    let opts = IndexOptions { hnsw: HnswParams::default(), seal_threshold: 64, lexical: true };
    let coll = db.create_collection_with("docs", schema, opts)?;
    for start in (0..300).step_by(50) {
        let mut tx = db.begin_write()?;
        for i in start..start + 50 {
            let doc = Doc::new(format!("doc-{i}")).content(format!("document number {i}")).meta("n", i as i64);
            tx.upsert_with_vectors(&coll, doc, vec![vector(i)])?;
        }
        tx.commit()?;
        db.wait_for_maintenance()?;
    }
    let mut tx = db.begin_write()?;
    for i in (0..300).step_by(7) {
        tx.delete(&coll, &format!("doc-{i}"))?;
    }
    tx.commit()?;
    db.wait_for_maintenance()?;
    // Below the seal threshold: these stay in the write buffer.
    let mut tx = db.begin_write()?;
    for i in 300..320 {
        let doc = Doc::new(format!("doc-{i}")).content(format!("document number {i}")).meta("n", i as i64);
        tx.upsert_with_vectors(&coll, doc, vec![vector(i)])?;
    }
    tx.commit()?;
    db.wait_for_maintenance()?;
    println!("{:?}", coll.index_stats());
    Ok(())
}
