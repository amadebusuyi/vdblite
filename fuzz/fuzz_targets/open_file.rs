//! Arbitrary bytes as a database file: opening and reading must return
//! errors, never panic. Seed the corpus with real `.vdb` files for depth.
#![no_main]

use libfuzzer_sys::fuzz_target;
use vdblite::Db;

fuzz_target!(|data: &[u8]| {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("f.vdb");
    std::fs::write(&path, data).unwrap();
    if let Ok(db) = Db::open(&path) {
        for name in db.collections().unwrap_or_default() {
            if let Ok(coll) = db.collection(&name) {
                let dim = coll.schema().dim;
                let _ = coll.search(&vec![0.1; dim]).k(3).chunks();
                let _ = coll.scan(None);
            }
        }
    }
});
