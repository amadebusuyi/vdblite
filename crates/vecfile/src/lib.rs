//! vecfile: an embedded, single-file vector database.
//!
//! Open it like SQLite, store documents, and search them by meaning. Large
//! documents are split into chunks along their structure; search returns the
//! chunks that matched, the passages around them, their sections, or whole
//! documents ranked by their best chunks.
//!
//! ```
//! use vecfile::{Agg, Chunking, Db, Doc, Metric, Quant, Schema, meta};
//!
//! # fn main() -> vecfile::Result<()> {
//! # let dir = tempfile::tempdir().unwrap();
//! let db = Db::open(dir.path().join("notes.vdb"))?;
//! let docs = db.create_collection("docs", Schema {
//!     dim: 3,
//!     metric: Metric::Cosine,
//!     quant: Quant::Int8,
//!     chunking: Chunking::Structural { target_tokens: 400, overlap_tokens: 50 },
//! })?;
//!
//! // Any model works: implement `Embedder`, or pass a closure.
//! let embedder = |texts: &[&str]| -> Vec<Vec<f32>> {
//!     texts.iter().map(|t| vec![t.len() as f32, 1.0, t.matches("rust").count() as f32]).collect()
//! };
//!
//! let mut tx = db.begin_write()?;
//! tx.upsert(&docs, Doc::new("readme-42").content("# Intro\n\nrust rust rust").meta("lang", "en"), &embedder)?;
//! tx.commit()?;
//!
//! let hits = docs.search(&[10.0, 1.0, 3.0]).k(10).filter(meta("lang").eq("en")).chunks()?;
//! assert_eq!(hits[0].key, "readme-42");
//! let top_docs = docs.search(&[10.0, 1.0, 3.0]).k(5).documents(Agg::MaxSim)?;
//! assert_eq!(top_docs.len(), 1);
//! let full = docs.get("readme-42")?.unwrap();
//! assert!(full.content.starts_with("# Intro"));
//! # Ok(())
//! # }
//! ```

mod collection;
mod db;
mod doc;
mod error;
mod filter;
mod lexical;
mod maintenance;
mod schema;
mod tx;

pub use collection::{Agg, ChunkHit, Collection, DocHit, IndexStats, Passage, Search};
pub use db::{DEFAULT_CACHE_BYTES, Db, OpenOptions};
pub use doc::{ChunkInfo, Doc, DocInfo, Document, Embedder, Meta, MetaValue, Upserted};
pub use error::{BoxError, Error, Result};
pub use filter::{Field, Filter, meta};
pub use schema::{Chunking, IndexOptions, Quant, Schema};
pub use tx::WriteTx;
pub use vecfile_index::Metric;
pub use vecfile_index::hnsw::HnswParams;

/// Lower-level building blocks: distance kernels, the index, chunking.
pub mod advanced {
    pub use vecfile_chunk as chunk;
    pub use vecfile_index as index;
}
