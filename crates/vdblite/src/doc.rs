use std::collections::BTreeMap;
use std::ops::Range;

pub use vdblite_core::meta::MetaValue;

use crate::error::BoxError;

/// Document metadata: field name to value.
pub type Meta = BTreeMap<String, MetaValue>;

/// A document to write: a key, text content and metadata.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Doc {
    pub(crate) key: String,
    pub(crate) content: String,
    pub(crate) meta: Meta,
    pub(crate) format: Option<String>,
}

impl Doc {
    /// A document identified by `key`, unique within its collection.
    pub fn new(key: impl Into<String>) -> Self {
        Doc {
            key: key.into(),
            ..Doc::default()
        }
    }

    pub fn content(mut self, text: impl Into<String>) -> Self {
        self.content = text.into();
        self
    }

    /// Sets a metadata field, replacing any earlier value.
    pub fn meta(mut self, field: impl Into<String>, value: impl Into<MetaValue>) -> Self {
        self.meta.insert(field.into(), value.into());
        self
    }

    /// A format hint for chunking: `"markdown"`, `"code"`, `"html"`,
    /// `"text"`, a file name, an extension or a MIME type. Without one the
    /// format is sniffed from the content.
    pub fn format(mut self, hint: impl Into<String>) -> Self {
        self.format = Some(hint.into());
        self
    }

    pub fn key(&self) -> &str {
        &self.key
    }
}

/// A stored document.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub id: u64,
    pub key: String,
    pub content: String,
    pub meta: Meta,
    /// Starts at 1 and increases with every change.
    pub version: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub format: &'static str,
    pub chunk_count: usize,
}

/// A stored document's summary, without its content.
#[derive(Clone, Debug, PartialEq)]
pub struct DocInfo {
    pub id: u64,
    pub key: String,
    pub meta: Meta,
    pub version: u64,
    pub created_ms: u64,
    pub updated_ms: u64,
    pub content_len: u64,
    pub chunk_count: usize,
}

/// A stored chunk's position in its document.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkInfo {
    pub chunk_id: u32,
    pub ordinal: u32,
    pub range: Range<usize>,
    pub overlap_start: usize,
    pub heading_path: Vec<String>,
    pub tokens: u32,
}

/// What an upsert did.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Upserted {
    pub doc_id: u64,
    pub version: u64,
    /// Chunks the document now has.
    pub chunks: usize,
    /// Chunks that needed new embeddings (the rest were unchanged and reused).
    pub embedded: usize,
    /// False when content and metadata were identical, so nothing was written.
    pub changed: bool,
}

/// Turns text into vectors. Implement this for your model, or pass a closure.
pub trait Embedder: Send + Sync {
    /// One vector per input text, in order.
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BoxError>;
}

impl<F> Embedder for F
where
    F: Fn(&[&str]) -> Vec<Vec<f32>> + Send + Sync,
{
    fn embed(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>, BoxError> {
        Ok(self(texts))
    }
}
