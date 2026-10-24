//! Collections: reads, search and result hydration.

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use parking_lot::{Mutex, RwLock};
use redb::{ReadTransaction, ReadableTable};
use vecfile_chunk::Format;
use vecfile_core::changes;
use vecfile_core::codec::{decompress, read_segment, read_vector};
use vecfile_core::records::{Change, ChunkRecord, CollectionRecord, DocRecord, decode};
use vecfile_core::tables::{
    BLOBS, CATALOG, CHANGES, CHUNKS, DOCS, KEYS, LEXICAL, META_INDEX, POSTINGS, SEGMENT_DATA,
    SEGMENTS, TERM_DF, VECTORS,
};
use vecfile_index::{Metric, Neighbor, VectorIndex};

use crate::db::{DbRef, load_index};
use crate::doc::{ChunkInfo, Doc, DocInfo, Document, Embedder, Meta, MetaValue};
use crate::error::{Error, Result, guard};
use crate::filter::Filter;
use crate::maintenance::FileVectors;
use crate::schema::{self, IndexOptions, Schema};
use crate::tx::{chunk_content, content_fingerprint};

pub(crate) struct CollInner {
    pub id: u32,
    pub name: String,
    pub schema: Schema,
    pub opts: IndexOptions,
    pub index: RwLock<VectorIndex>,
    /// Whether the collection has a keyword index.
    pub lexical: AtomicBool,
    /// Held while sealing or compacting, so only one runs at a time.
    pub maint: Mutex<()>,
    /// The last background maintenance failure, until reported.
    pub maint_error: Mutex<Option<Error>>,
    /// In a reading process: the last change-log entry applied to `index`.
    pub synced: Mutex<u64>,
}

impl CollInner {
    pub fn new(rec: &CollectionRecord, schema: Schema, opts: IndexOptions) -> Self {
        let index = VectorIndex::new(schema::index_config(&schema, &opts));
        Self::with_index(rec, schema, opts, index)
    }

    pub fn with_index(
        rec: &CollectionRecord,
        schema: Schema,
        opts: IndexOptions,
        index: VectorIndex,
    ) -> Self {
        CollInner {
            id: rec.id,
            name: rec.name.clone(),
            schema,
            opts,
            index: RwLock::new(index),
            lexical: AtomicBool::new(opts.lexical),
            maint: Mutex::new(()),
            maint_error: Mutex::new(None),
            synced: Mutex::new(0),
        }
    }

    /// In a reading process, brings the index up to the snapshot `rx` by
    /// replaying the writer's change log (or reloading, if this process has
    /// fallen too far behind). The index then reflects at least every commit
    /// `rx` sees.
    pub fn sync(&self, rx: &ReadTransaction) -> Result<()> {
        let mut synced = self.synced.lock();
        let log = match rx.open_table(CHANGES) {
            Ok(t) => t,
            // A version 1 file not yet upgraded: nothing writes to it.
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let Some(entries) = changes::read_after(&log, self.id, *synced)? else {
            return self.reload(rx, &mut synced);
        };
        let Some(&(last, _)) = entries.last() else {
            return Ok(());
        };

        // Read everything the entries need from the file first, so the
        // index is locked only to apply them.
        let headers = rx.open_table(SEGMENTS)?;
        let data = rx.open_table(SEGMENT_DATA)?;
        let vectors = rx.open_table(VECTORS)?;
        let dim = self.schema.dim;
        enum Step {
            Write(Vec<u32>, Vec<(u32, Vec<f32>)>),
            Seal(u64, vecfile_index::segment::SealedSegment),
            Compact(
                Vec<u64>,
                Option<(u64, vecfile_index::segment::SealedSegment)>,
            ),
        }
        let mut steps = Vec::with_capacity(entries.len());
        for (_, change) in entries {
            steps.push(match change {
                Change::Write { deletes, inserts } => {
                    let mut found = Vec::with_capacity(inserts.len());
                    for id in inserts {
                        // Gone if a later commit (also visible in `rx`)
                        // deleted it; that commit's entry removes it again.
                        if let Some(v) = vectors.get((self.id, id))? {
                            let mut buf = vec![0f32; dim];
                            read_vector(v.value(), &mut buf)?;
                            found.push((id, buf));
                        }
                    }
                    Step::Write(deletes, found)
                }
                Change::Seal { segment } => match headers.get((self.id, segment))? {
                    Some(_) => {
                        Step::Seal(segment, read_segment(&headers, &data, self.id, segment)?)
                    }
                    // Already merged away by a later compaction.
                    None => return self.reload(rx, &mut synced),
                },
                Change::Compact { removed, added } => {
                    let replacement = match added {
                        Some(id) if headers.get((self.id, id))?.is_some() => {
                            Some((id, read_segment(&headers, &data, self.id, id)?))
                        }
                        Some(_) => return self.reload(rx, &mut synced),
                        None => None,
                    };
                    Step::Compact(removed, replacement)
                }
            });
        }

        let mut idx = self.index.write();
        for step in steps {
            match step {
                Step::Write(deletes, inserts) => {
                    for id in deletes {
                        idx.delete(id);
                    }
                    for (id, v) in inserts {
                        idx.insert(id, &v);
                    }
                }
                Step::Seal(id, seg) => idx.install_loaded(id, Arc::new(seg)),
                Step::Compact(removed, replacement) => {
                    if !removed.iter().all(|&r| idx.has_sealed(r)) {
                        drop(idx);
                        return self.reload(rx, &mut synced);
                    }
                    idx.replace_segments(&removed, replacement);
                }
            }
        }
        drop(idx);
        self.lexical.store(
            rx.open_table(LEXICAL)?
                .get(self.id)?
                .is_some_and(|v| crate::lexical::read_stats(v.value()).is_some()),
            Ordering::Release,
        );
        *synced = last;
        Ok(())
    }

    /// Replaces the index with a fresh load from `rx`.
    fn reload(&self, rx: &ReadTransaction, synced: &mut u64) -> Result<()> {
        let rec = match rx.open_table(CATALOG)?.get(self.name.as_str())? {
            Some(v) => decode::<CollectionRecord>(v.value(), "collection record")?,
            None => return Ok(()),
        };
        if rec.id != self.id {
            // Dropped and recreated: this handle's collection is gone.
            return Ok(());
        }
        let loaded = load_index(rx, &rec)?;
        *self.index.write() = loaded.index;
        self.lexical.store(loaded.lexical, Ordering::Release);
        *synced = loaded.seq;
        Ok(())
    }
}

pub(crate) fn format_to_u8(f: Format) -> u8 {
    match f {
        Format::Markdown => 0,
        Format::Code => 1,
        Format::Html => 2,
        Format::Text => 3,
    }
}

pub(crate) fn format_from_u8(b: u8) -> Format {
    match b {
        0 => Format::Markdown,
        1 => Format::Code,
        2 => Format::Html,
        _ => Format::Text,
    }
}

/// A handle to a collection of documents. Cheap to clone; safe to share.
#[derive(Clone)]
pub struct Collection {
    pub(crate) inner: Arc<CollInner>,
    pub(crate) db: DbRef,
}

/// Index size and shape, for monitoring.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IndexStats {
    /// Live chunks in the index.
    pub chunks: usize,
    pub sealed_segments: usize,
    /// Chunks in the write buffer, searched exactly until sealed.
    pub buffered: usize,
    /// Bytes the index keeps in memory.
    pub memory_bytes: usize,
    /// The last background maintenance failure, if it has not been
    /// reported by [`crate::Db::wait_for_maintenance`] yet. Maintenance
    /// retries after the next commit.
    pub maintenance_error: Option<String>,
}

/// How chunk scores combine into a document score.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Agg {
    /// The document's best chunk.
    MaxSim,
    /// The sum of its best `n` chunk scores.
    SumTop(usize),
}

/// A matching chunk.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkHit {
    pub doc_id: u64,
    pub key: String,
    pub chunk_id: u32,
    pub ordinal: u32,
    /// Byte range of the chunk in the document.
    pub range: Range<usize>,
    /// Higher is better. For a vector search: cosine similarity, inner
    /// product, or negative squared L2 distance. For a hybrid search (see
    /// [`Search::text`]): the fused rank score; compare `vector_score` and
    /// `lexical_score` for absolute values.
    pub score: f32,
    /// The vector similarity, if the vector search found this chunk.
    pub vector_score: Option<f32>,
    /// The BM25 score, if the keyword search found this chunk.
    pub lexical_score: Option<f32>,
    pub text: String,
    pub heading_path: Vec<String>,
    pub meta: Meta,
}

/// A ranked chunk before hydration.
#[derive(Clone, Copy, Debug)]
struct Hit {
    id: u32,
    score: f32,
    vector: Option<f32>,
    lexical: Option<f32>,
}

/// Reciprocal Rank Fusion constant: the standard 60 keeps any single list's
/// top rank from dominating.
const RRF_K: f32 = 60.0;

/// A contiguous span of a document around one or more hits.
#[derive(Clone, Debug, PartialEq)]
pub struct Passage {
    pub doc_id: u64,
    pub key: String,
    pub range: Range<usize>,
    pub text: String,
    /// The best score of the hits inside the passage.
    pub score: f32,
    /// Chunk ordinals the passage covers.
    pub ordinals: Range<u32>,
    pub heading_path: Vec<String>,
    pub meta: Meta,
}

/// A document ranked by its chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct DocHit {
    pub doc_id: u64,
    pub key: String,
    pub score: f32,
    pub meta: Meta,
    /// The document's matching chunks, best first.
    pub chunks: Vec<ChunkHit>,
}

impl Collection {
    pub fn name(&self) -> &str {
        &self.inner.name
    }

    pub fn schema(&self) -> &Schema {
        &self.inner.schema
    }

    pub fn index_options(&self) -> &IndexOptions {
        &self.inner.opts
    }

    /// The index's size and shape. In a reading process, this first catches
    /// up with the writer, if it can.
    pub fn index_stats(&self) -> IndexStats {
        let _ = self.sync();
        let maintenance_error = self
            .inner
            .maint_error
            .lock()
            .as_ref()
            .map(ToString::to_string);
        let idx = self.inner.index.read();
        IndexStats {
            chunks: idx.len(),
            sealed_segments: idx.sealed().len(),
            buffered: idx.mutable().len(),
            memory_bytes: idx.memory_bytes(),
            maintenance_error,
        }
    }

    fn read(&self) -> Result<Reader> {
        Ok(Reader {
            rx: self.db.store.begin_read()?,
            coll: self.inner.id,
            docs: HashMap::new(),
        })
    }

    /// In a reading process, catches the index up with the writer's commits.
    fn sync(&self) -> Result<()> {
        if self.db.store.is_reader() {
            guard(|| self.inner.sync(&self.db.store.begin_read()?))?;
        }
        Ok(())
    }

    /// Number of documents.
    pub fn len(&self) -> Result<usize> {
        guard(|| self.len_inner())
    }

    fn len_inner(&self) -> Result<usize> {
        let rx = self.db.store.begin_read()?;
        let c = self.inner.id;
        Ok(rx.open_table(DOCS)?.range((c, 0)..=(c, u64::MAX))?.count())
    }

    pub fn is_empty(&self) -> Result<bool> {
        Ok(self.len()? == 0)
    }

    /// Looks a document up by key.
    pub fn get(&self, key: &str) -> Result<Option<Document>> {
        guard(|| self.get_inner(key))
    }

    fn get_inner(&self, key: &str) -> Result<Option<Document>> {
        let mut r = self.read()?;
        let Some(id) = r.doc_id(key)? else {
            return Ok(None);
        };
        let d = r.doc(id)?.ok_or_else(|| missing(format!("doc {id}")))?;
        let content = d.content.clone();
        let rec = &d.record;
        Ok(Some(Document {
            id,
            key: rec.key.clone(),
            content,
            meta: rec.meta.iter().cloned().collect(),
            version: rec.version,
            created_ms: rec.created_ms,
            updated_ms: rec.updated_ms,
            format: format_from_u8(rec.format).as_str(),
            chunk_count: rec.chunks.len(),
        }))
    }

    /// The chunks of a document, in order.
    pub fn chunks(&self, key: &str) -> Result<Vec<ChunkInfo>> {
        guard(|| self.chunks_inner(key))
    }

    fn chunks_inner(&self, key: &str) -> Result<Vec<ChunkInfo>> {
        let mut r = self.read()?;
        let Some(id) = r.doc_id(key)? else {
            return Ok(Vec::new());
        };
        let chunk_ids = r
            .doc(id)?
            .ok_or_else(|| missing(format!("doc {id}")))?
            .record
            .chunks
            .clone();
        let mut out = Vec::with_capacity(chunk_ids.len());
        for cid in chunk_ids {
            let ch = r
                .chunk(cid)?
                .ok_or_else(|| missing(format!("chunk {cid}")))?;
            let d = r.doc(id)?.expect("loaded above");
            out.push(ChunkInfo {
                chunk_id: cid,
                ordinal: ch.ordinal,
                range: ch.start as usize..ch.end as usize,
                overlap_start: ch.overlap_start as usize,
                heading_path: d.heading_path(ch.section),
                tokens: ch.tokens,
            });
        }
        Ok(out)
    }

    /// Documents whose metadata matches `filter` (all documents for `None`),
    /// in id order, without their content.
    pub fn scan(&self, filter: Option<&Filter>) -> Result<Vec<DocInfo>> {
        guard(|| self.scan_inner(filter))
    }

    fn scan_inner(&self, filter: Option<&Filter>) -> Result<Vec<DocInfo>> {
        let rx = self.db.store.begin_read()?;
        let c = self.inner.id;
        let mut out = Vec::new();
        for row in rx.open_table(DOCS)?.range((c, 0)..=(c, u64::MAX))? {
            let (k, v) = row?;
            let rec: DocRecord = decode(v.value(), "document")?;
            let meta: Meta = rec.meta.iter().cloned().collect();
            if filter.is_none_or(|f| f.matches(&meta)) {
                out.push(DocInfo {
                    id: k.value().1,
                    key: rec.key,
                    meta,
                    version: rec.version,
                    created_ms: rec.created_ms,
                    updated_ms: rec.updated_ms,
                    content_len: rec.content_len,
                    chunk_count: rec.chunks.len(),
                });
            }
        }
        Ok(out)
    }

    /// The embedding inputs `content` would produce, one per chunk, in order.
    /// Use with [`crate::WriteTx::upsert_with_vectors`].
    pub fn embed_inputs(&self, content: &str, format_hint: Option<&str>) -> Vec<String> {
        let chunked = chunk_content(&self.inner, content, format_hint);
        chunked
            .chunks
            .iter()
            .map(|c| chunked.embed_text(content, c))
            .collect()
    }

    /// The chunk texts that need new vectors to upsert `doc`, as
    /// (hash, embedding input) pairs; chunks unchanged since the stored
    /// version are left out. Embed the texts, then pass the vectors keyed by
    /// hash to [`crate::WriteTx::upsert_with_embeddings`].
    pub fn plan_embeddings(&self, doc: &Doc) -> Result<Vec<(u64, String)>> {
        guard(|| self.plan_embeddings_inner(doc))
    }

    fn plan_embeddings_inner(&self, doc: &Doc) -> Result<Vec<(u64, String)>> {
        let chunked = chunk_content(&self.inner, &doc.content, doc.format.as_deref());
        let mut r = self.read()?;
        let mut old_hashes: HashMap<u64, usize> = HashMap::new();
        if let Some(id) = r.doc_id(&doc.key)? {
            let d = r.doc(id)?.ok_or_else(|| missing(format!("doc {id}")))?;
            let rec = &d.record;
            let meta: Vec<(String, MetaValue)> = doc.meta.clone().into_iter().collect();
            let unchanged = rec.content_hash == content_fingerprint(&doc.content)
                && rec.content_len == doc.content.len() as u64
                && rec.meta == meta
                && rec.format == format_to_u8(chunked.format);
            if unchanged {
                return Ok(Vec::new());
            }
            for cid in rec.chunks.clone() {
                if let Some(ch) = r.chunk(cid)? {
                    *old_hashes.entry(ch.hash).or_default() += 1;
                }
            }
        }
        let mut out: Vec<(u64, String)> = Vec::new();
        for ch in &chunked.chunks {
            match old_hashes.get_mut(&ch.hash) {
                Some(n) if *n > 0 => *n -= 1,
                _ if out.iter().any(|(h, _)| *h == ch.hash) => {}
                _ => out.push((ch.hash, chunked.embed_text(&doc.content, ch))),
            }
        }
        Ok(out)
    }

    /// Whether the collection has a keyword index for hybrid search.
    pub fn has_lexical(&self) -> bool {
        let _ = self.sync();
        self.inner.lexical.load(Ordering::Acquire)
    }

    /// Builds the keyword index for a collection created without one, from
    /// the stored text (no re-embedding). Returns false if it already had one.
    pub fn enable_lexical(&self) -> Result<bool> {
        let db = crate::Db {
            inner: self.db.clone(),
        };
        let mut tx = db.begin_write()?;
        let built = tx.enable_lexical(self)?;
        tx.commit()?;
        Ok(built)
    }

    /// Starts a vector search.
    pub fn search(&self, query: &[f32]) -> Search<'_> {
        Search {
            coll: self,
            query: query.to_vec(),
            text: None,
            k: 10,
            ef: None,
            filter: None,
        }
    }

    /// Embeds `text` and starts a search with it.
    pub fn search_text(&self, text: &str, embedder: &dyn Embedder) -> Result<Search<'_>> {
        let mut v = embedder.embed(&[text]).map_err(Error::Embedder)?;
        if v.len() != 1 {
            return Err(Error::Embedder(
                format!("embedder returned {} vectors for 1 text", v.len()).into(),
            ));
        }
        Ok(self.search(&v.remove(0)))
    }

    fn score(&self, dist: f32) -> f32 {
        match self.inner.schema.metric {
            Metric::Cosine => 1.0 - dist,
            Metric::Dot | Metric::L2 => -dist,
        }
    }
}

fn missing(what: String) -> Error {
    Error::Storage(vecfile_core::StorageError::Corrupt(format!(
        "{what} is referenced but missing"
    )))
}

/// A search being configured. Finish it with [`Search::chunks`],
/// [`Search::expand`], [`Search::section`] or [`Search::documents`].
pub struct Search<'c> {
    coll: &'c Collection,
    query: Vec<f32>,
    text: Option<String>,
    k: usize,
    ef: Option<usize>,
    filter: Option<Filter>,
}

impl Search<'_> {
    /// Results to return (default 10).
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Graph beam width; higher is slower and more accurate. Defaults to the
    /// collection's `ef_search`.
    pub fn ef(mut self, ef: usize) -> Self {
        self.ef = Some(ef);
        self
    }

    /// Only consider documents whose metadata matches.
    pub fn filter(mut self, f: Filter) -> Self {
        self.filter = Some(f);
        self
    }

    /// Makes this a hybrid search: also rank chunks by BM25 over `text`
    /// (usually the text the query vector was embedded from) and fuse both
    /// rankings with Reciprocal Rank Fusion. Exact words, identifiers and
    /// rare names then match even when the embedding misses them.
    ///
    /// Without a keyword index (see [`Collection::has_lexical`]), or when
    /// `text` has no searchable words, this stays a vector search.
    pub fn text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }

    /// The `n` best chunks and a reader over the same snapshot they came from.
    fn raw(&self, n: usize) -> Result<(Reader, Vec<Hit>)> {
        let c = &self.coll.inner;
        if self.query.len() != c.schema.dim {
            return Err(Error::Dimension {
                expected: c.schema.dim,
                got: self.query.len(),
            });
        }
        if self.query.iter().any(|x| !x.is_finite()) {
            return Err(crate::error::invalid(
                "query must not contain NaN or infinity",
            ));
        }
        let reader = self.coll.read()?;
        if self.coll.db.store.is_reader() {
            c.sync(&reader.rx)?;
        }
        let bitmap = match &self.filter {
            Some(f) => Some(f.eval(&reader.rx.open_table(META_INDEX)?, c.id)?),
            None => None,
        };
        if bitmap.as_ref().is_some_and(|b| b.is_empty()) || n == 0 {
            return Ok((reader, Vec::new()));
        }
        let lexical_query = self.text.as_deref().filter(|t| {
            c.lexical.load(Ordering::Acquire) && !crate::lexical::query_terms(t).is_empty()
        });
        // Each list contributes more candidates than requested, so a chunk
        // ranked moderately by both can surface in the fused top n.
        let pool = if lexical_query.is_some() {
            (n * 2).max(40)
        } else {
            n
        };

        let store = FileVectors::new(reader.rx.open_table(VECTORS)?, c.id);
        let ef = self.ef.unwrap_or(c.opts.hnsw.ef_search);
        let vector_hits: Vec<Neighbor> =
            c.index
                .read()
                .search(&self.query, pool, ef, bitmap.as_ref(), &store);
        if let Some(e) = store.take_error() {
            return Err(e.into());
        }
        let Some(text) = lexical_query else {
            let hits = vector_hits
                .iter()
                .map(|h| {
                    let s = self.coll.score(h.dist);
                    Hit {
                        id: h.id,
                        score: s,
                        vector: Some(s),
                        lexical: None,
                    }
                })
                .collect();
            return Ok((reader, hits));
        };

        let stats = reader
            .rx
            .open_table(LEXICAL)?
            .get(c.id)?
            .and_then(|v| crate::lexical::read_stats(v.value()))
            .unwrap_or_else(crate::lexical::empty_stats);
        let lexical_hits = crate::lexical::search(
            &reader.rx.open_table(POSTINGS)?,
            &reader.rx.open_table(TERM_DF)?,
            stats,
            c.id,
            text,
            bitmap.as_ref(),
            pool,
        )?;
        let mut fused: HashMap<u32, Hit> = HashMap::new();
        for (rank, h) in vector_hits.iter().enumerate() {
            let e = fused.entry(h.id).or_insert(Hit {
                id: h.id,
                score: 0.0,
                vector: None,
                lexical: None,
            });
            e.score += 1.0 / (RRF_K + rank as f32 + 1.0);
            e.vector = Some(self.coll.score(h.dist));
        }
        for (rank, &(id, s)) in lexical_hits.iter().enumerate() {
            let e = fused.entry(id).or_insert(Hit {
                id,
                score: 0.0,
                vector: None,
                lexical: None,
            });
            e.score += 1.0 / (RRF_K + rank as f32 + 1.0);
            e.lexical = Some(s);
        }
        let mut hits: Vec<Hit> = fused.into_values().collect();
        // Equal fused scores are common (rank 1 in one list, 2 in the other);
        // break ties by agreement, then keyword strength, then similarity.
        let key = |h: &Hit| {
            (
                h.vector.is_some() as u8 + h.lexical.is_some() as u8,
                h.lexical.unwrap_or(f32::MIN),
                h.vector.unwrap_or(f32::MIN),
            )
        };
        hits.sort_by(|a, b| {
            let (ka, kb) = (key(a), key(b));
            b.score
                .total_cmp(&a.score)
                .then(kb.0.cmp(&ka.0))
                .then(kb.1.total_cmp(&ka.1))
                .then(kb.2.total_cmp(&ka.2))
                .then(a.id.cmp(&b.id))
        });
        hits.truncate(n);
        Ok((reader, hits))
    }

    /// The best-matching chunks with their text and position.
    pub fn chunks(self) -> Result<Vec<ChunkHit>> {
        guard(|| self.chunks_inner())
    }

    fn chunks_inner(self) -> Result<Vec<ChunkHit>> {
        let (mut r, hits) = self.raw(self.k)?;
        let mut out = Vec::with_capacity(hits.len());
        for h in hits {
            if let Some(hit) = r.chunk_hit(&h)? {
                out.push(hit);
            }
        }
        Ok(out)
    }

    /// Each hit widened by `n` chunks on either side; passages of the same
    /// document that touch are merged.
    pub fn expand(self, n: u32) -> Result<Vec<Passage>> {
        guard(|| self.expand_inner(n))
    }

    fn expand_inner(self, n: u32) -> Result<Vec<Passage>> {
        let (mut r, hits) = self.raw(self.k)?;
        let mut spans: Vec<(u64, Range<u32>, f32)> = Vec::new();
        for h in hits {
            let Some(ch) = r.chunk(h.id)? else { continue };
            let Some(d) = r.doc(ch.doc)? else { continue };
            let last = d.record.chunks.len().saturating_sub(1) as u32;
            let span = ch.ordinal.saturating_sub(n)..(ch.ordinal + n).min(last) + 1;
            spans.push((ch.doc, span, h.score));
        }
        let merged = merge_spans(spans);
        let mut out = Vec::with_capacity(merged.len());
        for (doc, ords, score) in merged {
            out.push(r.passage(doc, ords, score, None)?);
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }

    /// The whole heading section each hit belongs to (the whole document for
    /// hits before the first heading), one passage per section.
    pub fn section(self) -> Result<Vec<Passage>> {
        guard(|| self.section_inner())
    }

    fn section_inner(self) -> Result<Vec<Passage>> {
        let (mut r, hits) = self.raw(self.k)?;
        let mut best: Vec<((u64, Option<u32>), f32)> = Vec::new();
        for h in hits {
            let Some(ch) = r.chunk(h.id)? else { continue };
            let key = (ch.doc, ch.section);
            let score = h.score;
            match best.iter_mut().find(|(k, _)| *k == key) {
                Some((_, s)) => *s = s.max(score),
                None => best.push((key, score)),
            }
        }
        let mut out = Vec::with_capacity(best.len());
        for ((doc, section), score) in best {
            let Some(d) = r.doc(doc)? else { continue };
            let range = match section.and_then(|s| d.record.sections.get(s as usize)) {
                Some(s) => s.range.0 as usize..s.range.1 as usize,
                None => 0..d.content.len(),
            };
            let chunk_ids = d.record.chunks.clone();
            let ords = r.ordinals_in(&chunk_ids, &range)?;
            out.push(r.passage(doc, ords, score, Some((range, section)))?);
        }
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        Ok(out)
    }

    /// Whole documents ranked by their best chunks.
    pub fn documents(self, agg: Agg) -> Result<Vec<DocHit>> {
        guard(|| self.documents_inner(agg))
    }

    fn documents_inner(self, agg: Agg) -> Result<Vec<DocHit>> {
        // Over-fetch chunks so k distinct documents are likely to appear.
        let n = (self.k * 8).max(50);
        let (mut r, hits) = self.raw(n)?;
        let mut by_doc: Vec<(u64, Vec<ChunkHit>)> = Vec::new();
        for h in hits {
            let Some(hit) = r.chunk_hit(&h)? else {
                continue;
            };
            match by_doc.iter_mut().find(|(d, _)| *d == hit.doc_id) {
                Some((_, v)) => v.push(hit),
                None => by_doc.push((hit.doc_id, vec![hit])),
            }
        }
        let mut out: Vec<DocHit> = by_doc
            .into_iter()
            .map(|(doc_id, chunks)| {
                let score = match agg {
                    Agg::MaxSim => chunks[0].score,
                    Agg::SumTop(t) => chunks.iter().take(t.max(1)).map(|c| c.score).sum(),
                };
                DocHit {
                    doc_id,
                    key: chunks[0].key.clone(),
                    score,
                    meta: chunks[0].meta.clone(),
                    chunks,
                }
            })
            .collect();
        out.sort_by(|a, b| b.score.total_cmp(&a.score));
        out.truncate(self.k);
        Ok(out)
    }
}

/// Merges spans of the same document that overlap or touch, keeping the
/// best score, in order of each document's best hit.
fn merge_spans(mut spans: Vec<(u64, Range<u32>, f32)>) -> Vec<(u64, Range<u32>, f32)> {
    spans.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.start.cmp(&b.1.start)));
    let mut out: Vec<(u64, Range<u32>, f32)> = Vec::new();
    for (doc, r, s) in spans {
        match out.last_mut() {
            Some((d, last, best)) if *d == doc && r.start <= last.end => {
                last.end = last.end.max(r.end);
                *best = best.max(s);
            }
            _ => out.push((doc, r, s)),
        }
    }
    out
}

/// A loaded document and its decompressed content.
struct LoadedDoc {
    record: DocRecord,
    content: String,
}

impl LoadedDoc {
    fn heading_path(&self, section: Option<u32>) -> Vec<String> {
        let mut path = Vec::new();
        let mut cur = section;
        while let Some(i) = cur {
            let Some(s) = self.record.sections.get(i as usize) else {
                break;
            };
            let text = self
                .content
                .get(s.heading.0 as usize..s.heading.1 as usize)
                .unwrap_or("");
            path.push(text.trim().to_owned());
            cur = s.parent.filter(|&p| p < i);
        }
        path.reverse();
        path
    }
}

/// Reads documents and chunks from one snapshot, caching documents.
struct Reader {
    rx: ReadTransaction,
    coll: u32,
    docs: HashMap<u64, Option<LoadedDoc>>,
}

impl Reader {
    fn doc_id(&self, key: &str) -> Result<Option<u64>> {
        Ok(self
            .rx
            .open_table(KEYS)?
            .get((self.coll, key))?
            .map(|v| v.value()))
    }

    fn doc(&mut self, id: u64) -> Result<Option<&LoadedDoc>> {
        if let Entry::Vacant(e) = self.docs.entry(id) {
            let loaded = match self.rx.open_table(DOCS)?.get((self.coll, id))? {
                None => None,
                Some(bytes) => {
                    let record: DocRecord = decode(bytes.value(), "document")?;
                    let blobs = self.rx.open_table(BLOBS)?;
                    let blob = blobs
                        .get((self.coll, id))?
                        .ok_or_else(|| missing(format!("content of doc {id}")))?;
                    let raw = decompress(blob.value(), record.content_len)?;
                    let content = String::from_utf8(raw).map_err(|_| {
                        vecfile_core::StorageError::Corrupt(format!(
                            "doc {id} content is not UTF-8"
                        ))
                    })?;
                    Some(LoadedDoc { record, content })
                }
            };
            e.insert(loaded);
        }
        Ok(self.docs[&id].as_ref())
    }

    fn chunk(&self, id: u32) -> Result<Option<ChunkRecord>> {
        match self.rx.open_table(CHUNKS)?.get((self.coll, id))? {
            Some(v) => Ok(Some(decode(v.value(), "chunk")?)),
            None => Ok(None),
        }
    }

    /// Ordinals of the chunks (from `chunk_ids`) that intersect `range`.
    fn ordinals_in(&self, chunk_ids: &[u32], range: &Range<usize>) -> Result<Range<u32>> {
        let mut lo = u32::MAX;
        let mut hi = 0;
        for &id in chunk_ids {
            let Some(ch) = self.chunk(id)? else { continue };
            if (ch.start as usize) < range.end && (ch.end as usize) > range.start {
                lo = lo.min(ch.ordinal);
                hi = hi.max(ch.ordinal + 1);
            }
        }
        Ok(if lo == u32::MAX { 0..0 } else { lo..hi })
    }

    fn chunk_hit(&mut self, h: &Hit) -> Result<Option<ChunkHit>> {
        // A chunk deleted after the index snapshot was taken is skipped.
        let Some(ch) = self.chunk(h.id)? else {
            return Ok(None);
        };
        let Some(d) = self.doc(ch.doc)? else {
            return Ok(None);
        };
        let range = ch.start as usize..ch.end as usize;
        let text = d
            .content
            .get(range.clone())
            .ok_or_else(|| missing(format!("range of chunk {}", h.id)))?
            .to_owned();
        Ok(Some(ChunkHit {
            doc_id: ch.doc,
            key: d.record.key.clone(),
            chunk_id: h.id,
            ordinal: ch.ordinal,
            range,
            score: h.score,
            vector_score: h.vector,
            lexical_score: h.lexical,
            text,
            heading_path: d.heading_path(ch.section),
            meta: d.record.meta.iter().cloned().collect(),
        }))
    }

    /// A passage over chunk ordinals `ords` of `doc`, or over an explicit
    /// byte range (sections).
    fn passage(
        &mut self,
        doc: u64,
        ords: Range<u32>,
        score: f32,
        explicit: Option<(Range<usize>, Option<u32>)>,
    ) -> Result<Passage> {
        let chunk_ids = self
            .doc(doc)?
            .ok_or_else(|| missing(format!("doc {doc}")))?
            .record
            .chunks
            .clone();
        let (range, section) = match explicit {
            Some((r, s)) => (r, s),
            None => {
                let first = self
                    .chunk(chunk_ids[ords.start as usize])?
                    .ok_or_else(|| missing("chunk".into()))?;
                let last = self
                    .chunk(chunk_ids[ords.end as usize - 1])?
                    .ok_or_else(|| missing("chunk".into()))?;
                (first.start as usize..last.end as usize, first.section)
            }
        };
        let d = self.doc(doc)?.expect("loaded above");
        Ok(Passage {
            doc_id: doc,
            key: d.record.key.clone(),
            text: d
                .content
                .get(range.clone())
                .ok_or_else(|| missing("passage range".into()))?
                .to_owned(),
            range,
            score,
            ordinals: ords,
            heading_path: d.heading_path(section),
            meta: d.record.meta.iter().cloned().collect(),
        })
    }
}
