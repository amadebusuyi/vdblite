//! The write transaction.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::MutexGuard;
use redb::{ReadableTable, WriteTransaction};
use roaring::RoaringBitmap;
use vecfile_chunk::{ApproxTokenizer, Chunked};
use vecfile_core::changes;
use vecfile_core::codec::decompress;
use vecfile_core::codec::{bitmap_bytes, compress, read_bitmap, vector_bytes};
use vecfile_core::meta::{self, ALL_KEY, MetaValue};
use vecfile_core::records::{
    Change, ChunkRecord, CollectionRecord, DocRecord, LexicalStats, SectionRecord, decode, encode,
};
use vecfile_core::tables::{
    BLOBS, CATALOG, CHANGES, CHUNK_TERMS, CHUNKS, DEAD, DOCS, KEYS, LEXICAL, META_INDEX, PENDING,
    POSTINGS, TERM_DF, VECTORS,
};

use crate::collection::{CollInner, Collection, format_to_u8};
use crate::db::{Db, DbRef};
use crate::doc::{Doc, Embedder, Upserted};
use crate::error::{Error, Result, invalid};
use crate::lexical;
use crate::maintenance;

/// Index changes to apply once the transaction commits.
struct Changes {
    coll: Arc<CollInner>,
    inserts: HashMap<u32, Vec<f32>>,
    deletes: Vec<u32>,
}

/// The single write transaction. Dropping it without [`WriteTx::commit`]
/// discards every change.
pub struct WriteTx<'db> {
    db: &'db Db,
    tx: Option<WriteTransaction>,
    records: HashMap<u32, CollectionRecord>,
    bitmaps: HashMap<(u32, Vec<u8>), RoaringBitmap>,
    changes: HashMap<u32, Changes>,
    /// Keyword-index totals per collection touched (`None`: no keyword index).
    lexical: HashMap<u32, Option<LexicalStats>>,
    /// Set when a write failed part-way; such a transaction can't commit.
    poisoned: bool,
    // Declared last so the lock is released after the transaction drops.
    _writer: MutexGuard<'db, ()>,
}

enum Vectors<'a> {
    Embed(&'a dyn Embedder),
    Given(Vec<Vec<f32>>),
    /// Vectors for the chunks that need them, keyed by chunk hash, as
    /// planned by [`Collection::plan_embeddings`].
    ByHash(HashMap<u64, Vec<f32>>),
}

/// Identifies a document's content *as chunked*: the chunker version is the
/// hash seed, so a chunker change makes every document look changed once and
/// get re-chunked (version 0 matches files written before versioning).
pub(crate) fn content_fingerprint(content: &str) -> u64 {
    xxhash_rust::xxh3::xxh3_64_with_seed(content.as_bytes(), vecfile_chunk::VERSION)
}

pub(crate) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis() as u64)
}

/// Chunks content the way the collection's schema says to.
pub(crate) fn chunk_content(coll: &CollInner, content: &str, hint: Option<&str>) -> Chunked {
    let format = vecfile_chunk::detect(content, hint);
    match coll.schema.chunking.config() {
        Some(cfg) => vecfile_chunk::chunk(content, format, &cfg),
        None => vecfile_chunk::single(content, format, &ApproxTokenizer),
    }
}

impl<'db> WriteTx<'db> {
    pub(crate) fn begin(db: &'db Db) -> Result<Self> {
        let store = db.inner.store.writer()?;
        let writer = db.inner.writer.lock();
        let tx = store.begin_write()?;
        Ok(WriteTx {
            db,
            tx: Some(tx),
            records: HashMap::new(),
            bitmaps: HashMap::new(),
            changes: HashMap::new(),
            lexical: HashMap::new(),
            poisoned: false,
            _writer: writer,
        })
    }

    fn tx(&self) -> &WriteTransaction {
        self.tx.as_ref().expect("transaction is open until commit")
    }

    fn check_coll(&self, coll: &Collection) -> Result<()> {
        if !DbRef::ptr_eq(&coll.db, &self.db.inner) {
            return Err(invalid("collection belongs to a different database"));
        }
        if !self
            .db
            .inner
            .collections
            .read()
            .contains_key(&coll.inner.name)
        {
            return Err(Error::NoSuchCollection(coll.inner.name.clone()));
        }
        Ok(())
    }

    fn record(&mut self, coll: &CollInner) -> Result<&mut CollectionRecord> {
        if !self.records.contains_key(&coll.id) {
            let table = self.tx().open_table(CATALOG)?;
            let bytes = table
                .get(coll.name.as_str())?
                .ok_or_else(|| Error::NoSuchCollection(coll.name.clone()))?;
            let rec: CollectionRecord = decode(bytes.value(), "collection record")?;
            drop(bytes);
            drop(table);
            self.records.insert(coll.id, rec);
        }
        Ok(self.records.get_mut(&coll.id).expect("inserted above"))
    }

    fn bitmap(&mut self, coll: u32, key: &[u8]) -> Result<&mut RoaringBitmap> {
        let k = (coll, key.to_vec());
        if !self.bitmaps.contains_key(&k) {
            let table = self.tx().open_table(META_INDEX)?;
            let b = match table.get((coll, key))? {
                Some(v) => read_bitmap(v.value())?,
                None => RoaringBitmap::new(),
            };
            drop(table);
            self.bitmaps.insert(k.clone(), b);
        }
        Ok(self.bitmaps.get_mut(&k).expect("inserted above"))
    }

    fn changes(&mut self, coll: &Arc<CollInner>) -> &mut Changes {
        self.changes.entry(coll.id).or_insert_with(|| Changes {
            coll: coll.clone(),
            inserts: HashMap::new(),
            deletes: Vec::new(),
        })
    }

    /// The collection's keyword-index totals, or `None` if it has no index.
    fn lex_stats(&mut self, c: &CollInner) -> Result<Option<LexicalStats>> {
        if !self.lexical.contains_key(&c.id) {
            let stats = match self.tx().open_table(LEXICAL)?.get(c.id)? {
                Some(v) => lexical::read_stats(v.value()),
                None => None,
            };
            self.lexical.insert(c.id, stats);
        }
        Ok(self.lexical[&c.id])
    }

    fn lex_tables(&self) -> Result<lexical::Tables<'_>> {
        let tx = self.tx();
        Ok(lexical::Tables {
            postings: tx.open_table(POSTINGS)?,
            df: tx.open_table(TERM_DF)?,
            chunk_terms: tx.open_table(CHUNK_TERMS)?,
        })
    }

    /// Builds a keyword index for a collection created without one, from the
    /// text already stored (nothing is re-embedded). Returns false if it
    /// already had one.
    pub fn enable_lexical(&mut self, coll: &Collection) -> Result<bool> {
        let r = self.enable_lexical_inner(coll);
        self.poison_on_storage_error(r)
    }

    fn enable_lexical_inner(&mut self, coll: &Collection) -> Result<bool> {
        self.check_coll(coll)?;
        let c = &coll.inner;
        if self.lex_stats(c)?.is_some() {
            return Ok(false);
        }
        // Drop whatever an older layout left behind, then rebuild.
        {
            let cid = c.id;
            let tx = self.tx();
            tx.open_table(POSTINGS)?
                .retain_in((cid, "", 0)..(cid + 1, "", 0), |_, _| false)?;
            tx.open_table(TERM_DF)?
                .retain_in((cid, "")..(cid + 1, ""), |_, _| false)?;
            tx.open_table(CHUNK_TERMS)?
                .retain_in((cid, 0)..=(cid, u32::MAX), |_, _| false)?;
        }
        let mut stats = lexical::empty_stats();
        let docs: Vec<DocRecord> = {
            let table = self.tx().open_table(DOCS)?;
            let rows = table.range((c.id, 0)..=(c.id, u64::MAX))?;
            rows.map(|row| -> Result<DocRecord> { Ok(decode(row?.1.value(), "document")?) })
                .collect::<Result<_>>()?
        };
        {
            let mut t = self.lex_tables()?;
            let tx = self.tx.as_ref().expect("open");
            let blobs = tx.open_table(BLOBS)?;
            let chunks = tx.open_table(CHUNKS)?;
            let keys = tx.open_table(KEYS)?;
            for d in docs {
                let Some(id) = keys.get((c.id, d.key.as_str()))?.map(|v| v.value()) else {
                    continue;
                };
                let Some(blob) = blobs.get((c.id, id))? else {
                    continue;
                };
                let content =
                    String::from_utf8(decompress(blob.value(), d.content_len)?).unwrap_or_default();
                for &chunk in &d.chunks {
                    let Some(bytes) = chunks.get((c.id, chunk))? else {
                        continue;
                    };
                    let ch: ChunkRecord = decode(bytes.value(), "chunk")?;
                    let text = lexical::chunk_text(
                        &content,
                        &d.sections,
                        ch.section,
                        ch.start as usize..ch.end as usize,
                    );
                    lexical::index_chunk(&mut t, c.id, chunk, &text, &mut stats)?;
                }
            }
        }
        self.lexical.insert(c.id, Some(stats));
        Ok(true)
    }

    /// Inserts or replaces a document, chunking its content and embedding
    /// each chunk. Chunks whose text is unchanged since the last version keep
    /// their vectors, so a small edit re-embeds only what it touched.
    pub fn upsert(
        &mut self,
        coll: &Collection,
        doc: Doc,
        embedder: &dyn Embedder,
    ) -> Result<Upserted> {
        let r = self.upsert_inner(coll, doc, Vectors::Embed(embedder));
        self.poison_on_storage_error(r)
    }

    /// Inserts or replaces a document with caller-supplied vectors, one per
    /// chunk in order. [`Collection::embed_inputs`] shows the chunks the
    /// content will produce.
    pub fn upsert_with_vectors(
        &mut self,
        coll: &Collection,
        doc: Doc,
        vectors: Vec<Vec<f32>>,
    ) -> Result<Upserted> {
        let r = self.upsert_inner(coll, doc, Vectors::Given(vectors));
        self.poison_on_storage_error(r)
    }

    /// Inserts or replaces a document using vectors computed outside the
    /// transaction: run [`Collection::plan_embeddings`], embed the texts it
    /// returns, and pass the vectors keyed by the planned hashes. Chunks that
    /// are unchanged keep their vectors, as with [`WriteTx::upsert`].
    ///
    /// Fails with [`Error::StaleEmbeddings`] if the document changed since
    /// planning and a needed vector is missing; plan again and retry.
    pub fn upsert_with_embeddings(
        &mut self,
        coll: &Collection,
        doc: Doc,
        embeddings: HashMap<u64, Vec<f32>>,
    ) -> Result<Upserted> {
        let r = self.upsert_inner(coll, doc, Vectors::ByHash(embeddings));
        self.poison_on_storage_error(r)
    }

    /// Validation errors leave the transaction untouched, but a storage
    /// error may strike between writes, so it bars the transaction from
    /// committing.
    fn poison_on_storage_error<T>(&mut self, r: Result<T>) -> Result<T> {
        if matches!(r, Err(Error::Storage(_))) {
            self.poisoned = true;
        }
        r
    }

    fn upsert_inner(
        &mut self,
        coll: &Collection,
        doc: Doc,
        vectors: Vectors<'_>,
    ) -> Result<Upserted> {
        self.check_coll(coll)?;
        let c = &coll.inner;
        let cid = c.id;
        if doc.key.is_empty() {
            return Err(invalid("document key must not be empty"));
        }
        if let Some(f) = doc.meta.keys().find(|f| !meta::valid_field(f)) {
            return Err(invalid(format!("invalid metadata field name {f:?}")));
        }

        let existing = self
            .tx()
            .open_table(KEYS)?
            .get((cid, doc.key.as_str()))?
            .map(|v| v.value());
        let old: Option<DocRecord> = match existing {
            Some(id) => {
                let docs = self.tx().open_table(DOCS)?;
                let bytes = docs.get((cid, id))?.ok_or_else(|| {
                    vecfile_core::StorageError::Corrupt(format!(
                        "key {:?} points at missing doc {id}",
                        doc.key
                    ))
                })?;
                Some(decode(bytes.value(), "document")?)
            }
            None => None,
        };

        let chunked = chunk_content(c, &doc.content, doc.format.as_deref());
        let content_hash = content_fingerprint(&doc.content);
        let meta_pairs: Vec<(String, MetaValue)> = doc.meta.into_iter().collect();
        let format = format_to_u8(chunked.format);

        // Unchanged content and metadata: nothing to do. Caller-supplied vectors
        // are always written, since they may come from a different model.
        if let Some(o) = &old
            && o.content_hash == content_hash
            && o.content_len == doc.content.len() as u64
            && o.meta == meta_pairs
            && o.format == format
            && !matches!(vectors, Vectors::Given(_))
        {
            return Ok(Upserted {
                doc_id: existing.expect("old implies id"),
                version: o.version,
                chunks: o.chunks.len(),
                embedded: 0,
                changed: false,
            });
        }

        // Reuse old chunk ids (and vectors) whose embedding input is unchanged.
        let mut reusable: HashMap<u64, Vec<u32>> = HashMap::new();
        if let Some(o) = &old
            && !matches!(vectors, Vectors::Given(_))
        {
            let chunks = self.tx().open_table(CHUNKS)?;
            for &id in o.chunks.iter().rev() {
                if let Some(bytes) = chunks.get((cid, id))? {
                    let rec: ChunkRecord = decode(bytes.value(), "chunk")?;
                    reusable.entry(rec.hash).or_default().push(id);
                }
            }
        }
        let mut assigned: Vec<Option<u32>> = chunked
            .chunks
            .iter()
            .map(|ch| reusable.get_mut(&ch.hash).and_then(Vec::pop))
            .collect();

        // Embed (or take) vectors for the chunks that need them.
        let need: Vec<usize> = (0..assigned.len())
            .filter(|&i| assigned[i].is_none())
            .collect();
        let new_vectors: Vec<Vec<f32>> = match vectors {
            Vectors::Given(v) => {
                if v.len() != chunked.chunks.len() {
                    return Err(invalid(format!(
                        "{} vectors for {} chunks",
                        v.len(),
                        chunked.chunks.len()
                    )));
                }
                v
            }
            Vectors::Embed(e) if !need.is_empty() => {
                let texts: Vec<String> = need
                    .iter()
                    .map(|&i| chunked.embed_text(&doc.content, &chunked.chunks[i]))
                    .collect();
                let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
                let out = e.embed(&refs).map_err(Error::Embedder)?;
                if out.len() != need.len() {
                    return Err(Error::Embedder(
                        format!(
                            "embedder returned {} vectors for {} texts",
                            out.len(),
                            need.len()
                        )
                        .into(),
                    ));
                }
                out
            }
            Vectors::Embed(_) => Vec::new(),
            Vectors::ByHash(map) => need
                .iter()
                .map(|&i| {
                    let hash = chunked.chunks[i].hash;
                    map.get(&hash).cloned().ok_or(Error::StaleEmbeddings {
                        key: doc.key.clone(),
                    })
                })
                .collect::<Result<_>>()?,
        };
        for v in &new_vectors {
            if v.len() != c.schema.dim {
                return Err(Error::Dimension {
                    expected: c.schema.dim,
                    got: v.len(),
                });
            }
            if v.iter().any(|x| !x.is_finite()) {
                return Err(invalid("vectors must not contain NaN or infinity"));
            }
        }

        let need_count = need.len();
        let first_new = self.record(c)?.next_chunk;
        if first_new.checked_add(need_count as u32).is_none() {
            return Err(invalid("collection has run out of chunk ids (2^32)"));
        }

        // Retire old chunks that were not reused. No validation error can
        // happen past this point; storage errors poison the transaction.
        let reused: std::collections::HashSet<u32> = assigned.iter().flatten().copied().collect();
        let old_chunks: Vec<u32> = old.as_ref().map(|o| o.chunks.clone()).unwrap_or_default();
        for &id in old_chunks.iter().filter(|id| !reused.contains(id)) {
            self.delete_chunk(c, id)?;
        }
        if let Some(o) = &old {
            self.unindex_meta(cid, &o.meta, &old_chunks)?;
        }

        // Allocate and write new chunks.
        let rec = self.record(c)?;
        let doc_id = match existing {
            Some(id) => id,
            None => {
                let id = rec.next_doc;
                rec.next_doc += 1;
                id
            }
        };
        rec.next_chunk = first_new + need_count as u32;
        // Caller-supplied vectors skip reuse, so they always fill every slot here.
        let mut fresh: Vec<(u32, Vec<f32>)> = Vec::with_capacity(need_count);
        let empty = assigned.iter_mut().filter(|s| s.is_none());
        for ((id, slot), v) in (first_new..).zip(empty).zip(new_vectors) {
            *slot = Some(id);
            fresh.push((id, v));
        }
        {
            let tx = self.tx.as_ref().expect("open");
            let mut vt = tx.open_table(VECTORS)?;
            let mut pt = tx.open_table(PENDING)?;
            for (id, v) in &fresh {
                vt.insert((cid, *id), vector_bytes(v))?;
                pt.insert((cid, *id), ())?;
            }
        }
        let chunk_ids: Vec<u32> = assigned
            .iter()
            .map(|s| s.expect("every chunk assigned"))
            .collect();
        {
            let tx = self.tx.as_ref().expect("open");
            let mut ct = tx.open_table(CHUNKS)?;
            for (ch, &id) in chunked.chunks.iter().zip(&chunk_ids) {
                let r = ChunkRecord {
                    doc: doc_id,
                    ordinal: ch.ordinal,
                    start: ch.range.start as u64,
                    end: ch.range.end as u64,
                    overlap_start: ch.overlap_start as u64,
                    section: ch.section,
                    tokens: ch.tokens,
                    hash: ch.hash,
                };
                ct.insert((cid, id), encode(&r).as_slice())?;
            }
        }
        if let Some(mut stats) = self.lex_stats(c)? {
            let fresh_ids: std::collections::HashSet<u32> =
                fresh.iter().map(|(id, _)| *id).collect();
            let mut t = self.lex_tables()?;
            for (ch, &id) in chunked.chunks.iter().zip(&chunk_ids) {
                if fresh_ids.contains(&id) {
                    let path = chunked.heading_path(&doc.content, ch).join(" ");
                    let text = format!("{path}\n{}", &doc.content[ch.range.clone()]);
                    lexical::index_chunk(&mut t, cid, id, &text, &mut stats)?;
                }
            }
            drop(t);
            self.lexical.insert(cid, Some(stats));
        }
        let changes = self.changes(c);
        for (id, v) in fresh {
            changes.deletes.retain(|&d| d != id);
            changes.inserts.insert(id, v);
        }
        self.index_meta(cid, &meta_pairs, &chunk_ids)?;

        let now = now_ms();
        let version = old.as_ref().map_or(1, |o| o.version + 1);
        let record = DocRecord {
            key: doc.key.clone(),
            meta: meta_pairs,
            format,
            content_hash,
            content_len: doc.content.len() as u64,
            version,
            created_ms: old.as_ref().map_or(now, |o| o.created_ms),
            updated_ms: now,
            chunks: chunk_ids.clone(),
            sections: chunked
                .sections
                .iter()
                .map(|s| SectionRecord {
                    level: s.level,
                    heading: (s.heading.start as u64, s.heading.end as u64),
                    range: (s.range.start as u64, s.range.end as u64),
                    parent: s.parent,
                })
                .collect(),
        };
        {
            let tx = self.tx.as_ref().expect("open");
            tx.open_table(DOCS)?
                .insert((cid, doc_id), encode(&record).as_slice())?;
            tx.open_table(BLOBS)?
                .insert((cid, doc_id), compress(doc.content.as_bytes()).as_slice())?;
            if existing.is_none() {
                tx.open_table(KEYS)?
                    .insert((cid, doc.key.as_str()), doc_id)?;
            }
        }
        Ok(Upserted {
            doc_id,
            version,
            chunks: chunk_ids.len(),
            embedded: need_count,
            changed: true,
        })
    }

    /// Deletes a document. Returns false if no document has this key.
    pub fn delete(&mut self, coll: &Collection, key: &str) -> Result<bool> {
        let r = self.delete_inner(coll, key);
        self.poison_on_storage_error(r)
    }

    fn delete_inner(&mut self, coll: &Collection, key: &str) -> Result<bool> {
        self.check_coll(coll)?;
        let c = &coll.inner;
        let cid = c.id;
        let Some(id) = self
            .tx()
            .open_table(KEYS)?
            .get((cid, key))?
            .map(|v| v.value())
        else {
            return Ok(false);
        };
        let rec: DocRecord = {
            let docs = self.tx().open_table(DOCS)?;
            let bytes = docs.get((cid, id))?.ok_or_else(|| {
                vecfile_core::StorageError::Corrupt(format!(
                    "key {key:?} points at missing doc {id}"
                ))
            })?;
            decode(bytes.value(), "document")?
        };
        for &chunk in &rec.chunks {
            self.delete_chunk(c, chunk)?;
        }
        self.unindex_meta(cid, &rec.meta, &rec.chunks)?;
        let tx = self.tx();
        tx.open_table(DOCS)?.remove((cid, id))?;
        tx.open_table(BLOBS)?.remove((cid, id))?;
        tx.open_table(KEYS)?.remove((cid, key))?;
        Ok(true)
    }

    fn delete_chunk(&mut self, c: &Arc<CollInner>, id: u32) -> Result<()> {
        let cid = c.id;
        if let Some(mut stats) = self.lex_stats(c)? {
            let mut t = self.lex_tables()?;
            lexical::unindex_chunk(&mut t, cid, id, &mut stats)?;
            drop(t);
            self.lexical.insert(cid, Some(stats));
        }
        let was_pending = {
            let tx = self.tx();
            tx.open_table(VECTORS)?.remove((cid, id))?;
            tx.open_table(CHUNKS)?.remove((cid, id))?;
            tx.open_table(PENDING)?.remove((cid, id))?.is_some()
        };
        let inserted_here = self.changes(c).inserts.remove(&id).is_some();
        if inserted_here {
            return Ok(());
        }
        if !was_pending && let Some(seg) = c.index.read().sealed_segment_of(id) {
            self.tx().open_table(DEAD)?.insert((cid, id), seg)?;
        }
        self.changes(c).deletes.push(id);
        Ok(())
    }

    fn index_meta(&mut self, cid: u32, m: &[(String, MetaValue)], chunks: &[u32]) -> Result<()> {
        let ids = || chunks.iter().copied();
        self.bitmap(cid, ALL_KEY)?.extend(ids());
        for (field, value) in m {
            for key in meta::index_keys(field, value) {
                self.bitmap(cid, &key)?.extend(ids());
            }
        }
        Ok(())
    }

    fn unindex_meta(&mut self, cid: u32, m: &[(String, MetaValue)], chunks: &[u32]) -> Result<()> {
        let remove = |b: &mut RoaringBitmap| {
            for &id in chunks {
                b.remove(id);
            }
        };
        remove(self.bitmap(cid, ALL_KEY)?);
        for (field, value) in m {
            for key in meta::index_keys(field, value) {
                remove(self.bitmap(cid, &key)?);
            }
        }
        Ok(())
    }

    /// Makes every change durable, then updates the in-memory indexes.
    ///
    /// Sealing the write buffer into a graph happens in the background
    /// afterwards. Only a buffer that has outgrown a full segment (writes
    /// outpacing sealing) is sealed before this returns.
    pub fn commit(mut self) -> Result<()> {
        if self.poisoned {
            return Err(invalid(
                "an earlier write in this transaction failed; it was rolled back",
            ));
        }
        let tx = self.tx.take().expect("commit runs once");
        {
            let mut catalog = tx.open_table(CATALOG)?;
            for rec in self.records.values() {
                catalog.insert(rec.name.as_str(), encode(rec).as_slice())?;
            }
            let mut lex = tx.open_table(LEXICAL)?;
            for (cid, stats) in &self.lexical {
                if let Some(stats) = stats {
                    lex.insert(*cid, encode(stats).as_slice())?;
                }
            }
            let mut index = tx.open_table(META_INDEX)?;
            for ((cid, key), bitmap) in &self.bitmaps {
                if bitmap.is_empty() {
                    index.remove((*cid, key.as_slice()))?;
                } else {
                    index.insert((*cid, key.as_slice()), bitmap_bytes(bitmap).as_slice())?;
                }
            }
            // The index changes below, for processes reading the file.
            let mut log = tx.open_table(CHANGES)?;
            for ch in self.changes.values() {
                if ch.deletes.is_empty() && ch.inserts.is_empty() {
                    continue;
                }
                let mut inserts: Vec<u32> = ch.inserts.keys().copied().collect();
                inserts.sort_unstable();
                let change = Change::Write {
                    deletes: ch.deletes.clone(),
                    inserts,
                };
                changes::append(&mut log, ch.coll.id, &change)?;
            }
        }
        tx.commit()?;

        for (cid, stats) in &self.lexical {
            if stats.is_some()
                && let Some(c) = self
                    .db
                    .inner
                    .collections
                    .read()
                    .values()
                    .find(|c| c.id == *cid)
            {
                c.lexical.store(true, std::sync::atomic::Ordering::Release);
            }
        }
        let changes: Vec<Changes> = self.changes.drain().map(|(_, ch)| ch).collect();
        for ch in &changes {
            let mut idx = ch.coll.index.write();
            for &id in &ch.deletes {
                idx.delete(id);
            }
            for (&id, v) in &ch.inserts {
                idx.insert(id, v);
            }
        }
        // Let other writers (and maintenance) in before waiting on anything.
        let db = self.db;
        drop(self);
        let mut due = false;
        for ch in &changes {
            let (backlog, ready) = {
                let idx = ch.coll.index.read();
                let over = idx.mutable().len() >= idx.config().max_segment;
                (over, idx.should_seal() || idx.needs_compaction(false))
            };
            if backlog {
                maintenance::seal_backlog(&db.inner, &ch.coll)
                    .map_err(|e| Error::Maintenance(Box::new(e)))?;
            }
            due |= ready;
        }
        if due {
            db.inner.worker.request(&db.inner);
        }
        Ok(())
    }

    /// Discards every change (the same as dropping the transaction).
    pub fn rollback(mut self) -> Result<()> {
        if let Some(tx) = self.tx.take() {
            tx.abort()?;
        }
        Ok(())
    }
}
