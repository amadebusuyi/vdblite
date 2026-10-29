//! Node.js bindings for vdblite (napi-rs).
//!
//! This is the native layer; `index.js` wraps it with the friendlier API
//! (embedding functions, transactions, retries). Anything that touches the
//! file runs on the libuv thread pool and returns a Promise, so a commit that
//! seals a segment never blocks the event loop.
//!
//! Errors carry a stable code as a `CODE: message` prefix, which `index.js`
//! turns into `err.code`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{Value, json};
use vdblite::{Agg, Chunking, Db, Doc, Filter, OpenOptions};
use vdblite_bindings as shared;

// ---------------------------------------------------------------- errors

fn to_napi(e: vdblite::Error) -> napi::Error {
    napi::Error::from_reason(format!("{}: {e}", shared::code_of(&e)))
}

fn invalid(msg: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(format!("INVALID_ARGUMENT: {msg}"))
}

trait OrNapi<T> {
    fn or_napi(self) -> Result<T>;
}

impl<T> OrNapi<T> for std::result::Result<T, vdblite::Error> {
    fn or_napi(self) -> Result<T> {
        self.map_err(to_napi)
    }
}

// ---------------------------------------------------------------- JSON

// Metadata, filters and result shapes are shared with the C ABI
// (bindings/shared), so every language speaks the same JSON.

fn parse_filter(v: &Value) -> Result<Filter> {
    shared::parse_filter(v).map_err(invalid)
}

// ---------------------------------------------------------------- input types

#[napi(object)]
pub struct ChunkingOptions {
    /// "structural" (default) or "none" (each document is one chunk).
    pub kind: Option<String>,
    pub target_tokens: Option<u32>,
    pub overlap_tokens: Option<u32>,
}

#[napi(object)]
pub struct CollectionOptions {
    pub dim: u32,
    /// "cosine" (default), "dot" or "l2".
    pub metric: Option<String>,
    /// "int8" (default: int8 graph walk, f16 re-ranking), "binary" (1-bit
    /// graph walk, f16 re-ranking) or "none" (f32).
    pub quantization: Option<String>,
    pub chunking: Option<ChunkingOptions>,
    pub seal_threshold: Option<u32>,
    /// Keep a keyword index for hybrid search. Default true.
    pub lexical: Option<bool>,
    pub m: Option<u32>,
    pub ef_construction: Option<u32>,
    pub ef_search: Option<u32>,
}

#[napi(object)]
pub struct DocInput {
    pub key: String,
    pub content: String,
    pub meta: Option<Value>,
    pub format: Option<String>,
}

#[napi(object)]
pub struct PlannedVector {
    pub hash: String,
    pub vector: Float32Array,
}

#[napi(object)]
pub struct WriteOp {
    /// "upsert" or "delete".
    pub op: String,
    pub collection: String,
    pub key: String,
    pub content: Option<String>,
    pub meta: Option<Value>,
    pub format: Option<String>,
    /// Vectors for planned chunks, from `planEmbeddings`.
    pub embeddings: Option<Vec<PlannedVector>>,
    /// One vector per chunk, in order (bypasses chunk reuse).
    pub vectors: Option<Vec<Float32Array>>,
}

#[napi(object)]
pub struct BulkLoadOptions {
    /// Chunks between durable commits in a file opened with
    /// `openExclusive` (default `DEFAULT_CHECKPOINT`).
    pub checkpoint_every: Option<f64>,
}

#[napi(object)]
pub struct BulkDoc {
    pub key: String,
    pub content: String,
    pub meta: Option<Value>,
    pub format: Option<String>,
    /// One vector per chunk, in order.
    pub vectors: Vec<Float32Array>,
}

#[napi(object)]
pub struct SearchOptions {
    pub k: Option<u32>,
    pub ef: Option<u32>,
    pub filter: Option<Value>,
    /// Query text for hybrid search (BM25 fused with the vector ranking).
    pub text: Option<String>,
}

#[napi(object)]
pub struct OpenInput {
    /// Bytes of file pages to cache in this process.
    pub cache_bytes: Option<f64>,
}

#[napi(object)]
pub struct IndexStats {
    pub chunks: u32,
    pub sealed_segments: u32,
    pub buffered: u32,
    pub memory_bytes: f64,
    /// The last background maintenance failure, until reported.
    pub maintenance_error: Option<String>,
}

#[napi(object)]
pub struct CollectionInfo {
    pub name: String,
    pub dim: u32,
    pub metric: String,
    pub quantization: String,
    pub chunking: String,
    pub target_tokens: Option<u32>,
    pub overlap_tokens: Option<u32>,
}

fn doc_from_input(
    key: String,
    content: String,
    meta: Option<&Value>,
    format: Option<String>,
) -> Result<Doc> {
    shared::doc_from_input(key, content, meta, format).map_err(invalid)
}

// ---------------------------------------------------------------- output JSON

/// A JSON result handed to JavaScript as a plain value (task results need
/// `TypeName`, which `serde_json::Value` lacks).
pub struct Json(Value);

impl TypeName for Json {
    fn type_name() -> &'static str {
        "any"
    }

    fn value_type() -> ValueType {
        ValueType::Unknown
    }
}

impl ToNapiValue for Json {
    unsafe fn to_napi_value(env: napi::sys::napi_env, val: Self) -> Result<napi::sys::napi_value> {
        // SAFETY: forwarded unchanged from our caller's contract.
        unsafe { Value::to_napi_value(env, val.0) }
    }
}

// ---------------------------------------------------------------- database

type Shared = Arc<RwLock<Option<Db>>>;

fn db_of(shared: &Shared) -> Result<Db> {
    shared
        .read()
        .map_err(|_| invalid("database lock poisoned"))?
        .clone()
        .ok_or_else(|| napi::Error::from_reason("CLOSED: the database is closed"))
}

/// An open vdblite database file.
#[napi]
pub struct Database {
    db: Shared,
    path: String,
}

#[napi]
impl Database {
    /// Opens (or creates) the database file at `path` for writing; other
    /// processes may open it read-only alongside.
    #[napi(factory)]
    pub fn open(path: String, options: Option<OpenInput>) -> Result<Database> {
        Self::open_as(path, OpenOptions::new(), options)
    }

    /// Opens for writing with no other process allowed, not even readers.
    #[napi(factory)]
    pub fn open_exclusive(path: String, options: Option<OpenInput>) -> Result<Database> {
        Self::open_as(path, OpenOptions::new().exclusive(true), options)
    }

    /// Opens an existing file read-only, alongside its writer.
    #[napi(factory)]
    pub fn open_read_only(path: String, options: Option<OpenInput>) -> Result<Database> {
        Self::open_as(path, OpenOptions::new().read_only(true), options)
    }

    fn open_as(path: String, mut o: OpenOptions, input: Option<OpenInput>) -> Result<Database> {
        if let Some(bytes) = input.and_then(|i| i.cache_bytes) {
            if !(bytes.is_finite() && bytes >= 0.0) {
                return Err(invalid("cacheBytes must be a non-negative number"));
            }
            o = o.cache_bytes(bytes as usize);
        }
        Self::wrap(Db::open_with(&path, &o).or_napi()?, path)
    }

    fn wrap(db: Db, path: String) -> Result<Database> {
        Ok(Database {
            db: Arc::new(RwLock::new(Some(db))),
            path,
        })
    }

    #[napi(getter)]
    pub fn read_only(&self) -> Result<bool> {
        Ok(db_of(&self.db)?.is_read_only())
    }

    #[napi(getter)]
    pub fn path(&self) -> String {
        self.path.clone()
    }

    /// Closes the database. The file lock is released once in-flight
    /// operations finish; later calls fail with code `CLOSED`.
    #[napi]
    pub fn close(&self) {
        if let Ok(mut g) = self.db.write() {
            g.take();
        }
    }

    #[napi(getter)]
    pub fn closed(&self) -> bool {
        self.db.read().map(|g| g.is_none()).unwrap_or(true)
    }

    #[napi]
    pub fn collections(&self) -> Result<Vec<String>> {
        db_of(&self.db)?.collections().or_napi()
    }

    #[napi]
    pub fn has_collection(&self, name: String) -> Result<bool> {
        Ok(db_of(&self.db)?.collection(&name).is_ok())
    }

    #[napi]
    pub fn create_collection(
        &self,
        name: String,
        options: CollectionOptions,
    ) -> Result<Collection> {
        let chunking = options.chunking.as_ref();
        let (schema, opts) = shared::CollectionOptions {
            dim: options.dim as usize,
            metric: options.metric.as_deref(),
            quantization: options.quantization.as_deref(),
            chunking: chunking.and_then(|c| c.kind.as_deref()),
            target_tokens: chunking.and_then(|c| c.target_tokens).map(|v| v as usize),
            overlap_tokens: chunking.and_then(|c| c.overlap_tokens).map(|v| v as usize),
            seal_threshold: options.seal_threshold.map(|v| v as usize),
            lexical: options.lexical,
            m: options.m.map(|v| v as usize),
            ef_construction: options.ef_construction.map(|v| v as usize),
            ef_search: options.ef_search.map(|v| v as usize),
        }
        .build()
        .map_err(invalid)?;
        db_of(&self.db)?
            .create_collection_with(&name, schema, opts)
            .or_napi()?;
        Ok(Collection {
            db: self.db.clone(),
            name,
        })
    }

    #[napi]
    pub fn collection(&self, name: String) -> Result<Collection> {
        db_of(&self.db)?.collection(&name).or_napi()?;
        Ok(Collection {
            db: self.db.clone(),
            name,
        })
    }

    #[napi]
    pub fn drop_collection(&self, name: String) -> Result<()> {
        db_of(&self.db)?.drop_collection(&name).or_napi()
    }

    /// Applies upserts and deletes atomically in one transaction.
    #[napi(ts_return_type = "Promise<any>")]
    pub fn write(&self, ops: Vec<WriteOp>) -> Result<AsyncTask<WriteTask>> {
        let db = db_of(&self.db)?;
        let ops = ops
            .into_iter()
            .map(|o| -> Result<Op> {
                match o.op.as_str() {
                    "delete" => Ok(Op::Delete {
                        coll: o.collection,
                        key: o.key,
                    }),
                    "upsert" => {
                        let doc = doc_from_input(
                            o.key,
                            o.content.unwrap_or_default(),
                            o.meta.as_ref(),
                            o.format,
                        )?;
                        let vectors = match (o.embeddings, o.vectors) {
                            (Some(e), None) => Vectors::ByHash(
                                e.into_iter()
                                    .map(|p| {
                                        Ok((
                                            shared::parse_hash(&p.hash).map_err(invalid)?,
                                            p.vector.to_vec(),
                                        ))
                                    })
                                    .collect::<Result<_>>()?,
                            ),
                            (None, Some(v)) => {
                                Vectors::Given(v.into_iter().map(|a| a.to_vec()).collect())
                            }
                            (None, None) => Vectors::ByHash(HashMap::new()),
                            (Some(_), Some(_)) => {
                                return Err(invalid("pass embeddings or vectors, not both"));
                            }
                        };
                        Ok(Op::Upsert {
                            coll: o.collection,
                            doc,
                            vectors,
                        })
                    }
                    other => Err(invalid(format!("unknown op {other:?} (upsert, delete)"))),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(AsyncTask::new(WriteTask { db, ops }))
    }

    /// Seals buffered chunks and merges small segments in every collection.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn optimize(&self) -> Result<AsyncTask<OptimizeTask>> {
        Ok(AsyncTask::new(OptimizeTask {
            db: db_of(&self.db)?,
        }))
    }

    /// Waits for background sealing and compaction to catch up.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn wait_for_maintenance(&self) -> Result<AsyncTask<WaitTask>> {
        Ok(AsyncTask::new(WaitTask {
            db: db_of(&self.db)?,
        }))
    }

    /// Shrinks a database file that no one has open.
    #[napi]
    pub fn compact_file(path: String) -> Result<bool> {
        Db::compact_file(&path).or_napi()
    }
}

enum Vectors {
    ByHash(HashMap<u64, Vec<f32>>),
    Given(Vec<Vec<f32>>),
}

enum Op {
    Upsert {
        coll: String,
        doc: Doc,
        vectors: Vectors,
    },
    Delete {
        coll: String,
        key: String,
    },
}

pub struct WriteTask {
    db: Db,
    ops: Vec<Op>,
}

impl Task for WriteTask {
    type Output = Value;
    type JsValue = Json;

    fn compute(&mut self) -> Result<Value> {
        let mut colls: HashMap<String, vdblite::Collection> = HashMap::new();
        let mut tx = self.db.begin_write().or_napi()?;
        let mut results = Vec::with_capacity(self.ops.len());
        for op in std::mem::take(&mut self.ops) {
            let name = match &op {
                Op::Upsert { coll, .. } | Op::Delete { coll, .. } => coll.clone(),
            };
            if !colls.contains_key(&name) {
                colls.insert(name.clone(), self.db.collection(&name).or_napi()?);
            }
            let coll = &colls[&name];
            results.push(match op {
                Op::Delete { key, .. } => {
                    let deleted = tx.delete(coll, &key).or_napi()?;
                    shared::deleted_json(&name, &key, deleted)
                }
                Op::Upsert { doc, vectors, .. } => {
                    let key = doc.key().to_owned();
                    let up = match vectors {
                        Vectors::ByHash(m) => tx.upsert_with_embeddings(coll, doc, m),
                        Vectors::Given(v) => tx.upsert_with_vectors(coll, doc, v),
                    }
                    .or_napi()?;
                    shared::upserted_json(&name, &key, &up)
                }
            });
        }
        let warning = match tx.commit() {
            Ok(()) => Value::Null,
            // The data is committed; only sealing an oversized write buffer
            // afterwards failed.
            Err(e @ vdblite::Error::Maintenance(_)) => Value::String(e.to_string()),
            Err(e) => return Err(to_napi(e)),
        };
        Ok(json!({ "results": results, "warning": warning }))
    }

    fn resolve(&mut self, _env: Env, output: Value) -> Result<Json> {
        Ok(Json(output))
    }
}

pub struct OptimizeTask {
    db: Db,
}

impl Task for OptimizeTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        self.db.optimize().or_napi()
    }

    fn resolve(&mut self, _env: Env, _output: ()) -> Result<()> {
        Ok(())
    }
}

pub struct WaitTask {
    db: Db,
}

impl Task for WaitTask {
    type Output = ();
    type JsValue = ();

    fn compute(&mut self) -> Result<()> {
        self.db.wait_for_maintenance().or_napi()
    }

    fn resolve(&mut self, _env: Env, _output: ()) -> Result<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------- collection

/// A collection in an open database.
#[napi]
pub struct Collection {
    db: Shared,
    name: String,
}

enum Read {
    Plan(Doc),
    Search {
        query: Vec<f32>,
        k: usize,
        ef: Option<usize>,
        filter: Option<Filter>,
        text: Option<String>,
        mode: Mode,
    },
    Get(String),
    Chunks(String),
    Scan(Option<Filter>),
    Count,
    EnableLexical,
}

enum Mode {
    Chunks,
    Documents(Agg),
    Expand(u32),
    Sections,
}

pub struct ReadTask {
    coll: vdblite::Collection,
    read: Option<Read>,
}

impl Task for ReadTask {
    type Output = Value;
    type JsValue = Json;

    fn compute(&mut self) -> Result<Value> {
        let c = &self.coll;
        Ok(match self.read.take().expect("computed once") {
            Read::Plan(doc) => shared::plan_json(&c.plan_embeddings(&doc).or_napi()?),
            Read::Search {
                query,
                k,
                ef,
                filter,
                text,
                mode,
            } => {
                let mut s = c.search(&query).k(k);
                if let Some(t) = text {
                    s = s.text(t);
                }
                if let Some(ef) = ef {
                    s = s.ef(ef);
                }
                if let Some(f) = filter {
                    s = s.filter(f);
                }
                match mode {
                    Mode::Chunks => Value::Array(
                        s.chunks()
                            .or_napi()?
                            .iter()
                            .map(shared::chunk_hit_json)
                            .collect(),
                    ),
                    Mode::Expand(n) => Value::Array(
                        s.expand(n)
                            .or_napi()?
                            .iter()
                            .map(shared::passage_json)
                            .collect(),
                    ),
                    Mode::Sections => Value::Array(
                        s.section()
                            .or_napi()?
                            .iter()
                            .map(shared::passage_json)
                            .collect(),
                    ),
                    Mode::Documents(agg) => Value::Array(
                        s.documents(agg)
                            .or_napi()?
                            .iter()
                            .map(shared::doc_hit_json)
                            .collect(),
                    ),
                }
            }
            Read::Get(key) => c
                .get(&key)
                .or_napi()?
                .as_ref()
                .map_or(Value::Null, shared::document_json),
            Read::Chunks(key) => Value::Array(
                c.chunks(&key)
                    .or_napi()?
                    .iter()
                    .map(shared::chunk_info_json)
                    .collect(),
            ),
            Read::Scan(filter) => Value::Array(
                c.scan(filter.as_ref())
                    .or_napi()?
                    .iter()
                    .map(shared::doc_info_json)
                    .collect(),
            ),
            Read::Count => json!(c.len().or_napi()?),
            Read::EnableLexical => json!(c.enable_lexical().or_napi()?),
        })
    }

    fn resolve(&mut self, _env: Env, output: Value) -> Result<Json> {
        Ok(Json(output))
    }
}

#[napi]
impl Collection {
    fn handle(&self) -> Result<vdblite::Collection> {
        db_of(&self.db)?.collection(&self.name).or_napi()
    }

    fn task(&self, read: Read) -> Result<AsyncTask<ReadTask>> {
        Ok(AsyncTask::new(ReadTask {
            coll: self.handle()?,
            read: Some(read),
        }))
    }

    fn search_task(
        &self,
        query: Float32Array,
        options: Option<SearchOptions>,
        mode: Mode,
    ) -> Result<AsyncTask<ReadTask>> {
        let o = options.unwrap_or(SearchOptions {
            k: None,
            ef: None,
            filter: None,
            text: None,
        });
        let filter = match &o.filter {
            None | Some(Value::Null) => None,
            Some(f) => Some(parse_filter(f)?),
        };
        self.task(Read::Search {
            query: query.to_vec(),
            k: o.k.unwrap_or(10) as usize,
            ef: o.ef.map(|e| e as usize),
            filter,
            text: o.text,
            mode,
        })
    }

    #[napi(getter)]
    pub fn name(&self) -> String {
        self.name.clone()
    }

    /// Whether the collection has a keyword index for hybrid search.
    #[napi]
    pub fn has_lexical(&self) -> Result<bool> {
        Ok(self.handle()?.has_lexical())
    }

    /// Builds the keyword index from stored text (no re-embedding).
    /// Resolves to false if it already existed.
    #[napi(ts_return_type = "Promise<boolean>")]
    pub fn enable_lexical(&self) -> Result<AsyncTask<ReadTask>> {
        self.task(Read::EnableLexical)
    }

    #[napi]
    pub fn info(&self) -> Result<CollectionInfo> {
        let c = self.handle()?;
        let s = c.schema();
        let (chunking, target, overlap) = match s.chunking {
            Chunking::None => ("none", None, None),
            Chunking::Structural {
                target_tokens,
                overlap_tokens,
            } => (
                "structural",
                Some(target_tokens as u32),
                Some(overlap_tokens as u32),
            ),
        };
        Ok(CollectionInfo {
            name: self.name.clone(),
            dim: s.dim as u32,
            metric: shared::metric_name(s.metric).into(),
            quantization: shared::quant_name(s.quant).into(),
            chunking: chunking.into(),
            target_tokens: target,
            overlap_tokens: overlap,
        })
    }

    #[napi]
    pub fn stats(&self) -> Result<IndexStats> {
        let s = self.handle()?.index_stats();
        Ok(IndexStats {
            chunks: s.chunks as u32,
            sealed_segments: s.sealed_segments as u32,
            buffered: s.buffered as u32,
            memory_bytes: s.memory_bytes as f64,
            maintenance_error: s.maintenance_error,
        })
    }

    /// The embedding inputs `content` would produce, one per chunk.
    #[napi]
    pub fn embed_inputs(&self, content: String, format: Option<String>) -> Result<Vec<String>> {
        Ok(self.handle()?.embed_inputs(&content, format.as_deref()))
    }

    /// Chunk texts that need new vectors to upsert `doc`: `[{ hash, text }]`.
    #[napi(ts_return_type = "Promise<Array<{ hash: string; text: string }>>")]
    pub fn plan_embeddings(&self, doc: DocInput) -> Result<AsyncTask<ReadTask>> {
        let doc = doc_from_input(doc.key, doc.content, doc.meta.as_ref(), doc.format)?;
        self.task(Read::Plan(doc))
    }

    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn search(
        &self,
        query: Float32Array,
        options: Option<SearchOptions>,
    ) -> Result<AsyncTask<ReadTask>> {
        self.search_task(query, options, Mode::Chunks)
    }

    /// `agg`: "max" (default) or "sum" with `top` chunks (default 3).
    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn search_documents(
        &self,
        query: Float32Array,
        options: Option<SearchOptions>,
        agg: Option<String>,
        top: Option<u32>,
    ) -> Result<AsyncTask<ReadTask>> {
        let agg = shared::agg(agg.as_deref(), top.map(|t| t as usize)).map_err(invalid)?;
        self.search_task(query, options, Mode::Documents(agg))
    }

    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn search_expand(
        &self,
        query: Float32Array,
        neighbors: u32,
        options: Option<SearchOptions>,
    ) -> Result<AsyncTask<ReadTask>> {
        self.search_task(query, options, Mode::Expand(neighbors))
    }

    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn search_sections(
        &self,
        query: Float32Array,
        options: Option<SearchOptions>,
    ) -> Result<AsyncTask<ReadTask>> {
        self.search_task(query, options, Mode::Sections)
    }

    #[napi(ts_return_type = "Promise<any>")]
    pub fn get(&self, key: String) -> Result<AsyncTask<ReadTask>> {
        self.task(Read::Get(key))
    }

    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn chunks(&self, key: String) -> Result<AsyncTask<ReadTask>> {
        self.task(Read::Chunks(key))
    }

    #[napi(ts_return_type = "Promise<any[]>")]
    pub fn scan(&self, filter: Option<Value>) -> Result<AsyncTask<ReadTask>> {
        let filter = match &filter {
            None | Some(Value::Null) => None,
            Some(f) => Some(parse_filter(f)?),
        };
        self.task(Read::Scan(filter))
    }

    #[napi(ts_return_type = "Promise<number>")]
    pub fn count(&self) -> Result<AsyncTask<ReadTask>> {
        self.task(Read::Count)
    }

    /// Starts a bulk load; one per collection runs at a time.
    #[napi]
    pub fn bulk_load(&self, options: Option<BulkLoadOptions>) -> Result<BulkLoad> {
        let mut load = db_of(&self.db)?.bulk_load(&self.handle()?).or_napi()?;
        if let Some(n) = options.and_then(|o| o.checkpoint_every) {
            if !(n.is_finite() && n >= 1.0) {
                return Err(invalid(
                    "checkpointEvery must be a positive number of chunks",
                ));
            }
            load = load.checkpoint_every(n as usize);
        }
        Ok(BulkLoad {
            load: Arc::new(Mutex::new(Some(load))),
        })
    }
}

// ---------------------------------------------------------------- bulk load

/// Chunks between durable commits of a bulk load, by default.
#[napi]
pub const DEFAULT_CHECKPOINT: u32 = vdblite::DEFAULT_CHECKPOINT as u32;

/// A bulk load in progress (see `Collection.bulkLoad`).
#[napi]
pub struct BulkLoad {
    /// `None` once finished.
    load: Arc<Mutex<Option<vdblite::BulkLoad>>>,
}

enum BulkStep {
    Add(Vec<(Doc, Vec<Vec<f32>>)>),
    Checkpoint,
    Finish,
}

pub struct BulkTask {
    load: Arc<Mutex<Option<vdblite::BulkLoad>>>,
    step: Option<BulkStep>,
}

impl Task for BulkTask {
    type Output = u32;
    type JsValue = u32;

    fn compute(&mut self) -> Result<u32> {
        let finished = || invalid("the bulk load has finished");
        let mut guard = self.load.lock().unwrap_or_else(|e| e.into_inner());
        match self.step.take().expect("computed once") {
            BulkStep::Add(docs) => {
                let load = guard.as_mut().ok_or_else(finished)?;
                Ok(load.add(docs).or_napi()? as u32)
            }
            BulkStep::Checkpoint => {
                guard
                    .as_mut()
                    .ok_or_else(finished)?
                    .checkpoint()
                    .or_napi()?;
                Ok(0)
            }
            BulkStep::Finish => {
                let load = guard.take().ok_or_else(finished)?;
                drop(guard);
                load.finish().or_napi()?;
                Ok(0)
            }
        }
    }

    fn resolve(&mut self, _env: Env, output: u32) -> Result<u32> {
        Ok(output)
    }
}

#[napi]
impl BulkLoad {
    fn task(&self, step: BulkStep) -> AsyncTask<BulkTask> {
        AsyncTask::new(BulkTask {
            load: self.load.clone(),
            step: Some(step),
        })
    }

    /// Adds documents in one transaction; resolves to how many were written.
    #[napi(ts_return_type = "Promise<number>")]
    pub fn add(&self, docs: Vec<BulkDoc>) -> Result<AsyncTask<BulkTask>> {
        let docs = docs
            .into_iter()
            .map(|d| {
                let vectors = d.vectors.iter().map(|v| v.to_vec()).collect();
                Ok((
                    doc_from_input(d.key, d.content, d.meta.as_ref(), d.format)?,
                    vectors,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(self.task(BulkStep::Add(docs)))
    }

    /// Makes everything added so far durable.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn checkpoint(&self) -> AsyncTask<BulkTask> {
        self.task(BulkStep::Checkpoint)
    }

    /// Builds the index over everything added, with every core, and makes it
    /// searchable.
    #[napi(ts_return_type = "Promise<void>")]
    pub fn finish(&self) -> AsyncTask<BulkTask> {
        self.task(BulkStep::Finish)
    }

    /// Whether `finish` has been called.
    #[napi(getter)]
    pub fn finished(&self) -> bool {
        // Never wait on the event loop: a held lock means an add or
        // checkpoint is running, so the load hasn't finished.
        match self.load.try_lock() {
            Ok(g) => g.is_none(),
            Err(std::sync::TryLockError::Poisoned(e)) => e.into_inner().is_none(),
            Err(std::sync::TryLockError::WouldBlock) => false,
        }
    }
}
