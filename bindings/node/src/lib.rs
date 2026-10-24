//! Node.js bindings for vecfile (napi-rs).
//!
//! This is the native layer; `index.js` wraps it with the friendlier API
//! (embedding functions, transactions, retries). Anything that touches the
//! file runs on the libuv thread pool and returns a Promise, so a commit that
//! seals a segment never blocks the event loop.
//!
//! Errors carry a stable code as a `CODE: message` prefix, which `index.js`
//! turns into `err.code`.

use std::collections::HashMap;
use std::ops::Bound;
use std::sync::{Arc, RwLock};

use napi::bindgen_prelude::*;
use napi_derive::napi;
use serde_json::{Number, Value, json};
use vecfile::{
    Agg, ChunkHit, Chunking, Db, Doc, Filter, HnswParams, IndexOptions, MetaValue, Metric, Passage,
    Quant, Schema,
};

// ---------------------------------------------------------------- errors

fn code_of(e: &vecfile::Error) -> &'static str {
    use vecfile::Error as E;
    match e {
        E::Storage(_) => "STORAGE",
        E::Dimension { .. } => "DIMENSION",
        E::CollectionExists(_) => "COLLECTION_EXISTS",
        E::NoSuchCollection(_) => "NO_SUCH_COLLECTION",
        E::InvalidArgument(_) => "INVALID_ARGUMENT",
        E::Embedder(_) => "EMBEDDER",
        E::StaleEmbeddings { .. } => "STALE_EMBEDDINGS",
        E::Version { .. } => "VERSION",
        E::Maintenance(_) => "MAINTENANCE",
        E::ReadOnly => "READ_ONLY",
    }
}

fn to_napi(e: vecfile::Error) -> napi::Error {
    napi::Error::from_reason(format!("{}: {e}", code_of(&e)))
}

fn invalid(msg: impl std::fmt::Display) -> napi::Error {
    napi::Error::from_reason(format!("INVALID_ARGUMENT: {msg}"))
}

trait OrNapi<T> {
    fn or_napi(self) -> Result<T>;
}

impl<T> OrNapi<T> for std::result::Result<T, vecfile::Error> {
    fn or_napi(self) -> Result<T> {
        self.map_err(to_napi)
    }
}

// ---------------------------------------------------------------- JSON <-> metadata

fn meta_from_json(v: &Value, field: &str) -> Result<MetaValue> {
    Ok(match v {
        Value::Null => MetaValue::Null,
        Value::Bool(b) => MetaValue::Bool(*b),
        Value::Number(n) => match n.as_i64() {
            Some(i) => MetaValue::Int(i),
            None => MetaValue::Float(n.as_f64().unwrap_or(f64::NAN)),
        },
        Value::String(s) => MetaValue::Str(s.clone()),
        Value::Array(items) => MetaValue::List(
            items
                .iter()
                .map(|x| match x {
                    Value::Array(_) | Value::Object(_) => Err(invalid(format!(
                        "metadata field {field:?}: lists may only hold scalars"
                    ))),
                    x => meta_from_json(x, field),
                })
                .collect::<Result<_>>()?,
        ),
        Value::Object(_) => {
            return Err(invalid(format!(
                "metadata field {field:?}: nested objects are not supported"
            )));
        }
    })
}

fn meta_to_json(v: &MetaValue) -> Value {
    match v {
        MetaValue::Null => Value::Null,
        MetaValue::Bool(b) => Value::Bool(*b),
        MetaValue::Int(i) => json!(i),
        MetaValue::Float(f) => Number::from_f64(*f).map_or(Value::Null, Value::Number),
        MetaValue::Str(s) => Value::String(s.clone()),
        MetaValue::List(items) => Value::Array(items.iter().map(meta_to_json).collect()),
    }
}

fn meta_map_to_json(m: &vecfile::Meta) -> Value {
    Value::Object(
        m.iter()
            .map(|(k, v)| (k.clone(), meta_to_json(v)))
            .collect(),
    )
}

/// Parses a Mongo-style filter:
/// `{ lang: "en", n: { $gte: 1, $lt: 5 }, tags: { $in: ["a"] }, $or: [...], $not: {...} }`.
fn parse_filter(v: &Value) -> Result<Filter> {
    let Value::Object(map) = v else {
        return Err(invalid("a filter must be an object"));
    };
    let mut parts = Vec::new();
    for (key, cond) in map {
        let list = |what: &str| -> Result<Vec<Filter>> {
            match cond {
                Value::Array(items) => items.iter().map(parse_filter).collect(),
                _ => Err(invalid(format!("{what} takes an array of filters"))),
            }
        };
        parts.push(match key.as_str() {
            "$and" => Filter::And(list("$and")?),
            "$or" => Filter::Or(list("$or")?),
            "$not" => !parse_filter(cond)?,
            k if k.starts_with('$') => return Err(invalid(format!("unknown operator {k}"))),
            field => parse_condition(field, cond)?,
        });
    }
    Ok(match parts.len() {
        1 => parts.pop().expect("one part"),
        _ => Filter::And(parts),
    })
}

fn parse_condition(field: &str, cond: &Value) -> Result<Filter> {
    let Value::Object(ops) = cond else {
        return Ok(match cond {
            Value::Array(_) => Filter::In(field.into(), scalars(field, cond)?),
            v => Filter::Eq(field.into(), meta_from_json(v, field)?),
        });
    };
    let num = |op: &str, v: &Value| {
        v.as_f64()
            .ok_or_else(|| invalid(format!("{field}.{op} needs a number")))
    };
    let mut parts = Vec::new();
    let (mut lo, mut hi) = (Bound::Unbounded, Bound::Unbounded);
    for (op, v) in ops {
        match op.as_str() {
            "$eq" => parts.push(Filter::Eq(field.into(), meta_from_json(v, field)?)),
            "$ne" => parts.push(!Filter::Eq(field.into(), meta_from_json(v, field)?)),
            "$in" => parts.push(Filter::In(field.into(), scalars(field, v)?)),
            "$nin" => parts.push(!Filter::In(field.into(), scalars(field, v)?)),
            "$gt" => lo = Bound::Excluded(num(op, v)?),
            "$gte" => lo = Bound::Included(num(op, v)?),
            "$lt" => hi = Bound::Excluded(num(op, v)?),
            "$lte" => hi = Bound::Included(num(op, v)?),
            "$exists" => match v.as_bool() {
                Some(true) => parts.push(Filter::Exists(field.into())),
                Some(false) => parts.push(!Filter::Exists(field.into())),
                None => return Err(invalid(format!("{field}.$exists needs true or false"))),
            },
            other => return Err(invalid(format!("unknown operator {other} on {field:?}"))),
        }
    }
    if !matches!((lo, hi), (Bound::Unbounded, Bound::Unbounded)) {
        parts.push(Filter::Range {
            field: field.into(),
            lo,
            hi,
        });
    }
    match parts.len() {
        0 => Err(invalid(format!("empty condition on {field:?}"))),
        1 => Ok(parts.pop().expect("one part")),
        _ => Ok(Filter::And(parts)),
    }
}

fn scalars(field: &str, v: &Value) -> Result<Vec<MetaValue>> {
    match v {
        Value::Array(items) => items.iter().map(|x| meta_from_json(x, field)).collect(),
        _ => Err(invalid(format!("{field}: $in/$nin need an array"))),
    }
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
    /// "int8" (default) or "none" (full f32 vectors in memory).
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
pub struct SearchOptions {
    pub k: Option<u32>,
    pub ef: Option<u32>,
    pub filter: Option<Value>,
    /// Query text for hybrid search (BM25 fused with the vector ranking).
    pub text: Option<String>,
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
    let mut doc = Doc::new(key).content(content);
    if let Some(f) = format {
        doc = doc.format(f);
    }
    match meta {
        None | Some(Value::Null) => {}
        Some(Value::Object(m)) => {
            for (k, v) in m {
                doc = doc.meta(k.clone(), meta_from_json(v, k)?);
            }
        }
        Some(_) => return Err(invalid("meta must be an object")),
    }
    Ok(doc)
}

fn parse_hash(h: &str) -> Result<u64> {
    u64::from_str_radix(h, 16).map_err(|_| invalid(format!("invalid chunk hash {h:?}")))
}

fn hash_str(h: u64) -> String {
    format!("{h:016x}")
}

// ---------------------------------------------------------------- output JSON

fn chunk_hit_json(h: &ChunkHit) -> Value {
    json!({
        "key": h.key,
        "docId": h.doc_id,
        "chunkId": h.chunk_id,
        "ordinal": h.ordinal,
        "byteStart": h.range.start,
        "byteEnd": h.range.end,
        "score": h.score,
        "vectorScore": h.vector_score,
        "lexicalScore": h.lexical_score,
        "text": h.text,
        "headingPath": h.heading_path,
        "meta": meta_map_to_json(&h.meta),
    })
}

fn passage_json(p: &Passage) -> Value {
    json!({
        "key": p.key,
        "docId": p.doc_id,
        "byteStart": p.range.start,
        "byteEnd": p.range.end,
        "score": p.score,
        "text": p.text,
        "ordinals": [p.ordinals.start, p.ordinals.end],
        "headingPath": p.heading_path,
        "meta": meta_map_to_json(&p.meta),
    })
}

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

/// An open vecfile database file.
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
    pub fn open(path: String) -> Result<Database> {
        Self::wrap(Db::open(&path).or_napi()?, path)
    }

    /// Opens for writing with no other process allowed, not even readers.
    #[napi(factory)]
    pub fn open_exclusive(path: String) -> Result<Database> {
        Self::wrap(Db::open_exclusive(&path).or_napi()?, path)
    }

    /// Opens an existing file read-only, alongside its writer.
    #[napi(factory)]
    pub fn open_read_only(path: String) -> Result<Database> {
        Self::wrap(Db::open_read_only(&path).or_napi()?, path)
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
        let metric = match options.metric.as_deref().unwrap_or("cosine") {
            "cosine" => Metric::Cosine,
            "dot" => Metric::Dot,
            "l2" => Metric::L2,
            m => return Err(invalid(format!("unknown metric {m:?} (cosine, dot, l2)"))),
        };
        let quant = match options.quantization.as_deref().unwrap_or("int8") {
            "int8" => Quant::Int8,
            "none" => Quant::None,
            q => return Err(invalid(format!("unknown quantization {q:?} (int8, none)"))),
        };
        let chunking = match &options.chunking {
            None => Chunking::default(),
            Some(c) => match c.kind.as_deref().unwrap_or("structural") {
                "none" => Chunking::None,
                "structural" => Chunking::Structural {
                    target_tokens: c.target_tokens.unwrap_or(400) as usize,
                    overlap_tokens: c.overlap_tokens.unwrap_or(50) as usize,
                },
                k => {
                    return Err(invalid(format!(
                        "unknown chunking {k:?} (structural, none)"
                    )));
                }
            },
        };
        let defaults = HnswParams::default();
        let opts = IndexOptions {
            hnsw: HnswParams {
                m: options.m.map_or(defaults.m, |v| v as usize),
                ef_construction: options
                    .ef_construction
                    .map_or(defaults.ef_construction, |v| v as usize),
                ef_search: options.ef_search.map_or(defaults.ef_search, |v| v as usize),
                ..defaults
            },
            seal_threshold: options
                .seal_threshold
                .map_or(IndexOptions::default().seal_threshold, |v| v as usize),
            lexical: options.lexical.unwrap_or(true),
        };
        let schema = Schema {
            dim: options.dim as usize,
            metric,
            quant,
            chunking,
        };
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
                                    .map(|p| Ok((parse_hash(&p.hash)?, p.vector.to_vec())))
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
        let mut colls: HashMap<String, vecfile::Collection> = HashMap::new();
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
                    json!({ "op": "delete", "collection": name, "key": key, "deleted": deleted })
                }
                Op::Upsert { doc, vectors, .. } => {
                    let key = doc.key().to_owned();
                    let up = match vectors {
                        Vectors::ByHash(m) => tx.upsert_with_embeddings(coll, doc, m),
                        Vectors::Given(v) => tx.upsert_with_vectors(coll, doc, v),
                    }
                    .or_napi()?;
                    json!({
                        "op": "upsert", "collection": name, "key": key, "docId": up.doc_id,
                        "version": up.version, "chunks": up.chunks, "embedded": up.embedded,
                        "changed": up.changed,
                    })
                }
            });
        }
        let warning = match tx.commit() {
            Ok(()) => Value::Null,
            // The data is committed; only sealing an oversized write buffer
            // afterwards failed.
            Err(e @ vecfile::Error::Maintenance(_)) => Value::String(e.to_string()),
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
    coll: vecfile::Collection,
    read: Option<Read>,
}

impl Task for ReadTask {
    type Output = Value;
    type JsValue = Json;

    fn compute(&mut self) -> Result<Value> {
        let c = &self.coll;
        Ok(match self.read.take().expect("computed once") {
            Read::Plan(doc) => Value::Array(
                c.plan_embeddings(&doc)
                    .or_napi()?
                    .into_iter()
                    .map(|(h, text)| json!({ "hash": hash_str(h), "text": text }))
                    .collect(),
            ),
            Read::Search { query, k, ef, filter, text, mode } => {
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
                    Mode::Chunks => Value::Array(s.chunks().or_napi()?.iter().map(chunk_hit_json).collect()),
                    Mode::Expand(n) => Value::Array(s.expand(n).or_napi()?.iter().map(passage_json).collect()),
                    Mode::Sections => Value::Array(s.section().or_napi()?.iter().map(passage_json).collect()),
                    Mode::Documents(agg) => Value::Array(
                        s.documents(agg)
                            .or_napi()?
                            .iter()
                            .map(|d| {
                                json!({
                                    "key": d.key, "docId": d.doc_id, "score": d.score,
                                    "meta": meta_map_to_json(&d.meta),
                                    "chunks": d.chunks.iter().map(chunk_hit_json).collect::<Vec<_>>(),
                                })
                            })
                            .collect(),
                    ),
                }
            }
            Read::Get(key) => match c.get(&key).or_napi()? {
                None => Value::Null,
                Some(d) => json!({
                    "key": d.key, "docId": d.id, "content": d.content, "meta": meta_map_to_json(&d.meta),
                    "version": d.version, "createdMs": d.created_ms, "updatedMs": d.updated_ms,
                    "format": d.format, "chunkCount": d.chunk_count,
                }),
            },
            Read::Chunks(key) => Value::Array(
                c.chunks(&key)
                    .or_napi()?
                    .iter()
                    .map(|ch| {
                        json!({
                            "chunkId": ch.chunk_id, "ordinal": ch.ordinal, "byteStart": ch.range.start,
                            "byteEnd": ch.range.end, "overlapStart": ch.overlap_start,
                            "headingPath": ch.heading_path, "tokens": ch.tokens,
                        })
                    })
                    .collect(),
            ),
            Read::Scan(filter) => Value::Array(
                c.scan(filter.as_ref())
                    .or_napi()?
                    .iter()
                    .map(|d| {
                        json!({
                            "key": d.key, "docId": d.id, "meta": meta_map_to_json(&d.meta),
                            "version": d.version, "createdMs": d.created_ms, "updatedMs": d.updated_ms,
                            "contentLength": d.content_len, "chunkCount": d.chunk_count,
                        })
                    })
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
    fn handle(&self) -> Result<vecfile::Collection> {
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
            metric: match s.metric {
                Metric::Cosine => "cosine",
                Metric::Dot => "dot",
                Metric::L2 => "l2",
            }
            .into(),
            quantization: if s.quant == Quant::None {
                "none"
            } else {
                "int8"
            }
            .into(),
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
        let agg = match agg.as_deref().unwrap_or("max") {
            "max" => Agg::MaxSim,
            "sum" => Agg::SumTop(top.unwrap_or(3) as usize),
            a => return Err(invalid(format!("unknown agg {a:?} (max, sum)"))),
        };
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
}
