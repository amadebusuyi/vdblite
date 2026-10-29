//! JSON conversions shared by the language bindings (Node, and the C ABI
//! that Go, Java, .NET, Swift and C++ build on): metadata, Mongo-style
//! filters, collection options, and the result shapes. Every binding speaks
//! the same JSON, so a filter or a result looks the same in each language.
//!
//! Errors are messages for an invalid argument; each binding wraps them in
//! its own error type.

use std::collections::HashMap;
use std::ops::Bound;

use serde_json::{Map, Number, Value, json};
use vdblite::{
    Agg, ChunkHit, ChunkInfo, Chunking, Doc, DocHit, DocInfo, Document, Filter, HnswParams,
    IndexOptions, IndexStats, MetaValue, Metric, Passage, Quant, Schema, Upserted,
};

/// An invalid argument, as a message.
pub type Result<T> = std::result::Result<T, String>;

/// The stable code for an error, the same in every binding.
pub fn code_of(e: &vdblite::Error) -> &'static str {
    use vdblite::Error as E;
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

// ---------------------------------------------------------------- metadata

pub fn meta_from_json(v: &Value, field: &str) -> Result<MetaValue> {
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
                    Value::Array(_) | Value::Object(_) => Err(format!(
                        "metadata field {field:?}: lists may only hold scalars"
                    )),
                    x => meta_from_json(x, field),
                })
                .collect::<Result<_>>()?,
        ),
        Value::Object(_) => {
            return Err(format!(
                "metadata field {field:?}: nested objects are not supported"
            ));
        }
    })
}

pub fn meta_to_json(v: &MetaValue) -> Value {
    match v {
        MetaValue::Null => Value::Null,
        MetaValue::Bool(b) => Value::Bool(*b),
        MetaValue::Int(i) => json!(i),
        MetaValue::Float(f) => Number::from_f64(*f).map_or(Value::Null, Value::Number),
        MetaValue::Str(s) => Value::String(s.clone()),
        MetaValue::List(items) => Value::Array(items.iter().map(meta_to_json).collect()),
    }
}

pub fn meta_map_to_json(m: &vdblite::Meta) -> Value {
    Value::Object(
        m.iter()
            .map(|(k, v)| (k.clone(), meta_to_json(v)))
            .collect(),
    )
}

/// A document to write. `meta` must be an object, or null.
pub fn doc_from_input(
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
        Some(_) => return Err("meta must be an object".into()),
    }
    Ok(doc)
}

/// A document from `{ key, content, meta?, format? }`.
pub fn doc_from_json(v: &Value) -> Result<Doc> {
    let Value::Object(o) = v else {
        return Err("a document must be an object".into());
    };
    doc_from_input(
        str_field(o, "key")?.ok_or("a document needs a key")?,
        str_field(o, "content")?.unwrap_or_default(),
        o.get("meta"),
        str_field(o, "format")?,
    )
}

// ---------------------------------------------------------------- filters

/// Parses a Mongo-style filter:
/// `{ lang: "en", n: { $gte: 1, $lt: 5 }, tags: { $in: ["a"] }, $or: [...], $not: {...} }`.
pub fn parse_filter(v: &Value) -> Result<Filter> {
    let Value::Object(map) = v else {
        return Err("a filter must be an object".into());
    };
    let mut parts = Vec::new();
    for (key, cond) in map {
        let list = |what: &str| -> Result<Vec<Filter>> {
            match cond {
                Value::Array(items) => items.iter().map(parse_filter).collect(),
                _ => Err(format!("{what} takes an array of filters")),
            }
        };
        parts.push(match key.as_str() {
            "$and" => Filter::And(list("$and")?),
            "$or" => Filter::Or(list("$or")?),
            "$not" => !parse_filter(cond)?,
            k if k.starts_with('$') => return Err(format!("unknown operator {k}")),
            field => parse_condition(field, cond)?,
        });
    }
    Ok(match parts.len() {
        1 => parts.pop().expect("one part"),
        _ => Filter::And(parts),
    })
}

/// A filter, or none when `v` is absent or null.
pub fn optional_filter(v: Option<&Value>) -> Result<Option<Filter>> {
    match v {
        None | Some(Value::Null) => Ok(None),
        Some(f) => parse_filter(f).map(Some),
    }
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
            .ok_or_else(|| format!("{field}.{op} needs a number"))
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
                None => return Err(format!("{field}.$exists needs true or false")),
            },
            other => return Err(format!("unknown operator {other} on {field:?}")),
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
        0 => Err(format!("empty condition on {field:?}")),
        1 => Ok(parts.pop().expect("one part")),
        _ => Ok(Filter::And(parts)),
    }
}

fn scalars(field: &str, v: &Value) -> Result<Vec<MetaValue>> {
    match v {
        Value::Array(items) => items.iter().map(|x| meta_from_json(x, field)).collect(),
        _ => Err(format!("{field}: $in/$nin need an array")),
    }
}

// ---------------------------------------------------------------- options

pub fn metric(name: &str) -> Result<Metric> {
    match name {
        "cosine" => Ok(Metric::Cosine),
        "dot" => Ok(Metric::Dot),
        "l2" => Ok(Metric::L2),
        m => Err(format!("unknown metric {m:?} (cosine, dot, l2)")),
    }
}

pub fn metric_name(m: Metric) -> &'static str {
    match m {
        Metric::Cosine => "cosine",
        Metric::Dot => "dot",
        Metric::L2 => "l2",
    }
}

pub fn quant(name: &str) -> Result<Quant> {
    match name {
        "int8" => Ok(Quant::Int8),
        "binary" => Ok(Quant::Binary),
        "none" => Ok(Quant::None),
        q => Err(format!("unknown quantization {q:?} (int8, binary, none)")),
    }
}

pub fn quant_name(q: Quant) -> &'static str {
    match q {
        Quant::Int8 => "int8",
        Quant::Binary => "binary",
        Quant::None => "none",
    }
}

pub fn chunking(
    kind: Option<&str>,
    target: Option<usize>,
    overlap: Option<usize>,
) -> Result<Chunking> {
    match kind.unwrap_or("structural") {
        "none" => Ok(Chunking::None),
        "structural" => Ok(Chunking::Structural {
            target_tokens: target.unwrap_or(400),
            overlap_tokens: overlap.unwrap_or(50),
        }),
        k => Err(format!("unknown chunking {k:?} (structural, none)")),
    }
}

/// The parts of a collection's options that every binding spells the same
/// way; each option is `None` for its default.
#[derive(Default)]
pub struct CollectionOptions<'a> {
    pub dim: usize,
    pub metric: Option<&'a str>,
    pub quantization: Option<&'a str>,
    pub chunking: Option<&'a str>,
    pub target_tokens: Option<usize>,
    pub overlap_tokens: Option<usize>,
    pub seal_threshold: Option<usize>,
    pub lexical: Option<bool>,
    pub m: Option<usize>,
    pub ef_construction: Option<usize>,
    pub ef_search: Option<usize>,
}

impl CollectionOptions<'_> {
    pub fn build(&self) -> Result<(Schema, IndexOptions)> {
        let schema = Schema {
            dim: self.dim,
            metric: metric(self.metric.unwrap_or("cosine"))?,
            quant: quant(self.quantization.unwrap_or("int8"))?,
            chunking: chunking(self.chunking, self.target_tokens, self.overlap_tokens)?,
        };
        let defaults = HnswParams::default();
        let opts = IndexOptions {
            hnsw: HnswParams {
                m: self.m.unwrap_or(defaults.m),
                ef_construction: self.ef_construction.unwrap_or(defaults.ef_construction),
                ef_search: self.ef_search.unwrap_or(defaults.ef_search),
                ..defaults
            },
            seal_threshold: self
                .seal_threshold
                .unwrap_or(IndexOptions::default().seal_threshold),
            lexical: self.lexical.unwrap_or(true),
        };
        Ok((schema, opts))
    }
}

/// Collection options from JSON:
/// `{ dim, metric?, quantization?, chunking?: { kind?, targetTokens?, overlapTokens? },
///    sealThreshold?, lexical?, m?, efConstruction?, efSearch? }`.
pub fn collection_options(v: &Value) -> Result<(Schema, IndexOptions)> {
    let Value::Object(o) = v else {
        return Err("collection options must be an object".into());
    };
    let chunk = match o.get("chunking") {
        None | Some(Value::Null) => None,
        Some(Value::Object(c)) => Some(c),
        Some(_) => return Err("chunking must be an object".into()),
    };
    let empty = Map::new();
    let c = chunk.unwrap_or(&empty);
    CollectionOptions {
        dim: uint_field(o, "dim")?.ok_or("collection options need dim")?,
        metric: str_ref(o, "metric")?,
        quantization: str_ref(o, "quantization")?,
        chunking: str_ref(c, "kind")?,
        target_tokens: uint_field(c, "targetTokens")?,
        overlap_tokens: uint_field(c, "overlapTokens")?,
        seal_threshold: uint_field(o, "sealThreshold")?,
        lexical: bool_field(o, "lexical")?,
        m: uint_field(o, "m")?,
        ef_construction: uint_field(o, "efConstruction")?,
        ef_search: uint_field(o, "efSearch")?,
    }
    .build()
}

pub fn agg(name: Option<&str>, top: Option<usize>) -> Result<Agg> {
    match name.unwrap_or("max") {
        "max" => Ok(Agg::MaxSim),
        "sum" => Ok(Agg::SumTop(top.unwrap_or(3))),
        a => Err(format!("unknown agg {a:?} (max, sum)")),
    }
}

pub fn parse_hash(h: &str) -> Result<u64> {
    u64::from_str_radix(h, 16).map_err(|_| format!("invalid chunk hash {h:?}"))
}

pub fn hash_str(h: u64) -> String {
    format!("{h:016x}")
}

// ---------------------------------------------------------------- JSON fields

fn str_ref<'a>(o: &'a Map<String, Value>, name: &str) -> Result<Option<&'a str>> {
    match o.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(s)) => Ok(Some(s)),
        Some(_) => Err(format!("{name} must be a string")),
    }
}

pub fn str_field(o: &Map<String, Value>, name: &str) -> Result<Option<String>> {
    Ok(str_ref(o, name)?.map(str::to_owned))
}

pub fn uint_field(o: &Map<String, Value>, name: &str) -> Result<Option<usize>> {
    match o.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => v
            .as_u64()
            .map(|n| Some(n as usize))
            .ok_or_else(|| format!("{name} must be a non-negative integer")),
    }
}

pub fn bool_field(o: &Map<String, Value>, name: &str) -> Result<Option<bool>> {
    match o.get(name) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::Bool(b)) => Ok(Some(*b)),
        Some(_) => Err(format!("{name} must be true or false")),
    }
}

// ---------------------------------------------------------------- results

pub fn chunk_hit_json(h: &ChunkHit) -> Value {
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

pub fn passage_json(p: &Passage) -> Value {
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

pub fn doc_hit_json(d: &DocHit) -> Value {
    json!({
        "key": d.key, "docId": d.doc_id, "score": d.score,
        "meta": meta_map_to_json(&d.meta),
        "chunks": d.chunks.iter().map(chunk_hit_json).collect::<Vec<_>>(),
    })
}

pub fn document_json(d: &Document) -> Value {
    json!({
        "key": d.key, "docId": d.id, "content": d.content, "meta": meta_map_to_json(&d.meta),
        "version": d.version, "createdMs": d.created_ms, "updatedMs": d.updated_ms,
        "format": d.format, "chunkCount": d.chunk_count,
    })
}

pub fn doc_info_json(d: &DocInfo) -> Value {
    json!({
        "key": d.key, "docId": d.id, "meta": meta_map_to_json(&d.meta),
        "version": d.version, "createdMs": d.created_ms, "updatedMs": d.updated_ms,
        "contentLength": d.content_len, "chunkCount": d.chunk_count,
    })
}

pub fn chunk_info_json(c: &ChunkInfo) -> Value {
    json!({
        "chunkId": c.chunk_id, "ordinal": c.ordinal, "byteStart": c.range.start,
        "byteEnd": c.range.end, "overlapStart": c.overlap_start,
        "headingPath": c.heading_path, "tokens": c.tokens,
    })
}

pub fn upserted_json(collection: &str, key: &str, up: &Upserted) -> Value {
    json!({
        "op": "upsert", "collection": collection, "key": key, "docId": up.doc_id,
        "version": up.version, "chunks": up.chunks, "embedded": up.embedded,
        "changed": up.changed,
    })
}

pub fn deleted_json(collection: &str, key: &str, deleted: bool) -> Value {
    json!({ "op": "delete", "collection": collection, "key": key, "deleted": deleted })
}

pub fn plan_json(plan: &[(u64, String)]) -> Value {
    Value::Array(
        plan.iter()
            .map(|(h, text)| json!({ "hash": hash_str(*h), "text": text }))
            .collect(),
    )
}

pub fn info_json(name: &str, s: &Schema) -> Value {
    let mut v = json!({
        "name": name,
        "dim": s.dim,
        "metric": metric_name(s.metric),
        "quantization": quant_name(s.quant),
    });
    match s.chunking {
        Chunking::None => v["chunking"] = json!("none"),
        Chunking::Structural {
            target_tokens,
            overlap_tokens,
        } => {
            v["chunking"] = json!("structural");
            v["targetTokens"] = json!(target_tokens);
            v["overlapTokens"] = json!(overlap_tokens);
        }
    }
    v
}

pub fn stats_json(s: &IndexStats) -> Value {
    json!({
        "chunks": s.chunks,
        "sealedSegments": s.sealed_segments,
        "buffered": s.buffered,
        "memoryBytes": s.memory_bytes,
        "maintenanceError": s.maintenance_error,
    })
}

/// Vectors keyed by chunk hash, from `planEmbeddings` hashes and their
/// vectors in the same order.
pub fn by_hash(hashes: &[String], vectors: Vec<Vec<f32>>) -> Result<HashMap<u64, Vec<f32>>> {
    if hashes.len() != vectors.len() {
        return Err(format!(
            "{} chunk hashes but {} vectors",
            hashes.len(),
            vectors.len()
        ));
    }
    hashes
        .iter()
        .zip(vectors)
        .map(|(h, v)| Ok((parse_hash(h)?, v)))
        .collect()
}
