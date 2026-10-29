//! C ABI for vdblite. The declarations are in `include/vdblite.h`, which
//! documents each function; the Go, Java, .NET, Swift and C++ bindings are
//! thin layers over it.
//!
//! Conventions:
//! - Functions return a status: `VDB_OK` (0) or an error code, with the
//!   message from `vdb_last_error` on the same thread.
//! - Strings in are NUL-terminated UTF-8. Structured values (options,
//!   documents, metadata, filters, results) are JSON in the shapes of
//!   `vdblite_bindings`, the same as the Node API.
//! - Strings out are allocated here and freed with `vdb_string_free`.
//! - Vectors travel as one flat `float` array; a JSON field says how many
//!   vectors each item takes, in order.
//! - Handles are thread-safe, except that freeing one must not race with
//!   other calls on it.
//! - A panic is caught and returned as `VDB_ERR_PANIC`.

use std::cell::RefCell;
use std::ffi::{CStr, CString, c_char};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Mutex, RwLock};

use serde_json::{Value, json};
use vdblite::{Db, Doc, OpenOptions};
use vdblite_bindings as shared;

// ---------------------------------------------------------------- status codes

pub const VDB_OK: i32 = 0;
pub const VDB_ERR_STORAGE: i32 = 1;
pub const VDB_ERR_DIMENSION: i32 = 2;
pub const VDB_ERR_COLLECTION_EXISTS: i32 = 3;
pub const VDB_ERR_NO_SUCH_COLLECTION: i32 = 4;
pub const VDB_ERR_INVALID_ARGUMENT: i32 = 5;
pub const VDB_ERR_EMBEDDER: i32 = 6;
pub const VDB_ERR_STALE_EMBEDDINGS: i32 = 7;
pub const VDB_ERR_VERSION: i32 = 8;
pub const VDB_ERR_MAINTENANCE: i32 = 9;
pub const VDB_ERR_READ_ONLY: i32 = 10;
pub const VDB_ERR_CLOSED: i32 = 11;
pub const VDB_ERR_PANIC: i32 = 12;

pub const VDB_OPEN_READ_ONLY: u32 = 1;
pub const VDB_OPEN_EXCLUSIVE: u32 = 2;

/// An error on its way to the caller: a status code and a message.
struct Fail(i32, String);

type Res<T> = Result<T, Fail>;

impl From<vdblite::Error> for Fail {
    fn from(e: vdblite::Error) -> Self {
        let code = match shared::code_of(&e) {
            "STORAGE" => VDB_ERR_STORAGE,
            "DIMENSION" => VDB_ERR_DIMENSION,
            "COLLECTION_EXISTS" => VDB_ERR_COLLECTION_EXISTS,
            "NO_SUCH_COLLECTION" => VDB_ERR_NO_SUCH_COLLECTION,
            "INVALID_ARGUMENT" => VDB_ERR_INVALID_ARGUMENT,
            "EMBEDDER" => VDB_ERR_EMBEDDER,
            "STALE_EMBEDDINGS" => VDB_ERR_STALE_EMBEDDINGS,
            "VERSION" => VDB_ERR_VERSION,
            "MAINTENANCE" => VDB_ERR_MAINTENANCE,
            "READ_ONLY" => VDB_ERR_READ_ONLY,
            _ => VDB_ERR_STORAGE,
        };
        Fail(code, e.to_string())
    }
}

fn invalid(msg: impl Into<String>) -> Fail {
    Fail(VDB_ERR_INVALID_ARGUMENT, msg.into())
}

fn closed() -> Fail {
    Fail(VDB_ERR_CLOSED, "the database is closed".into())
}

thread_local! {
    static LAST_ERROR: RefCell<CString> = RefCell::new(CString::default());
}

fn set_last_error(msg: &str) {
    let c = CString::new(msg.replace('\0', " ")).unwrap_or_default();
    LAST_ERROR.with(|e| *e.borrow_mut() = c);
}

/// Runs `f`, turning its error or panic into a status code and the
/// thread's last error message.
fn status(f: impl FnOnce() -> Res<()>) -> i32 {
    let outcome = catch_unwind(AssertUnwindSafe(f)).unwrap_or_else(|panic| {
        let msg = panic
            .downcast_ref::<&str>()
            .map(|s| (*s).to_owned())
            .or_else(|| panic.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "unknown panic".into());
        Err(Fail(VDB_ERR_PANIC, format!("vdblite panicked: {msg}")))
    });
    match outcome {
        Ok(()) => {
            set_last_error("");
            VDB_OK
        }
        Err(Fail(code, msg)) => {
            set_last_error(&msg);
            code
        }
    }
}

// ---------------------------------------------------------------- arguments

/// # Safety
/// `p` is null or a NUL-terminated string valid for the call.
unsafe fn opt_str<'a>(p: *const c_char, what: &str) -> Res<Option<&'a str>> {
    if p.is_null() {
        return Ok(None);
    }
    // SAFETY: non-null, and NUL-terminated by the caller's contract.
    let s = unsafe { CStr::from_ptr(p) };
    s.to_str()
        .map(Some)
        .map_err(|_| invalid(format!("{what} is not valid UTF-8")))
}

/// # Safety
/// As for `opt_str`.
unsafe fn req_str<'a>(p: *const c_char, what: &str) -> Res<&'a str> {
    // SAFETY: forwarded from our caller.
    unsafe { opt_str(p, what) }?.ok_or_else(|| invalid(format!("{what} is null")))
}

/// # Safety
/// As for `opt_str`.
unsafe fn opt_json(p: *const c_char, what: &str) -> Res<Option<Value>> {
    // SAFETY: forwarded from our caller.
    match unsafe { opt_str(p, what) }? {
        None => Ok(None),
        Some(s) => serde_json::from_str(s)
            .map(Some)
            .map_err(|e| invalid(format!("{what} is not valid JSON: {e}"))),
    }
}

/// # Safety
/// As for `opt_str`.
unsafe fn req_json(p: *const c_char, what: &str) -> Res<Value> {
    // SAFETY: forwarded from our caller.
    unsafe { opt_json(p, what) }?.ok_or_else(|| invalid(format!("{what} is null")))
}

/// # Safety
/// `p` is null only if `n` is 0; otherwise it points to `n` floats.
unsafe fn floats<'a>(p: *const f32, n: usize) -> Res<&'a [f32]> {
    if n == 0 {
        return Ok(&[]);
    }
    if p.is_null() {
        return Err(invalid("vectors is null"));
    }
    // SAFETY: non-null and `n` long by the caller's contract.
    Ok(unsafe { std::slice::from_raw_parts(p, n) })
}

/// # Safety
/// `out` is null or valid for a write.
unsafe fn put<T>(out: *mut T, v: T) -> Res<()> {
    if out.is_null() {
        return Err(invalid("output pointer is null"));
    }
    // SAFETY: non-null and writable by the caller's contract.
    unsafe { out.write(v) };
    Ok(())
}

/// # Safety
/// As for `put`.
unsafe fn put_json(out: *mut *mut c_char, v: &Value) -> Res<()> {
    let s = CString::new(v.to_string()).map_err(|_| invalid("result holds a NUL byte"))?;
    // SAFETY: forwarded from our caller.
    unsafe { put(out, s.into_raw()) }
}

/// # Safety
/// `p` is null or a live handle from this library.
unsafe fn handle<'a, T>(p: *const T, what: &str) -> Res<&'a T> {
    // SAFETY: a live handle by the caller's contract.
    unsafe { p.as_ref() }.ok_or_else(|| invalid(format!("{what} is null")))
}

/// Takes vectors from the front of `rest`, `count` of them, `dim` long.
fn take_vectors(rest: &mut &[f32], count: usize, dim: usize) -> Res<Vec<Vec<f32>>> {
    let need = count
        .checked_mul(dim)
        .ok_or_else(|| invalid("too many vectors"))?;
    if rest.len() < need {
        return Err(invalid(format!(
            "ran out of vectors: needed {need} more floats, {} left",
            rest.len()
        )));
    }
    let (now, later) = rest.split_at(need);
    *rest = later;
    Ok(now.chunks(dim.max(1)).map(<[f32]>::to_vec).collect())
}

fn obj<'a>(v: &'a Value, what: &str) -> Res<&'a serde_json::Map<String, Value>> {
    v.as_object()
        .ok_or_else(|| invalid(format!("{what} must be an object")))
}

fn str_field(o: &serde_json::Map<String, Value>, name: &str) -> Res<Option<String>> {
    shared::str_field(o, name).map_err(invalid)
}

fn uint_field(o: &serde_json::Map<String, Value>, name: &str) -> Res<Option<usize>> {
    shared::uint_field(o, name).map_err(invalid)
}

// ---------------------------------------------------------------- library

/// The library version, e.g. "0.1.0". Static; don't free it.
#[unsafe(no_mangle)]
pub extern "C" fn vdb_version() -> *const c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// The message of the last failed call on this thread ("" after a call
/// that succeeded). Valid until the next call on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn vdb_last_error() -> *const c_char {
    LAST_ERROR.with(|e| e.borrow().as_ptr())
}

/// Frees a string returned by this library. Null is ignored.
///
/// # Safety
/// `s` is null or a string from this library, not yet freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_string_free(s: *mut c_char) {
    if !s.is_null() {
        // SAFETY: allocated by `CString::into_raw` here, freed once.
        drop(unsafe { CString::from_raw(s) });
    }
}

// ---------------------------------------------------------------- database

type Shared = Arc<RwLock<Option<Db>>>;

/// An open database file.
pub struct VdbDb {
    db: Shared,
}

fn db_of(shared: &Shared) -> Res<Db> {
    shared
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .clone()
        .ok_or_else(closed)
}

/// Opens (or creates) a database file. `flags`: `VDB_OPEN_READ_ONLY`
/// and/or `VDB_OPEN_EXCLUSIVE`. `cache_bytes`: 0 for the default.
///
/// # Safety
/// `path` is a NUL-terminated string; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_open(
    path: *const c_char,
    flags: u32,
    cache_bytes: u64,
    out: *mut *mut VdbDb,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let path = unsafe { req_str(path, "path") }?;
        let mut o = OpenOptions::new()
            .read_only(flags & VDB_OPEN_READ_ONLY != 0)
            .exclusive(flags & VDB_OPEN_EXCLUSIVE != 0);
        if cache_bytes > 0 {
            o = o.cache_bytes(cache_bytes as usize);
        }
        let db = Db::open_with(path, &o)?;
        let handle = Box::new(VdbDb {
            db: Arc::new(RwLock::new(Some(db))),
        });
        // SAFETY: by our caller's contract.
        unsafe { put(out, Box::into_raw(handle)) }
    })
}

/// Closes the database: later calls on it, and on its collections, fail
/// with `VDB_ERR_CLOSED`. The file is released once in-flight calls and
/// unfinished bulk loads are done. The handle stays valid until freed.
///
/// # Safety
/// `db` is null or a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_close(db: *const VdbDb) {
    // SAFETY: by our caller's contract.
    if let Some(db) = unsafe { db.as_ref() } {
        db.db.write().unwrap_or_else(|e| e.into_inner()).take();
    }
}

/// Closes (if open) and frees a database handle. Null is ignored.
///
/// # Safety
/// `db` is null or a live handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_db_free(db: *mut VdbDb) {
    if !db.is_null() {
        // SAFETY: from `Box::into_raw` in `vdb_open`, freed once.
        let db = unsafe { Box::from_raw(db) };
        db.db.write().unwrap_or_else(|e| e.into_inner()).take();
    }
}

/// # Safety
/// `db` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_is_read_only(db: *const VdbDb, out: *mut bool) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let db = db_of(&unsafe { handle(db, "db") }?.db)?;
        // SAFETY: by our caller's contract.
        unsafe { put(out, db.is_read_only()) }
    })
}

/// The collection names, as a JSON array of strings.
///
/// # Safety
/// `db` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_collections(db: *const VdbDb, out: *mut *mut c_char) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let db = db_of(&unsafe { handle(db, "db") }?.db)?;
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &json!(db.collections()?)) }
    })
}

/// A collection in an open database.
pub struct VdbCollection {
    db: Shared,
    name: String,
}

impl VdbCollection {
    fn get(&self) -> Res<vdblite::Collection> {
        Ok(db_of(&self.db)?.collection(&self.name)?)
    }
}

/// Creates a collection. `options_json`:
/// `{"dim": 768, "metric"?, "quantization"?, "chunking"?: {"kind"?, "targetTokens"?,
/// "overlapTokens"?}, "sealThreshold"?, "lexical"?, "m"?, "efConstruction"?, "efSearch"?}`.
///
/// # Safety
/// `db` is a live handle; strings are NUL-terminated; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_create_collection(
    db: *const VdbDb,
    name: *const c_char,
    options_json: *const c_char,
    out: *mut *mut VdbCollection,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (db, name, options) = unsafe {
            (
                handle(db, "db")?,
                req_str(name, "name")?,
                req_json(options_json, "options")?,
            )
        };
        let (schema, opts) = shared::collection_options(&options).map_err(invalid)?;
        db_of(&db.db)?.create_collection_with(name, schema, opts)?;
        let c = Box::new(VdbCollection {
            db: db.db.clone(),
            name: name.to_owned(),
        });
        // SAFETY: by our caller's contract.
        unsafe { put(out, Box::into_raw(c)) }
    })
}

/// An existing collection (`VDB_ERR_NO_SUCH_COLLECTION` if there is none).
///
/// # Safety
/// `db` is a live handle; `name` is NUL-terminated; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_get_collection(
    db: *const VdbDb,
    name: *const c_char,
    out: *mut *mut VdbCollection,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (db, name) = unsafe { (handle(db, "db")?, req_str(name, "name")?) };
        db_of(&db.db)?.collection(name)?;
        let c = Box::new(VdbCollection {
            db: db.db.clone(),
            name: name.to_owned(),
        });
        // SAFETY: by our caller's contract.
        unsafe { put(out, Box::into_raw(c)) }
    })
}

/// # Safety
/// `db` is a live handle; `name` is NUL-terminated.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_drop_collection(db: *const VdbDb, name: *const c_char) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (db, name) = unsafe { (handle(db, "db")?, req_str(name, "name")?) };
        Ok(db_of(&db.db)?.drop_collection(name)?)
    })
}

/// Applies upserts and deletes atomically, in one transaction.
/// `ops_json` is an array of
/// `{"op": "upsert", "collection", "key", "content", "meta"?, "format"?, "vectors": n}`
/// (n vectors, one per chunk, from `vectors`),
/// `{"op": "upsert", ..., "embeddings": ["hash", ...]}` (vectors for the
/// chunks `vdb_plan_embeddings` asked for, one per hash, from `vectors`), or
/// `{"op": "delete", "collection", "key"}`. Vectors are taken from
/// `vectors` (`n_floats` long) in op order, each as long as its collection's
/// dimension. `out` gets `{"results": [...], "warning": null | "..."}`.
///
/// # Safety
/// `db` is a live handle; `ops_json` is NUL-terminated; `vectors` holds
/// `n_floats` floats; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_write(
    db: *const VdbDb,
    ops_json: *const c_char,
    vectors: *const f32,
    n_floats: usize,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (db, ops, mut rest) = unsafe {
            (
                handle(db, "db")?,
                req_json(ops_json, "ops")?,
                floats(vectors, n_floats)?,
            )
        };
        let db = db_of(&db.db)?;
        let Value::Array(ops) = ops else {
            return Err(invalid("ops must be an array"));
        };
        let mut colls: std::collections::HashMap<String, vdblite::Collection> = Default::default();
        let mut tx = db.begin_write()?;
        let mut results = Vec::with_capacity(ops.len());
        for op in &ops {
            let o = obj(op, "an op")?;
            let name =
                str_field(o, "collection")?.ok_or_else(|| invalid("an op needs a collection"))?;
            if !colls.contains_key(&name) {
                colls.insert(name.clone(), db.collection(&name)?);
            }
            let coll = &colls[&name];
            let key = str_field(o, "key")?.ok_or_else(|| invalid("an op needs a key"))?;
            match str_field(o, "op")?.as_deref() {
                Some("delete") => {
                    let deleted = tx.delete(coll, &key)?;
                    results.push(shared::deleted_json(&name, &key, deleted));
                }
                Some("upsert") => {
                    let doc = shared::doc_from_json(op).map_err(invalid)?;
                    let dim = coll.schema().dim;
                    let up = match (o.get("embeddings"), uint_field(o, "vectors")?) {
                        (Some(Value::Array(hashes)), None) => {
                            let hashes = hashes
                                .iter()
                                .map(|h| {
                                    h.as_str()
                                        .map(str::to_owned)
                                        .ok_or_else(|| invalid("embeddings must be hash strings"))
                                })
                                .collect::<Res<Vec<_>>>()?;
                            let vs = take_vectors(&mut rest, hashes.len(), dim)?;
                            let by_hash = shared::by_hash(&hashes, vs).map_err(invalid)?;
                            tx.upsert_with_embeddings(coll, doc, by_hash)?
                        }
                        (None, Some(n)) => {
                            let vs = take_vectors(&mut rest, n, dim)?;
                            tx.upsert_with_vectors(coll, doc, vs)?
                        }
                        (None, None) => tx.upsert_with_embeddings(coll, doc, Default::default())?,
                        _ => return Err(invalid("pass embeddings or vectors, not both")),
                    };
                    results.push(shared::upserted_json(&name, &key, &up));
                }
                other => {
                    return Err(invalid(format!("unknown op {other:?} (upsert, delete)")));
                }
            }
        }
        if !rest.is_empty() {
            return Err(invalid(format!(
                "{} floats left over after the ops",
                rest.len()
            )));
        }
        let warning = match tx.commit() {
            Ok(()) => Value::Null,
            // The data is committed; only sealing an oversized write buffer
            // afterwards failed.
            Err(e @ vdblite::Error::Maintenance(_)) => Value::String(e.to_string()),
            Err(e) => return Err(e.into()),
        };
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &json!({ "results": results, "warning": warning })) }
    })
}

/// Seals buffered chunks and merges small segments in every collection.
///
/// # Safety
/// `db` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_optimize(db: *const VdbDb) -> i32 {
    // SAFETY: by our caller's contract.
    status(|| Ok(db_of(&unsafe { handle(db, "db") }?.db)?.optimize()?))
}

/// Blocks until background sealing and compaction catch up with every
/// commit so far.
///
/// # Safety
/// `db` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_wait_for_maintenance(db: *const VdbDb) -> i32 {
    // SAFETY: by our caller's contract.
    status(|| Ok(db_of(&unsafe { handle(db, "db") }?.db)?.wait_for_maintenance()?))
}

/// Rewrites a closed file without its free space; `shrank` (may be null)
/// says whether it got smaller.
///
/// # Safety
/// `path` is NUL-terminated; `shrank` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_compact_file(path: *const c_char, shrank: *mut bool) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let path = unsafe { req_str(path, "path") }?;
        let s = Db::compact_file(path)?;
        if !shrank.is_null() {
            // SAFETY: non-null and writable by our caller's contract.
            unsafe { shrank.write(s) };
        }
        Ok(())
    })
}

// ---------------------------------------------------------------- collection

/// Frees a collection handle. Null is ignored.
///
/// # Safety
/// `c` is null or a live handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_collection_free(c: *mut VdbCollection) {
    if !c.is_null() {
        // SAFETY: from `Box::into_raw` here, freed once.
        drop(unsafe { Box::from_raw(c) });
    }
}

/// `{"name", "dim", "metric", "quantization", "chunking", "targetTokens"?, "overlapTokens"?}`.
///
/// # Safety
/// `c` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_collection_info(
    c: *const VdbCollection,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let c = unsafe { handle(c, "collection") }?;
        let info = shared::info_json(&c.name, c.get()?.schema());
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &info) }
    })
}

/// `{"chunks", "sealedSegments", "buffered", "memoryBytes", "maintenanceError"}`.
///
/// # Safety
/// `c` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_collection_stats(
    c: *const VdbCollection,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let stats = shared::stats_json(&unsafe { handle(c, "collection") }?.get()?.index_stats());
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &stats) }
    })
}

/// The texts to embed for `content`, one per chunk, as a JSON array of
/// strings. `format` (may be null): "markdown", "code", "html", "text", a
/// file name, extension or MIME type; sniffed if null.
///
/// # Safety
/// `c` is a live handle; strings are NUL-terminated or (`format`) null;
/// `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_embed_inputs(
    c: *const VdbCollection,
    content: *const c_char,
    format: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, content, format) = unsafe {
            (
                handle(c, "collection")?,
                req_str(content, "content")?,
                opt_str(format, "format")?,
            )
        };
        let inputs = c.get()?.embed_inputs(content, format);
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &json!(inputs)) }
    })
}

/// The chunks of `doc_json` (`{"key", "content", "meta"?, "format"?}`) that
/// need new vectors, as `[{"hash", "text"}]`; unchanged chunks reuse their
/// stored vectors. Pass the hashes to `vdb_write` as `embeddings`.
///
/// # Safety
/// `c` is a live handle; `doc_json` is NUL-terminated; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_plan_embeddings(
    c: *const VdbCollection,
    doc_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, doc) = unsafe { (handle(c, "collection")?, req_json(doc_json, "doc")?) };
        let doc = shared::doc_from_json(&doc).map_err(invalid)?;
        let plan = shared::plan_json(&c.get()?.plan_embeddings(&doc)?);
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &plan) }
    })
}

/// Searches with `query` (`dim` floats). `options_json` (may be null):
/// `{"mode"?: "chunks" | "documents" | "expand" | "sections", "k"?, "ef"?,
/// "filter"?, "text"?, "agg"?: "max" | "sum", "top"?, "neighbors"?}`.
/// `out` gets a JSON array of chunk hits, document hits or passages.
///
/// # Safety
/// `c` is a live handle; `query` holds `dim` floats; `options_json` is
/// NUL-terminated or null; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_search(
    c: *const VdbCollection,
    query: *const f32,
    dim: usize,
    options_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, query, options) = unsafe {
            (
                handle(c, "collection")?,
                floats(query, dim)?,
                opt_json(options_json, "options")?,
            )
        };
        let options = options.unwrap_or_else(|| json!({}));
        let o = obj(&options, "search options")?;
        let coll = c.get()?;
        let mut s = coll.search(query).k(uint_field(o, "k")?.unwrap_or(10));
        if let Some(ef) = uint_field(o, "ef")? {
            s = s.ef(ef);
        }
        if let Some(f) = shared::optional_filter(o.get("filter")).map_err(invalid)? {
            s = s.filter(f);
        }
        if let Some(t) = str_field(o, "text")? {
            s = s.text(t);
        }
        let hits = match str_field(o, "mode")?.as_deref().unwrap_or("chunks") {
            "chunks" => s.chunks()?.iter().map(shared::chunk_hit_json).collect(),
            "documents" => {
                let agg = shared::agg(str_field(o, "agg")?.as_deref(), uint_field(o, "top")?)
                    .map_err(invalid)?;
                s.documents(agg)?.iter().map(shared::doc_hit_json).collect()
            }
            "expand" => {
                let n = uint_field(o, "neighbors")?.unwrap_or(1);
                let n = u32::try_from(n).map_err(|_| invalid("neighbors is too large"))?;
                s.expand(n)?.iter().map(shared::passage_json).collect()
            }
            "sections" => s.section()?.iter().map(shared::passage_json).collect(),
            m => {
                return Err(invalid(format!(
                    "unknown mode {m:?} (chunks, documents, expand, sections)"
                )));
            }
        };
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &Value::Array(hits)) }
    })
}

/// The document with `key`, with its content, or `null`.
///
/// # Safety
/// `c` is a live handle; `key` is NUL-terminated; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_get(
    c: *const VdbCollection,
    key: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, key) = unsafe { (handle(c, "collection")?, req_str(key, "key")?) };
        let doc = c.get()?.get(key)?;
        let v = doc.as_ref().map_or(Value::Null, shared::document_json);
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &v) }
    })
}

/// The chunks of the document with `key`, as a JSON array.
///
/// # Safety
/// `c` is a live handle; `key` is NUL-terminated; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_chunks(
    c: *const VdbCollection,
    key: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, key) = unsafe { (handle(c, "collection")?, req_str(key, "key")?) };
        let chunks: Vec<_> = c
            .get()?
            .chunks(key)?
            .iter()
            .map(shared::chunk_info_json)
            .collect();
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &Value::Array(chunks)) }
    })
}

/// Summaries of the documents matching `filter_json` (null: all), as a
/// JSON array.
///
/// # Safety
/// `c` is a live handle; `filter_json` is NUL-terminated or null; `out` is
/// writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_scan(
    c: *const VdbCollection,
    filter_json: *const c_char,
    out: *mut *mut c_char,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (c, filter) = unsafe { (handle(c, "collection")?, opt_json(filter_json, "filter")?) };
        let filter = shared::optional_filter(filter.as_ref()).map_err(invalid)?;
        let docs: Vec<_> = c
            .get()?
            .scan(filter.as_ref())?
            .iter()
            .map(shared::doc_info_json)
            .collect();
        // SAFETY: by our caller's contract.
        unsafe { put_json(out, &Value::Array(docs)) }
    })
}

/// # Safety
/// `c` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_count(c: *const VdbCollection, out: *mut u64) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let n = unsafe { handle(c, "collection") }?.get()?.len()?;
        // SAFETY: by our caller's contract.
        unsafe { put(out, n as u64) }
    })
}

/// Whether the collection keeps a keyword index for hybrid search.
///
/// # Safety
/// `c` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_has_lexical(c: *const VdbCollection, out: *mut bool) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let has = unsafe { handle(c, "collection") }?.get()?.has_lexical();
        // SAFETY: by our caller's contract.
        unsafe { put(out, has) }
    })
}

/// Builds the keyword index from stored text, without re-embedding.
/// `built` (may be null) is false if it already existed.
///
/// # Safety
/// `c` is a live handle; `built` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_enable_lexical(c: *const VdbCollection, built: *mut bool) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let b = unsafe { handle(c, "collection") }?
            .get()?
            .enable_lexical()?;
        if !built.is_null() {
            // SAFETY: non-null and writable by our caller's contract.
            unsafe { built.write(b) };
        }
        Ok(())
    })
}

// ---------------------------------------------------------------- bulk load

/// A bulk load in progress.
pub struct VdbBulk {
    /// `None` once finished.
    load: Mutex<Option<vdblite::BulkLoad>>,
    dim: usize,
}

/// Chunks between durable commits of a bulk load, by default.
#[unsafe(no_mangle)]
pub extern "C" fn vdb_default_checkpoint() -> u64 {
    vdblite::DEFAULT_CHECKPOINT as u64
}

/// Starts a bulk load into the collection: added documents are readable at
/// once and searchable after `vdb_bulk_finish`. `checkpoint_every`: chunks
/// between durable commits in a file opened with `VDB_OPEN_EXCLUSIVE`
/// (0 for the default).
///
/// # Safety
/// `c` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_load(
    c: *const VdbCollection,
    checkpoint_every: u64,
    out: *mut *mut VdbBulk,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let c = unsafe { handle(c, "collection") }?;
        let coll = c.get()?;
        let mut load = db_of(&c.db)?.bulk_load(&coll)?;
        if checkpoint_every > 0 {
            load = load.checkpoint_every(checkpoint_every as usize);
        }
        let b = Box::new(VdbBulk {
            load: Mutex::new(Some(load)),
            dim: coll.schema().dim,
        });
        // SAFETY: by our caller's contract.
        unsafe { put(out, Box::into_raw(b)) }
    })
}

impl VdbBulk {
    fn with(&self, f: impl FnOnce(&mut vdblite::BulkLoad) -> Res<()>) -> Res<()> {
        let mut guard = self.load.lock().unwrap_or_else(|e| e.into_inner());
        f(guard
            .as_mut()
            .ok_or_else(|| invalid("the bulk load has finished"))?)
    }
}

/// Adds documents in one transaction. `docs_json` is an array of
/// `{"key", "content", "meta"?, "format"?, "vectors": n}`, each taking n
/// vectors (one per chunk) from `vectors` in order. `written` (may be
/// null) gets the number of documents written.
///
/// # Safety
/// `b` is a live handle; `docs_json` is NUL-terminated; `vectors` holds
/// `n_floats` floats; `written` is null or writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_add(
    b: *const VdbBulk,
    docs_json: *const c_char,
    vectors: *const f32,
    n_floats: usize,
    written: *mut u64,
) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let (b, docs, mut rest) = unsafe {
            (
                handle(b, "bulk load")?,
                req_json(docs_json, "docs")?,
                floats(vectors, n_floats)?,
            )
        };
        let Value::Array(items) = docs else {
            return Err(invalid("docs must be an array"));
        };
        let mut docs: Vec<(Doc, Vec<Vec<f32>>)> = Vec::with_capacity(items.len());
        for item in &items {
            let n = uint_field(obj(item, "a document")?, "vectors")?
                .ok_or_else(|| invalid("each document needs vectors (a count)"))?;
            let vs = take_vectors(&mut rest, n, b.dim)?;
            docs.push((shared::doc_from_json(item).map_err(invalid)?, vs));
        }
        if !rest.is_empty() {
            return Err(invalid(format!(
                "{} floats left over after the documents",
                rest.len()
            )));
        }
        b.with(|load| {
            let n = load.add(docs)?;
            if !written.is_null() {
                // SAFETY: non-null and writable by our caller's contract.
                unsafe { written.write(n as u64) };
            }
            Ok(())
        })
    })
}

/// Makes everything added so far durable (only matters with
/// `VDB_OPEN_EXCLUSIVE`).
///
/// # Safety
/// `b` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_checkpoint(b: *const VdbBulk) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        unsafe { handle(b, "bulk load") }?.with(|load| Ok(load.checkpoint()?))
    })
}

/// Builds the index over everything added and makes it searchable. Other
/// writes wait until it's done; reads and searches don't.
///
/// # Safety
/// `b` is a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_finish(b: *const VdbBulk) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let b = unsafe { handle(b, "bulk load") }?;
        let load = b
            .load
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .ok_or_else(|| invalid("the bulk load has finished"))?;
        Ok(load.finish()?)
    })
}

/// Whether `vdb_bulk_finish` has been called.
///
/// # Safety
/// `b` is a live handle; `out` is writable.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_finished(b: *const VdbBulk, out: *mut bool) -> i32 {
    status(|| {
        // SAFETY: by our caller's contract.
        let b = unsafe { handle(b, "bulk load") }?;
        let done = b.load.lock().unwrap_or_else(|e| e.into_inner()).is_none();
        // SAFETY: by our caller's contract.
        unsafe { put(out, done) }
    })
}

/// Frees a bulk load. An unfinished one keeps what was added, which
/// background maintenance indexes. Null is ignored.
///
/// # Safety
/// `b` is null or a live handle, not used again.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn vdb_bulk_free(b: *mut VdbBulk) {
    if !b.is_null() {
        // SAFETY: from `Box::into_raw` here, freed once.
        drop(unsafe { Box::from_raw(b) });
    }
}
