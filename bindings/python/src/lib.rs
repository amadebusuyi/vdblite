//! Python bindings for vdblite (PyO3).
//!
//! A thin layer over the Rust API for vectors the caller computes: open a
//! file, create or open collections, upsert documents with their vectors,
//! and search. Every call that touches the file or the index releases the
//! GIL, so Python threads search in parallel.
//!
//! The file closes when the `Db` and every `Collection` taken from it are
//! gone.

use std::path::PathBuf;
use std::sync::Mutex;

use pyo3::buffer::PyBuffer;
use pyo3::create_exception;
use pyo3::exceptions::{PyException, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use vdblite::{Chunking, Doc, HnswParams, IndexOptions, Metric, OpenOptions, Quant, Schema};

create_exception!(vdblite, VdbliteError, PyException);

fn err(e: vdblite::Error) -> PyErr {
    VdbliteError::new_err(e.to_string())
}

fn metric(name: &str) -> PyResult<Metric> {
    match name.to_ascii_lowercase().as_str() {
        "cosine" => Ok(Metric::Cosine),
        "dot" | "ip" => Ok(Metric::Dot),
        "l2" => Ok(Metric::L2),
        _ => Err(PyValueError::new_err(format!(
            "metric must be cosine, dot or l2, not {name:?}"
        ))),
    }
}

fn quant(name: &str) -> PyResult<Quant> {
    match name.to_ascii_lowercase().as_str() {
        "int8" => Ok(Quant::Int8),
        "none" | "f32" => Ok(Quant::None),
        "binary" | "1bit" => Ok(Quant::Binary),
        _ => Err(PyValueError::new_err(format!(
            "quant must be int8, binary or none, not {name:?}"
        ))),
    }
}

/// An open `.vdb` file.
#[pyclass(frozen, module = "vdblite")]
struct Db {
    db: vdblite::Db,
}

#[pymethods]
impl Db {
    /// Opens (or creates) the database at `path`. One process may open a
    /// file for writing; others may open it with `read_only=True` alongside.
    /// `exclusive=True` shuts other processes out and halves commit flushes.
    #[new]
    #[pyo3(signature = (path, *, read_only = false, exclusive = false, cache_mb = None))]
    fn new(
        py: Python<'_>,
        path: PathBuf,
        read_only: bool,
        exclusive: bool,
        cache_mb: Option<usize>,
    ) -> PyResult<Self> {
        let mut options = OpenOptions::new().read_only(read_only).exclusive(exclusive);
        if let Some(mb) = cache_mb {
            options = options.cache_bytes(mb << 20);
        }
        let db = py
            .detach(|| vdblite::Db::open_with(&path, &options))
            .map_err(err)?;
        Ok(Db { db })
    }

    /// Creates a collection of unchunked documents with one vector each.
    #[pyo3(signature = (
        name, dim, *, metric = "cosine", quant = "int8", m = 16, ef_construction = 200,
        ef_search = 64, seal_threshold = None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_collection(
        &self,
        name: &str,
        dim: usize,
        metric: &str,
        quant: &str,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        seal_threshold: Option<usize>,
    ) -> PyResult<Collection> {
        let schema = Schema {
            dim,
            metric: self::metric(metric)?,
            quant: self::quant(quant)?,
            chunking: Chunking::None,
        };
        let defaults = IndexOptions::default();
        let options = IndexOptions {
            hnsw: HnswParams {
                m,
                ef_construction,
                ef_search,
                ..HnswParams::default()
            },
            seal_threshold: seal_threshold.unwrap_or(defaults.seal_threshold),
            lexical: false,
        };
        let coll = self
            .db
            .create_collection_with(name, schema, options)
            .map_err(err)?;
        Ok(Collection {
            db: self.db.clone(),
            coll,
        })
    }

    /// An existing collection.
    fn collection(&self, name: &str) -> PyResult<Collection> {
        let coll = self.db.collection(name).map_err(err)?;
        Ok(Collection {
            db: self.db.clone(),
            coll,
        })
    }

    fn collections(&self) -> PyResult<Vec<String>> {
        self.db.collections().map_err(err)
    }

    fn drop_collection(&self, py: Python<'_>, name: &str) -> PyResult<()> {
        py.detach(|| self.db.drop_collection(name)).map_err(err)
    }

    /// Seals buffered vectors into graphs and merges segments, using every
    /// core. Run it after a bulk load.
    fn optimize(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.db.optimize()).map_err(err)
    }

    /// Blocks until background maintenance has caught up with every commit.
    fn wait_for_maintenance(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.db.wait_for_maintenance()).map_err(err)
    }

    /// Rewrites a closed file without its free space. Returns whether it
    /// shrank.
    #[staticmethod]
    fn compact_file(py: Python<'_>, path: PathBuf) -> PyResult<bool> {
        py.detach(|| vdblite::Db::compact_file(&path)).map_err(err)
    }
}

/// A collection in an open database.
#[pyclass(frozen, module = "vdblite")]
struct Collection {
    db: vdblite::Db,
    coll: vdblite::Collection,
}

#[pymethods]
impl Collection {
    /// Inserts or replaces documents, one vector each, in one transaction.
    /// `contents` default to the keys: a document needs some content to
    /// hold its chunk. Returns the number written.
    #[pyo3(signature = (keys, vectors, contents = None))]
    fn upsert_vectors(
        &self,
        py: Python<'_>,
        keys: Vec<String>,
        vectors: Vec<Vec<f32>>,
        contents: Option<Vec<String>>,
    ) -> PyResult<usize> {
        if vectors.len() != keys.len() || contents.as_ref().is_some_and(|c| c.len() != keys.len()) {
            return Err(PyValueError::new_err(
                "keys, vectors and contents must have the same length",
            ));
        }
        let n = keys.len();
        py.detach(|| {
            let mut tx = self.db.begin_write()?;
            let mut contents = contents.map(Vec::into_iter);
            for (key, v) in keys.into_iter().zip(vectors) {
                let content = match contents.as_mut().and_then(Iterator::next) {
                    Some(c) => c,
                    None => key.clone(),
                };
                tx.upsert_with_vectors(&self.coll, Doc::new(key).content(content), vec![v])?;
            }
            tx.commit()
        })
        .map_err(err)?;
        Ok(n)
    }

    /// The `k` nearest documents as `(key, score)`, best first. Higher
    /// scores are closer: cosine similarity, inner product, or negative
    /// squared L2 distance. `ef` widens the graph search (default: the
    /// collection's `ef_search`).
    #[pyo3(signature = (vector, k = 10, *, ef = None))]
    fn search(
        &self,
        py: Python<'_>,
        vector: Vec<f32>,
        k: usize,
        ef: Option<usize>,
    ) -> PyResult<Vec<(String, f32)>> {
        py.detach(|| {
            let mut s = self.coll.search(&vector).k(k);
            if let Some(ef) = ef {
                s = s.ef(ef);
            }
            s.keys()
        })
        .map_err(err)
    }

    /// Starts a bulk load: documents go in quickly, and are indexed once,
    /// with every core, by `finish()`. Until then they can be read but not
    /// searched. `checkpoint_every` chunks, what's loaded so far is made
    /// durable.
    #[pyo3(signature = (*, checkpoint_every = None))]
    fn bulk_load(&self, checkpoint_every: Option<usize>) -> PyResult<BulkLoad> {
        let mut load = self.db.bulk_load(&self.coll).map_err(err)?;
        if let Some(n) = checkpoint_every {
            load = load.checkpoint_every(n);
        }
        Ok(BulkLoad {
            load: Mutex::new(Some(load)),
        })
    }

    /// Index size and state.
    fn index_stats<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let s = self.coll.index_stats();
        let d = PyDict::new(py);
        d.set_item("chunks", s.chunks)?;
        d.set_item("sealed_segments", s.sealed_segments)?;
        d.set_item("buffered", s.buffered)?;
        d.set_item("memory_bytes", s.memory_bytes)?;
        d.set_item("maintenance_error", s.maintenance_error)?;
        Ok(d)
    }
}

/// The rows of a 2-D float32 buffer (a NumPy array, say), or of a list of
/// lists.
fn rows(vectors: &Bound<'_, PyAny>) -> PyResult<Vec<Vec<f32>>> {
    if let Ok(buf) = PyBuffer::<f32>::get(vectors) {
        let shape = buf.shape();
        if shape.len() != 2 {
            return Err(PyValueError::new_err("vectors must be two-dimensional"));
        }
        let dim = shape[1];
        let flat = buf.to_vec(vectors.py())?;
        return Ok(flat.chunks(dim.max(1)).map(<[f32]>::to_vec).collect());
    }
    vectors.extract()
}

/// A bulk load in progress (see `Collection.bulk_load`).
#[pyclass(frozen, module = "vdblite")]
struct BulkLoad {
    load: Mutex<Option<vdblite::BulkLoad>>,
}

impl BulkLoad {
    fn with<T: Send>(
        &self,
        py: Python<'_>,
        f: impl FnOnce(&mut vdblite::BulkLoad) -> vdblite::Result<T> + Send,
    ) -> PyResult<T> {
        py.detach(|| {
            let mut guard = self.load.lock().unwrap_or_else(|e| e.into_inner());
            let load = guard
                .as_mut()
                .ok_or_else(|| VdbliteError::new_err("the bulk load has finished"))?;
            f(load).map_err(err)
        })
    }
}

#[pymethods]
impl BulkLoad {
    /// Adds documents, one vector each (`vectors`: a 2-D float32 array, or
    /// a list of lists). `contents` default to the keys. Returns how many
    /// were written.
    #[pyo3(signature = (keys, vectors, contents = None))]
    fn add(
        &self,
        py: Python<'_>,
        keys: Vec<String>,
        vectors: &Bound<'_, PyAny>,
        contents: Option<Vec<String>>,
    ) -> PyResult<usize> {
        let vectors = rows(vectors)?;
        if vectors.len() != keys.len() || contents.as_ref().is_some_and(|c| c.len() != keys.len()) {
            return Err(PyValueError::new_err(
                "keys, vectors and contents must have the same length",
            ));
        }
        self.with(py, move |load| {
            let mut contents = contents.map(Vec::into_iter);
            let docs = keys.into_iter().zip(vectors).map(|(key, v)| {
                let content = contents
                    .as_mut()
                    .and_then(Iterator::next)
                    .unwrap_or_else(|| key.clone());
                (Doc::new(key).content(content), vec![v])
            });
            load.add(docs)
        })
    }

    /// Makes everything added so far durable.
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        self.with(py, |load| load.checkpoint())
    }

    /// Builds the index over everything added, and makes it searchable.
    fn finish(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let load = self
                .load
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .take()
                .ok_or_else(|| VdbliteError::new_err("the bulk load has finished"))?;
            load.finish().map_err(err)
        })
    }
}

#[pymodule(name = "vdblite")]
fn vdblite_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Db>()?;
    m.add_class::<Collection>()?;
    m.add_class::<BulkLoad>()?;
    m.add("VdbliteError", m.py().get_type::<VdbliteError>())?;
    Ok(())
}
