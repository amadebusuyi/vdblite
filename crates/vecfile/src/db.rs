use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, RwLock};
use redb::{ConcurrencyMode, ReadableDatabase, ReadableTable};
use vecfile_core::codec::{self, read_vector};
use vecfile_core::records::{CollectionRecord, decode, encode};
use vecfile_core::tables::{
    BLOBS, CATALOG, CHANGES, CHUNKS, DEAD, DOCS, KEYS, META_INDEX, PENDING, SEGMENT_DATA, SEGMENTS,
    SETTINGS, VECTORS,
};
use vecfile_core::tables::{CHUNK_TERMS, LEXICAL, POSTINGS, TERM_DF};
use vecfile_core::{FORMAT_VERSION, StorageError, changes};
use vecfile_index::VectorIndex;
use vecfile_index::index::SealedEntry;

use crate::collection::{CollInner, Collection};
use crate::error::{Error, Result};
use crate::maintenance::{self, Worker};
use crate::schema::{self, IndexOptions, Schema};
use crate::tx::WriteTx;

const VERSION_KEY: &str = "format_version";
const NEXT_COLLECTION_KEY: &str = "next_collection_id";

/// An open vecfile database: one file, opened like SQLite.
///
/// `Db` is cheap to clone and safe to share across threads. One write
/// transaction runs at a time; searches and reads run concurrently with it.
///
/// One process at a time opens a file for writing ([`Db::open`]); any number
/// of other processes may open it read-only alongside ([`Db::open_read_only`])
/// and see each commit as it lands.
#[derive(Clone)]
pub struct Db {
    pub(crate) inner: DbRef,
}

/// The storage engine handle: read-write in the writing process, read-only
/// elsewhere.
pub(crate) enum Store {
    Writer(redb::Database),
    Reader(redb::ReadOnlyDatabase),
}

impl Store {
    pub(crate) fn begin_read(&self) -> Result<redb::ReadTransaction> {
        Ok(match self {
            Store::Writer(d) => d.begin_read()?,
            Store::Reader(d) => d.begin_read()?,
        })
    }

    /// The read-write handle, or [`Error::ReadOnly`].
    pub(crate) fn writer(&self) -> Result<&redb::Database> {
        match self {
            Store::Writer(d) => Ok(d),
            Store::Reader(_) => Err(Error::ReadOnly),
        }
    }

    pub(crate) fn is_reader(&self) -> bool {
        matches!(self, Store::Reader(_))
    }
}

pub(crate) struct DbInner {
    pub(crate) path: PathBuf,
    /// Dropped by hand in `Drop` so a panic while closing can be contained.
    pub(crate) store: ManuallyDrop<Store>,
    pub(crate) collections: RwLock<HashMap<String, Arc<CollInner>>>,
    /// Held while writing to the file: by the write transaction, and by
    /// maintenance while it installs a segment, so index changes are applied
    /// in commit order.
    pub(crate) writer: Mutex<()>,
    /// Seals and compacts in the background (writers only).
    pub(crate) worker: Arc<Worker>,
}

impl Drop for DbInner {
    fn drop(&mut self) {
        // Closing flushes allocator state, which redb can panic on when the
        // file is corrupted; contain that rather than panicking in a destructor
        // (which would abort if already unwinding).
        let store = &mut self.store;
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            // SAFETY: `store` is dropped exactly once, here, and never used again.
            unsafe { ManuallyDrop::drop(store) }
        }));
    }
}

/// A handle to the open database, held by every [`Db`] and [`Collection`].
///
/// The background maintenance thread holds the database only for short
/// spells while it reads or writes the file. When the last handle is
/// dropped it waits out such a spell, so the file is closed on the dropping
/// thread before `drop` returns and can be reopened at once.
pub(crate) struct DbRef(ManuallyDrop<Arc<DbInner>>);

impl DbRef {
    fn new(inner: DbInner) -> Self {
        DbRef(ManuallyDrop::new(Arc::new(inner)))
    }

    pub(crate) fn ptr_eq(a: &DbRef, b: &DbRef) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    pub(crate) fn downgrade(&self) -> Weak<DbInner> {
        Arc::downgrade(&self.0)
    }
}

impl Deref for DbRef {
    type Target = DbInner;

    fn deref(&self) -> &DbInner {
        &self.0
    }
}

impl Clone for DbRef {
    fn clone(&self) -> Self {
        DbRef(ManuallyDrop::new(Arc::clone(&self.0)))
    }
}

impl Drop for DbRef {
    fn drop(&mut self) {
        // SAFETY: `self.0` is taken exactly once, here, and never used again.
        let inner = unsafe { ManuallyDrop::take(&mut self.0) };
        let worker = inner.worker.clone();
        worker.release(inner);
    }
}

/// How a process opens a file.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// The one writer, with reading processes allowed alongside.
    Shared,
    /// The one writer, with no other process allowed.
    Exclusive,
    /// A reader, alongside a shared writer or other readers.
    ReadOnly,
}

fn builder(mode: Mode) -> redb::Builder {
    let mut b = redb::Builder::new();
    b.set_concurrency_mode(match mode {
        Mode::Exclusive => ConcurrencyMode::ExclusiveWriter,
        // One writing process, any number of reading ones.
        Mode::Shared | Mode::ReadOnly => ConcurrencyMode::SingleWriter,
    });
    b
}

impl Db {
    /// Opens the database at `path` for reading and writing, creating it if
    /// it does not exist. Only one process may have a file open this way at
    /// a time; others can open it with [`Db::open_read_only`].
    ///
    /// Recovery after a crash is automatic: the file always opens at its
    /// last committed transaction. A file written by an older version of
    /// vecfile is upgraded in place.
    pub fn open(path: impl AsRef<Path>) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        crate::error::guard(|| Self::open_inner(path, Mode::Shared))
    }

    /// Like [`Db::open`], but no other process may open the file, not even
    /// to read it. Each commit then needs one disk flush instead of the two
    /// that keep reading processes safe, which roughly halves the latency of
    /// small commits.
    pub fn open_exclusive(path: impl AsRef<Path>) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        crate::error::guard(|| Self::open_inner(path, Mode::Exclusive))
    }

    /// Opens an existing database for reading, alongside the process that
    /// writes it (if any). Searches and reads see every commit the writer
    /// makes, as it lands. Writing through this handle fails with
    /// [`Error::ReadOnly`].
    ///
    /// The file must be consistent: either open in a writing process, or
    /// closed cleanly. After a writer crash, opening it for writing again
    /// repairs it.
    pub fn open_read_only(path: impl AsRef<Path>) -> Result<Db> {
        let path = path.as_ref().to_path_buf();
        crate::error::guard(|| Self::open_inner(path, Mode::ReadOnly))
    }

    fn open_inner(path: PathBuf, mode: Mode) -> Result<Db> {
        let read_only = mode == Mode::ReadOnly;
        let store = if read_only {
            let store = Store::Reader(builder(mode).open_read_only(&path)?);
            check_version(&store)?;
            store
        } else {
            let db = builder(mode).create(&path)?;
            init_settings(&db)?;
            Store::Writer(db)
        };
        let db = Db {
            inner: DbRef::new(DbInner {
                path,
                store: ManuallyDrop::new(store),
                collections: RwLock::new(HashMap::new()),
                writer: Mutex::new(()),
                worker: Arc::new(Worker::default()),
            }),
        };
        db.load_collections()?;
        if !read_only {
            // Pick up maintenance a previous session left due.
            let due = db
                .inner
                .collections
                .read()
                .values()
                .any(|c| maintenance::is_due(c));
            if due {
                db.inner.worker.request(&db.inner);
            }
        }
        Ok(db)
    }

    pub fn path(&self) -> &Path {
        &self.inner.path
    }

    /// Whether this handle was opened with [`Db::open_read_only`].
    pub fn is_read_only(&self) -> bool {
        self.inner.store.is_reader()
    }

    /// Shrinks a closed database file by returning free pages to the OS.
    ///
    /// The file is copy-on-write, so space freed by updates, deletes and
    /// segment merges is reused by later writes but never handed back on its
    /// own. Run this while no process has the file open.
    /// Returns whether anything was reclaimed.
    pub fn compact_file(path: impl AsRef<Path>) -> Result<bool> {
        let path = path.as_ref();
        crate::error::guard(|| {
            let mut store = redb::Database::open(path)?;
            Ok(store.compact().map_err(redb::Error::from)?)
        })
    }

    fn load_collections(&self) -> Result<()> {
        let rx = self.inner.store.begin_read()?;
        let catalog = match rx.open_table(CATALOG) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let mut map = HashMap::new();
        for row in catalog.iter()? {
            let (_, v) = row?;
            let rec: CollectionRecord = decode(v.value(), "collection record")?;
            let inner = load_collection(&rx, &rec)?;
            map.insert(rec.name.clone(), Arc::new(inner));
        }
        *self.inner.collections.write() = map;
        Ok(())
    }

    /// In a reading process, brings the set of collections up to date with
    /// the writer's: new ones are loaded, dropped ones forgotten.
    fn refresh_catalog(&self) -> Result<()> {
        if !self.inner.store.is_reader() {
            return Ok(());
        }
        let rx = self.inner.store.begin_read()?;
        let catalog = match rx.open_table(CATALOG) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let mut records = Vec::new();
        for row in catalog.iter()? {
            let (_, v) = row?;
            records.push(decode::<CollectionRecord>(v.value(), "collection record")?);
        }
        let current: HashMap<String, u32> = self
            .inner
            .collections
            .read()
            .iter()
            .map(|(name, c)| (name.clone(), c.id))
            .collect();
        let fresh: Vec<&CollectionRecord> = records
            .iter()
            .filter(|r| current.get(&r.name) != Some(&r.id))
            .collect();
        if fresh.is_empty() && current.len() == records.len() {
            return Ok(());
        }
        let mut loaded = Vec::with_capacity(fresh.len());
        for rec in fresh {
            loaded.push((rec.name.clone(), Arc::new(load_collection(&rx, rec)?)));
        }
        let mut map = self.inner.collections.write();
        map.retain(|name, c| records.iter().any(|r| &r.name == name && r.id == c.id));
        map.extend(loaded);
        Ok(())
    }

    /// Creates a collection with default index options.
    pub fn create_collection(&self, name: &str, schema: Schema) -> Result<Collection> {
        self.create_collection_with(name, schema, IndexOptions::default())
    }

    /// Creates a collection with explicit index options.
    pub fn create_collection_with(
        &self,
        name: &str,
        schema: Schema,
        opts: IndexOptions,
    ) -> Result<Collection> {
        let store = self.inner.store.writer()?;
        schema::validate(&schema, &opts)?;
        if name.is_empty() {
            return Err(crate::error::invalid("collection name must not be empty"));
        }
        let _w = self.inner.writer.lock();
        if self.inner.collections.read().contains_key(name) {
            return Err(Error::CollectionExists(name.to_owned()));
        }
        let tx = store.begin_write()?;
        let rec = {
            let mut settings = tx.open_table(SETTINGS)?;
            let id = match settings.get(NEXT_COLLECTION_KEY)? {
                Some(v) => decode::<u32>(v.value(), "next collection id")?,
                None => 1,
            };
            settings.insert(NEXT_COLLECTION_KEY, encode(&(id + 1)).as_slice())?;
            let rec = schema::to_record(id, name, &schema, &opts);
            tx.open_table(CATALOG)?
                .insert(name, encode(&rec).as_slice())?;
            if opts.lexical {
                tx.open_table(LEXICAL)?
                    .insert(id, encode(&crate::lexical::empty_stats()).as_slice())?;
            }
            rec
        };
        tx.commit()?;
        let inner = Arc::new(CollInner::new(&rec, schema, opts));
        self.inner
            .collections
            .write()
            .insert(name.to_owned(), inner.clone());
        Ok(Collection {
            inner,
            db: self.inner.clone(),
        })
    }

    /// Opens an existing collection.
    pub fn collection(&self, name: &str) -> Result<Collection> {
        self.refresh_catalog()?;
        let inner = self
            .inner
            .collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NoSuchCollection(name.to_owned()))?;
        Ok(Collection {
            inner,
            db: self.inner.clone(),
        })
    }

    /// Names of all collections, sorted.
    pub fn collections(&self) -> Result<Vec<String>> {
        self.refresh_catalog()?;
        let mut names: Vec<String> = self.inner.collections.read().keys().cloned().collect();
        names.sort();
        Ok(names)
    }

    /// Deletes a collection and everything in it.
    pub fn drop_collection(&self, name: &str) -> Result<()> {
        let store = self.inner.store.writer()?;
        let _w = self.inner.writer.lock();
        let inner = self
            .inner
            .collections
            .read()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::NoSuchCollection(name.to_owned()))?;
        let c = inner.id;
        let tx = store.begin_write()?;
        {
            tx.open_table(CATALOG)?.remove(name)?;
            tx.open_table(DOCS)?
                .retain_in((c, 0)..=(c, u64::MAX), |_, _| false)?;
            tx.open_table(BLOBS)?
                .retain_in((c, 0)..=(c, u64::MAX), |_, _| false)?;
            tx.open_table(CHUNKS)?
                .retain_in((c, 0)..=(c, u32::MAX), |_, _| false)?;
            tx.open_table(VECTORS)?
                .retain_in((c, 0)..=(c, u32::MAX), |_, _| false)?;
            tx.open_table(PENDING)?
                .retain_in((c, 0)..=(c, u32::MAX), |_, _| false)?;
            tx.open_table(DEAD)?
                .retain_in((c, 0)..=(c, u32::MAX), |_, _| false)?;
            tx.open_table(SEGMENTS)?
                .retain_in((c, 0)..=(c, u64::MAX), |_, _| false)?;
            tx.open_table(SEGMENT_DATA)?
                .retain_in((c, 0, 0)..=(c, u64::MAX, u8::MAX), |_, _| false)?;
            tx.open_table(CHANGES)?
                .retain_in((c, 0)..=(c, u64::MAX), |_, _| false)?;
            let mut keys = tx.open_table(KEYS)?;
            keys.retain_in((c, "")..(c + 1, ""), |_, _| false)?;
            let mut idx = tx.open_table(META_INDEX)?;
            idx.retain_in((c, &[][..])..(c + 1, &[][..]), |_, _| false)?;
            tx.open_table(POSTINGS)?
                .retain_in((c, "", 0)..(c + 1, "", 0), |_, _| false)?;
            tx.open_table(TERM_DF)?
                .retain_in((c, "")..(c + 1, ""), |_, _| false)?;
            tx.open_table(CHUNK_TERMS)?
                .retain_in((c, 0)..=(c, u32::MAX), |_, _| false)?;
            tx.open_table(LEXICAL)?.remove(c)?;
        }
        tx.commit()?;
        self.inner.collections.write().remove(name);
        Ok(())
    }

    /// Starts the write transaction. Blocks while another one is open.
    pub fn begin_write(&self) -> Result<WriteTx<'_>> {
        WriteTx::begin(self)
    }

    /// Seals buffered chunks into graphs and merges small segments in every
    /// collection, so searches fan out over fewer segments. This runs on the
    /// calling thread, using every core; writes may continue meanwhile.
    /// Commits schedule the cheaper parts of this in the background; run it
    /// after bulk loads.
    pub fn optimize(&self) -> Result<()> {
        self.inner.store.writer()?;
        let colls: Vec<Arc<CollInner>> = self.inner.collections.read().values().cloned().collect();
        for c in colls {
            maintenance::optimize(&self.inner, &c)?;
        }
        Ok(())
    }

    /// Waits until background maintenance (sealing write buffers into graphs,
    /// rebuilding segments with many deletes) has caught up with every
    /// commit so far, and returns the first error it hit since the last call.
    ///
    /// Commits never wait for maintenance, so this is how a caller (a test,
    /// a benchmark, a bulk load) gets a settled index.
    pub fn wait_for_maintenance(&self) -> Result<()> {
        self.inner.worker.wait_idle();
        let colls: Vec<Arc<CollInner>> = self.inner.collections.read().values().cloned().collect();
        for c in colls {
            if let Some(e) = c.maint_error.lock().take() {
                return Err(Error::Maintenance(Box::new(e)));
            }
        }
        Ok(())
    }
}

/// Creates or upgrades the file's tables in a writing process.
fn init_settings(store: &redb::Database) -> Result<()> {
    // A file already in the current format needs no write (and no flush).
    {
        let rx = store.begin_read()?;
        let current = match rx.open_table(SETTINGS) {
            Ok(settings) => match settings.get(VERSION_KEY)? {
                Some(v) => decode::<u32>(v.value(), "format version")? == FORMAT_VERSION,
                None => false,
            },
            Err(redb::TableError::TableDoesNotExist(_)) => false,
            Err(e) => return Err(e.into()),
        };
        if current {
            return Ok(());
        }
    }
    let tx = store.begin_write()?;
    {
        let mut settings = tx.open_table(SETTINGS)?;
        let found = match settings.get(VERSION_KEY)? {
            Some(v) => Some(decode::<u32>(v.value(), "format version")?),
            None => None,
        };
        match found {
            // Version 1 lacks only the change log, created below.
            Some(1) | None => {
                settings.insert(VERSION_KEY, encode(&FORMAT_VERSION).as_slice())?;
            }
            Some(FORMAT_VERSION) => {}
            Some(v) => {
                return Err(Error::Version {
                    found: v,
                    supported: FORMAT_VERSION,
                });
            }
        }
        // Create every table up front so readers never see a missing one.
        tx.open_table(CATALOG)?;
        tx.open_table(DOCS)?;
        tx.open_table(KEYS)?;
        tx.open_table(BLOBS)?;
        tx.open_table(CHUNKS)?;
        tx.open_table(VECTORS)?;
        tx.open_table(PENDING)?;
        tx.open_table(DEAD)?;
        tx.open_table(META_INDEX)?;
        tx.open_table(SEGMENTS)?;
        tx.open_table(SEGMENT_DATA)?;
        tx.open_table(POSTINGS)?;
        tx.open_table(TERM_DF)?;
        tx.open_table(CHUNK_TERMS)?;
        tx.open_table(LEXICAL)?;
        tx.open_table(CHANGES)?;
    }
    tx.commit()?;
    Ok(())
}

/// Checks a file opened read-only is in a format this build can read.
fn check_version(store: &Store) -> Result<()> {
    let rx = store.begin_read()?;
    let settings = match rx.open_table(SETTINGS) {
        Ok(t) => t,
        // Never initialized by a writer: an empty database.
        Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
        Err(e) => return Err(e.into()),
    };
    match settings.get(VERSION_KEY)? {
        Some(v) => match decode::<u32>(v.value(), "format version")? {
            1 | FORMAT_VERSION => Ok(()),
            v => Err(Error::Version {
                found: v,
                supported: FORMAT_VERSION,
            }),
        },
        None => Ok(()),
    }
}

/// Rebuilds a collection's in-memory index from the file.
pub(crate) fn load_collection(
    rx: &redb::ReadTransaction,
    rec: &CollectionRecord,
) -> Result<CollInner> {
    let (schema, opts) = schema::from_record(rec)?;
    let loaded = load_index(rx, rec)?;
    let opts = IndexOptions {
        lexical: loaded.lexical,
        ..opts
    };
    let coll = CollInner::with_index(rec, schema, opts, loaded.index);
    *coll.synced.lock() = loaded.seq;
    Ok(coll)
}

/// A collection's index as stored in one snapshot of the file.
pub(crate) struct LoadedIndex {
    pub index: VectorIndex,
    /// Whether the collection has a usable keyword index.
    pub lexical: bool,
    /// The last change-log entry the index reflects.
    pub seq: u64,
}

/// Loads a collection's index from the file: sealed segments, their
/// tombstones, and the vectors still waiting in the write buffer.
pub(crate) fn load_index(
    rx: &redb::ReadTransaction,
    rec: &CollectionRecord,
) -> Result<LoadedIndex> {
    let (schema, opts) = schema::from_record(rec)?;
    let mut index = VectorIndex::new(schema::index_config(&schema, &opts));
    let c = rec.id;

    let headers = rx.open_table(SEGMENTS)?;
    let data = rx.open_table(SEGMENT_DATA)?;
    let ids: Vec<u64> = headers
        .range((c, 0)..=(c, u64::MAX))?
        .map(|row| Ok::<_, Error>(row?.0.value().1))
        .collect::<Result<_>>()?;
    // Segments decode independently; load them in parallel.
    let segments: Vec<SealedEntry> = {
        use rayon::prelude::*;
        ids.par_iter()
            .map(|&id| {
                let seg = codec::read_segment(&headers, &data, c, id)?;
                Ok(SealedEntry::new(id, Arc::new(seg)))
            })
            .collect::<Result<_>>()?
    };
    let mut entries: HashMap<u64, SealedEntry> = segments.into_iter().map(|e| (e.id, e)).collect();
    for row in rx.open_table(DEAD)?.range((c, 0)..=(c, u32::MAX))? {
        let (k, v) = row?;
        let chunk = k.value().1;
        let entry = entries.get_mut(&v.value()).ok_or_else(|| {
            StorageError::Corrupt(format!(
                "tombstone for chunk {chunk} names a missing segment"
            ))
        })?;
        if let Some(local) = entry.segment.local_of(chunk) {
            entry.tombstones.set(local);
        }
    }
    for (_, e) in entries {
        index.restore_sealed(e);
    }

    let vectors = rx.open_table(VECTORS)?;
    let mut buf = vec![0f32; schema.dim];
    for row in rx.open_table(PENDING)?.range((c, 0)..=(c, u32::MAX))? {
        let (k, _) = row?;
        let chunk = k.value().1;
        let v = vectors
            .get((c, chunk))?
            .ok_or_else(|| StorageError::Corrupt(format!("pending chunk {chunk} has no vector")))?;
        read_vector(v.value(), &mut buf)?;
        index.insert(chunk, &buf);
    }
    // An index in an older layout counts as absent until rebuilt.
    let lexical = rx
        .open_table(LEXICAL)?
        .get(c)?
        .is_some_and(|v| crate::lexical::read_stats(v.value()).is_some());
    let seq = match rx.open_table(CHANGES) {
        Ok(t) => changes::last_seq(&t, c)?,
        // A version 1 file not yet upgraded has no log.
        Err(redb::TableError::TableDoesNotExist(_)) => 0,
        Err(e) => return Err(e.into()),
    };
    Ok(LoadedIndex {
        index,
        lexical,
        seq,
    })
}
