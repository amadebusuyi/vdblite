use std::collections::HashMap;
use std::mem::ManuallyDrop;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::{Mutex, RwLock};
use redb::{ConcurrencyMode, ReadableDatabase, ReadableTable, ReadableTableMetadata};
use vdblite_core::codec::read_vector;
use vdblite_core::container::{self, Container, Lease, OpenError};
use vdblite_core::records::{CollectionRecord, SegmentRecord, decode, encode};
use vdblite_core::segments::{self, io};
use vdblite_core::tables::{
    BLOBS, BULK, CATALOG, CHANGES, CHUNKS, DEAD, DOCS, FREE_REGIONS, KEYS, META_INDEX, PENDING,
    RESERVED_REGIONS, SEGMENT_DATA, SEGMENTS, SETTINGS, VECTORS,
};
use vdblite_core::tables::{CHUNK_TERMS, LEXICAL, POSTINGS, TERM_DF};
use vdblite_core::{FORMAT_VERSION, StorageError, changes};
use vdblite_index::VectorIndex;
use vdblite_index::index::SealedEntry;

use crate::collection::{CollInner, Collection};
use crate::error::{Error, Result};
use crate::maintenance::{self, Worker};
use crate::schema::{self, IndexOptions, Schema};
use crate::tx::WriteTx;

const VERSION_KEY: &str = "format_version";
const NEXT_COLLECTION_KEY: &str = "next_collection_id";
/// The region epoch: raised by every commit that frees regions.
pub(crate) const EPOCH_KEY: &str = "region_epoch";

/// An open vdblite database: one file, opened like SQLite.
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
    /// The file: redb's pages and the mapped segment regions.
    pub(crate) container: Arc<Container>,
    /// The writer's current region epoch; in a reading process, the newest
    /// one it has seen. Segments are mapped leasing it.
    pub(crate) epoch: AtomicU64,
    /// Opened with [`Db::open_exclusive`]: no other process shares the file.
    pub(crate) exclusive: bool,
    pub(crate) collections: RwLock<HashMap<String, Arc<CollInner>>>,
    /// Held while writing to the file: by the write transaction, and by
    /// maintenance while it installs a segment, so index changes are applied
    /// in commit order.
    pub(crate) writer: Mutex<()>,
    /// Seals and compacts in the background (writers only).
    pub(crate) worker: Arc<Worker>,
}

impl DbInner {
    /// The epoch to lease for segments mapped now.
    pub(crate) fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }

    /// Leases the current epoch. A reading process takes this before it
    /// begins the read transaction it loads segments from, so the writer
    /// can't reuse any region that snapshot references.
    pub(crate) fn pin(&self) -> Result<Lease> {
        self.container
            .lease(self.epoch())
            .map_err(|e| Error::Storage(io(e)))
    }

    /// In a reading process, notes the epoch a snapshot was taken at.
    pub(crate) fn observe(&self, rx: &redb::ReadTransaction) -> Result<()> {
        if self.store.is_reader() {
            let e = read_epoch(rx)?;
            self.epoch.fetch_max(e, Ordering::AcqRel);
        }
        Ok(())
    }
}

/// The region epoch a snapshot records (zero before any region was freed).
pub(crate) fn read_epoch(rx: &redb::ReadTransaction) -> Result<u64> {
    match rx.open_table(SETTINGS) {
        Ok(t) => match t.get(EPOCH_KEY)? {
            Some(v) => Ok(decode::<u64>(v.value(), "region epoch")?),
            None => Ok(0),
        },
        Err(redb::TableError::TableDoesNotExist(_)) => Ok(0),
        Err(e) => Err(e.into()),
    }
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

/// Bytes of file pages the storage engine caches by default (on top of
/// the operating system's own file cache). Measured on 1M 768-dim vectors,
/// 256 MiB searches as fast as 1 GiB; 64 MiB is about 30% slower.
pub const DEFAULT_CACHE_BYTES: usize = 256 << 20;

/// How to open a database; see [`Db::open_with`].
#[derive(Clone, Debug)]
pub struct OpenOptions {
    read_only: bool,
    exclusive: bool,
    cache_bytes: usize,
}

impl Default for OpenOptions {
    fn default() -> Self {
        OpenOptions {
            read_only: false,
            exclusive: false,
            cache_bytes: DEFAULT_CACHE_BYTES,
        }
    }
}

impl OpenOptions {
    /// Read and write, with reading processes allowed alongside, and the
    /// default cache: what [`Db::open`] does.
    pub fn new() -> Self {
        Self::default()
    }

    /// Open read-only, alongside the writing process; see
    /// [`Db::open_read_only`].
    pub fn read_only(mut self, read_only: bool) -> Self {
        self.read_only = read_only;
        self
    }

    /// Shut every other process out; see [`Db::open_exclusive`].
    pub fn exclusive(mut self, exclusive: bool) -> Self {
        self.exclusive = exclusive;
        self
    }

    /// Bytes of file pages to cache in this process (default
    /// [`DEFAULT_CACHE_BYTES`]). The vector index is held in memory
    /// separately; this cache serves documents, records and the vectors read
    /// to re-rank results. The operating system caches the file too, so a
    /// small cache mostly costs system calls, not disk reads.
    pub fn cache_bytes(mut self, bytes: usize) -> Self {
        self.cache_bytes = bytes;
        self
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

fn builder(mode: Mode, cache_bytes: usize) -> redb::Builder {
    let mut b = redb::Builder::new();
    b.set_cache_size(cache_bytes);
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
    /// vdblite is upgraded in place.
    pub fn open(path: impl AsRef<Path>) -> Result<Db> {
        Self::open_with(path, &OpenOptions::new())
    }

    /// Opens the database at `path` as `options` say.
    pub fn open_with(path: impl AsRef<Path>, options: &OpenOptions) -> Result<Db> {
        let mode = match (options.read_only, options.exclusive) {
            (false, false) => Mode::Shared,
            (false, true) => Mode::Exclusive,
            (true, false) => Mode::ReadOnly,
            (true, true) => {
                return Err(crate::error::invalid(
                    "a read-only handle can't be exclusive",
                ));
            }
        };
        let path = path.as_ref().to_path_buf();
        crate::error::guard(|| Self::open_inner(path, mode, options.cache_bytes))
    }

    /// Like [`Db::open`], but no other process may open the file, not even
    /// to read it. Each commit then needs one disk flush instead of the two
    /// that keep reading processes safe, which roughly halves the latency of
    /// small commits.
    pub fn open_exclusive(path: impl AsRef<Path>) -> Result<Db> {
        Self::open_with(path, &OpenOptions::new().exclusive(true))
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
        Self::open_with(path, &OpenOptions::new().read_only(true))
    }

    fn open_inner(path: PathBuf, mode: Mode, cache_bytes: usize) -> Result<Db> {
        let read_only = mode == Mode::ReadOnly;
        let container = open_container(&path, read_only)?;
        let backend = container.redb_backend();
        let store = if read_only {
            let store =
                Store::Reader(builder(mode, cache_bytes).open_read_only_with_backend(backend)?);
            check_version(&store)?;
            store
        } else {
            let db = builder(mode, cache_bytes).create_with_backend(backend)?;
            init_settings(&db, &container)?;
            Store::Writer(db)
        };
        let epoch = read_epoch(&store.begin_read()?)?;
        let db = Db {
            inner: DbRef::new(DbInner {
                path,
                store: ManuallyDrop::new(store),
                container,
                epoch: AtomicU64::new(epoch),
                exclusive: mode == Mode::Exclusive,
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
            let before = std::fs::metadata(path)
                .map_err(|e| Error::Storage(io(e)))?
                .len();
            let tmp = path.with_extension("vdb-compact");
            let _ = std::fs::remove_file(&tmp);
            {
                let container = open_container(path, false)?;
                // Exclusive: fails while any other process has the file open.
                let mut store = exclusive(&container)?;
                init_settings(&store, &container)?;
                store.compact().map_err(redb::Error::from)?;
                // Live regions, packed after redb's pages in a fresh file.
                let records: Vec<((u32, u64), SegmentRecord)> = {
                    let rx = store.begin_read()?;
                    rx.open_table(SEGMENTS)?
                        .iter()?
                        .map(|row| {
                            let (k, v) = row?;
                            Ok((k.value(), decode(v.value(), "segment record")?))
                        })
                        .collect::<Result<_>>()?
                };
                let regions: Vec<(u64, u64)> = records.iter().map(|(_, r)| r.region).collect();
                let placed = container::rewrite(&container, &tmp, &regions)
                    .map_err(|e| Error::Storage(io(e)))?;
                drop(store);
                let fresh = Container::open(&tmp, true).map_err(|e| match e {
                    OpenError::Io(e) => Error::Storage(io(e)),
                    _ => Error::Storage(StorageError::Corrupt("compacted copy".into())),
                })?;
                let store = exclusive(&fresh)?;
                let tx = store.begin_write()?;
                {
                    let mut headers = tx.open_table(SEGMENTS)?;
                    for ((key, mut rec), at) in records.into_iter().zip(placed) {
                        rec.region.0 = at;
                        headers.insert(key, encode(&rec).as_slice())?;
                    }
                    tx.open_table(FREE_REGIONS)?.retain(|_, _| false)?;
                    tx.open_table(RESERVED_REGIONS)?.retain(|_, _| false)?;
                }
                tx.commit()?;
            }
            container::replace(&tmp, path).map_err(|e| Error::Storage(io(e)))?;
            let after = std::fs::metadata(path)
                .map_err(|e| Error::Storage(io(e)))?
                .len();
            Ok(after < before)
        })
    }

    fn load_collections(&self) -> Result<()> {
        let _pin = self.inner.pin()?;
        let rx = self.inner.store.begin_read()?;
        self.inner.observe(&rx)?;
        let catalog = match rx.open_table(CATALOG) {
            Ok(t) => t,
            Err(redb::TableError::TableDoesNotExist(_)) => return Ok(()),
            Err(e) => return Err(e.into()),
        };
        let mut map = HashMap::new();
        for row in catalog.iter()? {
            let (_, v) = row?;
            let rec: CollectionRecord = decode(v.value(), "collection record")?;
            let inner = load_collection(&rx, &rec, &self.inner)?;
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
        let _pin = self.inner.pin()?;
        let rx = self.inner.store.begin_read()?;
        self.inner.observe(&rx)?;
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
            loaded.push((
                rec.name.clone(),
                Arc::new(load_collection(&rx, rec, &self.inner)?),
            ));
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
        let mut epoch = self.inner.epoch();
        {
            // Its segments' regions become free once no process can read them.
            let mut free = tx.open_table(FREE_REGIONS)?;
            let mut regions = Vec::new();
            for row in tx.open_table(SEGMENTS)?.range((c, 0)..=(c, u64::MAX))? {
                let (_, v) = row?;
                regions.push(decode::<SegmentRecord>(v.value(), "segment record")?.region);
            }
            let mut bulk = tx.open_table(BULK)?;
            let checkpoints: Vec<(u64, u64)> = bulk
                .range((c, 0)..=(c, u64::MAX))?
                .map(|row| {
                    let (_, v) = row?;
                    Ok(
                        decode::<vdblite_core::records::BulkRecord>(v.value(), "bulk checkpoint")?
                            .region,
                    )
                })
                .collect::<Result<_>>()?;
            bulk.retain_in((c, 0)..=(c, u64::MAX), |_, _| false)?;
            for r in checkpoints {
                segments::release(&mut free, r, 0)?;
            }
            drop(bulk);
            if !regions.is_empty() {
                epoch += 1;
                tx.open_table(SETTINGS)?
                    .insert(EPOCH_KEY, encode(&epoch).as_slice())?;
                for r in regions {
                    segments::release(&mut free, r, epoch)?;
                }
            }
            drop(free);
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
        self.inner.epoch.store(epoch, Ordering::Release);
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

/// The redb store in `c`, opened with every other process shut out.
fn exclusive(c: &Arc<Container>) -> Result<redb::Database> {
    let mut b = redb::Builder::new();
    b.set_concurrency_mode(ConcurrencyMode::ExclusiveWriter);
    Ok(b.create_with_backend(c.redb_backend())?)
}

/// Opens the file's container, upgrading a file from before regions (a
/// bare redb file) when opening it for writing.
fn open_container(path: &Path, read_only: bool) -> Result<Arc<Container>> {
    let storage = |e: OpenError| match e {
        OpenError::Io(e) => Error::Storage(io(e)),
        OpenError::Corrupt(m) => Error::Storage(StorageError::Corrupt(m)),
        OpenError::Legacy => Error::Version {
            found: 2,
            supported: FORMAT_VERSION,
        },
    };
    match Container::open(path, !read_only) {
        Err(OpenError::Legacy) if !read_only => {
            // The rewrite needs the file to itself: an exclusive open fails
            // while any other process has it.
            {
                let mut b = redb::Builder::new();
                b.set_concurrency_mode(ConcurrencyMode::ExclusiveWriter);
                b.open(path)?;
            }
            container::upgrade_legacy(path).map_err(|e| Error::Storage(io(e)))?;
            Container::open(path, true).map_err(storage)
        }
        other => other.map_err(storage),
    }
}

/// Creates or upgrades the file's tables in a writing process, and returns
/// to the free list any region a crash left reserved.
fn init_settings(store: &redb::Database, container: &Arc<Container>) -> Result<()> {
    // A file already in the current format, with nothing reserved, needs no
    // write (and no flush).
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
        let reserved = match rx.open_table(RESERVED_REGIONS) {
            Ok(t) => !t.is_empty()?,
            Err(redb::TableError::TableDoesNotExist(_)) => false,
            Err(e) => return Err(e.into()),
        };
        if current && !reserved {
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
        let migrate = match found {
            // A new file.
            None => false,
            // Version 1 lacks the change log, created below; versions 1 and
            // 2 keep segments as values.
            Some(1 | 2) => true,
            Some(FORMAT_VERSION) => false,
            Some(v) => {
                return Err(Error::Version {
                    found: v,
                    supported: FORMAT_VERSION,
                });
            }
        };
        settings.insert(VERSION_KEY, encode(&FORMAT_VERSION).as_slice())?;
        drop(settings);
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
        tx.open_table(BULK)?;
        let mut free = tx.open_table(FREE_REGIONS)?;
        // Never referenced by a committed record: reusable at once.
        let mut reserved = tx.open_table(RESERVED_REGIONS)?;
        let left: Vec<(u64, u64)> = reserved
            .iter()?
            .map(|r| r.map(|(k, v)| (k.value(), v.value())))
            .collect::<Result<_, _>>()?;
        for (offset, len) in left {
            reserved.remove(offset)?;
            segments::release(&mut free, (offset, len), 0)?;
        }
        if migrate {
            migrate_segments(&tx, &mut free, container)?;
        }
    }
    tx.commit()?;
    Ok(())
}

/// Moves a version 2 file's sealed segments from `segment_data` values into
/// regions, with their chunks' full vectors, which leave the `vectors`
/// table (it keeps only the write buffer's).
fn migrate_segments(
    tx: &redb::WriteTransaction,
    free: &mut segments::FreeTable<'_>,
    container: &Arc<Container>,
) -> Result<()> {
    let mut headers = tx.open_table(SEGMENTS)?;
    let mut data = tx.open_table(SEGMENT_DATA)?;
    let mut vectors = tx.open_table(VECTORS)?;
    let keys: Vec<(u32, u64)> = headers
        .iter()?
        .map(|r| r.map(|(k, _)| k.value()))
        .collect::<Result<_, _>>()?;
    let mut records = Vec::with_capacity(keys.len());
    let chunks = tx.open_table(CHUNKS)?;
    let docs = tx.open_table(DOCS)?;
    for &(coll, id) in &keys {
        let mut seg = segments::read_legacy_segment(&headers, &data, &vectors, coll, id)?;
        crate::collection::label_with_keys(&chunks, &docs, coll, &mut seg)?;
        let at = segments::allocate(free, container, segments::region_len(&seg))?;
        records.push(((coll, id), segments::write_segment(container, &seg, at)?));
        for &chunk in seg.ids() {
            vectors.remove((coll, chunk))?;
        }
    }
    container.sync().map_err(io)?;
    for (key, rec) in records {
        headers.insert(key, encode(&rec).as_slice())?;
    }
    data.retain(|_, _| false)?;
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
            FORMAT_VERSION => Ok(()),
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
    db: &DbInner,
) -> Result<CollInner> {
    let (schema, opts) = schema::from_record(rec)?;
    let loaded = load_index(rx, rec, db)?;
    let opts = IndexOptions {
        lexical: loaded.lexical,
        ..opts
    };
    let coll = CollInner::with_index(rec, schema, opts, loaded.index);
    *coll.synced.lock() = loaded.seq;
    if !db.store.is_reader() {
        let leftover = match rx.open_table(BULK) {
            Ok(t) => t.range((rec.id, 0)..=(rec.id, u64::MAX))?.next().is_some(),
            Err(redb::TableError::TableDoesNotExist(_)) => false,
            Err(e) => return Err(e.into()),
        };
        coll.bulk_leftover
            .store(leftover, std::sync::atomic::Ordering::Release);
    }
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

/// Loads a collection's index from the file: sealed segments (mapped, not
/// copied), their tombstones, and the vectors still in the write buffer.
pub(crate) fn load_index(
    rx: &redb::ReadTransaction,
    rec: &CollectionRecord,
    db: &DbInner,
) -> Result<LoadedIndex> {
    let (schema, opts) = schema::from_record(rec)?;
    let mut index = VectorIndex::new(schema::index_config(&schema, &opts));
    let c = rec.id;

    let headers = rx.open_table(SEGMENTS)?;
    let records: Vec<(u64, SegmentRecord)> = headers
        .range((c, 0)..=(c, u64::MAX))?
        .map(|row| {
            let (k, v) = row?;
            Ok((k.value().1, decode(v.value(), "segment record")?))
        })
        .collect::<Result<_>>()?;
    // Segments check their checksums independently; map them in parallel.
    let epoch = db.epoch();
    let segments: Vec<SealedEntry> = {
        use rayon::prelude::*;
        records
            .par_iter()
            .map(|(id, rec)| {
                let seg = segments::map_segment(&db.container, rec, epoch)?;
                Ok(SealedEntry::new(*id, Arc::new(seg)))
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
