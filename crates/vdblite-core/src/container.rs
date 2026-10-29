//! The `.vdb` file: redb's pages and immutable regions in one file.
//!
//! redb owns a *logical* file, which this module maps onto *extents* of the
//! physical one. The rest of the physical file holds *regions*: immutable,
//! page-aligned byte ranges (sealed segments) that every process maps
//! read-only and reads in place.
//!
//! ```text
//! [superblock A | superblock B | extent | region | region | extent | ...]
//! ```
//!
//! **Superblocks.** Two 64 KiB slots, each holding the magic, a sequence
//! number, redb's logical length and its extent list, then an xxh3
//! checksum. A change writes the slot not holding the newest copy and syncs
//! it before anything relies on it, so one valid copy always survives a
//! crash. Extents only ever get added, and only grow the file.
//!
//! **Regions.** Written once, synced, and only then referenced by a redb
//! record committed in the same transaction as the rest of the change, so a
//! crash leaves at most unreferenced space. The space of a region nothing
//! references any more is reused only when no process can still be reading
//! it (see [`Container::reusable`]).
//!
//! **Epochs.** The writer stamps freed regions with an epoch it raises at
//! every free. Every mapped region carries a [`Lease`] on the epoch it was
//! mapped at, and each process holds a shared byte-range lock on the oldest
//! epoch it still leases (bytes [`EPOCH_LOCK_BASE`]` + epoch`). A region
//! freed at epoch `f` is reusable once no process holds a lock below `f`.
//! A crashed process's locks go with it.

use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io;
use std::ops::Bound;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use memmap2::{Mmap, MmapOptions};
use parking_lot::{Mutex, RwLock};
use redb::StorageBackend;
use redb::backends::FileBackend;
use vdblite_index::Backing;
use xxhash_rust::xxh3::xxh3_64;

/// First bytes of a `.vdb` file.
pub const MAGIC: [u8; 8] = *b"VDBLITE\x01";
/// redb's magic number: a file starting with it is a database written
/// before regions existed (a bare redb file).
const REDB_MAGIC: [u8; 4] = *b"redb";
/// Layout version of the superblock.
const CONTAINER_VERSION: u32 = 1;
const SLOT_SIZE: u64 = 64 * 1024;
/// Physical bytes before the first extent or region.
pub const HEADER_SIZE: u64 = 2 * SLOT_SIZE;
/// Alignment of every extent and region.
pub const PAGE: u64 = 4096;
/// Bytes of a slot before the extent list.
const SLOT_FIXED: usize = 8 + 4 + 4 + 8 + 8 + 4 + 4;
/// Most extents a slot can list.
const MAX_EXTENTS: usize = (SLOT_SIZE as usize - SLOT_FIXED - 8) / 16;
const MIN_EXTENT: u64 = 1 << 20;
const MAX_EXTENT: u64 = 256 << 20;

/// Base of the epoch lock bytes, well clear of redb's (from `1 << 62`).
pub const EPOCH_LOCK_BASE: u64 = 1 << 61;
/// Held exclusively while a new file's superblocks are first written.
const INIT_LOCK_BYTE: u64 = EPOCH_LOCK_BASE - 1;

fn byte(b: u64) -> (Bound<u64>, Bound<u64>) {
    (Bound::Included(b), Bound::Included(b))
}

fn round_up(x: u64, to: u64) -> u64 {
    x.div_ceil(to) * to
}

fn backend_io(e: redb::BackendError) -> io::Error {
    io::Error::other(e.to_string())
}

/// Why a file could not be opened as a container.
#[derive(Debug)]
pub enum OpenError {
    Io(io::Error),
    /// A bare redb file from before regions existed; see [`upgrade_legacy`].
    Legacy,
    /// Neither superblock is valid, or the file isn't a vdblite file.
    Corrupt(String),
}

impl From<io::Error> for OpenError {
    fn from(e: io::Error) -> Self {
        OpenError::Io(e)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Extent {
    phys: u64,
    len: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct Superblock {
    seq: u64,
    logical_len: u64,
    extents: Vec<Extent>,
}

impl Superblock {
    fn capacity(&self) -> u64 {
        self.extents.iter().map(|e| e.len).sum()
    }

    fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(SLOT_FIXED + 16 * self.extents.len() + 8);
        out.extend_from_slice(&MAGIC);
        out.extend_from_slice(&CONTAINER_VERSION.to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        out.extend_from_slice(&self.seq.to_le_bytes());
        out.extend_from_slice(&self.logical_len.to_le_bytes());
        out.extend_from_slice(&(self.extents.len() as u32).to_le_bytes());
        out.extend_from_slice(&0u32.to_le_bytes());
        for e in &self.extents {
            out.extend_from_slice(&e.phys.to_le_bytes());
            out.extend_from_slice(&e.len.to_le_bytes());
        }
        let sum = xxh3_64(&out);
        out.extend_from_slice(&sum.to_le_bytes());
        out
    }

    /// Decodes a slot; `None` if it is torn, corrupt or never written.
    fn decode(slot: &[u8]) -> Option<Superblock> {
        let u32_at = |i: usize| u32::from_le_bytes(slot[i..i + 4].try_into().unwrap());
        let u64_at = |i: usize| u64::from_le_bytes(slot[i..i + 8].try_into().unwrap());
        if slot.len() < SLOT_FIXED + 8 || slot[..8] != MAGIC || u32_at(8) != CONTAINER_VERSION {
            return None;
        }
        let count = u32_at(32) as usize;
        if count > MAX_EXTENTS {
            return None;
        }
        let body = SLOT_FIXED + 16 * count;
        if u64_at(body) != xxh3_64(&slot[..body]) {
            return None;
        }
        let extents = (0..count)
            .map(|i| Extent {
                phys: u64_at(SLOT_FIXED + 16 * i),
                len: u64_at(SLOT_FIXED + 16 * i + 8),
            })
            .collect();
        let sb = Superblock {
            seq: u64_at(16),
            logical_len: u64_at(24),
            extents,
        };
        sb.valid().then_some(sb)
    }

    /// Extents are page-aligned, inside the file's body, and cover the
    /// logical length.
    fn valid(&self) -> bool {
        self.extents.iter().all(|e| {
            e.phys >= HEADER_SIZE
                && e.phys.is_multiple_of(PAGE)
                && e.len > 0
                && e.len.is_multiple_of(PAGE)
                && e.phys.checked_add(e.len).is_some()
        }) && self.logical_len <= self.capacity()
    }

    /// The physical offset of logical byte `off`, and how many bytes of the
    /// same extent follow it.
    fn translate(&self, off: u64) -> Option<(u64, u64)> {
        let mut start = 0;
        for e in &self.extents {
            if off < start + e.len {
                let within = off - start;
                return Some((e.phys + within, e.len - within));
            }
            start += e.len;
        }
        None
    }
}

/// A read-only mapping of the whole file, as long as it was when mapped.
struct Mapping {
    map: Mmap,
}

/// In-process accounting of leased epochs, and the lock that publishes the
/// oldest one to other processes.
#[derive(Default)]
struct Epochs {
    live: BTreeMap<u64, usize>,
    /// The epoch whose lock byte this process holds shared.
    held: Option<u64>,
}

/// One `.vdb` file, open in this process.
pub struct Container {
    path: PathBuf,
    /// All physical reads, writes and locks go through redb's backend, so
    /// this process has one lock owner for the file.
    file: FileBackend,
    /// A second handle, used only to map the file. It stays open as long as
    /// the container: closing a descriptor can drop a process's POSIX
    /// record locks, so none is closed early.
    map_file: File,
    writable: bool,
    sb: RwLock<Superblock>,
    /// Highest logical length redb has had; growing back below it must
    /// zero the bytes, as redb expects new space to read as zeros.
    high_water: Mutex<u64>,
    /// The physical end: where the next extent or region goes (writers).
    end: Mutex<u64>,
    mapping: Mutex<Option<Arc<Mapping>>>,
    epochs: Mutex<Epochs>,
}

impl std::fmt::Debug for Container {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Container")
            .field("path", &self.path)
            .finish()
    }
}

impl Container {
    /// Opens the container at `path`, creating it when `writable` and absent
    /// (or empty). A bare redb file is [`OpenError::Legacy`].
    pub fn open(path: &Path, writable: bool) -> Result<Arc<Container>, OpenError> {
        let mut opts = OpenOptions::new();
        opts.read(true);
        if writable {
            opts.write(true).create(true);
        }
        let file = opts.open(path)?;
        let map_file = File::open(path)?;
        let backend =
            FileBackend::new(file).map_err(|e| OpenError::Io(io::Error::other(e.to_string())))?;

        let len = StorageBackend::len(&backend)?;
        if len == 0 && writable {
            Self::initialize(&backend)?;
        }
        let sb = Self::read_superblock(&backend)?;
        let len = StorageBackend::len(&backend)?;
        let high = sb.logical_len;
        Ok(Arc::new(Container {
            path: path.to_path_buf(),
            file: backend,
            map_file,
            writable,
            sb: RwLock::new(sb),
            high_water: Mutex::new(high),
            end: Mutex::new(round_up(len.max(HEADER_SIZE), PAGE)),
            mapping: Mutex::new(None),
            epochs: Mutex::new(Epochs::default()),
        }))
    }

    /// Writes the superblocks of a new, empty file. Two processes creating
    /// the same file at once serialize on a lock byte.
    fn initialize(backend: &FileBackend) -> Result<(), OpenError> {
        backend
            .lock_range(byte(INIT_LOCK_BYTE).0, byte(INIT_LOCK_BYTE).1)
            .map_err(|e| OpenError::Io(backend_io(e)))?;
        let result = (|| {
            if StorageBackend::len(backend)? == 0 {
                let sb = Superblock {
                    seq: 1,
                    logical_len: 0,
                    extents: Vec::new(),
                };
                backend.set_len(HEADER_SIZE)?;
                backend.write(0, &sb.encode())?;
                backend.sync_data()?;
            }
            Ok::<_, io::Error>(())
        })();
        let _ = backend.unlock_range(byte(INIT_LOCK_BYTE).0, byte(INIT_LOCK_BYTE).1);
        result.map_err(OpenError::Io)
    }

    /// Reads both superblock slots and returns the newest valid one.
    fn read_superblock(backend: &FileBackend) -> Result<Superblock, OpenError> {
        let len = StorageBackend::len(backend)?;
        if len < 8 {
            return Err(OpenError::Corrupt("file too short for a header".into()));
        }
        let mut magic = [0u8; 8];
        backend.read(0, &mut magic)?;
        if magic[..4] == REDB_MAGIC {
            return Err(OpenError::Legacy);
        }
        if magic != MAGIC {
            return Err(OpenError::Corrupt("not a vdblite file".into()));
        }
        if len < HEADER_SIZE {
            return Err(OpenError::Corrupt("file too short for its header".into()));
        }
        let mut best: Option<Superblock> = None;
        let mut slot = vec![0u8; SLOT_SIZE as usize];
        for i in 0..2 {
            backend.read(i * SLOT_SIZE, &mut slot)?;
            if let Some(sb) = Superblock::decode(&slot)
                && best.as_ref().is_none_or(|b| sb.seq > b.seq)
            {
                best = Some(sb);
            }
        }
        best.ok_or_else(|| OpenError::Corrupt("no valid superblock".into()))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn is_writable(&self) -> bool {
        self.writable
    }

    /// In a reading process, picks up superblock changes the writer made.
    fn refresh(&self) -> io::Result<()> {
        if self.writable {
            return Ok(());
        }
        match Self::read_superblock(&self.file) {
            Ok(sb) => {
                let mut cur = self.sb.write();
                if sb.seq > cur.seq {
                    *cur = sb;
                }
                Ok(())
            }
            Err(OpenError::Io(e)) => Err(e),
            // The writer may be mid-way through a slot; the other one is
            // still valid, so a failure here is transient. Keep the old one.
            Err(_) => Ok(()),
        }
    }

    // ------------------------------------------------------------ redb's file

    fn logical_len(&self) -> io::Result<u64> {
        self.refresh()?;
        Ok(self.sb.read().logical_len)
    }

    fn read_logical(&self, mut off: u64, mut out: &mut [u8]) -> io::Result<()> {
        let mut refreshed = false;
        while !out.is_empty() {
            let hit = self.sb.read().translate(off);
            let Some((phys, avail)) = hit else {
                // A reading process may not know an extent the writer added
                // yet: reload the superblock once (retrying a moment if the
                // writer is between slots), then give up.
                if !self.writable && !refreshed {
                    for _ in 0..50 {
                        self.refresh()?;
                        if self.sb.read().translate(off).is_some() {
                            break;
                        }
                        std::thread::sleep(std::time::Duration::from_millis(2));
                    }
                    refreshed = true;
                    continue;
                }
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("logical offset {off} is past the end of the file"),
                ));
            };
            let n = (avail as usize).min(out.len());
            self.file.read(phys, &mut out[..n])?;
            off += n as u64;
            out = &mut out[n..];
        }
        Ok(())
    }

    fn write_logical(&self, mut off: u64, mut data: &[u8]) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only"));
        }
        while !data.is_empty() {
            let (phys, avail) = self.sb.read().translate(off).ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("write at logical offset {off} past the end"),
                )
            })?;
            let n = (avail as usize).min(data.len());
            self.file.write(phys, &data[..n])?;
            off += n as u64;
            data = &data[n..];
        }
        Ok(())
    }

    /// Sets redb's logical length, adding extents as needed. The new
    /// superblock is synced before this returns, so nothing redb writes in
    /// the new space can be referenced while the space is unknown on disk.
    fn set_logical_len(&self, len: u64) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only"));
        }
        let mut sb = self.sb.write();
        let old_len = sb.logical_len;
        let mut cap = sb.capacity();
        while cap < len {
            let want = (len - cap).max(cap.clamp(MIN_EXTENT, MAX_EXTENT));
            let size = round_up(want, PAGE);
            if sb.extents.len() >= MAX_EXTENTS {
                return Err(io::Error::other("the file has reached its extent limit"));
            }
            let phys = self.allocate_end(size)?;
            sb.extents.push(Extent { phys, len: size });
            cap += size;
        }
        // Growing back over bytes an earlier, longer file held: zero them.
        let mut high = self.high_water.lock();
        if len > old_len && old_len < *high {
            let stop = len.min(*high);
            let zeros = vec![0u8; 1 << 20];
            let mut at = old_len;
            while at < stop {
                let n = ((stop - at) as usize).min(zeros.len());
                drop(sb);
                self.write_logical(at, &zeros[..n])?;
                sb = self.sb.write();
                at += n as u64;
            }
        }
        *high = (*high).max(len);
        drop(high);
        sb.logical_len = len;
        sb.seq += 1;
        let slot = (sb.seq % 2) * SLOT_SIZE;
        self.file.write(slot, &sb.encode())?;
        self.file.sync_data()
    }

    /// redb's header (its first 320 bytes, where every commit records its
    /// transaction), so a reading process can tell cheaply whether anything
    /// was committed since it last looked.
    pub fn redb_header(&self) -> io::Result<Vec<u8>> {
        let mut out = vec![0u8; 320.min(self.sb.read().logical_len as usize)];
        self.read_logical(0, &mut out)?;
        Ok(out)
    }

    // ------------------------------------------------------------ regions

    /// Reserves `len` bytes (rounded up to a page) at the physical end of
    /// the file and returns their offset.
    pub fn allocate_end(&self, len: u64) -> io::Result<u64> {
        let mut end = self.end.lock();
        let at = *end;
        let new_end = at + round_up(len, PAGE);
        self.file.set_len(new_end)?;
        *end = new_end;
        Ok(at)
    }

    /// Writes bytes at a physical offset (a region being filled).
    pub fn write_at(&self, phys: u64, data: &[u8]) -> io::Result<()> {
        if !self.writable {
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "read-only"));
        }
        self.file.write(phys, data)
    }

    /// Reads bytes at a physical offset.
    pub fn read_at(&self, phys: u64, out: &mut [u8]) -> io::Result<()> {
        self.file.read(phys, out)
    }

    /// Flushes written regions to disk.
    pub fn sync(&self) -> io::Result<()> {
        self.file.sync_data()
    }

    /// The physical end of the file (writers: where the next allocation goes).
    pub fn physical_len(&self) -> io::Result<u64> {
        Ok(if self.writable {
            *self.end.lock()
        } else {
            StorageBackend::len(&self.file)?
        })
    }

    /// Maps `len` bytes at physical `offset`, leasing `epoch` for as long as
    /// the returned region (or any [`vdblite_index::Buf`] made from it) lives.
    pub fn map(self: &Arc<Self>, offset: u64, len: u64, epoch: u64) -> io::Result<Region> {
        let end = offset
            .checked_add(len)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "region overflows"))?;
        let lease = self.lease(epoch)?;
        let mapping = self.mapping_covering(end)?;
        Ok(Region {
            mapping,
            start: offset as usize,
            len: len as usize,
            _lease: lease,
        })
    }

    fn mapping_covering(&self, end: u64) -> io::Result<Arc<Mapping>> {
        let mut cur = self.mapping.lock();
        if let Some(m) = &*cur
            && m.map.len() as u64 >= end
        {
            return Ok(m.clone());
        }
        let file_len = self.map_file.metadata()?.len();
        if file_len < end {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("region ends at {end}, past the file's {file_len} bytes"),
            ));
        }
        // SAFETY: the mapping is read-only, and the bytes vdblite reads
        // through it are regions, which nothing writes while a lease can
        // see them (see the module docs). Other bytes of the file change
        // under the mapping, but are never read through it.
        let map = unsafe {
            MmapOptions::new()
                .len(file_len as usize)
                .map(&self.map_file)?
        };
        let m = Arc::new(Mapping { map });
        *cur = Some(m.clone());
        Ok(m)
    }

    // ------------------------------------------------------------ epochs

    /// Leases `epoch` for this process until the lease drops.
    pub fn lease(self: &Arc<Self>, epoch: u64) -> io::Result<Lease> {
        let mut ep = self.epochs.lock();
        *ep.live.entry(epoch).or_insert(0) += 1;
        self.rehold(&mut ep)?;
        Ok(Lease {
            container: self.clone(),
            epoch,
        })
    }

    fn release(&self, epoch: u64) {
        let mut ep = self.epochs.lock();
        if let Some(n) = ep.live.get_mut(&epoch) {
            *n -= 1;
            if *n == 0 {
                ep.live.remove(&epoch);
            }
        }
        let _ = self.rehold(&mut ep);
    }

    /// Moves this process's epoch lock to its oldest leased epoch: the new
    /// lock is taken before the old one goes, so there is never a gap.
    fn rehold(&self, ep: &mut Epochs) -> io::Result<()> {
        let want = ep.live.keys().next().copied();
        if want == ep.held {
            return Ok(());
        }
        if let Some(w) = want {
            let (s, e) = byte(EPOCH_LOCK_BASE + w);
            self.file.lock_shared_range(s, e).map_err(backend_io)?;
        }
        if let Some(h) = ep.held {
            let (s, e) = byte(EPOCH_LOCK_BASE + h);
            self.file.unlock_range(s, e).map_err(backend_io)?;
        }
        ep.held = want;
        Ok(())
    }

    /// Whether space freed at epoch `freed_at` can be reused: no lease in
    /// this process, and no lock in any other, on an older epoch.
    pub fn reusable(&self, freed_at: u64) -> io::Result<bool> {
        let ep = self.epochs.lock();
        if ep
            .live
            .keys()
            .next()
            .is_some_and(|&oldest| oldest < freed_at)
        {
            return Ok(false);
        }
        if freed_at == 0 {
            return Ok(true);
        }
        // This process's own lock (if any) is at or above `freed_at`, so the
        // queried range doesn't overlap it, as the backend requires.
        let busy = self
            .file
            .query_lock_range(
                Bound::Included(EPOCH_LOCK_BASE),
                Bound::Included(EPOCH_LOCK_BASE + freed_at - 1),
            )
            .map_err(backend_io)?;
        Ok(!busy)
    }

    /// The redb backend for this container.
    pub fn redb_backend(self: &Arc<Self>) -> RedbBackend {
        RedbBackend(self.clone())
    }
}

/// A held epoch; see the module docs.
pub struct Lease {
    container: Arc<Container>,
    epoch: u64,
}

impl Lease {
    pub fn epoch(&self) -> u64 {
        self.epoch
    }
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.container.release(self.epoch);
    }
}

/// A mapped region: its bytes stay readable, unchanged, while it lives.
pub struct Region {
    mapping: Arc<Mapping>,
    start: usize,
    len: usize,
    _lease: Lease,
}

impl Backing for Region {
    fn bytes(&self) -> &[u8] {
        &self.mapping.map[self.start..self.start + self.len]
    }
}

/// redb's view of a container: its logical file, plus the file's locks.
#[derive(Debug)]
pub struct RedbBackend(Arc<Container>);

impl StorageBackend for RedbBackend {
    fn len(&self) -> Result<u64, io::Error> {
        self.0.logical_len()
    }

    fn read(&self, offset: u64, out: &mut [u8]) -> Result<(), io::Error> {
        self.0.read_logical(offset, out)
    }

    fn set_len(&self, len: u64) -> Result<(), io::Error> {
        self.0.set_logical_len(len)
    }

    fn sync_data(&self) -> Result<(), io::Error> {
        self.0.file.sync_data()
    }

    fn write(&self, offset: u64, data: &[u8]) -> Result<(), io::Error> {
        self.0.write_logical(offset, data)
    }

    fn close(&self) -> Result<(), io::Error> {
        // The container outlives redb's handle (mapped regions), and closes
        // the file when it goes; redb's locks are released here.
        Ok(())
    }

    fn try_lock_range(
        &self,
        start: Bound<u64>,
        end: Bound<u64>,
    ) -> Result<bool, redb::BackendError> {
        self.0.file.try_lock_range(start, end)
    }

    fn try_lock_shared_range(
        &self,
        start: Bound<u64>,
        end: Bound<u64>,
    ) -> Result<bool, redb::BackendError> {
        self.0.file.try_lock_shared_range(start, end)
    }

    fn lock_range(&self, start: Bound<u64>, end: Bound<u64>) -> Result<(), redb::BackendError> {
        self.0.file.lock_range(start, end)
    }

    fn lock_shared_range(
        &self,
        start: Bound<u64>,
        end: Bound<u64>,
    ) -> Result<(), redb::BackendError> {
        self.0.file.lock_shared_range(start, end)
    }

    fn unlock_range(&self, start: Bound<u64>, end: Bound<u64>) -> Result<(), redb::BackendError> {
        self.0.file.unlock_range(start, end)
    }

    fn query_lock_range(
        &self,
        start: Bound<u64>,
        end: Bound<u64>,
    ) -> Result<bool, redb::BackendError> {
        self.0.file.query_lock_range(start, end)
    }
}

impl Drop for Container {
    fn drop(&mut self) {
        let _ = self.file.close();
    }
}

/// Opens the redb store inside a `.vdb` file for writing, bypassing
/// vdblite: for tools and tests that inspect or alter tables directly.
pub fn open_store(path: &Path) -> Result<redb::Database, crate::StorageError> {
    let c = Container::open(path, true).map_err(|e| match e {
        OpenError::Io(e) => crate::StorageError::Io(e),
        OpenError::Legacy => crate::StorageError::Corrupt("a file from before regions".into()),
        OpenError::Corrupt(m) => crate::StorageError::Corrupt(m),
    })?;
    let mut b = redb::Builder::new();
    b.set_concurrency_mode(redb::ConcurrencyMode::SingleWriter);
    Ok(b.create_with_backend(c.redb_backend())?)
}

/// Writes a compact copy of `src` to `dst`: redb's logical file in one
/// extent, then the given regions packed one after another. Returns each
/// region's new physical offset, in the order given. The caller rewrites
/// the records that point at regions, then renames `dst` over the original.
pub fn rewrite(src: &Container, dst: &Path, regions: &[(u64, u64)]) -> io::Result<Vec<u64>> {
    let logical_len = src.sb.read().logical_len;
    let extent_len = round_up(logical_len.max(PAGE), PAGE);
    let mut placed = Vec::with_capacity(regions.len());
    let mut end = HEADER_SIZE + extent_len;
    for &(_, len) in regions {
        placed.push(end);
        end += round_up(len, PAGE);
    }
    let out = OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(dst)?;
    out.set_len(end)?;
    let sb = Superblock {
        seq: 1,
        logical_len,
        extents: vec![Extent {
            phys: HEADER_SIZE,
            len: extent_len,
        }],
    };
    write_all_at(&out, 0, &sb.encode())?;
    let mut buf = vec![0u8; 8 << 20];
    let mut at = 0u64;
    while at < logical_len {
        let n = ((logical_len - at) as usize).min(buf.len());
        src.read_logical(at, &mut buf[..n])?;
        write_all_at(&out, HEADER_SIZE + at, &buf[..n])?;
        at += n as u64;
    }
    for (&(from, len), &to) in regions.iter().zip(&placed) {
        let mut done = 0u64;
        while done < len {
            let n = ((len - done) as usize).min(buf.len());
            src.read_at(from + done, &mut buf[..n])?;
            write_all_at(&out, to + done, &buf[..n])?;
            done += n as u64;
        }
    }
    out.sync_all()?;
    Ok(placed)
}

/// Renames `tmp` over `path` and makes the rename durable.
pub fn replace(tmp: &Path, path: &Path) -> io::Result<()> {
    std::fs::rename(tmp, path)?;
    if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty())
        && let Ok(d) = File::open(dir)
    {
        // Best effort where directories can't be synced.
        let _ = d.sync_all();
    }
    Ok(())
}

/// Whether the file at `path` is a bare redb file from before regions.
pub fn is_legacy(path: &Path) -> io::Result<bool> {
    let mut magic = [0u8; 4];
    let mut f = File::open(path)?;
    use std::io::Read;
    Ok(f.read_exact(&mut magic).is_ok() && magic == REDB_MAGIC)
}

/// Rewrites a bare redb file as a container holding the same redb file,
/// atomically (a new file renamed over the old one). The caller must have
/// the file to itself.
pub fn upgrade_legacy(path: &Path) -> io::Result<()> {
    let src = File::open(path)?;
    let len = src.metadata()?.len();
    let tmp = path.with_extension("vdb-upgrade");
    let _ = std::fs::remove_file(&tmp);
    {
        let dst = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&tmp)?;
        let extent_len = round_up(len.max(PAGE), PAGE);
        let sb = Superblock {
            seq: 1,
            logical_len: len,
            extents: vec![Extent {
                phys: HEADER_SIZE,
                len: extent_len,
            }],
        };
        dst.set_len(HEADER_SIZE + extent_len)?;
        write_all_at(&dst, 0, &sb.encode())?;
        let mut buf = vec![0u8; 8 << 20];
        let mut at = 0u64;
        while at < len {
            let n = ((len - at) as usize).min(buf.len());
            read_exact_at(&src, at, &mut buf[..n])?;
            write_all_at(&dst, HEADER_SIZE + at, &buf[..n])?;
            at += n as u64;
        }
        dst.sync_all()?;
    }
    replace(&tmp, path)
}

#[cfg(unix)]
fn write_all_at(f: &File, off: u64, data: &[u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.write_all_at(data, off)
}

#[cfg(unix)]
fn read_exact_at(f: &File, off: u64, out: &mut [u8]) -> io::Result<()> {
    use std::os::unix::fs::FileExt;
    f.read_exact_at(out, off)
}

#[cfg(windows)]
fn write_all_at(f: &File, mut off: u64, mut data: &[u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !data.is_empty() {
        let n = f.seek_write(data, off)?;
        off += n as u64;
        data = &data[n..];
    }
    Ok(())
}

#[cfg(windows)]
fn read_exact_at(f: &File, mut off: u64, mut out: &mut [u8]) -> io::Result<()> {
    use std::os::windows::fs::FileExt;
    while !out.is_empty() {
        let n = f.seek_read(out, off)?;
        if n == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        off += n as u64;
        out = &mut out[n..];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn superblocks_round_trip_and_reject_damage() {
        let sb = Superblock {
            seq: 7,
            logical_len: 5000,
            extents: vec![
                Extent {
                    phys: HEADER_SIZE,
                    len: 4096,
                },
                Extent {
                    phys: HEADER_SIZE + 8192,
                    len: 4096,
                },
            ],
        };
        let mut slot = sb.encode();
        slot.resize(SLOT_SIZE as usize, 0);
        assert_eq!(Superblock::decode(&slot), Some(sb.clone()));
        assert_eq!(sb.translate(4095), Some((HEADER_SIZE + 4095, 1)));
        assert_eq!(sb.translate(4096), Some((HEADER_SIZE + 8192, 4096)));
        assert_eq!(sb.translate(8192), None);
        for i in [0, 9, 17, 30, 40, 50] {
            let mut bad = slot.clone();
            bad[i] ^= 0x10;
            assert_eq!(Superblock::decode(&bad), None, "byte {i}");
        }
    }

    #[test]
    fn redb_lives_inside_a_container() {
        use redb::{ReadableDatabase, ReadableTable, ReadableTableMetadata, TableDefinition};
        const T: TableDefinition<u64, &[u8]> = TableDefinition::new("t");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("c.vdb");
        let c = Container::open(&path, true).unwrap();
        let region_at;
        {
            let db = redb::Database::builder()
                .create_with_backend(c.redb_backend())
                .unwrap();
            let tx = db.begin_write().unwrap();
            {
                let mut t = tx.open_table(T).unwrap();
                for i in 0..2000u64 {
                    t.insert(i, vec![i as u8; 900].as_slice()).unwrap();
                }
            }
            tx.commit().unwrap();
            // A region between redb's extents.
            region_at = c.allocate_end(10_000).unwrap();
            c.write_at(region_at, &[7u8; 10_000]).unwrap();
            c.sync().unwrap();
            let tx = db.begin_write().unwrap();
            {
                let mut t = tx.open_table(T).unwrap();
                for i in 2000..4000u64 {
                    t.insert(i, vec![i as u8; 900].as_slice()).unwrap();
                }
            }
            tx.commit().unwrap();
        }
        drop(c);

        let c = Container::open(&path, false).unwrap();
        let db = redb::Database::builder()
            .open_read_only_with_backend(c.redb_backend())
            .unwrap();
        let rx = db.begin_read().unwrap();
        let t = rx.open_table(T).unwrap();
        assert_eq!(t.len().unwrap(), 4000);
        assert_eq!(
            t.get(3999).unwrap().unwrap().value(),
            &[3999u64 as u8; 900][..]
        );
        let r = c.map(region_at, 10_000, 1).unwrap();
        assert!(r.bytes().iter().all(|&b| b == 7));
    }

    #[test]
    fn legacy_files_upgrade_in_place() {
        use redb::{ReadableDatabase, ReadableTable, TableDefinition};
        const T: TableDefinition<u64, u64> = TableDefinition::new("t");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("old.vdb");
        {
            let db = redb::Database::create(&path).unwrap();
            let tx = db.begin_write().unwrap();
            tx.open_table(T).unwrap().insert(1, 42).unwrap();
            tx.commit().unwrap();
        }
        assert!(is_legacy(&path).unwrap());
        assert!(matches!(
            Container::open(&path, false),
            Err(OpenError::Legacy)
        ));
        upgrade_legacy(&path).unwrap();
        assert!(!is_legacy(&path).unwrap());
        let c = Container::open(&path, true).unwrap();
        let db = redb::Database::builder()
            .create_with_backend(c.redb_backend())
            .unwrap();
        let rx = db.begin_read().unwrap();
        assert_eq!(
            rx.open_table(T).unwrap().get(1).unwrap().unwrap().value(),
            42
        );
    }

    #[test]
    fn freed_space_waits_for_older_leases() {
        let dir = tempfile::tempdir().unwrap();
        let c = Container::open(&dir.path().join("e.vdb"), true).unwrap();
        assert!(c.reusable(5).unwrap(), "nothing leased");
        let old = c.lease(3).unwrap();
        assert!(
            !c.reusable(5).unwrap(),
            "a lease on epoch 3 can see space freed at 5"
        );
        assert!(
            c.reusable(3).unwrap(),
            "space freed at 3 was never visible at 3"
        );
        let newer = c.lease(6).unwrap();
        drop(old);
        assert!(c.reusable(5).unwrap());
        assert!(!c.reusable(7).unwrap());
        drop(newer);
        assert!(c.reusable(100).unwrap());
    }
}
