//! Sealing and compaction.
//!
//! Both run in three steps, so the slow part never blocks writes or
//! searches:
//!
//! 1. Take what they need from the in-memory index: a copy of the write
//!    buffer, or a compaction plan and handles to its source segments.
//! 2. Build the new graph with no lock held. Reserve space for its region
//!    (a short transaction), and write and sync the region, still unlocked.
//! 3. Under the writer lock, commit the segment's record in one transaction
//!    (with a change-log entry for reading processes), then swap the
//!    segment, mapped from its region, into the index.
//!
//! Writes that land during step 2 are reconciled in step 3: a chunk deleted
//! meanwhile is tombstoned in the new segment, on disk and in memory.
//!
//! A background thread per database ([`Worker`]) runs them after commits;
//! [`crate::Db::optimize`] runs them on the caller's thread. A per-collection
//! lock keeps the two from working on the same collection at once.

use std::sync::atomic::Ordering;
use std::sync::{Arc, OnceLock, Weak};

use parking_lot::{Condvar, Mutex};
use vdblite_core::changes;
use vdblite_core::records::{Change, SegmentRecord, decode, encode};
use vdblite_core::segments::{self, io};
use vdblite_core::tables::{
    CHANGES, CHUNKS, DEAD, DOCS, FREE_REGIONS, PENDING, RESERVED_REGIONS, SEGMENTS, SETTINGS,
    VECTORS,
};
use vdblite_index::segment::SealedSegment;

use crate::collection::CollInner;
use crate::db::{DbInner, DbRef, EPOCH_KEY};
use crate::error::Result;

/// Background builds that run alongside searches use half the cores, in
/// their own pool, so searches (which fan out on rayon's global pool) never
/// queue behind a graph build, and at low priority, so the OS runs searches
/// first when cores are short. With nothing searching, or a caller waiting
/// for maintenance, they use every core instead (see [`Access::build`]).
fn build_pool() -> &'static rayon::ThreadPool {
    static POOL: OnceLock<rayon::ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        rayon::ThreadPoolBuilder::new()
            .num_threads((cores / 2).max(1))
            .thread_name(|i| format!("vdblite-build-{i}"))
            .start_handler(|_| lower_priority())
            .build()
            .expect("spawning the index build threads")
    })
}

/// Lowers the calling thread's scheduling priority (best effort).
fn lower_priority() {
    #[cfg(target_vendor = "apple")]
    {
        // SAFETY: sets the QoS class of the calling thread only; no pointers.
        let _ =
            unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_UTILITY, 0) };
    }
    #[cfg(target_os = "linux")]
    {
        // SAFETY: raises the calling thread's nice value (per-thread on
        // Linux); no pointers.
        let _ = unsafe { libc::nice(10) };
    }
}

/// Whether a collection has a seal, a compaction or an interrupted bulk
/// load to finish.
pub(crate) fn is_due(c: &CollInner) -> bool {
    if c.bulk_leftover.load(Ordering::Acquire) && c.bulk.lock().is_none() {
        return true;
    }
    let idx = c.index.read();
    idx.should_seal() || idx.needs_compaction(false)
}

/// How maintenance reaches the database.
enum Access<'a> {
    /// The caller holds the database open, and builds on its own thread.
    Caller(&'a DbInner),
    /// The background thread, which holds the database only while it
    /// touches the file, and builds in the background pool.
    Background(&'a Worker, &'a Weak<DbInner>),
}

impl Access<'_> {
    /// Runs `f` with the database, or returns `None` if it was closed.
    fn with<R>(&self, f: impl FnOnce(&DbInner) -> Result<R>) -> Result<Option<R>> {
        match self {
            Access::Caller(db) => f(db).map(Some),
            Access::Background(worker, weak) => match worker.enter(weak) {
                Some(held) => f(&held).map(Some),
                None => Ok(None),
            },
        }
    }

    fn build<R: Send>(&self, f: impl FnOnce() -> R + Send) -> R {
        match self {
            Access::Caller(_) => f(),
            // Nothing to share the cores with, or someone waiting on this:
            // every core, at normal priority.
            Access::Background(worker, _)
                if worker.waiting.load(Ordering::Acquire) > 0
                    || vdblite_index::index::searches_in_flight() == 0 =>
            {
                f()
            }
            Access::Background(..) => build_pool().install(f),
        }
    }
}

/// Whether `coll` is still the collection registered under its name (it
/// was not dropped). Call with the writer lock held.
fn registered(db: &DbInner, coll: &Arc<CollInner>) -> bool {
    db.collections
        .read()
        .get(&coll.name)
        .is_some_and(|c| Arc::ptr_eq(c, coll))
}

/// Seals and merges everything in a collection, on the caller's thread.
pub(crate) fn optimize(db: &DbInner, coll: &Arc<CollInner>) -> Result<()> {
    let _m = coll.maint.lock();
    let access = Access::Caller(db);
    seal(&access, coll)?;
    compact(&access, coll, true)
}

/// Seals the write buffer on the caller's thread if it holds more than a
/// full segment: back-pressure for writes that outpace background sealing.
/// Waits for background work on the collection first.
pub(crate) fn seal_backlog(db: &DbInner, coll: &Arc<CollInner>) -> Result<()> {
    let _m = coll.maint.lock();
    let over = {
        let idx = coll.index.read();
        idx.mutable().len() >= idx.config().max_segment
    };
    if over {
        seal(&Access::Caller(db), coll)?;
    }
    Ok(())
}

/// The background pass over one collection.
fn run_due(access: &Access<'_>, coll: &Arc<CollInner>) -> Result<()> {
    let _m = coll.maint.lock();
    if coll.bulk_leftover.load(Ordering::Acquire) && coll.bulk.lock().is_none() {
        access.with(|db| crate::bulk::finish_leftover(db, coll))?;
    }
    if coll.index.read().should_seal() {
        seal(access, coll)?;
    }
    if coll.index.read().needs_compaction(false) {
        compact(access, coll, false)?;
    }
    Ok(())
}

/// Reserves region space for `seg` (a short transaction under the writer
/// lock) and writes the region outside it. Returns the record to commit,
/// or `None` if the database closed meanwhile.
fn write_region(access: &Access<'_>, seg: &SealedSegment) -> Result<Option<SegmentRecord>> {
    let len = segments::region_len(seg);
    let Some(offset) = access.with(|db| {
        let _w = db.writer.lock();
        let tx = db.store.writer()?.begin_write()?;
        let offset = {
            let mut free = tx.open_table(FREE_REGIONS)?;
            let offset = segments::allocate(&mut free, &db.container, len)?;
            tx.open_table(RESERVED_REGIONS)?.insert(offset, len)?;
            offset
        };
        tx.commit()?;
        Ok(offset)
    })?
    else {
        return Ok(None);
    };
    access.with(|db| {
        let rec = segments::write_segment(&db.container, seg, offset)?;
        db.container.sync().map_err(io)?;
        Ok(rec)
    })
}

/// Seals the write buffer into a new segment. Call with `coll.maint` held.
fn seal(access: &Access<'_>, coll: &Arc<CollInner>) -> Result<()> {
    let id = coll.index.write().allocate_segment_id();
    let (cfg, snapshot) = {
        let idx = coll.index.read();
        (*idx.config(), idx.snapshot_mutable())
    };
    let Some(snapshot) = snapshot else {
        return Ok(());
    };
    let mut pending = access.build(move || snapshot.build(&cfg, id));
    // Its rows' document keys, so searches name results without lookups.
    let labelled = access.with(|db| {
        let rx = db.store.begin_read()?;
        crate::collection::label_with_keys(
            &rx.open_table(CHUNKS)?,
            &rx.open_table(DOCS)?,
            coll.id,
            &mut pending.segment,
        )
    })?;
    if labelled.is_none() {
        return Ok(());
    }
    let Some(rec) = write_region(access, &pending.segment)? else {
        return Ok(());
    };
    access.with(|db| {
        let _w = db.writer.lock();
        if !registered(db, coll) {
            return Ok(());
        }
        let tx = db.store.writer()?.begin_write()?;
        {
            tx.open_table(RESERVED_REGIONS)?.remove(rec.region.0)?;
            tx.open_table(SEGMENTS)?
                .insert((coll.id, id), encode(&rec).as_slice())?;
            let mut pending_rows = tx.open_table(PENDING)?;
            let mut vectors = tx.open_table(VECTORS)?;
            let mut dead = tx.open_table(DEAD)?;
            for &chunk in pending.segment.ids() {
                if pending_rows.remove((coll.id, chunk))?.is_some() {
                    // Its full vector now lives in the segment's region.
                    vectors.remove((coll.id, chunk))?;
                } else {
                    // Deleted while the segment was building.
                    dead.insert((coll.id, chunk), id)?;
                }
            }
            let mut log = tx.open_table(CHANGES)?;
            changes::append(&mut log, coll.id, &Change::Seal { segment: id })?;
        }
        tx.commit()?;
        // Serve it from the region: the heap copy goes.
        pending.segment = segments::map_segment(&db.container, &rec, db.epoch())?;
        coll.index.write().install_sealed(pending);
        Ok(())
    })?;
    Ok(())
}

/// Rebuilds segments with too many tombstones and, with `merge`, merges
/// small neighbouring segments. Call with `coll.maint` held.
fn compact(access: &Access<'_>, coll: &Arc<CollInner>, merge: bool) -> Result<()> {
    let (cfg, jobs) = {
        let idx = coll.index.read();
        (*idx.config(), idx.plan_compaction(merge))
    };
    for job in jobs {
        // Handles to the source segments: their vectors are read in place.
        let input = coll.index.read().compaction_input(&job);
        let built = access.build(|| input.build(&cfg));
        drop(input);
        let rec = match &built {
            Some(seg) => match write_region(access, seg)? {
                Some(rec) => Some(rec),
                None => return Ok(()),
            },
            None => None,
        };

        let applied = access.with(|db| {
            let _w = db.writer.lock();
            if !registered(db, coll) {
                return Ok(false);
            }
            let replacement = built.map(|seg| (coll.index.write().allocate_segment_id(), seg));
            let tx = db.store.writer()?.begin_write()?;
            let epoch = db.epoch() + 1;
            {
                let mut headers = tx.open_table(SEGMENTS)?;
                let mut free = tx.open_table(FREE_REGIONS)?;
                if let (Some((id, _)), Some(rec)) = (&replacement, &rec) {
                    tx.open_table(RESERVED_REGIONS)?.remove(rec.region.0)?;
                    headers.insert((coll.id, *id), encode(rec).as_slice())?;
                }
                // The old regions are free once nothing can still read them.
                for &old in &job.remove {
                    if let Some(v) = headers.remove((coll.id, old))? {
                        let r: SegmentRecord = decode(v.value(), "segment record")?;
                        segments::release(&mut free, r.region, epoch)?;
                    }
                }
                tx.open_table(SETTINGS)?
                    .insert(EPOCH_KEY, encode(&epoch).as_slice())?;
                // Tombstones in the removed segments move to the replacement
                // if it holds that chunk (deleted after planning), or go.
                let mut dead = tx.open_table(DEAD)?;
                let idx = coll.index.read();
                for e in idx.sealed().iter().filter(|e| job.remove.contains(&e.id)) {
                    for local in e.tombstones.iter_ones() {
                        let chunk = e.segment.ids()[local as usize];
                        match &replacement {
                            Some((id, seg)) if seg.local_of(chunk).is_some() => {
                                dead.insert((coll.id, chunk), *id)?;
                            }
                            _ => {
                                dead.remove((coll.id, chunk))?;
                            }
                        }
                    }
                }
                drop(idx);
                let mut log = tx.open_table(CHANGES)?;
                let change = Change::Compact {
                    removed: job.remove.clone(),
                    added: replacement.as_ref().map(|r| r.0),
                };
                changes::append(&mut log, coll.id, &change)?;
            }
            tx.commit()?;
            db.epoch.store(epoch, Ordering::Release);
            let replacement = match (replacement, &rec) {
                (Some((id, _)), Some(rec)) => {
                    Some((id, segments::map_segment(&db.container, rec, epoch)?))
                }
                _ => None,
            };
            coll.index.write().apply_compaction(&job, replacement);
            Ok(true)
        })?;
        if applied != Some(true) {
            return Ok(());
        }
    }
    Ok(())
}

/// The background maintenance thread of one database, and the handshake
/// that keeps it from holding the database open after its last handle is
/// dropped.
///
/// The thread keeps only a weak reference. It upgrades it for short spells
/// ([`Worker::enter`]) and never once the database is closing; the last
/// handle waits out a spell in progress ([`Worker::release`]).
#[derive(Default)]
pub(crate) struct Worker {
    state: Mutex<WorkerState>,
    cv: Condvar,
    /// Callers blocked in [`Worker::wait_idle`].
    waiting: std::sync::atomic::AtomicUsize,
}

#[derive(Default)]
struct WorkerState {
    /// Work was requested since the thread last looked.
    queued: bool,
    /// The thread is working through a pass.
    running: bool,
    /// The thread exists.
    started: bool,
    /// The last handle is gone or going: take no new reference.
    closed: bool,
    /// References to the database the thread holds right now.
    holding: usize,
}

/// A spell of access to the database from the background thread.
struct Held<'w> {
    db: Option<Arc<DbInner>>,
    worker: &'w Worker,
}

impl std::ops::Deref for Held<'_> {
    type Target = DbInner;

    fn deref(&self) -> &DbInner {
        self.db.as_ref().expect("held until dropped")
    }
}

impl Drop for Held<'_> {
    fn drop(&mut self) {
        let mut st = self.worker.state.lock();
        // Dropped under the lock, so a handle being released never sees this
        // reference half gone. It is never the last: the last handle waits
        // for `holding` to reach zero before letting go.
        drop(self.db.take());
        st.holding -= 1;
        self.worker.cv.notify_all();
    }
}

impl Worker {
    /// Asks the thread for a maintenance pass, starting it if needed.
    pub(crate) fn request(self: &Arc<Self>, db: &DbRef) {
        let mut st = self.state.lock();
        if st.closed {
            return;
        }
        st.queued = true;
        if !st.started {
            st.started = true;
            let worker = self.clone();
            let weak = db.downgrade();
            let spawned = std::thread::Builder::new()
                .name("vdblite-maintenance".into())
                .spawn(move || worker.run(weak));
            if spawned.is_err() {
                // No thread: the next request tries again; `optimize` and
                // write back-pressure still work on callers' threads.
                st.started = false;
                st.queued = false;
            }
        }
        self.cv.notify_all();
    }

    /// Blocks until no pass is queued or running.
    pub(crate) fn wait_idle(&self) {
        self.waiting.fetch_add(1, Ordering::AcqRel);
        let mut st = self.state.lock();
        while st.started && (st.queued || st.running) {
            self.cv.wait(&mut st);
        }
        drop(st);
        self.waiting.fetch_sub(1, Ordering::AcqRel);
    }

    fn run(self: Arc<Self>, weak: Weak<DbInner>) {
        loop {
            {
                let mut st = self.state.lock();
                st.running = false;
                self.cv.notify_all();
                while !st.queued && !st.closed {
                    self.cv.wait(&mut st);
                }
                if st.closed {
                    st.started = false;
                    self.cv.notify_all();
                    return;
                }
                st.queued = false;
                st.running = true;
            }
            let colls: Vec<Arc<CollInner>> = match self.enter(&weak) {
                Some(held) => held.collections.read().values().cloned().collect(),
                None => continue,
            };
            let access = Access::Background(&self, &weak);
            for c in colls {
                // A panic inside the storage engine (a corrupted file) must
                // not kill the thread with `running` set.
                let result = crate::error::guard(|| run_due(&access, &c));
                let mut slot = c.maint_error.lock();
                match result {
                    Ok(()) => *slot = None,
                    Err(e) => *slot = Some(e),
                }
            }
        }
    }

    /// Takes a reference to the database for one spell, unless it is
    /// closing.
    fn enter(&self, weak: &Weak<DbInner>) -> Option<Held<'_>> {
        let mut st = self.state.lock();
        if st.closed {
            return None;
        }
        let db = weak.upgrade()?;
        st.holding += 1;
        drop(st);
        Some(Held {
            db: Some(db),
            worker: self,
        })
    }

    /// Drops a handle's reference. If it is the last handle, waits for the
    /// thread to finish its spell and stops it from taking another, so the
    /// database is dropped (and the file closed) right here.
    pub(crate) fn release(&self, inner: Arc<DbInner>) {
        let mut st = self.state.lock();
        loop {
            let others = Arc::strong_count(&inner) - 1 - st.holding;
            if others > 0 {
                // Not the last handle; dropping under the lock keeps the
                // count honest for a concurrent last one.
                drop(inner);
                return;
            }
            if st.holding == 0 {
                break;
            }
            self.cv.wait(&mut st);
        }
        st.closed = true;
        self.cv.notify_all();
        drop(st);
        drop(inner);
    }
}

impl std::fmt::Debug for Worker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Worker")
    }
}
