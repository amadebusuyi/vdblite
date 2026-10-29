//! Bulk loads (see [`BulkLoad`]): their vectors stay out of the `vectors`
//! table and the write buffer, and go to one graph built on a background
//! thread while the load goes on ([`SegmentBuilder`]). Each batch's vectors
//! are checkpointed in a region recorded with its documents ([`BulkRecord`]
//! in the `bulk` table), which is what a crashed or dropped load is
//! finished from.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::thread::JoinHandle;

use redb::{ReadableTable, WriteTransaction};
use roaring::RoaringBitmap;
use vdblite_core::changes;
use vdblite_core::records::{BulkRecord, Change, decode, encode};
use vdblite_core::segments::{self, io};
use vdblite_core::tables::{BULK, CHANGES, CHUNKS, DEAD, FREE_REGIONS, SEGMENTS};
use vdblite_index::segment::{SealedSegment, SegmentBuilder};

use crate::collection::{CollInner, Collection};
use crate::db::{Db, DbInner};
use crate::doc::Doc;
use crate::error::{Error, Result, invalid};
use crate::tx::WriteTx;

/// Chunks between durable checkpoints, by default.
pub const DEFAULT_CHECKPOINT: usize = 1_000_000;

/// What other transactions need to know about a collection's bulk load.
#[derive(Default)]
pub(crate) struct BulkState {
    /// Chunks of the load deleted (or replaced) before it finished.
    pub removed: RoaringBitmap,
}

/// Batches queued for the builder thread before `add` waits for it.
const QUEUED_BATCHES: usize = 4;

/// The thread building a bulk load's segment.
struct Builder {
    batches: SyncSender<(Vec<u32>, Vec<f32>)>,
    /// Set when the load is dropped unfinished: the thread stops early.
    cancel: Arc<AtomicBool>,
    thread: JoinHandle<Option<SealedSegment>>,
}

impl Builder {
    fn spawn(c: &CollInner) -> Result<Self> {
        let cfg = *c.index.read().config();
        let (batches, rx) = sync_channel(QUEUED_BATCHES);
        let cancel = Arc::new(AtomicBool::new(false));
        let stop = cancel.clone();
        let thread = std::thread::Builder::new()
            .name("vdblite-bulk-build".into())
            .spawn(move || {
                let b = SegmentBuilder::new(cfg.dim, cfg.metric, &cfg.hnsw, cfg.quantization);
                build(b, rx, &stop)
            })
            .map_err(|e| Error::Storage(vdblite_core::StorageError::Io(e)))?;
        Ok(Builder {
            batches,
            cancel,
            thread,
        })
    }

    /// The finished segment, or `None` if the thread failed.
    fn finish(self) -> Option<SealedSegment> {
        drop(self.batches);
        self.thread.join().ok().flatten()
    }

    fn cancel(self) {
        self.cancel.store(true, Ordering::Release);
    }
}

fn build(
    mut b: SegmentBuilder,
    rx: Receiver<(Vec<u32>, Vec<f32>)>,
    cancel: &AtomicBool,
) -> Option<SealedSegment> {
    while let Ok((mut ids, mut vectors)) = rx.recv() {
        if cancel.load(Ordering::Acquire) {
            return None;
        }
        // Whatever else is queued joins this step.
        while let Ok((i, v)) = rx.try_recv() {
            ids.extend(i);
            vectors.extend(v);
        }
        b.add(&ids, &vectors);
    }
    (!cancel.load(Ordering::Acquire)).then(|| b.finish())
}

/// A bulk load in progress: many documents into one collection, indexed
/// once.
///
/// A bulk load writes documents as ordinary transactions do, except that
/// their vectors go to one graph, built with every core while the load goes
/// on; [`BulkLoad::finish`] waits for the last of it and makes it
/// searchable. Nothing is sealed or merged along the way, so each vector is
/// inserted into a graph once, and most of the build overlaps the writes.
///
/// Durability: each [`BulkLoad::add`] is one transaction, and its documents
/// and vectors always commit together. In a file opened with
/// [`Db::open_exclusive`], batches commit without waiting for the disk, and
/// every [`checkpoint_every`](BulkLoad::checkpoint_every) chunks one
/// commits durably; a crash loses the load since the last durable commit
/// and nothing before it. (In a file other processes may share, every batch
/// commits durably.) The next writer to open the file finishes whatever a
/// crashed load committed, in the background; dropping a load without
/// finishing it does the same.
///
/// Until `finish` returns, the load's documents can be read by key but not
/// found by search.
pub struct BulkLoad {
    db: Db,
    coll: Collection,
    /// `None` once finished, or if the builder thread died.
    builder: Option<Builder>,
    /// Each chunk's document key.
    keys: HashMap<u32, String>,
    /// Chunks added since the last durable commit.
    since_durable: usize,
    checkpoint_every: usize,
    finished: bool,
}

impl Db {
    /// Starts a bulk load into `coll`; see [`BulkLoad`]. One bulk load per
    /// collection runs at a time.
    pub fn bulk_load(&self, coll: &Collection) -> Result<BulkLoad> {
        self.inner.store.writer()?;
        let mut state = coll.inner.bulk.lock();
        if state.is_some() {
            return Err(invalid(format!(
                "collection {:?} already has a bulk load under way",
                coll.inner.name
            )));
        }
        let builder = Builder::spawn(&coll.inner)?;
        *state = Some(BulkState::default());
        Ok(BulkLoad {
            db: self.clone(),
            coll: coll.clone(),
            builder: Some(builder),
            keys: HashMap::new(),
            since_durable: 0,
            checkpoint_every: DEFAULT_CHECKPOINT,
            finished: false,
        })
    }
}

impl BulkLoad {
    /// Chunks between durable commits in an exclusively opened file
    /// (default [`DEFAULT_CHECKPOINT`]).
    pub fn checkpoint_every(mut self, chunks: usize) -> Self {
        self.checkpoint_every = chunks.max(1);
        self
    }

    /// Inserts or replaces documents with their vectors (one per chunk, as
    /// [`WriteTx::upsert_with_vectors`]), in one transaction. Returns how
    /// many were written. Batches of thousands keep the per-commit cost
    /// small.
    pub fn add(&mut self, docs: impl IntoIterator<Item = (Doc, Vec<Vec<f32>>)>) -> Result<usize> {
        let mut tx = WriteTx::begin_bulk(&self.db, &self.coll)?;
        let mut n = 0;
        for (doc, vectors) in docs {
            tx.upsert_with_vectors(&self.coll, doc, vectors)?;
            n += 1;
        }
        let durable = !self.db.inner.exclusive || self.since_durable >= self.checkpoint_every;
        if let Some(sink) = tx.commit_bulk(durable)? {
            self.since_durable = if durable {
                0
            } else {
                self.since_durable + sink.ids.len()
            };
            self.keys.extend(sink.ids.iter().copied().zip(sink.keys));
            // A builder that died is noticed at `finish`, which rebuilds
            // from the checkpoints.
            if let Some(b) = &self.builder
                && b.batches.send((sink.ids, sink.vectors)).is_err()
            {
                self.builder = None;
            }
        }
        Ok(n)
    }

    /// Makes everything added so far durable.
    pub fn checkpoint(&mut self) -> Result<()> {
        let db = &self.db.inner;
        let _w = db.writer.lock();
        // An empty durable commit makes every commit before it durable;
        // their checkpoint regions must be on disk first.
        db.container.sync().map_err(io)?;
        db.store.writer()?.begin_write()?.commit()?;
        self.since_durable = 0;
        Ok(())
    }

    /// Waits for the index over everything loaded and makes it searchable.
    /// Other writers go on until it is ready to install.
    pub fn finish(mut self) -> Result<()> {
        self.finished = true;
        let result = self.build_and_install();
        *self.coll.inner.bulk.lock() = None;
        result
    }

    fn build_and_install(&mut self) -> Result<()> {
        let db = &self.db.inner;
        let c = &self.coll.inner;
        let Some(mut seg) = self.builder.take().and_then(Builder::finish) else {
            // The builder failed: build again from the checkpoints.
            return finish_leftover(db, c);
        };
        seg.set_labels(|g| self.keys.get(&g).map(String::as_bytes));
        let _w = db.writer.lock();
        // Rows deleted or replaced during the load stay in the graph as
        // tombstones, as rows deleted while a seal builds do.
        let removed = c
            .bulk
            .lock()
            .as_ref()
            .map(|b| b.removed.clone())
            .unwrap_or_default();
        let dead: Vec<u32> = removed
            .iter()
            .filter(|&id| seg.local_of(id).is_some())
            .collect();
        install(db, c, (!seg.is_empty()).then_some(seg), &dead)
    }
}

impl Drop for BulkLoad {
    fn drop(&mut self) {
        if self.finished {
            return;
        }
        if let Some(b) = self.builder.take() {
            b.cancel();
        }
        // Keep what was loaded: checkpoint it, and let maintenance finish it.
        if self.checkpoint().is_ok() {
            self.coll.inner.bulk_leftover.store(true, Ordering::Release);
            self.db.inner.worker.request(&self.db.inner);
        }
        *self.coll.inner.bulk.lock() = None;
    }
}

/// The rows of `ids`/`vectors` that `keep` accepts.
fn live_rows(
    ids: Vec<u32>,
    vectors: Vec<f32>,
    dim: usize,
    keep: impl Fn(u32) -> bool,
) -> (Vec<u32>, Vec<f32>) {
    if ids.iter().all(|&id| keep(id)) {
        return (ids, vectors);
    }
    let mut out_ids = Vec::with_capacity(ids.len());
    let mut out = Vec::with_capacity(vectors.len());
    for (&id, v) in ids.iter().zip(vectors.chunks_exact(dim)) {
        if keep(id) {
            out_ids.push(id);
            out.extend_from_slice(v);
        }
    }
    (out_ids, out)
}

/// Writes `built` and commits it in place of the collection's bulk
/// checkpoints, with `dead` (global ids in it) as tombstones. Call with the
/// write lock held.
fn install(
    db: &DbInner,
    c: &Arc<CollInner>,
    built: Option<SealedSegment>,
    dead: &[u32],
) -> Result<()> {
    let id = c.index.write().allocate_segment_id();
    let tx = db.store.writer()?.begin_write()?;
    let rec = match &built {
        Some(seg) => {
            let mut free = tx.open_table(FREE_REGIONS)?;
            let at = segments::allocate(&mut free, &db.container, segments::region_len(seg))?;
            let rec = segments::write_segment(&db.container, seg, at)?;
            db.container.sync().map_err(io)?;
            tx.open_table(SEGMENTS)?
                .insert((c.id, id), encode(&rec).as_slice())?;
            let mut dead_rows = tx.open_table(DEAD)?;
            for &chunk in dead {
                dead_rows.insert((c.id, chunk), id)?;
            }
            let mut log = tx.open_table(CHANGES)?;
            changes::append(&mut log, c.id, &Change::Seal { segment: id })?;
            Some(rec)
        }
        None => None,
    };
    drop_checkpoints(&tx, c.id)?;
    tx.commit()?;
    if let Some(rec) = rec {
        let seg = segments::map_segment(&db.container, &rec, db.epoch())?;
        c.index.write().install_loaded(id, Arc::new(seg), dead);
    }
    Ok(())
}

/// Deletes a collection's bulk checkpoints and frees their regions. Only
/// the writing process ever reads them, never through a mapping, so their
/// space is reusable at once.
fn drop_checkpoints(tx: &WriteTransaction, coll: u32) -> Result<()> {
    let mut bulk = tx.open_table(BULK)?;
    let mut free = tx.open_table(FREE_REGIONS)?;
    let rows: Vec<((u32, u64), BulkRecord)> = bulk
        .range((coll, 0)..=(coll, u64::MAX))?
        .map(|row| {
            let (k, v) = row?;
            Ok((k.value(), decode(v.value(), "bulk checkpoint")?))
        })
        .collect::<Result<_>>()?;
    for (key, rec) in rows {
        bulk.remove(key)?;
        segments::release(&mut free, rec.region, 0)?;
    }
    Ok(())
}

/// Finishes the checkpoints a crashed (or dropped) bulk load left: builds
/// their live chunks into a segment. Run by maintenance in the writer.
pub(crate) fn finish_leftover(db: &DbInner, c: &Arc<CollInner>) -> Result<()> {
    let _w = db.writer.lock();
    let rx = db.store.begin_read()?;
    let mut ids = Vec::new();
    let mut vectors = Vec::new();
    for row in rx.open_table(BULK)?.range((c.id, 0)..=(c.id, u64::MAX))? {
        let rec: BulkRecord = decode(row?.1.value(), "bulk checkpoint")?;
        if rec.dim as usize != c.schema.dim && rec.count > 0 {
            return Err(Error::Storage(vdblite_core::StorageError::Corrupt(
                "bulk checkpoint has the wrong dimension".into(),
            )));
        }
        let (i, v) = segments::read_bulk(&db.container, &rec)?;
        ids.extend(i);
        vectors.extend(v);
    }
    // Chunks deleted since their checkpoint are gone from `chunks`.
    let chunks = rx.open_table(CHUNKS)?;
    let mut live = RoaringBitmap::new();
    for &id in &ids {
        if chunks.get((c.id, id))?.is_some() {
            live.insert(id);
        }
    }
    drop((chunks, rx));
    let (ids, vectors) = live_rows(ids, vectors, c.schema.dim, |id| live.contains(id));
    let built = if ids.is_empty() {
        None
    } else {
        let mut seg = c.index.read().config().build_segment(&ids, &vectors);
        let rx = db.store.begin_read()?;
        crate::collection::label_with_keys(
            &rx.open_table(CHUNKS)?,
            &rx.open_table(vdblite_core::tables::DOCS)?,
            c.id,
            &mut seg,
        )?;
        Some(seg)
    };
    install(db, c, built, &[])?;
    c.bulk_leftover.store(false, Ordering::Release);
    Ok(())
}
