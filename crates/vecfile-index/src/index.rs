//! A collection's vector index: sealed segments plus one mutable segment.
//!
//! Search fans out over every segment in parallel, merges the int8 candidates
//! into one bounded heap of `k × rerank`, then re-ranks those with
//! full-precision vectors read from a [`VectorStore`].
//!
//! The index holds no full-precision vectors for sealed segments; it only
//! needs the store for re-ranking and for rebuilding segments.

use std::cell::RefCell;
use std::sync::Arc;

use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::distance::{Metric, kernels};
use crate::hnsw::HnswParams;
use crate::segment::{
    Bitset, MutableSegment, Quantization, SealedSegment, SearchParams, SegmentScratch, VectorStore,
    prepare,
};
use crate::topk::{Neighbor, TopK};

/// Settings fixed when a collection is created.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IndexConfig {
    pub dim: usize,
    pub metric: Metric,
    pub hnsw: HnswParams,
    /// Mutable segment size at which it should be sealed.
    pub seal_threshold: usize,
    /// Candidates re-ranked per result (`k × rerank`).
    pub rerank: usize,
    /// Filter selectivity below which a segment is scanned instead of walked.
    pub brute_force_below: f64,
    /// Tombstone fraction at which a sealed segment is rebuilt.
    pub rebuild_dead_ratio: f64,
    /// Largest segment that merging will produce.
    pub max_segment: usize,
    /// In-memory representation of sealed segments.
    pub quantization: Quantization,
}

impl IndexConfig {
    pub fn new(dim: usize, metric: Metric) -> Self {
        IndexConfig {
            dim,
            metric,
            hnsw: HnswParams::default(),
            seal_threshold: 65_536,
            rerank: 4,
            brute_force_below: 0.02,
            rebuild_dead_ratio: 0.2,
            max_segment: 262_144,
            quantization: Quantization::Int8,
        }
    }

    /// Builds a sealed segment with this configuration. `vectors` are
    /// row-major, one per id, already prepared (see [`prepare`]).
    pub fn build_segment(&self, ids: &[u32], vectors: &[f32]) -> SealedSegment {
        SealedSegment::build_with(
            ids,
            vectors,
            self.dim,
            self.metric,
            &self.hnsw,
            self.quantization,
        )
    }
}

/// A sealed segment with its deletion state.
#[derive(Clone, Debug)]
pub struct SealedEntry {
    /// Stable identifier, used by storage to name the segment's pages.
    pub id: u64,
    pub segment: Arc<SealedSegment>,
    pub tombstones: Bitset,
}

impl SealedEntry {
    pub fn new(id: u64, segment: Arc<SealedSegment>) -> Self {
        let tombstones = Bitset::new(segment.len());
        SealedEntry {
            id,
            segment,
            tombstones,
        }
    }

    pub fn live(&self) -> usize {
        self.segment.len() - self.tombstones.count_ones()
    }

    pub fn dead_ratio(&self) -> f64 {
        if self.segment.is_empty() {
            0.0
        } else {
            self.tombstones.count_ones() as f64 / self.segment.len() as f64
        }
    }
}

thread_local! {
    static SCRATCH: RefCell<SegmentScratch> = RefCell::new(SegmentScratch::default());
}

/// A copy of the mutable segment's contents, taken so a seal can build its
/// graph without holding any lock on the index.
pub struct MutableSnapshot {
    ids: Vec<u32>,
    vectors: Vec<f32>,
    generations: Vec<u64>,
}

impl MutableSnapshot {
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Builds the sealed segment (the slow part of a seal).
    pub fn build(self, cfg: &IndexConfig, id: u64) -> PendingSeal {
        let segment = cfg.build_segment(&self.ids, &self.vectors);
        PendingSeal {
            id,
            segment,
            generations: self.ids.into_iter().zip(self.generations).collect(),
        }
    }
}

/// A segment built from a snapshot of the mutable segment, waiting to be
/// installed.
pub struct PendingSeal {
    pub id: u64,
    pub segment: SealedSegment,
    generations: Vec<(u32, u64)>,
}

/// One unit of compaction: segments to remove and the live ids to rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionJob {
    pub remove: Vec<u64>,
    pub live: Vec<u32>,
}

/// What changed when a sealed segment was installed or rebuilt, so storage
/// can persist it.
#[derive(Debug, Default)]
pub struct Installed {
    pub added: Vec<u64>,
    pub removed: Vec<u64>,
}

#[derive(Clone, Debug)]
pub struct VectorIndex {
    cfg: IndexConfig,
    sealed: Vec<SealedEntry>,
    mutable: MutableSegment,
    next_segment_id: u64,
}

impl VectorIndex {
    pub fn new(cfg: IndexConfig) -> Self {
        VectorIndex {
            cfg,
            sealed: Vec::new(),
            mutable: MutableSegment::new(cfg.dim),
            next_segment_id: 1,
        }
    }

    pub fn config(&self) -> &IndexConfig {
        &self.cfg
    }

    pub fn sealed(&self) -> &[SealedEntry] {
        &self.sealed
    }

    pub fn mutable(&self) -> &MutableSegment {
        &self.mutable
    }

    /// Live vectors across all segments.
    pub fn len(&self) -> usize {
        self.mutable.len() + self.sealed.iter().map(SealedEntry::live).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Bytes resident for search (sealed segments plus the mutable buffer).
    pub fn memory_bytes(&self) -> usize {
        self.sealed
            .iter()
            .map(|e| e.segment.memory_bytes() + e.tombstones.len() / 8)
            .sum::<usize>()
            + 4 * (self.mutable.vectors().len() + self.mutable.ids().len())
    }

    /// Adds a sealed segment loaded from storage.
    pub fn restore_sealed(&mut self, entry: SealedEntry) {
        self.next_segment_id = self.next_segment_id.max(entry.id + 1);
        self.sealed.push(entry);
        self.sealed
            .sort_by_key(|e| e.segment.ids().first().copied());
    }

    /// Inserts a vector into the mutable segment. It is normalized for cosine.
    /// The id must not already be live in a sealed segment.
    pub fn insert(&mut self, id: u32, vector: &[f32]) {
        assert_eq!(vector.len(), self.cfg.dim, "vector length must equal dim");
        if self.cfg.metric == Metric::Cosine {
            let mut v = vector.to_vec();
            prepare(self.cfg.metric, &mut v);
            self.mutable.insert(id, &v);
        } else {
            self.mutable.insert(id, vector);
        }
    }

    /// Deletes a vector. Returns the sealed segment id it was tombstoned in, or
    /// `None` if it was in the mutable segment or not present.
    pub fn delete(&mut self, id: u32) -> Option<u64> {
        if self.mutable.remove(id) {
            return None;
        }
        for e in &mut self.sealed {
            if let Some(local) = e.segment.local_of(id) {
                e.tombstones.set(local);
                return Some(e.id);
            }
        }
        None
    }

    /// The sealed segment holding a live vector, if any.
    pub fn sealed_segment_of(&self, id: u32) -> Option<u64> {
        self.sealed.iter().find_map(|e| {
            let local = e.segment.local_of(id)?;
            (!e.tombstones.get(local)).then_some(e.id)
        })
    }

    pub fn should_seal(&self) -> bool {
        self.mutable.len() >= self.cfg.seal_threshold
    }

    /// Reserves an id for a segment about to be built, so storage can write
    /// it under its final name before it is installed.
    pub fn allocate_segment_id(&mut self) -> u64 {
        let id = self.next_segment_id;
        self.next_segment_id += 1;
        id
    }

    /// Copies the mutable segment's contents for a seal, or `None` if it is
    /// empty. Build the copy with [`MutableSnapshot::build`] outside any
    /// lock, then install it with [`Self::install_sealed`].
    pub fn snapshot_mutable(&self) -> Option<MutableSnapshot> {
        if self.mutable.is_empty() {
            return None;
        }
        Some(MutableSnapshot {
            ids: self.mutable.ids().to_vec(),
            vectors: self.mutable.vectors().to_vec(),
            generations: self.mutable.generations().to_vec(),
        })
    }

    /// Builds a sealed segment from the current mutable contents without
    /// changing the index.
    pub fn build_from_mutable(&self, id: u64) -> Option<PendingSeal> {
        Some(self.snapshot_mutable()?.build(&self.cfg, id))
    }

    /// Installs a segment built by [`Self::build_from_mutable`]. Vectors that
    /// were updated or deleted while it was building are tombstoned in the new
    /// segment; updated ones stay in the mutable segment.
    pub fn install_sealed(&mut self, pending: PendingSeal) -> &SealedEntry {
        let id = pending.id;
        self.next_segment_id = self.next_segment_id.max(id + 1);
        let mut entry = SealedEntry::new(id, Arc::new(pending.segment));
        for (global, generation) in pending.generations {
            if self.mutable.generation(global) == Some(generation) {
                self.mutable.remove(global);
            } else if let Some(local) = entry.segment.local_of(global) {
                entry.tombstones.set(local);
            }
        }
        self.sealed.push(entry);
        self.sort_sealed();
        let pos = self
            .sealed
            .iter()
            .position(|e| e.id == id)
            .expect("just inserted");
        &self.sealed[pos]
    }

    /// Installs a segment another process sealed, read back from storage.
    /// Its vectors leave the mutable segment; any no longer there were
    /// deleted before the seal and are tombstoned.
    pub fn install_loaded(&mut self, id: u64, segment: Arc<SealedSegment>) {
        self.next_segment_id = self.next_segment_id.max(id + 1);
        let mut entry = SealedEntry::new(id, segment);
        for (local, &global) in entry.segment.ids().iter().enumerate() {
            if !self.mutable.remove(global) {
                entry.tombstones.set(local as u32);
            }
        }
        self.sealed.push(entry);
        self.sort_sealed();
    }

    /// Whether a sealed segment with this id is installed.
    pub fn has_sealed(&self, id: u64) -> bool {
        self.sealed.iter().any(|e| e.id == id)
    }

    fn sort_sealed(&mut self) {
        self.sealed
            .sort_by_key(|e| e.segment.ids().first().copied());
    }

    /// Whether [`Self::compact`] would do anything.
    pub fn needs_compaction(&self, merge: bool) -> bool {
        !self.compaction_groups(merge).is_empty()
    }

    fn compaction_groups(&self, merge: bool) -> Vec<Vec<usize>> {
        let rebuild = |e: &SealedEntry| e.dead_ratio() >= self.cfg.rebuild_dead_ratio;
        let mut groups = Vec::new();
        let mut run: Vec<usize> = Vec::new();
        let mut run_live = 0;
        let flush = |run: &mut Vec<usize>, groups: &mut Vec<Vec<usize>>| {
            if run.len() > 1 || run.first().is_some_and(|&i| rebuild(&self.sealed[i])) {
                groups.push(std::mem::take(run));
            }
            run.clear();
        };
        for (i, e) in self.sealed.iter().enumerate() {
            let mergeable = merge && e.live() < self.cfg.max_segment / 2;
            if mergeable && run_live + e.live() <= self.cfg.max_segment {
                run.push(i);
                run_live += e.live();
                continue;
            }
            flush(&mut run, &mut groups);
            run_live = 0;
            if mergeable {
                run.push(i);
                run_live = e.live();
            } else if rebuild(e) {
                groups.push(vec![i]);
            }
        }
        flush(&mut run, &mut groups);
        groups
    }

    /// Plans compaction: segments whose dead ratio passed the threshold are
    /// rebuilt, and with `merge`, runs of adjacent segments whose live vectors
    /// fit in `max_segment` are merged (fewer segments, less fan-out).
    pub fn plan_compaction(&self, merge: bool) -> Vec<CompactionJob> {
        self.compaction_groups(merge)
            .into_iter()
            .map(|g| {
                let mut job = CompactionJob {
                    remove: Vec::new(),
                    live: Vec::new(),
                };
                for i in g {
                    let e = &self.sealed[i];
                    job.remove.push(e.id);
                    job.live.extend(
                        e.segment
                            .ids()
                            .iter()
                            .enumerate()
                            .filter(|&(local, _)| !e.tombstones.get(local as u32))
                            .map(|(_, &g)| g),
                    );
                }
                job
            })
            .collect()
    }

    /// Builds the replacement segment for a job, reading vectors from
    /// `store`. Needs only `&self`, so it can run outside a write lock.
    /// Returns `None` when nothing in the job is still live.
    pub fn build_compaction(
        &self,
        job: &CompactionJob,
        store: &dyn VectorStore,
    ) -> Option<SealedSegment> {
        let mut ids = Vec::with_capacity(job.live.len());
        let mut vectors = Vec::with_capacity(job.live.len() * self.cfg.dim);
        let mut buf = vec![0f32; self.cfg.dim];
        for &global in &job.live {
            if store.read(global, &mut buf) {
                prepare(self.cfg.metric, &mut buf);
                ids.push(global);
                vectors.extend_from_slice(&buf);
            }
        }
        (!ids.is_empty()).then(|| self.cfg.build_segment(&ids, &vectors))
    }

    /// Swaps the segments a job removes for its replacement (if any).
    /// Anything deleted from the old segments since planning stays deleted.
    pub fn apply_compaction(
        &mut self,
        job: &CompactionJob,
        replacement: Option<(u64, SealedSegment)>,
    ) {
        self.replace_segments(&job.remove, replacement);
    }

    /// Removes sealed segments and installs their replacement (if any),
    /// carrying over their tombstones, so anything deleted from them stays
    /// deleted.
    pub fn replace_segments(&mut self, remove: &[u64], replacement: Option<(u64, SealedSegment)>) {
        let mut dead = Vec::new();
        for e in self.sealed.iter().filter(|e| remove.contains(&e.id)) {
            dead.extend(
                e.tombstones
                    .iter_ones()
                    .map(|l| e.segment.ids()[l as usize]),
            );
        }
        self.sealed.retain(|e| !remove.contains(&e.id));
        if let Some((id, seg)) = replacement {
            self.next_segment_id = self.next_segment_id.max(id + 1);
            let mut entry = SealedEntry::new(id, Arc::new(seg));
            for g in dead {
                if let Some(local) = entry.segment.local_of(g) {
                    entry.tombstones.set(local);
                }
            }
            self.sealed.push(entry);
        }
        self.sort_sealed();
    }

    /// Seals the mutable segment in place (build + install).
    pub fn seal(&mut self) -> Option<u64> {
        let id = self.allocate_segment_id();
        let pending = self.build_from_mutable(id)?;
        Some(self.install_sealed(pending).id)
    }

    /// Plans, builds and applies compaction in one go. Returns which segment
    /// ids were added and removed.
    pub fn compact(&mut self, store: &dyn VectorStore, merge: bool) -> Installed {
        let mut out = Installed::default();
        for job in self.plan_compaction(merge) {
            let built = self.build_compaction(&job, store);
            let replacement = built.map(|seg| (self.allocate_segment_id(), seg));
            out.removed.extend(&job.remove);
            out.added.extend(replacement.as_ref().map(|r| r.0));
            self.apply_compaction(&job, replacement);
        }
        out
    }

    /// The `k` nearest live vectors to `query`, closest first, with exact
    /// distances. `filter`, when given, restricts results to its ids.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        filter: Option<&RoaringBitmap>,
        store: &dyn VectorStore,
    ) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.cfg.dim, "query length must equal dim");
        if k == 0 {
            return Vec::new();
        }
        let mut q = query.to_vec();
        prepare(self.cfg.metric, &mut q);
        let n = k * self.cfg.rerank.max(1);
        let params = SearchParams {
            n,
            ef,
            brute_force_below: self.cfg.brute_force_below,
        };

        let per_segment = |e: &SealedEntry| {
            let mut out = Vec::with_capacity(n);
            SCRATCH.with(|s| {
                e.segment.candidates(
                    &q,
                    &params,
                    Some(&e.tombstones),
                    filter,
                    &mut s.borrow_mut(),
                    &mut out,
                )
            });
            out
        };
        let lists: Vec<Vec<Neighbor>> = if self.sealed.len() > 1 {
            self.sealed.par_iter().map(per_segment).collect()
        } else {
            self.sealed.iter().map(per_segment).collect()
        };

        let mut merged = TopK::new(n);
        for c in lists.iter().flatten() {
            merged.push(c.id, c.dist);
        }
        let mut exact = Vec::new();
        self.mutable
            .candidates(&q, self.cfg.metric, n, filter, &mut exact);
        for c in &exact {
            merged.push(c.id, c.dist);
        }

        if self.cfg.quantization == Quantization::None {
            // Every candidate distance is already exact.
            let mut all = merged.into_sorted_vec();
            all.truncate(k);
            return all;
        }
        let kern = kernels();
        let mut buf = vec![0f32; self.cfg.dim];
        let mut top = TopK::new(k);
        for c in merged.into_sorted_vec() {
            if let Some(v) = self.mutable.get(c.id) {
                top.push(c.id, self.cfg.metric.distance(kern, &q, v));
            } else if store.read(c.id, &mut buf) {
                prepare(self.cfg.metric, &mut buf);
                top.push(c.id, self.cfg.metric.distance(kern, &q, &buf));
            }
        }
        top.into_sorted_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force;
    use crate::recall::recall_at_k;
    use crate::rng::SplitMix64;
    use std::collections::HashMap;

    fn random(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dim).map(|_| r.next_f64_open() as f32).collect()
    }

    fn cfg(dim: usize) -> IndexConfig {
        IndexConfig {
            seal_threshold: 1000,
            ..IndexConfig::new(dim, Metric::L2)
        }
    }

    fn filled(n: usize, dim: usize) -> (VectorIndex, HashMap<u32, Vec<f32>>, Vec<f32>) {
        let data = random(n, dim, 1);
        let mut idx = VectorIndex::new(cfg(dim));
        let mut store = HashMap::new();
        for (i, v) in data.chunks_exact(dim).enumerate() {
            idx.insert(i as u32, v);
            store.insert(i as u32, v.to_vec());
            if idx.should_seal() {
                idx.seal();
            }
        }
        (idx, store, data)
    }

    #[test]
    fn multi_segment_search_matches_brute_force() {
        let dim = 16;
        let (idx, store, data) = filled(4500, dim);
        assert_eq!(idx.sealed().len(), 4);
        assert_eq!(idx.mutable().len(), 500);
        let queries = random(50, dim, 2);
        let found: Vec<Vec<u32>> = queries
            .chunks_exact(dim)
            .map(|q| {
                idx.search(q, 10, 64, None, &store)
                    .iter()
                    .map(|n| n.id)
                    .collect()
            })
            .collect();
        let truth: Vec<Vec<u32>> = brute_force::search_batch(&data, dim, Metric::L2, &queries, 10)
            .iter()
            .map(|r| r.iter().map(|n| n.id).collect())
            .collect();
        let r = recall_at_k(&found, &truth, 10);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn deleted_vectors_never_return() {
        let dim = 8;
        let (mut idx, mut store, data) = filled(2500, dim);
        for id in (0..2500u32).filter(|i| i % 3 == 0) {
            idx.delete(id);
            store.remove(&id);
        }
        for q in data.chunks_exact(dim).take(30) {
            let hits = idx.search(q, 10, 64, None, &store);
            assert_eq!(hits.len(), 10);
            assert!(hits.iter().all(|n| n.id % 3 != 0));
        }
        assert_eq!(idx.len(), 2500 - 834);
    }

    #[test]
    fn compaction_rebuilds_and_merges() {
        let dim = 8;
        let (mut idx, mut store, data) = filled(3000, dim);
        assert_eq!(idx.sealed().len(), 3);
        // Kill half of the first segment and 90% of the other two.
        for id in 0..500u32 {
            idx.delete(id);
            store.remove(&id);
        }
        for id in (1000..3000u32).filter(|i| i % 10 != 0) {
            idx.delete(id);
            store.remove(&id);
        }
        let before = idx.len();
        assert!(idx.needs_compaction(false));
        let changes = idx.compact(&store, false);
        assert_eq!(
            changes.removed.len(),
            3,
            "every segment is over the dead ratio"
        );
        assert_eq!(changes.added.len(), 3);
        assert_eq!(idx.len(), before);
        let changes = idx.compact(&store, true);
        assert_eq!(changes.removed.len(), 3);
        assert_eq!(changes.added.len(), 1, "small neighbours merge into one");
        assert!(!idx.needs_compaction(true));
        assert_eq!(idx.len(), before);
        assert!(idx.sealed().iter().all(|e| e.tombstones.count_ones() == 0));
        let q = &data[1010 * dim..1011 * dim];
        assert_eq!(idx.search(q, 1, 64, None, &store)[0].id, 1010);
    }

    #[test]
    fn filters_restrict_results() {
        let dim = 8;
        let (idx, store, data) = filled(3000, dim);
        let allowed: RoaringBitmap = (0..3000u32).filter(|i| i % 7 == 0).collect();
        let q = &data[..dim];
        let hits = idx.search(q, 10, 64, Some(&allowed), &store);
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|n| allowed.contains(n.id)));
        assert_eq!(hits[0].id, 0);
    }

    #[test]
    fn updates_during_seal_are_kept() {
        let dim = 4;
        let mut idx = VectorIndex::new(cfg(dim));
        let mut store = HashMap::new();
        for i in 0..10u32 {
            let v = vec![i as f32; dim];
            idx.insert(i, &v);
            store.insert(i, v);
        }
        let id = idx.allocate_segment_id();
        let seg = idx.build_from_mutable(id).unwrap();
        idx.insert(3, &[100.0; 4]);
        store.insert(3, vec![100.0; 4]);
        idx.delete(4);
        store.remove(&4);
        idx.install_sealed(seg);
        assert_eq!(idx.mutable().ids(), &[3]);
        assert_eq!(idx.len(), 9);
        assert_eq!(idx.search(&[100.0; 4], 1, 16, None, &store)[0].id, 3);
        assert!(
            idx.search(&[4.0; 4], 9, 16, None, &store)
                .iter()
                .all(|n| n.id != 4)
        );
    }

    #[test]
    fn snapshot_builds_outside_the_index() {
        let dim = 4;
        let mut idx = VectorIndex::new(cfg(dim));
        let mut store = HashMap::new();
        for i in 0..10u32 {
            let v = vec![i as f32; dim];
            idx.insert(i, &v);
            store.insert(i, v);
        }
        let snap = idx.snapshot_mutable().unwrap();
        assert_eq!(snap.len(), 10);
        // The index keeps changing while the copy builds.
        idx.insert(10, &[10.0; 4]);
        store.insert(10, vec![10.0; 4]);
        idx.delete(2);
        store.remove(&2);
        let id = idx.allocate_segment_id();
        let pending = snap.build(idx.config(), id);
        idx.install_sealed(pending);
        assert_eq!(idx.mutable().ids(), &[10]);
        assert_eq!(idx.len(), 10);
        assert_eq!(idx.sealed_segment_of(5), Some(id));
        assert_eq!(idx.sealed_segment_of(2), None, "deleted during the build");
    }

    #[test]
    fn replicas_follow_seals_and_compactions() {
        // A writer index, and a replica that only sees inserts, deletes and
        // segments read back from storage, as a reading process does.
        let dim = 4;
        let mut writer = VectorIndex::new(cfg(dim));
        let mut replica = VectorIndex::new(cfg(dim));
        let mut store = HashMap::new();
        for i in 0..1500u32 {
            let v = vec![i as f32; dim];
            writer.insert(i, &v);
            replica.insert(i, &v);
            store.insert(i, v);
        }
        writer.delete(7);
        replica.delete(7);
        let id = writer.allocate_segment_id();
        let pending = writer.build_from_mutable(id).unwrap();
        let sealed = Arc::new(pending.segment.clone());
        // Deleted after the snapshot but before the seal reached the replica.
        writer.delete(8);
        replica.delete(8);
        writer.install_sealed(pending);
        replica.install_loaded(id, sealed);
        assert_eq!(replica.len(), writer.len());
        assert!(replica.has_sealed(id));
        assert_eq!(replica.sealed_segment_of(8), None);
        assert_eq!(replica.sealed_segment_of(9), Some(id));

        for i in 0..400u32 {
            writer.delete(i);
            replica.delete(i);
            store.remove(&i);
        }
        let jobs = writer.plan_compaction(false);
        assert_eq!(jobs.len(), 1);
        let built = writer.build_compaction(&jobs[0], &store).unwrap();
        writer.delete(900);
        replica.delete(900);
        let new_id = writer.allocate_segment_id();
        writer.apply_compaction(&jobs[0], Some((new_id, built.clone())));
        replica.replace_segments(&jobs[0].remove, Some((new_id, built)));
        assert_eq!(replica.len(), writer.len());
        assert!(!replica.has_sealed(id));
        assert_eq!(replica.sealed_segment_of(900), None);
        let q = [1000.0f32; 4];
        assert_eq!(
            replica.search(&q, 5, 64, None, &store),
            writer.search(&q, 5, 64, None, &store)
        );
    }

    #[test]
    fn deletes_between_plan_and_apply_stick() {
        let dim = 4;
        let (mut idx, mut store, _) = filled(2000, dim);
        for id in 0..300u32 {
            idx.delete(id);
            store.remove(&id);
        }
        let jobs = idx.plan_compaction(false);
        assert_eq!(jobs.len(), 1);
        let built = idx.build_compaction(&jobs[0], &store);
        idx.delete(500);
        store.remove(&500);
        let id = idx.allocate_segment_id();
        idx.apply_compaction(&jobs[0], built.map(|s| (id, s)));
        assert_eq!(idx.len(), 2000 - 301);
        assert!(
            idx.search(&vec![0.5; dim], 2000, 64, None, &store)
                .iter()
                .all(|n| n.id != 500)
        );
    }

    #[test]
    fn f32_mode_is_exact() {
        let dim = 8;
        let mut c = cfg(dim);
        c.quantization = Quantization::None;
        let data = random(2500, dim, 7);
        let mut idx = VectorIndex::new(c);
        for (i, v) in data.chunks_exact(dim).enumerate() {
            idx.insert(i as u32, v);
            if idx.should_seal() {
                idx.seal();
            }
        }
        let q = random(1, dim, 8);
        let got = idx.search(&q, 5, 64, None, &HashMap::new());
        let want = brute_force::search(&data, dim, Metric::L2, &q, 5);
        assert_eq!(
            got.iter().map(|n| n.id).collect::<Vec<_>>(),
            want.iter().map(|n| n.id).collect::<Vec<_>>()
        );
    }

    #[test]
    fn cosine_normalizes() {
        let mut idx = VectorIndex::new(IndexConfig::new(2, Metric::Cosine));
        idx.insert(1, &[10.0, 0.0]);
        idx.insert(2, &[0.0, 3.0]);
        let hits = idx.search(&[1.0, 0.1], 2, 16, None, &HashMap::new());
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].dist < 0.01);
    }
}
