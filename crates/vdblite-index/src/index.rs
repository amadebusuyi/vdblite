//! A collection's vector index: sealed segments plus one mutable segment.
//!
//! Search fans out over every segment in parallel, keeps the best `k ×
//! rerank` int8 candidates, then re-ranks those with each segment's own
//! full-precision vectors (mapped from the file for a loaded segment, so
//! a re-rank read is a memory load, not a storage lookup).

use std::cell::RefCell;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::distance::{Metric, kernels};
use crate::hnsw::HnswParams;
use crate::segment::{
    Bitset, MutableSegment, Quantization, Row, SealedSegment, SearchParams, SegmentScratch, prepare,
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
            max_segment: 1_048_576,
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

/// Searches in flight in this process, across every index. A search fans
/// out over its segments only while there are idle cores to fan out to;
/// under load, each search runs on its caller's thread, which wastes nothing
/// on hand-offs between threads.
static IN_FLIGHT: AtomicUsize = AtomicUsize::new(0);

/// Searches running in this process right now, across every index.
pub fn searches_in_flight() -> usize {
    IN_FLIGHT.load(Ordering::Relaxed)
}

struct InFlight(usize);

impl InFlight {
    fn enter() -> Self {
        InFlight(IN_FLIGHT.fetch_add(1, Ordering::Relaxed) + 1)
    }
}

impl Drop for InFlight {
    fn drop(&mut self) {
        IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
    }
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

/// One unit of compaction: segments to remove and the live rows to rebuild.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompactionJob {
    pub remove: Vec<u64>,
    /// A segment (one of `remove`) to grow rather than rebuild: its graph
    /// and codes are kept, and the other segments' vectors are inserted.
    pub base: Option<u64>,
    /// Live rows to build from, as (segment id, local ids): every removed
    /// segment's without a base, the other segments' with one.
    pub live: Vec<(u64, Vec<u32>)>,
}

impl CompactionJob {
    /// Rows the job inserts (or builds from, without a base).
    pub fn live_rows(&self) -> usize {
        self.live.iter().map(|(_, l)| l.len()).sum()
    }
}

/// What a compaction builds from, taken from the index so the build can run
/// with no lock held: the segment to grow (if any) and the rows to add.
pub struct CompactionInput {
    pub base: Option<Arc<SealedSegment>>,
    pub rows: Vec<(Arc<SealedSegment>, Vec<u32>)>,
}

impl CompactionInput {
    /// Builds the replacement segment; `None` if there is nothing to build
    /// from. Vectors come from the source segments, already prepared.
    pub fn build(&self, cfg: &IndexConfig) -> Option<SealedSegment> {
        let count: usize = self.rows.iter().map(|(_, l)| l.len()).sum();
        let mut ids = Vec::with_capacity(count);
        let mut vectors = Vec::with_capacity(count * cfg.dim);
        for (seg, locals) in &self.rows {
            for &local in locals {
                ids.push(seg.ids()[local as usize]);
                seg.vector(local).extend_into(&mut vectors);
            }
        }
        let built = match &self.base {
            Some(base) => base.extend(&ids, &vectors, &cfg.hnsw),
            None if ids.is_empty() => return None,
            None => cfg.build_segment(&ids, &vectors),
        };
        // Labels carry over from the sources, when they all have them.
        let sources: Vec<&SealedSegment> = self
            .base
            .iter()
            .map(|b| &**b)
            .chain(self.rows.iter().map(|(s, _)| &**s))
            .collect();
        if sources.iter().all(|s| s.has_labels()) {
            Some(built.with_labels(|g| {
                sources
                    .iter()
                    .find_map(|s| s.local_of(g).and_then(|l| s.label(l)))
            }))
        } else {
            Some(built)
        }
    }
}

/// Background maintenance merges a run of adjacent segments once it is this
/// long. Merges grow the run's largest segment, so each vector is inserted
/// into a large graph about once rather than rebuilt at every merge.
const BACKGROUND_MERGE_RUN: usize = 4;

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

    /// The label a sealed segment stores for a vector (see
    /// [`SealedSegment::with_labels`]); `None` for the write buffer or a
    /// segment without labels.
    pub fn label_of(&self, id: u32) -> Option<&[u8]> {
        self.sealed.iter().find_map(|e| {
            let local = e.segment.local_of(id)?;
            e.segment.label(local)
        })
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

    /// Installs a segment another process sealed, read back from storage,
    /// with the global ids storage has tombstoned in it. Its vectors leave
    /// the mutable segment.
    pub fn install_loaded(&mut self, id: u64, segment: Arc<SealedSegment>, dead: &[u32]) {
        self.next_segment_id = self.next_segment_id.max(id + 1);
        let mut entry = SealedEntry::new(id, segment);
        for &global in entry.segment.ids() {
            self.mutable.remove(global);
        }
        for &global in dead {
            if let Some(local) = entry.segment.local_of(global) {
                entry.tombstones.set(local);
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

    /// Runs of adjacent segments whose live vectors fit in `max_segment`
    /// merge: any run of two or more with `merge`, otherwise only runs of
    /// [`BACKGROUND_MERGE_RUN`] or more. A segment outside such a run whose
    /// dead ratio passed the threshold is rebuilt alone.
    fn compaction_groups(&self, merge: bool) -> Vec<Vec<usize>> {
        let rebuild = |e: &SealedEntry| e.dead_ratio() >= self.cfg.rebuild_dead_ratio;
        let min_run = if merge { 2 } else { BACKGROUND_MERGE_RUN };
        let mut groups = Vec::new();
        let mut run: Vec<usize> = Vec::new();
        let mut run_live = 0;
        let flush = |run: &mut Vec<usize>, groups: &mut Vec<Vec<usize>>| {
            if run.len() >= min_run {
                groups.push(std::mem::take(run));
            } else {
                groups.extend(
                    run.iter()
                        .filter(|&&i| rebuild(&self.sealed[i]))
                        .map(|&i| vec![i]),
                );
            }
            run.clear();
        };
        for (i, e) in self.sealed.iter().enumerate() {
            if !run.is_empty() && run_live + e.live() > self.cfg.max_segment {
                flush(&mut run, &mut groups);
                run_live = 0;
            }
            run.push(i);
            run_live += e.live();
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
                // Grow the largest clean member if it holds at least half the
                // live vectors; otherwise inserting would cost about as much
                // as rebuilding, and a rebuild sheds the tombstones.
                let total: usize = g.iter().map(|&i| self.sealed[i].live()).sum();
                let base = (g.len() > 1)
                    .then(|| {
                        g.iter()
                            .map(|&i| &self.sealed[i])
                            .filter(|e| e.dead_ratio() < self.cfg.rebuild_dead_ratio)
                            .max_by_key(|e| e.live())
                    })
                    .flatten()
                    .filter(|e| e.live() * 2 >= total)
                    .map(|e| e.id);
                let mut job = CompactionJob {
                    remove: Vec::new(),
                    base,
                    live: Vec::new(),
                };
                for i in g {
                    let e = &self.sealed[i];
                    job.remove.push(e.id);
                    if Some(e.id) == base {
                        continue;
                    }
                    let live = (0..e.segment.len() as u32)
                        .filter(|&local| !e.tombstones.get(local))
                        .collect();
                    job.live.push((e.id, live));
                }
                job
            })
            .collect()
    }

    /// The sealed segment with this id.
    pub fn sealed_by_id(&self, id: u64) -> Option<&SealedEntry> {
        self.sealed.iter().find(|e| e.id == id)
    }

    /// What a job builds from (see [`CompactionInput`]). Cheap: it clones
    /// segment handles, not vectors.
    pub fn compaction_input(&self, job: &CompactionJob) -> CompactionInput {
        let segment = |id: u64| {
            self.sealed_by_id(id)
                .map(|e| e.segment.clone())
                .expect("a compaction job names installed segments")
        };
        CompactionInput {
            base: job.base.map(segment),
            rows: job
                .live
                .iter()
                .map(|(id, locals)| (segment(*id), locals.clone()))
                .collect(),
        }
    }

    /// Builds the replacement segment for a job. Returns `None` when nothing
    /// in the job is still live.
    pub fn build_compaction(&self, job: &CompactionJob) -> Option<SealedSegment> {
        self.compaction_input(job).build(&self.cfg)
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
    pub fn compact(&mut self, merge: bool) -> Installed {
        let mut out = Installed::default();
        for job in self.plan_compaction(merge) {
            let built = self.build_compaction(&job);
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
        let in_flight = InFlight::enter();
        let fan_out = self.sealed.len() > 1
            && in_flight.0 * self.sealed.len() <= rayon::current_num_threads();
        let lists: Vec<Vec<Neighbor>> = if fan_out {
            self.sealed.par_iter().map(per_segment).collect()
        } else {
            self.sealed.iter().map(per_segment).collect()
        };

        // Candidates with where they came from: a sealed segment's index, or
        // `MUTABLE` for the write buffer, whose distances are already exact.
        const MUTABLE: usize = usize::MAX;
        let mut cands: Vec<(Neighbor, usize)> = Vec::with_capacity(n * (lists.len() + 1));
        for (si, list) in lists.iter().enumerate() {
            cands.extend(list.iter().map(|&c| (c, si)));
        }
        let mut exact = Vec::new();
        self.mutable
            .candidates(&q, self.cfg.metric, n, filter, &mut exact);
        cands.extend(exact.into_iter().map(|c| (c, MUTABLE)));
        if cands.len() > n {
            cands.select_nth_unstable_by(n - 1, |a, b| a.0.cmp(&b.0));
            cands.truncate(n);
        }

        if self.cfg.quantization == Quantization::None {
            // Every candidate distance is already exact.
            let mut all: Vec<Neighbor> = cands.into_iter().map(|(c, _)| c).collect();
            all.sort_unstable();
            all.truncate(k);
            return all;
        }
        // Find each candidate's full vector and start loading them all, so
        // the re-rank reads overlap instead of stalling one by one.
        let rows: Vec<(Neighbor, Option<Row>)> = cands
            .iter()
            .map(|&(c, si)| {
                let v = (si != MUTABLE).then(|| {
                    let seg = &self.sealed[si].segment;
                    let local = seg.local_of(c.id).expect("a candidate is in its segment");
                    seg.vector(local)
                });
                if let Some(v) = v {
                    v.prefetch();
                }
                (c, v)
            })
            .collect();
        let kern = kernels();
        let mut top = TopK::new(k);
        let mut scratch = Vec::new();
        for (c, v) in rows {
            let d = match v {
                Some(v) => v.distance(self.cfg.metric, kern, &q, &mut scratch),
                None => c.dist,
            };
            top.push(c.id, d);
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

    fn filled(n: usize, dim: usize) -> (VectorIndex, Vec<f32>) {
        let data = random(n, dim, 1);
        let mut idx = VectorIndex::new(cfg(dim));
        for (i, v) in data.chunks_exact(dim).enumerate() {
            idx.insert(i as u32, v);
            if idx.should_seal() {
                idx.seal();
            }
        }
        (idx, data)
    }

    #[test]
    fn multi_segment_search_matches_brute_force() {
        let dim = 16;
        let (idx, data) = filled(4500, dim);
        assert_eq!(idx.sealed().len(), 4);
        assert_eq!(idx.mutable().len(), 500);
        let queries = random(50, dim, 2);
        let found: Vec<Vec<u32>> = queries
            .chunks_exact(dim)
            .map(|q| idx.search(q, 10, 64, None).iter().map(|n| n.id).collect())
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
        let (mut idx, data) = filled(2500, dim);
        for id in (0..2500u32).filter(|i| i % 3 == 0) {
            idx.delete(id);
        }
        for q in data.chunks_exact(dim).take(30) {
            let hits = idx.search(q, 10, 64, None);
            assert_eq!(hits.len(), 10);
            assert!(hits.iter().all(|n| n.id % 3 != 0));
        }
        assert_eq!(idx.len(), 2500 - 834);
    }

    #[test]
    fn compaction_rebuilds_and_merges() {
        let dim = 8;
        let (mut idx, data) = filled(3000, dim);
        assert_eq!(idx.sealed().len(), 3);
        // Kill half of the first segment and 90% of the other two.
        for id in 0..500u32 {
            idx.delete(id);
        }
        for id in (1000..3000u32).filter(|i| i % 10 != 0) {
            idx.delete(id);
        }
        let before = idx.len();
        assert!(idx.needs_compaction(false));
        let changes = idx.compact(false);
        assert_eq!(
            changes.removed.len(),
            3,
            "every segment is over the dead ratio"
        );
        assert_eq!(changes.added.len(), 3);
        assert_eq!(idx.len(), before);
        let changes = idx.compact(true);
        assert_eq!(changes.removed.len(), 3);
        assert_eq!(changes.added.len(), 1, "small neighbours merge into one");
        assert!(!idx.needs_compaction(true));
        assert_eq!(idx.len(), before);
        assert!(idx.sealed().iter().all(|e| e.tombstones.count_ones() == 0));
        let q = &data[1010 * dim..1011 * dim];
        assert_eq!(idx.search(q, 1, 64, None)[0].id, 1010);
    }

    #[test]
    fn filters_restrict_results() {
        let dim = 8;
        let (idx, data) = filled(3000, dim);
        let allowed: RoaringBitmap = (0..3000u32).filter(|i| i % 7 == 0).collect();
        let q = &data[..dim];
        let hits = idx.search(q, 10, 64, Some(&allowed));
        assert_eq!(hits.len(), 10);
        assert!(hits.iter().all(|n| allowed.contains(n.id)));
        assert_eq!(hits[0].id, 0);
    }

    #[test]
    fn updates_during_seal_are_kept() {
        let dim = 4;
        let mut idx = VectorIndex::new(cfg(dim));
        for i in 0..10u32 {
            let v = vec![i as f32; dim];
            idx.insert(i, &v);
        }
        let id = idx.allocate_segment_id();
        let seg = idx.build_from_mutable(id).unwrap();
        idx.insert(3, &[100.0; 4]);
        idx.delete(4);
        idx.install_sealed(seg);
        assert_eq!(idx.mutable().ids(), &[3]);
        assert_eq!(idx.len(), 9);
        assert_eq!(idx.search(&[100.0; 4], 1, 16, None)[0].id, 3);
        assert!(idx.search(&[4.0; 4], 9, 16, None).iter().all(|n| n.id != 4));
    }

    #[test]
    fn snapshot_builds_outside_the_index() {
        let dim = 4;
        let mut idx = VectorIndex::new(cfg(dim));
        for i in 0..10u32 {
            let v = vec![i as f32; dim];
            idx.insert(i, &v);
        }
        let snap = idx.snapshot_mutable().unwrap();
        assert_eq!(snap.len(), 10);
        // The index keeps changing while the copy builds.
        idx.insert(10, &[10.0; 4]);
        idx.delete(2);
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
        for i in 0..1500u32 {
            let v = vec![i as f32; dim];
            writer.insert(i, &v);
            replica.insert(i, &v);
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
        // Storage tombstoned 8 in the new segment (deleted during the build).
        replica.install_loaded(id, sealed, &[8]);
        assert_eq!(replica.len(), writer.len());
        assert!(replica.has_sealed(id));
        assert_eq!(replica.sealed_segment_of(8), None);
        assert_eq!(replica.sealed_segment_of(9), Some(id));

        for i in 0..400u32 {
            writer.delete(i);
            replica.delete(i);
        }
        let jobs = writer.plan_compaction(false);
        assert_eq!(jobs.len(), 1);
        let built = writer.build_compaction(&jobs[0]).unwrap();
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
            replica.search(&q, 5, 64, None),
            writer.search(&q, 5, 64, None)
        );
    }

    #[test]
    fn merges_grow_the_largest_segment() {
        let dim = 8;
        let data = random(4500, dim, 1);
        let mut idx = VectorIndex::new(cfg(dim));
        let add = |idx: &mut VectorIndex, range: std::ops::Range<usize>| {
            for i in range {
                let v = &data[i * dim..(i + 1) * dim];
                idx.insert(i as u32, v);
                if idx.should_seal() {
                    idx.seal();
                }
            }
        };
        // Equal segments: no member dominates, so the merge rebuilds.
        add(&mut idx, 0..3000);
        let jobs = idx.plan_compaction(true);
        assert_eq!((jobs.len(), jobs[0].base), (1, None));
        idx.compact(true);
        assert_eq!(idx.sealed().len(), 1);
        // A big segment and a small one: the merge grows the big one.
        add(&mut idx, 3000..4500);
        assert_eq!(idx.sealed().len(), 2);
        let jobs = idx.plan_compaction(true);
        assert_eq!(jobs.len(), 1);
        let job = &jobs[0];
        let big = idx.sealed().iter().max_by_key(|e| e.live()).unwrap().id;
        assert_eq!(job.base, Some(big));
        assert_eq!(
            job.live_rows(),
            1000,
            "only the small segment's rows are inserted"
        );
        let built = idx.build_compaction(job).unwrap();
        idx.delete(10); // deleted during the build: stays deleted
        let id = idx.allocate_segment_id();
        idx.apply_compaction(job, Some((id, built)));
        assert_eq!(idx.sealed().len(), 1);
        assert_eq!(idx.len(), 4500 - 1);
        assert_eq!(idx.sealed_segment_of(10), None);
        for i in [777usize, 3500] {
            let q = &data[i * dim..(i + 1) * dim];
            assert_eq!(idx.search(q, 1, 64, None)[0].id, i as u32);
        }
    }

    #[test]
    fn background_merges_wait_for_a_run() {
        let dim = 4;
        let (idx, _) = filled(3000, dim);
        assert_eq!(idx.sealed().len(), 3);
        assert!(
            !idx.needs_compaction(false),
            "three segments: no background merge yet"
        );
        assert!(idx.needs_compaction(true));
        let (idx, _) = filled(4000, dim);
        assert!(
            idx.needs_compaction(false),
            "four segments merge in the background"
        );
    }

    #[test]
    fn deletes_between_plan_and_apply_stick() {
        let dim = 4;
        let (mut idx, _) = filled(2000, dim);
        for id in 0..300u32 {
            idx.delete(id);
        }
        let jobs = idx.plan_compaction(false);
        assert_eq!(jobs.len(), 1);
        let built = idx.build_compaction(&jobs[0]);
        idx.delete(500);
        let id = idx.allocate_segment_id();
        idx.apply_compaction(&jobs[0], built.map(|s| (id, s)));
        assert_eq!(idx.len(), 2000 - 301);
        assert!(
            idx.search(&vec![0.5; dim], 2000, 64, None)
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
        let got = idx.search(&q, 5, 64, None);
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
        let hits = idx.search(&[1.0, 0.1], 2, 16, None);
        assert_eq!(hits[0].id, 1);
        assert!(hits[0].dist < 0.01);
    }
}
