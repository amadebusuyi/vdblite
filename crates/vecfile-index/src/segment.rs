//! Segments: the unit of indexing.
//!
//! New vectors land in a [`MutableSegment`], searched exactly by brute force.
//! When it grows large enough it is sealed into a [`SealedSegment`]: an HNSW
//! graph walked with int8 codes. Sealed segments never change; deletes are
//! tombstone bits kept beside them (see [`crate::index`]).
//!
//! Vectors are identified by global `u32` ids. Inside a sealed segment they
//! are renumbered densely (`0..len`) in ascending global-id order, so a global
//! id is found by binary search and a filter bitmap can be range-counted.

use std::collections::HashMap;

use roaring::RoaringBitmap;

use crate::distance::{Metric, kernels, normalize};
use crate::hnsw::{self, Graph, GraphParts, HnswParams, SearchScratch};
use crate::quant::ScalarQuantizer;
use crate::topk::{Neighbor, TopK};

/// Source of full-precision vectors, used for re-ranking and rebuilds.
pub trait VectorStore: Sync {
    /// Copies the vector with global id `id` into `out`; false if absent.
    fn read(&self, id: u32, out: &mut [f32]) -> bool;
}

/// A vector store backed by a map, for tests and in-memory use.
impl VectorStore for HashMap<u32, Vec<f32>> {
    fn read(&self, id: u32, out: &mut [f32]) -> bool {
        match self.get(&id) {
            Some(v) => {
                out.copy_from_slice(v);
                true
            }
            None => false,
        }
    }
}

/// Dense vectors where global id `i` is row `i`.
pub struct DenseStore<'a> {
    pub data: &'a [f32],
    pub dim: usize,
}

impl VectorStore for DenseStore<'_> {
    fn read(&self, id: u32, out: &mut [f32]) -> bool {
        let i = id as usize;
        match self.data.get(i * self.dim..(i + 1) * self.dim) {
            Some(v) => {
                out.copy_from_slice(v);
                true
            }
            None => false,
        }
    }
}

/// A fixed-size bitset over a sealed segment's local ids.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Bitset {
    words: Vec<u64>,
    len: usize,
    ones: usize,
}

impl Bitset {
    pub fn new(len: usize) -> Self {
        Bitset {
            words: vec![0; len.div_ceil(64)],
            len,
            ones: 0,
        }
    }

    #[inline]
    pub fn get(&self, i: u32) -> bool {
        self.words[i as usize / 64] >> (i % 64) & 1 == 1
    }

    /// Sets bit `i`; returns true if it was clear.
    pub fn set(&mut self, i: u32) -> bool {
        let w = &mut self.words[i as usize / 64];
        let mask = 1u64 << (i % 64);
        let was_clear = *w & mask == 0;
        *w |= mask;
        self.ones += usize::from(was_clear);
        was_clear
    }

    pub fn count_ones(&self) -> usize {
        self.ones
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn iter_ones(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.len as u32).filter(|&i| self.get(i))
    }
}

/// Per-thread buffers for segment search.
#[derive(Debug, Default)]
pub struct SegmentScratch {
    graph: SearchScratch,
    /// The query prepared for asymmetric distances.
    qprep: Vec<f32>,
    hits: Vec<Neighbor>,
}

/// How a segment search behaves.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SearchParams {
    /// Candidates wanted from the segment (`k × rerank factor`).
    pub n: usize,
    /// Graph beam width; raised to at least `n`.
    pub ef: usize,
    /// Below this filter selectivity, always scan the matching rows. Above
    /// it, a cost estimate picks between scanning and walking the graph.
    pub brute_force_below: f64,
}

/// How a sealed segment stores vectors for the graph walk.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum Quantization {
    /// int8 codes in memory; full vectors re-rank the final candidates.
    #[default]
    Int8,
    /// Full f32 vectors in memory; distances are exact, no re-ranking.
    None,
}

/// An immutable HNSW segment, walked with int8 codes or f32 vectors.
#[derive(Clone, Debug, PartialEq)]
pub struct SealedSegment {
    dim: usize,
    metric: Metric,
    graph: Graph,
    quant: ScalarQuantizer,
    codes: Vec<i8>,
    corr: Vec<f32>,
    /// Full vectors, kept only for [`Quantization::None`].
    vectors: Vec<f32>,
    ids: Vec<u32>,
}

/// The raw arrays of a [`SealedSegment`], for serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentParts {
    pub dim: usize,
    pub metric: Metric,
    pub graph: GraphParts,
    pub quant_offset: Vec<f32>,
    pub quant_scale: f32,
    pub codes: Vec<i8>,
    pub corr: Vec<f32>,
    pub vectors: Vec<f32>,
    pub ids: Vec<u32>,
}

impl SealedSegment {
    /// Builds a segment from global ids and their row-major vectors. Vectors
    /// must already be normalized for [`Metric::Cosine`].
    pub fn build(
        ids: &[u32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
        params: &HnswParams,
    ) -> Self {
        Self::build_with(ids, vectors, dim, metric, params, Quantization::Int8)
    }

    /// [`Self::build`] with a choice of in-memory representation.
    /// A segment holding this one's vectors plus `ids`, built by inserting
    /// the new vectors into this segment's graph rather than rebuilding it,
    /// so the cost is that of the new vectors only.
    ///
    /// `own_vectors` are this segment's vectors, row `i` for local id `i`
    /// (a deleted one may be its decoded code: it only steers the build).
    /// `vectors` are the new ones, one row per id; ids must not already be
    /// here. All vectors must be prepared (see [`prepare`]). Codes are kept
    /// as they are, and new ones use this segment's quantizer.
    pub fn extend(
        &self,
        own_vectors: &[f32],
        ids: &[u32],
        vectors: &[f32],
        params: &HnswParams,
    ) -> Self {
        let dim = self.dim;
        assert_eq!(
            own_vectors.len(),
            self.len() * dim,
            "one vector per own row"
        );
        assert_eq!(ids.len() * dim, vectors.len(), "one vector per id");
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_unstable_by_key(|&i| ids[i]);

        // Merge the two ascending id lists; remember where each row lands.
        let n = self.len() + ids.len();
        let mut merged_ids = Vec::with_capacity(n);
        let mut remap = Vec::with_capacity(self.len());
        let mut source = Vec::with_capacity(n); // (from_self, row)
        let (mut a, mut b) = (0, 0);
        while a < self.len() || b < order.len() {
            let take_own = b == order.len() || (a < self.len() && self.ids[a] < ids[order[b]]);
            if take_own {
                remap.push(merged_ids.len() as u32);
                merged_ids.push(self.ids[a]);
                source.push((true, a));
                a += 1;
            } else {
                let id = ids[order[b]];
                assert!(
                    merged_ids.last().is_none_or(|&l| l < id),
                    "ids must be unique and new"
                );
                merged_ids.push(id);
                source.push((false, order[b]));
                b += 1;
            }
        }
        assert!(
            merged_ids.windows(2).all(|w| w[0] < w[1]),
            "ids must be new"
        );

        let mut all = Vec::with_capacity(n * dim);
        for &(own, row) in &source {
            let v = if own { own_vectors } else { vectors };
            all.extend_from_slice(&v[row * dim..(row + 1) * dim]);
        }
        let k = kernels();
        let metric = self.metric;
        let graph = hnsw::extend(&self.graph, &remap, n, params, |a, b| {
            let (a, b) = (a as usize, b as usize);
            metric.distance(
                k,
                &all[a * dim..(a + 1) * dim],
                &all[b * dim..(b + 1) * dim],
            )
        });

        let (codes, corr, full) = if self.vectors.is_empty() {
            let mut codes = vec![0i8; n * dim];
            let mut corr = vec![0f32; n];
            for (i, &(own, row)) in source.iter().enumerate() {
                let out = &mut codes[i * dim..(i + 1) * dim];
                if own {
                    out.copy_from_slice(self.code(row as u32));
                    corr[i] = self.corr[row];
                } else {
                    corr[i] = self.quant.encode(&vectors[row * dim..(row + 1) * dim], out);
                }
            }
            (codes, corr, Vec::new())
        } else {
            (Vec::new(), Vec::new(), all)
        };
        SealedSegment {
            dim,
            metric,
            graph,
            quant: self.quant.clone(),
            codes,
            corr,
            vectors: full,
            ids: merged_ids,
        }
    }

    pub fn build_with(
        ids: &[u32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
        params: &HnswParams,
        quantization: Quantization,
    ) -> Self {
        assert_eq!(ids.len() * dim, vectors.len(), "one vector per id");
        let mut order: Vec<usize> = (0..ids.len()).collect();
        order.sort_unstable_by_key(|&i| ids[i]);
        let sorted_ids: Vec<u32> = order.iter().map(|&i| ids[i]).collect();
        assert!(
            sorted_ids.windows(2).all(|w| w[0] < w[1]),
            "ids must be unique"
        );
        let mut sorted = Vec::with_capacity(vectors.len());
        for &i in &order {
            sorted.extend_from_slice(&vectors[i * dim..(i + 1) * dim]);
        }

        let k = kernels();
        let graph = hnsw::build(sorted_ids.len(), params, |a, b| {
            let (a, b) = (a as usize, b as usize);
            metric.distance(
                k,
                &sorted[a * dim..(a + 1) * dim],
                &sorted[b * dim..(b + 1) * dim],
            )
        });
        let quant = ScalarQuantizer::train(&sorted, dim, metric);
        let (codes, corr, vectors) = match quantization {
            Quantization::Int8 => {
                let mut codes = vec![0i8; sorted.len()];
                let corr = sorted
                    .chunks_exact(dim)
                    .zip(codes.chunks_exact_mut(dim))
                    .map(|(v, c)| quant.encode(v, c))
                    .collect();
                (codes, corr, Vec::new())
            }
            Quantization::None => (Vec::new(), Vec::new(), sorted),
        };
        SealedSegment {
            dim,
            metric,
            graph,
            quant,
            codes,
            corr,
            vectors,
            ids: sorted_ids,
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn metric(&self) -> Metric {
        self.metric
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn quantization(&self) -> Quantization {
        if self.vectors.is_empty() && !self.ids.is_empty() {
            Quantization::Int8
        } else {
            Quantization::None
        }
    }

    /// Global ids in this segment, ascending; local id `i` is `ids()[i]`.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// The local id of a global id, if this segment holds it.
    pub fn local_of(&self, global: u32) -> Option<u32> {
        self.ids.binary_search(&global).ok().map(|i| i as u32)
    }

    /// Bytes resident while searching: graph, codes, corrections and ids.
    pub fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
            + self.codes.len()
            + 4 * (self.corr.len() + self.vectors.len() + self.ids.len() + self.dim)
    }

    #[inline]
    fn code(&self, local: u32) -> &[i8] {
        let i = local as usize;
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    fn vector(&self, local: u32) -> &[f32] {
        let i = local as usize;
        &self.vectors[i * self.dim..(i + 1) * self.dim]
    }

    /// Appends up to `p.n` approximate nearest neighbours of `query` to `out`
    /// as (global id, distance). Distances are int8 approximations, or exact
    /// for [`Quantization::None`]. `query` must be normalized for cosine.
    ///
    /// Rows set in `tombstones` are skipped, and when `filter` is given only
    /// global ids in it are returned. Allocates nothing once `s` is warm.
    pub fn candidates(
        &self,
        query: &[f32],
        p: &SearchParams,
        tombstones: Option<&Bitset>,
        filter: Option<&RoaringBitmap>,
        s: &mut SegmentScratch,
        out: &mut Vec<Neighbor>,
    ) {
        if self.is_empty() || p.n == 0 {
            return;
        }
        let k = kernels();
        if !self.vectors.is_empty() {
            let dist = |local: u32| self.metric.distance(k, query, self.vector(local));
            let prefetch = |local: u32| crate::distance::prefetch(self.vector(local));
            return self.candidates_with(
                dist,
                prefetch,
                p,
                tombstones,
                filter,
                &mut s.graph,
                &mut s.hits,
                out,
            );
        }
        let prefetch = |local: u32| crate::distance::prefetch(self.code(local));
        // The query stays f32; only the stored vectors are quantized.
        s.qprep.resize(self.dim, 0.0);
        let qconst = self.quant.prepare_query(query, &mut s.qprep);
        let qprep = &s.qprep;
        let dist = |local: u32| self.quant.asym_distance(k, qprep, qconst, self.code(local));
        self.candidates_with(
            dist,
            prefetch,
            p,
            tombstones,
            filter,
            &mut s.graph,
            &mut s.hits,
            out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn candidates_with<D: Fn(u32) -> f32, P: Fn(u32)>(
        &self,
        dist: D,
        prefetch: P,
        p: &SearchParams,
        tombstones: Option<&Bitset>,
        filter: Option<&RoaringBitmap>,
        graph_scratch: &mut SearchScratch,
        hits: &mut Vec<Neighbor>,
        out: &mut Vec<Neighbor>,
    ) {
        let dead = |local: u32| tombstones.is_some_and(|t| t.get(local));

        let ef = p.ef.max(p.n);
        if let Some(f) = filter {
            let first = self.ids[0];
            let last = self.ids[self.ids.len() - 1];
            let matching = f.range_cardinality(first..=last);
            let selectivity = matching as f64 / self.len() as f64;
            // A filtered walk must visit about ef / selectivity nodes to collect
            // ef matches, each costing up to 2M distances; a scan costs one
            // distance per matching row. Take whichever is cheaper.
            let walk_cost = ef as f64 / selectivity.max(1e-9) * 2.0 * self.graph.m() as f64;
            if selectivity < p.brute_force_below || (matching as f64) < walk_cost {
                let mut top = TopK::new(p.n);
                for global in f.range(first..=last) {
                    if let Some(local) = self.local_of(global)
                        && !dead(local)
                    {
                        top.push(global, dist(local));
                    }
                }
                out.extend(top.into_sorted_vec());
                return;
            }
        }

        let accept = |local: u32| {
            !dead(local) && filter.is_none_or(|f| f.contains(self.ids[local as usize]))
        };
        self.graph
            .search_prefetching(dist, accept, prefetch, ef, graph_scratch, hits);
        out.extend(
            hits.iter()
                .take(p.n)
                .map(|h| Neighbor::new(self.ids[h.id as usize], h.dist)),
        );
    }

    /// The stored vector of a local id: decoded from int8, or exact.
    pub fn decode(&self, local: u32, out: &mut [f32]) {
        if self.vectors.is_empty() {
            self.quant.decode(self.code(local), out);
        } else {
            out.copy_from_slice(self.vector(local));
        }
    }

    pub fn to_parts(&self) -> SegmentParts {
        SegmentParts {
            dim: self.dim,
            metric: self.metric,
            graph: self.graph.to_parts(),
            quant_offset: self.quant.offset().to_vec(),
            quant_scale: self.quant.scale(),
            codes: self.codes.clone(),
            corr: self.corr.clone(),
            vectors: self.vectors.clone(),
            ids: self.ids.clone(),
        }
    }

    /// Rebuilds a segment from its arrays, validating their shapes.
    pub fn from_parts(p: SegmentParts) -> Result<Self, String> {
        let graph = Graph::from_parts(p.graph)?;
        let n = graph.len();
        if p.dim == 0 || p.quant_offset.len() != p.dim {
            return Err("quantizer dimension mismatch".into());
        }
        let int8 = p.codes.len() == n * p.dim && p.corr.len() == n && p.vectors.is_empty();
        let full = p.vectors.len() == n * p.dim && p.codes.is_empty() && p.corr.is_empty();
        if !(int8 || full) || p.ids.len() != n {
            return Err("segment arrays disagree on length".into());
        }
        if !p.ids.windows(2).all(|w| w[0] < w[1]) {
            return Err("segment ids are not strictly ascending".into());
        }
        if !(p.quant_scale.is_finite() && p.quant_scale > 0.0) {
            return Err("invalid quantizer scale".into());
        }
        Ok(SealedSegment {
            dim: p.dim,
            metric: p.metric,
            graph,
            quant: ScalarQuantizer::from_parts(p.metric, p.quant_offset, p.quant_scale),
            codes: p.codes,
            corr: p.corr,
            vectors: p.vectors,
            ids: p.ids,
        })
    }
}

/// The write buffer: exact f32 vectors searched by brute force.
#[derive(Clone, Debug, Default)]
pub struct MutableSegment {
    dim: usize,
    ids: Vec<u32>,
    vectors: Vec<f32>,
    /// Bumped on every insert, so a seal can tell whether a vector changed
    /// while the segment was being built.
    gens: Vec<u64>,
    next_gen: u64,
    pos: HashMap<u32, usize>,
}

impl MutableSegment {
    pub fn new(dim: usize) -> Self {
        MutableSegment {
            dim,
            ..Default::default()
        }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    pub fn contains(&self, id: u32) -> bool {
        self.pos.contains_key(&id)
    }

    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    pub fn vectors(&self) -> &[f32] {
        &self.vectors
    }

    /// Generation stamps, aligned with [`Self::ids`].
    pub fn generations(&self) -> &[u64] {
        &self.gens
    }

    pub fn generation(&self, id: u32) -> Option<u64> {
        self.pos.get(&id).map(|&i| self.gens[i])
    }

    pub fn get(&self, id: u32) -> Option<&[f32]> {
        self.pos
            .get(&id)
            .map(|&i| &self.vectors[i * self.dim..(i + 1) * self.dim])
    }

    /// Inserts or replaces a vector.
    pub fn insert(&mut self, id: u32, v: &[f32]) {
        assert_eq!(v.len(), self.dim, "vector length must equal dim");
        self.next_gen += 1;
        if let Some(&i) = self.pos.get(&id) {
            self.vectors[i * self.dim..(i + 1) * self.dim].copy_from_slice(v);
            self.gens[i] = self.next_gen;
        } else {
            self.pos.insert(id, self.ids.len());
            self.ids.push(id);
            self.gens.push(self.next_gen);
            self.vectors.extend_from_slice(v);
        }
    }

    /// Removes a vector; returns false if it was not here.
    pub fn remove(&mut self, id: u32) -> bool {
        let Some(i) = self.pos.remove(&id) else {
            return false;
        };
        let last = self.ids.len() - 1;
        if i != last {
            let moved = self.ids[last];
            self.ids[i] = moved;
            self.gens[i] = self.gens[last];
            self.vectors
                .copy_within(last * self.dim..(last + 1) * self.dim, i * self.dim);
            self.pos.insert(moved, i);
        }
        self.ids.pop();
        self.gens.pop();
        self.vectors.truncate(last * self.dim);
        true
    }

    /// Appends up to `n` exact nearest neighbours of `query` to `out`.
    pub fn candidates(
        &self,
        query: &[f32],
        metric: Metric,
        n: usize,
        filter: Option<&RoaringBitmap>,
        out: &mut Vec<Neighbor>,
    ) {
        let k = kernels();
        let mut top = TopK::new(n);
        for (&id, v) in self
            .ids
            .iter()
            .zip(self.vectors.chunks_exact(self.dim.max(1)))
        {
            if filter.is_none_or(|f| f.contains(id)) {
                top.push(id, metric.distance(k, query, v));
            }
        }
        out.extend(top.into_sorted_vec());
    }
}

/// Normalizes `v` in place when the metric needs unit vectors.
pub fn prepare(metric: Metric, v: &mut [f32]) {
    if metric == Metric::Cosine {
        normalize(v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force;
    use crate::rng::SplitMix64;

    fn random(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dim).map(|_| r.next_f64_open() as f32).collect()
    }

    fn params(n: usize) -> SearchParams {
        SearchParams {
            n,
            ef: 64,
            brute_force_below: 0.02,
        }
    }

    #[test]
    fn candidates_cover_true_neighbours() {
        let dim = 24;
        let n = 4000;
        let data = random(n, dim, 1);
        let ids: Vec<u32> = (0..n as u32).map(|i| i * 3 + 7).collect();
        let seg = SealedSegment::build(&ids, &data, dim, Metric::L2, &HnswParams::default());
        let q = random(1, dim, 2);
        let mut out = Vec::new();
        seg.candidates(
            &q,
            &params(40),
            None,
            None,
            &mut SegmentScratch::default(),
            &mut out,
        );
        let truth = brute_force::search(&data, dim, Metric::L2, &q, 10);
        let found: Vec<u32> = out.iter().map(|n| n.id).collect();
        let hits = truth
            .iter()
            .filter(|t| found.contains(&ids[t.id as usize]))
            .count();
        assert!(
            hits >= 9,
            "only {hits}/10 true neighbours among 40 candidates"
        );
    }

    #[test]
    fn extended_segments_keep_codes_and_find_everything() {
        let dim = 24;
        let data = random(3000, dim, 21);
        let row = |i: usize| &data[i * dim..(i + 1) * dim];
        // Global ids interleave: the base holds even ids, the new rows odd.
        let base_ids: Vec<u32> = (0..2000u32).map(|i| i * 2).collect();
        let base_data: Vec<f32> = (0..2000).flat_map(|i| row(i).to_vec()).collect();
        let new_ids: Vec<u32> = (0..1000u32).rev().map(|i| i * 2 + 1).collect();
        let new_data: Vec<f32> = (0..1000)
            .rev()
            .flat_map(|i| row(2000 + i).to_vec())
            .collect();
        let base = SealedSegment::build(
            &base_ids,
            &base_data,
            dim,
            Metric::L2,
            &HnswParams::default(),
        );
        let grown = base.extend(&base_data, &new_ids, &new_data, &HnswParams::default());
        assert_eq!(grown.len(), 3000);
        assert!(grown.ids().windows(2).all(|w| w[0] < w[1]));
        assert_eq!(SealedSegment::from_parts(grown.to_parts()).unwrap(), grown);
        // The base's codes are reused unchanged.
        for (local, &g) in base.ids().iter().enumerate() {
            let gl = grown.local_of(g).unwrap();
            assert_eq!(grown.code(gl), base.code(local as u32));
        }
        // Old and new rows are both found as their own nearest neighbour.
        let mut s = SegmentScratch::default();
        for (g, v) in [(0u32, row(0)), (1u32, row(2000)), (1999u32, row(2999))] {
            let mut out = Vec::new();
            grown.candidates(v, &params(1), None, None, &mut s, &mut out);
            assert_eq!(out[0].id, g);
        }
    }

    #[test]
    fn tombstones_and_filters_exclude_rows() {
        let dim = 8;
        let n = 3000;
        let data = random(n, dim, 3);
        let ids: Vec<u32> = (0..n as u32).collect();
        let seg = SealedSegment::build(&ids, &data, dim, Metric::L2, &HnswParams::default());
        let mut tomb = Bitset::new(n);
        for i in (0..n as u32).step_by(2) {
            tomb.set(i);
        }
        let mut s = SegmentScratch::default();
        let mut out = Vec::new();
        seg.candidates(
            &data[..dim],
            &params(20),
            Some(&tomb),
            None,
            &mut s,
            &mut out,
        );
        assert_eq!(out.len(), 20);
        assert!(out.iter().all(|n| n.id % 2 == 1));

        // Selective filter takes the brute-force path; a broad one walks the graph.
        for allowed in [
            RoaringBitmap::from_iter([5u32, 77, 901]),
            (0..n as u32).step_by(3).collect(),
        ] {
            out.clear();
            seg.candidates(
                &data[..dim],
                &params(20),
                None,
                Some(&allowed),
                &mut s,
                &mut out,
            );
            assert_eq!(out.len(), allowed.len().min(20) as usize);
            assert!(out.iter().all(|n| allowed.contains(n.id)));
        }
    }

    #[test]
    fn parts_round_trip() {
        let dim = 4;
        let data = random(300, dim, 4);
        let ids: Vec<u32> = (100..400).collect();
        let seg = SealedSegment::build(&ids, &data, dim, Metric::Dot, &HnswParams::default());
        let back = SealedSegment::from_parts(seg.to_parts()).unwrap();
        assert_eq!(back, seg);
        let mut bad = seg.to_parts();
        bad.ids.swap(0, 1);
        assert!(SealedSegment::from_parts(bad).is_err());
    }

    #[test]
    fn f32_segments_are_exact() {
        let dim = 8;
        let n = 2000;
        let data = random(n, dim, 5);
        let ids: Vec<u32> = (0..n as u32).collect();
        let seg = SealedSegment::build_with(
            &ids,
            &data,
            dim,
            Metric::L2,
            &HnswParams::default(),
            Quantization::None,
        );
        assert_eq!(seg.quantization(), Quantization::None);
        let q = random(1, dim, 6);
        let mut out = Vec::new();
        seg.candidates(
            &q,
            &params(10),
            None,
            None,
            &mut SegmentScratch::default(),
            &mut out,
        );
        let truth = brute_force::search(&data, dim, Metric::L2, &q, 1);
        assert_eq!(out[0].id, truth[0].id);
        assert_eq!(out[0].dist, truth[0].dist);
        assert_eq!(SealedSegment::from_parts(seg.to_parts()).unwrap(), seg);
    }

    #[test]
    fn mutable_insert_remove() {
        let mut m = MutableSegment::new(2);
        m.insert(1, &[0.0, 0.0]);
        m.insert(2, &[1.0, 1.0]);
        m.insert(3, &[2.0, 2.0]);
        assert!(m.remove(1));
        assert!(!m.remove(1));
        assert_eq!(m.get(3), Some(&[2.0, 2.0][..]));
        m.insert(2, &[5.0, 5.0]);
        let mut out = Vec::new();
        m.candidates(&[5.0, 5.0], Metric::L2, 1, None, &mut out);
        assert_eq!(out[0].id, 2);
        assert_eq!(m.len(), 2);
    }
}
