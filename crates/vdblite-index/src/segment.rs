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

use half::f16;
use half::slice::HalfFloatSliceExt;
use rayon::prelude::*;
use roaring::RoaringBitmap;

use crate::buf::Buf;
use crate::distance::{Kernels, Metric, kernels, normalize};
use crate::hnsw::{self, Graph, GraphParts, HnswParams, SearchScratch};
use crate::quant::ScalarQuantizer;
use crate::rabitq::{self, BinaryQuantizer, BinaryQuery};
use crate::topk::{Neighbor, TopK};

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
    bquery: BinaryQuery,
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
    /// int8 codes walk the graph; full vectors re-rank the final candidates.
    #[default]
    Int8,
    /// Full f32 vectors walk the graph; distances are exact, no re-ranking.
    None,
    /// 1-bit codes ([`crate::rabitq`]) walk the graph, an eighth of int8's
    /// size; full vectors re-rank. The graph is built on int8 codes.
    Binary,
}

/// A segment's full-precision vectors (prepared: unit length for cosine),
/// row `i` for local id `i`.
#[derive(Clone, Debug, PartialEq)]
pub enum FullVectors {
    /// What [`Quantization::None`] segments walk.
    F32(Buf<f32>),
    /// What int8 segments keep: re-ranking a few hundred candidates in half
    /// precision orders them as f32 does, at half the size on disk and in
    /// the page cache. (Converting back and forth is lossless after the
    /// first rounding, so rebuilds don't drift.)
    F16(Buf<f16>),
}

impl FullVectors {
    /// `vectors` as they are, or narrowed to half precision.
    pub fn new(vectors: Vec<f32>, half: bool) -> Self {
        if half {
            let mut out = vec![f16::ZERO; vectors.len()];
            out.par_chunks_mut(1 << 16)
                .zip(vectors.par_chunks(1 << 16))
                .for_each(|(o, v)| o.convert_from_f32_slice(v));
            FullVectors::F16(out.into())
        } else {
            FullVectors::F32(vectors.into())
        }
    }

    /// Values (rows × dimension).
    pub fn len(&self) -> usize {
        match self {
            FullVectors::F32(v) => v.len(),
            FullVectors::F16(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn bytes(&self) -> &[u8] {
        match self {
            FullVectors::F32(v) => bytemuck::cast_slice(v),
            FullVectors::F16(v) => bytemuck::cast_slice(v),
        }
    }

    #[inline]
    fn row(&self, i: usize, dim: usize) -> Row<'_> {
        let r = i * dim..(i + 1) * dim;
        match self {
            FullVectors::F32(v) => Row::F32(&v[r]),
            FullVectors::F16(v) => Row::F16(&v[r]),
        }
    }
}

/// One row of [`FullVectors`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Row<'a> {
    F32(&'a [f32]),
    F16(&'a [f16]),
}

impl Row<'_> {
    /// Starts loading the row into cache.
    #[inline]
    pub fn prefetch(self) {
        match self {
            Row::F32(v) => crate::distance::prefetch(v),
            Row::F16(v) => crate::distance::prefetch(v),
        }
    }

    /// Appends the row to `out` as f32.
    pub fn extend_into(self, out: &mut Vec<f32>) {
        match self {
            Row::F32(v) => out.extend_from_slice(v),
            Row::F16(v) => {
                let at = out.len();
                out.resize(at + v.len(), 0.0);
                v.convert_to_f32_slice(&mut out[at..]);
            }
        }
    }

    /// The distance from `query` (prepared) to this row. A half-precision
    /// row is widened into `scratch` first.
    #[inline]
    pub fn distance(
        self,
        metric: Metric,
        k: &Kernels,
        query: &[f32],
        scratch: &mut Vec<f32>,
    ) -> f32 {
        match self {
            Row::F32(v) => metric.distance(k, query, v),
            Row::F16(v) => {
                scratch.resize(v.len(), 0.0);
                v.convert_to_f32_slice(scratch);
                metric.distance(k, query, scratch)
            }
        }
    }
}

/// An immutable HNSW segment, walked with int8 codes or f32 vectors.
///
/// Every segment keeps its full-precision vectors ([`FullVectors`]: f16
/// for int8 segments). They re-rank candidates, rebuild and grow graphs,
/// and are what [`Quantization::None`] walks. Its arrays
/// are [`Buf`]s, so a segment loaded from a file borrows them from a mapped
/// region instead of copying them.
#[derive(Clone, Debug, PartialEq)]
pub struct SealedSegment {
    dim: usize,
    metric: Metric,
    graph: Graph,
    quant: ScalarQuantizer,
    /// int8 codes, row-major by local id ([`Quantization::Int8`] only).
    codes: Buf<i8>,
    /// 1-bit codes, `binary.words()` per row ([`Quantization::Binary`]
    /// only).
    bits: Buf<u64>,
    binary: Option<Box<BinaryQuantizer>>,
    /// Per row: the int8 correction, or [`rabitq::FACTORS`] binary factors.
    corr: Buf<f32>,
    full: FullVectors,
    ids: Buf<u32>,
    /// Optional per-row labels (opaque bytes; vdblite stores each chunk's
    /// document key): row `i` is `label_bytes[label_offsets[i]..
    /// label_offsets[i + 1]]`. Both empty when the segment has none.
    label_offsets: Buf<u32>,
    label_bytes: Buf<u8>,
}

/// The arrays of a [`SealedSegment`], for serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentParts {
    pub dim: usize,
    pub metric: Metric,
    pub graph: GraphParts,
    pub quant_offset: Vec<f32>,
    pub quant_scale: f32,
    /// Empty but for [`Quantization::Int8`].
    pub codes: Buf<i8>,
    /// Empty but for [`Quantization::Binary`].
    pub bits: Buf<u64>,
    /// The binary quantizer's [`BinaryQuantizer::state`]; empty but for
    /// [`Quantization::Binary`].
    pub binary_state: Vec<f32>,
    pub corr: Buf<f32>,
    pub full: FullVectors,
    pub ids: Buf<u32>,
    /// Empty, or one more than the rows.
    pub label_offsets: Buf<u32>,
    pub label_bytes: Buf<u8>,
}

impl SealedSegment {
    /// Builds a segment from global ids and their row-major vectors. Vectors
    /// must already be prepared (see [`prepare`]).
    pub fn build(
        ids: &[u32],
        vectors: &[f32],
        dim: usize,
        metric: Metric,
        params: &HnswParams,
    ) -> Self {
        Self::build_with(ids, vectors, dim, metric, params, Quantization::Int8)
    }

    /// A segment holding this one's vectors plus `ids`, built by inserting
    /// the new vectors into this segment's graph rather than rebuilding it,
    /// so the cost is that of the new vectors only.
    ///
    /// `vectors` are the new ones, one row per id, prepared (see
    /// [`prepare`]); ids must not already be here. Codes are kept as they
    /// are, and new ones use this segment's quantizer.
    pub fn extend(&self, ids: &[u32], vectors: &[f32], params: &HnswParams) -> Self {
        let dim = self.dim;
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

        let mut full = Vec::with_capacity(n * dim);
        for &(from_self, r) in &source {
            if from_self {
                self.full.row(r, dim).extend_into(&mut full);
            } else {
                full.extend_from_slice(&vectors[r * dim..(r + 1) * dim]);
            }
        }
        let quantization = self.quantization();
        // The graph grows on int8 codes, as it was built: a binary segment
        // encodes them afresh (they are cheap) and keeps only its bits.
        let (codes, corr) = match quantization {
            Quantization::Int8 => {
                let mut codes = vec![0i8; n * dim];
                let mut corr = vec![0f32; n];
                for (i, &(own, r)) in source.iter().enumerate() {
                    let out = &mut codes[i * dim..(i + 1) * dim];
                    if own {
                        out.copy_from_slice(self.code(r as u32));
                        corr[i] = self.corr[r];
                    } else {
                        corr[i] = self.quant.encode(&vectors[r * dim..(r + 1) * dim], out);
                    }
                }
                (codes, corr)
            }
            Quantization::Binary => encode_int8(&self.quant, &full, dim),
            Quantization::None => (Vec::new(), Vec::new()),
        };
        let graph = if quantization == Quantization::None {
            let (dist, prefetch) = full_distance(self.metric, &full, dim);
            hnsw::extend_prefetching(&self.graph, &remap, n, params, dist, prefetch)
        } else {
            let (dist, prefetch) = code_distance(&self.quant, &codes, &corr, dim);
            hnsw::extend_prefetching(&self.graph, &remap, n, params, dist, prefetch)
        };
        let (codes, bits, corr) = match &self.binary {
            Some(bq) => {
                let (bits, factors) = encode_binary(bq, &full);
                (Vec::new(), bits, factors)
            }
            None => (codes, Vec::new(), corr),
        };
        SealedSegment {
            dim,
            metric: self.metric,
            graph,
            quant: self.quant.clone(),
            codes: codes.into(),
            bits: bits.into(),
            binary: self.binary.clone(),
            corr: corr.into(),
            full: FullVectors::new(full, quantization != Quantization::None),
            ids: merged_ids.into(),
            label_offsets: Buf::empty(),
            label_bytes: Buf::empty(),
        }
    }

    /// [`Self::build`] with a choice of how the graph is walked.
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

        let quant = ScalarQuantizer::train(&sorted, dim, metric);
        let n = sorted_ids.len();
        let (codes, corr, graph) = match quantization {
            // The graph is built on int8 codes: a quarter of the memory
            // traffic of the full vectors.
            Quantization::Int8 | Quantization::Binary => {
                let (codes, corr) = encode_int8(&quant, &sorted, dim);
                // Measured on 200K real 768-dim embeddings: a quarter of the
                // f32 build time, and the same recall at every ef.
                let (dist, prefetch) = code_distance(&quant, &codes, &corr, dim);
                let graph = hnsw::build_prefetching(n, params, dist, prefetch);
                (codes, corr, graph)
            }
            Quantization::None => {
                let (dist, prefetch) = full_distance(metric, &sorted, dim);
                let graph = hnsw::build_prefetching(n, params, dist, prefetch);
                (Vec::new(), Vec::new(), graph)
            }
        };
        let (codes, bits, binary, corr) = if quantization == Quantization::Binary {
            let bq = BinaryQuantizer::train(&sorted, dim, metric);
            let (bits, factors) = encode_binary(&bq, &sorted);
            (Vec::new(), bits, Some(Box::new(bq)), factors)
        } else {
            (codes, Vec::new(), None, corr)
        };
        SealedSegment {
            dim,
            metric,
            graph,
            quant,
            codes: codes.into(),
            bits: bits.into(),
            binary,
            corr: Buf::owned(corr),
            full: FullVectors::new(sorted, quantization != Quantization::None),
            ids: sorted_ids.into(),
            label_offsets: Buf::empty(),
            label_bytes: Buf::empty(),
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

    /// This segment walked with `quantization` instead: same graph, same
    /// vectors (as precise as this segment kept them), new codes.
    pub fn requantize(&self, quantization: Quantization) -> Self {
        let mut full = Vec::with_capacity(self.full.len());
        for local in 0..self.len() as u32 {
            self.vector(local).extend_into(&mut full);
        }
        let (codes, bits, binary, corr) = match quantization {
            Quantization::Int8 => {
                let (codes, corr) = encode_int8(&self.quant, &full, self.dim);
                (codes, Vec::new(), None, corr)
            }
            Quantization::Binary => {
                let bq = BinaryQuantizer::train(&full, self.dim, self.metric);
                let (bits, factors) = encode_binary(&bq, &full);
                (Vec::new(), bits, Some(Box::new(bq)), factors)
            }
            Quantization::None => (Vec::new(), Vec::new(), None, Vec::new()),
        };
        SealedSegment {
            dim: self.dim,
            metric: self.metric,
            graph: self.graph.clone(),
            quant: self.quant.clone(),
            codes: codes.into(),
            bits: bits.into(),
            binary,
            corr: corr.into(),
            full: FullVectors::new(full, quantization != Quantization::None),
            ids: self.ids.clone(),
            label_offsets: self.label_offsets.clone(),
            label_bytes: self.label_bytes.clone(),
        }
    }

    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    pub fn quantization(&self) -> Quantization {
        if self.binary.is_some() {
            Quantization::Binary
        } else if self.codes.is_empty() && !self.ids.is_empty() {
            Quantization::None
        } else {
            Quantization::Int8
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

    /// Sets each row's label to `label_of(global id)`. If any row has none,
    /// the segment keeps no labels at all.
    pub fn set_labels<L: AsRef<[u8]>>(&mut self, label_of: impl Fn(u32) -> Option<L>) {
        let mut offsets = Vec::with_capacity(self.len() + 1);
        let mut bytes = Vec::new();
        offsets.push(0u32);
        for &g in self.ids.iter() {
            let Some(l) = label_of(g) else {
                self.label_offsets = Buf::empty();
                self.label_bytes = Buf::empty();
                return;
            };
            bytes.extend_from_slice(l.as_ref());
            offsets.push(u32::try_from(bytes.len()).expect("labels exceed 4 GiB"));
        }
        self.label_offsets = offsets.into();
        self.label_bytes = bytes.into();
    }

    /// [`Self::set_labels`], by value.
    pub fn with_labels<L: AsRef<[u8]>>(mut self, label_of: impl Fn(u32) -> Option<L>) -> Self {
        self.set_labels(label_of);
        self
    }

    /// The label of a local id, if the segment has labels.
    #[inline]
    pub fn label(&self, local: u32) -> Option<&[u8]> {
        let i = local as usize;
        let start = *self.label_offsets.get(i)? as usize;
        let end = *self.label_offsets.get(i + 1)? as usize;
        self.label_bytes.get(start..end)
    }

    /// Whether every row has a label.
    pub fn has_labels(&self) -> bool {
        !self.label_offsets.is_empty()
    }

    /// Whether the arrays are borrowed from shared memory (a mapped file)
    /// rather than owned by this process.
    pub fn is_shared(&self) -> bool {
        self.ids.is_shared()
    }

    /// Bytes the graph walk touches: graph, codes, corrections and ids (and
    /// the full vectors when they are what's walked). Re-ranking reads a few
    /// full vectors per search on top of this.
    pub fn memory_bytes(&self) -> usize {
        let walked = match self.quantization() {
            Quantization::Int8 => self.codes.len(),
            Quantization::Binary => 8 * self.bits.len(),
            Quantization::None => self.full.bytes().len(),
        };
        self.graph.memory_bytes() + walked + 4 * (self.corr.len() + self.ids.len() + self.dim)
    }

    #[inline]
    fn code(&self, local: u32) -> &[i8] {
        let i = local as usize;
        &self.codes[i * self.dim..(i + 1) * self.dim]
    }

    /// The full-precision vector of a local id, prepared.
    #[inline]
    pub fn vector(&self, local: u32) -> Row<'_> {
        self.full.row(local as usize, self.dim)
    }

    /// Every full-precision vector, row `i` for local id `i`.
    pub fn vectors(&self) -> &FullVectors {
        &self.full
    }

    /// A row of a [`Quantization::None`] segment, whose vectors are f32.
    #[inline]
    fn f32_vector(&self, local: u32) -> &[f32] {
        match self.vector(local) {
            Row::F32(v) => v,
            Row::F16(_) => unreachable!("unquantized segments keep f32 vectors"),
        }
    }

    /// Appends up to `p.n` approximate nearest neighbours of `query` to `out`
    /// as (global id, distance). Distances are int8 approximations, or exact
    /// for [`Quantization::None`]. `query` must be prepared.
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
        if self.quantization() == Quantization::None {
            let dist = |local: u32| self.metric.distance(k, query, self.f32_vector(local));
            let prefetch = |local: u32| crate::distance::prefetch(self.f32_vector(local));
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
        if let Some(bq) = &self.binary {
            bq.prepare_query(k, query, &mut s.bquery);
            let words = bq.words();
            let bquery = &s.bquery;
            let bits = |local: u32| {
                let i = local as usize;
                &self.bits[i * words..(i + 1) * words]
            };
            let factors = |local: u32| {
                let i = local as usize * rabitq::FACTORS;
                &self.corr[i..i + rabitq::FACTORS]
            };
            let dist = |local: u32| bq.distance(k, bquery, bits(local), factors(local));
            let prefetch = |local: u32| {
                crate::distance::prefetch(bits(local));
                crate::distance::prefetch(factors(local));
            };
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

    /// The arrays, borrowed, for writing the segment out.
    pub fn arrays(&self) -> SegmentArrays<'_> {
        let (levels, level0, upper_off, upper) = self.graph.arrays();
        SegmentArrays {
            dim: self.dim,
            metric: self.metric,
            m: self.graph.m() as u32,
            entry: self.graph.entry(),
            max_level: self.graph.max_level(),
            levels,
            level0,
            upper_off,
            upper,
            quant_offset: self.quant.offset(),
            quant_scale: self.quant.scale(),
            codes: &self.codes,
            bits: &self.bits,
            binary_state: self.binary.as_ref().map_or(&[], |b| b.state()),
            corr: &self.corr,
            full: &self.full,
            ids: &self.ids,
            label_offsets: &self.label_offsets,
            label_bytes: &self.label_bytes,
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
            bits: self.bits.clone(),
            binary_state: self
                .binary
                .as_ref()
                .map_or_else(Vec::new, |b| b.state().to_vec()),
            corr: self.corr.clone(),
            full: self.full.clone(),
            ids: self.ids.clone(),
            label_offsets: self.label_offsets.clone(),
            label_bytes: self.label_bytes.clone(),
        }
    }

    /// Rebuilds a segment from its arrays, validating their shapes.
    pub fn from_parts(p: SegmentParts) -> Result<Self, String> {
        let graph = Graph::from_parts(p.graph)?;
        let n = graph.len();
        if p.dim == 0 || p.quant_offset.len() != p.dim {
            return Err("quantizer dimension mismatch".into());
        }
        let binary = if p.binary_state.is_empty() {
            None
        } else {
            Some(Box::new(
                BinaryQuantizer::from_state(p.metric, p.dim, p.binary_state)
                    .ok_or("binary quantizer state has the wrong length")?,
            ))
        };
        let int8 = binary.is_none()
            && p.codes.len() == n * p.dim
            && p.bits.is_empty()
            && p.corr.len() == n;
        let bin = binary.as_ref().is_some_and(|b| {
            p.codes.is_empty()
                && p.bits.len() == n * b.words()
                && p.corr.len() == n * rabitq::FACTORS
        });
        let none = binary.is_none() && p.codes.is_empty() && p.bits.is_empty() && p.corr.is_empty();
        if !(int8 || bin || none) || p.full.len() != n * p.dim || p.ids.len() != n {
            return Err("segment arrays disagree on length".into());
        }
        if none && !matches!(p.full, FullVectors::F32(_)) {
            return Err("an unquantized segment needs f32 vectors".into());
        }
        if !p.ids.windows(2).all(|w| w[0] < w[1]) {
            return Err("segment ids are not strictly ascending".into());
        }
        if !(p.quant_scale.is_finite() && p.quant_scale > 0.0) {
            return Err("invalid quantizer scale".into());
        }
        if !p.label_offsets.is_empty() {
            let o = &p.label_offsets;
            if o.len() != n + 1
                || o[0] != 0
                || !o.windows(2).all(|w| w[0] <= w[1])
                || o[n] as usize != p.label_bytes.len()
            {
                return Err("segment labels are malformed".into());
            }
        } else if !p.label_bytes.is_empty() {
            return Err("segment label bytes without offsets".into());
        }
        Ok(SealedSegment {
            dim: p.dim,
            metric: p.metric,
            graph,
            quant: ScalarQuantizer::from_parts(p.metric, p.quant_offset, p.quant_scale),
            codes: p.codes,
            bits: p.bits,
            binary,
            corr: p.corr,
            full: p.full,
            ids: p.ids,
            label_offsets: p.label_offsets,
            label_bytes: p.label_bytes,
        })
    }
}

/// Rows a [`SegmentBuilder`] collects before it trains its quantizer and
/// starts building the graph.
pub const STREAM_TRAIN_ROWS: usize = 50_000;

/// Builds a sealed segment while its rows arrive, so the graph is mostly
/// built by the time the last row is in.
///
/// Rows wait until [`STREAM_TRAIN_ROWS`] have arrived; the quantizer is
/// trained on those, and from then on every [`SegmentBuilder::add`]
/// encodes its rows and inserts them into the graph with all of rayon's
/// threads. Training on the first rows assumes they are representative,
/// as a shuffled load's are; the quantizer's clamping keeps a later shift
/// from doing more than cost some precision.
pub struct SegmentBuilder {
    dim: usize,
    metric: Metric,
    quantization: Quantization,
    graph: hnsw::GraphBuilder,
    train_rows: usize,
    quant: Option<ScalarQuantizer>,
    ids: Vec<u32>,
    /// Rows not yet in the graph: waiting for the quantizer.
    waiting: Vec<f32>,
    codes: Vec<i8>,
    corr: Vec<f32>,
    /// Full vectors: f32 for [`Quantization::None`], f16 otherwise.
    full32: Vec<f32>,
    full16: Vec<f16>,
}

impl SegmentBuilder {
    pub fn new(
        dim: usize,
        metric: Metric,
        params: &HnswParams,
        quantization: Quantization,
    ) -> Self {
        assert!(dim > 0, "dim must be positive");
        SegmentBuilder {
            dim,
            metric,
            quantization,
            graph: hnsw::GraphBuilder::new(params),
            train_rows: STREAM_TRAIN_ROWS,
            quant: None,
            ids: Vec::new(),
            waiting: Vec::new(),
            codes: Vec::new(),
            corr: Vec::new(),
            full32: Vec::new(),
            full16: Vec::new(),
        }
    }

    /// Rows to collect before training (default [`STREAM_TRAIN_ROWS`]).
    pub fn train_after(mut self, rows: usize) -> Self {
        self.train_rows = rows.max(1);
        self
    }

    /// Rows added so far.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Global ids added so far, ascending.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Adds rows: global ids, each above every id added before, and their
    /// prepared vectors. Returns once they are in the graph (or waiting
    /// for the quantizer).
    pub fn add(&mut self, ids: &[u32], vectors: &[f32]) {
        let dim = self.dim;
        assert_eq!(ids.len() * dim, vectors.len(), "one vector per id");
        let ascending = ids.windows(2).all(|w| w[0] < w[1])
            && ids
                .first()
                .is_none_or(|&f| self.ids.last().is_none_or(|&l| l < f));
        assert!(ascending, "ids must ascend across adds");
        self.ids.extend_from_slice(ids);
        if self.quantization == Quantization::None {
            self.full32.extend_from_slice(vectors);
            let (dist, prefetch) = full_distance(self.metric, &self.full32, dim);
            self.graph.grow(self.ids.len(), dist, prefetch);
            return;
        }
        let at = self.full16.len();
        self.full16.resize(at + vectors.len(), f16::ZERO);
        self.full16[at..].convert_from_f32_slice(vectors);
        self.waiting.extend_from_slice(vectors);
        if self.quant.is_some() || self.waiting.len() >= self.train_rows * dim {
            self.insert_waiting();
        }
    }

    /// Encodes the waiting rows (training the quantizer first, if need be)
    /// and inserts them into the graph.
    fn insert_waiting(&mut self) {
        let dim = self.dim;
        let quant = match &self.quant {
            Some(q) => q,
            None => self
                .quant
                .insert(ScalarQuantizer::train(&self.waiting, dim, self.metric)),
        };
        let (codes, corr) = encode_int8(quant, &self.waiting, dim);
        self.codes.extend(codes);
        self.corr.extend(corr);
        self.waiting.clear();
        let (dist, prefetch) = code_distance(quant, &self.codes, &self.corr, dim);
        self.graph.grow(self.ids.len(), dist, prefetch);
    }

    /// Inserts whatever is still waiting and returns the segment.
    pub fn finish(mut self) -> SealedSegment {
        let dim = self.dim;
        if self.quantization != Quantization::None {
            self.insert_waiting();
        }
        let quant = self
            .quant
            .take()
            .unwrap_or_else(|| ScalarQuantizer::train(&self.full32, dim, self.metric));
        let full = if self.quantization == Quantization::None {
            FullVectors::F32(self.full32.into())
        } else {
            FullVectors::F16(self.full16.into())
        };
        let (codes, bits, binary, corr) = if self.quantization == Quantization::Binary {
            // Trained on a sample: the centroid is all it learns.
            let sample = self.ids.len().min(self.train_rows);
            let mut rows = Vec::with_capacity(sample * dim);
            for local in 0..sample {
                full.row(local, dim).extend_into(&mut rows);
            }
            let bq = BinaryQuantizer::train(&rows, dim, self.metric);
            let (bits, factors) = encode_binary_rows(&bq, &full, self.ids.len());
            (Vec::new(), bits, Some(Box::new(bq)), factors)
        } else {
            (self.codes, Vec::new(), None, self.corr)
        };
        SealedSegment {
            dim,
            metric: self.metric,
            graph: self.graph.finish(),
            quant,
            codes: codes.into(),
            bits: bits.into(),
            binary,
            corr: corr.into(),
            full,
            ids: self.ids.into(),
            label_offsets: Buf::empty(),
            label_bytes: Buf::empty(),
        }
    }
}

/// [`encode_binary`] for the first `n` rows of `full`, widened a block at a
/// time.
fn encode_binary_rows(bq: &BinaryQuantizer, full: &FullVectors, n: usize) -> (Vec<u64>, Vec<f32>) {
    const BLOCK: usize = 4096;
    let (dim, words) = (bq.dim(), bq.words());
    let mut bits = vec![0u64; n * words];
    let mut factors = vec![0f32; n * rabitq::FACTORS];
    let k = kernels();
    bits.par_chunks_mut(BLOCK * words)
        .zip(factors.par_chunks_mut(BLOCK * rabitq::FACTORS))
        .enumerate()
        .for_each(|(block, (b, f))| {
            let mut row = Vec::with_capacity(dim);
            for (i, (b, f)) in b
                .chunks_exact_mut(words)
                .zip(f.as_chunks_mut::<{ rabitq::FACTORS }>().0)
                .enumerate()
            {
                row.clear();
                full.row(block * BLOCK + i, dim).extend_into(&mut row);
                bq.encode(k, &row, b, f);
            }
        });
    (bits, factors)
}

/// int8 codes and corrections for row-major `rows`.
fn encode_int8(quant: &ScalarQuantizer, rows: &[f32], dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut codes = vec![0i8; rows.len()];
    let corr = rows
        .par_chunks_exact(dim)
        .zip(codes.par_chunks_exact_mut(dim))
        .map(|(v, c)| quant.encode(v, c))
        .collect();
    (codes, corr)
}

/// 1-bit codes and their factors for row-major `rows`.
fn encode_binary(bq: &BinaryQuantizer, rows: &[f32]) -> (Vec<u64>, Vec<f32>) {
    let (dim, words) = (bq.dim(), bq.words());
    let n = rows.len() / dim;
    let mut bits = vec![0u64; n * words];
    let mut factors = vec![0f32; n * rabitq::FACTORS];
    let k = kernels();
    rows.par_chunks_exact(dim)
        .zip(bits.par_chunks_exact_mut(words))
        .zip(factors.par_chunks_exact_mut(rabitq::FACTORS))
        .for_each(|((v, b), f)| bq.encode(k, v, b, f));
    (bits, factors)
}

/// Distance between two rows' int8 codes (in the metric's units), and a
/// prefetch of a row's code, for building a graph.
fn code_distance<'a>(
    quant: &'a ScalarQuantizer,
    codes: &'a [i8],
    corr: &'a [f32],
    dim: usize,
) -> (
    impl Fn(u32, u32) -> f32 + Sync + 'a,
    impl Fn(u32) + Sync + 'a,
) {
    let k = kernels();
    let code = move |i: u32| &codes[i as usize * dim..(i as usize + 1) * dim];
    (
        move |a: u32, b: u32| {
            quant.distance(k, code(a), corr[a as usize], code(b), corr[b as usize])
        },
        move |i: u32| crate::distance::prefetch(code(i)),
    )
}

/// Distance between two rows' full vectors, and a prefetch of a row.
fn full_distance(
    metric: Metric,
    rows: &[f32],
    dim: usize,
) -> (
    impl Fn(u32, u32) -> f32 + Sync + '_,
    impl Fn(u32) + Sync + '_,
) {
    let k = kernels();
    let row = move |i: u32| &rows[i as usize * dim..(i as usize + 1) * dim];
    (
        move |a: u32, b: u32| metric.distance(k, row(a), row(b)),
        move |i: u32| crate::distance::prefetch(row(i)),
    )
}

/// A segment's arrays, borrowed (see [`SealedSegment::arrays`]).
pub struct SegmentArrays<'a> {
    pub dim: usize,
    pub metric: Metric,
    pub m: u32,
    pub entry: u32,
    pub max_level: u8,
    pub levels: &'a [u8],
    pub level0: &'a [u32],
    pub upper_off: &'a [u32],
    pub upper: &'a [u32],
    pub quant_offset: &'a [f32],
    pub quant_scale: f32,
    pub codes: &'a [i8],
    pub bits: &'a [u64],
    pub binary_state: &'a [f32],
    pub corr: &'a [f32],
    pub full: &'a FullVectors,
    pub ids: &'a [u32],
    pub label_offsets: &'a [u32],
    pub label_bytes: &'a [u8],
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
    fn streamed_segments_find_true_neighbours() {
        let (dim, n) = (64, 4000);
        let data = random(n, dim, 31);
        let ids: Vec<u32> = (0..n as u32).map(|i| i * 2 + 1).collect();
        let q = random(1, dim, 32);
        let truth = brute_force::search(&data, dim, Metric::L2, &q, 10);
        for (quantization, wanted) in [
            (Quantization::Int8, 40),
            (Quantization::None, 40),
            (Quantization::Binary, 100),
        ] {
            let mut b = SegmentBuilder::new(dim, Metric::L2, &HnswParams::default(), quantization)
                .train_after(1000);
            for (i, rows) in data.chunks(500 * dim).enumerate() {
                b.add(&ids[i * 500..i * 500 + rows.len() / dim], rows);
            }
            let seg = b.finish();
            assert_eq!(seg.quantization(), quantization);
            assert_eq!(seg.ids(), &ids[..]);
            assert_eq!(SealedSegment::from_parts(seg.to_parts()).unwrap(), seg);
            let mut out = Vec::new();
            let mut s = SegmentScratch::default();
            seg.candidates(&q, &params(wanted), None, None, &mut s, &mut out);
            let found: Vec<u32> = out.iter().map(|n| n.id).collect();
            let hits = truth
                .iter()
                .filter(|t| found.contains(&ids[t.id as usize]))
                .count();
            assert!(hits >= 9, "{quantization:?}: {hits}/10 among {wanted}");
        }
        // Too few rows to train before the end: trained at the end.
        let mut b =
            SegmentBuilder::new(dim, Metric::L2, &HnswParams::default(), Quantization::Int8);
        b.add(&ids[..300], &data[..300 * dim]);
        assert_eq!(b.finish().len(), 300);
        let empty =
            SegmentBuilder::new(dim, Metric::L2, &HnswParams::default(), Quantization::Int8);
        assert!(empty.finish().is_empty());
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
        let grown = base.extend(&new_ids, &new_data, &HnswParams::default());
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
        let mut ids = bad.ids.into_vec();
        ids.swap(0, 1);
        bad.ids = ids.into();
        assert!(SealedSegment::from_parts(bad).is_err());
        let mut short = seg.to_parts();
        short.full = FullVectors::F16(vec![f16::ZERO; short.full.len() - 1].into());
        assert!(SealedSegment::from_parts(short).is_err());
    }

    #[test]
    fn int8_segments_keep_half_precision_vectors() {
        let dim = 16;
        let data = random(200, dim, 6);
        let ids: Vec<u32> = (0..200).collect();
        let p = HnswParams::default();
        let seg = SealedSegment::build(&ids, &data, dim, Metric::L2, &p);
        assert!(matches!(seg.vectors(), FullVectors::F16(_)));
        let mut row = Vec::new();
        seg.vector(7).extend_into(&mut row);
        for (a, b) in row.iter().zip(&data[7 * dim..8 * dim]) {
            assert!((a - b).abs() <= b.abs() / 1024.0, "{a} vs {b}");
        }
        // Growing a segment widens and narrows its rows again: no drift.
        let grown = seg.extend(&[500], &data[..dim], &p);
        assert_eq!(grown.vector(7), seg.vector(7));

        let binary =
            SealedSegment::build_with(&ids, &data, dim, Metric::L2, &p, Quantization::Binary);
        assert_eq!(binary.quantization(), Quantization::Binary);
        assert_eq!(
            SealedSegment::from_parts(binary.to_parts()).unwrap(),
            binary
        );
        let rq = seg.requantize(Quantization::Binary);
        assert_eq!(rq.quantization(), Quantization::Binary);
        assert_eq!((rq.graph(), rq.vectors()), (seg.graph(), seg.vectors()));
        assert_eq!(
            rq.requantize(Quantization::Int8).quantization(),
            Quantization::Int8
        );

        let exact = SealedSegment::build_with(&ids, &data, dim, Metric::L2, &p, Quantization::None);
        assert!(matches!(exact.vectors(), FullVectors::F32(_)));
        let mut parts = exact.to_parts();
        parts.full = seg.to_parts().full;
        assert!(SealedSegment::from_parts(parts).is_err());
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
