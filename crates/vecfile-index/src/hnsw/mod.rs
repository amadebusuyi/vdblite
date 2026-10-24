//! Hierarchical Navigable Small World graphs (Malkov & Yashunin, 2016).
//!
//! A built [`Graph`] is stored as flat `u32` arrays so it can be written to
//! and read from a file without pointer fix-ups:
//!
//! - `level0`: one fixed-width slot per node, `[count, n1 .. n_2M]`.
//! - `upper_off`: per node, the start of its upper-layer slots in `upper`
//!   (`u32::MAX` for the ~(1 - 1/M) of nodes that live only on layer 0).
//! - `upper`: for a node of level L, L slots of `[count, n1 .. n_M]`.
//!
//! The graph knows nothing about vectors. Search and build take a distance
//! closure, so the same graph serves f32, int8 or any other representation.

mod build;

use std::cmp::Reverse;
use std::collections::BinaryHeap;

pub use build::build;

use crate::distance::{Metric, kernels};
use crate::topk::Neighbor;
use crate::visited::Visited;

/// Tuning knobs for building and searching an HNSW graph.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HnswParams {
    /// Neighbours per node on upper layers; layer 0 allows `2 * m`.
    pub m: usize,
    /// Beam width while inserting. Higher builds slower, searches better.
    pub ef_construction: usize,
    /// Default beam width while searching.
    pub ef_search: usize,
    /// `false` builds a single-layer graph (every node on layer 0).
    pub layered: bool,
    /// Seed for level assignment.
    pub seed: u64,
}

impl Default for HnswParams {
    fn default() -> Self {
        HnswParams {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            layered: true,
            seed: 0x5EED_CAFE,
        }
    }
}

/// An immutable HNSW graph over nodes `0..len`.
#[derive(Clone, Debug, PartialEq)]
pub struct Graph {
    m: u32,
    entry: u32,
    max_level: u8,
    levels: Vec<u8>,
    level0: Vec<u32>,
    upper_off: Vec<u32>,
    upper: Vec<u32>,
}

/// The raw arrays of a [`Graph`], for serialization.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphParts {
    pub m: u32,
    pub entry: u32,
    pub max_level: u8,
    pub levels: Vec<u8>,
    pub level0: Vec<u32>,
    pub upper_off: Vec<u32>,
    pub upper: Vec<u32>,
}

/// Reusable buffers for [`Graph::search`]; keep one per thread.
#[derive(Clone, Debug, Default)]
pub struct SearchScratch {
    visited: Visited,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
}

impl SearchScratch {
    pub fn new() -> Self {
        Self::default()
    }
}

impl Graph {
    pub fn len(&self) -> usize {
        self.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.levels.is_empty()
    }

    pub fn m(&self) -> usize {
        self.m as usize
    }

    pub fn entry(&self) -> u32 {
        self.entry
    }

    pub fn max_level(&self) -> u8 {
        self.max_level
    }

    pub fn level(&self, node: u32) -> u8 {
        self.levels[node as usize]
    }

    /// Bytes held by the graph arrays.
    pub fn memory_bytes(&self) -> usize {
        self.levels.len() + 4 * (self.level0.len() + self.upper_off.len() + self.upper.len())
    }

    /// Neighbours of `node` on `level`.
    #[inline]
    pub fn neighbors(&self, node: u32, level: u8) -> &[u32] {
        if level == 0 {
            let w = 2 * self.m as usize + 1;
            let slot = &self.level0[node as usize * w..(node as usize + 1) * w];
            &slot[1..1 + slot[0] as usize]
        } else {
            let w = self.m as usize + 1;
            let base = self.upper_off[node as usize] as usize + (level as usize - 1) * w;
            let slot = &self.upper[base..base + w];
            &slot[1..1 + slot[0] as usize]
        }
    }

    /// Beam search for the nodes closest to a query.
    ///
    /// `dist(node)` gives the query's distance to a node. Nodes for which
    /// `accept(node)` is false still route the search but are never returned,
    /// which is how deletes and filters work. Writes up to `ef` results into
    /// `out`, closest first. Allocates nothing once `scratch` has warmed up.
    pub fn search<D, A>(
        &self,
        mut dist: D,
        accept: A,
        ef: usize,
        scratch: &mut SearchScratch,
        out: &mut Vec<Neighbor>,
    ) where
        D: FnMut(u32) -> f32,
        A: Fn(u32) -> bool,
    {
        out.clear();
        if self.is_empty() || ef == 0 {
            return;
        }
        let mut ep = self.entry;
        let mut ep_d = dist(ep);
        for level in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                for &n in self.neighbors(ep, level) {
                    let d = dist(n);
                    if d < ep_d {
                        ep = n;
                        ep_d = d;
                        changed = true;
                    }
                }
            }
        }

        let s = scratch;
        s.visited.reset(self.len());
        s.candidates.clear();
        s.results.clear();
        s.visited.insert(ep);
        s.candidates.push(Reverse(Neighbor::new(ep, ep_d)));
        if accept(ep) {
            s.results.push(Neighbor::new(ep, ep_d));
        }
        while let Some(Reverse(c)) = s.candidates.pop() {
            if s.results.len() >= ef && c.dist > s.results.peek().map_or(f32::INFINITY, |r| r.dist)
            {
                break;
            }
            for &n in self.neighbors(c.id, 0) {
                if !s.visited.insert(n) {
                    continue;
                }
                let d = dist(n);
                let worst = s.results.peek().map_or(f32::INFINITY, |r| r.dist);
                if s.results.len() < ef || d < worst {
                    s.candidates.push(Reverse(Neighbor::new(n, d)));
                    if accept(n) {
                        s.results.push(Neighbor::new(n, d));
                        if s.results.len() > ef {
                            s.results.pop();
                        }
                    }
                }
            }
        }
        out.extend(s.results.drain());
        out.sort_unstable();
    }

    pub fn into_parts(self) -> GraphParts {
        GraphParts {
            m: self.m,
            entry: self.entry,
            max_level: self.max_level,
            levels: self.levels,
            level0: self.level0,
            upper_off: self.upper_off,
            upper: self.upper,
        }
    }

    pub fn to_parts(&self) -> GraphParts {
        self.clone().into_parts()
    }

    /// Rebuilds a graph from its arrays, checking every invariant search
    /// relies on, so a corrupted file is an error rather than a panic.
    pub fn from_parts(p: GraphParts) -> Result<Graph, String> {
        let n = p.levels.len();
        let m = p.m as usize;
        if m == 0 && n > 0 {
            return Err("m is zero".into());
        }
        if n > u32::MAX as usize {
            return Err("too many nodes".into());
        }
        let w0 = 2 * m + 1;
        if p.level0.len() != n * w0 {
            return Err(format!(
                "level0 has {} words, expected {}",
                p.level0.len(),
                n * w0
            ));
        }
        if p.upper_off.len() != n {
            return Err("upper_off length differs from node count".into());
        }
        if n == 0 {
            return Ok(Graph {
                m: p.m,
                entry: 0,
                max_level: 0,
                ..Graph::empty(p.m)
            });
        }
        if p.entry as usize >= n {
            return Err("entry out of range".into());
        }
        if p.levels[p.entry as usize] != p.max_level {
            return Err("entry is not on the top level".into());
        }
        // A neighbour listed on `level` must itself live on that level, or
        // search would look up upper slots the neighbour doesn't have.
        let levels = &p.levels;
        let check_slot = |slot: &[u32], cap: usize, level: u8| -> Result<(), String> {
            let c = slot[0] as usize;
            if c > cap {
                return Err(format!("neighbour count {c} exceeds {cap}"));
            }
            for &x in &slot[1..1 + c] {
                if x as usize >= n {
                    return Err("neighbour id out of range".into());
                }
                if levels[x as usize] < level {
                    return Err(format!(
                        "neighbour {x} listed on level {level} lives below it"
                    ));
                }
            }
            Ok(())
        };
        for slot in p.level0.chunks_exact(w0) {
            check_slot(slot, 2 * m, 0)?;
        }
        let wu = m + 1;
        for (node, (&lvl, &off)) in p.levels.iter().zip(&p.upper_off).enumerate() {
            if lvl > p.max_level {
                return Err(format!("node {node} level {lvl} above max {}", p.max_level));
            }
            if lvl == 0 {
                continue;
            }
            let start = off as usize;
            let end = start + lvl as usize * wu;
            if off == u32::MAX || end > p.upper.len() {
                return Err(format!("node {node} upper slots out of range"));
            }
            for (i, slot) in p.upper[start..end].chunks_exact(wu).enumerate() {
                check_slot(slot, m, i as u8 + 1)?;
            }
        }
        Ok(Graph {
            m: p.m,
            entry: p.entry,
            max_level: p.max_level,
            levels: p.levels,
            level0: p.level0,
            upper_off: p.upper_off,
            upper: p.upper,
        })
    }

    fn empty(m: u32) -> Graph {
        Graph {
            m,
            entry: 0,
            max_level: 0,
            levels: Vec::new(),
            level0: Vec::new(),
            upper_off: Vec::new(),
            upper: Vec::new(),
        }
    }
}

/// An HNSW index over full-precision vectors held in memory.
///
/// This is the reference index: segments use int8 codes for the walk and
/// full vectors only for re-ranking (see [`crate::segment`]).
#[derive(Clone, Debug)]
pub struct HnswIndex {
    pub graph: Graph,
    pub vectors: Vec<f32>,
    pub dim: usize,
    pub metric: Metric,
}

impl HnswIndex {
    /// Builds an index over row-major `vectors`, in parallel.
    pub fn build(vectors: Vec<f32>, dim: usize, metric: Metric, params: &HnswParams) -> Self {
        assert!(
            dim > 0 && vectors.len().is_multiple_of(dim),
            "vectors must be n x dim"
        );
        let n = vectors.len() / dim;
        let k = kernels();
        let graph = build(n, params, |a, b| {
            let (a, b) = (a as usize, b as usize);
            metric.distance(
                k,
                &vectors[a * dim..(a + 1) * dim],
                &vectors[b * dim..(b + 1) * dim],
            )
        });
        HnswIndex {
            graph,
            vectors,
            dim,
            metric,
        }
    }

    /// The `k` approximate nearest neighbours of `query`, closest first.
    pub fn search(
        &self,
        query: &[f32],
        k: usize,
        ef: usize,
        scratch: &mut SearchScratch,
    ) -> Vec<Neighbor> {
        assert_eq!(query.len(), self.dim, "query length must equal dim");
        let kern = kernels();
        let mut out = Vec::with_capacity(ef.max(k));
        self.graph.search(
            |i| {
                let i = i as usize;
                self.metric
                    .distance(kern, query, &self.vectors[i * self.dim..(i + 1) * self.dim])
            },
            |_| true,
            ef.max(k),
            scratch,
            &mut out,
        );
        out.truncate(k);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute_force;
    use crate::recall::recall_at_k;
    use crate::rng::SplitMix64;

    pub(crate) fn random_vectors(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut r = SplitMix64::new(seed);
        (0..n * dim).map(|_| r.next_f64_open() as f32).collect()
    }

    fn recall(index: &HnswIndex, queries: &[f32], k: usize, ef: usize) -> f64 {
        let mut s = SearchScratch::new();
        let found: Vec<Vec<u32>> = queries
            .chunks_exact(index.dim)
            .map(|q| {
                index
                    .search(q, k, ef, &mut s)
                    .iter()
                    .map(|n| n.id)
                    .collect()
            })
            .collect();
        let truth: Vec<Vec<u32>> =
            brute_force::search_batch(&index.vectors, index.dim, index.metric, queries, k)
                .iter()
                .map(|r| r.iter().map(|n| n.id).collect())
                .collect();
        recall_at_k(&found, &truth, k)
    }

    #[test]
    fn high_recall_on_random_data() {
        let dim = 16;
        let index = HnswIndex::build(
            random_vectors(5000, dim, 1),
            dim,
            Metric::L2,
            &HnswParams::default(),
        );
        let r = recall(&index, &random_vectors(100, dim, 2), 10, 64);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn flat_graph_also_works() {
        let dim = 16;
        let params = HnswParams {
            layered: false,
            ..HnswParams::default()
        };
        let index = HnswIndex::build(random_vectors(3000, dim, 3), dim, Metric::L2, &params);
        assert_eq!(index.graph.max_level(), 0);
        let r = recall(&index, &random_vectors(100, dim, 4), 10, 64);
        assert!(r > 0.95, "recall {r}");
    }

    #[test]
    fn tiny_graphs() {
        for n in [0usize, 1, 2, 5] {
            let index = HnswIndex::build(
                random_vectors(n, 4, 5),
                4,
                Metric::L2,
                &HnswParams::default(),
            );
            let hits = index.search(&[0.5; 4], 10, 16, &mut SearchScratch::new());
            assert_eq!(hits.len(), n);
        }
    }

    #[test]
    fn rejected_nodes_route_but_are_not_returned() {
        let dim = 8;
        let index = HnswIndex::build(
            random_vectors(2000, dim, 6),
            dim,
            Metric::L2,
            &HnswParams::default(),
        );
        let q = &index.vectors[..dim];
        let mut out = Vec::new();
        index.graph.search(
            |i| {
                Metric::L2.distance(
                    kernels(),
                    q,
                    &index.vectors[i as usize * dim..(i as usize + 1) * dim],
                )
            },
            |i| i % 2 == 1,
            32,
            &mut SearchScratch::new(),
            &mut out,
        );
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|n| n.id % 2 == 1));
    }

    #[test]
    fn parts_round_trip_and_validate() {
        let index = HnswIndex::build(
            random_vectors(500, 4, 7),
            4,
            Metric::L2,
            &HnswParams::default(),
        );
        let parts = index.graph.to_parts();
        assert_eq!(Graph::from_parts(parts.clone()).unwrap(), index.graph);

        let mut bad = parts.clone();
        bad.level0[1] = 10_000;
        assert!(Graph::from_parts(bad).is_err());
        let mut bad = parts.clone();
        bad.level0[0] = 99;
        assert!(Graph::from_parts(bad).is_err());
        let mut bad = parts.clone();
        bad.entry = 9999;
        assert!(Graph::from_parts(bad).is_err());

        // A layer-0-only node listed as an upper-layer neighbour.
        let mut bad = parts.clone();
        let low = bad.levels.iter().position(|&l| l == 0).unwrap() as u32;
        let high = bad.levels.iter().position(|&l| l > 0).unwrap();
        let off = bad.upper_off[high] as usize;
        bad.upper[off] = 1;
        bad.upper[off + 1] = low;
        assert!(Graph::from_parts(bad).is_err());
    }
}
