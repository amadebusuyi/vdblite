//! Parallel HNSW construction.
//!
//! Nodes are inserted concurrently with rayon. Each node's adjacency lists sit
//! behind their own lock, and a thread holds at most one node lock at a time,
//! so inserts cannot deadlock. A node that raises the graph's top level holds
//! `grow` for its whole insert, as in hnswlib, so the entry point only ever
//! moves to a fully linked node.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use parking_lot::{Mutex, RwLock};
use rayon::prelude::*;

use super::{Graph, HnswParams};
use crate::rng::SplitMix64;
use crate::topk::Neighbor;
use crate::visited::Visited;

/// Nodes inserted one at a time before going parallel, so early inserts see a
/// connected graph rather than racing on a handful of nodes.
const SERIAL_PREFIX: usize = 1024;

/// Builds a graph over nodes `0..n`, where `dist(a, b)` is the distance
/// between two nodes.
pub fn build<F>(n: usize, params: &HnswParams, dist: F) -> Graph
where
    F: Fn(u32, u32) -> f32 + Sync,
{
    assert!(params.m >= 2, "m must be at least 2");
    assert!(n <= u32::MAX as usize, "too many nodes");
    let levels = assign_levels(n, params);
    if n == 0 {
        return Graph::empty(params.m as u32);
    }
    let b = Builder {
        m: params.m,
        ef: params.ef_construction.max(params.m),
        links: levels
            .iter()
            .map(|&l| Mutex::new(vec![Vec::new(); l as usize + 1]))
            .collect(),
        entry: RwLock::new((0, levels[0])),
        grow: Mutex::new(()),
        levels,
        dist,
    };

    let prefix = n.min(SERIAL_PREFIX);
    let mut scratch = Scratch::new(n);
    for q in 1..prefix {
        b.insert(q as u32, &mut scratch);
    }
    (prefix..n)
        .into_par_iter()
        .for_each_init(|| Scratch::new(n), |s, q| b.insert(q as u32, s));
    b.freeze()
}

/// Grows an existing graph instead of building from scratch.
///
/// The result has nodes `0..n`. Node `i` of `base` becomes node
/// `remap[i]`, keeping its level and links; the nodes not in `remap` are
/// inserted the same way [`build`] inserts them. `dist(a, b)` works on the
/// new numbering. The cost is that of inserting the new nodes only, so
/// growing a large graph by a little is cheap.
pub fn extend<F>(base: &Graph, remap: &[u32], n: usize, params: &HnswParams, dist: F) -> Graph
where
    F: Fn(u32, u32) -> f32 + Sync,
{
    assert_eq!(remap.len(), base.len(), "one new position per base node");
    assert!(n <= u32::MAX as usize, "too many nodes");
    if base.is_empty() {
        return build(n, params, dist);
    }
    let mut is_base = vec![false; n];
    for &r in remap {
        assert!(
            (r as usize) < n && !is_base[r as usize],
            "remap must be injective into 0..n"
        );
        is_base[r as usize] = true;
    }
    // New nodes get levels from the same distribution, seeded apart from
    // the base's.
    let fresh_levels = assign_levels(
        n,
        &HnswParams {
            seed: params.seed ^ (base.len() as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15),
            ..*params
        },
    );
    let mut levels = vec![0u8; n];
    let mut links: Vec<Mutex<Vec<Vec<u32>>>> = Vec::with_capacity(n);
    for (node, &fresh) in fresh_levels.iter().enumerate() {
        if !is_base[node] {
            levels[node] = fresh;
        }
    }
    for (old, &new) in remap.iter().enumerate() {
        levels[new as usize] = base.level(old as u32);
    }
    for &l in &levels {
        links.push(Mutex::new(vec![Vec::new(); l as usize + 1]));
    }
    for (old, &new) in remap.iter().enumerate() {
        let mut g = links[new as usize].lock();
        for (level, list) in g.iter_mut().enumerate() {
            list.extend(
                base.neighbors(old as u32, level as u8)
                    .iter()
                    .map(|&x| remap[x as usize]),
            );
        }
    }
    let b = Builder {
        m: params.m,
        ef: params.ef_construction.max(params.m),
        links,
        entry: RwLock::new((remap[base.entry() as usize], base.max_level())),
        grow: Mutex::new(()),
        levels,
        dist,
    };
    let fresh: Vec<u32> = (0..n as u32).filter(|&i| !is_base[i as usize]).collect();
    fresh
        .into_par_iter()
        .for_each_init(|| Scratch::new(n), |s, q| b.insert(q, s));
    b.freeze()
}

fn assign_levels(n: usize, params: &HnswParams) -> Vec<u8> {
    if !params.layered {
        return vec![0; n];
    }
    let ml = 1.0 / (params.m as f64).ln();
    let mut rng = SplitMix64::new(params.seed);
    (0..n)
        .map(|_| (-rng.next_f64_open().ln() * ml).floor().min(16.0) as u8)
        .collect()
}

struct Builder<F> {
    m: usize,
    ef: usize,
    levels: Vec<u8>,
    links: Vec<Mutex<Vec<Vec<u32>>>>,
    entry: RwLock<(u32, u8)>,
    grow: Mutex<()>,
    dist: F,
}

struct Scratch {
    visited: Visited,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
    found: Vec<Neighbor>,
    entries: Vec<Neighbor>,
    buf: Vec<u32>,
}

impl Scratch {
    fn new(n: usize) -> Self {
        Scratch {
            visited: Visited::new(n),
            candidates: BinaryHeap::new(),
            results: BinaryHeap::new(),
            found: Vec::new(),
            entries: Vec::new(),
            buf: Vec::new(),
        }
    }
}

impl<F: Fn(u32, u32) -> f32 + Sync> Builder<F> {
    fn cap(&self, level: usize) -> usize {
        if level == 0 { 2 * self.m } else { self.m }
    }

    fn copy_neighbors(&self, node: u32, level: usize, buf: &mut Vec<u32>) {
        buf.clear();
        let g = self.links[node as usize].lock();
        if let Some(list) = g.get(level) {
            buf.extend_from_slice(list);
        }
    }

    fn insert(&self, q: u32, s: &mut Scratch) {
        let lq = self.levels[q as usize];
        let snapshot = *self.entry.read();
        let grow_guard = (lq > snapshot.1).then(|| self.grow.lock());
        let (mut ep, top) = if grow_guard.is_some() {
            *self.entry.read()
        } else {
            snapshot
        };

        let mut ep_d = (self.dist)(q, ep);
        for level in ((lq as usize + 1)..=top as usize).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                self.copy_neighbors(ep, level, &mut s.buf);
                for &nb in &s.buf {
                    let d = (self.dist)(q, nb);
                    if d < ep_d {
                        ep = nb;
                        ep_d = d;
                        changed = true;
                    }
                }
            }
        }

        s.entries.clear();
        s.entries.push(Neighbor::new(ep, ep_d));
        for level in (0..=lq.min(top) as usize).rev() {
            self.search_layer(q, level, s);
            let selected = self.select(&s.found, self.m);
            self.links[q as usize].lock()[level] = selected.clone();
            let cap = self.cap(level);
            for &nb in &selected {
                let mut g = self.links[nb as usize].lock();
                let list = &mut g[level];
                if list.contains(&q) {
                    continue;
                }
                if list.len() < cap {
                    list.push(q);
                } else {
                    let mut cands: Vec<Neighbor> = list
                        .iter()
                        .chain(std::iter::once(&q))
                        .map(|&x| Neighbor::new(x, (self.dist)(nb, x)))
                        .collect();
                    cands.sort_unstable();
                    *list = self.select(&cands, cap);
                }
            }
            std::mem::swap(&mut s.entries, &mut s.found);
        }

        if grow_guard.is_some() && lq > top {
            *self.entry.write() = (q, lq);
        }
    }

    /// Fills `s.found` with up to `ef` nodes near `q` on `level`, closest first,
    /// starting from `s.entries`.
    fn search_layer(&self, q: u32, level: usize, s: &mut Scratch) {
        s.visited.reset(self.levels.len());
        s.candidates.clear();
        s.results.clear();
        s.visited.insert(q);
        for &e in &s.entries {
            if s.visited.insert(e.id) {
                s.candidates.push(Reverse(e));
                s.results.push(e);
            }
        }
        while let Some(Reverse(c)) = s.candidates.pop() {
            let worst = s.results.peek().map_or(f32::INFINITY, |r| r.dist);
            if s.results.len() >= self.ef && c.dist > worst {
                break;
            }
            self.copy_neighbors(c.id, level, &mut s.buf);
            for &nb in &s.buf {
                if !s.visited.insert(nb) {
                    continue;
                }
                let d = (self.dist)(q, nb);
                let worst = s.results.peek().map_or(f32::INFINITY, |r| r.dist);
                if s.results.len() < self.ef || d < worst {
                    s.candidates.push(Reverse(Neighbor::new(nb, d)));
                    s.results.push(Neighbor::new(nb, d));
                    if s.results.len() > self.ef {
                        s.results.pop();
                    }
                }
            }
        }
        s.found.clear();
        s.found.extend(s.results.drain());
        s.found.sort_unstable();
    }

    /// The neighbour-selection heuristic (paper, Algorithm 4): walk candidates
    /// closest first and keep one only if it is closer to the base node than
    /// to every neighbour already kept. This spreads links across directions
    /// instead of clustering them, which keeps the graph navigable.
    fn select(&self, cands: &[Neighbor], m: usize) -> Vec<u32> {
        let mut kept: Vec<Neighbor> = Vec::with_capacity(m);
        for &c in cands {
            if kept.len() >= m {
                break;
            }
            if kept.iter().all(|r| (self.dist)(c.id, r.id) > c.dist) {
                kept.push(c);
            }
        }
        kept.into_iter().map(|n| n.id).collect()
    }

    fn freeze(self) -> Graph {
        let n = self.levels.len();
        let m = self.m;
        let (w0, wu) = (2 * m + 1, m + 1);
        let mut level0 = vec![0u32; n * w0];
        let mut upper_off = vec![u32::MAX; n];
        let mut upper = Vec::new();
        for (node, links) in self.links.into_iter().enumerate() {
            let links = links.into_inner();
            let slot = &mut level0[node * w0..(node + 1) * w0];
            slot[0] = links[0].len() as u32;
            slot[1..1 + links[0].len()].copy_from_slice(&links[0]);
            if links.len() > 1 {
                upper_off[node] =
                    u32::try_from(upper.len()).expect("upper layers exceed u32 offsets");
                for list in &links[1..] {
                    let start = upper.len();
                    upper.resize(start + wu, 0);
                    upper[start] = list.len() as u32;
                    upper[start + 1..start + 1 + list.len()].copy_from_slice(list);
                }
            }
        }
        let (entry, max_level) = self.entry.into_inner();
        Graph {
            m: m as u32,
            entry,
            max_level,
            levels: self.levels,
            level0,
            upper_off,
            upper,
        }
    }
}
