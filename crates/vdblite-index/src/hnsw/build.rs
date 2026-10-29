//! Parallel HNSW construction.
//!
//! Nodes are inserted concurrently with rayon. Each node's adjacency lists
//! sit behind their own read-write lock: the walks every insert makes only
//! read neighbour lists, so they run side by side, and a lock is held for
//! writing only to add a link. A thread holds at most one node lock at a
//! time, so inserts cannot deadlock. A node that raises the graph's top
//! level holds `grow` for its whole insert, as in hnswlib, so the entry point
//! only ever moves to a fully linked node.
//!
//! Every link keeps its distance, so pruning a full neighbour list costs
//! only the selection heuristic's comparisons, not a fresh distance to every
//! neighbour.

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

/// A link whose distance isn't known yet (inherited from a base graph);
/// it is computed the first time its list is pruned.
const UNKNOWN: f32 = f32::NAN;

/// Builds a graph over nodes `0..n`, where `dist(a, b)` is the distance
/// between two nodes. It must be symmetric.
pub fn build<F>(n: usize, params: &HnswParams, dist: F) -> Graph
where
    F: Fn(u32, u32) -> f32 + Sync,
{
    build_prefetching(n, params, dist, |_| {})
}

/// [`build`], calling `prefetch(node)` shortly before computing distances to
/// a node, so its data can load from memory alongside other nodes'.
pub fn build_prefetching<F, P>(n: usize, params: &HnswParams, dist: F, prefetch: P) -> Graph
where
    F: Fn(u32, u32) -> f32 + Sync,
    P: Fn(u32) + Sync,
{
    let mut g = GraphBuilder::new(params);
    g.grow(n, dist, prefetch);
    g.finish()
}

/// A graph built in steps, for nodes that arrive over time: each
/// [`GraphBuilder::grow`] inserts the nodes added since the last one. The
/// result is the graph [`build`] makes over the same nodes in the same
/// order (up to the races of parallel insertion).
pub struct GraphBuilder {
    params: HnswParams,
    rng: SplitMix64,
    state: State,
}

impl GraphBuilder {
    pub fn new(params: &HnswParams) -> Self {
        assert!(params.m >= 2, "m must be at least 2");
        GraphBuilder {
            params: *params,
            rng: SplitMix64::new(params.seed),
            state: State {
                m: params.m,
                ef: params.ef_construction.max(params.m),
                levels: Vec::new(),
                links: Vec::new(),
                entry: RwLock::new((0, 0)),
                grow: Mutex::new(()),
            },
        }
    }

    /// Nodes so far.
    pub fn len(&self) -> usize {
        self.state.levels.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Adds nodes `len()..n` and inserts them, in parallel. `dist` and
    /// `prefetch` cover nodes `0..n`.
    pub fn grow<F, P>(&mut self, n: usize, dist: F, prefetch: P)
    where
        F: Fn(u32, u32) -> f32 + Sync,
        P: Fn(u32) + Sync,
    {
        assert!(n <= u32::MAX as usize, "too many nodes");
        let start = self.len();
        if n <= start {
            return;
        }
        let ml = 1.0 / (self.params.m as f64).ln();
        for _ in start..n {
            let level = level_of(&mut self.rng, self.params.layered, ml);
            self.state.levels.push(level);
            self.state
                .links
                .push(RwLock::new(vec![Vec::new(); level as usize + 1]));
        }
        if start == 0 {
            *self.state.entry.get_mut() = (0, self.state.levels[0]);
        }
        let b = Builder {
            state: &self.state,
            dist,
            prefetch,
        };
        let serial = n.min(SERIAL_PREFIX).max(start);
        let mut scratch = Scratch::new(n);
        for q in start.max(1)..serial {
            b.insert(q as u32, &mut scratch);
        }
        (serial..n)
            .into_par_iter()
            .for_each(|q| with_scratch(|s| b.insert(q as u32, s)));
    }

    pub fn finish(self) -> Graph {
        if self.is_empty() {
            return Graph::empty(self.params.m as u32);
        }
        self.state.freeze()
    }
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
    extend_prefetching(base, remap, n, params, dist, |_| {})
}

/// [`extend`] with a prefetch hook, as [`build_prefetching`].
pub fn extend_prefetching<F, P>(
    base: &Graph,
    remap: &[u32],
    n: usize,
    params: &HnswParams,
    dist: F,
    prefetch: P,
) -> Graph
where
    F: Fn(u32, u32) -> f32 + Sync,
    P: Fn(u32) + Sync,
{
    assert_eq!(remap.len(), base.len(), "one new position per base node");
    assert!(n <= u32::MAX as usize, "too many nodes");
    if base.is_empty() {
        return build_prefetching(n, params, dist, prefetch);
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
    for (node, &fresh) in fresh_levels.iter().enumerate() {
        if !is_base[node] {
            levels[node] = fresh;
        }
    }
    for (old, &new) in remap.iter().enumerate() {
        levels[new as usize] = base.level(old as u32);
    }
    let links: Vec<RwLock<Vec<Vec<Neighbor>>>> = levels
        .iter()
        .map(|&l| RwLock::new(vec![Vec::new(); l as usize + 1]))
        .collect();
    for (old, &new) in remap.iter().enumerate() {
        let mut g = links[new as usize].write();
        for (level, list) in g.iter_mut().enumerate() {
            list.extend(
                base.neighbors(old as u32, level as u8)
                    .iter()
                    .map(|&x| Neighbor::new(remap[x as usize], UNKNOWN)),
            );
        }
    }
    let state = State {
        m: params.m,
        ef: params.ef_construction.max(params.m),
        links,
        entry: RwLock::new((remap[base.entry() as usize], base.max_level())),
        grow: Mutex::new(()),
        levels,
    };
    let b = Builder {
        state: &state,
        dist,
        prefetch,
    };
    let fresh: Vec<u32> = (0..n as u32).filter(|&i| !is_base[i as usize]).collect();
    fresh
        .into_par_iter()
        .for_each(|q| with_scratch(|s| b.insert(q, s)));
    state.freeze()
}

fn assign_levels(n: usize, params: &HnswParams) -> Vec<u8> {
    let ml = 1.0 / (params.m as f64).ln();
    let mut rng = SplitMix64::new(params.seed);
    (0..n)
        .map(|_| level_of(&mut rng, params.layered, ml))
        .collect()
}

fn level_of(rng: &mut SplitMix64, layered: bool, ml: f64) -> u8 {
    if layered {
        (-rng.next_f64_open().ln() * ml).floor().min(16.0) as u8
    } else {
        0
    }
}

/// A graph under construction.
struct State {
    m: usize,
    ef: usize,
    levels: Vec<u8>,
    /// Per node, per level: its links and their distances to it.
    links: Vec<RwLock<Vec<Vec<Neighbor>>>>,
    entry: RwLock<(u32, u8)>,
    grow: Mutex<()>,
}

/// Inserts nodes into a [`State`] with a given distance.
struct Builder<'a, F, P> {
    state: &'a State,
    dist: F,
    prefetch: P,
}

impl<F, P> std::ops::Deref for Builder<'_, F, P> {
    type Target = State;

    fn deref(&self) -> &State {
        self.state
    }
}

thread_local! {
    /// Each thread's insert buffers, kept between inserts and between
    /// builds: the visited set alone is two bytes per node, too much to
    /// allocate per rayon task when a streamed build grows a big graph a
    /// little at a time.
    static SCRATCH: std::cell::RefCell<Scratch> = std::cell::RefCell::new(Scratch::new(0));
}

fn with_scratch<R>(f: impl FnOnce(&mut Scratch) -> R) -> R {
    SCRATCH.with(|s| f(&mut s.borrow_mut()))
}

struct Scratch {
    visited: Visited,
    candidates: BinaryHeap<Reverse<Neighbor>>,
    results: BinaryHeap<Neighbor>,
    found: Vec<Neighbor>,
    entries: Vec<Neighbor>,
    buf: Vec<u32>,
    fresh: Vec<u32>,
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
            fresh: Vec::new(),
        }
    }
}

impl<F: Fn(u32, u32) -> f32 + Sync, P: Fn(u32) + Sync> Builder<'_, F, P> {
    fn cap(&self, level: usize) -> usize {
        if level == 0 { 2 * self.m } else { self.m }
    }

    fn copy_neighbors(&self, node: u32, level: usize, buf: &mut Vec<u32>) {
        buf.clear();
        let g = self.links[node as usize].read();
        if let Some(list) = g.get(level) {
            buf.extend(list.iter().map(|n| n.id));
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
            self.links[q as usize].write()[level] = selected.clone();
            let cap = self.cap(level);
            for back in &selected {
                // The neighbour's distance to q is the one q found to it.
                let mut g = self.links[back.id as usize].write();
                let list = &mut g[level];
                if list.iter().any(|x| x.id == q) {
                    continue;
                }
                let link = Neighbor::new(q, back.dist);
                if list.len() < cap {
                    list.push(link);
                } else {
                    let mut cands: Vec<Neighbor> = list.clone();
                    cands.push(link);
                    for c in &mut cands {
                        if c.dist.is_nan() {
                            c.dist = (self.dist)(back.id, c.id);
                        }
                    }
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
            // Unvisited neighbours first, all prefetched, so their loads
            // overlap rather than stall one after another.
            s.fresh.clear();
            for &nb in &s.buf {
                if s.visited.insert(nb) {
                    s.fresh.push(nb);
                }
            }
            for &nb in &s.fresh {
                (self.prefetch)(nb);
            }
            for &nb in &s.fresh {
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
    fn select(&self, cands: &[Neighbor], m: usize) -> Vec<Neighbor> {
        let mut kept: Vec<Neighbor> = Vec::with_capacity(m);
        for &c in cands {
            if kept.len() >= m {
                break;
            }
            if kept.iter().all(|r| (self.dist)(c.id, r.id) > c.dist) {
                kept.push(c);
            }
        }
        kept
    }
}

impl State {
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
            for (dst, l) in slot[1..1 + links[0].len()].iter_mut().zip(&links[0]) {
                *dst = l.id;
            }
            if links.len() > 1 {
                upper_off[node] =
                    u32::try_from(upper.len()).expect("upper layers exceed u32 offsets");
                for list in &links[1..] {
                    let start = upper.len();
                    upper.resize(start + wu, 0);
                    upper[start] = list.len() as u32;
                    for (dst, l) in upper[start + 1..start + 1 + list.len()]
                        .iter_mut()
                        .zip(list)
                    {
                        *dst = l.id;
                    }
                }
            }
        }
        let (entry, max_level) = self.entry.into_inner();
        Graph {
            m: m as u32,
            entry,
            max_level,
            levels: self.levels.into(),
            level0: level0.into(),
            upper_off: upper_off.into(),
            upper: upper.into(),
        }
    }
}
