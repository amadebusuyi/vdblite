# Phase 1 — In-memory HNSW

Phase 1 proves recall and speed in memory, with full f32 vectors, before
anything is quantized or written to a file.

## Decisions

**Layout first.** The built graph is flat `u32` arrays (`level0` fixed-width
slots of `[count, n1..n_2M]`, `upper_off`, packed `upper`), the layout the
file will hold. Nothing is converted between build and search.

**The graph knows no vectors.** Build and search take a distance closure, so
one graph serves f32 (this phase), int8 (Phase 2) and anything later.

**Parallel build with a lock per node.** rayon inserts nodes concurrently;
each node's lists have their own `parking_lot::Mutex`, and a thread holds at
most one at a time, so inserts cannot deadlock. A node that raises the top
level holds a `grow` lock for its whole insert (as hnswlib does), so the entry
point only ever moves to a fully linked node. The first 1,024 nodes are
inserted serially so early inserts see a connected graph.

**Neighbour selection** uses the paper's heuristic (Algorithm 4): keep a
candidate only if it is closer to the base node than to every neighbour kept
so far. New nodes get `M` links; layer 0 holds up to `2M`.

**Visited set** is a `u16` epoch array, cleared in O(1) by bumping the
epoch and wiped once every 65,535 searches.

**Filtered and deleted nodes route but are not returned**: search takes an
`accept` predicate; rejected nodes still enter the candidate queue.

**Corrupt input is an error.** `Graph::from_parts` checks every count and
neighbour id, so a damaged file can't cause an out-of-bounds panic in the
search loop.

## Gate result

Passed on SIFT1M (1M x 128, M=16, efConstruction=200): recall@10 0.958 at
ef=64 with p99 341 µs, single thread. Build about 2 minutes on 16 threads.
See `BENCHMARKS.md`.
