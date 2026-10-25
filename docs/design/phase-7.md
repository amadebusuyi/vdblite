# Phase 7 — Search throughput, memory and recall

Goal: 10,000 queries per second on 16 threads, less memory, and better
recall for the work spent than the other engines in the embedded
comparison. Measured on 1M real 768-dimension embeddings (arXiv titles,
InstructorXL; `scripts/fetch-arxiv768.sh`) with `examples/qps.rs`, on an
8-core / 16-thread laptop that also runs other work, so every comparison
below is an interleaved A/B with medians, and recall (which is
deterministic) is the anchor.

## Where the time went

At the start, five segments of about 200K, ef=64: 3,500 QPS on 16 threads,
recall@10 0.967. A sampling profile of the 16-thread run:

| | Share of CPU |
| --- | --- |
| int8 distance kernel | ~49% |
| graph walk (heaps, visited set, memory stalls) | ~26% |
| hand-offs between threads (fan-out) | ~16% |
| storage (re-rank reads through redb) | ~3% |

A query computed 6,685 distances: every segment's graph is walked in full,
so five segments cost nearly five searches. Per distance, a 1M index cost
about 250 ns against about 147 ns for a 20K one that fits in cache: the walk
is memory-bound (each distance reads a 768-byte code from a random place),
and the loop's overhead dwarfs the kernel's 28 ns.

## Decisions

**One large graph per collection.** One graph over 1M walks about 1.4K
distances at ef=64 against 6.7K for five segments, and at equal recall
(0.97) gives about twice the throughput. Segments now grow to 16 × the seal
threshold (about 1M chunks) instead of 4 ×.

**Merges grow the largest segment.** Rebuilding a 1M graph took 17 minutes,
too slow to do at every merge. `hnsw::extend` inserts new nodes into an
existing graph (the same insertion a build uses), keeping its links, so a
merge costs the inserts only; `SealedSegment::extend` keeps the base's int8
codes and quantizer. A merge grows its largest member when that member is
clean and holds at least half the rows; otherwise it rebuilds, which also
sheds tombstones. Tombstones in a grown base stay in the graph and move to
the new segment like any compaction's. Background maintenance merges once
four adjacent segments accumulate, so each vector is inserted into a large
graph about once. A grown graph's recall matched or beat a rebuilt one's
(0.9431 vs 0.9376 at ef=64).

**Asymmetric distances.** The walk compares the f32 query against int8
codes (`x ≈ o + s·(c + 128)`, so `q·x = C + Σ (s·q_i)·c_i` with `C` per
query) instead of quantizing the query too. New f32×int8 kernels (scalar,
AVX2, NEON). Recall was identical to the symmetric form, which says
quantization isn't what limits recall here; it is kept because it reads no
per-vector correction (one cache line fewer per distance) and measured about
4% faster.

**Fan out only while cores are idle.** A search spreads its segments over
the thread pool only when `in-flight searches × segments ≤ threads`. No
throughput change under load (the hand-off samples were idle spinning), but
no wasted hand-offs either.

**Prefetch.** The walk collects a node's unvisited neighbours, prefetches
their codes, then computes, and prefetches the next candidate's neighbour
list. +2–4%, within noise; kept because it's cheap and helps more on
machines with higher memory latency.

**Page cache 256 MiB.** redb caches file pages on top of the OS. 256 MiB
searched as fast as the old 1 GiB default (64 MiB was 30% slower), and caps
up to 750 MB of growth in a long-running process. `OpenOptions::cache_bytes`
changes it.

**`Search::chunk_ids`.** Ids and scores without reading documents: the
cheapest result shape, and what other engines' benchmarks return.

## Measured, not adopted

- **Deeper re-ranking.** Re-ranking 80 candidates instead of 40 gained no
  recall at ef=128, and each extra one cost about 38 µs: the f32 vectors
  (3 GB) don't fit the page cache, so most re-rank reads go to the file.
  Recall has to come from the graph.
- **A faster int8 kernel.** At 768 dimensions the walk is memory-bound;
  the kernel's arithmetic is not the limit.

## Graph parameters

Recall at a given ef is limited by the graph, so the links per node (M)
and the build beam (efConstruction) were swept on 200K of the real vectors
(`examples/graph_sweep.rs`; every graph built first, throughput measured in
five interleaved rounds, median):

| M | ef=32 | ef=64 | ef=128 | ef=256 | index |
| --- | --- | --- | --- | --- | --- |
| 16 | 0.910 · 30.6K | 0.948 · 22.3K | 0.979 · 12.9K | 0.991 · 7.3K | 184 MB |
| 32 | 0.954 · 19.0K | 0.974 · 13.4K | 0.992 · 7.8K | 0.997 · 4.4K | 209 MB |

(recall@10 · QPS on 16 threads; efConstruction 200). At equal recall the
two are level up to about 0.975, and M=32 is about 7% faster at 0.99 and
reaches 0.997 where M=16 can't within ef=256, for 14% more index memory.
efConstruction 400 added at most 0.003 recall for twice the build time.
The default stays M=16, efConstruction 200; collections that need ≥ 0.99
recall should set `hnsw.m = 32`.

## Known costs

- **Ingest is slower while merges run.** Bulk-loading 1M took 1,258 s
  (794 docs/s) against 849 s before, with commit p50 984 ms against 410 ms.
  The background merges read and rewrite segments of up to 1M vectors, and
  commits (two disk flushes each in shared mode) wait behind that I/O. It
  then needs no 17-minute rebuild to reach one graph.
- **The file grows** as merges rewrite segments copy-on-write (8.6 GB
  against 6.4 GB for 1M at 768 dims); `Db::compact_file` reclaims it.
