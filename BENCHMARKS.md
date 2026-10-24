# Benchmarks

One entry per gate review, newest first. Record the machine with every entry.

All entries below: Intel Core i9-9980HK (8 cores / 16 threads, AVX2, no
AVX-512), 32 GB RAM, macOS, Rust 1.99.1, kernels `avx2`. HNSW M=16,
efConstruction=200 throughout. Latencies are single-query (one query at a
time; multi-segment searches fan out over threads). recall@10 is tie-aware
unless marked "by id".

## 2026-10-23 — Phase 5 gate: the whole database

SIFT1M as one million single-chunk documents in a `.vdb` file, through the
public API (`examples/sift_db.rs`). Ingest in 10K-document transactions with
the default 64K seal threshold, then `optimize()`.

| Step | Result |
| --- | --- |
| Ingest 1M documents | 75 s (13.3K docs/s), 47 s of it in commits and sealing |
| `optimize()` (seal + merge 14 → 5 segments) | 76 s |
| File size / after `Db::compact_file` | 2.16 GB / 1.38 GB |
| Reopen (load 5 segments) | 0.35–0.41 s |
| Resident index | 278 MB |

| Search incl. hydration from the file | recall@10 (by id) | p50 | p99 |
| --- | --- | --- | --- |
| First pass after reopen (cold cache), ef=64 | 0.992 | 2.2 ms | 4.9 ms |
| Warm, ef=32 | 0.981 | 320 µs | 784 µs |
| Warm, ef=64 | 0.992 | 355 µs | 488 µs |
| Warm, ef=128 | 0.998 | 508 µs | 1.06 ms |

Durability and robustness: crash test (10 SIGKILLs mid-write, run 4 times)
lost no acknowledged commit. Corrupted files (1–8 flipped bits): 120 rounds
with every redb panic contained as an error; a 1,000-round run found one file
(round 413) where redb double-panics and aborts the process — see
`docs/design/phase-5.md`.

The headline target at full size (`examples/synthetic768.rs`, synthetic
768-dim cosine vectors shaped like embeddings, 4 x 256K segments):

| Target from the guide | Measured | |
| --- | --- | --- |
| ≤ ~1 GB RAM | 918 MB resident index (f32 vectors stay in the file: 3.07 GB) | pass |
| < 5 ms p99 | ef=32: 624 µs · ef=64: 464 µs · ef=128: 768 µs | pass |
| recall@10 ≥ 0.95 | 0.9997–0.9999 | pass on synthetic data only |

The synthetic set is evidently easier than real embeddings (recall near 1.0
even at ef=32), so treat recall as unproven at 768 dims until it's measured
on a real embedding set; memory and latency carry over. Building the four
256K-row graphs took 11.4 minutes (16 threads), roughly 6x the 128-dim build.
Ground truth by exact search: 9.8 s for 1,000 queries.

## 2026-10-23 — Phase 4 gate: chunking

15 tests, including 800 random property-test documents across Markdown, code,
HTML and text: chunk ranges always tile the document (byte-exact rebuild),
respect the size cap, and form well-nested section trees. Not yet measured:
retrieval hit rate on a labelled question set (needs your documents and model).

## 2026-10-23 — Phase 3 gate: segments, deletes, filters

SIFT1M split into sealed segments (`examples/segments_sift.rs`), int8 walk +
f32 re-rank.

| Configuration, ef=64 | recall@10 | p50 | p99 | concurrent QPS |
| --- | --- | --- | --- | --- |
| 16 x 64K segments, layered | 0.998 | 359 µs | 600 µs | 4,417 |
| 16 x 64K segments, flat graph | 0.9985 | 441 µs | 649 µs | 3,810 |
| 4 x 256K segments (after merge) | 0.992 | 276 µs | 519 µs | 14,045 |
| 1 x 1M segment (Phase 2) | 0.965 | 167 µs | 213 µs | 42,732 |

Filtered search, 16 x 64K segments, ef=64:

| Selectivity | recall@10 | p50 | p99 | Path chosen |
| --- | --- | --- | --- | --- |
| 50% | 0.998 | 654 µs | 1.09 ms | graph walk |
| 10% | 1.000 | 2.15 ms | 3.57 ms | scan |
| 3% | 1.000 | 576 µs | 1.36 ms | scan |
| 1% | 1.000 | 163 µs | 365 µs | scan |
| 0.1% | 1.000 | 54 µs | 154 µs | scan |

Before the cost model (fixed 2% threshold), 10% and 3% took 4.7 ms and
16.9 ms at p50.

## 2026-10-23 — Phase 2 gate: int8 + re-rank

SIFT1M, one sealed segment, int8 graph walk, top 40 re-ranked with f32.

| ef | recall@10 | QPS | p50 | p99 |
| --- | --- | --- | --- | --- |
| 16 | 0.923 | 7,466 | 133 µs | 208 µs |
| 32 | 0.923 | 8,846 | 114 µs | 151 µs |
| 64 | 0.965 | 6,107 | 167 µs | 213 µs |
| 128 | 0.990 | 3,524 | 292 µs | 365 µs |
| 256 | 0.998 | 1,970 | 527 µs | 640 µs |

(ef is raised to at least k x 4 = 40, so ef 16 and 32 run the same search.)
Resident index 277.7 MB: int8 codes 128 MB, graph ~141 MB, ids and
corrections 8 MB. The warm per-segment query path makes zero allocations
(counting-allocator test).

## 2026-10-23 — Phase 1 gate: in-memory f32 HNSW

SIFT1M, one graph over f32 vectors (`examples/hnsw_sift.rs`). Build ~2 min on
16 threads; graph 141 MB; peak RSS 1.03 GB (vectors + graph + build state).

| ef | recall@10 | QPS | p50 | p99 |
| --- | --- | --- | --- | --- |
| 16 | 0.778 | 11,440 | 79 µs | 196 µs |
| 32 | 0.886 | 8,065 | 118 µs | 293 µs |
| 64 | 0.958 | 5,249 | 190 µs | 341 µs |
| 128 | 0.985 | 2,483 | 393 µs | 779 µs |
| 256 | 0.995 | 1,429 | 687 µs | 1.33 ms |

## 2026-10-23 — Phase 0 gate

Machine: Intel Core i9-9980HK (8 cores / 16 threads, AVX2, no AVX-512), macOS, Rust 1.99.1.
Kernels selected at runtime: `avx2`.

### Correctness

| Check | Result |
| --- | --- |
| Unit and property tests (24), AVX2 and forced scalar | Pass |
| SIMD f32 vs scalar, relative 1e-5, dims 0–1099 | Pass (AVX2) |
| SIMD int8 vs exact i64 reference, dims 0–1099 and 65,536 | Pass (AVX2) |
| SIFT1M exact search, 10K queries, recall@1 / @10 / @100 by id | 0.9914 / 0.9994 / 0.9999 |
| Same, counting distance ties | 1.0000 / 1.0000 / 1.0000 |

All by-id misses are exact distance ties caused by duplicate SIFT base vectors.

### Distance kernels (criterion, median)

| Kernel | dim 128 scalar | dim 128 AVX2 | dim 768 scalar | dim 768 AVX2 |
| --- | --- | --- | --- | --- |
| dot_f32 | 81.2 ns | 10.8 ns | 719 ns | 30.0 ns |
| l2_sq_f32 | 99.5 ns | 11.7 ns | 704 ns | 37.8 ns |
| dot_i8 | 20.7 ns | 9.0 ns | 105 ns | 28.4 ns |
| l2_sq_i8 | 25.8 ns | 10.5 ns | 115 ns | 30.5 ns |

### Exact search

| Workload | Time |
| --- | --- |
| 100K x 128, 1 query, top-10 | 3.3 ms |
| 100K x 128, 64 queries batched, top-10 | 15.0 ms (0.23 ms/query) |
| SIFT1M (1M x 128), 10K queries, top-100, 16 threads | 21.9 s (457 QPS, 2.19 ms/query) |
