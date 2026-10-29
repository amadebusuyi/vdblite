# vdblite vs Zvec

vdblite 0.1.0 measured against Zvec 0.8.0 on the same laptop, with the same
graph settings, on three datasets:

| Dataset | Vectors | Dims | Metric | Harness |
| --- | --- | --- | --- | --- |
| Cohere 1M | 1,000,000 | 768 | cosine | VectorDBBench, top 100 |
| SIFT1M | 1,000,000 | 128 | L2 | embedded harness, top 10 |
| arXiv 500K (titles, InstructorXL) | 500,000 | 768 | cosine | embedded harness, top 10 |

Runs from 2026-10-26 (Cohere) and 2026-10-27 (SIFT1M, arXiv).

## Summary

- **vdblite loads faster on two of the three datasets.**
  - **SIFT1M:** 75 s against Zvec's 102 s.
  - **arXiv:** 91 s against 280 s.
  - **Cohere:** close in Docker (vdblite 330 s, Zvec 365 s). Natively, Zvec is a little faster (194 s vs 209 s).
- **Under concurrent load, vdblite has more throughput.**
  - **Cohere, through VectorDBBench, where both engines run through Python:**
    vdblite is 13% faster at 8 processes (2,484 vs 2,195 QPS) and equal at
    12–16, at about the same recall.
- **Single-query speed depends on the dataset.**
  - **Cohere:** vdblite's p99 is 2.7 ms against Zvec's 10.0 ms.
  - **SIFT1M:** vdblite is faster at the same recall.
  - **arXiv:** Zvec is about 2× faster at the same recall (1.26 ms vs
    2.60 ms at ~0.993 recall@10).
- **Zvec gets more recall from the same ef setting on every dataset.** At
  ef 64, recall@10 is 0.9658 vs 0.9564 on SIFT1M and 0.9501 vs 0.9400 on
  arXiv. vdblite needs a higher setting to match it.
- **vdblite uses less memory and starts faster.**
  - **Memory after searching:** 517 MB vs 813 MB (SIFT1M) and 1.13 GB vs
    1.69 GB (arXiv).
  - **Cohere:** a vdblite search process holds 37 MB of private memory,
    because all processes share one memory-mapped copy of the index.
  - **First query after opening:** 9–27 ms against Zvec's 157–348 ms.
- **The two are about the same on disk once vdblite's file is compacted.**
  Before compaction vdblite's files are larger.

## Setup

| | |
| --- | --- |
| Machine | Intel Core i9-9980HK (8 cores / 16 threads, AVX2), 32 GB RAM, macOS |
| Index, both engines | HNSW, M = 16, efConstruction = 200, int8 codes for the graph walk, full-precision re-rank of k × 4 candidates |
| Versions | vdblite 0.1.0, Zvec 0.8.0 |

The two harnesses are described in their sections below.

## Cohere 1M (VectorDBBench)

VectorDBBench commit 139f8a3d (`scripts/vectordbbench/`), both engines
through their Python bindings, one search process per client.

| | |
| --- | --- |
| Queries | 1,000 test queries, k = 100 |
| Default search | ef 400 for both (vdblite raises its ef 64 to k × 4 re-rank candidates) |
| High-recall search | ef 800 for both |
| Concurrency | 30 s at each level |
| Serial latency | All 1,000 queries, one at a time |

| | vdblite (int8) | Zvec 0.8 |
| --- | --- | --- |
| Load (insert + optimize) | 209 s | **194 s** |
| QPS, 8 processes | **2,484** | 2,195 |
| QPS, 12 / 16 processes | 2,491 / 2,463 | 2,477 / 2,472 |
| Serial p50 / p99 | 2.1 / **2.7 ms** | 2.1 / 10.0 ms |
| Recall@100 | 0.9717 | **0.9746** |
| ef 800: QPS at 8 processes / best | **1,472 / 1,515** | 1,249 / 1,329 |
| ef 800: recall@100 | 0.9866 | 0.9875 |
| File size after load | 4.5 GB (2.6 GB after `compact_file`) | **3.8 GB** |
| Private memory, 1 / 8 search processes | 37 MB / 299 MB | not measured |

vdblite's peak RSS per search process is 1.34 GB. That figure counts pages
of the file mapped into memory: every process shares them, and the OS can
drop them when memory runs low, so it isn't memory the process owns.

After a bulk load the file keeps the space its load checkpoints used.
Later writes reuse that space, but the file only shrinks when
`compact_file` runs (12.5 s, 4.5 GB → 2.6 GB).

### In identical Docker containers

Both engines run from the same image, one after the other, with databases
on a Docker volume inside the VM (`scripts/vectordbbench/docker-run.sh`).
QPS is the best over 1–16 processes.

| | vdblite | Zvec 0.8 |
| --- | --- | --- |
| Load | **330 s** | 365 s |
| QPS, default (cold / warm run) | **2,076 / 2,227** | 1,918 / 1,881 |
| Recall@100, default | 0.972 | **0.9747** |
| Serial p50 / p99, default | **2.4 / 3.5 ms** | 2.9 / 4.8 ms |
| ef 800: QPS / recall | **1,206** / 0.9865 | 1,153 / **0.9876** |
| ef 800: serial p50 / p99 | **3.3** / 7.2 ms | 4.8 / 7.2 ms |

Both engines load more slowly in Docker than natively. The VM's
peak-memory figure (11.7 GB for vdblite, 11.9 GB for Zvec) includes the
page cache of each engine's files, so it doesn't compare the engines'
own memory.

## SIFT1M and arXiv 500K (embedded harness)

`scripts/compare/embedded/run.sh` runs each engine in the same Linux
container image, one at a time:

1. **Build:** a build container bulk-loads the base vectors in 10K batches
   and builds the index.
2. **Search:** a fresh container drops the page cache, opens the database,
   and runs one cold pass. Then it runs a warm pass at each ef setting,
   followed by a concurrent run at ef 64 with 16 threads.

Recall@10 is by id, against exact ground truth over 1,000 queries.
vdblite's search latencies return ids only, as Zvec's do; reading back the
stored records is listed separately.

**The two engines are driven differently.** vdblite runs from Rust
(`examples/ann_compare.rs`); Zvec runs from Python (`bench.py`), and its
concurrent run uses Python threads. A query against a 10-vector collection
gives each engine's fixed per-query cost: 1–4 µs for vdblite and about
300 µs for Zvec (314 µs on SIFT1M, 297 µs on arXiv). The Zvec latencies
below include that cost.

### Load, size and memory

| | vdblite, SIFT1M | Zvec, SIFT1M | vdblite, arXiv | Zvec, arXiv |
| --- | --- | --- | --- | --- |
| Ingest (load + index) | **74.7 s** (70.5 + 4.2) | 102.3 s (22.2 + 80.1) | **91.4 s** (72.5 + 18.9) | 279.6 s (86.1 + 193.5) |
| On disk | 1,074 MB (866 MB compacted) | **861 MB** | 2,137 MB (1,392 MB compacted) | 2,056 MB |
| Open | 227 ms | **100 ms** | 1,404 ms | **168 ms** |
| First query after open | **9.2 ms** | 157 ms | **26.5 ms** | 348 ms |
| Cold pass p50 / p99, ef 64 | **177 µs / 4.3 ms** | 733 µs / 10.1 ms | **2.8 / 23.5 ms** | 7.2 / 84.0 ms |
| Memory added by opening and searching | **517 MB** | 813 MB | **1,132 MB** | 1,686 MB |

`compact_file` takes 4.2 s on SIFT1M and 4.6 s on arXiv. The memory row is
the search process's resident memory after the warm passes, minus what it
used before opening the database.

### Recall and latency by ef (warm, one query at a time)

SIFT1M:

| ef | vdblite recall@10 | vdblite p50 / p99 | Zvec recall@10 | Zvec p50 / p99 |
| --- | --- | --- | --- | --- |
| 40 | 0.9113 | 76 / 117 µs | 0.9276 | 381 / 491 µs |
| 64 | 0.9564 | 123 / 172 µs | 0.9658 | 411 / 525 µs |
| 128 | 0.9857 | 187 / 269 µs | 0.9913 | 479 / 602 µs |
| 256 | 0.9953 | 323 / 427 µs | 0.9979 | 630 / 843 µs |
| 512 | 0.9972 | 588 / 752 µs | 0.9997 | 980 / 1,315 µs |

arXiv 500K:

| ef | vdblite recall@10 | vdblite p50 / p99 | Zvec recall@10 | Zvec p50 / p99 |
| --- | --- | --- | --- | --- |
| 40 | 0.9088 | 375 / 671 µs | 0.9166 | 3,957 µs / 268 ms ¹ |
| 64 | 0.9400 | 539 / 793 µs | 0.9501 | 647 / 945 µs |
| 128 | 0.9729 | 810 / 1,091 µs | 0.9809 | 861 / 1,288 µs |
| 256 | 0.9883 | 1,463 / 2,160 µs | 0.9922 | 1,264 / 2,233 µs |
| 512 | 0.9934 | 2,602 / 3,338 µs | 0.9966 | 2,220 / 6,688 µs |

¹ An outlier with no recorded cause: the timed pass had its own warm-up
pass first, yet its mean was 95 ms. Every other Zvec arXiv setting runs at
0.6–2.2 ms p50.

**At the same recall, the ranking differs between the two datasets.**

- **SIFT1M, ~0.99 recall@10:** vdblite takes 187–323 µs (ef 128–256),
  Zvec 479 µs (ef 128). Zvec's ~314 µs fixed cost is most of its time here.
  Without it, Zvec's search work at ef 128 is about 165 µs for 0.9913
  recall, slightly less than vdblite's 187 µs for 0.9857.
- **SIFT1M, ~0.998:** vdblite takes 588 µs (ef 512), Zvec 630 µs (ef 256).
- **arXiv, ~0.98–0.99:** vdblite takes 1,463 µs (ef 256, 0.9883), Zvec
  861–1,264 µs (ef 128–256, 0.9809–0.9922).
- **arXiv, ~0.993:** vdblite takes 2,602 µs (ef 512), Zvec 1,264 µs (ef
  256). Zvec is about 2× faster here, even including its fixed cost.

### Concurrent throughput (ef 64, 16 threads)

| | vdblite | Zvec |
| --- | --- | --- |
| SIFT1M | 41,606 QPS (27,724 reading back records) | 7,374 QPS |
| arXiv 500K | 12,383 QPS (9,594 reading back records) | 3,918 QPS |

Treat this as a comparison of the two harnesses, not of the engines. Zvec's
16 threads are Python threads paying its ~300 µs per-query cost, while
vdblite's are Rust threads. Recall at ef 64 also differs: Zvec is about
0.01 higher on both datasets. The Cohere results above, where both engines
run through Python processes, are the fairer throughput comparison.

vdblite's "reading back records" figures fetch each hit's stored document
from the file. That takes vdblite's ef 64 p50 to 285 µs on SIFT1M and
816 µs on arXiv.

## vdblite with 1-bit codes

vdblite can walk the graph with 1-bit RaBitQ codes instead of int8
(`quant="binary"`). int8 is the default. On Cohere 1M:

| Through VectorDBBench | int8 (default) | 1-bit |
| --- | --- | --- |
| Load | 209 s | 222 s |
| QPS, 8 / 16 processes | 2,484 / 2,463 | 1,810 / 1,944 |
| Recall@100 | 0.9717 | 0.9835 |
| Serial p50 / p99 | 2.1 / 2.7 ms | 2.5 / 3.7 ms |
| Private memory, 1 / 8 processes | 37 / 299 MB | 40 / 316 MB |
| Peak RSS, 1 process | 1.34 GB | 0.94 GB |
| File size after load | 4.5 GB | 3.9 GB |

The 1-bit run uses 8 × k re-rank candidates by default (int8 uses 4 × k),
which is why it trades throughput for recall here. On the same graph, with
the same re-rank depth and search setting, the two code sizes reach the
same recall at about the same speed (`examples/walk_sweep.rs`, 8 threads,
k = 100):

| Codes | Graph-walk data | rerank 4, ef 800 | rerank 8, ef 800 | rerank 16, ef 800 |
| --- | --- | --- | --- | --- |
| int8 | 918 MB | 0.9867 · 1,696 QPS | 0.9867 · 1,534 | 0.9927 · 818 |
| 1-bit | 254 MB | 0.9748 · 2,838 | 0.9834 · 2,212 | 0.9916 · 1,291 |

So 1-bit codes are a way to cut the memory a search touches by 3.6×, not
a speed-up.

## Graph build time

Building a 1M × 768 graph (`examples/stream_build.rs`) takes 131.7 s in one
shot, or 134.4 s when streamed in 10K batches. During a bulk load the build
runs alongside the writes and sets the pace: 209 s through VectorDBBench,
and 170 s for a Python bulk load straight from a NumPy file.

## Caveats

- **Each configuration ran once.** Repeat runs on this machine have moved
  p50 by about 10% and p99 by up to 2×, so treat small differences as noise.
- **The benchmark client shares the CPU with the engine** in every harness.
- **Different languages drive the engines in the embedded harness:** Rust
  for vdblite, Python for Zvec. That favours vdblite on latency and
  especially on concurrent throughput.
- **Zvec's private memory wasn't measured on Cohere.** The embedded
  harness's resident-memory figures are the only memory comparison.
- **Container resources weren't recorded.** The embedded and Docker
  VectorDBBench runs were in Docker's Linux VM, which reports 16 vCPUs and
  23.2 GB now.

## Sources

- `BENCHMARKS.md`, Phase 8 entry (2026-10-26)
- `data/vectordbbench/cohere1m-zvec.log`, `cohere1m-vdblite-step4.log`,
  `docker-vdblite.log`, `docker-zvec.log`
- `target/compare/embedded/sift-1000000/{vdblite,zvec}-{build,search}.json`
  and `sift-1000000-phase8.log`
- `target/compare/embedded/arxiv768-500000/{vdblite,zvec}-{build,search}.json`
  and `arxiv768-500000-phase8.log`
- `scripts/vectordbbench/`, `scripts/compare/embedded/`
