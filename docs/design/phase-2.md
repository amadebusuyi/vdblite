# Phase 2 — int8 quantization and re-ranking

Phase 2 makes the index memory-lean: the graph walk uses int8 codes, and only
the final `k x 4` candidates touch full f32 vectors.

## Decisions

**Shared scale, per-dimension offset.** The guide sketched per-dimension
scale *and* offset. With a per-dimension scale, an int8 distance is a
weighted sum the integer kernels can't compute. With one scale `s` shared by
all dimensions and a per-dimension offset `o_i`:

- L2: `|x - y|² = s² · Σ (c_i - d_i)²` — exact for the quantized vectors.
- Dot/cosine: `x · y = s² · Σ c_i d_i + corr(c) + corr(d) + K`, with one f32
  `corr` stored per vector and `K` a constant.

So int8 distances come out in real units. That matters in Phase 3: candidates
from segments with different quantizers merge into one heap directly.

**Robust range.** Offsets and the scale come from each dimension's 0.1% and
99.9% quantiles (sampled from up to 20K rows), so a few outliers don't waste
codes; values outside are clamped.

**Symmetric queries.** The query is quantized with the segment's quantizer,
so the walk is pure int8 kernel calls.

**Build with f32, search with int8.** The graph is built from full vectors
(better neighbour choices); the codes are only for walking it.

**Zero allocation on the warm path.** `SegmentScratch` holds the visited set,
heaps, query code and hit list. A test with a counting global allocator
asserts zero allocations across 100 warm segment searches.

**f32-only mode.** `Quantization::None` keeps f32 vectors in memory and
skips re-ranking: the guide's mitigation for models that lose recall under
int8.

## Gate result

Passed on SIFT1M as a single segment: recall@10 0.965 at ef=64 (f32 HNSW:
0.958), p99 213 µs, resident index 278 MB vs 512 MB of f32 vectors plus the
graph. At 768 dims the same layout projects to about 0.92 GB.

## Carried forward

int8 kernels are no faster per element than f32 FMA kernels (both ~28 ns at
768 dims on AVX2), so the int8 win is memory and bandwidth, not compute.
`vpmaddubsw` or VNNI could change that.
