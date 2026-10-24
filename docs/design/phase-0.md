# Phase 0 — Foundations

Phase 0 builds what every later phase is measured with: distance kernels,
exact search, a dataset loader and a recall metric. No index, no storage.

## Decisions

**Distance convention.** Every metric returns a distance where smaller is
closer: L2 is squared (no `sqrt`), Dot is `-(a·b)`, Cosine is `1 - a·b`.
One ordering everywhere means one top-k heap and one merge.

**Cosine requires unit vectors.** Cosine is computed as a dot product, so
vectors must be normalized on insert (`distance::normalize`). That makes
cosine cost the same as dot instead of three accumulations.

**Kernel dispatch.** A `Kernels` struct of function pointers is chosen once
(`OnceLock`) from runtime CPU detection: AVX2+FMA on x86_64, NEON on aarch64,
scalar otherwise. `VECFILE_KERNELS=scalar` forces the fallback. Hot loops
fetch `kernels()` once and call through the pointers. A SIMD set can only be
obtained after detection succeeds, so the pointers are safe to call.
AVX-512 is left for later: this dev machine lacks it, so it couldn't be tested here.

**Tolerance.** int8 kernels must match scalar exactly (i32 sums, exact up to
`MAX_DIM` = 65,536). f32 kernels reassociate the sum, so "within 1e-5" is
measured relative to the sum of absolute terms, which bounds rounding error
for any summation order.

**Recall counts ties.** SIFT1M contains duplicate base vectors. Exact search
scored 0.9914 recall@1 by id; all 86 misses were distance ties (checked exactly on the first 1K queries). The
harness reports recall by id and with ties counted as hits (ann-benchmarks'
definition, 1e-3 tolerance), and gates on the tie-aware number.

**Brute force is tiled.** `search_batch` runs a tile of up to 32 queries
against L2-sized blocks of base vectors, sized down so small batches still
use every thread. It is the ground truth for later phases, so it should be fast
enough to re-run often.

## Gate result

Passed. See `BENCHMARKS.md` (2026-10-23 entry).

## Carried forward

- int8 AVX2 is no faster than f32 AVX2 per element (both ~28 ns at 768 dims): the
  widen-to-i16 path does 16 lanes per `vpmaddwd`. Phase 2 (quantization) should
  try `vpmaddubsw` with an unsigned offset, or VNNI where available.
- NEON kernels compile cleanly but haven't run yet; CI's `macos-14` job runs them.
- Benchmark noise on a laptop measured ±15% with no code change, so a 5%
  regression gate needs a quiet, fixed runner.
