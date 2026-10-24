# Phase 3 — Segments, deletes and filters

Phase 3 turns one graph into a live index: writes land in a mutable segment,
which is sealed into immutable HNSW segments; deletes are tombstones;
filters are Roaring bitmaps.

## Decisions

**Global `u32` chunk ids, dense local ids.** Each sealed segment renumbers its
vectors `0..len` in ascending global-id order. A global id is found by binary
search, and a filter bitmap can be range-counted per segment
(`range_cardinality`) to estimate selectivity for free.

**Merge then re-rank.** Each segment returns `k x 4` int8 candidates (in real
units, see Phase 2); they merge into one bounded heap, and only that heap is
re-ranked with f32. The mutable segment is searched exactly.

**Seal without blocking readers.** `build_from_mutable` needs only `&self`;
the slow graph build runs under a read lock, and `install_sealed` swaps it in.
Each mutable entry carries a generation stamp, so anything updated or deleted
during the build is tombstoned in the new segment rather than resurrected.

**Filters: cost model, not a fixed 2%.** The guide said "brute force below
~2% selectivity". On 64K-row segments that threshold made 3%-selective
queries take 17 ms (p50). A filtered walk must visit about
`ef / selectivity` nodes, each costing up to `2M` distances, to collect `ef`
matches; a scan costs one distance per matching row. Each segment now picks
the cheaper. 2% remains a floor below which it always scans.

**Compaction is plan / build / apply.** Planning and building need only
`&self`, so rebuilds run while searches continue. A segment past 20%
tombstones is rebuilt; with `merge`, runs of neighbouring segments are merged
up to `max_segment` (default 4x the seal size). Deletes between plan and
apply are carried into the new segment.

## Open decision: keep the upper layers?

Kept. On 16 x 64K SIFT segments the flat graph matched the layered one
(recall@10 0.9985 vs 0.9979 at ef=64, p99 649 vs 600 µs) and saved only
4.6 MB (1.7%). There is no case for the extra code path yet; revisit at
higher dimensions.

## Gate result

Passed. 16 x 64K segments: recall@10 0.998 at ef=64, p99 600 µs. Filtered
p99 at 50/10/3/1/0.1% selectivity: 1.1/3.6/1.4/0.37/0.15 ms with recall
≥ 0.998. Merging to 4 x 256K triples concurrent throughput (4.4K → 14K QPS).
