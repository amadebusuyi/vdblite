# Phase 8 — Faster loads, less memory, more throughput

Goal, on 1M Cohere vectors (768 dims, cosine, top 100) through
VectorDBBench, against the Phase 7 numbers on the same laptop:

| | Phase 7 | Target | Zvec 0.8 (same machine, same graph settings) |
| --- | --- | --- | --- |
| Load (insert + optimize) | 1,126 s | ≤ 225 s (5×) | 194 s |
| Peak memory, one search process | 2.0 GB | ≤ 0.67 GB (3×) | |
| Memory, 8 search processes | ~16 GB | ≤ 1 GB | |
| QPS, 8 processes, recall 0.97 | 993 | ≥ 2,000 (2×) | 2,195 |
| Recall@100, default / high setting | 0.973 / 0.988 | ≥ 0.97 / ≥ 0.985 | 0.975 / 0.988 |

Zvec reaches every target on this hardware with M=16, efConstruction=200,
int8 codes and an f32 refine, so none of them asks for more than the
machine can do.

## Where the time and memory go

All three gaps trace back to one Phase 5 decision: vectors and sealed
segments are redb values.

- **Queries.** A top-100 query re-ranks 400 candidates. Each is a B-tree
  lookup and a `pread` of a 3 KB value; a profile of the Cohere file spent
  over 90% of query time there and under 5% walking the graph.
- **Memory.** Opening a file copies every segment out of redb into private
  memory: the 918 MB index peaks at 2.0 GB while it loads, and every
  process holds its own copy. VectorDBBench searches with one process per
  client; at 16 the laptop swapped and throughput fell to 36 QPS.
- **Loads.** 915 of the 1,126 seconds are `optimize`. A profile of a bulk
  load: graph inserts computing f32 distances on 3 KB vectors take nearly
  all the CPU; each vector is inserted about 1.7 times (into its 64K
  segment, then again by the merge); background builds use half the cores
  at low priority while the caller waits; and a 16-thread build keeps only
  4–6 threads busy. A one-shot 1M graph takes 17 minutes against Zvec's
  135 s. Merges also rewrite whole segments copy-on-write, which is why the
  file reaches 8.6 GB.

## Decisions

**One file, with mapped regions.** The `.vdb` stays a single file. vdblite
implements redb's `StorageBackend` and owns the file layout: redb sees a
logical file, and the physical file holds both redb's pages and immutable,
page-aligned regions that every process maps read-only.

The physical file is two superblock slots (64 KiB each, checksummed,
written alternately like redb's own dual header) followed by *extents* and
*regions*. redb's logical file is a list of extents (1 MiB, doubling to at
most 256 MiB) recorded in the superblock; everything else is region space.
The extent list changes only when redb's file grows: under the writer lock,
with the new superblock synced before redb uses the space. New redb space
reads as zeros, as the trait requires.

redb has no public way to open a read-only database on a custom backend, so
vdblite builds against a fork (`crates/vdblite-redb`, see its `FORK.md`)
that adds the one constructor; nothing else differs from redb 4.4.0.

A region (one segment's graph, codes and ids; a block of re-rank vectors) is
written into region arenas, synced, and only then committed: the redb record
that points at it (arena, offset, length, checksum) lands in the same
transaction as everything else, so a crash leaves at most unreferenced
space, which the next open reclaims. Regions are never modified.

Space from regions that are no longer referenced (merged away, dropped) is
reused only when no process can still be reading it. Every commit that
frees a region bumps an epoch stored in the file, and the freed space joins
a coalescing free list stamped with it. Each process holds a shared
byte-range lock at an offset derived from the oldest epoch it still has
mappings or snapshots from (a *lease*); a region is reused once no lease in
any process is older than its stamp. A crashed process's leases go with its
locks. Regions being written outside the writer lock are recorded as
reserved, so a crash mid-write leaves space the next writer reclaims. When
the file grows, readers remap on their next search; old mappings stay valid
because regions never move.

`compact_file` rewrites the file without free space, as now.

What the user sees doesn't change: one file to copy, back up, or send.

**Re-rank from the map.** Each segment's region holds its full-precision
vectors in local-id order, so a re-rank read is an indexed memory load, not
a lookup. Segments walked with int8 codes store them as f16 (half the
space; rebuilds widen and narrow them losslessly); unquantized segments,
which walk them, keep f32. Result keys come from labels in the segment's
region (each row's document key) instead of two B-tree lookups per hit.

A reading process checks redb's header (one 320-byte read) before a search
and skips the snapshot, epoch lease and sync entirely when nothing was
committed since its last one; the read transaction for documents begins
only if the search needs one.

**Bulk loads.** A bulk-load handle, `db.bulk_load(&coll)`, takes rows
(a numpy buffer from Python, without per-row objects). Each `add` is one
transaction: documents as usual, and the batch's vectors written to a
region recorded in that same transaction, never through the `vectors`
table or the write buffer. No seals or merges run during the load;
`finish()` builds one graph over everything with every core, installs it as
a segment, and frees the batch regions.

- Durability: redb requires durable commits when other processes may share
  the file, so there every `add` is durable. A file opened with
  `open_exclusive` commits batches without fsync and durably every
  `checkpoint_every` rows (1M by default); a crash loses the load since the
  last durable commit and nothing before it.
- Rows are not searchable (in this or other processes) until `finish()`.
- Dropping the handle without `finish()`, or a crash, keeps what was
  committed: the next writer finishes it in the background.
- Ordinary transactions keep their guarantee: no acknowledged write lost.

**Faster graph builds.** Build on the int8 codes (int8 × int8 distances: a
quarter of the f32 build time on 200K real embeddings, the same recall at
every ef; quantize before building), insert every
vector once, use all cores whenever nothing is searching or the caller is
waiting, and fix the builder's parallel scaling (per-node locks, no shared
hot spots) until 16 threads keep 16 cores busy. Seals in streaming ingest
extend the largest segment instead of building small graphs to merge later;
regions make that cheap, since extending writes new code blocks and a new
graph, not a copy of the old codes.

**Smaller codes for the walk.** Phase 7 found quantization isn't what
limits recall, so the walk can use fewer bits. As built: 1-bit RaBitQ
codes (`Quantization::Binary`; random rotation, sign bits, three f32
factors per row, a 4-bit query compared by popcount), re-ranked from the
mapped f16 vectors at twice int8's depth. The graph is still built on int8
codes. On Cohere 1M they take the walked index from 918 MB to 254 MB at
about int8's throughput for the same recall, so they are an option, not the
default. 4-bit codes were skipped: 1-bit already shows the walk is not
bandwidth-bound on this machine.

**Streamed bulk builds.** A bulk load feeds its rows to a builder thread
as they commit; the quantizer is trained on the first 50K rows and every
later batch is inserted into the growing graph with all cores, so most of
the build overlaps the writes. Bulk checkpoints of quantized collections
are stored in f16, halving their writes and the space they leave free.

## Steps and gates

Each step ends with the full suite (unit, property, crash, corruption,
multi-process) and a VectorDBBench Cohere 1M run on this laptop, recorded in
`BENCHMARKS.md`. Recall@100 must stay ≥ 0.97 at the default setting and
≥ 0.985 at the high one through every step.

1. **Single-file regions and mapped segments.** Storage backend, arena map,
   regions, reader table, free list, `compact_file`; segments and re-rank
   vectors move into regions; re-rank and keys from the map.
   Gate: ≥ 1,800 QPS at 8 processes; peak memory per search process
   ≤ 1.0 GB; 8 processes ≤ 2 GB in total; no crash or corruption test
   regressions; a file written by Phase 7 upgrades on open.
2. **Bulk loads and the build.** Bulk-load handle and Python buffer API,
   int8 builds, every vector inserted once, full-core scheduling, builder
   scaling. Gate: load ≤ 225 s; a one-shot 1M × 768 graph ≤ 150 s.
3. **Query path.** Re-rank depth, f16 re-rank vectors, smaller page cache.
   Gate: ≥ 2,000 QPS at 8 processes; serial p50 ≤ 2 ms.
4. **Smaller codes.** 4-bit, then 1-bit codes. Gate: one search process
   ≤ 0.67 GB at peak, with recall held.

## Tests added

- **Crash:** kill points between region write, region sync, record commit
  and free-list update; between arena allocation and map sync; during a
  bulk load, before and after a checkpoint.
- **Corruption:** flipped bits in the arena map, region headers and region
  bodies (each region carries a checksum, checked when mapped).
- **Processes:** readers mapping while the writer merges, frees and grows
  the file; a reader killed while holding a slot; no region reused while a
  slot can still see it.
- **Fuzz:** the arena map and region headers.

## Risks

| Risk | Impact | Mitigation |
| --- | --- | --- |
| A storage-layout bug corrupts files | High | Regions immutable and checksummed; crash and corruption suites above; fuzzing; the arena map is the only mutable structure outside redb, and it is dual-copy |
| Region space reused while still mapped | High | Epoch-gated free list behind byte-range-locked reader slots; a dedicated multi-process test |
| Smaller codes cost recall on some models | Medium | Per-collection code size, int8 kept as the fallback; gated on Cohere and arxiv |
| Bulk-load rows lost on a crash | Low | Documented; checkpoints and resumable progress |
| Windows mapping semantics | Medium | Regions are never deleted, only reused once unmapped everywhere, which needs no file deletion |

## Results

VectorDBBench, Cohere 1M, this laptop (full table in `BENCHMARKS.md`):

| | Phase 7 | Phase 8 (int8, default) | Target | |
| --- | --- | --- | --- | --- |
| Load | 1,126 s | 209 s | ≤ 225 s | met (5.4×) |
| QPS, 8 processes | 993 | 2,484 warm | ≥ 2,000 | met (2.5×) |
| Serial p50 | | 2.0–2.1 ms | ≤ 2 ms | met at step 3, 2.1 ms at step 4 |
| Recall@100, default / ef 800 | 0.973 / 0.988 | 0.9717 / 0.9866 | ≥ 0.97 / ≥ 0.985 | met |
| Private memory, 1 search process | 2.0 GB | 37 MB | ≤ 0.67 GB | met |
| Private memory, 8 processes | ~16 GB | 299 MB | ≤ 1 GB | met |

Peak RSS per search process is 1.34 GB (0.94 GB with 1-bit codes): it
counts the file's mapped pages, which every process shares and the OS can
drop at will, so it is not memory the process owns. A search touches the
codes and graph (about 0.9 GB with int8, 0.25 GB with 1-bit) plus a few
hundred re-rank rows each.

What changed from the plan:

- **Dropping a load keeps it.** The plan said dropping the handle discards
  the load; instead its committed batches are finished in the background,
  which loses nothing a caller has been told was written.
- **Durability per add.** redb refuses non-durable commits when other
  processes may share the file, so only `open_exclusive` files batch their
  fsyncs.
- **A redb fork.** One public constructor added (`crates/vdblite-redb`).
- **The file keeps the bulk checkpoints' space.** It is freed and reused
  by later writes but the file doesn't shrink on its own: 4.5 GB after the
  load (3.8 GB for Zvec); `compact_file` takes it to 2.6 GB in 12.5 s.
- **Re-rank depth** stays k × 4 for int8: k × 2 loses recall at the
  default ef (0.938).
