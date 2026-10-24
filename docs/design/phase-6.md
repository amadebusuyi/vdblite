# Phase 6 — Background maintenance, reading processes, steady-state numbers

Phase 5 made a search index that lives in a file. A review of it as a
database found four gaps: commits stalled for graph builds, only one process
could open a file, the benchmarks measured only a settled index, and open
time at 768 dimensions was unknown. Phase 6 closes them.

## Background maintenance

**Commits don't seal.** A commit makes its data durable, applies its index
changes, and returns. Sealing the write buffer into a graph, and rebuilding
segments with many deletes, run on a background thread, in three steps:

1. Copy what's needed from the index under a read lock: the write buffer's
   vectors, or a compaction plan.
2. Build the graph with no lock held. Commits and searches go on.
3. Under the writer lock, write the segment in one transaction, then swap it
   into the index.

Writes that land during step 2 are reconciled in step 3. A chunk deleted
while its seal was building gets a tombstone row on disk and a tombstone in
memory (the generation check `install_sealed` already had). For compaction,
tombstones in the removed segments move to the replacement if it holds that
chunk (deleted after planning) and are dropped otherwise. Compaction reads
live vectors in batches of 4,096, so the thread never holds the file long.

**Builds get their own pool.** Searches fan out over segments on rayon's
global pool; background builds run in a separate pool with half the cores,
so a search never queues behind a graph build (they still share the CPU).
`optimize` runs on the caller's thread with every core, and no longer holds
the writer lock throughout: writes continue while it builds.

**Back-pressure.** If writes outpace sealing, the buffer grows. A commit that
leaves it over a full segment (4 × the seal threshold: 262,144 chunks, about
800 MB at 768 dims) seals it on the caller's thread before returning. That is
the only way a commit waits for a build.

**Closing.** The thread holds the database only in short spells. The last
handle waits out a spell in progress and stops the next, so the file closes
on the dropping thread and can be reopened at once. A build still running
then is discarded; its chunks are buffered on disk and sealed after the next
open.

**Errors.** Background failures are kept per collection: `IndexStats::
maintenance_error` shows the last one, `Db::wait_for_maintenance` returns it,
and the next commit retries. `commit` returns `Error::Maintenance` only when a
back-pressure seal fails (the data is committed either way).

`Db::wait_for_maintenance` blocks until background work has caught up; tests
and bulk loads use it to get a settled index.

## Reading processes

**One writer, any number of readers.** redb's `SingleWriter` concurrency mode
(the `experimental-multiprocess` feature) lets one process open a file for
writing and others open it read-only, coordinated with byte-range locks;
each new read transaction sees the writer's latest durable commit.
`Db::open` uses it; `Db::open_read_only` is the reader. The price: every
commit is two-phase (two disk flushes instead of one). `Db::open_exclusive`
keeps the old single-flush commits for applications that don't need readers.
Both modes write the same format.

**Readers replay a change log.** The vector index lives in memory, so a
reader's copy, loaded at open, would go stale. Each transaction that changes
an index appends what it did to the `changes` table, in the same
transaction: `Write { deletes, inserts }`, `Seal { segment }`, or
`Compact { removed, added }`. Before each search a reader applies the entries
newer than its last, from the same snapshot it hydrates results from, so its
index reflects at least that snapshot. Vectors and segments are read first;
the index is write-locked only to apply them.

**Falling behind.** The log keeps the newest 1,024 entries per collection. A
reader further behind, or one that meets a segment a later compaction has
already removed, reloads the collection from the file, as open does.

**Collections.** Readers re-read the (small) catalog in `Db::collection` and
`Db::collections` to pick up collections created or dropped by the writer;
`collections()` now returns a `Result`.

**Format version 2** adds the `changes` table. A writer upgrades a version 1
file in place when it opens it (only the table is created); readers read
version 1 files as they are; older builds refuse version 2 files rather than
write to them without the log.

**Opening no longer writes.** A writer opening a file already in the current
format skips the settings transaction, which had cost a commit (now two
flushes) on every open.

**Limits.**

- redb marks multi-process support experimental. vecfile's own tests cover
  readers in the same process and in another one, including the writer
  being killed and a new writer recovering the file while a reader stays
  open.
- A reader can't open a file whose writer crashed until a writer has opened
  (and repaired) it.
- Read-only processes skip redb's corruption check at open, which can't run
  while another process may be writing.

## Open time

**Still a copy, not a memory map.** Segment arrays live in redb values, and
redb doesn't expose a value's place in the file. Reading them in place would
need an aligned region of our own outside redb: the second pager Phase 5
decided against. So open still copies each segment into memory, but once
instead of twice (straight from the stored value into the typed array), and
segments load in parallel.

## Steady-state benchmark

`crates/vecfile/examples/steady_state.rs` measures what an application sees
while writing and searching at once, through the public API: commit latency
during bulk ingest, open time and resident memory, search on the index as
background maintenance left it and after `optimize` (recall against exact
search), filters including numeric ranges over a unique field, hybrid
search, search with the write buffer filling, search and commits during a
seal, single-document commits by collection size and open mode, and a
reading process following the writer.

Real embeddings come from `scripts/fetch-arxiv768.sh`: 1M InstructorXL
embeddings (768 dims) of arXiv paper titles, with the titles as document
text for hybrid search. `--synthetic` uses the generator from the Phase 5
headline run instead.

## Gate result

See `BENCHMARKS.md`, 2026-10-24.
