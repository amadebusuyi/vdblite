# Phase 5 — Single-file storage, public API, durability

Phase 5 freezes the proven index into a file and puts the guide's API on top.

## Open decision: custom pager or redb?

**redb (Option B), and further than the guide suggested.** The guide
recommended redb for documents and metadata plus "our own aligned region for
vectors and the graph". Two regions in one file means writing a pager anyway,
which is the risk the guide most wanted to avoid. So everything lives in redb
tables, including sealed segments (one value per array) and f32 vectors.

What this costs, against the guide's design:

| Guide | Here | Effect |
| --- | --- | --- |
| mmap'd graph and codes, zero-copy | segments copied into memory at open | 1M x 128 opens in ~0.4 s; resident set is the same (codes + graph were resident anyway) |
| f32 vectors read through mmap for re-ranking | 40 B-tree lookups per query through redb's page cache | warm p99 stays under 0.5 ms at 1M |
| WAL beside the file | redb's copy-on-write commits with a checksummed dual header | same guarantee (no acknowledged write lost), no WAL file |
| SQLite-style multi-process readers | one process holds the file lock; threads share it | a second process gets an error instead of a reader |

The mmap path can come back later without changing the API: segment arrays
are plain little-endian buffers, so an aligned region (or a redb version that
exposes aligned values) can be read in place.

## Decisions

**Tables, keyed collection-first.** One set of tables serves every
collection; each key starts with the collection's `u32` id (see
`vdblite-core/src/tables.rs`). Dropping a collection is range deletes.

**What is persisted vs rebuilt.** Persisted: documents, zstd blobs, chunk
records, f32 vectors, the metadata index, sealed segments, tombstones (`dead`:
chunk to segment), and which chunks are still buffered (`pending`). At open,
segments load, tombstones apply, and buffered chunks' vectors are re-inserted
into the mutable segment. Nothing needs a rebuild after a crash.

**Commit, then apply.** A write transaction writes records and vectors, then
commits; only then are index changes applied in memory. Readers take the
index read lock and a redb snapshot, and hydrate every result from that one
snapshot: a chunk deleted after the index was read is skipped, never shown
with the wrong text.

**Sealing and compaction run after commit**, under the writer lock but
outside the index lock (graph builds don't block searches), each in its own
transaction. A crash between the data commit and a seal leaves the chunks in
`pending`; they're re-buffered at open and sealed later. If maintenance fails
after a successful data commit, `commit` returns `Error::Maintenance`: the data
is safe, and maintenance retries on the next commit.

**Failed writes poison the transaction.** Every validation (dimensions,
NaN, field names, vector counts, id space) runs before an upsert writes
anything, so a rejected upsert leaves the transaction usable. A storage error
can strike between writes, so it marks the transaction poisoned and `commit`
refuses it.

**Incremental re-embedding.** Chunks are matched to the previous version by
the hash of their embedding input; matches keep their chunk id, vector and
index entry. Metadata-only changes re-embed nothing.

**Filters are evaluated twice, consistently.** Search turns a `Filter` into a
Roaring bitmap from the inverted index (numbers share one order-preserving
encoding, so ranges are one key range). `scan` evaluates the same `Filter`
against each document's metadata. A test checks both give the same documents
for every filter shape.

**Validation everywhere data comes off disk.** Every array length, graph
invariant (including that upper-layer neighbours live on that layer) and
record decode is checked; corrupt data is `Error::Storage(Corrupt)`.

## Tests

- **Crash:** the test binary re-runs itself as a writer and is SIGKILLed at
  10 points, including mid-commit and mid-seal. After each kill, every commit
  it reported must be present, deletes must hold, and the index must match the
  records exactly.
- **Corruption:** copies of a real file with 1–8 flipped bits are opened,
  searched, read and scanned. redb itself panics on some of them (while
  loading allocator state at open, parsing pages, or flushing at close), so
  VDBLite catches panics at `Db::open`, every read entry point, and when the
  handle is dropped, turning them into `Error::Storage`. CI runs 120 rounds
  with no escaped panic. A 1,000-round run found one file (about 1 in 400)
  where redb panics again inside a destructor while unwinding, which aborts
  the process; nothing outside redb can contain that. That file is kept as an
  ignored test (`redb_double_panic_reproduction`) for an upstream report.
  Containment needs an unwinding build: with `panic = "abort"`, any of these
  panics ends the process. This concerns damaged files (bit rot, tampering),
  not crashes: crash safety above is a separate guarantee.
- **Concurrency:** four reader threads search and hydrate continuously while
  the writer commits, deletes and seals. The guide's `loom` suggestion targets
  hand-written lock-free primitives; there are none here (the shared state is
  `parking_lot::RwLock` plus redb MVCC), so loom has nothing to check.
- **Fuzzing:** cargo-fuzz targets in `fuzz/` (chunker, segment decoding,
  whole-file open) for nightly.

## Hybrid search (the guide's v1.1), added after v1

A BM25 keyword index sits beside the vectors (`postings`, `term_df`,
`chunk_terms`, `lexical` tables), written in the same transaction as each
chunk. `Search::text(query)` runs BM25 next to the vector search and fuses the
two rankings with Reciprocal Rank Fusion (k = 60). Each list contributes
`max(2n, 40)` candidates so chunks ranked moderately by both can surface.

- **Tokenizer:** lowercase words of letters, digits and `_`, minus a short
  English stopword list. Identifiers also index their parts: `SealedSegment`
  → `sealedsegment`, `sealed`, `segment`; `MAX_DIM` → `max_dim`, `max`, `dim`.
  Parts count half as much as whole words, so "info" ranks an `info()` method
  above `ChunkInfo` in the keyword list.
- **Ties:** equal fused scores (rank 1 in one list, 2 in the other) are broken
  by agreement (found by both), then keyword score, then similarity.
- **Index versioning:** the `lexical` stats carry a layout version; an index
  in an older layout is ignored until `enable_lexical()` rebuilds it from
  stored text (no re-embedding).
- **What's indexed:** each chunk's heading path plus its own bytes, not its
  overlap, which belongs to the previous chunk.
- **Scores:** a hybrid hit's `score` is the fused rank score; `vector_score`
  and `lexical_score` give the absolute values, and either is absent if that
  list didn't find the chunk.
- **Existing files:** collections created before this (or with `lexical:
  false`) are vector-only until `Collection::enable_lexical()` builds the index
  from stored text, with no re-embedding.
- **Chunker versioning:** a document's content fingerprint is seeded with
  `vdblite_chunk::VERSION`, so a chunker change re-chunks every document on its
  next write while unchanged chunks keep their vectors.

## Not in v1 (as the guide planned)

Python bindings, a bundled ONNX embedder. (Node bindings now exist in
`bindings/node`.)
`Db::compact_file` shrinks the file offline; online space reclamation is
redb's page reuse.
