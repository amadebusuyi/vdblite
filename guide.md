# Embedded Vector Store (Rust) — Engineering Design

Oct 23, 2026 · @Adebowale

## Goals and non-goals

We are building a single-file, embedded vector database in Rust (working name `vdblite`). An app opens it like SQLite: `Db::open("notes.vdb")`, with no server. It stores documents like a regular database, splits large content into searchable chunks, and returns whole documents or the exact parts that matched.

**Goals**

- **One file, embedded.** A library crate that keeps all data in a single file, with a lock file and WAL beside it during writes. Bindings for Python and Node come later.
- **Records with structure.** Collections of documents with an ID, typed metadata, raw content and one or more embeddings.
- **Smart chunking.** Large content is split by structure (headings, paragraphs, code blocks). The parent document can be rebuilt byte for byte from its chunks.
- **Two retrieval modes.** Return the best chunks with their offsets in the parent, or return whole parent documents ranked by their best chunks.
- **Fast and lean.** Search over 1M chunks of 768 dimensions in under 5 ms at p99, with recall@10 of at least 0.95, using about 1 GB of RAM or less.
- **Durable.** No acknowledged write is lost after a crash.

**Non-goals for v1**

- No network server, replication or sharding.
- No SQL parser. Queries go through a typed Rust API.
- No built-in embedding model. Callers pass vectors in, or plug in an `Embedder` trait.
- One writer at a time, many readers (the SQLite model).

## Architecture overview

The system is one crate with four layers. Writes run left to right: chunk, embed, store. Reads enter the query planner, which narrows candidates with the indexes before any text is read.

&#91;embedded content: architecture · API, 3 services, 3 engines, 1 file\]

The Embedder is dashed because it sits outside the core: callers pass vectors in, or plug in a model. Suggested workspace crates: `vdblite-core` (storage and WAL), `vdblite-index` (HNSW and quantization), `vdblite-chunk`, and `vdblite` (the public API).

## Data model and public API

The model has three levels: a **collection** holds **documents**, and each document owns ordered **chunks**. Only chunks carry vectors; documents carry content and metadata.

| Entity | Key | Fields | Notes |
| --- | --- | --- | --- |
| Collection | name | dim, metric (cosine / dot / L2), quantization, chunking policy, index params | Fixed at creation, stored in the catalog |
| Document | `doc_id` (u64, or a user string mapped to u64) | metadata (typed map), content blob, content hash, version, created / updated time | Content stored once, compressed with zstd |
| Chunk | `chunk_id` (dense u32 inside its segment) | `doc_id`, ordinal, byte range \[start, end) in the parent, heading path, token count, vector | Text is a slice of the parent; it is never copied |

Chunks point into the parent by byte range, so storage holds each byte once. Rebuilding the whole document is a single blob read.

```rust
let db = Db::open("notes.vdb")?;
let docs = db.create_collection("docs", Schema {
    dim: 768,
    metric: Metric::Cosine,
    quant: Quant::Int8,
    chunking: Chunking::Structural { target_tokens: 400, overlap_tokens: 50 },
})?;

// Write: content is chunked; the embedder (or caller-supplied vectors) fills each chunk
let mut tx = db.begin_write()?;
tx.upsert(&docs, Doc::new("readme-42").content(text).meta("lang", "en"), &embedder)?;
tx.commit()?;

// Read: part-level search
let hits = docs.search(&query_vec).k(10).filter(meta("lang").eq("en")).chunks()?;
// Read: whole documents ranked by their best chunks
let top_docs = docs.search(&query_vec).k(5).documents(Agg::MaxSim)?;
let full = docs.get("readme-42")?;   // plain lookup by key
```

Core operations: `upsert`, `delete`, `get`, `scan` (by metadata filter), `search().chunks()`, `search().documents()`, `search().expand(n)` (a hit plus n neighbouring chunks).

## Chunking engine

Chunking splits on structure first and size second. Every chunk remembers where it sits in the parent, so any hit can be widened back to its section or to the whole document.

**Pipeline**

1. **Detect the format** from a hint or by sniffing: Markdown, source code, HTML, or plain text.
2. **Parse into a section tree.** Markdown uses headings; code uses functions and classes (tree-sitter, behind a feature flag); plain text uses blank-line paragraphs and sentences.
3. **Pack greedily.** Walk the tree and merge sibling blocks until the target size (default 400 tokens) is reached. Never split mid-sentence or mid-code-block unless one block alone exceeds the hard cap (default 800 tokens).
4. **Add overlap.** Prepend about 50 tokens from the previous chunk. Overlap is stored as a range, not as copied text.
5. **Add context.** Each chunk keeps its heading path (for example `Guide > Install > Linux`). The path is embedded together with the chunk text, which improves recall, but it is not stored twice.
6. **Record the range** `[start, end)` and the ordinal.

**Retrieval shapes this enables**

| Mode | Returns | Use it for |
| --- | --- | --- |
| `chunks()` | Matching chunks with doc ID, offsets and score | Precise passages, citations |
| `expand(n)` | Each hit plus n chunks either side, merged when they touch | Context for an LLM prompt |
| `section()` | The whole heading section the hit belongs to | Docs and manuals |
| `documents(agg)` | Parent documents ranked by MaxSim, or the sum of their top-3 chunk scores | "Which files are about X" |

**Updates.** Each chunk is hashed. When a document changes, only chunks whose hash changed are re-embedded, which saves most of the embedding cost on small edits.

Token counts use a pluggable tokenizer trait; the default counts approximate tokens as bytes / 4.

## Single-file storage format

The `.vdb` file is a set of fixed 4 KB pages addressed by page number, read through mmap. Writes go to a write-ahead log (`.vdb-wal`) first and are copied into the main file at checkpoint, the same model SQLite uses in WAL mode.

| Region | Contents | Layout |
| --- | --- | --- |
| Header (page 0) | Magic bytes, format version, page size, catalog root, free-list root, checksum | 4 KB |
| Catalog | Collections and their schemas; the segment list per collection | B-tree pages |
| Document store | Doc ID to (metadata, content blob pointer, version) | B+tree keyed by u64 |
| Blob pages | zstd-compressed document content; large blobs span overflow pages | Page chains |
| Chunk table | Per chunk: doc ID, ordinal, byte range, hash | Fixed 32-byte rows, columnar |
| Vector pages | Quantized vectors (int8), and full f32 vectors for re-ranking | Contiguous, 64-byte aligned |
| Graph pages | HNSW adjacency in fixed slots | `u32` arrays (see the index section) |
| Metadata indexes | Inverted index: field value to a Roaring bitmap of chunk IDs | B-tree of bitmaps |

**Segments.** New writes land in a small mutable segment. When it reaches about 64K chunks, it is sealed: its index is built and written as immutable pages. Search runs over all segments and merges the results. Background compaction merges small segments and drops deleted rows.

**Durability.** Every page carries a CRC32C checksum. A commit appends its frames to the WAL and calls fsync. Recovery replays committed frames and discards a torn tail.

**Concurrency.** One writer holds a file lock. Readers take a snapshot of the latest committed WAL frame and never block the writer.

**Build or borrow?** Writing the pager is the costliest part of the project. Option A: a custom pager (full control over vector alignment). Option B: [redb](https://github.com/cberner/redb) for documents and metadata, plus our own aligned region for vectors and the graph. Option B is recommended for v1.

## Vector index: memory-lean HNSW

Each sealed segment has its own HNSW graph, stored as flat `u32` arrays that are mmap'd straight from the file. In-memory vectors are int8; full f32 vectors stay on disk and are touched only for re-ranking.

```rust
struct SegmentIndex<'a> {
    level0: &'a [u32],   // N slots of [count, n1..n_2M], fixed width, mmap'd
    upper_off: &'a [u32], // offsets for nodes with level > 0 (about N/M of them)
    upper: &'a [u32],     // packed neighbour lists for the upper layers
    levels: &'a [u8],
    codes: &'a [i8],      // N x dim int8 vectors, 64-byte aligned
    scales: &'a [f32],    // per-dimension scale and offset for int8
    full: &'a [f32],      // f32 vectors on disk, used to re-rank only
    entry: u32,
}
```

**Design choices**

- **Dense u32 IDs inside each segment.** External doc and chunk IDs live in the chunk table. This halves graph memory compared with u64.
- **Neighbour selection heuristic** from the HNSW paper (diversity pruning). Defaults M = 16, efConstruction = 200, efSearch = 64, all tunable per collection.
- **int8 scalar quantization** with per-dimension min and max. The graph walk uses int8 distances; the top `k × 4` results are re-ranked with f32.
- **Visited set** as a reusable `u16` epoch array per thread, so searches allocate nothing.
- **Deletes** mark a tombstone bit. Tombstoned nodes still route the search but never appear in results. Compaction rebuilds the graph once a segment passes 20% tombstones.
- **Filtered search.** The metadata bitmap is checked during the walk. When a filter matches less than about 2% of a segment, brute-force scan of the matching rows instead.
- **Experiments for later:** a flat graph with no upper layers, BFS reordering of node IDs for cache locality, and binary or RaBitQ codes.

**Memory estimate, 1M chunks at 768 dimensions:** int8 vectors 768 MB, graph about 140 MB, chunk table 32 MB. That is about 0.95 GB resident, against about 3.3 GB for plain f32 HNSW.

## Query path and hybrid retrieval

A search fans out over the segments in parallel, merges their candidates, re-ranks them with full-precision vectors, and only then reads any text. Text and blobs are never touched until the final top-k is known.

1. **Plan.** Resolve the metadata filter to a Roaring bitmap per segment. Choose a graph walk or brute force from the filter's selectivity.
2. **Fan out.** Search each segment in parallel with rayon: the mutable segment by brute force, sealed segments by HNSW over int8 codes.
3. **Merge** into one bounded heap of `k × 4` candidates.
4. **Re-rank** with f32 vectors read from mmap.
5. **Shape the result.** For `chunks()`, return as is. For `documents()`, group by doc ID and score each document by MaxSim. For `expand()` or `section()`, widen using the chunk ranges.
6. **Hydrate.** Read metadata and slice text out of the parent blob, only for the final k results.

**Hybrid search (v1.1).** Add a BM25 inverted index over chunk text. Run it alongside the vector search and merge the two lists with Reciprocal Rank Fusion (k = 60). This fixes the usual misses on exact names, error codes and identifiers.

## Performance strategy

Speed comes from three places: SIMD distance kernels, zero-copy reads through mmap, and zero allocation on the query path. Every hot path gets a criterion benchmark before it gets optimized.

| Area | Technique | Crate or tool |
| --- | --- | --- |
| Distance | Hand-written AVX2, AVX-512 and NEON kernels for int8 and f32, selected once at startup | `std::arch`, or `simsimd` |
| Memory layout | Vectors 64-byte aligned and contiguous; casts with no copy | `bytemuck`, `zerocopy` |
| I/O | mmap reads; `madvise(WILLNEED)` on graph pages; sequential WAL appends | `memmap2` |
| Allocation | Per-thread scratch (heaps, visited set) reused across queries; arena allocation during index builds | `bumpalo` |
| Parallelism | Segments searched in parallel; index builds in parallel with striped locks | `rayon`, `parking_lot` |
| Prefetch | Prefetch neighbour vectors one hop ahead during the graph walk | `_mm_prefetch` |
| Compression | zstd for document blobs; Roaring for metadata bitmaps | `zstd`, `roaring` |
| Build | `lto = "fat"`, `codegen-units = 1`, `panic = "abort"`, `target-cpu` set per release | Cargo profile |

For profiling, use `cargo flamegraph`, `perf stat` (cache misses per query) and `dhat` (allocations per query, which should be zero).

## Engineering workflow

The build runs in six phases over about 13 weeks of focused work. Each phase ends at a gate with a measured pass condition. The index comes before storage on purpose: recall and speed get proven in memory before they are frozen into a file format.

&#91;embedded content: engineering workflow · 6 phases, 6 exit gates\]

Week counts are planning estimates for one developer. Each phase works the same way:

1. **Design note.** Write a short note per phase on the doc's open decisions, before coding.
2. **Test first.** Write the correctness test (against brute force, or byte-exact rebuild) before the optimization.
3. **Simple, then fast.** Build the plain version, record a benchmark, then optimize against that number.
4. **Gate review.** Run the full benchmark suite and record results in a `BENCHMARKS.md` log. Move on only when the gate passes.

**Phase 0 starter checklist**

- [ ] Create the Cargo workspace with the four crates named above
- [ ] Download SIFT1M and its ground truth; write the loader
- [ ] Write scalar distance functions plus proptests
- [ ] Add AVX2 and NEON kernels with runtime dispatch
- [ ] Write brute-force top-k and the recall@k harness
- [ ] Set up CI: tests, clippy, benchmarks on main

## Testing and benchmarking

Three kinds of check guard the project: correctness against brute force, crash safety, and performance tracked over time. A change that lowers recall or raises p99 latency by more than 5% fails CI.

| Kind | What | How |
| --- | --- | --- |
| Unit | Distance kernels match the scalar code within 1e-5; chunk ranges rebuild the parent byte for byte | `proptest` |
| Recall | recall@10 compared with exact brute-force search | SIFT1M, GloVe-100, plus a 768-dim text-embedding set |
| Crash | Kill the process at random points during writes; the reopened file passes its checksums and holds every committed write | A fault-injection harness |
| Fuzz | File parser and WAL replay given corrupted bytes | `cargo-fuzz` |
| Concurrency | Readers alongside a writer; no torn reads | `loom` on the core primitives |
| Performance | QPS, p50 / p99 latency, resident memory, index build time, file size | `criterion`, plus a nightly run of the ann-benchmarks harness |
| Chunking quality | Retrieval hit rate on a small labelled question set from your own documents | A custom eval script |

## Risks and open decisions

The biggest risk is scope: a custom pager plus an index plus chunking is several projects in one. Borrowing redb for the record store and keeping the index as our own region cuts that risk the most.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Custom pager bugs corrupt data | High | Use redb in v1; checksum every page; crash-test in CI |
| Deletes and updates degrade the HNSW graph | Medium | Tombstones plus rebuild at 20%; segments keep rebuilds small |
| int8 quantization loses recall on some embedding models | Medium | Re-rank with f32; keep an f32-only mode per collection |
| Strict filters return too few results | Medium | Fall back to brute force under about 2% selectivity |
| Chunking settings don't suit every content type | Medium | Policy set per collection; tune with the chunking eval |

**Open decisions**

- [ ] Custom pager or redb for documents and metadata (recommended: redb).
- [ ] Keep upper HNSW layers, or try a flat graph after Phase 3 benchmarks.
- [ ] Default embedding dimension and model for your own use (for example a 384-dim or 768-dim model).
- [ ] Ship a bundled embedder (ONNX) behind a feature flag, or vectors from callers only.
- [ ] Which bindings come first: Python (PyO3) or Node (napi-rs).
