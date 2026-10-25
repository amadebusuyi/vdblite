# vecfile

An embedded, single-file vector database in Rust. Open it like SQLite, store
documents, and search them by meaning: no server.

```rust
use vecfile::{Agg, Chunking, Db, Doc, Metric, Quant, Schema, meta};

let db = Db::open("notes.vdb")?;
let docs = db.create_collection("docs", Schema {
    dim: 768,
    metric: Metric::Cosine,
    quant: Quant::Int8,
    chunking: Chunking::Structural { target_tokens: 400, overlap_tokens: 50 },
})?;

let mut tx = db.begin_write()?;
tx.upsert(&docs, Doc::new("readme-42").content(text).meta("lang", "en"), &embedder)?;
tx.commit()?;

let hits = docs.search(&query_vec).k(10).filter(meta("lang").eq("en")).chunks()?;
let top_docs = docs.search(&query_vec).k(5).documents(Agg::MaxSim)?;
let full = docs.get("readme-42")?;
```

**Node.js:** see [bindings/node](bindings/node/README.md) (`npm run build`,
then `npm install` the folder into your project). For a working app, see
[examples/notes-search](examples/notes-search): semantic search over a folder
of notes with a local embedding model.

`embedder` is anything implementing `vecfile::Embedder`, including a closure
`|texts: &[&str]| -> Vec<Vec<f32>>`. Or pass vectors directly with
`upsert_with_vectors`.

## What it does

- **One file.** Documents, metadata, vectors and index live in one `.vdb`
  file (on [redb](https://github.com/cberner/redb): ACID, checksummed,
  crash-safe). One writing process (`Db::open`); any number of other
  processes can read it alongside (`Db::open_read_only`) and see each commit
  as it lands.
- **Commits don't wait for the index.** Sealing new chunks into graphs and
  rebuilding segments after deletes run in the background; searches and
  writes carry on meanwhile.
- **Structural chunking.** Markdown, HTML, code and plain text split on
  headings, paragraphs, code blocks and definitions. Chunks are byte ranges, so
  a document rebuilds byte for byte. Edits re-embed only the chunks they touch.
- **Four result shapes.** `chunks()` (passages with offsets), `expand(n)`
  (hits plus neighbours, merged), `section()` (the whole heading section),
  `documents(agg)` (documents ranked by their best chunks).
- **Hybrid search.** BM25 keyword matching fused with vector similarity
  (`.text(query)`), so exact identifiers and rare names are found too.
- **Metadata filters.** Equality, sets, numeric ranges, existence, and/or/not,
  over Roaring bitmaps.
- **Memory-lean HNSW.** int8 codes for the graph walk, f32 re-ranking from
  the file, SIMD kernels (AVX2, NEON) chosen at runtime, zero allocation per
  segment search.

## Performance

On an 8-core Intel i9 laptop (AVX2), SIFT1M (1M x 128):

| SIFT1M, ef=64, warm | recall@10 | p50 | p99 |
| --- | --- | --- | --- |
| Full API, 1M documents in a `.vdb`, incl. reading results from the file | 0.992 | 355 µs | 488 µs |

On 1M real 768-dim text embeddings (arXiv titles), one graph, 16 threads:

| ids only | recall@10 | QPS |
| --- | --- | --- |
| ef=64 | 0.938 | 17,081 |
| ef=128 | 0.969 | 9,319 |
| ef=256 | 0.986 | 3,885 |

The resident index is about 920 MB (int8 codes and graph; f32 vectors stay
in the file). Collections that need recall of 0.99 or more do better with
`hnsw.m = 32`. Full results, including filtered search and the caveats,
are in [BENCHMARKS.md](BENCHMARKS.md).

## Layout

| Crate | Contents |
| --- | --- |
| `vecfile` | Public API: `Db`, collections, transactions, search |
| `vecfile-core` | File layout, records, codecs |
| `vecfile-index` | Distance kernels, HNSW, quantization, segments |
| `vecfile-chunk` | Format detection and structural chunking |
| `bindings/node` | Node.js addon (napi-rs), JS API and TypeScript types |

Design notes per phase are in [docs/design/](docs/design/).

## Development

```sh
cargo test --workspace --release          # unit, property, crash and corruption tests
scripts/fetch-sift1m.sh                    # benchmark data (~160 MB)
cargo run --release -p vecfile-index --example sift_recall    -- data/sift
cargo run --release -p vecfile-index --example hnsw_sift      -- data/sift
cargo run --release -p vecfile-index --example segments_sift  -- data/sift --filters
cargo run --release -p vecfile-index --example synthetic768
cargo run --release -p vecfile       --example sift_db        -- data/sift
scripts/fetch-arxiv768.sh                  # 1M real 768-dim embeddings (~3.2 GB download)
cargo run --release -p vecfile       --example steady_state   -- --data data/arxiv768
cargo bench -p vecfile-index --bench distance --bench brute_force
```

Fuzz targets (nightly): `cd fuzz && cargo +nightly fuzz run chunk`.
