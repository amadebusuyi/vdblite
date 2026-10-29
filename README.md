# VDBLite

An embedded, single-file vector database in Rust. Open it like SQLite, store
documents, and search them by meaning: no server.

```sh
cargo add vdblite                                  # Rust
npm install vdblite                                # Node.js / TypeScript
go get github.com/OWNER/vdblite/bindings/go        # Go
dotnet add package VDBLite                         # C# / .NET
```

Also for Java and Kotlin (Maven Central), Swift (SwiftPM) and C/C++
(prebuilt libraries); see [Languages](#languages). Every package ships
prebuilt native code, so none of them needs a Rust toolchain.

```rust
use vdblite::{Agg, Chunking, Db, Doc, Metric, Quant, Schema, meta};

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

For a working app, see [examples/notes-search](examples/notes-search)
(Node.js): semantic search over a folder of notes with a local embedding
model.

`embedder` is anything implementing `vdblite::Embedder`, including a closure
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
- **Memory-lean HNSW.** int8 collections walk the graph on int8 codes and
  re-rank from f16 vectors; unquantized collections walk f32 vectors. Both
  are memory-mapped from the file and shared across processes. SIMD kernels
  (AVX2, NEON) chosen at runtime, zero allocation per segment search.

## Languages

| Language | Package | Docs |
| --- | --- | --- |
| Rust | [`vdblite`](https://crates.io/crates/vdblite) on crates.io | [docs.rs](https://docs.rs/vdblite) |
| JavaScript / TypeScript | [`vdblite`](https://www.npmjs.com/package/vdblite) on npm | [bindings/node](bindings/node/README.md) |
| Python | `vdblite-py` (vectors in, keys out; for now built from source) | [bindings/python](bindings/python) |
| Go | `github.com/OWNER/vdblite/bindings/go` | [bindings/go](bindings/go/README.md) |
| Java, Kotlin | `io.github.OWNER:vdblite` on Maven Central | [bindings/java](bindings/java/README.md) |
| C#, F# (.NET) | [`VDBLite`](https://www.nuget.org/packages/VDBLite) on NuGet | [bindings/dotnet](bindings/dotnet/README.md) |
| Swift (macOS, iOS) | `github.com/OWNER/vdblite-swift` with SwiftPM | [bindings/swift](bindings/swift/README.md) |
| C, C++ | archives on [GitHub releases](https://github.com/OWNER/vdblite/releases) | [bindings/c](bindings/c/README.md) |

Every language reads and writes the same `.vdb` files and speaks the same
JSON shapes for metadata, filters and results. Go, Java, .NET, Swift and
C++ are thin layers over one C API ([vdblite.h](bindings/c/include/vdblite.h)).

## Performance

[VectorDBBench](https://github.com/zilliztech/VectorDBBench) on Cohere 1M
(768 dims, cosine, top 100), on an 8-core Intel i9 laptop (AVX2), through the
Python binding with one search process per client. Zvec, another embedded
engine, ran on the same machine with the same graph settings:

| Cohere 1M | VDBLite (int8, default) | VDBLite (1-bit) | Zvec |
| --- | --- | --- | --- |
| Load (insert + optimize) | 209 s | 222 s | 194 s |
| QPS, 8 processes (warm) | 2,484 | 1,810 | 2,195 |
| Serial p50 / p99 | 2.1 / 2.7 ms | 2.5 / 3.7 ms | 2.1 / 10.0 ms |
| Recall@100 | 0.9717 | 0.9835 | 0.9746 |
| ef 800: QPS at 8 / recall | 1,472 / 0.9866 | | 1,329 / 0.9875 |
| Private memory per search process | 37 MB | 40 MB | |
| File size after the load (after `compact_file`) | 4.5 GB (2.6 GB) | 3.9 GB | 3.8 GB |

The walked index (int8 codes and graph) is about 920 MB, memory-mapped from
the file and shared by every process that opens it, so 8 search processes
use 299 MB of private memory between them. f16 re-rank vectors stay in the
file. With 1-bit codes (`Quant::Binary`) the walked index is about 254 MB.

Earlier in-process measurements (before the Phase 8 storage changes), SIFT1M
(1M x 128), full API, ef=64, warm: recall@10 0.992, p50 355 µs, p99 488 µs.
Collections that need recall of 0.99 or more do better with `hnsw.m = 32`.
Full results, including filtered search and the caveats, are in
[BENCHMARKS.md](BENCHMARKS.md).

## Compatibility

- **Rust:** 1.95 or newer.
- **Node.js:** 18 or newer. Prebuilt addons for macOS (arm64, x64), Linux
  (x64, arm64; glibc 2.28+ or musl/Alpine) and Windows (x64, arm64). Other
  platforms build from source with `npm run build` in
  [bindings/node](bindings/node).
- **Go:** 1.22 or newer with cgo; macOS (arm64, amd64), Linux (amd64, arm64;
  glibc 2.28+), Windows amd64 (MinGW).
- **Java:** 17 or newer; macOS, Linux (glibc 2.28+), Windows, on x64 and arm64.
- **.NET:** 8 or newer; macOS, Linux (glibc 2.28+ or musl), Windows, on x64
  and arm64.
- **Swift:** 5.9 or newer; macOS 12+ and iOS 15+ (devices and simulator).
- **C/C++:** C99 / C++17; the same platforms as Node.js, plus Windows MinGW.
- **File format:** each `.vdb` records its format version. A newer VDBLite
  upgrades an older file in place the first time a writer opens it
  (`Db::open`); until then, read-only handles get `Error::Version`, so open
  it for writing once after upgrading. An older VDBLite refuses a newer file
  with an error rather than misreading it. Upgrade the application before
  copying newer files to it.
- **API:** while the version is 0.x, minor releases may change the API;
  [CHANGELOG.md](CHANGELOG.md) lists what changed.

## Deploying

VDBLite runs inside your application, so you deploy the application with the
`.vdb` file on a disk it can reach:

- Put the file on a local disk or a block volume attached to the machine (an
  EBS or Persistent Disk volume, a Fly.io or Railway volume, a VPS disk).
  Every process that opens the file memory-maps its sealed index segments,
  and memory maps and byte-range locks are unreliable over network file
  systems (NFS, EFS, SMB, Azure Files), so don't use those. Serverless
  functions don't keep their disk.
- One process writes a file at a time; other processes on the same machine
  can open it read-only. Scale up rather than out.
- Leave RAM for the OS page cache beyond the resident index: re-ranking reads
  vectors from the file.
- Back up by copying the file while no write is in progress, or a volume
  snapshot. `Db::compact_file` shrinks a closed file.

## Layout

| Crate | Contents |
| --- | --- |
| `vdblite` | Public API: `Db`, collections, transactions, search |
| `vdblite-core` | File layout, records, codecs |
| `vdblite-index` | Distance kernels, HNSW, quantization, segments |
| `vdblite-chunk` | Format detection and structural chunking |
| `bindings/node` | Node.js addon (napi-rs), JS API and TypeScript types |
| `bindings/c` | C API (`vdblite.h`), C++ wrapper, shared and static libraries |
| `bindings/go`, `bindings/java`, `bindings/dotnet`, `bindings/swift` | Language packages over the C API |
| `bindings/shared` | JSON conversions (metadata, filters, results) shared by Node and the C API |

Design notes per phase are in [docs/design/](docs/design/).

## Development

```sh
cargo test --workspace --release          # unit, property, crash and corruption tests
scripts/fetch-sift1m.sh                    # benchmark data (~160 MB)
cargo run --release -p vdblite-index --example sift_recall    -- data/sift
cargo run --release -p vdblite-index --example hnsw_sift      -- data/sift
cargo run --release -p vdblite-index --example segments_sift  -- data/sift --filters
cargo run --release -p vdblite-index --example synthetic768
cargo run --release -p vdblite       --example sift_db        -- data/sift
scripts/fetch-arxiv768.sh                  # 1M real 768-dim embeddings (~3.2 GB download)
cargo run --release -p vdblite       --example steady_state   -- --data data/arxiv768
cargo bench -p vdblite-index --bench distance --bench brute_force
```

Fuzz targets (nightly): `cd fuzz && cargo +nightly fuzz run chunk`.

Language bindings: `bindings/c/scripts/test.sh` (C, C++);
`node bindings/c/scripts/ffi.mjs dev` builds the C library and places it for
Go (`go test ./...` in bindings/go), Java (`mvn test` in bindings/java) and
.NET (`dotnet test tests/VDBLite.Tests` in bindings/dotnet); Swift:
`bindings/swift/scripts/build-xcframework.sh --host`, then `swift test`.

## Releasing

```sh
node scripts/release/version.mjs bump 0.2.0   # every version number, and Cargo.lock
# add a "## [0.2.0]" section to CHANGELOG.md
git commit -am "Release 0.2.0" && git tag v0.2.0 && git push --follow-tags
```

The tag starts [.github/workflows/release.yml](.github/workflows/release.yml):
CI, prebuilt addons for every platform (each tested on that platform), then
crates.io and npm, a GitHub release, and an install check from the registries.
See [RELEASING.md](RELEASING.md) for the one-time setup.

## License

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE). Unless you
explicitly state otherwise, any contribution you submit for inclusion is
licensed as above, without additional terms.
