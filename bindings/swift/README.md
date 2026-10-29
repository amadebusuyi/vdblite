# VDBLite for Swift

An embedded, single-file vector database: no server, one `.vdb` file. Made
for on-device search on macOS and iOS: the core is Rust, shipped as an
XCFramework, so there is nothing to compile. macOS 12+, iOS 15+.

## Install

In `Package.swift`:

```swift
.package(url: "https://github.com/OWNER/vdblite-swift", from: "0.1.0")
// and in your target: .product(name: "VDBLite", package: "vdblite-swift")
```

Or in Xcode: File → Add Package Dependencies, with that URL.

## Use

```swift
import VDBLite

// Any function from texts to vectors: Core ML, MLX, an API...
let embed: Embedder = { texts in try await model.embed(texts) }

let db = try Database.open(url.path)
let docs = try db.getOrCreateCollection("docs", options: CollectionOptions(dim: 384))

try await docs.upsert(Doc(key: "note-1", content: markdown, meta: ["tag": "travel", "year": 2024]), embed: embed)

let q = try await embed(["where did we stay in Lisbon?"])[0]
let hits = try docs.search(q, options: SearchOptions(
    k: 5,
    filter: ["year": ["$gte": 2023]],
    text: "Lisbon" // optional: hybrid keyword + vector search
))
for h in hits { print(h.key, h.score, h.text) }
```

Content is split into chunks along its structure (Markdown headings, HTML,
code, paragraphs). Re-upserting an edited document with an embedder embeds
only the chunks whose text changed.

| Method | Returns |
| --- | --- |
| `search(q, options:)` | best chunks |
| `searchDocuments(q, agg:top:)` | documents ranked by their chunks |
| `searchExpand(q, neighbors:)` | hits widened by neighbouring chunks |
| `searchSections(q)` | the whole heading section around each hit |
| `get`, `chunks`, `scan`, `count`, `info`, `stats` | reading |
| `db.write([.upsert(...), .delete(...)])` | several changes, atomically |
| `docs.bulkLoad()` → `add`, `finish` | fastest way to fill a collection |

Filters and metadata are literals (`["lang": "en", "year": ["$gte": 2023]]`);
operators: `$eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not`.
Failures throw `VdbliteError`, whose `code` says what kind.

Calls block (the heavy ones use every core), so call them off the main
actor. Every type is thread-safe.

## Development

This package's source lives in [bindings/swift](https://github.com/OWNER/vdblite/tree/main/bindings/swift)
of the main repository; releases copy it to `vdblite-swift`.
`scripts/build-xcframework.sh --host` builds `CVDBLite.xcframework` for
this Mac (without Xcode); without `--host` it builds every slice (needs
Xcode). Then `swift test`.
