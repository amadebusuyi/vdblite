# Changelog

All notable changes to VDBLite. Versions follow [Semantic Versioning](https://semver.org);
while the version is 0.x, a minor release (0.1 → 0.2) may break the API.
The file format has its own version (see [Compatibility](README.md#compatibility)).

## [0.1.0]

First public release.

- Single-file `.vdb` databases on redb: ACID, checksummed, crash-safe; one
  writer and any number of readers, including read-only handles in other
  processes.
- Structural chunking of Markdown, HTML, code and plain text, with chunks
  stored as byte ranges so documents rebuild byte for byte.
- HNSW search with its index memory-mapped from the file and shared across
  processes; SIMD kernels (AVX2, NEON) chosen at runtime, metadata filters
  on Roaring bitmaps, and hybrid BM25 + vector search.
- Result shapes: chunks, expanded neighbours, sections, and documents ranked
  by their chunks.
- Bulk loads (`Db::bulk_load`, Node `collection.bulkLoad()`): documents are
  written as they arrive while one graph builds beside them, so `finish`
  mostly waits for the tail.
- Quantization: int8 codes with f16 re-ranking (default), 1-bit RaBitQ codes
  with f16 re-ranking (`Quant::Binary`, Node `"binary"`), or exact f32
  vectors (`Quant::None`, Node `"none"`).
- Node.js package `vdblite` with prebuilt addons for macOS, Linux (glibc and
  musl) and Windows, on x64 and arm64.
- A C API (`vdblite.h`) with a C++ wrapper, and packages over it for Go,
  Java (Maven Central), .NET (NuGet) and Swift (SwiftPM, macOS and iOS),
  each shipping prebuilt native code. Every language uses the same file
  format and the same JSON for metadata, filters and results.
