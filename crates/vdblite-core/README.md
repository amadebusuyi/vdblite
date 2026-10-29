# vdblite-core

The storage layer: table layout, persisted records, metadata encoding and the codecs for vectors, bitmaps and sealed segments, on [redb](https://github.com/cberner/redb).

This is an internal crate of [VDBLite](https://crates.io/crates/vdblite), an
embedded, single-file vector database. Its API follows VDBLite's needs and may
change in any release; depend on `vdblite` instead.

Licensed under the [Apache License, Version 2.0](LICENSE-APACHE).
