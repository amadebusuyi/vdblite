# vdblite-redb

This is [redb](https://github.com/cberner/redb) 4.4.0, by Christopher
Berner, unchanged except for one public method:

```rust
impl Builder {
    /// Opens an existing database read-only on a custom storage backend.
    pub fn open_read_only_with_backend(&self, backend: impl StorageBackend)
        -> Result<ReadOnlyDatabase, DatabaseError>;
}
```

redb can already open a database for writing on a custom backend
(`create_with_backend`), but only open one read-only from a path. VDBLite
keeps redb's pages inside its own single-file layout (see
`docs/design/phase-8.md`), so its reading processes need the read-only open
on its backend too.

The addition is the only difference from the published crate; the library
is still named `redb`. It is offered upstream, and VDBLite will go back to
the published crate once redb has it.

redb is licensed under either the MIT license or the Apache License,
Version 2.0, at your option; this fork is used under the Apache License,
Version 2.0 (LICENSE-APACHE). `xxh3.rs` keeps its own MIT notice in its header.
