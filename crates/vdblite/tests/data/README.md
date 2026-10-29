# Test data

- `redb-abort.vdb`: a damaged file on which redb double-panics (see
  `tests/corruption.rs`, `redb_double_panic_reproduction`).
- `v2-phase7.vdb`: a file written by vdblite as of Phase 7 (format version
  2, a bare redb file), for `tests/readers.rs::phase_7_files_are_upgraded`.
  One collection, `docs`: 8-dim cosine, int8, unchunked, seal threshold 64,
  keyword index on. Documents `doc-0` .. `doc-319`, each with content
  `document number {i}`, metadata `n = i`, and the pseudo-random vector of
  `fixture_vector` in `tests/readers.rs`; every 7th of `doc-0` ..
  `doc-299` deleted; `doc-300` .. `doc-319` left in the write buffer, the rest
  sealed into 3 segments. Written by `make_v2_fixture.rs` (in this
  directory) as an example of the Phase 7 code (commit 839a56f).
