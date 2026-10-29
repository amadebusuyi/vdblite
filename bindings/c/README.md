# VDBLite for C and C++

An embedded, single-file vector database: no server, one `.vdb` file. This
is its C API (`vdblite.h`), a header-only C++17 wrapper (`vdblite.hpp`), and
the shared and static libraries they link.

## Install

Download the archive for your platform from the
[releases](https://github.com/OWNER/vdblite/releases): `vdblite-c-<version>-<platform>.tar.gz`
for macOS (arm64, x64), Linux (x64, arm64; glibc 2.28+ or musl) and Windows
(x64, arm64; MSVC, and x64 MinGW). Each holds `include/` and `lib/`:

| Platform | Shared | Static |
| --- | --- | --- |
| macOS | `libvdblite.dylib` | `libvdblite.a` |
| Linux | `libvdblite.so` | `libvdblite.a` |
| Windows (MSVC) | `vdblite.dll` + import library `vdblite.dll.lib` | `vdblite.lib` |
| Windows (MinGW) | `vdblite.dll` | `libvdblite.a` |

The static library needs some system libraries too: `lib/link.txt` lists
them for that platform (on Linux, `-lgcc_s -lutil -lrt -lpthread -lm -ldl -lc`).

```sh
cc app.c -I vdblite/include vdblite/lib/libvdblite.a -lpthread -ldl -lm      # static
c++ app.cpp -std=c++17 -I vdblite/include -L vdblite/lib -lvdblite         # shared
```

Or build it: `cargo build -p vdblite-c --profile release-ffi` writes the
libraries to `target/release-ffi/`.

## Use (C)

```c
#include "vdblite.h"

vdb_db *db;
vdb_collection *docs;
if (vdb_open("notes.vdb", 0, 0, &db) != VDB_OK) { fprintf(stderr, "%s\n", vdb_last_error()); return 1; }
vdb_create_collection(db, "docs", "{\"dim\": 3, \"chunking\": {\"kind\": \"none\"}}", &docs);

// Upserts and deletes in one transaction; each upsert names how many
// vectors it takes from the flat array.
float vectors[] = {1, 0, 0, 0, 1, 0};
char *out;
vdb_write(db,
  "[{\"op\": \"upsert\", \"collection\": \"docs\", \"key\": \"a\", \"content\": \"alpha\","
  "  \"meta\": {\"lang\": \"en\"}, \"vectors\": 1},"
  " {\"op\": \"upsert\", \"collection\": \"docs\", \"key\": \"b\", \"content\": \"beta\", \"vectors\": 1}]",
  vectors, 6, &out);
vdb_string_free(out);

float query[] = {1, 0.1f, 0};
vdb_search(docs, query, 3, "{\"k\": 5, \"filter\": {\"lang\": \"en\"}}", &out);
puts(out); // [{"key":"a","score":0.99,"text":"alpha","meta":{"lang":"en"},...}]
vdb_string_free(out);

vdb_collection_free(docs);
vdb_db_free(db);
```

- Every function returns `VDB_OK` or an error code (`VDB_ERR_*`);
  `vdb_last_error()` has the message, on the same thread.
- Options, documents, metadata, filters and results are JSON, in the same
  shapes as the [Node.js API](../node/README.md): see the comments in
  [vdblite.h](include/vdblite.h). Free strings the library returns with
  `vdb_string_free`.
- Chunked collections: `vdb_embed_inputs` gives the texts to embed for a
  document, one per chunk; pass that many vectors. `vdb_plan_embeddings`
  gives only the chunks whose text changed, for re-embedding just those.
- Bulk loads (`vdb_bulk_load`, `vdb_bulk_add`, `vdb_bulk_finish`) write
  documents as they come and build the index once at the end.
- Handles are thread-safe; calls block. One process writes a file at a
  time (`vdb_open`); others can read it alongside (`VDB_OPEN_READ_ONLY`).

## Use (C++)

```cpp
#include "vdblite.hpp"

auto db = vdblite::Database::open("notes.vdb");
auto docs = db.create_collection("docs", R"({"dim": 3})");
db.write(R"([{"op": "upsert", "collection": "docs", "key": "a", "content": "alpha", "vectors": 1}])", {1, 0, 0});
std::string hits = docs.search({1, 0, 0}, R"({"k": 5})"); // parse with your JSON library
```

Failures throw `vdblite::Error`, whose `code()` is a `VDB_ERR_*` constant.
Handles close and free themselves.

## Development

`bindings/c/scripts/test.sh` builds the library and runs the C and C++
tests. `node bindings/c/scripts/ffi.mjs` builds it for each platform and
places it for the Go, Java and .NET bindings.
