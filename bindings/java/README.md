# VDBLite for Java

An embedded, single-file vector database: no server, one `.vdb` file. The
core is Rust; this jar calls it through [JNA](https://github.com/java-native-access/jna)
and bundles the native library for macOS (arm64, x64), Linux (x64, arm64;
glibc 2.28+) and Windows (x64, arm64). Java 17 or newer; works from Kotlin
and other JVM languages.

## Install

Maven:

```xml
<dependency>
  <groupId>io.github.OWNER</groupId>
  <artifactId>vdblite</artifactId>
  <version>0.1.0</version>
</dependency>
```

Gradle: `implementation("io.github.OWNER:vdblite:0.1.0")`

## Use

```java
import io.vdblite.*;

// Any function from texts to vectors: OpenAI, a local model...
Embedder embed = texts -> myModel.embed(texts);

try (Database db = Database.open(Path.of("notes.vdb"))) {
  Collection docs = db.getOrCreateCollection("docs", CollectionOptions.dim(1536));

  docs.upsert(new Doc("readme", markdown).meta("lang", "en").meta("year", 2024), embed);

  float[] q = embed.embed(List.of("how do I install it?")).get(0);
  List<ChunkHit> hits = docs.search(q, SearchOptions.k(5)
      .filter(Map.of("lang", "en", "year", Map.of("$gte", 2023)))
      .text("install")); // optional: hybrid keyword + vector search
  for (ChunkHit h : hits) System.out.println(h.key() + " " + h.score() + " " + h.text());
}
```

Content is split into chunks along its structure (Markdown headings, HTML,
code, paragraphs). Re-upserting an edited document with an `Embedder`
embeds only the chunks whose text changed.

| Method | Returns |
| --- | --- |
| `search(q, options)` | best chunks |
| `searchDocuments`, `searchDocumentsSum(q, top, options)` | documents ranked by their chunks |
| `searchExpand(q, neighbors, options)` | hits widened by neighbouring chunks |
| `searchSections(q, options)` | the whole heading section around each hit |
| `get`, `chunks`, `scan(filter)`, `count`, `info`, `stats` | reading |
| `db.write(WriteOp.upsert(...), WriteOp.delete(...))` | several changes, atomically |
| `docs.bulkLoad()` → `add`, `finish` | fastest way to fill a collection |

Filters are Mongo-style maps: `$eq $ne $in $nin $gt $gte $lt $lte $exists
$and $or $not`. Failures throw `VdbliteException`, whose `code()` says what
kind.

`Database` and `Collection` are thread-safe; calls block. One process writes
a file at a time; others can `Database.openReadOnly(path)` alongside and see
each commit as it lands.

On a platform the jar doesn't include, build the library
(`cargo build -p vdblite-c --profile release-ffi`) and pass
`-Dvdblite.library.path=/path/to/libvdblite.so`.

## Development

`node bindings/c/scripts/ffi.mjs dev` builds the library for this machine
into `src/main/resources/`; then `mvn test`.
