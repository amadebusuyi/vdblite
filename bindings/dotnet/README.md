# VDBLite for .NET

An embedded, single-file vector database: no server, one `.vdb` file. The
core is Rust; this package calls it through P/Invoke and includes the native
library for Windows (x64, arm64), macOS (arm64, x64) and Linux (x64, arm64;
glibc 2.28+ or musl/Alpine). .NET 8 or newer.

## Install

```sh
dotnet add package VDBLite
```

## Use

```csharp
using VDBLite;

// Any function from texts to vectors: OpenAI, Azure OpenAI, ONNX...
Embedder embed = async texts => await myModel.EmbedAsync(texts);

using var db = Database.Open("notes.vdb");
var docs = db.GetOrCreateCollection("docs", new CollectionOptions(1536));

await docs.UpsertAsync(new Doc("readme", markdown,
    new Dictionary<string, object?> { ["lang"] = "en", ["year"] = 2024 }), embed);

var q = (await embed(["how do I install it?"]))[0];
var hits = docs.Search(q, new SearchOptions
{
    K = 5,
    Filter = new Dictionary<string, object?>
    {
        ["lang"] = "en",
        ["year"] = new Dictionary<string, object?> { ["$gte"] = 2023 },
    },
    Text = "install", // optional: hybrid keyword + vector search
});
foreach (var h in hits) Console.WriteLine($"{h.Key} {h.Score} {h.Text}");
```

Content is split into chunks along its structure (Markdown headings, HTML,
code, paragraphs). Re-upserting an edited document with `UpsertAsync`
embeds only the chunks whose text changed.

| Method | Returns |
| --- | --- |
| `Search(q, options)` | best chunks |
| `SearchDocuments(q, options, "max" \| "sum", top)` | documents ranked by their chunks |
| `SearchExpand(q, neighbors, options)` | hits widened by neighbouring chunks |
| `SearchSections(q, options)` | the whole heading section around each hit |
| `Get`, `Chunks`, `Scan(filter)`, `Count`, `Info`, `Stats` | reading |
| `db.Write(WriteOp.Upsert(...), WriteOp.Delete(...))` | several changes, atomically |
| `docs.BulkLoad()` → `Add`, `Finish` | fastest way to fill a collection |

Filters are Mongo-style: `$eq $ne $in $nin $gt $gte $lt $lte $exists $and
$or $not`. Failures throw `VdbliteException`, whose `Code` says what kind.

`Database` and `Collection` are thread-safe; calls block, so wrap them in
`Task.Run` to keep a UI thread free. One process writes a file at a time;
others can `Database.OpenReadOnly(path)` alongside and see each commit as it
lands.

## Development

`node bindings/c/scripts/ffi.mjs dev` builds the library for this machine
into `runtimes/`; then `dotnet test tests/VDBLite.Tests`.
