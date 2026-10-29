# VDBLite for Go

An embedded, single-file vector database: no server, one `.vdb` file. The
core is Rust; this package links it statically with cgo.

## Install

```sh
go get github.com/OWNER/vdblite/bindings/go
```

The module includes the prebuilt library for macOS (arm64, amd64), Linux
(amd64, arm64; glibc 2.28+) and Windows (amd64, with MinGW gcc), so there
is nothing else to install and no Rust needed, only cgo's C compiler. The
download is large (the static libraries are about 70 MB together).

On another platform, or with your own build of the library, build with
`-tags vdblite_custom` and point cgo at it:

```sh
CGO_LDFLAGS="/path/to/libvdblite.a -lpthread -ldl -lm" go build -tags vdblite_custom
```

## Use

```go
import vdblite "github.com/OWNER/vdblite/bindings/go"

db, err := vdblite.Open("notes.vdb", nil)
if err != nil { ... }
defer db.Close()

docs, err := db.CreateCollection("docs", vdblite.CollectionOptions{Dim: 1536})

// Any function from texts to vectors: OpenAI, Ollama, a local model...
embed := func(texts []string) ([][]float32, error) { ... }

_, err = docs.UpsertText(vdblite.Doc{
    Key: "readme", Content: markdown,
    Meta: vdblite.Meta{"lang": "en", "year": 2024},
}, embed)

q, _ := embed([]string{"how do I install it?"})
hits, err := docs.Search(q[0], &vdblite.SearchOptions{
    K:      5,
    Filter: vdblite.Filter{"lang": "en", "year": vdblite.Filter{"$gte": 2023}},
    Text:   "install", // optional: hybrid keyword + vector search
})
for _, h := range hits {
    fmt.Println(h.Key, h.Score, h.HeadingPath, h.Text)
}
```

Content is split into chunks along its structure (Markdown headings, HTML,
code, paragraphs). Re-upserting an edited document with `UpsertText` embeds
only the chunks whose text changed.

| Method | Returns |
| --- | --- |
| `Search(q, opts)` | best chunks |
| `SearchDocuments(q, opts, "max" \| "sum", top)` | documents ranked by their chunks |
| `SearchExpand(q, neighbors, opts)` | hits widened by neighbouring chunks |
| `SearchSections(q, opts)` | the whole heading section around each hit |
| `Get`, `Chunks`, `Scan(filter)`, `Count`, `Info`, `Stats` | reading |
| `db.Write(vdblite.Upsert(...), vdblite.Delete(...))` | several changes, atomically |
| `docs.BulkLoad(0)` → `Add`, `Finish` | fastest way to fill a collection |

Filters are Mongo-style: `$eq $ne $in $nin $gt $gte $lt $lte $exists $and
$or $not`. Errors are `*vdblite.Error`; test them with
`errors.Is(err, vdblite.CodeNoSuchCollection)`.

Everything is safe to use from many goroutines. One process writes a file
at a time; others can open it with `&vdblite.OpenOptions{ReadOnly: true}`
and see each commit as it lands.

## Development

`node bindings/c/scripts/ffi.mjs dev` builds the library for this machine
into `lib/`; then `go test ./...`.
