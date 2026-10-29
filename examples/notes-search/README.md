# notes-search

Semantic search over a folder of notes, docs or code, from the command line.
It uses [VDBLite](../../bindings/node) for storage and search and a local
embedding model (all-MiniLM-L6-v2 via transformers.js), so there's no server,
no API key, and everything lives in one file: `notes.vdb`.

## Setup

```sh
# once: build the vdblite addon from this checkout (needs Rust)
(cd ../../bindings/node && npm run build)

npm install
# outside this repo, depend on the published package instead: npm install vdblite
```

The first index or search downloads the model (~90 MB) into `.model-cache/`.

## Use

```sh
node cli.js index ~/Documents/notes        # add a folder (re-run any time; default: current folder)
node cli.js search "how do I rotate the api keys"
node cli.js search "deploy steps" --context    # include neighbouring chunks
node cli.js search "error handling" --files    # rank whole files instead
node cli.js search "retry logic" --ext rs --k 10
node cli.js search "MAX_DIM"                   # exact identifiers work too
node cli.js search "vague words" --all         # show weak matches anyway
node cli.js stats
node cli.js forget ~/Documents/notes
```

Try it on this repository:

```sh
node cli.js index ../..
node cli.js search "why did we choose redb instead of writing our own pager"
```

Re-running `index` is cheap: unchanged files are skipped, and in a changed
file only the chunks whose text changed are re-embedded (editing one paragraph
re-embeds about 2 chunks). Deleted files are removed from the index.

Set `NOTES_DB=/path/to/file.vdb` to keep the index somewhere else.

## How it works

| File | Role |
| --- | --- |
| `embedder.js` | Loads the model and turns texts into 384-dim vectors |
| `cli.js` | Walks folders, upserts files with metadata (`root`, `path`, `ext`, `modified`), searches |

Searches are hybrid: keyword (BM25) matches are fused with embedding
similarity, so exact names like `MAX_DIM` are found as reliably as paraphrases.
Each result says why it matched (`similarity 0.54 · keyword match`). When
nothing shares a keyword with the query and the best similarity is weak, the
tool says "No close matches" instead of listing noise.

Files are chunked by structure: Markdown headings, HTML sections, code
definitions. Results show the heading path of each match
(`Guide › Install › Linux`). Search filters on metadata, e.g. `--ext md`
becomes `{ ext: "md" }`.

## Notes

- **Intel Macs:** `package.json` pins `onnxruntime-node` to 1.22.0, the last
  release with Intel macOS binaries. On Apple Silicon, Linux or Windows you
  can remove the `overrides` entry.
- **Swapping the model:** any embedder works. To use an API instead, replace
  `embed()` in `embedder.js` and set `DIM` to the model's dimension. Then
  delete `notes.vdb`, because a collection's dimension is fixed when it's
  created.
