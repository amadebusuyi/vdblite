# vecfile for Node.js

An embedded, single-file vector database: no server, one `.vdb` file. The
core is Rust; this package is a native addon with a JS API and TypeScript
types.

## Install

Build the addon once (needs Rust, from https://rustup.rs), then install the
folder into your project:

```sh
cd /path/to/vector-db/bindings/node && npm run build
cd /path/to/your-project && npm install /path/to/vector-db/bindings/node
```

`npm install <folder>` links the folder, so rebuilding updates your project.
For a copy instead (to commit or ship), run `npm pack` here and install the
`.tgz`. The addon is built for the machine it was built on.

## Use

```js
import { open } from 'vecfile' // or: const { open } = require('vecfile')

// Any async function from texts to vectors: OpenAI, Voyage, transformers.js…
const embed = async (texts) => {
  const res = await openai.embeddings.create({ model: 'text-embedding-3-small', input: texts })
  return res.data.map((d) => d.embedding)
}

const db = open('notes.vdb')
const docs = db.getOrCreateCollection('docs', { dim: 1536, metric: 'cosine' })

await docs.upsert({ key: 'readme', content: markdown, meta: { lang: 'en', year: 2024 } }, embed)

const hits = await docs.searchText('how do I install it?', embed, {
  k: 5,
  filter: { lang: 'en', year: { $gte: 2023 } },
})
// [{ key, text, score, headingPath, meta, byteStart, byteEnd, ... }]

db.close()
```

Content is split into chunks along its structure (Markdown headings, HTML,
code, paragraphs). Re-upserting an edited document embeds only the chunks
whose text changed, so you pay your embedding API for the edit, not the file.

### Searching

| Method | Returns |
| --- | --- |
| `search(vector, { k, ef, filter })` | best chunks |
| `searchDocuments(vector, { k, agg: 'max' \| 'sum', top })` | documents ranked by their chunks |
| `searchExpand(vector, { k, neighbors })` | hits widened by neighbouring chunks, merged |
| `searchSections(vector, { k })` | the whole heading section around each hit |
| `searchText(text, embed, { mode, ...options })` | embeds, then any of the above (hybrid by default) |

Pass `text` to any search to make it **hybrid**: BM25 keyword matching is
fused with vector similarity, so exact identifiers, error codes and rare names
are found even when the embedding misses them. Each chunk hit carries
`vectorScore` and `lexicalScore` (null when that side didn't find it).
`searchText` does this automatically; pass `hybrid: false` to opt out.
Collections made before hybrid search can build the keyword index from stored
text with `await collection.enableLexical()`.

Filters are Mongo-style: `$eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not`.
Equality on a list field (`tags: ['a', 'b']`) matches any element.

### Writing

```js
await docs.upsertMany(items, embed, { batchSize: 256 }) // bulk load
await docs.delete('readme')

// Several changes, committed atomically:
await db.transaction((tx) => {
  tx.upsert(docs, { key: 'a', content: '...' }, embed)
  tx.delete(docs, 'b')
})

// Or bring your own vectors, one per chunk:
const inputs = docs.embedInputs(content)        // the chunk texts
await docs.upsert({ key: 'c', content }, vectorsFor(inputs))
```

After a big load, `await db.optimize()` merges index segments for faster
search. `Database.compactFile(path)` shrinks a closed file.

### Reading

`get(key)` (with content), `chunks(key)`, `scan(filter)` (summaries), `count()`,
`info()`, `stats()`.

## Notes

- Everything that touches the file runs off the main thread and returns a Promise.
- One process writes a file at a time (`open`); any number of other
  processes can read it alongside (`openReadOnly`) and see each commit as it
  lands. `openExclusive` shuts readers out in exchange for faster small
  commits. Within a process, any number of concurrent reads run alongside
  writes.
- Commits return without waiting for chunks to be sealed into the graph
  index; that happens in the background. `await db.waitForMaintenance()`
  waits for it (after a bulk load, or in tests).
- Errors have a `code`: `DIMENSION`, `NO_SUCH_COLLECTION`, `COLLECTION_EXISTS`,
  `INVALID_ARGUMENT`, `EMBEDDER`, `STORAGE`, `READ_ONLY`, `CLOSED`, …
- `byteStart`/`byteEnd` are UTF-8 byte offsets: `Buffer.from(content).subarray(byteStart, byteEnd)`.

## Development

```sh
npm run build   # release build of the addon
npm test
```
