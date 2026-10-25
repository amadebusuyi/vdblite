'use strict'

// vecfile for Node.js: an embedded, single-file vector database.
// The native addon (vecfile.node) does the work; this file adds embedding
// functions, transactions and batching on top.

const path = require('node:path')

let native
try {
  native = require(path.join(__dirname, 'vecfile.node'))
} catch (e) {
  throw new Error(
    `vecfile: native addon not found or failed to load (${e.message}). ` +
      `Build it with \`npm run build\` in ${__dirname}.`,
  )
}

/** Native errors arrive as "CODE: message"; expose the code as err.code. */
function fixError(e) {
  if (e && typeof e.message === 'string') {
    const m = /^([A-Z_]+): ([\s\S]*)$/.exec(e.message)
    if (m) {
      e.code = m[1]
      e.message = m[2]
    }
  }
  return e
}

/** Runs a native call; errors thrown synchronously (argument checks) and
 * rejections both get their code. */
async function call(fn) {
  try {
    return await fn()
  } catch (e) {
    throw fixError(e)
  }
}

function callSync(fn) {
  try {
    return fn()
  } catch (e) {
    throw fixError(e)
  }
}

function toF32(v, what) {
  if (v instanceof Float32Array) return v
  if (Array.isArray(v) || ArrayBuffer.isView(v)) return Float32Array.from(v)
  throw new TypeError(`${what} must be an array of numbers or a Float32Array`)
}

function docInput(doc) {
  if (!doc || typeof doc.key !== 'string') throw new TypeError('a document needs a string key')
  return {
    key: doc.key,
    content: doc.content == null ? '' : String(doc.content),
    meta: doc.meta ?? undefined,
    format: doc.format ?? undefined,
  }
}

const EMBED_BATCH = 128
const MAX_STALE_RETRIES = 3

async function runEmbedder(embed, texts) {
  const out = []
  for (let i = 0; i < texts.length; i += EMBED_BATCH) {
    const batch = texts.slice(i, i + EMBED_BATCH)
    const vectors = await embed(batch)
    if (!vectors || vectors.length !== batch.length) {
      const err = new Error(
        `embedder returned ${vectors ? vectors.length : 'nothing'} vectors for ${batch.length} texts`,
      )
      err.code = 'EMBEDDER'
      throw err
    }
    for (const v of vectors) out.push(toF32(v, 'an embedding'))
  }
  return out
}

class Database {
  /** Opens (or creates) the database file at `path` for writing. Other
   * processes may open it read-only alongside (see `openReadOnly`). */
  static open(path, options) {
    return new Database(callSync(() => native.Database.open(path, options)))
  }

  /** Opens for writing with no other process allowed, not even readers;
   * small commits are then about twice as fast. */
  static openExclusive(path, options) {
    return new Database(callSync(() => native.Database.openExclusive(path, options)))
  }

  /** Opens an existing file read-only, alongside the process writing it.
   * Reads and searches see each of its commits as it lands; writes fail
   * with code 'READ_ONLY'. */
  static openReadOnly(path, options) {
    return new Database(callSync(() => native.Database.openReadOnly(path, options)))
  }

  /** Shrinks a database file that is not open anywhere. */
  static compactFile(path) {
    return callSync(() => native.Database.compactFile(path))
  }

  constructor(inner) {
    this._inner = inner
  }

  get path() {
    return this._inner.path
  }

  get closed() {
    return this._inner.closed
  }

  get readOnly() {
    return callSync(() => this._inner.readOnly)
  }

  /** Releases the file. Later calls fail with code 'CLOSED'. */
  close() {
    this._inner.close()
  }

  collections() {
    return callSync(() => this._inner.collections())
  }

  hasCollection(name) {
    return callSync(() => this._inner.hasCollection(name))
  }

  createCollection(name, options) {
    if (!options || !options.dim) throw new TypeError('createCollection needs options.dim')
    return new Collection(this, callSync(() => this._inner.createCollection(name, options)))
  }

  collection(name) {
    return new Collection(this, callSync(() => this._inner.collection(name)))
  }

  /** Opens `name`, creating it with `options` if it doesn't exist. */
  getOrCreateCollection(name, options) {
    return this.hasCollection(name) ? this.collection(name) : this.createCollection(name, options)
  }

  dropCollection(name) {
    callSync(() => this._inner.dropCollection(name))
  }

  /** Seals buffered chunks and merges segments; run after bulk loads. */
  async optimize() {
    await call(() => this._inner.optimize())
  }

  /** Waits until background sealing and compaction have caught up with
   * every commit so far; rejects with the first error they hit. */
  async waitForMaintenance() {
    await call(() => this._inner.waitForMaintenance())
  }

  /**
   * Runs `fn(tx)` and commits everything it queued atomically.
   *
   * `tx.upsert(collection, doc, embedOrVectors)` and `tx.delete(collection, key)`
   * only queue work. Embedding happens after `fn` returns and before the
   * commit, and only for chunks whose text changed since the stored version.
   * Resolves to `{ value, results }`, where `value` is what `fn` returned.
   */
  async transaction(fn) {
    const ops = []
    const tx = {
      upsert(collection, doc, embed) {
        assertCollection(collection)
        if (typeof embed !== 'function' && !Array.isArray(embed)) {
          throw new TypeError('upsert needs an embed function or an array of vectors (one per chunk)')
        }
        ops.push({ op: 'upsert', collection, doc: docInput(doc), embed })
      },
      delete(collection, key) {
        assertCollection(collection)
        ops.push({ op: 'delete', collection, key: String(key) })
      },
    }
    const value = await fn(tx)
    if (ops.length === 0) return { value, results: [] }

    // The last write to a document wins, so earlier ones are dropped; this
    // also keeps planned embeddings valid within the batch.
    const lastIndex = new Map()
    ops.forEach((o, i) => lastIndex.set(`${o.collection.name}\0${o.op === 'delete' ? o.key : o.doc.key}`, i))
    const final = ops.filter((o, i) => lastIndex.get(`${o.collection.name}\0${o.op === 'delete' ? o.key : o.doc.key}`) === i)

    for (let attempt = 0; ; attempt++) {
      const nativeOps = await this._prepare(final)
      try {
        const out = await call(() => this._inner.write(nativeOps))
        if (out.warning) process.emitWarning(out.warning, { code: 'VECFILE_MAINTENANCE' })
        return { value, results: out.results }
      } catch (e) {
        // A document changed between planning and writing: plan again.
        if (e.code === 'STALE_EMBEDDINGS' && attempt < MAX_STALE_RETRIES) continue
        throw e
      }
    }
  }

  async _prepare(ops) {
    const nativeOps = []
    const pending = new Map() // embed fn -> [{ op, plan }]
    for (const o of ops) {
      const coll = o.collection
      if (o.op === 'delete') {
        nativeOps.push({ op: 'delete', collection: coll.name, key: o.key })
        continue
      }
      const nativeOp = { op: 'upsert', collection: coll.name, ...o.doc }
      nativeOps.push(nativeOp)
      if (Array.isArray(o.embed)) {
        nativeOp.vectors = o.embed.map((v) => toF32(v, 'a vector'))
        continue
      }
      const plan = await call(() => coll._inner.planEmbeddings(o.doc))
      nativeOp.embeddings = []
      if (plan.length === 0) continue
      if (!pending.has(o.embed)) pending.set(o.embed, [])
      pending.get(o.embed).push({ nativeOp, plan })
    }
    // One batched embedder pass per embedding function.
    for (const [embed, items] of pending) {
      const texts = items.flatMap((it) => it.plan.map((p) => p.text))
      const vectors = await runEmbedder(embed, texts)
      let i = 0
      for (const it of items) {
        for (const p of it.plan) it.nativeOp.embeddings.push({ hash: p.hash, vector: vectors[i++] })
      }
    }
    return nativeOps
  }
}

function assertCollection(c) {
  if (!(c instanceof Collection)) throw new TypeError('expected a Collection')
}

class Collection {
  constructor(db, inner) {
    this.db = db
    this._inner = inner
  }

  get name() {
    return this._inner.name
  }

  /** Schema: dim, metric, quantization, chunking. */
  info() {
    return callSync(() => this._inner.info())
  }

  /** Index size: chunks, sealedSegments, buffered, memoryBytes, and
   * maintenanceError if background maintenance last failed. */
  stats() {
    return callSync(() => this._inner.stats())
  }

  /** The texts that would be embedded for `content`, one per chunk. */
  embedInputs(content, format) {
    return callSync(() => this._inner.embedInputs(content, format))
  }

  /** Inserts or replaces one document. */
  async upsert(doc, embed) {
    const { results } = await this.db.transaction((tx) => tx.upsert(this, doc, embed))
    return results[0]
  }

  /** Upserts many documents, committing every `batchSize` (default 256). */
  async upsertMany(docs, embed, { batchSize = 256 } = {}) {
    const results = []
    for (let i = 0; i < docs.length; i += batchSize) {
      const batch = docs.slice(i, i + batchSize)
      const out = await this.db.transaction((tx) => {
        for (const d of batch) tx.upsert(this, d, embed)
      })
      results.push(...out.results)
    }
    return results
  }

  /** Deletes a document; resolves to whether it existed. */
  async delete(key) {
    const { results } = await this.db.transaction((tx) => tx.delete(this, key))
    return results[0].deleted
  }

  async get(key) {
    return call(() => this._inner.get(String(key)))
  }

  async chunks(key) {
    return call(() => this._inner.chunks(String(key)))
  }

  /** Document summaries (no content) matching an optional filter. */
  async scan(filter) {
    return call(() => this._inner.scan(filter ?? null))
  }

  async count() {
    return call(() => this._inner.count())
  }

  /** The best-matching chunks. Options: { k = 10, ef, filter }. */
  async search(query, options) {
    return call(() => this._inner.search(toF32(query, 'the query'), options))
  }

  /** Documents ranked by their best chunks. Options add { agg: 'max' | 'sum', top }. */
  async searchDocuments(query, options = {}) {
    const { agg, top, ...rest } = options
    return call(() => this._inner.searchDocuments(toF32(query, 'the query'), rest, agg, top))
  }

  /** Each hit widened by `neighbors` chunks either side (default 1), merged. */
  async searchExpand(query, options = {}) {
    const { neighbors = 1, ...rest } = options
    return call(() => this._inner.searchExpand(toF32(query, 'the query'), neighbors, rest))
  }

  /** The whole heading section around each hit. */
  async searchSections(query, options) {
    return call(() => this._inner.searchSections(toF32(query, 'the query'), options))
  }

  /** Whether the collection has a keyword index for hybrid search. */
  hasLexical() {
    return callSync(() => this._inner.hasLexical())
  }

  /** Builds the keyword index from stored text, without re-embedding. */
  async enableLexical() {
    return call(() => this._inner.enableLexical())
  }

  /**
   * Embeds `text` with `embed`, then searches. `options.mode` picks the shape.
   * The search is hybrid (vector + keywords) when the collection has a keyword
   * index, unless `options.hybrid` is false.
   */
  async searchText(text, embed, options = {}) {
    const [vector] = await runEmbedder(embed, [text])
    const { mode = 'chunks', hybrid = true, ...opts } = options
    const rest = hybrid ? { ...opts, text } : opts
    switch (mode) {
      case 'chunks':
        return this.search(vector, rest)
      case 'documents':
        return this.searchDocuments(vector, rest)
      case 'expand':
        return this.searchExpand(vector, rest)
      case 'sections':
        return this.searchSections(vector, rest)
      default:
        throw new TypeError(`unknown mode ${mode} (chunks, documents, expand, sections)`)
    }
  }
}

/** Opens (or creates) a database file. */
/** Opens (or creates) a database file. `options.cacheBytes` sets the page
 * cache size. */
function open(path, options) {
  return Database.open(path, options)
}

module.exports = { open, Database, Collection }
