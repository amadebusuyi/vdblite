import assert from 'node:assert/strict'
import fs from 'node:fs'
import os from 'node:os'
import path from 'node:path'
import { test } from 'node:test'

import { DEFAULT_CHECKPOINT, Database, open } from '../index.mjs'

const DIM = 64

/** Deterministic bag-of-words vector: texts that share words are close. */
function embedOne(text) {
  const v = new Float32Array(DIM)
  for (const w of text.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean)) {
    let h = 2166136261
    for (const c of w) h = Math.imul(h ^ c.charCodeAt(0), 16777619) >>> 0
    v[h % DIM] += 1
    v[(h >>> 16) % DIM] += 0.5
  }
  v[DIM - 1] += 0.01
  return v
}

/** An async embedder (like an API call) that counts what it embeds. */
function counting() {
  const fn = async (texts) => {
    fn.texts += texts.length
    await new Promise((r) => setTimeout(r, 1))
    return texts.map(embedOne)
  }
  fn.texts = 0
  return fn
}

function article(topic, n) {
  let s = `# ${topic} guide\n\nAn introduction to ${topic} number ${n}.\n\n`
  for (let i = 0; i < 3; i++) {
    s += `## Part ${i}\n\nThis part covers ${topic} detail ${i} for article ${n}. It explains ${topic} carefully.\n\n`
  }
  return s
}

function tmpDb() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'vdblite-node-'))
  return path.join(dir, 'test.vdb')
}

const opts = { dim: DIM, chunking: { targetTokens: 20, overlapTokens: 4 } }
const TOPICS = ['rust', 'python', 'gardening', 'astronomy', 'cooking', 'chess']

async function fill(db, n = 24, embed = counting()) {
  const docs = db.createCollection('docs', { ...opts, sealThreshold: 64 })
  const items = Array.from({ length: n }, (_, i) => ({
    key: `doc-${i}`,
    content: article(TOPICS[i % 6], i),
    meta: { topic: TOPICS[i % 6], n: i, tags: [TOPICS[i % 6], i % 2 ? 'odd' : 'even'] },
  }))
  await docs.upsertMany(items, embed, { batchSize: 8 })
  await db.waitForMaintenance()
  return docs
}

test('write, search, reopen', async () => {
  const file = tmpDb()
  let db = open(file)
  const docs = await fill(db)
  assert.ok(docs.stats().sealedSegments >= 1)
  const hits = await docs.search(embedOne('astronomy detail'), { k: 5 })
  assert.equal(hits.length, 5)
  assert.ok(hits.every((h) => h.text.includes('astronomy')))
  assert.ok(hits[0].score >= hits[4].score)
  assert.deepEqual(hits[0].headingPath.slice(0, 1), ['astronomy guide'])
  db.close()
  assert.equal(db.closed, true)
  await assert.rejects(docs.count(), { code: 'CLOSED' })

  db = Database.open(file)
  assert.deepEqual(db.collections(), ['docs'])
  const again = db.collection('docs')
  assert.equal(await again.count(), 24)
  const doc = await again.get('doc-3')
  assert.equal(doc.content, article('astronomy', 3))
  assert.deepEqual(doc.meta.tags, ['astronomy', 'odd'])
  assert.equal(doc.format, 'markdown')
  assert.equal(await again.get('nope'), null)
  const chunks = await again.chunks('doc-3')
  const bytes = Buffer.from(doc.content)
  assert.equal(chunks.map((c) => bytes.subarray(c.byteStart, c.byteEnd).toString()).join(''), doc.content)
  db.close()
})

test('only changed chunks are re-embedded', async () => {
  const db = open(tmpDb())
  const docs = db.createCollection('docs', opts)
  const embed = counting()
  const text = article('rust', 1)
  const first = await docs.upsert({ key: 'a', content: text }, embed)
  assert.equal(first.embedded, first.chunks)
  assert.equal(embed.texts, first.chunks)

  const same = await docs.upsert({ key: 'a', content: text }, embed)
  assert.equal(same.changed, false)
  assert.equal(embed.texts, first.chunks)

  const before = embed.texts
  const edited = await docs.upsert({ key: 'a', content: text.replace('detail 2 for article 1', 'detail 2, revised') }, embed)
  assert.ok(edited.embedded >= 1 && edited.embedded < edited.chunks, JSON.stringify(edited))
  assert.equal(embed.texts - before, edited.embedded)
  assert.equal(edited.version, 2)
  db.close()
})

test('mongo-style filters on search and scan', async () => {
  const db = open(tmpDb())
  const docs = await fill(db)
  const q = embedOne('guide part detail')
  const keys = async (filter) => {
    const fromSearch = new Set((await docs.search(q, { k: 1000, filter })).map((h) => Number(h.key.slice(4))))
    const fromScan = new Set((await docs.scan(filter)).map((d) => Number(d.key.slice(4))))
    assert.deepEqual([...fromSearch].sort((a, b) => a - b), [...fromScan].sort((a, b) => a - b))
    return [...fromScan].sort((a, b) => a - b)
  }
  assert.deepEqual(await keys({ topic: 'chess' }), [5, 11, 17, 23])
  assert.deepEqual(await keys({ tags: 'chess' }), [5, 11, 17, 23])
  assert.deepEqual(await keys({ topic: { $in: ['rust', 'chess'] }, n: { $lt: 12 } }), [0, 5, 6, 11])
  assert.deepEqual(await keys({ n: { $gte: 3, $lte: 6 } }), [3, 4, 5, 6])
  assert.deepEqual(await keys({ $or: [{ n: { $lt: 2 } }, { n: { $gt: 21 } }] }), [0, 1, 22, 23])
  assert.deepEqual(await keys({ topic: { $ne: 'rust' }, n: { $lt: 7 } }), [1, 2, 3, 4, 5])
  assert.deepEqual(await keys({ $not: { n: { $gte: 3 } } }), [0, 1, 2])
  assert.deepEqual(await keys({ missing: { $exists: true } }), [])
  await assert.rejects(docs.search(q, { filter: { n: { $near: 1 } } }), { code: 'INVALID_ARGUMENT' })
  db.close()
})

test('result shapes', async () => {
  const db = open(tmpDb())
  const docs = await fill(db, 12)
  const q = embedOne('gardening detail 1 article 2')
  const [best] = await docs.search(q, { k: 1 })
  assert.equal(best.key, 'doc-2')
  const [passage] = await docs.searchExpand(q, { k: 1, neighbors: 1 })
  assert.ok(passage.byteStart <= best.byteStart && best.byteEnd <= passage.byteEnd)
  const [section] = await docs.searchSections(q, { k: 1 })
  assert.ok(section.text.startsWith('## Part 1'))
  const top = await docs.searchDocuments(embedOne('cooking'), { k: 2 })
  assert.ok(top.every((d) => d.meta.topic === 'cooking'))
  const summed = await docs.searchDocuments(embedOne('cooking'), { k: 2, agg: 'sum', top: 3 })
  assert.ok(summed[0].score > top[0].score)
  const viaText = await docs.searchText('cooking', counting(), { mode: 'documents', k: 2 })
  assert.deepEqual(viaText.map((d) => d.key), top.map((d) => d.key))
  db.close()
})

test('transactions are atomic', async () => {
  const db = open(tmpDb())
  const docs = await fill(db, 6)
  const embed = counting()
  const { value, results } = await db.transaction(async (tx) => {
    tx.upsert(docs, { key: 'new', content: 'brand new text', meta: { fresh: true } }, embed)
    tx.delete(docs, 'doc-0')
    tx.upsert(docs, { key: 'temp', content: 'short lived' }, embed)
    tx.delete(docs, 'temp')
    return 42
  })
  assert.equal(value, 42)
  assert.deepEqual(results.map((r) => `${r.op}:${r.key}`), ['upsert:new', 'delete:doc-0', 'delete:temp'])
  assert.equal(await docs.count(), 6)
  assert.equal(await docs.get('doc-0'), null)

  // A throwing transaction commits nothing.
  await assert.rejects(
    db.transaction((tx) => {
      tx.delete(docs, 'doc-1')
      throw new Error('boom')
    }),
    /boom/,
  )
  // A failing embedder commits nothing either.
  await assert.rejects(
    db.transaction((tx) => {
      tx.delete(docs, 'doc-1')
      tx.upsert(docs, { key: 'x', content: 'text' }, async () => {
        throw new Error('model offline')
      })
    }),
    /model offline/,
  )
  assert.ok(await docs.get('doc-1'))
  db.close()
})

test('concurrent writers and readers', async () => {
  const db = open(tmpDb())
  const docs = db.createCollection('docs', { ...opts, sealThreshold: 32 })
  const embed = counting()
  const writes = Array.from({ length: 20 }, (_, i) =>
    docs.upsert({ key: `k${i}`, content: article(TOPICS[i % 6], i) }, embed),
  )
  const reads = Array.from({ length: 20 }, () => docs.search(embedOne('chess'), { k: 3 }))
  await Promise.all([...writes, ...reads])
  // The same key written concurrently twice still ends consistent.
  await Promise.all([
    docs.upsert({ key: 'k0', content: 'version A of the text' }, embed),
    docs.upsert({ key: 'k0', content: 'version B of the text' }, embed),
  ])
  assert.match((await docs.get('k0')).content, /version [AB]/)
  assert.equal(await docs.count(), 20)
  const stored = (await docs.scan()).reduce((n, d) => n + d.chunkCount, 0)
  assert.equal(docs.stats().chunks, stored)
  db.close()
})

test('caller vectors, options and errors', async () => {
  const db = open(tmpDb())
  const whole = db.createCollection('whole', { dim: 3, metric: 'l2', chunking: { kind: 'none' }, quantization: 'none' })
  await whole.upsert({ key: 'x', content: 'anything' }, [[1, 0, 0]])
  await whole.upsert({ key: 'y', content: 'other' }, [new Float32Array([0, 5, 0])])
  const [hit] = await whole.search([0.9, 0, 0], { k: 1 })
  assert.equal(hit.key, 'x')
  assert.deepEqual(whole.info(), { name: 'whole', dim: 3, metric: 'l2', quantization: 'none', chunking: 'none' })

  await assert.rejects(whole.search([1, 2], { k: 1 }), { code: 'DIMENSION' })
  await assert.rejects(whole.upsert({ key: 'z', content: 'bad' }, [[1, 2]]), { code: 'DIMENSION' })
  assert.throws(() => db.collection('nope'), { code: 'NO_SUCH_COLLECTION' })
  assert.throws(() => db.createCollection('whole', { dim: 3 }), { code: 'COLLECTION_EXISTS' })
  await assert.rejects(
    whole.upsert({ key: 'm', content: 'x', meta: { nested: { a: 1 } } }, [[0, 0, 1]]),
    { code: 'INVALID_ARGUMENT' },
  )
  assert.throws(() => open(db.path), { code: 'STORAGE' }, 'the file is locked while open')

  const docs = db.getOrCreateCollection('docs', opts)
  assert.equal(db.getOrCreateCollection('docs', opts).name, 'docs')
  const inputs = docs.embedInputs(article('chess', 1), 'md')
  await docs.upsert({ key: 'c', content: article('chess', 1), format: 'md' }, inputs.map(embedOne))
  assert.equal((await docs.chunks('c')).length, inputs.length)
  db.dropCollection('docs')
  assert.deepEqual(db.collections(), ['whole'])
  await db.optimize()
  db.close()
})

test('hybrid search finds exact terms', async () => {
  const db = open(tmpDb())
  const docs = await fill(db, 12)
  assert.equal(docs.hasLexical(), true)
  await docs.upsert(
    { key: 'cfg', content: '# Config\n\nSet ZX_QUUX_LIMIT to raise the ceiling.\n\n## Garden\n\nSoil and seeds.\n' },
    counting(),
  )
  const q = embedOne('gardening soil seeds')
  const plain = await docs.search(q, { k: 3 })
  assert.ok(plain.every((h) => !h.text.includes('ZX_QUUX') && h.lexicalScore === null))
  const hybrid = await docs.search(q, { k: 3, text: 'ZX_QUUX_LIMIT' })
  const hit = hybrid.find((h) => h.text.includes('ZX_QUUX_LIMIT'))
  assert.ok(hit && hit.lexicalScore > 0)
  const viaText = await docs.searchText('quux limit', counting(), { k: 3 })
  assert.ok(viaText.some((h) => h.text.includes('ZX_QUUX_LIMIT')))
  const optedOut = await docs.searchText('quux limit', counting(), { k: 3, hybrid: false })
  assert.ok(optedOut.every((h) => h.lexicalScore === null))
  db.close()

  // Collections created without the index can build it from stored text.
  const db2 = open(tmpDb())
  const bare = db2.createCollection('bare', { ...opts, lexical: false })
  await bare.upsert({ key: 'a', content: 'notes about quasars and pulsars' }, counting())
  assert.equal(bare.hasLexical(), false)
  assert.equal(await bare.enableLexical(), true)
  assert.equal(bare.hasLexical(), true)
  const [top] = await bare.search(embedOne('unrelated'), { k: 1, text: 'quasars' })
  assert.ok(top.lexicalScore > 0)
  db2.close()
})

test('read-only handles follow the writer', async () => {
  const file = tmpDb()
  const db = open(file)
  const docs = await fill(db, 12)
  const reader = Database.openReadOnly(file)
  assert.equal(reader.readOnly, true)
  assert.equal(db.readOnly, false)
  const rdocs = reader.collection('docs')
  assert.equal(await rdocs.count(), 12)

  // New commits are visible to the reader as soon as they land.
  await docs.upsert({ key: 'late', content: article('chess', 99) }, counting())
  assert.ok(await rdocs.get('late'))
  const hits = await rdocs.search(embedOne(article('chess', 99)), { k: 5 })
  assert.ok(hits.some((h) => h.key === 'late'))
  await db.waitForMaintenance()
  assert.deepEqual(
    { ...rdocs.stats(), memoryBytes: 0 },
    { ...docs.stats(), memoryBytes: 0 },
  )

  // The reader can't write.
  await assert.rejects(rdocs.upsert({ key: 'x', content: 'y' }, counting()), { code: 'READ_ONLY' })
  assert.throws(() => reader.createCollection('x', opts), { code: 'READ_ONLY' })
  assert.throws(() => Database.openExclusive(file), { code: 'STORAGE' }, 'readers are open')
  reader.close()
  db.close()

  // An exclusive writer shuts readers out.
  const solo = Database.openExclusive(file)
  assert.throws(() => Database.openReadOnly(file), { code: 'STORAGE' })
  solo.close()
})

test('bulk load', async () => {
  const file = tmpDb()
  let db = open(file)
  const docs = db.createCollection('docs', opts)
  const items = Array.from({ length: 30 }, (_, i) => ({
    key: `doc-${i}`,
    content: article(TOPICS[i % 6], i),
    meta: { topic: TOPICS[i % 6] },
  }))
  const embed = counting()

  const load = docs.bulkLoad()
  assert.throws(() => docs.bulkLoad(), { code: 'INVALID_ARGUMENT' })
  assert.equal(await load.add(items.slice(0, 20), embed), 20)
  // Caller vectors: one per chunk, in the order embedInputs gives.
  const own = items.slice(20).map((d) => ({ ...d, vectors: docs.embedInputs(d.content).map(embedOne) }))
  assert.equal(await load.add(own), 10)
  await assert.rejects(load.add([{ key: 'x', content: 'no vectors' }]), TypeError)

  // Readable before finish, but not searchable.
  assert.equal(await docs.count(), 30)
  assert.equal((await docs.get('doc-3')).meta.topic, 'astronomy')
  assert.deepEqual(await docs.search(embedOne('astronomy detail'), { k: 5 }), [])

  assert.equal(load.finished, false)
  await load.finish()
  assert.equal(load.finished, true)
  await assert.rejects(load.add(items.slice(0, 1), embed), { code: 'INVALID_ARGUMENT' })

  const hits = await docs.search(embedOne('astronomy detail'), { k: 5, filter: { topic: 'astronomy' } })
  assert.equal(hits.length, 5)
  assert.ok(hits.every((h) => h.text.includes('astronomy')))
  // A new load can start once the last one finished.
  await docs.bulkLoad().finish()
  db.close()

  db = open(file)
  assert.equal(await db.collection('docs').count(), 30)
  assert.equal((await db.collection('docs').search(embedOne('chess detail'), { k: 3 })).length, 3)
  db.close()

  // Exclusive files commit durably every checkpointEvery chunks, or on checkpoint().
  assert.equal(DEFAULT_CHECKPOINT, 1_000_000)
  db = Database.openExclusive(tmpDb())
  const ex = db.createCollection('docs', opts)
  assert.throws(() => ex.bulkLoad({ checkpointEvery: 0 }), { code: 'INVALID_ARGUMENT' })
  const exLoad = ex.bulkLoad({ checkpointEvery: 10 })
  await exLoad.add(items, embed)
  await exLoad.checkpoint()
  await exLoad.finish()
  assert.equal((await ex.search(embedOne('rust detail'), { k: 3 })).length, 3)
  db.close()
})

test('binary quantization', async () => {
  const db = open(tmpDb())
  const docs = db.createCollection('bin', { ...opts, quantization: 'binary', sealThreshold: 64 })
  assert.equal(docs.info().quantization, 'binary')
  const items = Array.from({ length: 24 }, (_, i) => ({ key: `doc-${i}`, content: article(TOPICS[i % 6], i) }))
  await docs.upsertMany(items, counting(), { batchSize: 8 })
  await db.waitForMaintenance()
  assert.ok(docs.stats().sealedSegments >= 1)
  const hits = await docs.search(embedOne('gardening detail'), { k: 5 })
  assert.equal(hits.length, 5)
  assert.ok(hits.every((h) => h.text.includes('gardening')))
  assert.throws(() => db.createCollection('bad', { dim: 8, quantization: 'int4' }), { code: 'INVALID_ARGUMENT' })
  db.close()
})
