// Post-release check: run in an empty project after `npm install vdblite`.
import fs from 'node:fs'
import { open } from 'vdblite'

/** Toy embedding: counts of word lengths. Enough to rank the two documents. */
const embed = async (texts) =>
  texts.map((t) => {
    const v = new Array(8).fill(0)
    for (const w of t.split(/\W+/)) v[w.length % 8] += 1
    return v
  })

fs.rmSync('smoke.vdb', { force: true })
const db = open('smoke.vdb')
const docs = db.getOrCreateCollection('docs', { dim: 8, metric: 'cosine' })
await docs.upsert({ key: 'install', content: '# Install\n\nRun npm install vdblite.' }, embed)
await docs.upsert({ key: 'pasta', content: '# Cooking\n\nBoil the pasta for ten minutes.' }, embed)
const [hit] = await docs.searchText('npm install vdblite', embed, { k: 1 })
db.close()
if (hit?.key !== 'install') throw new Error(`unexpected top hit: ${JSON.stringify(hit)}`)
console.log(`ok: vdblite on ${process.platform}-${process.arch}, top hit ${hit.key}`)
