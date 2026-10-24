#!/usr/bin/env node
// notes-search: index a folder of notes, docs or code, then search it by
// meaning. Everything lives in one file (notes.vdb) next to where you run it.
//
//   node cli.js index <folder>          add or refresh a folder
//   node cli.js search <query> [flags]  --k 5  --files  --context  --ext md  --all
//   node cli.js forget <folder>         remove a folder from the index
//   node cli.js stats

import fs from 'node:fs'
import path from 'node:path'

import { open } from 'vecfile'

import { DIM, embed } from './embedder.js'

const DB_PATH = process.env.NOTES_DB ?? 'notes.vdb'
const EXTENSIONS = new Set([
  '.md', '.markdown', '.mdx', '.txt', '.html', '.htm',
  '.js', '.mjs', '.ts', '.tsx', '.py', '.rs', '.go', '.java', '.rb', '.c', '.h', '.cpp', '.sh', '.sql',
])
const SKIP_DIRS = new Set(['node_modules', '.git', 'target', 'dist', 'build', '.model-cache'])
const MAX_FILE_BYTES = 1_000_000
// With this model, related passages usually have similarity 0.4 or more;
// below this, and with no keyword match, the best result is probably noise.
const WEAK_SCORE = 0.35

async function collection(db) {
  const notes = db.getOrCreateCollection('notes', {
    dim: DIM,
    metric: 'cosine',
    // MiniLM reads at most 256 word pieces, so keep chunks well under that.
    chunking: { targetTokens: 180, overlapTokens: 30 },
  })
  // Indexes made before hybrid search existed get a keyword index once,
  // built from the stored text (nothing is re-embedded).
  if (!notes.hasLexical()) {
    process.stdout.write('building the keyword index (one time)… ')
    await notes.enableLexical()
    console.log('done')
  }
  return notes
}

/** Every indexable file under `root`, recursively. */
function* walk(root) {
  for (const entry of fs.readdirSync(root, { withFileTypes: true })) {
    if (entry.name.startsWith('.') && entry.name !== '.') continue
    const full = path.join(root, entry.name)
    if (entry.isDirectory()) {
      if (!SKIP_DIRS.has(entry.name)) yield* walk(full)
    } else if (entry.isFile() && EXTENSIONS.has(path.extname(entry.name).toLowerCase())) {
      yield full
    }
  }
}

async function index(folder) {
  const root = path.resolve(folder)
  if (!fs.existsSync(root) || !fs.statSync(root).isDirectory()) throw new Error(`not a folder: ${folder}`)
  const db = open(DB_PATH)
  const notes = await collection(db)
  const started = Date.now()

  // Files already indexed under an enclosing folder keep that folder as their
  // root, so indexing a subfolder doesn't relabel (and rewrite) them.
  const rootOf = new Map((await notes.scan()).map((d) => [d.key, d.meta.root]))

  const docs = []
  for (const file of walk(root)) {
    const stat = fs.statSync(file)
    if (stat.size > MAX_FILE_BYTES) continue
    const prev = rootOf.get(file)
    const fileRoot = prev && isWithin(root, prev) ? prev : root
    docs.push({
      key: file, // absolute path: unique across indexed folders
      content: fs.readFileSync(file, 'utf8'),
      format: path.extname(file), // tells the chunker how to split it
      meta: {
        root: fileRoot,
        path: path.relative(fileRoot, file),
        ext: path.extname(file).slice(1).toLowerCase(),
        modified: Math.round(stat.mtimeMs),
      },
    })
  }
  console.log(`indexing ${docs.length} files from ${root}`)

  let changed = 0
  let embedded = 0
  let chunks = 0
  const batch = 32
  for (let i = 0; i < docs.length; i += batch) {
    const results = await notes.upsertMany(docs.slice(i, i + batch), embed)
    for (const r of results) {
      if (r.changed) changed++
      embedded += r.embedded
      chunks += r.chunks
    }
    process.stdout.write(`\r  ${Math.min(i + batch, docs.length)}/${docs.length} files`)
  }
  process.stdout.write('\n')

  // Files under this folder that disappeared since the last run.
  const present = new Set(docs.map((d) => d.key))
  const stale = (await notes.scan()).filter((d) => isWithin(d.key, root) && !present.has(d.key))
  if (stale.length) {
    await db.transaction((tx) => stale.forEach((d) => tx.delete(notes, d.key)))
  }

  const secs = ((Date.now() - started) / 1000).toFixed(1)
  console.log(
    `done in ${secs}s: ${changed} of ${docs.length} files new or changed, ${stale.length} removed; ` +
      `embedded ${embedded} of ${chunks} chunks (the rest were unchanged)`,
  )
  db.close()
}

/** Whether `p` is `dir` or inside it. */
function isWithin(p, dir) {
  const rel = path.relative(dir, p)
  return rel === '' || (!rel.startsWith('..') && !path.isAbsolute(rel))
}

function parseFlags(args) {
  const flags = { k: 5, files: false, context: false, ext: null, all: false }
  const words = []
  for (let i = 0; i < args.length; i++) {
    const a = args[i]
    if (a === '--k') flags.k = Number(args[++i])
    else if (a === '--files') flags.files = true
    else if (a === '--context') flags.context = true
    else if (a === '--ext') flags.ext = args[++i].replace(/^\./, '')
    else if (a === '--all') flags.all = true
    else words.push(a)
  }
  return { query: words.join(' '), flags }
}

function snippet(text, max = 280) {
  const flat = text.replace(/\s+/g, ' ').trim()
  return flat.length > max ? `${flat.slice(0, max)}…` : flat
}

/** Why a chunk matched, in words: its similarity and whether keywords hit. */
function why(hit) {
  const parts = []
  if (hit.vectorScore != null) parts.push(`similarity ${hit.vectorScore.toFixed(2)}`)
  if (hit.lexicalScore != null) parts.push('keyword match')
  return parts.join(' · ')
}

async function search(args) {
  const { query, flags } = parseFlags(args)
  if (!query) throw new Error('usage: search <query> [--k 5] [--files] [--context] [--ext md] [--all]')
  const db = open(DB_PATH)
  const notes = await collection(db)
  if ((await notes.count()) === 0) {
    console.log('the index is empty: run `node cli.js index <folder>` first')
    return db.close()
  }
  const filter = flags.ext ? { ext: flags.ext } : undefined
  const [vector] = await embed([query])
  // Hybrid: vector similarity plus keyword (BM25) matches, fused.
  const options = { k: flags.k, filter, text: query }

  // A match is worth showing if it shares keywords with the query, or if
  // its meaning is close; otherwise say so instead of listing noise.
  const isStrong = (h) => h.lexicalScore != null || (h.vectorScore ?? 0) >= WEAK_SCORE
  let hits = await notes.search(vector, options)
  const best = Math.max(0, ...hits.map((h) => h.vectorScore ?? 0))
  const strong = hits.some(isStrong)
  // Once something really matches, drop the noise ranked around it.
  if (strong && !flags.all) hits = hits.filter(isStrong)
  if (hits.length === 0 || (!strong && !flags.all)) {
    console.log(`No close matches for "${query}".` + (hits.length ? ' (--all shows the nearest anyway)' : ''))
    return db.close()
  }
  if (!strong) console.log(`(weak matches only; best similarity ${best.toFixed(2)})\n`)

  if (flags.files) {
    const files = await notes.searchDocuments(vector, options)
    for (const [i, f] of files.entries()) {
      const where = path.join(path.basename(f.meta.root), f.meta.path)
      console.log(`${i + 1}. ${where}  (${why(f.chunks[0])}; ${f.chunks.length} matching chunks)`)
      console.log(`   ${snippet(f.chunks[0].text, 160)}\n`)
    }
  } else if (flags.context) {
    const passages = await notes.searchExpand(vector, { ...options, neighbors: 1 })
    for (const [i, p] of passages.entries()) {
      const where = path.join(path.basename(p.meta.root), p.meta.path)
      const heading = p.headingPath.length ? `  § ${p.headingPath.join(' › ')}` : ''
      console.log(`${i + 1}. ${where}${heading}`)
      console.log(`   ${snippet(p.text, 600)}\n`)
    }
  } else {
    for (const [i, h] of hits.entries()) {
      const where = path.join(path.basename(h.meta.root), h.meta.path)
      const heading = h.headingPath.length ? `  § ${h.headingPath.join(' › ')}` : ''
      console.log(`${i + 1}. ${where}${heading}  (${why(h)})`)
      console.log(`   ${snippet(h.text)}\n`)
    }
  }
  db.close()
}

async function forget(folder) {
  const root = path.resolve(folder)
  const db = open(DB_PATH)
  const notes = await collection(db)
  const docs = (await notes.scan()).filter((d) => isWithin(d.key, root))
  await db.transaction((tx) => docs.forEach((d) => tx.delete(notes, d.key)))
  console.log(`removed ${docs.length} files under ${root}`)
  db.close()
}

async function stats() {
  const db = open(DB_PATH)
  const notes = await collection(db)
  const docs = await notes.scan()
  const roots = new Map()
  for (const d of docs) roots.set(d.meta.root, (roots.get(d.meta.root) ?? 0) + 1)
  const s = notes.stats()
  const size = fs.statSync(DB_PATH).size
  console.log(`${DB_PATH}: ${(size / 1e6).toFixed(1)} MB, ${docs.length} files, ${s.chunks} chunks`)
  console.log(`index: ${s.sealedSegments} sealed segments, ${s.buffered} buffered chunks, ${(s.memoryBytes / 1e6).toFixed(1)} MB in memory`)
  for (const [root, n] of roots) console.log(`  ${root}: ${n} files`)
  db.close()
}

const [command, ...args] = process.argv.slice(2)
const commands = {
  index: () => index(args[0] ?? '.'),
  search: () => search(args),
  forget: () => forget(args[0] ?? '.'),
  stats: () => stats(),
}

if (!commands[command]) {
  console.log('usage: node cli.js <index <folder> | search <query> | forget <folder> | stats>')
  process.exit(command ? 1 : 0)
}
try {
  await commands[command]()
} catch (e) {
  console.error(`error: ${e.message}${e.code ? ` (${e.code})` : ''}`)
  process.exit(1)
}
