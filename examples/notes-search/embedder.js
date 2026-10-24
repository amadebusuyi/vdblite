// A local sentence-embedding model (all-MiniLM-L6-v2, 384 dimensions) via
// transformers.js. Runs on the CPU with no API key; the model (~90 MB) is
// downloaded on first use and cached.

import { env, pipeline } from '@huggingface/transformers'

export const MODEL = 'Xenova/all-MiniLM-L6-v2'
export const DIM = 384

// Keep the model cache next to this project rather than inside node_modules,
// so reinstalling packages doesn't trigger a re-download.
env.cacheDir = new URL('./.model-cache/', import.meta.url).pathname

let extractor

/** Embeds texts into unit-length vectors, one Float32Array per text. */
export async function embed(texts) {
  extractor ??= pipeline('feature-extraction', MODEL, { dtype: 'fp32' })
  const model = await extractor
  // One text at a time: a batch is padded to its longest text, and on CPU
  // that waste costs more than batching saves (measured: 14.5 ms per chunk
  // singly vs 30+ ms in batches of 32 on mixed-length chunks).
  const vectors = []
  for (const text of texts) {
    const out = await model(text, { pooling: 'mean', normalize: true })
    vectors.push(out.data.slice(0, DIM))
  }
  return vectors
}
