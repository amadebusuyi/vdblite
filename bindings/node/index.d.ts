/** A metadata value: scalars, or lists of scalars. */
export type MetaValue = string | number | boolean | null | Array<string | number | boolean | null>

export type Meta = Record<string, MetaValue>

export type Vector = number[] | Float32Array

/** Turns texts into vectors, one per text, in order. May be async. */
export type Embedder = (texts: string[]) => Vector[] | Promise<Vector[]>

export interface DocInput {
  /** Unique within the collection. */
  key: string
  content: string
  meta?: Meta
  /** "markdown", "code", "html", "text", a file name, extension or MIME type. Sniffed if omitted. */
  format?: string
}

export interface CollectionOptions {
  dim: number
  /** Default "cosine". */
  metric?: 'cosine' | 'dot' | 'l2'
  /**
   * Default "int8": the graph walk uses int8 codes and re-ranks from f16
   * vectors. "binary" walks 1-bit codes (an eighth of int8's size) and
   * re-ranks more candidates from f16. "none" walks full f32 vectors. Either
   * way the vectors are memory-mapped from the file.
   */
  quantization?: 'int8' | 'binary' | 'none'
  /** Default { kind: "structural", targetTokens: 400, overlapTokens: 50 }. */
  chunking?: { kind?: 'structural' | 'none'; targetTokens?: number; overlapTokens?: number }
  /** Chunks buffered before they're sealed into a graph. Default 65536. */
  sealThreshold?: number
  /** Keep a BM25 keyword index for hybrid search. Default true. */
  lexical?: boolean
  m?: number
  efConstruction?: number
  efSearch?: number
}

/**
 * Mongo-style metadata filter, e.g.
 * `{ lang: "en", year: { $gte: 2020 }, tags: { $in: ["a", "b"] }, $or: [...] }`.
 * Operators: $eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not.
 * Equality on a list field matches any element.
 */
export type Filter = { [field: string]: unknown }

export interface SearchOptions {
  /** Results to return. Default 10. */
  k?: number
  /** Graph beam width; higher is slower and more accurate. */
  ef?: number
  filter?: Filter
  /**
   * Query text for hybrid search: chunks are also ranked by BM25 keyword
   * match and the two rankings are fused. Finds exact words, identifiers and
   * rare names that embeddings miss.
   */
  text?: string
}

export interface ChunkHit {
  key: string
  docId: number
  chunkId: number
  ordinal: number
  /** UTF-8 byte offsets of the chunk in the document. */
  byteStart: number
  byteEnd: number
  /**
   * Higher is better. Vector search: cosine similarity, dot product, or
   * negative squared L2. Hybrid search: the fused rank score (compare
   * vectorScore and lexicalScore for absolute values).
   */
  score: number
  /** Vector similarity, or null if only the keyword search found this chunk. */
  vectorScore: number | null
  /** BM25 score, or null if the keyword search didn't match this chunk. */
  lexicalScore: number | null
  text: string
  headingPath: string[]
  meta: Meta
}

export interface Passage {
  key: string
  docId: number
  byteStart: number
  byteEnd: number
  score: number
  text: string
  /** Chunk ordinals covered: [start, end). */
  ordinals: [number, number]
  headingPath: string[]
  meta: Meta
}

export interface DocHit {
  key: string
  docId: number
  score: number
  meta: Meta
  /** The document's matching chunks, best first. */
  chunks: ChunkHit[]
}

export interface Document {
  key: string
  docId: number
  content: string
  meta: Meta
  version: number
  createdMs: number
  updatedMs: number
  format: 'markdown' | 'code' | 'html' | 'text'
  chunkCount: number
}

export interface DocInfo {
  key: string
  docId: number
  meta: Meta
  version: number
  createdMs: number
  updatedMs: number
  contentLength: number
  chunkCount: number
}

export interface ChunkInfo {
  chunkId: number
  ordinal: number
  byteStart: number
  byteEnd: number
  overlapStart: number
  headingPath: string[]
  tokens: number
}

export interface UpsertResult {
  op: 'upsert'
  collection: string
  key: string
  docId: number
  version: number
  chunks: number
  /** Chunks that needed new embeddings; the rest were unchanged and reused. */
  embedded: number
  /** False when content and metadata were identical. */
  changed: boolean
}

export interface DeleteResult {
  op: 'delete'
  collection: string
  key: string
  deleted: boolean
}

export interface Transaction {
  /** Queue an upsert. `embed` is an embedder, or one vector per chunk (see `embedInputs`). */
  upsert(collection: Collection, doc: DocInput, embed: Embedder | Vector[]): void
  delete(collection: Collection, key: string): void
}

export interface CollectionInfo {
  name: string
  dim: number
  metric: 'cosine' | 'dot' | 'l2'
  quantization: 'int8' | 'binary' | 'none'
  chunking: 'structural' | 'none'
  targetTokens?: number
  overlapTokens?: number
}

export interface OpenOptions {
  /** Bytes of file pages to cache in this process (default 256 MiB). */
  cacheBytes?: number
}

export interface BulkLoadOptions {
  /**
   * Chunks between durable commits when the file was opened with
   * `openExclusive` (default `DEFAULT_CHECKPOINT`, 1,000,000). A crash loses
   * at most this much of the load. Otherwise every `add` commits durably.
   */
  checkpointEvery?: number
}

/** A document for `BulkLoad.add`, optionally with one vector per chunk. */
export interface BulkDocInput extends DocInput {
  vectors?: Vector[]
}

export interface IndexStats {
  chunks: number
  sealedSegments: number
  buffered: number
  memoryBytes: number
  /** The last background maintenance failure, until reported by waitForMaintenance(). */
  maintenanceError?: string
}

/** Errors carry `code`: STORAGE, DIMENSION, COLLECTION_EXISTS, NO_SUCH_COLLECTION,
 * INVALID_ARGUMENT, EMBEDDER, STALE_EMBEDDINGS, VERSION, MAINTENANCE, READ_ONLY, CLOSED. */
export interface VdbliteError extends Error {
  code?: string
}

export declare class Database {
  /** Opens (or creates) a file for writing; other processes may open it read-only alongside. */
  static open(path: string, options?: OpenOptions): Database
  /** Opens for writing with no other process allowed; small commits are about twice as fast. */
  static openExclusive(path: string, options?: OpenOptions): Database
  /** Opens an existing file read-only, following the writing process's commits. */
  static openReadOnly(path: string, options?: OpenOptions): Database
  static compactFile(path: string): boolean
  readonly path: string
  readonly closed: boolean
  readonly readOnly: boolean
  close(): void
  collections(): string[]
  hasCollection(name: string): boolean
  createCollection(name: string, options: CollectionOptions): Collection
  collection(name: string): Collection
  getOrCreateCollection(name: string, options: CollectionOptions): Collection
  dropCollection(name: string): void
  optimize(): Promise<void>
  /** Waits for background sealing and compaction to catch up with every commit so far. */
  waitForMaintenance(): Promise<void>
  transaction<T>(
    fn: (tx: Transaction) => T | Promise<T>,
  ): Promise<{ value: T; results: Array<UpsertResult | DeleteResult> }>
}

export declare class Collection {
  readonly db: Database
  readonly name: string
  info(): CollectionInfo
  stats(): IndexStats
  embedInputs(content: string, format?: string): string[]
  hasLexical(): boolean
  /** Builds the keyword index from stored text, without re-embedding. */
  enableLexical(): Promise<boolean>
  upsert(doc: DocInput, embed: Embedder | Vector[]): Promise<UpsertResult>
  upsertMany(docs: DocInput[], embed: Embedder, options?: { batchSize?: number }): Promise<UpsertResult[]>
  delete(key: string): Promise<boolean>
  get(key: string): Promise<Document | null>
  chunks(key: string): Promise<ChunkInfo[]>
  scan(filter?: Filter): Promise<DocInfo[]>
  count(): Promise<number>
  /**
   * Starts a bulk load: added documents are readable at once and searchable
   * after `finish()`, which builds the index once with every core. One bulk
   * load per collection at a time (a second fails with INVALID_ARGUMENT).
   */
  bulkLoad(options?: BulkLoadOptions): BulkLoad
  search(query: Vector, options?: SearchOptions): Promise<ChunkHit[]>
  searchDocuments(query: Vector, options?: SearchOptions & { agg?: 'max' | 'sum'; top?: number }): Promise<DocHit[]>
  searchExpand(query: Vector, options?: SearchOptions & { neighbors?: number }): Promise<Passage[]>
  searchSections(query: Vector, options?: SearchOptions): Promise<Passage[]>
  /** Hybrid by default when the collection has a keyword index; pass hybrid: false to opt out. */
  searchText(text: string, embed: Embedder, options?: SearchOptions & { mode?: 'chunks'; hybrid?: boolean }): Promise<ChunkHit[]>
  searchText(text: string, embed: Embedder, options: SearchOptions & { mode: 'documents'; agg?: 'max' | 'sum'; top?: number; hybrid?: boolean }): Promise<DocHit[]>
  searchText(text: string, embed: Embedder, options: SearchOptions & { mode: 'expand' | 'sections'; neighbors?: number; hybrid?: boolean }): Promise<Passage[]>
}

/**
 * A bulk load in progress. If it is never finished, what was added is kept
 * and indexed by background maintenance; after a crash, every committed
 * `add` is kept and indexed at the next open.
 */
export declare class BulkLoad {
  readonly collection: Collection
  readonly finished: boolean
  /**
   * Inserts or replaces documents in one transaction; resolves to how many
   * were written. `embed` embeds every chunk; without it, each document
   * needs `vectors`, one per chunk (see `Collection.embedInputs`).
   */
  add(docs: BulkDocInput[], embed?: Embedder): Promise<number>
  /** Makes everything added so far durable (only matters with `openExclusive`). */
  checkpoint(): Promise<void>
  /**
   * Builds the index over everything added and makes it searchable. Other
   * writes to the database wait until it's done; reads and searches don't.
   */
  finish(): Promise<void>
}

/** Chunks between durable commits of a bulk load, by default (1,000,000). */
export declare const DEFAULT_CHECKPOINT: number

export declare function open(path: string, options?: OpenOptions): Database
