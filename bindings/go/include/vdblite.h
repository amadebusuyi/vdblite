/*
 * vdblite: an embedded, single-file vector database. C API.
 *
 * Conventions
 * - Functions return VDB_OK (0) or an error code; vdb_last_error() gives
 *   the message on the same thread.
 * - Strings in are NUL-terminated UTF-8. Structured values (options,
 *   documents, metadata, filters, results) are JSON, in the same shapes as
 *   the Node.js API (camelCase fields).
 * - Strings out (char **out) are allocated by the library: free each with
 *   vdb_string_free.
 * - Vectors travel as one flat float array. A JSON field ("vectors": n)
 *   says how many vectors each item takes, in order; each is as long as
 *   its collection's dimension.
 * - Handles are thread-safe. Freeing a handle must not race with other
 *   calls on it. Calls block; run them on your own threads.
 *
 * Licensed under Apache-2.0.
 */
#ifndef VDBLITE_H
#define VDBLITE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------- status */

#define VDB_OK 0
#define VDB_ERR_STORAGE 1            /* the file or a transaction failed */
#define VDB_ERR_DIMENSION 2          /* a vector had the wrong length */
#define VDB_ERR_COLLECTION_EXISTS 3
#define VDB_ERR_NO_SUCH_COLLECTION 4
#define VDB_ERR_INVALID_ARGUMENT 5
#define VDB_ERR_EMBEDDER 6
#define VDB_ERR_STALE_EMBEDDINGS 7   /* the document changed since planning: plan again */
#define VDB_ERR_VERSION 8            /* the file's format version isn't supported */
#define VDB_ERR_MAINTENANCE 9        /* sealing or compacting failed; nothing was lost */
#define VDB_ERR_READ_ONLY 10
#define VDB_ERR_CLOSED 11
#define VDB_ERR_PANIC 12             /* a bug in vdblite: please report it */

/* The library version, e.g. "0.1.0". Static. */
const char *vdb_version(void);

/* The message of the last failed call on this thread ("" after a
 * successful call). Valid until the next call on this thread. */
const char *vdb_last_error(void);

/* Frees a string returned by the library. NULL is ignored. */
void vdb_string_free(char *s);

/* ------------------------------------------------------------- database */

typedef struct VdbDb vdb_db;
typedef struct VdbCollection vdb_collection;
typedef struct VdbBulk vdb_bulk;

#define VDB_OPEN_READ_ONLY 1u  /* follow another process's writes */
#define VDB_OPEN_EXCLUSIVE 2u  /* no other process, not even readers; faster small commits */

/* Opens (or creates) the database file at `path`. One process writes a
 * file at a time; others may open it with VDB_OPEN_READ_ONLY alongside.
 * `cache_bytes`: bytes of file pages to cache, or 0 for the default. */
int32_t vdb_open(const char *path, uint32_t flags, uint64_t cache_bytes, vdb_db **out);

/* Closes the database: later calls on it and its collections fail with
 * VDB_ERR_CLOSED. The handle stays valid until vdb_db_free. */
void vdb_close(const vdb_db *db);

/* Closes (if still open) and frees the handle. NULL is ignored. */
void vdb_db_free(vdb_db *db);

int32_t vdb_is_read_only(const vdb_db *db, bool *out);

/* The collection names: a JSON array of strings. */
int32_t vdb_collections(const vdb_db *db, char **out);

/* Creates a collection. `options_json`:
 *   {"dim": 768, "metric": "cosine" | "dot" | "l2",
 *    "quantization": "int8" | "binary" | "none",
 *    "chunking": {"kind": "structural" | "none", "targetTokens": 400, "overlapTokens": 50},
 *    "sealThreshold": 65536, "lexical": true, "m": 16, "efConstruction": 200, "efSearch": 64}
 * Everything but "dim" is optional. */
int32_t vdb_create_collection(const vdb_db *db, const char *name, const char *options_json,
                              vdb_collection **out);

/* An existing collection, or VDB_ERR_NO_SUCH_COLLECTION. */
int32_t vdb_get_collection(const vdb_db *db, const char *name, vdb_collection **out);

int32_t vdb_drop_collection(const vdb_db *db, const char *name);

/* Applies upserts and deletes atomically, in one transaction. `ops_json`
 * is an array of:
 *   {"op": "upsert", "collection", "key", "content", "meta"?, "format"?, "vectors": n}
 *       n vectors, one per chunk (see vdb_embed_inputs);
 *   {"op": "upsert", ..., "embeddings": ["hash", ...]}
 *       one vector per hash, for the chunks vdb_plan_embeddings asked for;
 *       unchanged chunks keep their stored vectors;
 *   {"op": "delete", "collection", "key"}
 * Vectors come from `vectors` (`n_floats` floats) in op order. `out` gets
 *   {"results": [{"op": "upsert", "key", "docId", "version", "chunks",
 *                 "embedded", "changed", ...} | {"op": "delete", "key", "deleted", ...}],
 *    "warning": null | "..."}  (a warning: committed, but sealing failed) */
int32_t vdb_write(const vdb_db *db, const char *ops_json, const float *vectors,
                  size_t n_floats, char **out);

/* Seals buffered chunks and merges small segments, using every core. */
int32_t vdb_optimize(const vdb_db *db);

/* Blocks until background sealing and compaction catch up. */
int32_t vdb_wait_for_maintenance(const vdb_db *db);

/* Rewrites a file nobody has open without its free space. `shrank` may be
 * NULL. */
int32_t vdb_compact_file(const char *path, bool *shrank);

/* ------------------------------------------------------------- collection */

/* Frees a collection handle. NULL is ignored. */
void vdb_collection_free(vdb_collection *c);

/* {"name", "dim", "metric", "quantization", "chunking", "targetTokens"?, "overlapTokens"?} */
int32_t vdb_collection_info(const vdb_collection *c, char **out);

/* {"chunks", "sealedSegments", "buffered", "memoryBytes", "maintenanceError"} */
int32_t vdb_collection_stats(const vdb_collection *c, char **out);

/* The texts to embed for `content`, one per chunk: a JSON array of strings.
 * `format` (may be NULL): "markdown", "code", "html", "text", a file name,
 * extension or MIME type. */
int32_t vdb_embed_inputs(const vdb_collection *c, const char *content, const char *format,
                         char **out);

/* The chunks of `doc_json` ({"key", "content", "meta"?, "format"?}) that
 * need new vectors: [{"hash", "text"}]. Embed the texts and pass the hashes
 * to vdb_write as "embeddings". */
int32_t vdb_plan_embeddings(const vdb_collection *c, const char *doc_json, char **out);

/* Searches with `query` (`dim` floats). `options_json` (may be NULL):
 *   {"mode": "chunks" | "documents" | "expand" | "sections", "k": 10, "ef",
 *    "filter": {...}, "text": "query words for hybrid search",
 *    "agg": "max" | "sum", "top": 3, "neighbors": 1}
 * Filters are Mongo-style: {"lang": "en", "year": {"$gte": 2020},
 * "tags": {"$in": ["a"]}, "$or": [...]}; operators $eq $ne $in $nin $gt
 * $gte $lt $lte $exists $and $or $not.
 * `out` gets a JSON array: chunk hits {"key", "docId", "chunkId", "ordinal",
 * "byteStart", "byteEnd", "score", "vectorScore", "lexicalScore", "text",
 * "headingPath", "meta"}, document hits {"key", "docId", "score", "meta",
 * "chunks"}, or passages {"key", "docId", "byteStart", "byteEnd", "score",
 * "text", "ordinals", "headingPath", "meta"}. */
int32_t vdb_search(const vdb_collection *c, const float *query, size_t dim,
                   const char *options_json, char **out);

/* The document with `key` ({"key", "docId", "content", "meta", "version",
 * "createdMs", "updatedMs", "format", "chunkCount"}), or "null". */
int32_t vdb_get(const vdb_collection *c, const char *key, char **out);

/* The chunks of a document: [{"chunkId", "ordinal", "byteStart", "byteEnd",
 * "overlapStart", "headingPath", "tokens"}]. */
int32_t vdb_chunks(const vdb_collection *c, const char *key, char **out);

/* Summaries of the documents matching `filter_json` (NULL: all). */
int32_t vdb_scan(const vdb_collection *c, const char *filter_json, char **out);

int32_t vdb_count(const vdb_collection *c, uint64_t *out);

int32_t vdb_has_lexical(const vdb_collection *c, bool *out);

/* Builds the keyword index from stored text. `built` (may be NULL) is false
 * if it already existed. */
int32_t vdb_enable_lexical(const vdb_collection *c, bool *built);

/* ------------------------------------------------------------- bulk load */

/* Chunks between durable commits of a bulk load, by default. */
uint64_t vdb_default_checkpoint(void);

/* Starts a bulk load: added documents are readable at once and searchable
 * after vdb_bulk_finish, which builds the index with every core. One per
 * collection at a time. `checkpoint_every`: chunks between durable commits
 * with VDB_OPEN_EXCLUSIVE, or 0 for the default. */
int32_t vdb_bulk_load(const vdb_collection *c, uint64_t checkpoint_every, vdb_bulk **out);

/* Adds documents in one transaction: `docs_json` is an array of
 * {"key", "content", "meta"?, "format"?, "vectors": n}, each taking n
 * vectors from `vectors`. `written` (may be NULL) gets the count. */
int32_t vdb_bulk_add(const vdb_bulk *b, const char *docs_json, const float *vectors,
                     size_t n_floats, uint64_t *written);

/* Makes everything added so far durable (matters with VDB_OPEN_EXCLUSIVE). */
int32_t vdb_bulk_checkpoint(const vdb_bulk *b);

/* Builds the index and makes everything added searchable. Other writes wait
 * until it's done; reads and searches don't. Always call it. */
int32_t vdb_bulk_finish(const vdb_bulk *b);

int32_t vdb_bulk_finished(const vdb_bulk *b, bool *out);

/* Frees a bulk load. One never finished keeps what was added, and
 * background maintenance indexes it. NULL is ignored. */
void vdb_bulk_free(vdb_bulk *b);

#ifdef __cplusplus
}
#endif

#endif /* VDBLITE_H */
