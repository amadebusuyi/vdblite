// Package vdblite is an embedded, single-file vector database: open a
// file, store documents, and search them by meaning, with no server.
//
// Documents are split into chunks along their structure (Markdown
// headings, HTML, code, paragraphs); search returns the matching chunks,
// the passages around them, their sections, or whole documents ranked by
// their best chunks. Bring vectors from any embedding model.
//
// Every method is safe to call from many goroutines. Calls block; the
// heavy ones (Optimize, BulkLoad.Finish) use every core.
package vdblite

// #cgo CFLAGS: -I${SRCDIR}/include
// #include <stdlib.h>
// #include "vdblite.h"
import "C"

import (
	"encoding/json"
	"errors"
	"fmt"
	"runtime"
	"unsafe"
)

// Version is the version of the vdblite library linked in.
func Version() string { return C.GoString(C.vdb_version()) }

// ---------------------------------------------------------------- errors

// Code identifies the kind of an Error.
type Code int

const (
	CodeStorage          Code = C.VDB_ERR_STORAGE
	CodeDimension        Code = C.VDB_ERR_DIMENSION
	CodeCollectionExists Code = C.VDB_ERR_COLLECTION_EXISTS
	CodeNoSuchCollection Code = C.VDB_ERR_NO_SUCH_COLLECTION
	CodeInvalidArgument  Code = C.VDB_ERR_INVALID_ARGUMENT
	CodeEmbedder         Code = C.VDB_ERR_EMBEDDER
	CodeStaleEmbeddings  Code = C.VDB_ERR_STALE_EMBEDDINGS
	CodeVersion          Code = C.VDB_ERR_VERSION
	CodeMaintenance      Code = C.VDB_ERR_MAINTENANCE
	CodeReadOnly         Code = C.VDB_ERR_READ_ONLY
	CodeClosed           Code = C.VDB_ERR_CLOSED
	CodePanic            Code = C.VDB_ERR_PANIC
)

var codeNames = map[Code]string{
	CodeStorage: "STORAGE", CodeDimension: "DIMENSION", CodeCollectionExists: "COLLECTION_EXISTS",
	CodeNoSuchCollection: "NO_SUCH_COLLECTION", CodeInvalidArgument: "INVALID_ARGUMENT",
	CodeEmbedder: "EMBEDDER", CodeStaleEmbeddings: "STALE_EMBEDDINGS", CodeVersion: "VERSION",
	CodeMaintenance: "MAINTENANCE", CodeReadOnly: "READ_ONLY", CodeClosed: "CLOSED", CodePanic: "PANIC",
}

func (c Code) String() string {
	if n, ok := codeNames[c]; ok {
		return n
	}
	return fmt.Sprintf("Code(%d)", int(c))
}

// Error is a failed call. Test its kind with errors.As and Code, or with
// errors.Is against a Code:
//
//	if errors.Is(err, vdblite.CodeNoSuchCollection) { ... }
type Error struct {
	Code    Code
	Message string
}

func (e *Error) Error() string { return fmt.Sprintf("vdblite: %s: %s", e.Code, e.Message) }

// Is reports whether target is this error's Code.
func (e *Error) Is(target error) bool {
	c, ok := target.(Code)
	return ok && c == e.Code
}

// Error lets a Code be the target of errors.Is.
func (c Code) Error() string { return c.String() }

// check turns a status into an error. It must run on the goroutine's
// thread that made the call, so callers lock the OS thread (see call).
func check(status C.int32_t) error {
	if status == C.VDB_OK {
		return nil
	}
	return &Error{Code: Code(status), Message: C.GoString(C.vdb_last_error())}
}

// call runs f, which makes one C call and returns its status, on one OS
// thread, so the thread-local error message belongs to that call.
func call(f func() C.int32_t) error {
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()
	return check(f())
}

// ---------------------------------------------------------------- C helpers

type cstrings []*C.char

// str returns a C copy of s, freed by free.
func (cs *cstrings) str(s string) *C.char {
	p := C.CString(s)
	*cs = append(*cs, p)
	return p
}

// opt is str, or NULL for "".
func (cs *cstrings) opt(s string) *C.char {
	if s == "" {
		return nil
	}
	return cs.str(s)
}

// json is str of v as JSON, or NULL for nil.
func (cs *cstrings) json(v any) (*C.char, error) {
	if v == nil {
		return nil, nil
	}
	b, err := json.Marshal(v)
	if err != nil {
		return nil, err
	}
	return cs.str(string(b)), nil
}

func (cs *cstrings) free() {
	for _, p := range *cs {
		C.free(unsafe.Pointer(p))
	}
}

// take decodes a JSON string from the library into v and frees it.
func take(s *C.char, v any) error {
	defer C.vdb_string_free(s)
	return json.Unmarshal([]byte(C.GoString(s)), v)
}

func floats(v []float32) (*C.float, C.size_t) {
	if len(v) == 0 {
		return nil, 0
	}
	return (*C.float)(unsafe.Pointer(&v[0])), C.size_t(len(v))
}

// flatten puts vectors back to back.
func flatten(dst []float32, vectors [][]float32) []float32 {
	for _, v := range vectors {
		dst = append(dst, v...)
	}
	return dst
}

// ---------------------------------------------------------------- types

// Meta is a document's metadata: strings, numbers, booleans, nil, or
// slices of those.
type Meta map[string]any

// Filter is a Mongo-style metadata filter, e.g.
//
//	vdblite.Filter{"lang": "en", "year": vdblite.Filter{"$gte": 2020}, "$or": []vdblite.Filter{...}}
//
// Operators: $eq $ne $in $nin $gt $gte $lt $lte $exists $and $or $not.
// Equality on a list field matches any element.
type Filter map[string]any

// Doc is a document to write.
type Doc struct {
	Key     string `json:"key"`
	Content string `json:"content"`
	Meta    Meta   `json:"meta,omitempty"`
	// Format: "markdown", "code", "html", "text", a file name, extension
	// or MIME type. Sniffed if empty.
	Format string `json:"format,omitempty"`
}

// Embedder turns texts into vectors, one per text, in order.
type Embedder func(texts []string) ([][]float32, error)

// OpenOptions configures Open. The zero value opens for writing.
type OpenOptions struct {
	// ReadOnly opens an existing file alongside the process writing it,
	// following its commits.
	ReadOnly bool
	// Exclusive shuts out every other process, readers included, for
	// faster small commits.
	Exclusive bool
	// CacheBytes of file pages to cache; 0 for the default (256 MiB).
	CacheBytes uint64
}

// Chunking says how documents are split. The zero value is structural
// chunking with the default sizes.
type Chunking struct {
	// Kind: "structural" (default) or "none" (each document is one chunk).
	Kind          string `json:"kind,omitempty"`
	TargetTokens  int    `json:"targetTokens,omitempty"`
	OverlapTokens int    `json:"overlapTokens,omitempty"`
}

// CollectionOptions configures a new collection. Only Dim is required.
type CollectionOptions struct {
	Dim int `json:"dim"`
	// Metric: "cosine" (default), "dot" or "l2".
	Metric string `json:"metric,omitempty"`
	// Quantization: "int8" (default: int8 graph walk, f16 re-ranking),
	// "binary" (1-bit graph walk, f16 re-ranking) or "none" (f32).
	Quantization string    `json:"quantization,omitempty"`
	Chunking     *Chunking `json:"chunking,omitempty"`
	// SealThreshold: chunks buffered before they're sealed into a graph.
	SealThreshold int `json:"sealThreshold,omitempty"`
	// Lexical keeps a BM25 keyword index for hybrid search (default true).
	Lexical        *bool `json:"lexical,omitempty"`
	M              int   `json:"m,omitempty"`
	EfConstruction int   `json:"efConstruction,omitempty"`
	EfSearch       int   `json:"efSearch,omitempty"`
}

// CollectionInfo describes a collection.
type CollectionInfo struct {
	Name          string `json:"name"`
	Dim           int    `json:"dim"`
	Metric        string `json:"metric"`
	Quantization  string `json:"quantization"`
	Chunking      string `json:"chunking"`
	TargetTokens  int    `json:"targetTokens,omitempty"`
	OverlapTokens int    `json:"overlapTokens,omitempty"`
}

// IndexStats describes a collection's index.
type IndexStats struct {
	Chunks         int   `json:"chunks"`
	SealedSegments int   `json:"sealedSegments"`
	Buffered       int   `json:"buffered"`
	MemoryBytes    int64 `json:"memoryBytes"`
	// MaintenanceError is the last background maintenance failure, until
	// WaitForMaintenance reports it.
	MaintenanceError string `json:"maintenanceError,omitempty"`
}

// SearchOptions refine a search. A nil *SearchOptions means the defaults.
type SearchOptions struct {
	// K results to return (default 10).
	K int `json:"k,omitempty"`
	// Ef widens the graph search: slower, more accurate.
	Ef     int    `json:"ef,omitempty"`
	Filter Filter `json:"filter,omitempty"`
	// Text makes the search hybrid: chunks are also ranked by BM25
	// keyword match and the rankings fused.
	Text string `json:"text,omitempty"`
}

// ChunkHit is a matching chunk.
type ChunkHit struct {
	Key     string `json:"key"`
	DocID   uint64 `json:"docId"`
	ChunkID uint32 `json:"chunkId"`
	Ordinal uint32 `json:"ordinal"`
	// ByteStart and ByteEnd are UTF-8 byte offsets in the document.
	ByteStart int `json:"byteStart"`
	ByteEnd   int `json:"byteEnd"`
	// Score: higher is better. Cosine similarity, dot product or negative
	// squared L2; in a hybrid search, the fused rank score.
	Score float32 `json:"score"`
	// VectorScore is nil if only the keyword search found this chunk.
	VectorScore *float32 `json:"vectorScore"`
	// LexicalScore is nil if the keyword search didn't match it.
	LexicalScore *float32 `json:"lexicalScore"`
	Text         string   `json:"text"`
	HeadingPath  []string `json:"headingPath"`
	Meta         Meta     `json:"meta"`
}

// DocHit is a document ranked by its chunks.
type DocHit struct {
	Key   string  `json:"key"`
	DocID uint64  `json:"docId"`
	Score float32 `json:"score"`
	Meta  Meta    `json:"meta"`
	// Chunks are the document's matching chunks, best first.
	Chunks []ChunkHit `json:"chunks"`
}

// Passage is a hit widened to neighbouring chunks or its section.
type Passage struct {
	Key       string  `json:"key"`
	DocID     uint64  `json:"docId"`
	ByteStart int     `json:"byteStart"`
	ByteEnd   int     `json:"byteEnd"`
	Score     float32 `json:"score"`
	Text      string  `json:"text"`
	// Ordinals of the chunks covered: [start, end).
	Ordinals    [2]uint32 `json:"ordinals"`
	HeadingPath []string  `json:"headingPath"`
	Meta        Meta      `json:"meta"`
}

// Document is a stored document with its content.
type Document struct {
	Key        string `json:"key"`
	DocID      uint64 `json:"docId"`
	Content    string `json:"content"`
	Meta       Meta   `json:"meta"`
	Version    uint64 `json:"version"`
	CreatedMs  uint64 `json:"createdMs"`
	UpdatedMs  uint64 `json:"updatedMs"`
	Format     string `json:"format"`
	ChunkCount int    `json:"chunkCount"`
}

// DocInfo summarises a stored document.
type DocInfo struct {
	Key           string `json:"key"`
	DocID         uint64 `json:"docId"`
	Meta          Meta   `json:"meta"`
	Version       uint64 `json:"version"`
	CreatedMs     uint64 `json:"createdMs"`
	UpdatedMs     uint64 `json:"updatedMs"`
	ContentLength int64  `json:"contentLength"`
	ChunkCount    int    `json:"chunkCount"`
}

// ChunkInfo describes one chunk of a stored document.
type ChunkInfo struct {
	ChunkID      uint32   `json:"chunkId"`
	Ordinal      uint32   `json:"ordinal"`
	ByteStart    int      `json:"byteStart"`
	ByteEnd      int      `json:"byteEnd"`
	OverlapStart int      `json:"overlapStart"`
	HeadingPath  []string `json:"headingPath"`
	Tokens       int      `json:"tokens"`
}

// WriteResult is the outcome of one op in DB.Write.
type WriteResult struct {
	// Op: "upsert" or "delete".
	Op         string `json:"op"`
	Collection string `json:"collection"`
	Key        string `json:"key"`
	// For upserts.
	DocID   uint64 `json:"docId"`
	Version uint64 `json:"version"`
	Chunks  int    `json:"chunks"`
	// Embedded counts the chunks that took new vectors; the rest were
	// unchanged and kept theirs.
	Embedded int `json:"embedded"`
	// Changed is false when content and metadata were identical.
	Changed bool `json:"changed"`
	// For deletes: whether the document existed.
	Deleted bool `json:"deleted"`
}

// ---------------------------------------------------------------- database

// DB is an open database file.
type DB struct {
	h *C.vdb_db
}

// Open opens (or creates) the database file at path. One process writes
// a file at a time; others may open it with ReadOnly alongside. opts may
// be nil.
func Open(path string, opts *OpenOptions) (*DB, error) {
	if opts == nil {
		opts = &OpenOptions{}
	}
	var flags C.uint32_t
	if opts.ReadOnly {
		flags |= C.VDB_OPEN_READ_ONLY
	}
	if opts.Exclusive {
		flags |= C.VDB_OPEN_EXCLUSIVE
	}
	var cs cstrings
	defer cs.free()
	var h *C.vdb_db
	if err := call(func() C.int32_t {
		return C.vdb_open(cs.str(path), flags, C.uint64_t(opts.CacheBytes), &h)
	}); err != nil {
		return nil, err
	}
	db := &DB{h: h}
	runtime.SetFinalizer(db, func(db *DB) { C.vdb_db_free(db.h) })
	return db, nil
}

// Close closes the database; later calls on it and its collections fail
// with CodeClosed.
func (db *DB) Close() error {
	C.vdb_close(db.h)
	runtime.KeepAlive(db)
	return nil
}

// ReadOnly reports whether the database was opened read-only.
func (db *DB) ReadOnly() (bool, error) {
	var out C.bool
	err := call(func() C.int32_t { return C.vdb_is_read_only(db.h, &out) })
	runtime.KeepAlive(db)
	return bool(out), err
}

// jsonCall makes a C call that returns a JSON string, and decodes it.
func jsonCall(v any, f func(out **C.char) C.int32_t) error {
	var out *C.char
	if err := call(func() C.int32_t { return f(&out) }); err != nil {
		return err
	}
	return take(out, v)
}

// Collections returns the collection names.
func (db *DB) Collections() ([]string, error) {
	var names []string
	err := jsonCall(&names, func(out **C.char) C.int32_t { return C.vdb_collections(db.h, out) })
	runtime.KeepAlive(db)
	return names, err
}

func (db *DB) wrap(h *C.vdb_collection, name string) *Collection {
	c := &Collection{h: h, db: db, name: name}
	runtime.SetFinalizer(c, func(c *Collection) { C.vdb_collection_free(c.h) })
	return c
}

// CreateCollection creates a collection.
func (db *DB) CreateCollection(name string, opts CollectionOptions) (*Collection, error) {
	var cs cstrings
	defer cs.free()
	o, err := cs.json(opts)
	if err != nil {
		return nil, err
	}
	var h *C.vdb_collection
	err = call(func() C.int32_t { return C.vdb_create_collection(db.h, cs.str(name), o, &h) })
	runtime.KeepAlive(db)
	if err != nil {
		return nil, err
	}
	return db.wrap(h, name), nil
}

// Collection returns an existing collection (CodeNoSuchCollection if
// there is none).
func (db *DB) Collection(name string) (*Collection, error) {
	var cs cstrings
	defer cs.free()
	var h *C.vdb_collection
	err := call(func() C.int32_t { return C.vdb_get_collection(db.h, cs.str(name), &h) })
	runtime.KeepAlive(db)
	if err != nil {
		return nil, err
	}
	return db.wrap(h, name), nil
}

// DropCollection deletes a collection and everything in it.
func (db *DB) DropCollection(name string) error {
	var cs cstrings
	defer cs.free()
	err := call(func() C.int32_t { return C.vdb_drop_collection(db.h, cs.str(name)) })
	runtime.KeepAlive(db)
	return err
}

// Op is one change in DB.Write: see Upsert and Delete.
type Op struct {
	op         string
	collection string
	doc        Doc
	vectors    [][]float32
	hashes     []string
}

// Upsert inserts or replaces a document, with one vector per chunk (see
// Collection.EmbedInputs).
func Upsert(c *Collection, doc Doc, vectors [][]float32) Op {
	return Op{op: "upsert", collection: c.name, doc: doc, vectors: vectors}
}

// Delete removes a document.
func Delete(c *Collection, key string) Op {
	return Op{op: "delete", collection: c.name, doc: Doc{Key: key}}
}

type opJSON struct {
	Op         string   `json:"op"`
	Collection string   `json:"collection"`
	Key        string   `json:"key"`
	Content    *string  `json:"content,omitempty"`
	Meta       Meta     `json:"meta,omitempty"`
	Format     string   `json:"format,omitempty"`
	Vectors    *int     `json:"vectors,omitempty"`
	Embeddings []string `json:"embeddings,omitempty"`
}

// Write applies upserts and deletes atomically, in one transaction.
func (db *DB) Write(ops ...Op) ([]WriteResult, error) {
	items := make([]opJSON, len(ops))
	var flat []float32
	for i, o := range ops {
		items[i] = opJSON{Op: o.op, Collection: o.collection, Key: o.doc.Key}
		if o.op != "upsert" {
			continue
		}
		items[i].Content, items[i].Meta, items[i].Format = &ops[i].doc.Content, o.doc.Meta, o.doc.Format
		if o.hashes != nil {
			items[i].Embeddings = o.hashes
		} else {
			n := len(o.vectors)
			items[i].Vectors = &n
		}
		flat = flatten(flat, o.vectors)
	}
	var cs cstrings
	defer cs.free()
	j, err := cs.json(items)
	if err != nil {
		return nil, err
	}
	var out struct {
		Results []WriteResult `json:"results"`
		Warning *string       `json:"warning"`
	}
	p, n := floats(flat)
	err = jsonCall(&out, func(o **C.char) C.int32_t { return C.vdb_write(db.h, j, p, n, o) })
	runtime.KeepAlive(db)
	runtime.KeepAlive(flat)
	if err != nil {
		return nil, err
	}
	if out.Warning != nil {
		// Committed; only sealing the write buffer afterwards failed.
		return out.Results, &Error{Code: CodeMaintenance, Message: *out.Warning}
	}
	return out.Results, nil
}

// Optimize seals buffered chunks and merges small segments in every
// collection, using every core.
func (db *DB) Optimize() error {
	err := call(func() C.int32_t { return C.vdb_optimize(db.h) })
	runtime.KeepAlive(db)
	return err
}

// WaitForMaintenance blocks until background sealing and compaction catch
// up with every commit so far.
func (db *DB) WaitForMaintenance() error {
	err := call(func() C.int32_t { return C.vdb_wait_for_maintenance(db.h) })
	runtime.KeepAlive(db)
	return err
}

// CompactFile rewrites a file nobody has open without its free space, and
// reports whether it shrank.
func CompactFile(path string) (bool, error) {
	var cs cstrings
	defer cs.free()
	var shrank C.bool
	err := call(func() C.int32_t { return C.vdb_compact_file(cs.str(path), &shrank) })
	return bool(shrank), err
}

// ---------------------------------------------------------------- collection

// Collection is a collection in an open database.
type Collection struct {
	h    *C.vdb_collection
	db   *DB
	name string
}

// Name returns the collection's name.
func (c *Collection) Name() string { return c.name }

// Info describes the collection.
func (c *Collection) Info() (CollectionInfo, error) {
	var info CollectionInfo
	err := jsonCall(&info, func(out **C.char) C.int32_t { return C.vdb_collection_info(c.h, out) })
	runtime.KeepAlive(c)
	return info, err
}

// Stats describes the collection's index.
func (c *Collection) Stats() (IndexStats, error) {
	var s IndexStats
	err := jsonCall(&s, func(out **C.char) C.int32_t { return C.vdb_collection_stats(c.h, out) })
	runtime.KeepAlive(c)
	return s, err
}

// EmbedInputs returns the texts to embed for content, one per chunk.
// format may be empty.
func (c *Collection) EmbedInputs(content, format string) ([]string, error) {
	var cs cstrings
	defer cs.free()
	var texts []string
	err := jsonCall(&texts, func(out **C.char) C.int32_t {
		return C.vdb_embed_inputs(c.h, cs.str(content), cs.opt(format), out)
	})
	runtime.KeepAlive(c)
	return texts, err
}

// Upsert inserts or replaces one document, with one vector per chunk.
func (c *Collection) Upsert(doc Doc, vectors [][]float32) (WriteResult, error) {
	r, err := c.db.Write(Upsert(c, doc, vectors))
	if len(r) == 0 {
		return WriteResult{}, err
	}
	return r[0], err
}

// UpsertText inserts or replaces one document, calling embed for the
// chunks that need vectors. An edited document re-embeds only the chunks
// whose text changed.
func (c *Collection) UpsertText(doc Doc, embed Embedder) (WriteResult, error) {
	for attempt := 0; ; attempt++ {
		var plan []struct {
			Hash string `json:"hash"`
			Text string `json:"text"`
		}
		var cs cstrings
		d, err := cs.json(doc)
		if err == nil {
			err = jsonCall(&plan, func(out **C.char) C.int32_t { return C.vdb_plan_embeddings(c.h, d, out) })
		}
		cs.free()
		runtime.KeepAlive(c)
		if err != nil {
			return WriteResult{}, err
		}
		hashes := make([]string, len(plan))
		texts := make([]string, len(plan))
		for i, p := range plan {
			hashes[i], texts[i] = p.Hash, p.Text
		}
		var vectors [][]float32
		if len(texts) > 0 {
			if vectors, err = embed(texts); err != nil {
				return WriteResult{}, &Error{Code: CodeEmbedder, Message: err.Error()}
			}
			if len(vectors) != len(texts) {
				return WriteResult{}, &Error{Code: CodeEmbedder, Message: fmt.Sprintf(
					"the embedder returned %d vectors for %d texts", len(vectors), len(texts))}
			}
		}
		op := Op{op: "upsert", collection: c.name, doc: doc, vectors: vectors, hashes: hashes}
		r, err := c.db.Write(op)
		// Another writer changed the document's chunks after planning.
		if errors.Is(err, CodeStaleEmbeddings) && attempt < 3 {
			continue
		}
		if len(r) == 0 {
			return WriteResult{}, err
		}
		return r[0], err
	}
}

// Delete removes a document and reports whether it existed.
func (c *Collection) Delete(key string) (bool, error) {
	r, err := c.db.Write(Delete(c, key))
	if len(r) == 0 {
		return false, err
	}
	return r[0].Deleted, err
}

func (c *Collection) search(query []float32, opts *SearchOptions, extra map[string]any, v any) error {
	o := map[string]any{}
	if opts != nil {
		b, err := json.Marshal(opts)
		if err != nil {
			return err
		}
		if err := json.Unmarshal(b, &o); err != nil {
			return err
		}
	}
	for k, x := range extra {
		o[k] = x
	}
	var cs cstrings
	defer cs.free()
	j, err := cs.json(o)
	if err != nil {
		return err
	}
	p, n := floats(query)
	err = jsonCall(v, func(out **C.char) C.int32_t { return C.vdb_search(c.h, p, n, j, out) })
	runtime.KeepAlive(c)
	runtime.KeepAlive(query)
	return err
}

// Search returns the chunks nearest to query, best first.
func (c *Collection) Search(query []float32, opts *SearchOptions) ([]ChunkHit, error) {
	var hits []ChunkHit
	err := c.search(query, opts, map[string]any{"mode": "chunks"}, &hits)
	return hits, err
}

// SearchDocuments ranks documents by their best chunk (agg "max", the
// default) or by the sum of their top chunks (agg "sum").
func (c *Collection) SearchDocuments(query []float32, opts *SearchOptions, agg string, top int) ([]DocHit, error) {
	extra := map[string]any{"mode": "documents"}
	if agg != "" {
		extra["agg"] = agg
	}
	if top > 0 {
		extra["top"] = top
	}
	var hits []DocHit
	err := c.search(query, opts, extra, &hits)
	return hits, err
}

// SearchExpand returns hits widened by neighbors chunks on each side,
// with overlapping passages merged.
func (c *Collection) SearchExpand(query []float32, neighbors int, opts *SearchOptions) ([]Passage, error) {
	var ps []Passage
	err := c.search(query, opts, map[string]any{"mode": "expand", "neighbors": neighbors}, &ps)
	return ps, err
}

// SearchSections returns the whole heading section around each hit.
func (c *Collection) SearchSections(query []float32, opts *SearchOptions) ([]Passage, error) {
	var ps []Passage
	err := c.search(query, opts, map[string]any{"mode": "sections"}, &ps)
	return ps, err
}

// Get returns the document with key, or nil if there is none.
func (c *Collection) Get(key string) (*Document, error) {
	var cs cstrings
	defer cs.free()
	var d *Document
	err := jsonCall(&d, func(out **C.char) C.int32_t { return C.vdb_get(c.h, cs.str(key), out) })
	runtime.KeepAlive(c)
	return d, err
}

// Chunks describes the chunks of the document with key.
func (c *Collection) Chunks(key string) ([]ChunkInfo, error) {
	var cs cstrings
	defer cs.free()
	var chunks []ChunkInfo
	err := jsonCall(&chunks, func(out **C.char) C.int32_t { return C.vdb_chunks(c.h, cs.str(key), out) })
	runtime.KeepAlive(c)
	return chunks, err
}

// Scan summarises the documents matching filter (nil: all of them).
func (c *Collection) Scan(filter Filter) ([]DocInfo, error) {
	var cs cstrings
	defer cs.free()
	var f *C.char
	if filter != nil {
		var err error
		if f, err = cs.json(filter); err != nil {
			return nil, err
		}
	}
	var docs []DocInfo
	err := jsonCall(&docs, func(out **C.char) C.int32_t { return C.vdb_scan(c.h, f, out) })
	runtime.KeepAlive(c)
	return docs, err
}

// Count returns the number of documents.
func (c *Collection) Count() (int, error) {
	var n C.uint64_t
	err := call(func() C.int32_t { return C.vdb_count(c.h, &n) })
	runtime.KeepAlive(c)
	return int(n), err
}

// HasLexical reports whether the collection keeps a keyword index.
func (c *Collection) HasLexical() (bool, error) {
	var out C.bool
	err := call(func() C.int32_t { return C.vdb_has_lexical(c.h, &out) })
	runtime.KeepAlive(c)
	return bool(out), err
}

// EnableLexical builds the keyword index from stored text, without
// re-embedding; false if it already existed.
func (c *Collection) EnableLexical() (bool, error) {
	var built C.bool
	err := call(func() C.int32_t { return C.vdb_enable_lexical(c.h, &built) })
	runtime.KeepAlive(c)
	return bool(built), err
}

// ---------------------------------------------------------------- bulk load

// DefaultCheckpoint is the chunks between durable commits of a bulk load,
// by default.
var DefaultCheckpoint = int(C.vdb_default_checkpoint())

// BulkLoad is a bulk load in progress. Always call Finish.
type BulkLoad struct {
	h *C.vdb_bulk
}

// BulkDoc is a document for BulkLoad.Add, with one vector per chunk.
type BulkDoc struct {
	Doc
	Vectors [][]float32
}

// BulkLoad starts a bulk load: added documents are readable at once and
// searchable after Finish, which builds the index once with every core.
// checkpointEvery: chunks between durable commits in a file opened with
// Exclusive (0 for DefaultCheckpoint). One bulk load per collection at a
// time.
func (c *Collection) BulkLoad(checkpointEvery int) (*BulkLoad, error) {
	var h *C.vdb_bulk
	err := call(func() C.int32_t { return C.vdb_bulk_load(c.h, C.uint64_t(checkpointEvery), &h) })
	runtime.KeepAlive(c)
	if err != nil {
		return nil, err
	}
	b := &BulkLoad{h: h}
	runtime.SetFinalizer(b, func(b *BulkLoad) { C.vdb_bulk_free(b.h) })
	return b, nil
}

// Add inserts or replaces documents in one transaction, and returns how
// many were written.
func (b *BulkLoad) Add(docs []BulkDoc) (int, error) {
	type item struct {
		Doc
		Vectors int `json:"vectors"`
	}
	items := make([]item, len(docs))
	var flat []float32
	for i, d := range docs {
		items[i] = item{Doc: d.Doc, Vectors: len(d.Vectors)}
		flat = flatten(flat, d.Vectors)
	}
	var cs cstrings
	defer cs.free()
	j, err := cs.json(items)
	if err != nil {
		return 0, err
	}
	p, n := floats(flat)
	var written C.uint64_t
	err = call(func() C.int32_t { return C.vdb_bulk_add(b.h, j, p, n, &written) })
	runtime.KeepAlive(b)
	runtime.KeepAlive(flat)
	return int(written), err
}

// Checkpoint makes everything added so far durable (only matters with
// Exclusive).
func (b *BulkLoad) Checkpoint() error {
	err := call(func() C.int32_t { return C.vdb_bulk_checkpoint(b.h) })
	runtime.KeepAlive(b)
	return err
}

// Finish builds the index over everything added and makes it searchable.
// Other writes wait until it's done; reads and searches don't.
func (b *BulkLoad) Finish() error {
	err := call(func() C.int32_t { return C.vdb_bulk_finish(b.h) })
	runtime.KeepAlive(b)
	return err
}

// Finished reports whether Finish has been called.
func (b *BulkLoad) Finished() bool {
	var out C.bool
	_ = call(func() C.int32_t { return C.vdb_bulk_finished(b.h, &out) })
	runtime.KeepAlive(b)
	return bool(out)
}
