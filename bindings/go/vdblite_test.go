package vdblite_test

import (
	"errors"
	"path/filepath"
	"strings"
	"testing"

	"github.com/OWNER/vdblite/bindings/go"
)

// embed is a toy embedder: [length, count of "rust", 1].
func embed(texts []string) ([][]float32, error) {
	out := make([][]float32, len(texts))
	for i, t := range texts {
		out[i] = []float32{float32(len(t)), float32(strings.Count(t, "rust")), 1}
	}
	return out, nil
}

func open(t *testing.T) (*vdblite.DB, string) {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.vdb")
	db, err := vdblite.Open(path, nil)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { db.Close() })
	return db, path
}

func TestRoundTrip(t *testing.T) {
	db, path := open(t)
	docs, err := db.CreateCollection("docs", vdblite.CollectionOptions{Dim: 3})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.CreateCollection("docs", vdblite.CollectionOptions{Dim: 3}); !errors.Is(err, vdblite.CodeCollectionExists) {
		t.Fatalf("want COLLECTION_EXISTS, got %v", err)
	}

	r, err := docs.UpsertText(vdblite.Doc{
		Key: "readme", Content: "# Intro\n\nrust rust rust\n\n## Usage\n\nOther words here.",
		Meta: vdblite.Meta{"lang": "en", "year": 2024, "tags": []string{"a", "b"}},
	}, embed)
	if err != nil {
		t.Fatal(err)
	}
	if !r.Changed || r.Chunks == 0 || r.Embedded != r.Chunks {
		t.Fatalf("unexpected upsert result %+v", r)
	}
	// Unchanged: nothing re-embedded.
	r, err = docs.UpsertText(vdblite.Doc{
		Key: "readme", Content: "# Intro\n\nrust rust rust\n\n## Usage\n\nOther words here.",
		Meta: vdblite.Meta{"lang": "en", "year": 2024, "tags": []string{"a", "b"}},
	}, embed)
	if err != nil || r.Changed || r.Embedded != 0 {
		t.Fatalf("re-upsert: %+v, %v", r, err)
	}

	q, _ := embed([]string{"rust rust rust"})
	hits, err := docs.Search(q[0], &vdblite.SearchOptions{K: 5, Filter: vdblite.Filter{"year": vdblite.Filter{"$gte": 2020}}})
	if err != nil || len(hits) == 0 || hits[0].Key != "readme" {
		t.Fatalf("search: %+v, %v", hits, err)
	}
	if hits[0].VectorScore == nil || len(hits[0].HeadingPath) == 0 || hits[0].Meta["lang"] != "en" {
		t.Fatalf("hit fields: %+v", hits[0])
	}
	hits, err = docs.Search(q[0], &vdblite.SearchOptions{Filter: vdblite.Filter{"tags": "z"}})
	if err != nil || len(hits) != 0 {
		t.Fatalf("filtered search: %+v, %v", hits, err)
	}
	hits, err = docs.Search(q[0], &vdblite.SearchOptions{Text: "Usage"})
	if err != nil || len(hits) == 0 || hits[0].LexicalScore == nil {
		t.Fatalf("hybrid search: %+v, %v", hits, err)
	}
	dh, err := docs.SearchDocuments(q[0], nil, "sum", 2)
	if err != nil || len(dh) != 1 || len(dh[0].Chunks) == 0 {
		t.Fatalf("documents: %+v, %v", dh, err)
	}
	ps, err := docs.SearchSections(q[0], nil)
	if err != nil || len(ps) == 0 || !strings.Contains(ps[0].Text, "rust") {
		t.Fatalf("sections: %+v, %v", ps, err)
	}
	if _, err := docs.SearchExpand(q[0], 1, nil); err != nil {
		t.Fatal(err)
	}
	if _, err := docs.Search([]float32{1, 2}, nil); !errors.Is(err, vdblite.CodeDimension) {
		t.Fatalf("want DIMENSION, got %v", err)
	}

	d, err := docs.Get("readme")
	if err != nil || d == nil || !strings.HasPrefix(d.Content, "# Intro") || d.Format != "markdown" {
		t.Fatalf("get: %+v, %v", d, err)
	}
	if d, err := docs.Get("missing"); err != nil || d != nil {
		t.Fatalf("get missing: %+v, %v", d, err)
	}
	chunks, err := docs.Chunks("readme")
	if err != nil || len(chunks) != r.Chunks {
		t.Fatalf("chunks: %+v, %v", chunks, err)
	}
	infos, err := docs.Scan(vdblite.Filter{"lang": "en"})
	if err != nil || len(infos) != 1 {
		t.Fatalf("scan: %+v, %v", infos, err)
	}
	info, err := docs.Info()
	if err != nil || info.Dim != 3 || info.Quantization != "int8" || info.Chunking != "structural" {
		t.Fatalf("info: %+v, %v", info, err)
	}

	// Atomic multi-op write with explicit vectors.
	inputs, err := docs.EmbedInputs("plain", "text")
	if err != nil || len(inputs) != 1 {
		t.Fatalf("embed inputs: %v, %v", inputs, err)
	}
	vs, _ := embed(inputs)
	res, err := db.Write(vdblite.Upsert(docs, vdblite.Doc{Key: "p", Content: "plain"}, vs), vdblite.Delete(docs, "readme"))
	if err != nil || len(res) != 2 || !res[1].Deleted {
		t.Fatalf("write: %+v, %v", res, err)
	}
	if n, _ := docs.Count(); n != 1 {
		t.Fatalf("count %d", n)
	}

	// A reader in the same process, alongside the writer.
	ro, err := vdblite.Open(path, &vdblite.OpenOptions{ReadOnly: true})
	if err != nil {
		t.Fatal(err)
	}
	rdocs, err := ro.Collection("docs")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := rdocs.Delete("p"); !errors.Is(err, vdblite.CodeReadOnly) {
		t.Fatalf("want READ_ONLY, got %v", err)
	}
	ro.Close()
	if _, err := rdocs.Count(); !errors.Is(err, vdblite.CodeClosed) {
		t.Fatalf("want CLOSED, got %v", err)
	}
	var e *vdblite.Error
	if _, err := db.Collection("nope"); !errors.As(err, &e) || e.Code != vdblite.CodeNoSuchCollection {
		t.Fatalf("want NO_SUCH_COLLECTION, got %v", err)
	}
}

func TestBulkLoad(t *testing.T) {
	db, _ := open(t)
	c, err := db.CreateCollection("v", vdblite.CollectionOptions{
		Dim: 2, Quantization: "binary", Chunking: &vdblite.Chunking{Kind: "none"},
	})
	if err != nil {
		t.Fatal(err)
	}
	load, err := c.BulkLoad(0)
	if err != nil {
		t.Fatal(err)
	}
	var batch []vdblite.BulkDoc
	for i := 0; i < 500; i++ {
		x := float32(i%7) + 1
		batch = append(batch, vdblite.BulkDoc{
			Doc:     vdblite.Doc{Key: string(rune('a'+i%26)) + string(rune('0'+i/26)), Content: "doc"},
			Vectors: [][]float32{{x, 1}},
		})
	}
	n, err := load.Add(batch)
	if err != nil || n != 500 {
		t.Fatalf("add: %d, %v", n, err)
	}
	if load.Finished() {
		t.Fatal("finished early")
	}
	if err := load.Finish(); err != nil {
		t.Fatal(err)
	}
	if !load.Finished() {
		t.Fatal("not finished")
	}
	if err := load.Finish(); !errors.Is(err, vdblite.CodeInvalidArgument) {
		t.Fatalf("second finish: %v", err)
	}
	hits, err := c.Search([]float32{7, 1}, &vdblite.SearchOptions{K: 3})
	if err != nil || len(hits) != 3 {
		t.Fatalf("search: %+v, %v", hits, err)
	}
	if s, err := c.Stats(); err != nil || s.Chunks != 500 {
		t.Fatalf("stats: %+v, %v", s, err)
	}
}
