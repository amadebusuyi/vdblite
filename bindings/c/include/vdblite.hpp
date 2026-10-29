// vdblite for C++17: RAII handles and exceptions over the C API
// (vdblite.h). Structured values are JSON strings in the same shapes as the
// C API; parse them with the JSON library you already use.
//
// Licensed under Apache-2.0.
#ifndef VDBLITE_HPP
#define VDBLITE_HPP

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vdblite.h"

namespace vdblite {

// A failed call: `code()` is one of the VDB_ERR_* constants.
class Error : public std::runtime_error {
 public:
  Error(int32_t code, const std::string &message) : std::runtime_error(message), code_(code) {}
  int32_t code() const noexcept { return code_; }

 private:
  int32_t code_;
};

namespace detail {

inline void check(int32_t status) {
  if (status != VDB_OK) throw Error(status, vdb_last_error());
}

// Takes ownership of a string from the library.
inline std::string take(char *s) {
  std::string out = s ? s : "";
  vdb_string_free(s);
  return out;
}

inline const char *c_str_or_null(const std::string &s) { return s.empty() ? nullptr : s.c_str(); }

struct DbFree {
  void operator()(vdb_db *p) const noexcept { vdb_db_free(p); }
};
struct CollectionFree {
  void operator()(vdb_collection *p) const noexcept { vdb_collection_free(p); }
};
struct BulkFree {
  void operator()(vdb_bulk *p) const noexcept { vdb_bulk_free(p); }
};

}  // namespace detail

inline std::string version() { return vdb_version(); }

// A bulk load in progress (see Collection::bulk_load). Always call finish().
class BulkLoad {
 public:
  // `docs_json`: [{"key", "content", "meta"?, "format"?, "vectors": n}];
  // `vectors`: every document's vectors, one per chunk, back to back.
  uint64_t add(const std::string &docs_json, const std::vector<float> &vectors) {
    uint64_t written = 0;
    detail::check(vdb_bulk_add(h_.get(), docs_json.c_str(), vectors.data(), vectors.size(), &written));
    return written;
  }
  void checkpoint() { detail::check(vdb_bulk_checkpoint(h_.get())); }
  void finish() { detail::check(vdb_bulk_finish(h_.get())); }
  bool finished() const {
    bool out = false;
    detail::check(vdb_bulk_finished(h_.get(), &out));
    return out;
  }

 private:
  friend class Collection;
  explicit BulkLoad(vdb_bulk *h) : h_(h) {}
  std::unique_ptr<vdb_bulk, detail::BulkFree> h_;
};

// A collection in an open database.
class Collection {
 public:
  // {"name", "dim", "metric", "quantization", "chunking", ...}
  std::string info() const { return json(vdb_collection_info); }
  // {"chunks", "sealedSegments", "buffered", "memoryBytes", "maintenanceError"}
  std::string stats() const { return json(vdb_collection_stats); }

  // The texts to embed for `content`, one per chunk: a JSON array of strings.
  std::string embed_inputs(const std::string &content, const std::string &format = "") const {
    char *out = nullptr;
    detail::check(vdb_embed_inputs(h_.get(), content.c_str(), detail::c_str_or_null(format), &out));
    return detail::take(out);
  }

  // [{"hash", "text"}]: the chunks of `doc_json` that need new vectors.
  std::string plan_embeddings(const std::string &doc_json) const {
    char *out = nullptr;
    detail::check(vdb_plan_embeddings(h_.get(), doc_json.c_str(), &out));
    return detail::take(out);
  }

  // `options_json`: {"mode", "k", "ef", "filter", "text", "agg", "top", "neighbors"}.
  std::string search(const std::vector<float> &query, const std::string &options_json = "") const {
    char *out = nullptr;
    detail::check(vdb_search(h_.get(), query.data(), query.size(), detail::c_str_or_null(options_json), &out));
    return detail::take(out);
  }

  // The document, or "null".
  std::string get(const std::string &key) const { return keyed(vdb_get, key); }
  std::string chunks(const std::string &key) const { return keyed(vdb_chunks, key); }

  std::string scan(const std::string &filter_json = "") const {
    char *out = nullptr;
    detail::check(vdb_scan(h_.get(), detail::c_str_or_null(filter_json), &out));
    return detail::take(out);
  }

  uint64_t count() const {
    uint64_t n = 0;
    detail::check(vdb_count(h_.get(), &n));
    return n;
  }

  bool has_lexical() const {
    bool out = false;
    detail::check(vdb_has_lexical(h_.get(), &out));
    return out;
  }

  bool enable_lexical() {
    bool built = false;
    detail::check(vdb_enable_lexical(h_.get(), &built));
    return built;
  }

  // `checkpoint_every`: 0 for the default.
  BulkLoad bulk_load(uint64_t checkpoint_every = 0) {
    vdb_bulk *b = nullptr;
    detail::check(vdb_bulk_load(h_.get(), checkpoint_every, &b));
    return BulkLoad(b);
  }

 private:
  friend class Database;
  explicit Collection(vdb_collection *h) : h_(h) {}

  std::string json(int32_t (*f)(const vdb_collection *, char **)) const {
    char *out = nullptr;
    detail::check(f(h_.get(), &out));
    return detail::take(out);
  }
  std::string keyed(int32_t (*f)(const vdb_collection *, const char *, char **), const std::string &key) const {
    char *out = nullptr;
    detail::check(f(h_.get(), key.c_str(), &out));
    return detail::take(out);
  }

  std::unique_ptr<vdb_collection, detail::CollectionFree> h_;
};

// An open database file. Closed when destroyed.
class Database {
 public:
  // `flags`: VDB_OPEN_READ_ONLY and/or VDB_OPEN_EXCLUSIVE.
  static Database open(const std::string &path, uint32_t flags = 0, uint64_t cache_bytes = 0) {
    vdb_db *h = nullptr;
    detail::check(vdb_open(path.c_str(), flags, cache_bytes, &h));
    return Database(h);
  }

  static bool compact_file(const std::string &path) {
    bool shrank = false;
    detail::check(vdb_compact_file(path.c_str(), &shrank));
    return shrank;
  }

  void close() { vdb_close(h_.get()); }

  bool read_only() const {
    bool out = false;
    detail::check(vdb_is_read_only(h_.get(), &out));
    return out;
  }

  // A JSON array of names.
  std::string collections() const {
    char *out = nullptr;
    detail::check(vdb_collections(h_.get(), &out));
    return detail::take(out);
  }

  // `options_json`: {"dim", "metric"?, "quantization"?, "chunking"?, ...}.
  Collection create_collection(const std::string &name, const std::string &options_json) {
    vdb_collection *c = nullptr;
    detail::check(vdb_create_collection(h_.get(), name.c_str(), options_json.c_str(), &c));
    return Collection(c);
  }

  Collection collection(const std::string &name) const {
    vdb_collection *c = nullptr;
    detail::check(vdb_get_collection(h_.get(), name.c_str(), &c));
    return Collection(c);
  }

  void drop_collection(const std::string &name) { detail::check(vdb_drop_collection(h_.get(), name.c_str())); }

  // Atomic upserts and deletes; see vdb_write for `ops_json`.
  std::string write(const std::string &ops_json, const std::vector<float> &vectors = {}) {
    char *out = nullptr;
    detail::check(vdb_write(h_.get(), ops_json.c_str(), vectors.data(), vectors.size(), &out));
    return detail::take(out);
  }

  void optimize() { detail::check(vdb_optimize(h_.get())); }
  void wait_for_maintenance() { detail::check(vdb_wait_for_maintenance(h_.get())); }

 private:
  explicit Database(vdb_db *h) : h_(h) {}
  std::unique_ptr<vdb_db, detail::DbFree> h_;
};

}  // namespace vdblite

#endif  // VDBLITE_HPP
