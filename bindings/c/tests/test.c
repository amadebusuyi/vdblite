/* Tests the C API end to end. Usage: test <scratch directory> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "vdblite.h"

static int failures = 0;

#define CHECK(cond)                                                       \
  do {                                                                    \
    if (!(cond)) {                                                        \
      fprintf(stderr, "%s:%d: CHECK(%s) failed: %s\n", __FILE__, __LINE__, \
              #cond, vdb_last_error());                                   \
      failures++;                                                         \
    }                                                                     \
  } while (0)

#define OK(call) CHECK((call) == VDB_OK)

static int contains(const char *s, const char *part) { return s && strstr(s, part) != NULL; }

int main(int argc, char **argv) {
  if (argc < 2) {
    fprintf(stderr, "usage: %s <scratch dir>\n", argv[0]);
    return 2;
  }
  char path[4096];
  snprintf(path, sizeof path, "%s/c-test.vdb", argv[1]);
  remove(path);

  CHECK(strlen(vdb_version()) > 0);

  vdb_db *db = NULL;
  OK(vdb_open(path, 0, 0, &db));

  vdb_collection *docs = NULL;
  OK(vdb_create_collection(db, "docs", "{\"dim\": 3, \"chunking\": {\"kind\": \"none\"}}", &docs));
  CHECK(vdb_create_collection(db, "docs", "{\"dim\": 3}", &docs) == VDB_ERR_COLLECTION_EXISTS);
  CHECK(contains(vdb_last_error(), "already exists"));

  /* Two upserts and their vectors, back to back. */
  const char *ops =
      "[{\"op\": \"upsert\", \"collection\": \"docs\", \"key\": \"a\", \"content\": \"alpha\","
      "  \"meta\": {\"lang\": \"en\", \"n\": 1}, \"vectors\": 1},"
      " {\"op\": \"upsert\", \"collection\": \"docs\", \"key\": \"b\", \"content\": \"beta\","
      "  \"meta\": {\"lang\": \"fr\", \"n\": 2}, \"vectors\": 1}]";
  float vectors[] = {1, 0, 0, 0, 1, 0};
  char *out = NULL;
  OK(vdb_write(db, ops, vectors, 6, &out));
  CHECK(contains(out, "\"changed\":true"));
  vdb_string_free(out);

  uint64_t n = 0;
  OK(vdb_count(docs, &n));
  CHECK(n == 2);

  float q[] = {1, 0.1f, 0};
  OK(vdb_search(docs, q, 3, "{\"k\": 2}", &out));
  CHECK(contains(out, "\"key\":\"a\""));
  CHECK(out && strstr(out, "\"a\"") < strstr(out, "\"b\""));
  vdb_string_free(out);

  OK(vdb_search(docs, q, 3, "{\"k\": 2, \"filter\": {\"lang\": \"fr\"}}", &out));
  CHECK(contains(out, "\"key\":\"b\"") && !contains(out, "\"key\":\"a\""));
  vdb_string_free(out);

  OK(vdb_search(docs, q, 3, "{\"mode\": \"documents\"}", &out));
  CHECK(contains(out, "\"chunks\":["));
  vdb_string_free(out);

  OK(vdb_search(docs, q, 3, "{\"text\": \"beta\"}", &out));
  CHECK(contains(out, "\"lexicalScore\""));
  vdb_string_free(out);

  CHECK(vdb_search(docs, q, 2, NULL, &out) == VDB_ERR_DIMENSION);
  CHECK(vdb_search(docs, q, 3, "{\"filter\": {\"n\": {\"$bad\": 1}}}", &out) == VDB_ERR_INVALID_ARGUMENT);
  CHECK(contains(vdb_last_error(), "$bad"));
  CHECK(vdb_write(db, ops, vectors, 3, &out) == VDB_ERR_INVALID_ARGUMENT);

  OK(vdb_get(docs, "a", &out));
  CHECK(contains(out, "\"content\":\"alpha\""));
  vdb_string_free(out);
  OK(vdb_get(docs, "missing", &out));
  CHECK(out && strcmp(out, "null") == 0);
  vdb_string_free(out);

  OK(vdb_scan(docs, "{\"n\": {\"$gte\": 2}}", &out));
  CHECK(contains(out, "\"key\":\"b\"") && !contains(out, "\"key\":\"a\""));
  vdb_string_free(out);

  /* Planned embeddings: an unchanged document needs none. */
  OK(vdb_plan_embeddings(docs, "{\"key\": \"a\", \"content\": \"alpha\"}", &out));
  CHECK(out && strcmp(out, "[]") == 0);
  vdb_string_free(out);
  OK(vdb_plan_embeddings(docs, "{\"key\": \"a\", \"content\": \"alpha 2\"}", &out));
  CHECK(contains(out, "\"hash\""));
  vdb_string_free(out);

  OK(vdb_write(db, "[{\"op\": \"delete\", \"collection\": \"docs\", \"key\": \"b\"}]", NULL, 0, &out));
  CHECK(contains(out, "\"deleted\":true"));
  vdb_string_free(out);

  /* Chunked collection, bulk loaded. */
  vdb_collection *notes = NULL;
  OK(vdb_create_collection(db, "notes", "{\"dim\": 2, \"quantization\": \"binary\"}", &notes));
  OK(vdb_embed_inputs(notes, "# Title\n\nSome text.", "markdown", &out));
  CHECK(out && out[0] == '[');
  vdb_string_free(out);

  vdb_bulk *load = NULL;
  OK(vdb_bulk_load(notes, 0, &load));
  float bulk_vectors[] = {1, 0, 0, 1};
  uint64_t written = 0;
  OK(vdb_bulk_add(load,
                  "[{\"key\": \"x\", \"content\": \"x\", \"vectors\": 1},"
                  " {\"key\": \"y\", \"content\": \"y\", \"vectors\": 1}]",
                  bulk_vectors, 4, &written));
  CHECK(written == 2);
  bool finished = true;
  OK(vdb_bulk_finished(load, &finished));
  CHECK(!finished);
  OK(vdb_bulk_finish(load));
  CHECK(vdb_bulk_finish(load) == VDB_ERR_INVALID_ARGUMENT);
  vdb_bulk_free(load);
  float nq[] = {0, 1};
  OK(vdb_search(notes, nq, 2, "{\"k\": 1}", &out));
  CHECK(contains(out, "\"key\":\"y\""));
  vdb_string_free(out);

  OK(vdb_collection_info(notes, &out));
  CHECK(contains(out, "\"quantization\":\"binary\""));
  vdb_string_free(out);
  OK(vdb_collections(db, &out));
  CHECK(contains(out, "\"notes\""));
  vdb_string_free(out);

  /* A reader alongside the writer. */
  vdb_db *reader = NULL;
  OK(vdb_open(path, VDB_OPEN_READ_ONLY, 0, &reader));
  vdb_collection *rdocs = NULL;
  OK(vdb_get_collection(reader, "docs", &rdocs));
  OK(vdb_count(rdocs, &n));
  CHECK(n == 1);
  CHECK(vdb_write(reader, "[{\"op\": \"delete\", \"collection\": \"docs\", \"key\": \"a\"}]", NULL, 0, &out) ==
        VDB_ERR_READ_ONLY);
  vdb_collection_free(rdocs);
  vdb_db_free(reader);

  CHECK(vdb_get_collection(db, "nope", &rdocs) == VDB_ERR_NO_SUCH_COLLECTION);

  vdb_close(db);
  CHECK(vdb_count(docs, &n) == VDB_ERR_CLOSED);
  vdb_collection_free(docs);
  vdb_collection_free(notes);
  vdb_db_free(db);

  bool shrank = false;
  OK(vdb_compact_file(path, &shrank));

  if (failures) {
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
  }
  printf("c: all checks passed\n");
  return 0;
}
