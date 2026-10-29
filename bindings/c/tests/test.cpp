// Tests the C++ wrapper. Usage: test_cpp <scratch directory>
#include <cstdio>
#include <iostream>
#include <string>

#include "vdblite.hpp"

#define CHECK(cond)                                                      \
  do {                                                                   \
    if (!(cond)) {                                                       \
      std::cerr << __FILE__ << ":" << __LINE__ << ": CHECK(" #cond ")\n"; \
      return 1;                                                          \
    }                                                                    \
  } while (0)

int main(int argc, char **argv) {
  if (argc < 2) return 2;
  std::string path = std::string(argv[1]) + "/cpp-test.vdb";
  std::remove(path.c_str());

  auto db = vdblite::Database::open(path);
  auto docs = db.create_collection("docs", R"({"dim": 3, "chunking": {"kind": "none"}})");
  auto res = db.write(R"([{"op": "upsert", "collection": "docs", "key": "a", "content": "alpha", "vectors": 1}])",
                      {1, 0, 0});
  CHECK(res.find("\"changed\":true") != std::string::npos);
  CHECK(docs.count() == 1);
  auto hits = docs.search({1, 0, 0}, R"({"k": 1})");
  CHECK(hits.find("\"key\":\"a\"") != std::string::npos);
  CHECK(docs.get("missing") == "null");

  try {
    docs.search({1, 0});
    CHECK(false);
  } catch (const vdblite::Error &e) {
    CHECK(e.code() == VDB_ERR_DIMENSION);
  }

  auto load = docs.bulk_load();
  CHECK(load.add(R"([{"key": "b", "content": "beta", "vectors": 1}])", {0, 1, 0}) == 1);
  load.finish();
  CHECK(load.finished());
  CHECK(docs.count() == 2);

  db.close();
  try {
    docs.count();
    CHECK(false);
  } catch (const vdblite::Error &e) {
    CHECK(e.code() == VDB_ERR_CLOSED);
  }
  std::cout << "c++: all checks passed\n";
  return 0;
}
