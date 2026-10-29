package io.vdblite;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class VdbliteTest {
  @TempDir Path dir;

  /** A toy embedder: [length, count of "rust", 1]. */
  static List<float[]> embed(List<String> texts) {
    List<float[]> out = new ArrayList<>();
    for (String t : texts) {
      out.add(new float[] {t.length(), t.split("rust", -1).length - 1, 1});
    }
    return out;
  }

  static final String README = "# Intro\n\nrust rust rust\n\n## Usage\n\nOther words here.";

  @Test
  void roundTrip() {
    Path path = dir.resolve("test.vdb");
    try (Database db = Database.open(path)) {
      assertFalse(Database.version().isEmpty());
      Collection docs = db.createCollection("docs", CollectionOptions.dim(3));
      VdbliteException dup =
          assertThrows(VdbliteException.class, () -> db.createCollection("docs", CollectionOptions.dim(3)));
      assertEquals(ErrorCode.COLLECTION_EXISTS, dup.code());
      assertTrue(db.hasCollection("docs"));

      Doc doc = new Doc("readme", README).meta("lang", "en").meta("year", 2024).meta("tags", List.of("a", "b"));
      WriteResult r = docs.upsert(doc, VdbliteTest::embed);
      assertTrue(r.changed());
      assertEquals(r.chunks(), r.embedded());
      WriteResult again = docs.upsert(doc, VdbliteTest::embed);
      assertFalse(again.changed());
      assertEquals(0, again.embedded());

      float[] q = embed(List.of("rust rust rust")).get(0);
      List<ChunkHit> hits = docs.search(q, SearchOptions.k(5).filter(Map.of("year", Map.of("$gte", 2020))));
      assertEquals("readme", hits.get(0).key());
      assertNotNull(hits.get(0).vectorScore());
      assertEquals("en", hits.get(0).meta().get("lang"));
      assertEquals(List.of("a", "b"), hits.get(0).meta().get("tags"));
      assertTrue(docs.search(q, new SearchOptions().filter(Map.of("tags", "z"))).isEmpty());
      ChunkHit lexical = docs.search(q, new SearchOptions().text("Usage")).get(0);
      assertNotNull(lexical.lexicalScore());

      assertEquals(1, docs.searchDocumentsSum(q, 2, null).size());
      assertTrue(docs.searchSections(q, null).get(0).text().contains("rust"));
      assertFalse(docs.searchExpand(q, 1, null).isEmpty());
      VdbliteException dim = assertThrows(VdbliteException.class, () -> docs.search(new float[] {1, 2}));
      assertEquals(ErrorCode.DIMENSION, dim.code());

      Document d = docs.get("readme");
      assertTrue(d.content().startsWith("# Intro"));
      assertEquals("markdown", d.format());
      assertNull(docs.get("missing"));
      assertEquals(r.chunks(), docs.chunks("readme").size());
      assertEquals(1, docs.scan(Map.of("lang", "en")).size());
      CollectionInfo info = docs.info();
      assertEquals(3, info.dim());
      assertEquals("int8", info.quantization());

      // Unicode survives the trip both ways.
      List<String> inputs = docs.embedInputs("héllo wörld ✓", "text");
      assertEquals(1, inputs.size());
      List<WriteResult> res =
          db.write(WriteOp.upsert(docs, new Doc("u", "héllo wörld ✓"), embed(inputs)), WriteOp.delete(docs, "readme"));
      assertTrue(res.get(1).deleted());
      assertEquals("héllo wörld ✓", docs.get("u").content());
      assertEquals(1, docs.count());

      try (Database ro = Database.openReadOnly(path)) {
        Collection rdocs = ro.collection("docs");
        assertEquals(ErrorCode.READ_ONLY, assertThrows(VdbliteException.class, () -> rdocs.delete("u")).code());
        ro.close();
        assertEquals(ErrorCode.CLOSED, assertThrows(VdbliteException.class, rdocs::count).code());
      }
      assertEquals(
          ErrorCode.NO_SUCH_COLLECTION, assertThrows(VdbliteException.class, () -> db.collection("nope")).code());
    }
  }

  @Test
  void bulkLoad() {
    try (Database db = Database.open(dir.resolve("bulk.vdb"))) {
      Collection c = db.createCollection("v", CollectionOptions.dim(2).quantization("binary").noChunking());
      BulkLoad load = c.bulkLoad();
      List<BulkLoad.Item> items = new ArrayList<>();
      for (int i = 0; i < 500; i++) {
        items.add(new BulkLoad.Item(new Doc("k" + i, "doc"), List.of(new float[] {i % 7 + 1, 1})));
      }
      assertEquals(500, load.add(items));
      assertFalse(load.isFinished());
      load.finish();
      assertTrue(load.isFinished());
      assertEquals(ErrorCode.INVALID_ARGUMENT, assertThrows(VdbliteException.class, load::finish).code());
      assertEquals(3, c.search(new float[] {7, 1}, SearchOptions.k(3)).size());
      assertEquals(500, c.stats().chunks());
      assertTrue(BulkLoad.DEFAULT_CHECKPOINT > 0);
    }
  }

  @Test
  void json() {
    Object v = Json.parse("{\"a\":[1,2.5,\"x\\n\\u00e9\",true,null],\"b\":{}}");
    assertEquals("{\"a\":[1,2.5,\"x\\né\",true,null],\"b\":{}}", Json.write(v));
  }
}
