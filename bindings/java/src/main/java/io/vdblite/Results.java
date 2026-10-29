package io.vdblite;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** Builds result records from the native library's JSON. */
final class Results {
  private Results() {}

  static ChunkHit chunkHit(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new ChunkHit(
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.integer(m, "chunkId"),
        Json.integer(m, "ordinal"),
        Json.integer(m, "byteStart"),
        Json.integer(m, "byteEnd"),
        Json.flt(m, "score"),
        Json.flt(m, "vectorScore"),
        Json.flt(m, "lexicalScore"),
        Json.str(m, "text"),
        Json.strings(m.get("headingPath")),
        Json.meta(m));
  }

  static DocHit docHit(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new DocHit(
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.flt(m, "score"),
        Json.meta(m),
        map(m.get("chunks"), Results::chunkHit));
  }

  static Passage passage(Object o) {
    Map<String, Object> m = Json.obj(o);
    List<Object> ord = Json.list(m.get("ordinals"));
    return new Passage(
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.integer(m, "byteStart"),
        Json.integer(m, "byteEnd"),
        Json.flt(m, "score"),
        Json.str(m, "text"),
        ((Number) ord.get(0)).intValue(),
        ((Number) ord.get(1)).intValue(),
        Json.strings(m.get("headingPath")),
        Json.meta(m));
  }

  static Document document(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new Document(
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.str(m, "content"),
        Json.meta(m),
        Json.num(m, "version"),
        Json.num(m, "createdMs"),
        Json.num(m, "updatedMs"),
        Json.str(m, "format"),
        Json.integer(m, "chunkCount"));
  }

  static DocInfo docInfo(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new DocInfo(
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.meta(m),
        Json.num(m, "version"),
        Json.num(m, "createdMs"),
        Json.num(m, "updatedMs"),
        Json.num(m, "contentLength"),
        Json.integer(m, "chunkCount"));
  }

  static ChunkInfo chunkInfo(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new ChunkInfo(
        Json.integer(m, "chunkId"),
        Json.integer(m, "ordinal"),
        Json.integer(m, "byteStart"),
        Json.integer(m, "byteEnd"),
        Json.integer(m, "overlapStart"),
        Json.strings(m.get("headingPath")),
        Json.integer(m, "tokens"));
  }

  static WriteResult writeResult(Object o) {
    Map<String, Object> m = Json.obj(o);
    return new WriteResult(
        Json.str(m, "op"),
        Json.str(m, "collection"),
        Json.str(m, "key"),
        Json.num(m, "docId"),
        Json.num(m, "version"),
        Json.integer(m, "chunks"),
        Json.integer(m, "embedded"),
        Json.bool(m, "changed"),
        Json.bool(m, "deleted"));
  }

  interface Conv<T> {
    T from(Object o);
  }

  static <T> List<T> map(Object list, Conv<T> f) {
    List<T> out = new ArrayList<>();
    for (Object o : Json.list(list)) out.add(f.from(o));
    return out;
  }
}
