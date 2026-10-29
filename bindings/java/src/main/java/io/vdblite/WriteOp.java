package io.vdblite;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** One change for {@link Database#write}: an upsert or a delete. */
public final class WriteOp {
  private final Map<String, Object> json;
  final List<float[]> vectors;

  private WriteOp(Map<String, Object> json, List<float[]> vectors) {
    this.json = json;
    this.vectors = vectors;
  }

  /** Inserts or replaces a document with one vector per chunk (see {@link Collection#embedInputs}). */
  public static WriteOp upsert(Collection collection, Doc doc, List<float[]> vectors) {
    Map<String, Object> m = doc.toJson();
    m.put("op", "upsert");
    m.put("collection", collection.name());
    m.put("vectors", vectors.size());
    return new WriteOp(m, new ArrayList<>(vectors));
  }

  /** An upsert with vectors for the planned chunks, by hash. */
  static WriteOp upsertPlanned(Collection collection, Doc doc, List<String> hashes, List<float[]> vectors) {
    Map<String, Object> m = doc.toJson();
    m.put("op", "upsert");
    m.put("collection", collection.name());
    m.put("embeddings", hashes);
    return new WriteOp(m, vectors);
  }

  public static WriteOp delete(Collection collection, String key) {
    return new WriteOp(Map.of("op", "delete", "collection", collection.name(), "key", key), List.of());
  }

  Map<String, Object> toJson() {
    return json;
  }
}
