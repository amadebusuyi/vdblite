package io.vdblite;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Options for a new collection. Only the dimension is required:
 *
 * <pre>{@code CollectionOptions.dim(768).metric("cosine").quantization("int8")}</pre>
 */
public final class CollectionOptions {
  private final Map<String, Object> m = new LinkedHashMap<>();
  private final Map<String, Object> chunking = new LinkedHashMap<>();

  private CollectionOptions(int dim) {
    m.put("dim", dim);
  }

  public static CollectionOptions dim(int dim) {
    return new CollectionOptions(dim);
  }

  /** "cosine" (default), "dot" or "l2". */
  public CollectionOptions metric(String metric) {
    m.put("metric", metric);
    return this;
  }

  /**
   * "int8" (default: int8 graph walk, f16 re-ranking), "binary" (1-bit
   * graph walk, f16 re-ranking) or "none" (f32).
   */
  public CollectionOptions quantization(String quantization) {
    m.put("quantization", quantization);
    return this;
  }

  /** Splits documents along their structure, into chunks of about this size. */
  public CollectionOptions structuralChunking(int targetTokens, int overlapTokens) {
    chunking.put("kind", "structural");
    chunking.put("targetTokens", targetTokens);
    chunking.put("overlapTokens", overlapTokens);
    return this;
  }

  /** Each document is one chunk with one vector. */
  public CollectionOptions noChunking() {
    chunking.clear();
    chunking.put("kind", "none");
    return this;
  }

  /** Chunks buffered before they're sealed into a graph (default 65536). */
  public CollectionOptions sealThreshold(int chunks) {
    m.put("sealThreshold", chunks);
    return this;
  }

  /** Keep a BM25 keyword index for hybrid search (default true). */
  public CollectionOptions lexical(boolean lexical) {
    m.put("lexical", lexical);
    return this;
  }

  /** HNSW graph parameters. */
  public CollectionOptions hnsw(int m, int efConstruction, int efSearch) {
    this.m.put("m", m);
    this.m.put("efConstruction", efConstruction);
    this.m.put("efSearch", efSearch);
    return this;
  }

  String toJson() {
    Map<String, Object> out = new LinkedHashMap<>(m);
    if (!chunking.isEmpty()) out.put("chunking", chunking);
    return Json.write(out);
  }
}
