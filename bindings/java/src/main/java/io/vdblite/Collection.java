package io.vdblite;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByteByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** A collection in an open database. Thread-safe. */
public final class Collection {
  private final Database db;
  private final Pointer h;
  private final String name;

  Collection(Database db, Pointer h, String name) {
    this.db = db;
    this.h = h;
    this.name = name;
    Database.CLEANER.register(this, () -> Native.LIB.vdb_collection_free(h));
  }

  public String name() {
    return name;
  }

  public Database database() {
    return db;
  }

  public CollectionInfo info() {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_collection_info(h, out));
    Map<String, Object> m = Json.obj(Native.take(out));
    return new CollectionInfo(
        Json.str(m, "name"),
        Json.integer(m, "dim"),
        Json.str(m, "metric"),
        Json.str(m, "quantization"),
        Json.str(m, "chunking"),
        Json.integer(m, "targetTokens"),
        Json.integer(m, "overlapTokens"));
  }

  public IndexStats stats() {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_collection_stats(h, out));
    Map<String, Object> m = Json.obj(Native.take(out));
    return new IndexStats(
        Json.num(m, "chunks"),
        Json.integer(m, "sealedSegments"),
        Json.num(m, "buffered"),
        Json.num(m, "memoryBytes"),
        Json.str(m, "maintenanceError"));
  }

  /** The texts to embed for {@code content}, one per chunk. {@code format} may be null. */
  public List<String> embedInputs(String content, String format) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_embed_inputs(h, content, format, out));
    return Json.strings(Native.take(out));
  }

  /** Inserts or replaces a document with one vector per chunk. */
  public WriteResult upsert(Doc doc, List<float[]> vectors) {
    return db.write(WriteOp.upsert(this, doc, vectors)).get(0);
  }

  /**
   * Inserts or replaces a document, calling {@code embed} for the chunks that need vectors. An
   * edited document re-embeds only the chunks whose text changed.
   */
  public WriteResult upsert(Doc doc, Embedder embed) {
    for (int attempt = 0; ; attempt++) {
      PointerByReference out = new PointerByReference();
      Native.check(Native.LIB.vdb_plan_embeddings(h, Json.write(doc.toJson()), out));
      List<String> hashes = new ArrayList<>();
      List<String> texts = new ArrayList<>();
      for (Object o : Json.list(Native.take(out))) {
        Map<String, Object> p = Json.obj(o);
        hashes.add(Json.str(p, "hash"));
        texts.add(Json.str(p, "text"));
      }
      List<float[]> vectors = texts.isEmpty() ? List.of() : embed(embed, texts);
      try {
        return db.write(List.of(WriteOp.upsertPlanned(this, doc, hashes, vectors))).get(0);
      } catch (VdbliteException e) {
        // Another writer changed the document's chunks after planning.
        if (e.code() != ErrorCode.STALE_EMBEDDINGS || attempt >= 3) throw e;
      }
    }
  }

  private static List<float[]> embed(Embedder embed, List<String> texts) {
    List<float[]> vectors;
    try {
      vectors = embed.embed(texts);
    } catch (Exception e) {
      throw new VdbliteException(ErrorCode.EMBEDDER, String.valueOf(e.getMessage()));
    }
    if (vectors == null || vectors.size() != texts.size()) {
      throw new VdbliteException(
          ErrorCode.EMBEDDER,
          "the embedder returned " + (vectors == null ? 0 : vectors.size()) + " vectors for " + texts.size() + " texts");
    }
    return vectors;
  }

  /** Deletes a document; true if it existed. */
  public boolean delete(String key) {
    return db.write(WriteOp.delete(this, key)).get(0).deleted();
  }

  private Object search(float[] query, SearchOptions options, String mode, Map<String, Object> extra) {
    SearchOptions o = options != null ? options : new SearchOptions();
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_search(h, query, query.length, o.toJson(mode, extra), out));
    return Native.take(out);
  }

  /** The chunks nearest to {@code query}, best first. {@code options} may be null. */
  public List<ChunkHit> search(float[] query, SearchOptions options) {
    return Results.map(search(query, options, "chunks", Map.of()), Results::chunkHit);
  }

  public List<ChunkHit> search(float[] query) {
    return search(query, null);
  }

  /** Documents ranked by their best chunk. */
  public List<DocHit> searchDocuments(float[] query, SearchOptions options) {
    return Results.map(search(query, options, "documents", Map.of()), Results::docHit);
  }

  /** Documents ranked by the sum of their {@code top} best chunks. */
  public List<DocHit> searchDocumentsSum(float[] query, int top, SearchOptions options) {
    return Results.map(
        search(query, options, "documents", Map.of("agg", "sum", "top", top)), Results::docHit);
  }

  /** Hits widened by {@code neighbors} chunks on each side, with overlaps merged. */
  public List<Passage> searchExpand(float[] query, int neighbors, SearchOptions options) {
    return Results.map(
        search(query, options, "expand", Map.of("neighbors", neighbors)), Results::passage);
  }

  /** The whole heading section around each hit. */
  public List<Passage> searchSections(float[] query, SearchOptions options) {
    return Results.map(search(query, options, "sections", Map.of()), Results::passage);
  }

  /** The document with {@code key}, or null. */
  public Document get(String key) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_get(h, key, out));
    Object d = Native.take(out);
    return d == null ? null : Results.document(d);
  }

  public List<ChunkInfo> chunks(String key) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_chunks(h, key, out));
    return Results.map(Native.take(out), Results::chunkInfo);
  }

  /** Summaries of the documents matching {@code filter} (null: all). */
  public List<DocInfo> scan(Map<String, ?> filter) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_scan(h, filter == null ? null : Json.write(filter), out));
    return Results.map(Native.take(out), Results::docInfo);
  }

  public long count() {
    LongByReference out = new LongByReference();
    Native.check(Native.LIB.vdb_count(h, out));
    return out.getValue();
  }

  public boolean hasLexical() {
    ByteByReference out = new ByteByReference();
    Native.check(Native.LIB.vdb_has_lexical(h, out));
    return out.getValue() != 0;
  }

  /** Builds the keyword index from stored text, without re-embedding; false if it existed. */
  public boolean enableLexical() {
    ByteByReference out = new ByteByReference();
    Native.check(Native.LIB.vdb_enable_lexical(h, out));
    return out.getValue() != 0;
  }

  /** Starts a bulk load with the default checkpoint interval. */
  public BulkLoad bulkLoad() {
    return bulkLoad(0);
  }

  /**
   * Starts a bulk load: added documents are readable at once and searchable after {@link
   * BulkLoad#finish}, which builds the index once with every core. {@code checkpointEvery}: chunks
   * between durable commits in a file opened exclusively (0 for {@link BulkLoad#DEFAULT_CHECKPOINT}).
   */
  public BulkLoad bulkLoad(long checkpointEvery) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_bulk_load(h, checkpointEvery, out));
    return new BulkLoad(this, out.getValue());
  }
}
