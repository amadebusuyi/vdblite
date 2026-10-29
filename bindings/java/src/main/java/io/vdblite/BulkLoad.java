package io.vdblite;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByteByReference;
import com.sun.jna.ptr.LongByReference;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * A bulk load in progress (see {@link Collection#bulkLoad}). Always call {@link #finish}: until
 * then, loaded documents can be read but not found by search.
 */
public final class BulkLoad {
  /** Chunks between durable commits, by default. */
  public static final long DEFAULT_CHECKPOINT = Native.LIB.vdb_default_checkpoint();

  private final Collection collection;
  private final Pointer h;

  BulkLoad(Collection collection, Pointer h) {
    this.collection = collection;
    this.h = h;
    Database.CLEANER.register(this, () -> Native.LIB.vdb_bulk_free(h));
  }

  /** A document for {@link #add}, with one vector per chunk. */
  public record Item(Doc doc, List<float[]> vectors) {}

  public Collection collection() {
    return collection;
  }

  /** Inserts or replaces documents in one transaction; returns how many were written. */
  public long add(List<Item> items) {
    List<Object> docs = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();
    for (Item it : items) {
      Map<String, Object> m = it.doc().toJson();
      m.put("vectors", it.vectors().size());
      docs.add(m);
      vectors.addAll(it.vectors());
    }
    float[] flat = Native.flatten(vectors);
    LongByReference written = new LongByReference();
    Native.check(Native.LIB.vdb_bulk_add(h, Json.write(docs), flat, flat.length, written));
    return written.getValue();
  }

  /** Makes everything added so far durable (only matters in a file opened exclusively). */
  public void checkpoint() {
    Native.check(Native.LIB.vdb_bulk_checkpoint(h));
  }

  /**
   * Builds the index over everything added and makes it searchable. Other writes wait until it's
   * done; reads and searches don't.
   */
  public void finish() {
    Native.check(Native.LIB.vdb_bulk_finish(h));
  }

  public boolean isFinished() {
    ByteByReference out = new ByteByReference();
    Native.check(Native.LIB.vdb_bulk_finished(h, out));
    return out.getValue() != 0;
  }
}
