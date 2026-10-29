package io.vdblite;

import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByteByReference;
import com.sun.jna.ptr.PointerByReference;
import java.lang.ref.Cleaner;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/**
 * An open vdblite database file. One process writes a file at a time; other
 * processes can open it read-only alongside and see each commit as it
 * lands. Thread-safe: share one instance across threads.
 *
 * <pre>{@code
 * try (Database db = Database.open(Path.of("notes.vdb"))) {
 *   Collection docs = db.getOrCreateCollection("docs", CollectionOptions.dim(768));
 *   docs.upsert(new Doc("readme", markdown).meta("lang", "en"), embedder);
 *   List<ChunkHit> hits = docs.search(queryVector, SearchOptions.k(5));
 * }
 * }</pre>
 */
public final class Database implements AutoCloseable {
  static final Cleaner CLEANER = Cleaner.create();

  final Pointer h;

  private Database(Pointer h) {
    this.h = h;
    CLEANER.register(this, () -> Native.LIB.vdb_db_free(h));
  }

  /** The version of the native library. */
  public static String version() {
    return Native.LIB.vdb_version();
  }

  /** Opens (or creates) a file for writing; other processes may read it alongside. */
  public static Database open(Path path) {
    return open(path, 0, 0);
  }

  /** Opens for writing with no other process allowed, not even readers; faster small commits. */
  public static Database openExclusive(Path path) {
    return open(path, Native.VDB_OPEN_EXCLUSIVE, 0);
  }

  /** Opens an existing file read-only, following the writing process's commits. */
  public static Database openReadOnly(Path path) {
    return open(path, Native.VDB_OPEN_READ_ONLY, 0);
  }

  /**
   * Opens with {@code readOnly} or {@code exclusive} access, caching {@code cacheBytes} of file
   * pages (0 for the default, 256 MiB).
   */
  public static Database open(Path path, boolean readOnly, boolean exclusive, long cacheBytes) {
    int flags = (readOnly ? Native.VDB_OPEN_READ_ONLY : 0) | (exclusive ? Native.VDB_OPEN_EXCLUSIVE : 0);
    return open(path, flags, cacheBytes);
  }

  private static Database open(Path path, int flags, long cacheBytes) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_open(path.toString(), flags, cacheBytes, out));
    return new Database(out.getValue());
  }

  /** Rewrites a file nobody has open without its free space; true if it shrank. */
  public static boolean compactFile(Path path) {
    ByteByReference shrank = new ByteByReference();
    Native.check(Native.LIB.vdb_compact_file(path.toString(), shrank));
    return shrank.getValue() != 0;
  }

  /** Closes the database; later calls on it and its collections throw {@link ErrorCode#CLOSED}. */
  @Override
  public void close() {
    Native.LIB.vdb_close(h);
  }

  public boolean isReadOnly() {
    ByteByReference out = new ByteByReference();
    Native.check(Native.LIB.vdb_is_read_only(h, out));
    return out.getValue() != 0;
  }

  public List<String> collections() {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_collections(h, out));
    return Json.strings(Native.take(out));
  }

  public boolean hasCollection(String name) {
    return collections().contains(name);
  }

  public Collection createCollection(String name, CollectionOptions options) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_create_collection(h, name, options.toJson(), out));
    return new Collection(this, out.getValue(), name);
  }

  /** An existing collection; throws {@link ErrorCode#NO_SUCH_COLLECTION} if there is none. */
  public Collection collection(String name) {
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_get_collection(h, name, out));
    return new Collection(this, out.getValue(), name);
  }

  public Collection getOrCreateCollection(String name, CollectionOptions options) {
    try {
      return collection(name);
    } catch (VdbliteException e) {
      if (e.code() != ErrorCode.NO_SUCH_COLLECTION) throw e;
      return createCollection(name, options);
    }
  }

  public void dropCollection(String name) {
    Native.check(Native.LIB.vdb_drop_collection(h, name));
  }

  /**
   * Applies upserts and deletes atomically, in one transaction.
   *
   * <pre>{@code db.write(WriteOp.upsert(docs, doc, vectors), WriteOp.delete(docs, "old"))}</pre>
   */
  public List<WriteResult> write(WriteOp... ops) {
    return write(List.of(ops));
  }

  public List<WriteResult> write(List<WriteOp> ops) {
    List<Object> items = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();
    for (WriteOp op : ops) {
      items.add(op.toJson());
      vectors.addAll(op.vectors);
    }
    float[] flat = Native.flatten(vectors);
    PointerByReference out = new PointerByReference();
    Native.check(Native.LIB.vdb_write(h, Json.write(items), flat, flat.length, out));
    Map<String, Object> res = Json.obj(Native.take(out));
    List<WriteResult> results = Results.map(res.get("results"), Results::writeResult);
    Object warning = res.get("warning");
    if (warning != null) {
      // Committed; only sealing the write buffer afterwards failed.
      throw new VdbliteException(ErrorCode.MAINTENANCE, (String) warning);
    }
    return results;
  }

  /** Seals buffered chunks and merges small segments in every collection, using every core. */
  public void optimize() {
    Native.check(Native.LIB.vdb_optimize(h));
  }

  /** Waits for background sealing and compaction to catch up with every commit so far. */
  public void waitForMaintenance() {
    Native.check(Native.LIB.vdb_wait_for_maintenance(h));
  }
}
