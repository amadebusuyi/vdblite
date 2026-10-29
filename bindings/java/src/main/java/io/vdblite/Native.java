package io.vdblite;

import com.sun.jna.Library;
import com.sun.jna.Pointer;
import com.sun.jna.ptr.ByteByReference;
import com.sun.jna.ptr.LongByReference;
import com.sun.jna.ptr.PointerByReference;
import java.util.Map;

/**
 * The C API (vdblite.h), loaded with JNA. The library comes from the
 * {@code vdblite.library.path} system property (a file), else from this
 * jar's resources for the platform, else from {@code jna.library.path} and
 * the system's library paths.
 */
interface Native extends Library {
  Native LIB = load();

  private static Native load() {
    Map<String, Object> options = Map.of(Library.OPTION_STRING_ENCODING, "UTF-8");
    String path = System.getProperty("vdblite.library.path");
    return com.sun.jna.Native.load(path != null ? path : "vdblite", Native.class, options);
  }

  int VDB_OPEN_READ_ONLY = 1;
  int VDB_OPEN_EXCLUSIVE = 2;

  String vdb_version();

  String vdb_last_error();

  void vdb_string_free(Pointer s);

  int vdb_open(String path, int flags, long cacheBytes, PointerByReference out);

  void vdb_close(Pointer db);

  void vdb_db_free(Pointer db);

  int vdb_is_read_only(Pointer db, ByteByReference out);

  int vdb_collections(Pointer db, PointerByReference out);

  int vdb_create_collection(Pointer db, String name, String optionsJson, PointerByReference out);

  int vdb_get_collection(Pointer db, String name, PointerByReference out);

  int vdb_drop_collection(Pointer db, String name);

  int vdb_write(Pointer db, String opsJson, float[] vectors, long nFloats, PointerByReference out);

  int vdb_optimize(Pointer db);

  int vdb_wait_for_maintenance(Pointer db);

  int vdb_compact_file(String path, ByteByReference shrank);

  void vdb_collection_free(Pointer c);

  int vdb_collection_info(Pointer c, PointerByReference out);

  int vdb_collection_stats(Pointer c, PointerByReference out);

  int vdb_embed_inputs(Pointer c, String content, String format, PointerByReference out);

  int vdb_plan_embeddings(Pointer c, String docJson, PointerByReference out);

  int vdb_search(Pointer c, float[] query, long dim, String optionsJson, PointerByReference out);

  int vdb_get(Pointer c, String key, PointerByReference out);

  int vdb_chunks(Pointer c, String key, PointerByReference out);

  int vdb_scan(Pointer c, String filterJson, PointerByReference out);

  int vdb_count(Pointer c, LongByReference out);

  int vdb_has_lexical(Pointer c, ByteByReference out);

  int vdb_enable_lexical(Pointer c, ByteByReference built);

  long vdb_default_checkpoint();

  int vdb_bulk_load(Pointer c, long checkpointEvery, PointerByReference out);

  int vdb_bulk_add(Pointer b, String docsJson, float[] vectors, long nFloats, LongByReference written);

  int vdb_bulk_checkpoint(Pointer b);

  int vdb_bulk_finish(Pointer b);

  int vdb_bulk_finished(Pointer b, ByteByReference out);

  void vdb_bulk_free(Pointer b);

  // ------------------------------------------------------------ helpers

  /** Throws for a failed status, with this thread's error message. */
  static void check(int status) {
    if (status != 0) throw new VdbliteException(ErrorCode.of(status), LIB.vdb_last_error());
  }

  /** Takes a JSON string the library returned, parses it, and frees it. */
  static Object take(PointerByReference out) {
    Pointer p = out.getValue();
    try {
      return Json.parse(p.getString(0, "UTF-8"));
    } finally {
      LIB.vdb_string_free(p);
    }
  }

  /** Vectors back to back, for the C API. */
  static float[] flatten(Iterable<float[]> vectors) {
    int n = 0;
    for (float[] v : vectors) n += v.length;
    float[] flat = new float[n];
    int at = 0;
    for (float[] v : vectors) {
      System.arraycopy(v, 0, flat, at, v.length);
      at += v.length;
    }
    return flat;
  }
}
