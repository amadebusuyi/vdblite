package io.vdblite;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * Refines a search:
 *
 * <pre>{@code SearchOptions.k(5).filter(Map.of("lang", "en", "year", Map.of("$gte", 2020)))}</pre>
 */
public final class SearchOptions {
  final Map<String, Object> m = new LinkedHashMap<>();

  public SearchOptions() {}

  /** Results to return (default 10). */
  public static SearchOptions k(int k) {
    return new SearchOptions().limit(k);
  }

  /** Results to return (default 10). */
  public SearchOptions limit(int k) {
    m.put("k", k);
    return this;
  }

  /** Graph beam width: higher is slower and more accurate. */
  public SearchOptions ef(int ef) {
    m.put("ef", ef);
    return this;
  }

  /**
   * A Mongo-style metadata filter: operators $eq $ne $in $nin $gt $gte $lt
   * $lte $exists $and $or $not. Equality on a list field matches any
   * element.
   */
  public SearchOptions filter(Map<String, ?> filter) {
    m.put("filter", filter);
    return this;
  }

  /**
   * Query text for hybrid search: chunks are also ranked by BM25 keyword
   * match and the two rankings fused.
   */
  public SearchOptions text(String text) {
    m.put("text", text);
    return this;
  }

  String toJson(String mode, Map<String, Object> extra) {
    Map<String, Object> out = new LinkedHashMap<>(m);
    out.put("mode", mode);
    out.putAll(extra);
    return Json.write(out);
  }
}
