package io.vdblite;

import java.util.LinkedHashMap;
import java.util.Map;

/**
 * A document to write: a key, its content, optional metadata and format.
 *
 * <pre>{@code new Doc("readme", markdown).meta("lang", "en").meta("year", 2024)}</pre>
 */
public final class Doc {
  private final String key;
  private final String content;
  private final Map<String, Object> meta = new LinkedHashMap<>();
  private String format;

  public Doc(String key, String content) {
    this.key = key;
    this.content = content;
  }

  /**
   * Adds a metadata field: a string, number, boolean, null, or a list of
   * those.
   */
  public Doc meta(String field, Object value) {
    meta.put(field, value);
    return this;
  }

  public Doc meta(Map<String, ?> fields) {
    meta.putAll(fields);
    return this;
  }

  /**
   * "markdown", "code", "html", "text", a file name, extension or MIME
   * type. Sniffed if not set.
   */
  public Doc format(String format) {
    this.format = format;
    return this;
  }

  public String key() {
    return key;
  }

  public String content() {
    return content;
  }

  Map<String, Object> toJson() {
    Map<String, Object> m = new LinkedHashMap<>();
    m.put("key", key);
    m.put("content", content);
    if (!meta.isEmpty()) m.put("meta", meta);
    if (format != null) m.put("format", format);
    return m;
  }
}
