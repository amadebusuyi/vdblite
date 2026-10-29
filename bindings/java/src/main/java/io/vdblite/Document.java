package io.vdblite;

import java.util.Map;

/** A stored document with its content. {@code format} is "markdown", "code", "html" or "text". */
public record Document(String key, long docId, String content, Map<String, Object> meta, long version, long createdMs, long updatedMs, String format, int chunkCount) {}
