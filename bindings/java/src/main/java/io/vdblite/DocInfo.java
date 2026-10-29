package io.vdblite;

import java.util.Map;

/** A summary of a stored document. */
public record DocInfo(String key, long docId, Map<String, Object> meta, long version, long createdMs, long updatedMs, long contentLength, int chunkCount) {}
