package io.vdblite;

import java.util.List;
import java.util.Map;

/** A document ranked by its chunks; {@code chunks} are its matching chunks, best first. */
public record DocHit(String key, long docId, float score, Map<String, Object> meta, List<ChunkHit> chunks) {}
