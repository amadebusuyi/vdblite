package io.vdblite;

import java.util.List;
import java.util.Map;

/** A chunk that matched a search. {@code byteStart}/{@code byteEnd} are UTF-8 byte offsets in the document. {@code score}: higher is better (cosine similarity, dot product or negative squared L2; in a hybrid search the fused rank score). {@code vectorScore} is null if only the keyword search found the chunk, {@code lexicalScore} if the keyword search didn't match it. */
public record ChunkHit(String key, long docId, int chunkId, int ordinal, int byteStart, int byteEnd, float score, Float vectorScore, Float lexicalScore, String text, List<String> headingPath, Map<String, Object> meta) {}
