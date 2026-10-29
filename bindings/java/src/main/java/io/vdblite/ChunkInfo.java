package io.vdblite;

import java.util.List;

/** One chunk of a stored document. */
public record ChunkInfo(int chunkId, int ordinal, int byteStart, int byteEnd, int overlapStart, List<String> headingPath, int tokens) {}
