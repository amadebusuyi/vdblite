package io.vdblite;

import java.util.List;
import java.util.Map;

/** A hit widened to neighbouring chunks or its whole section; it covers chunk ordinals [{@code ordinalStart}, {@code ordinalEnd}). */
public record Passage(String key, long docId, int byteStart, int byteEnd, float score, String text, int ordinalStart, int ordinalEnd, List<String> headingPath, Map<String, Object> meta) {}
