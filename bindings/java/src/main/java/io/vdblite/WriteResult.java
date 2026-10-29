package io.vdblite;

/** The outcome of one write. For an upsert: {@code embedded} counts the chunks that took new vectors (the rest kept theirs), and {@code changed} is false when content and metadata were identical. For a delete: {@code deleted} says whether the document existed. */
public record WriteResult(String op, String collection, String key, long docId, long version, int chunks, int embedded, boolean changed, boolean deleted) {}
