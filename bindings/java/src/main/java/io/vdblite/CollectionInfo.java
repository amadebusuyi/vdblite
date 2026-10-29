package io.vdblite;

/** A collection's settings. {@code targetTokens} and {@code overlapTokens} are 0 without structural chunking. */
public record CollectionInfo(String name, int dim, String metric, String quantization, String chunking, int targetTokens, int overlapTokens) {}
