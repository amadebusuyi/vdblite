package io.vdblite;

/** A collection's index. {@code maintenanceError} is the last background maintenance failure, or null. */
public record IndexStats(long chunks, int sealedSegments, long buffered, long memoryBytes, String maintenanceError) {}
