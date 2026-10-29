package io.vdblite;

import java.util.List;

/** Turns texts into vectors, one per text, in order: any embedding model. */
@FunctionalInterface
public interface Embedder {
  List<float[]> embed(List<String> texts) throws Exception;
}
