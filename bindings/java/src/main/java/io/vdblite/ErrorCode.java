package io.vdblite;

/** The kind of a {@link VdbliteException}. */
public enum ErrorCode {
  /** The file or a transaction failed, or stored bytes were invalid. */
  STORAGE(1),
  /** A vector had the wrong number of dimensions. */
  DIMENSION(2),
  COLLECTION_EXISTS(3),
  NO_SUCH_COLLECTION(4),
  INVALID_ARGUMENT(5),
  /** The embedder failed. */
  EMBEDDER(6),
  /** The document changed after its embeddings were planned. */
  STALE_EMBEDDINGS(7),
  /** The file's format version isn't supported by this build. */
  VERSION(8),
  /** Sealing or compacting the index failed; nothing committed was lost. */
  MAINTENANCE(9),
  READ_ONLY(10),
  CLOSED(11),
  /** A bug in vdblite: please report it. */
  PANIC(12);

  private final int status;

  ErrorCode(int status) {
    this.status = status;
  }

  /** The C API's status code. */
  public int status() {
    return status;
  }

  static ErrorCode of(int status) {
    for (ErrorCode c : values()) {
      if (c.status == status) return c;
    }
    return STORAGE;
  }
}
