package io.vdblite;

/** A failed vdblite call; {@link #code()} says what kind. */
public class VdbliteException extends RuntimeException {
  private final ErrorCode code;

  public VdbliteException(ErrorCode code, String message) {
    super(code + ": " + message);
    this.code = code;
  }

  public ErrorCode code() {
    return code;
  }
}
