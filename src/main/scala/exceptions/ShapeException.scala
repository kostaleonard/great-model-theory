package exceptions

/** Thrown when the user performs an operation on an incorrectly shaped NDArray.
  *
  * @param message
  *   The error message.
  */
class ShapeException(message: String) extends Exception(message) {

  /** Create with message and cause.
    *
    * @param message
    *   The error message.
    * @param cause
    *   The stack trace.
    */
  def this(message: String, cause: Throwable) {
    this(message)
    initCause(cause)
  }

  /** Create with cause.
    *
    * @param cause
    *   The stack trace.
    */
  def this(cause: Throwable) {
    this(Option(cause).map(_.toString).orNull, cause)
  }

  /** Create an empty exception.
    */
  def this() {
    this(null: String)
  }
}
