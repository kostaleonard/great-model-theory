package losses

import ndarray.NDArray

import scala.util.Try

/** A loss function.
  *
  * The loss function is used to compute the gradient for gradient descent and
  * other optimization algorithms.
  *
  * @tparam T
  *   The array element type.
  */
trait Loss[T] {

  /** Returns the loss on the inputs.
    *
    * @param y_true
    *   The ground truth tensor of shape (batchSize, d0, d1, ..., dN).
    * @param y_pred
    *   The prediction tensor of shape (batchSize, d0, d1, ..., dN).
    * @return
    *   The loss tensor of shape (batchSize, d0, d1, ..., dN-1).
    */
  def compute_loss(y_true: NDArray[T], y_pred: NDArray[T]): Try[NDArray[T]]
}
