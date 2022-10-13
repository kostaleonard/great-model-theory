package losses
import ndarray.NDArray

/** The mean squared error loss function.
  *
  * Computes the mean of the squared error by index. The mean is applied on the
  * last dimension.
  *
  * @tparam T
  *   The array element type.
  */
class MeanSquaredError[T] extends Loss[T] {

  /** Returns the mean squared error by index.
    *
    * The mean is applied on the last dimension.
    *
    * @param y_true
    *   The ground truth tensor of shape (batchSize, d0, d1, ..., dN).
    * @param y_pred
    *   The prediction tensor of shape (batchSize, d0, d1, ..., dN).
    *  @return
    *   The loss tensor of shape (batchSize, d0, d1, ..., dN-1).
    */
  override def compute_loss(y_true: NDArray[T], y_pred: NDArray[T]): NDArray[T] = ???
}
