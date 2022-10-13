package losses
import exceptions.ShapeException
import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/** The mean squared error loss function.
  *
  * Computes the mean of the squared error by index. The mean is applied on the
  * last dimension.
  *
  * @tparam T
  *   The array element type.
  */
class MeanSquaredError[T: ClassTag](implicit num: Fractional[T])
    extends Loss[T] {

  /** Returns the mean squared error by index.
    *
    * The mean is applied on the last dimension.
    *
    * @param y_true
    *   The ground truth tensor of shape (batchSize, d0, d1, ..., dN).
    * @param y_pred
    *   The prediction tensor of shape (batchSize, d0, d1, ..., dN).
    * @return
    *   The loss tensor of shape (batchSize, d0, d1, ..., dN-1).
    */
  override def compute_loss(
      y_true: NDArray[T],
      y_pred: NDArray[T]
  ): Try[NDArray[T]] =
    if (!(y_true.shape sameElements y_pred.shape))
      Failure(new ShapeException("Loss inputs must have matching shape"))
    else {
      val squaredError =
        (y_true - y_pred).get.map(diff => num.times(diff, diff))
      val axis = y_true.shape.length - 1
      val meanSquaredError = squaredError.reduce(
        arr =>
          num.div(
            arr.flatten().reduce(num.plus),
            num.fromInt(arr.flatten().length)
          ),
        axis
      )
      Success(meanSquaredError)
    }
}
