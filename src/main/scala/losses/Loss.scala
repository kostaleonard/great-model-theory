package losses

import autodifferentiation.DifferentiableFunction
import ndarray.NDArray

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
    * @param yTrue
    *   The ground truth tensor of shape (batchSize, d0, d1, ..., dN).
    * @param yPred
    *   The prediction tensor of shape (batchSize, d0, d1, ..., dN).
    * @return
    *   The loss tensor of shape (batchSize, d0, d1, ..., dN-1).
    */
  def compute_loss(yTrue: NDArray[T], yPred: NDArray[T]): NDArray[T]

  /** Returns the loss's computation graph.
    *
    * The computation graph defines the transformations the loss makes on
    * inputs. It is a composition of `DifferentiableFunction`s from which the
    * model can compute gradients during training.
    */
  def getComputationGraph: DifferentiableFunction[T]
}
