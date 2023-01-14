package layers

import autodifferentiation.DifferentiableFunction
import ndarray.NDArray

import scala.util.Try

/** A neural network layer.
  *
  * @tparam T
  *   The array element type.
  */
abstract class Layer[T] {

  def getComputationGraph: DifferentiableFunction[T]

  /** Returns the layer's transformation on the inputs.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  def apply(inputs: NDArray[T]): Try[NDArray[T]] = getComputationGraph.compute(inputs)
}
