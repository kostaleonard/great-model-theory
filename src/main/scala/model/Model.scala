package model

import layers.Layer
import ndarray.NDArray

import scala.annotation.tailrec
import scala.util.{Failure, Success, Try}

/** A neural network.
  *
  * @param layers
  *   The layers of the neural network.
  * @tparam T
  *   The array element type.
  */
class Model[T](layers: List[Layer[T]]) {

  /** Returns the output of the model on the inputs.
    *
    * @param inputs
    *   The inputs to the model.
    */
  def apply(inputs: NDArray[T]): Try[NDArray[T]] = {
    @tailrec
    def applyRecursive(
        layerInputs: NDArray[T],
        remainingLayers: List[Layer[T]]
    ): Try[NDArray[T]] = {
      if (remainingLayers.isEmpty) Success(layerInputs)
      else {
        val layerOutputs = remainingLayers.head.apply(layerInputs)
        layerOutputs match {
          case Success(arr) => applyRecursive(arr, remainingLayers.tail)
          case Failure(_)   => layerOutputs
        }
      }
    }
    applyRecursive(inputs, layers)
  }

  // TODO fit, predict (this is just apply), and evaluate
}
