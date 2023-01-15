package model

import layers.Layer
import ndarray.NDArray

import scala.annotation.tailrec
import scala.util.{Failure, Success, Try}

//TODO update docstring
/** A neural network.
  *
  * @param layers
  *   The layers of the neural network.
  * @tparam T
  *   The array element type.
  */
class Model[T](outputLayer: Layer[T]) {

  /** Returns the output of the model on the inputs.
    *
    * @param inputs
    *   The inputs to the model.
    */
  def apply(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = outputLayer(inputs)
}
