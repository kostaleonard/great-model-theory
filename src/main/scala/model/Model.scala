package model

import autodifferentiation.{Input, ModelParameter}
import layers.Layer
import ndarray.NDArray

import scala.util.Try

/** A neural network.
  *
  * @param outputLayer
  *   The final layer of the neural network.
  * @tparam T
  *   The array element type.
  */
case class Model[T](outputLayer: Layer[T]) {

  /** Returns the output of the model on the inputs.
    *
    * @param inputs
    *   The inputs to the model.
    */
  def apply(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = outputLayer(
    inputs
  )

  // TODO getLayers: List[Layer[T]]? Would need to change constructor, or guarantee that Layer has a previousLayer: Option[Layer[T]] field

  // TODO docstring
  // TODO we want this to preserve layer references
  def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Model[T] = Model(outputLayer.withUpdatedParameters(parameters))
}
