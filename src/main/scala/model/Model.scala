package model

import autodifferentiation.{Input, ModelParameter}
import layers.Layer
import ndarray.NDArray

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
  def apply(inputs: Map[Input[T], NDArray[T]]): NDArray[T] = outputLayer(
    inputs
  )

  /** Returns a new instance of the model with updated parameters.
    *
    * @param parameters
    *   A Map in which the keys are the current parameters and the values are
    *   the parameters that should replace them. Any keys not found in the model
    *   are ignored.
    */
  def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Model[T] = Model(outputLayer.withUpdatedParameters(parameters))
}
