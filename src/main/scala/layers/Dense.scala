package layers

import ndarray.NDArray

/**
 * A densely connected neural network layer.
 *
 * @param units The number of neurons in the layer.
 * */
class Dense[T](units: Int) extends Layer[T] {

  //TODO implement actual dense transformation
  /**
   * Returns the layer's transformation on the inputs.
   *
   * Multiplies all values of the input tensor by the learned layer weights.
   *
   * @param inputs The input tensor of arbitrary shape.
   */
  def call(inputs: NDArray[T]): NDArray[T] = inputs
}
