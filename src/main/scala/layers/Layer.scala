package layers

import ndarray.NDArray

/**
 * A neural network layer.
 */
abstract class Layer[T] {

  /**
   * Returns the layer's transformation on the inputs.
   *
   * @param inputs The input tensor of arbitrary shape.
   */
  def call(inputs: NDArray[T]): NDArray[T]
}
