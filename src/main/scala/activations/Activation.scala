package activations

import ndarray.NDArray

/** An activation function.
  *
  * An activation function is a typically nonlinear function that is applied
  * after a layer's transformation. The purpose of the activation function is to
  * introduce nonlinearity into the neural network's representation of the data,
  * expanding the range of functions that the network can model, or to otherwise
  * modify the data to have desirable qualities.
  *
  * @tparam T
  *   The array element type.
  */
trait Activation[T] {

  /** Returns the result of applying the activation function on the inputs.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  def activation(inputs: NDArray[T]): NDArray[T]
}
