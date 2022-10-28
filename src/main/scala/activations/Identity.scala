package activations

import ndarray.NDArray

/** The identity activation function.
  *
  * Leaves inputs unaltered.
  *
  * @tparam T
  *   The array element type.
  */
case class Identity[T]() extends Activation[T] {

  /** Returns the identity function on the inputs.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  override def activation(inputs: NDArray[T]): NDArray[T] = inputs
}
