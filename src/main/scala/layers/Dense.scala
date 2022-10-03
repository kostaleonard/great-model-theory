package layers

import ndarray.NDArray

//TODO eventually we want to not need/allow users to supply the input shape
/** A densely connected neural network layer.
  *
  * Implements the operation
  * [[outputs = activation(dot(inputs, kernel) + bias)]] where activation is the
  * element-wise activation function, kernel is a weights matrix created by the
  * layer, and bias is a bias vector created by the layer.
  *
  * If the input to the layer has a rank greater than 2, then this layer
  * computes the dot product between the inputs and the kernel along the last
  * axis of the inputs and axis 0 of the kernel. For example, if the inputs have
  * dimensions (batchSize, d0, d1), then we create a kernel with shape (d1,
  * units), and the kernel operates along axis 2 of the inputs, on every
  * sub-tensor of shape (1, 1, d1) (there are batchSize * d0 such sub-tensors).
  * The output in this case will have shape (batchSize, d0, units).
  *
  * @param inputShape
  *   The shape of the input arrays that will be passed into this layer. The
  *   first dimension is considered the batch dimension, and is ignored.
  * @param units
  *   The number of neurons in the layer.
  */
class Dense[T](inputShape: Array[Int], units: Int) extends Layer[T] {
  // Weights are randomly initialized, but biases are initialized to 0.
  val weights: NDArray[T] = NDArray.random[T](Array(inputShape.last, units))
  val biases: NDArray[T] = NDArray.zeros[T](Array(units))

  // TODO implement actual dense transformation
  /** Returns the layer's transformation on the inputs.
    *
    * Multiplies all values of the input tensor by the learned layer weights.
    *
    * @param inputs
    *   The input tensor of arbitrary shape.
    */
  def call(inputs: NDArray[T]): NDArray[T] = inputs
}
