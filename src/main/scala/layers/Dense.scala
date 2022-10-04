package layers

import ndarray.NDArray

import scala.reflect.ClassTag

//TODO add glorot uniform as function in companion object?
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
  * @param weightsInitialization
  *   The initial weights matrix to use. Must be of shape
  *   [[(inputShape.last, units)]]. If not provided, the weights matrix is
  *   randomly initialized.
  * @param biasesInitialization
  *   The initial biases vector to use. Must be of shape [[(units]]. If not
  *   provided, the biases vector is set to zero.
  * @tparam T
  *   The array element type.
  */
class Dense[T: ClassTag](
    inputShape: Array[Int],
    units: Int,
    weightsInitialization: Option[NDArray[T]] = None,
    biasesInitialization: Option[NDArray[T]] = None
) extends Layer[T] {
  val weights: NDArray[T] = weightsInitialization.getOrElse(
    NDArray.random[T](List(inputShape.last, units))
  )
  val biases: NDArray[T] =
    biasesInitialization.getOrElse(NDArray.zeros[T](List(units)))

  // TODO implement actual dense transformation
  /** Returns the layer's transformation on the inputs.
    *
    * Multiplies all values of the input tensor by the learned layer weights.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  def apply(inputs: NDArray[T]): NDArray[T] = inputs
}
