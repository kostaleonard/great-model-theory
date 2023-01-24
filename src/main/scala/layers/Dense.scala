package layers

import activations.{Activation, Identity}
import autodifferentiation.{Add, DifferentiableFunction, MatMul, ModelParameter}
import ndarray.NDArray

import scala.reflect.ClassTag

object Dense {

  def withInitialization[T: ClassTag](
      previousLayer: Layer[T],
      units: Int,
      weightsInitialization: Option[NDArray[T]] = None,
      biasesInitialization: Option[NDArray[T]] = None,
      activation: Activation[T] = Identity[T]()
  )(implicit num: Numeric[T]): Dense[T] = {
    val weights: NDArray[T] = weightsInitialization.getOrElse(
      NDArray.random[T](List(previousLayer.getOutputShape.last, units))
    )
    val biases: NDArray[T] =
      biasesInitialization.getOrElse(NDArray.zeros[T](List(units)))
    Dense(previousLayer, units, weights, biases, activation)
  }

  def withRandomWeights[T: ClassTag](
      previousLayer: Layer[T],
      units: Int,
      activation: Activation[T] = Identity[T]()
  )(implicit num: Numeric[T]): Dense[T] = {
    val weights =
      NDArray.random[T](List(previousLayer.getOutputShape.last, units))
    val biases = NDArray.zeros[T](List(units))
    Dense(previousLayer, units, weights, biases, activation)
  }
}

/** A densely connected neural network layer.
  *
  * Implements the operation outputs = activation(dot(inputs, kernel) + bias)
  * where activation is the element-wise activation function, kernel is a
  * weights matrix created by the layer, and bias is a bias vector created by
  * the layer.
  *
  * If the input to the layer has a rank greater than 2, then this layer
  * computes the dot product between the inputs and the kernel along the last
  * axis of the inputs and axis 0 of the kernel. For example, if the inputs have
  * dimensions (batchSize, d0, d1), then we create a kernel with shape (d1,
  * units), and the kernel operates along axis 2 of the inputs, on every
  * sub-tensor of shape (1, 1, d1) (there are batchSize * d0 such sub-tensors).
  * The output in this case will have shape (batchSize, d0, units).
  *
  * @param previousLayer
  *   The input to this layer.
  * @param units
  *   The number of neurons in the layer.
  * @param weights
  *   The weights matrix to use. Must be of shape (inputShape.last, units).
  * @param biases
  *   The biases vector to use. Must be of shape (units).
  * @param activation
  *   The activation function to apply after the dense transformation.
  * @tparam T
  *   The array element type.
  */
case class Dense[T: ClassTag] private (
    previousLayer: Layer[T],
    units: Int,
    weights: NDArray[T],
    biases: NDArray[T],
    activation: Activation[T]
)(implicit implicitNumeric: Numeric[T])
    extends Layer[T] {

  override def getComputationGraph: DifferentiableFunction[T] =
    Add(
      MatMul(
        previousLayer.getComputationGraph,
        ModelParameter("weights", weights)
      ),
      ModelParameter("biases", biases)
    )(implicitNumeric)
}
