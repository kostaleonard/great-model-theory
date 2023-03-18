package layers

import autodifferentiation.{Add, DifferentiableFunction, DotProduct, ModelParameter}
import ndarray.NDArray

import java.util
import scala.reflect.ClassTag

object Dense {

  /** Returns a Dense layer with user-defined weights and biases.
    *
    * @param previousLayer
    *   The input to this layer.
    * @param units
    *   The number of neurons in the layer.
    * @param weightsInitialization
    *   If supplied, the weights matrix to use. Must be of shape
    *   (previousLayer.getOutputShape.last, units). If not supplied, the layer
    *   is initialized with random weights.
    * @param biasesInitialization
    *   If supplied, the biases vector to use. Must be of shape (units). If not
    *   supplied, the layer is initialized with random biases.
    * @tparam T
    *   The array element type.
    */
  def withInitialization[T: ClassTag](
      previousLayer: Layer[T],
      units: Int,
      weightsInitialization: Option[ModelParameter[T]] = None,
      biasesInitialization: Option[ModelParameter[T]] = None
  )(implicit num: Numeric[T]): Dense[T] = {
    val outputShape = previousLayer.getOutputShape
    val weights = weightsInitialization.getOrElse(
      ModelParameter(
        s"weights@Dense($previousLayer)",
        NDArray.random[T](Array(outputShape.last.get, units))
      )
    )
    val biases =
      biasesInitialization.getOrElse(
        ModelParameter(
          s"biases@Dense($previousLayer)",
          NDArray.zeros[T](Array(units))
        )
      )
    Dense(previousLayer, units, weights, biases)
  }

  /** Returns a Dense layer with randomly-initialized weights and biases.
    *
    * @param previousLayer
    *   The input to this layer.
    * @param units
    *   The number of neurons in the layer.
    * @tparam T
    *   The array element type.
    */
  def withRandomWeights[T: ClassTag](
      previousLayer: Layer[T],
      units: Int
  )(implicit num: Numeric[T]): Dense[T] = {
    val outputShape = previousLayer.getOutputShape
    val weights =
      NDArray.random[T](Array(outputShape.last.get, units))
    val biases = NDArray.zeros[T](Array(units))
    Dense(
      previousLayer,
      units,
      ModelParameter(s"weights@Dense($previousLayer)", weights),
      ModelParameter(s"biases@Dense($previousLayer)", biases)
    )
  }
}

/** A densely connected neural network layer.
  *
  * Implements the operation outputs = dot(inputs, kernel) + bias where kernel
  * is a weights matrix created by the layer, and bias is a bias vector created
  * by the layer. It is typical to follow this layer with an activation function
  * layer like Sigmoid to introduce nonlinearity.
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
  *   The weights matrix to use. Must be of shape
  *   (previousLayer.getOutputShape.last, units).
  * @param biases
  *   The biases vector to use. Must be of shape (units).
  * @tparam T
  *   The array element type.
  */
case class Dense[T: ClassTag](
    previousLayer: Layer[T],
    units: Int,
    weights: ModelParameter[T],
    biases: ModelParameter[T]
)(implicit implicitNumeric: Numeric[T])
    extends Layer[T] {

  override def getComputationGraph: DifferentiableFunction[T] =
    Add(
      DotProduct(
        previousLayer.getComputationGraph,
        weights
      ),
      biases
    )

  override def withUpdatedParameters(
      parameters: util.IdentityHashMap[ModelParameter[T], ModelParameter[T]]
  ): Layer[T] =
    Dense(
      previousLayer.withUpdatedParameters(parameters),
      units,
      parameters.getOrDefault(weights, weights),
      parameters.getOrDefault(biases, biases)
    )
}
