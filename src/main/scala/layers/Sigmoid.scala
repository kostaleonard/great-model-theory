package layers

import autodifferentiation.{
  Add,
  Constant,
  DifferentiableFunction,
  Exp,
  Negate,
  Reciprocal
}
import ndarray.NDArray

import scala.reflect.ClassTag

/** A sigmoid neural network layer.
  *
  * Implements the operation outputs = 1 / (1 + exp(-inputs)). Typically used to
  * introduce nonlinearity into the model (i.e., an activation function). The
  * output will be the same shape as the input with all values in the range (0,
  * 1).
  *
  * @param previousLayer
  *   The input to this layer.
  * @tparam T
  *   The array element type.
  */
case class Sigmoid[T: ClassTag](previousLayer: Layer[T])(implicit
    implicitNumeric: Fractional[T]
) extends Layer[T] {

  override def getComputationGraph: DifferentiableFunction[T] =
    Reciprocal(
      Add(
        Constant(NDArray.ones[T](Array(1))),
        Exp(Negate(previousLayer.getComputationGraph))
      )
    )
}
