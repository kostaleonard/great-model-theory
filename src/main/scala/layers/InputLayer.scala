package layers
import autodifferentiation.{DifferentiableFunction, Input, ModelParameter}

import scala.reflect.ClassTag

/** An input to a neural network.
  *
  * @param input
  *   The placeholder input for the neural network, which is filled in with the
  *   actual NDArray[T] elements of the (training, prediction, etc.) dataset at
  *   run time.
  * @tparam T
  *   The array element type.
  */
case class InputLayer[T: ClassTag](input: Input[T]) extends Layer[T] {
  override def getComputationGraph: DifferentiableFunction[T] = input

  // This layer has no parameters.
  override def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Layer[T] = this
}
