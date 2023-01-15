package layers
import autodifferentiation.{DifferentiableFunction, Input}

//TODO docstring
case class InputLayer[T](input: Input[T]) extends Layer[T] {
  override def getComputationGraph: DifferentiableFunction[T] = input
}
