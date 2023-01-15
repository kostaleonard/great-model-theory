package layers
import autodifferentiation.{DifferentiableFunction, Input}

import scala.reflect.ClassTag

//TODO docstring
case class InputLayer[T: ClassTag](input: Input[T]) extends Layer[T] {
  override def getComputationGraph: DifferentiableFunction[T] = input
}
