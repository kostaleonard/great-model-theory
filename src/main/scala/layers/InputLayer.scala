package layers
import autodifferentiation.{DifferentiableFunction, Input}

case class InputLayer[T](name: String, shape: Array[Int]) extends Layer[T] {
  override val layerNum: Int = 0

  override def getComputationGraph: DifferentiableFunction[T] = Input(name, shape)
}
