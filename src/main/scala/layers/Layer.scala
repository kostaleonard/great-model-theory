package layers

import autodifferentiation.{DifferentiableFunction, Input}
import ndarray.NDArray

import scala.util.Try

/** A neural network layer.
  *
  * @tparam T
  *   The array element type.
  */
abstract class Layer[T] {
  //TODO better way to make input names unique--that is all this is for
  val layerNum: Int

  //TODO docstring
  def getComputationGraph: DifferentiableFunction[T]

  //TODO update docstring
  /** Returns the layer's transformation on the inputs.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  def apply(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = getComputationGraph.compute(inputs)

  def getInputs: Set[Input[T]] = getComputationGraph.getInputs

  def getOutputShape: Array[Int] = getComputationGraph.getOutputShape
}
