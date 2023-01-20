package layers

import autodifferentiation.{DifferentiableFunction, Input, ModelParameter}
import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.Try

/** A neural network layer.
  *
  * @tparam T
  *   The array element type.
  */
abstract class Layer[T: ClassTag] {

  // TODO docstring
  def getComputationGraph: DifferentiableFunction[T]

  // TODO update docstring
  /** Returns the layer's transformation on the inputs.
    *
    * @param inputs
    *   The input tensor of arbitrary shape. The first dimension is the batch
    *   dimension.
    */
  def apply(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    getComputationGraph.compute(inputs)

  def getInputs: Set[Input[T]] = getComputationGraph.getInputs

  def getOutputShape: Array[Int] = getComputationGraph.getOutputShape

  def getTrainableVariables: Set[ModelParameter[T]] = ???
}
