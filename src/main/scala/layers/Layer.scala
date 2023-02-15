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

  /** Returns the layer's computation graph.
    *
    * The computation graph defines the transformations the layer makes on
    * inputs. It is a composition of `DifferentiableFunction`s from which the
    * model can compute gradients during training.
    */
  def getComputationGraph: DifferentiableFunction[T]

  /** Returns the layer's transformation on the inputs.
    *
    * @param inputs
    *   A Map of `Input` objects to tensors of arbitrary shape.
    */
  def apply(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    getComputationGraph.compute(inputs)

  /** Returns the layer's `Input` objects. */
  def getInputs: Set[Input[T]] = getComputationGraph.getInputs

  /** Returns the layer's output shape with possible placeholder dimensions. */
  def getOutputShape: Try[Array[Option[Int]]] =
    getComputationGraph.getOutputShape

  /** Returns a new instance of the layer with updated parameters.
    *
    * @param parameters
    *   A Map in which the keys are the current parameters and the values are
    *   the parameters that should replace them. Any keys not found in the layer
    *   are ignored.
    */
  def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Layer[T]
}
