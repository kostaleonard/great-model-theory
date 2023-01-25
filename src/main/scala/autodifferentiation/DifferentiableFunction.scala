package autodifferentiation

import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

/** A function that is differentiable with respect to its variables.
  *
  * Used to define a compute graph for neural networks.
  *
  * @tparam T
  *   The array element type.
  */
trait DifferentiableFunction[T] {

  /** Returns the output of the function for the given input values.
    *
    * @param inputs
    *   A Map of `Input` objects to tensors of arbitrary shape.
    */
  def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]]

  /** Returns the gradient of the function with respect to a variable.
    *
    * @param withRespectToVariable
    *   The variable with which to compute the gradient. If we call this
    *   DifferentiableFunction y and the variable x, this operation produces
    *   dy/dx.
    */
  def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T]

  /** Returns the set of all inputs to the function. */
  def getInputs: Set[Input[T]]

  //TODO use Try
  /** Returns the output shape of the function. */
  def getOutputShape(implicit classTag: ClassTag[T]): Array[Int] =
    computeOnZeroInputs match {
      case Success(outputs) => outputs.shape
      case _                => ???
    }

  private def computeOnZeroInputs(implicit
      classTag: ClassTag[T]
  ): Try[NDArray[T]] = {
    val inputs = getInputs
    val zeroInputs =
      inputs.map(input => input -> NDArray.zeros[T](input.shape)).toMap
    compute(zeroInputs)
  }
}

/** A constant (has 0 gradient).
  *
  * @param value
  *   The constant's value.
  * @tparam T
  *   The array element type.
  */
case class Constant[T: ClassTag](value: NDArray[T])
    extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    Success(value)

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape))

  override def getInputs: Set[Input[T]] = Set.empty
}

/** A variable (has potentially non-zero gradient).
  *
  * @tparam T
  *   The array element type.
  */
abstract class Variable[T: ClassTag] extends DifferentiableFunction[T] {
  val name: String

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    if (withRespectToVariable == this) Constant(NDArray.ones(Array(1)))
    else Constant(NDArray.zeros(Array(1)))
}

/** A model parameter.
  *
  * @param name
  *   The name of the variable.
  * @param value
  *   The current value of the parameter.
  * @tparam T
  *   The array element type.
  */
case class ModelParameter[T: ClassTag](
    override val name: String,
    value: NDArray[T]
) extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    Success(value)

  override def getInputs: Set[Input[T]] = Set.empty
}

/** An input variable that users supply.
  *
  * Passes user-defined values to the computation graph.
  *
  * @param name
  *   The name of the variable.
  * @param shape
  *   The shape of the array containing the variable.
  * @tparam T
  *   The array element type.
  */
case class Input[T: ClassTag](override val name: String, shape: Array[Int])
    extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    inputs.get(this) match {
      case Some(value) => Success(value)
      case None =>
        Failure(new NoSuchElementException(f"Input $name is not defined"))
    }

  override def getInputs: Set[Input[T]] = Set(this)
}

/** Adds the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(
    implicit num: Numeric[T]
) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue + bValue
          case failure         => failure
        }
      case failure => failure
    }

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToVariable), b.gradient(withRespectToVariable))

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs
}

/** Matrix multiplies the results of two functions.
  *
  * @param a
  *   The left hand side.
  * @param b
  *   The right hand side.
  * @param num
  *   The implicit numeric conversion.
  * @tparam T
  *   The array element type.
  */
case class MatMul[T](
    a: DifferentiableFunction[T],
    b: DifferentiableFunction[T]
)(implicit num: Numeric[T])
    extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) =>
        b.compute(inputs) match {
          case Success(bValue) => aValue dot bValue
          case failure         => failure
        }
      case failure => failure
    }

  //TODO implement gradient
  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = ???

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs
}
