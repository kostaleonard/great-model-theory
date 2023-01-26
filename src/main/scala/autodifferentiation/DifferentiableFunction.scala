package autodifferentiation

import exceptions.ShapeException
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

  // TODO implement
  // TODO add tests for this under DifferentiableFunction ("A DifferentiableFunction should...")
  /** Returns the set of all inputs to the function. */
  def getInputs: Set[Input[T]]

  //TODO implement here if possible
  //TODO I don't think we need the classtag anymore
  /** Returns the output shape of the function with possible placeholders. */
  def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]]
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

  override def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]] = Success(value.shape.map(Some(_)))
}

/** A variable (has potentially non-zero gradient).
  *
  * @tparam T
  *   The array element type.
  */
trait Variable[T] extends DifferentiableFunction[T] {
  val name: String
  implicit val classTag: ClassTag[T]

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
case class ModelParameter[T](
    override val name: String,
    value: NDArray[T]
)(override implicit val classTag: ClassTag[T])
    extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    Success(value)

  override def getInputs: Set[Input[T]] = Set.empty

  override def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]] = Success(value.shape.map(Some(_)))
}

/** An input variable that users supply.
  *
  * Passes user-defined values to the computation graph.
  *
  * @param name
  *   The name of the variable.
  * @param shapeWithPlaceholders
  *   The shape of the array containing the variable, with None for dimensions
  *   that can vary at run time (e.g., the batch dimension).
  * @tparam T
  *   The array element type.
  */
case class Input[T](
    override val name: String,
    shapeWithPlaceholders: Array[Option[Int]]
)(override implicit val classTag: ClassTag[T])
    extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    inputs.get(this) match {
      case Some(value) =>
        if (value.shape.length != shapeWithPlaceholders.length)
          Failure(
            new ShapeException(
              s"Input $name expects values of shape ${shapeWithPlaceholders
                  .mkString("Array(", ", ", ")")}, but got ${value.shape
                  .mkString("Array(", ", ", ")")}"
            )
          )
        else if (
          shapeWithPlaceholders.indices.forall(idx =>
            shapeWithPlaceholders(idx) match {
              case Some(dimension) => value.shape(idx) == dimension
              case _               => true
            }
          )
        ) Success(value)
        else
          Failure(
            new ShapeException(
              s"Input $name expects values of shape ${shapeWithPlaceholders
                  .mkString("Array(", ", ", ")")}, but got ${value.shape
                  .mkString("Array(", ", ", ")")}"
            )
          )
      case None =>
        Failure(new NoSuchElementException(s"Input $name is not defined"))
    }

  override def getInputs: Set[Input[T]] = Set(this)

  override def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]] = Success(shapeWithPlaceholders)
}

//TODO test compute, gradient
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

  //TODO remember that None can get filled in with any value at runtime, so None in a must correspond with Some in b and vice versa
  override def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]] =
    a.getOutputShape match {
      case Success(aShape) => b.getOutputShape match {
        case Success(bShape) => ???
        case failure => failure
      }
      case failure => failure
    }
}

//TODO test compute, gradient, getOutputShape
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

  // TODO implement gradient
  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = ???

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs

  override def getOutputShape(implicit classTag: ClassTag[T]): Try[Array[Option[Int]]] = ???
}
