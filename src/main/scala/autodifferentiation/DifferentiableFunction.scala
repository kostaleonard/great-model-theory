package autodifferentiation

import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

//TODO docstring
trait DifferentiableFunction[T] {
  // TODO docstring
  def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]]

  // TODO docstring
  def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T]

  // TODO traverse graph and return inputs--you can use Scala reflect to get all of the case class parameters that are DifferentiableFunctions, check if they are Inputs, and recurse; children should not have to implement
  def getInputs: Set[Input[T]]

  // TODO can compute output shape from using placeholder inputs
  // TODO exhaustive case matching
  def getOutputShape(implicit classTag: ClassTag[T]): Array[Int] =
    computeOnZeroInputs match {
      case Success(outputs) => outputs.shape
      case _ => ???
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

case class Constant[T: ClassTag](value: NDArray[T])
    extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    Success(value)

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape.toList))

  override def getInputs: Set[Input[T]] = Set.empty
}

//TODO if we add implicit classtag to gradient and compute, we can make this a trait I think
abstract class Variable[T: ClassTag] extends DifferentiableFunction[T] {
  val name: String

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] =
    if (withRespectToVariable == this) Constant(NDArray.ones(List(1)))
    else Constant(NDArray.zeros(List(1)))
}

case class ModelParameter[T: ClassTag](
    override val name: String,
    value: NDArray[T]
) extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    Success(value)

  override def getInputs: Set[Input[T]] = Set.empty
}

case class Input[T: ClassTag](override val name: String, shape: Seq[Int])
    extends Variable[T] {
  // TODO if the given value for the input is of the wrong shape, also fail
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    inputs.get(this) match {
      case Some(value) => Success(value)
      case None =>
        Failure(new NoSuchElementException(f"Input $name is not defined"))
    }

  override def getInputs: Set[Input[T]] = Set(this)
}

case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(
    implicit num: Numeric[T]
) extends DifferentiableFunction[T] {
  // TODO can I define this short circuit binary computation somewhere? This will probably be a repeated code segment if I don't. Also could clean things up.
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

  override def gradient(
      withRespectToVariable: Variable[T]
  ): DifferentiableFunction[T] = ???

  override def getInputs: Set[Input[T]] = a.getInputs union b.getInputs
}
