package autodifferentiation

import ndarray.NDArray

import scala.reflect.ClassTag
import scala.util.{Failure, Success, Try}

//TODO docstring
trait DifferentiableFunction[T] {

  //TODO docstring
  def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]]

  //TODO docstring
  def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T]

  //TODO traverse graph and return inputs
  def getInputs: Set[Input[T]] = ???

  //TODO can compute output shape from using placeholder inputs
  def getOutputShape: Array[Int] = ???
}

case class Constant[T: ClassTag](value: NDArray[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = Success(value)

  override def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape.toList))
}

abstract class Variable[T: ClassTag] extends DifferentiableFunction[T] {
  val name: String

  override def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T] =
    if(withRespectToVariable == this) Constant(NDArray.ones(List(1)))
    else Constant(NDArray.zeros(List(1)))
}

case class ModelParameter[T: ClassTag](override val name: String, value: NDArray[T]) extends Variable[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = Success(value)
}

case class Input[T: ClassTag](override val name: String, shape: Array[Int]) extends Variable[T] {
  //TODO if the given value for the input is of the wrong shape, also fail
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = inputs.get(this) match {
    case Some(value) => Success(value)
    case None => Failure(new NoSuchElementException(f"Input $name is not defined"))
  }
}

case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(implicit num: Numeric[T]) extends DifferentiableFunction[T] {
  //TODO can I define this short circuit binary computation somewhere? This will probably be a repeated code segment if I don't. Also could clean things up.
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) => b.compute(inputs) match {
        case Success(bValue) => aValue + bValue
        case failure => failure
      }
      case failure => failure
    }

  override def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToVariable), b.gradient(withRespectToVariable))
}

case class Sigmoid[T](a: DifferentiableFunction[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = ???

  override def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T] = ???
}

case class MatMul[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[Input[T], NDArray[T]]): Try[NDArray[T]] = ???

  override def gradient(withRespectToVariable: Variable[T]): DifferentiableFunction[T] = ???
}
