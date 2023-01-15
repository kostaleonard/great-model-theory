package autodifferentiation

import ndarray.NDArray

import scala.util.{Failure, Success, Try}

//TODO docstring
trait DifferentiableFunction[T] {

  //TODO docstring
  def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]]

  //TODO docstring
  def gradient(withRespectToVariable: String): DifferentiableFunction[T]

  //TODO traverse graph and return inputs
  def getInputs: Set[Input[T]] = ???

  //TODO can compute output shape from using placeholder inputs
  def getOutputShape: Array[Int] = ???
}

case class Constant[T](value: NDArray[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = Success(value)

  override def gradient(withRespectToVariable: String): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape))
}

abstract case class Variable[T](name: String) extends DifferentiableFunction[T] {
  override def gradient(withRespectToVariable: String): DifferentiableFunction[T] =
    if(withRespectToVariable == name) Constant(NDArray.ones(Array(1)))
    else Constant(NDArray.zeros(Array(1)))
}

case class Parameter[T](override val name: String, value: NDArray[T]) extends Variable[T](name) {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = Success(value)
}

case class Input[T](override val name: String, shape: Array[Int]) extends Variable[T](name) {
  //TODO if the given value for the input is of the wrong shape, also fail
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = inputs.get(name) match {
    case Some(value) => Success(value)
    case None => Failure(new NoSuchElementException(f"Variable $name is not defined"))
  }
}

case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(implicit num: Numeric[T]) extends DifferentiableFunction[T] {
  //TODO can I define this short circuit binary computation somewhere? This will probably be a repeated code segment if I don't. Also could clean things up.
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] =
    a.compute(inputs) match {
      case Success(aValue) => b.compute(inputs) match {
        case Success(bValue) => aValue + bValue
        case _ => _
      }
      case _ => _
    }

  override def gradient(withRespectToVariable: String): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToVariable), b.gradient(withRespectToVariable))
}

case class Sigmoid[T](a: DifferentiableFunction[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = ???

  override def gradient(withRespectToVariable: String): DifferentiableFunction[T] = ???
}

case class MatMul[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = ???

  override def gradient(withRespectToVariable: String): DifferentiableFunction[T] = ???
}
