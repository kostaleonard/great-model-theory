package autodifferentiation

import ndarray.NDArray
import scala.util.{Failure, Success, Try}

//TODO docstring
trait DifferentiableFunction[T] {

  //TODO docstring
  def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]]

  //TODO docstring
  def gradient(withRespectToInput: String): DifferentiableFunction[T]
}

case class Constant[T](value: NDArray[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = Success(value)

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape))
}

case class Input[T](name: String) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): Try[NDArray[T]] = inputs.get(name) match {
    case Some(value) => Success(value)
    case None => Failure(new NoSuchElementException(f"Input $name is not defined"))
  }

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] =
    if(withRespectToInput == name) Constant(NDArray.ones(Array(1)))
    else Constant(NDArray.zeros(Array(1)))
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

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToInput), b.gradient(withRespectToInput))
}
