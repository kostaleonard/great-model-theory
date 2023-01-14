package autodifferentiation

import ndarray.NDArray

//TODO docstring
trait DifferentiableFunction[T] {

  def compute(inputs: Map[String, NDArray[T]]): NDArray[T]

  //TODO docstring
  //gradient of this function with respect to the arg at index
  def gradient(withRespectToInput: String): DifferentiableFunction[T]
}

case class Constant[T](value: NDArray[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): NDArray[T] = value

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] = Constant(NDArray.zeros(value.shape))
}

case class Input[T](name: String) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): NDArray[T] = inputs(name)

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] =
    if(withRespectToInput == name) Constant(NDArray.ones(Array(1)))
    else Constant(NDArray.zeros(Array(1)))
}

case class Add[T](a: DifferentiableFunction[T], b: DifferentiableFunction[T])(implicit num: Numeric[T]) extends DifferentiableFunction[T] {
  override def compute(inputs: Map[String, NDArray[T]]): NDArray[T] =
    (a.compute(inputs) + b.compute(inputs)).get

  override def gradient(withRespectToInput: String): DifferentiableFunction[T] =
    Add(a.gradient(withRespectToInput), b.gradient(withRespectToInput))
}
