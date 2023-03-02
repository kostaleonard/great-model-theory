package layers

import autodifferentiation.{
  DifferentiableFunction,
  Input,
  Mean,
  ModelParameter,
  Square,
  Subtract
}

import scala.reflect.ClassTag

/** Computes the mean squared error of the previous layer given outputs.
  *
  * @param previousLayer
  *   The input to this layer.
  * @tparam T
  *   The array element type.
  */
case class MeanSquaredError[T: ClassTag](previousLayer: Layer[T])(implicit
    numeric: Fractional[T]
) extends Layer[T] {
  val labelsInput: Input[T] = Input("yTrue", previousLayer.getOutputShape)

  override def getComputationGraph: DifferentiableFunction[T] =
    Mean(
      Square(
        Subtract(
          previousLayer.getComputationGraph,
          labelsInput
        )
      )
    )

  // This layer has no parameters.
  override def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Layer[T] = this
}
