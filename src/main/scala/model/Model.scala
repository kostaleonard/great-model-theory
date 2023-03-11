package model

import autodifferentiation.{Input, ModelParameter}
import layers.{Layer, MeanSquaredError}
import ndarray.NDArray

import scala.reflect.ClassTag

/** A neural network.
  *
  * @param outputLayer
  *   The final layer of the neural network.
  * @tparam T
  *   The array element type.
  */
case class Model[T: ClassTag](outputLayer: Layer[T]) {

  /** Returns the output of the model on the inputs.
    *
    * @param inputs
    *   The inputs to the model.
    */
  def apply(inputs: Map[Input[T], NDArray[T]]): NDArray[T] = outputLayer(
    inputs
  )

  /** Returns a new instance of the model with updated parameters.
    *
    * @param parameters
    *   A Map in which the keys are the current parameters and the values are
    *   the parameters that should replace them. Any keys not found in the model
    *   are ignored.
    */
  def withUpdatedParameters(
      parameters: Map[ModelParameter[T], ModelParameter[T]]
  ): Model[T] = Model(outputLayer.withUpdatedParameters(parameters))

  //TODO docstring
  def fit(
      inputs: Map[Input[T], NDArray[T]],
      labels: NDArray[T],
      epochs: Int,
      learningRate: Double = 1e-3
  )(implicit numeric: Fractional[T]): Model[T] = {
    var fittedModel = this
    val learningRateArray = NDArray[T](Array(learningRate.asInstanceOf[T]))
    (0 until epochs).foreach { epoch =>
      val nextStepLoss = MeanSquaredError(fittedModel.outputLayer)
      val inputsWithLabels =
        inputs + (nextStepLoss.labelsInput -> labels)
      val execution =
        nextStepLoss.getComputationGraph.computeAll(inputsWithLabels)
      val gradients =
        nextStepLoss.getComputationGraph.backpropagateAll(execution)
      val modelParameterGradients = gradients
        .filter(_._1 match {
          case ModelParameter(_, _) => true
          case _                    => false
        })
        .asInstanceOf[Map[ModelParameter[T], NDArray[T]]]
      val updatedParameters = modelParameterGradients.map {
        parameterAndGradient =>
          val parameter = parameterAndGradient._1
          val gradient = parameterAndGradient._2
          val newParameter = ModelParameter(
            parameter.name,
            parameter.value - gradient * learningRateArray
          )
          parameter -> newParameter
      }
      println(
        s"Epoch $epoch: loss=${execution.outputs(nextStepLoss.getComputationGraph).flatten().head}"
      )
      fittedModel = fittedModel.withUpdatedParameters(updatedParameters)
    }
    fittedModel
  }
}
