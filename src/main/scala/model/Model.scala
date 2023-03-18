package model

import autodifferentiation.{Input, ModelParameter}
import layers.{Layer, MeanSquaredError}
import ndarray.NDArray

import java.util
import scala.reflect.{ClassTag, classTag}

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
  def apply(inputs: Map[String, NDArray[T]]): NDArray[T] = outputLayer(
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
      parameters: util.IdentityHashMap[ModelParameter[T], ModelParameter[T]]
  ): Model[T] = Model(outputLayer.withUpdatedParameters(parameters))

  //TODO users shouldn't have to know about Input (in autodiff)
  /** Returns a model trained to predict the labels on the inputs.
    *
    * @param inputs
    *   The inputs to all input layers in the model. A Map of Input objects to
    *   the NDArrays that should fill them during training. The shape of each
    *   array depends on the learning task and model definition. In general, the
    *   first dimension is the batch dimension.
    * @param labels
    *   The ground truth labels the model should learn to predict on the input
    *   data. The labels should be in the same order as the inputs and should
    *   have the same number of examples. That is, the batch dimension should
    *   match.
    * @param epochs
    *   The number of complete passes to make over the data during gradient
    *   descent. After each pass, the algorithm generates new model parameters
    *   based on the gradient of the loss on the inputs.
    * @param learningRate
    *   Determines the magnitude of the weight update during gradient descent.
    *   The algorithm scales the gradient by this factor before updating the
    *   model's parameters. Lower values lead to slower convergence, but higher
    *   values can lead to instability or divergence. Typical values are on the
    *   order of 1e-3.
    */
  def fit(
      inputs: Map[String, NDArray[T]],
      labels: NDArray[T],
      epochs: Int,
      learningRate: Double = 1e-3
  )(implicit numeric: Fractional[T]): Model[T] = {
    var fittedModel = this
    val learningRateAsT = classTag[T] match {
      case _ if classTag[T] == classTag[Float] =>
        learningRate.toFloat.asInstanceOf[T]
      case _ if classTag[T] == classTag[Double] => learningRate.asInstanceOf[T]
    }
    (0 until epochs).foreach { epoch =>
      val nextStepLoss = MeanSquaredError(fittedModel.outputLayer)
      val inputsWithLabels =
        inputs + (nextStepLoss.labelsInput.name -> labels)
      //TODO because of our use of IdentityHashMap, we can only call getComputationGraph once--add issue
      val nextStepLossGraph = nextStepLoss.getComputationGraph
      val execution =
        nextStepLossGraph.computeAllComponentFunctions(inputsWithLabels)
      val gradients = {
        nextStepLossGraph.backpropagateAllComponentFunctions(execution)
      }
      val modelParameterGradients = new util.IdentityHashMap[ModelParameter[T], NDArray[T]]
      gradients.forEach((differentiableFunction, gradient) => differentiableFunction match {
        case ModelParameter(_, _) => modelParameterGradients.put(differentiableFunction.asInstanceOf[ModelParameter[T]], gradient)
        case _ => ;
      })
      val updatedParameters = new util.IdentityHashMap[ModelParameter[T], ModelParameter[T]]
      modelParameterGradients.forEach { (parameter, gradient) =>
        val newParameter = ModelParameter(
          parameter.name,
          parameter.value - (gradient * learningRateAsT)
        )
        updatedParameters.put(parameter, newParameter)
      }
      println(
        s"Epoch $epoch: loss=${execution.outputs.get(nextStepLossGraph).flatten().head}"
      )
      fittedModel = fittedModel.withUpdatedParameters(updatedParameters)
    }
    fittedModel
  }

  /** Returns the model's loss on the test set.
    *
    * @param inputs
    *   The inputs to all input layers in the model. A Map of Input objects to
    *   the NDArrays that should fill them during training. The shape of each
    *   array depends on the learning task and model definition. In general, the
    *   first dimension is the batch dimension.
    * @param labels
    *   The ground truth labels the model should learn to predict on the input
    *   data. The labels should be in the same order as the inputs and should
    *   have the same number of examples. That is, the batch dimension should
    *   match.
    */
  def evaluate(
      inputs: Map[Input[T], NDArray[T]],
      labels: NDArray[T]
  )(implicit numeric: Fractional[T]): NDArray[T] = ???

  /** Returns the model's predictions on the inputs.
    *
    * @param inputs
    *   The inputs to all input layers in the model. A Map of Input objects to
    *   the NDArrays that should fill them during training. The shape of each
    *   array depends on the learning task and model definition. In general, the
    *   first dimension is the batch dimension.
    * @return
    *   The model's predictions. These predictions will have the same shape as
    *   the model's output layer.
    */
  def predict(inputs: Map[Input[T], NDArray[T]]): NDArray[T] = ???
}
