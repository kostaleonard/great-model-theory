package model

import autodifferentiation.{Constant, Input, Mean, Square, Subtract}
import exceptions.ShapeException
import layers.{Dense, InputLayer}
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ModelSpec extends AnyFlatSpec with Matchers {

  "A Model" should "apply a single layer to the input" in {
    val numFeatures = 4
    val outputSize = 2
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, outputSize)
    val model = Model(dense)
    val sampleBatchSize = 2
    val inputs =
      Map(input -> NDArray.ones[Float](Array(sampleBatchSize, numFeatures)))
    val outputs = model(inputs)
    assert(outputs.shape sameElements Array(sampleBatchSize, outputSize))
    assert(!outputs.flatten().forall(_ == 1))
  }

  it should "apply multiple layers to the input" in {
    val numFeatures = 4
    val hiddenSize = 3
    val outputSize = 2
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense1 = Dense.withRandomWeights(inputLayer, hiddenSize)
    val dense2 = Dense.withRandomWeights(dense1, outputSize)
    val model = Model(dense2)
    val sampleBatchSize = 2
    val inputs =
      Map(input -> NDArray.ones[Float](Array(sampleBatchSize, numFeatures)))
    val outputs = model(inputs)
    assert(outputs.shape sameElements Array(sampleBatchSize, outputSize))
    assert(!outputs.flatten().forall(_ == 1))
    // Check that the model also applied layer 2.
    val dense1Outputs = dense1(inputs)
    assert(dense1Outputs arrayNotApproximatelyEquals outputs)
  }

  it should "fail to apply layers on incorrectly shaped input" in {
    val numFeatures = 4
    val outputSize = 2
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, outputSize)
    val model = Model(dense)
    val sampleBatchSize = 2
    val inputs =
      Map(input -> NDArray.ones[Float](Array(sampleBatchSize, numFeatures + 1)))
    assertThrows[ShapeException](model(inputs))
  }

  it should "perform gradient descent (Float)" in {
    val numExamples = 4
    val numFeatures = 3
    val numOutputs = 2
    // The function we are trying to model is f(x) = (x0 ^ 2 - x1, 2 * x2)
    val xTrain =
      NDArray[Float](Array(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(numExamples, numFeatures)
      )
    val yTrain = NDArray[Float](Array(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(numExamples, numOutputs)
    )
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, numOutputs)
    val model = Model(dense)
    val inputs = Map(input -> xTrain)
    val lossFunctionBefore = Mean(
      Square(Subtract(model.outputLayer.getComputationGraph, Constant(yTrain)))
    )
    val lossBefore = lossFunctionBefore.compute(inputs).flatten().head
    val fittedModel = model.fit(inputs, yTrain, 10, learningRate = 1e-2)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
  }

  it should "perform gradient descent (Double)" in {
    val numExamples = 4
    val numFeatures = 3
    val numOutputs = 2
    // The function we are trying to model is f(x) = (x0 ^ 2 - x1, 2 * x2)
    val xTrain =
      NDArray[Double](Array(1, 3, 2, 4, 9, 1, 2, 2, 2, 1, 0, -1)).reshape(
        Array(numExamples, numFeatures)
      )
    val yTrain = NDArray[Double](Array(-2, 4, 7, 2, 2, 4, 1, -2)).reshape(
      Array(numExamples, numOutputs)
    )
    val input = Input[Double]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, numOutputs)
    val model = Model(dense)
    val inputs = Map(input -> xTrain)
    val lossFunctionBefore = Mean(
      Square(Subtract(model.outputLayer.getComputationGraph, Constant(yTrain)))
    )
    val lossBefore = lossFunctionBefore.compute(inputs).flatten().head
    val fittedModel = model.fit(inputs, yTrain, 10, learningRate = 1e-2)
    val lossFunctionAfter = Mean(
      Square(
        Subtract(fittedModel.outputLayer.getComputationGraph, Constant(yTrain))
      )
    )
    val lossAfter = lossFunctionAfter.compute(inputs).flatten().head
    assert(lossAfter < lossBefore)
  }
}
