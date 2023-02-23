package model

import autodifferentiation.Input
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
}
