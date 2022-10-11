package model

import layers.Dense
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class ModelSpec extends AnyFlatSpec with Matchers {
  private val BATCH_SIZE_PLACEHOLDER = 1

  "A Model" should "apply a single layer to the input" in {
    val numFeatures = 4
    val outputSize = 2
    val dense =
      new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, numFeatures), outputSize)
    val model = new Model[Float](List(dense))
    val sampleBatchSize = 2
    val inputs = NDArray.ones[Float](List(sampleBatchSize, numFeatures))
    val outputs = model(inputs)
    assert(outputs.isSuccess)
    assert(outputs.get.shape sameElements Array(sampleBatchSize, outputSize))
    assert(!outputs.get.flatten().forall(_ == 1))
  }

  it should "apply multiple layers to the input" in {
    val numFeatures = 4
    val hiddenSize = 3
    val outputSize = 2
    val dense1 =
      new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, numFeatures), hiddenSize)
    val dense2 =
      new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, hiddenSize), outputSize)
    val model = new Model[Float](List(dense1, dense2))
    val sampleBatchSize = 2
    val inputs = NDArray.ones[Float](List(sampleBatchSize, numFeatures))
    val outputs = model(inputs)
    assert(outputs.isSuccess)
    assert(outputs.get.shape sameElements Array(sampleBatchSize, outputSize))
    assert(!outputs.get.flatten().forall(_ == 1))
    // Check that layer 2 was also applied.
    val dense1Outputs = dense1(inputs)
    assert(dense1Outputs.get arrayNotApproximatelyEquals outputs.get)
  }

  it should "fail to apply layers on incorrectly shaped input" in {
    val numFeatures = 4
    val outputSize = 2
    val dense =
      new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, numFeatures), outputSize)
    val model = new Model[Float](List(dense))
    val sampleBatchSize = 2
    val inputs = NDArray.ones[Float](List(sampleBatchSize, numFeatures + 1))
    val outputs = model(inputs)
    assert(outputs.isFailure)
  }

  it should "fail to apply layers when layer shapes do not match" in {
    val numFeatures = 4
    val hiddenSize = 3
    val outputSize = 2
    val dense1 =
      new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, numFeatures), hiddenSize)
    val dense2 = new Dense[Float](
      Array(BATCH_SIZE_PLACEHOLDER, hiddenSize + 1),
      outputSize
    )
    val model = new Model[Float](List(dense1, dense2))
    val sampleBatchSize = 2
    val inputs = NDArray.ones[Float](List(sampleBatchSize, numFeatures))
    val outputs = model(inputs)
    assert(outputs.isFailure)
  }
}
