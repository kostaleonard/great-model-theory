package layers

import autodifferentiation.{Input, ModelParameter}
import exceptions.ShapeException
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DenseSpec extends AnyFlatSpec with Matchers {

  "A Dense layer" should "have randomly initialized weights" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, 2)
    val head = dense.weights.value(Array(0, 0))
    assert(!dense.weights.value.flatten().forall(_ == head))
  }

  it should "have zero-initialized biases" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, 2)
    assert(dense.biases.value.flatten().forall(_ == 0))
  }

  it should "have weights and biases of the correct shapes" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, 2)
    assert(dense.weights.value.shape sameElements Array(4, 2))
    assert(dense.biases.value.shape sameElements Array(2))
  }

  it should "compute the dot product of the inputs and weights (rank 2)" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val units = 2
    val dense = Dense
      .withInitialization(
        inputLayer,
        units,
        weightsInitialization = Some(
          ModelParameter(
            "weights",
            NDArray.arange[Float](Array(numFeatures, units))
          )
        ),
        biasesInitialization = None
      )
    val sampleBatchSize = 2
    val inputs =
      Map(input -> NDArray.arange[Float](Array(sampleBatchSize, numFeatures)))
    val outputs = dense(inputs)
    assert(outputs.shape sameElements Array(sampleBatchSize, units))
    assert(
      outputs arrayApproximatelyEquals NDArray[Float](
        Array(28f, 34f, 76f, 98f)
      ).reshape(Array(sampleBatchSize, units))
    )
  }

  it should "compute the dot product of the inputs and weights (rank 3)" in {
    val numFeaturesRows = 4
    val numFeaturesCols = 3
    val input = Input[Float](
      "X",
      Array(None, Some(numFeaturesRows), Some(numFeaturesCols))
    )
    val inputLayer = InputLayer(input)
    val units = 2
    val dense = Dense
      .withInitialization(
        inputLayer,
        units,
        weightsInitialization = Some(
          ModelParameter(
            "weights",
            NDArray.arange[Float](Array(numFeaturesCols, units))
          )
        ),
        biasesInitialization = None
      )
    val sampleBatchSize = 2
    val inputs = Map(
      input -> NDArray.arange[Float](
        Array(sampleBatchSize, numFeaturesRows, numFeaturesCols)
      )
    )
    val outputs = dense(inputs)
    assert(
      outputs.shape sameElements Array(
        sampleBatchSize,
        numFeaturesRows,
        units
      )
    )
    assert(
      outputs arrayApproximatelyEquals NDArray[Float](
        Array(10f, 13f, 28f, 40f, 46f, 67f, 64f, 94f, 82f, 121f, 100f, 148f,
          118f, 175f, 136f, 202f)
      ).reshape(Array(sampleBatchSize, numFeaturesRows, units))
    )
  }

  it should "compute the dot product of the inputs and weights and add biases" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val units = 2
    val dense = Dense
      .withInitialization(
        inputLayer,
        units,
        weightsInitialization = Some(
          ModelParameter(
            "weights",
            NDArray.arange[Float](Array(numFeatures, units))
          )
        ),
        biasesInitialization =
          Some(ModelParameter("biases", NDArray.ones[Float](Array(units))))
      )
    val sampleBatchSize = 2
    val inputs =
      Map(input -> NDArray.arange[Float](Array(sampleBatchSize, numFeatures)))
    val outputs = dense(inputs)
    assert(outputs.shape sameElements Array(sampleBatchSize, units))
    assert(
      outputs arrayApproximatelyEquals NDArray[Float](
        Array(29f, 35f, 77f, 99f)
      ).reshape(Array(sampleBatchSize, units))
    )
  }

  it should "fail to transform incorrectly shaped inputs" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val dense = Dense.withRandomWeights(inputLayer, 2)
    val sampleBatchSize = 2
    val inputs = Map(
      input -> NDArray.arange[Float](Array(sampleBatchSize, numFeatures + 1))
    )
    assertThrows[ShapeException](dense(inputs))
  }
}
