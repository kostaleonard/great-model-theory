package layers

import autodifferentiation.Input
import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class SigmoidSpec extends AnyFlatSpec with Matchers {

  "A Sigmoid layer" should "compute the sigmoid function" in {
    val numFeatures = 3
    val input = Input[Double]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val sigmoid = Sigmoid(inputLayer)
    val batchSize = 2
    val inputs = Map(
      "X" -> NDArray[Double](Array(0, 1, 2, -3, 4, 0)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val outputs = sigmoid(inputs)
    val expected = NDArray[Double](
      Array(0.5, 0.73105852, 0.88079701, 0.04742591, 0.98201377, 0.5)
    ).reshape(Array(batchSize, numFeatures))
    assert(outputs arrayApproximatelyEquals expected)
  }

  it should "compute values in (0, 1) (Double)" in {
    val numFeatures = 3
    val input = Input[Double]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val sigmoid = Sigmoid(inputLayer)
    val batchSize = 2
    val inputs = Map(
      "X" -> NDArray[Double](Array(-1e6, -1e3, -10, 10, 1e3, 1e6)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val outputs = sigmoid(inputs)
    assert(outputs.flatten().forall(x => (0 <= x) && (x <= 1)))
  }

  it should "compute values in (0, 1) (Float)" in {
    val numFeatures = 3
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val sigmoid = Sigmoid(inputLayer)
    val batchSize = 2
    val inputs = Map(
      "X" -> NDArray[Float](Array(-1e6f, -1e3f, -10f, 10f, 1e3f, 1e6f)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val outputs = sigmoid(inputs)
    assert(outputs.flatten().forall(x => (0 <= x) && (x <= 1)))
  }
}
