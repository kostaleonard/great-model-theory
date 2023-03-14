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
      input -> NDArray[Double](Array(0, 1, 2, -3, 4, 0)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val outputs = sigmoid(inputs)
    val expected = NDArray[Double](
      Array(0.5, 0.73105852, 0.88079701, 0.04742591, 0.98201377, 0.5)
    ).reshape(Array(batchSize, numFeatures))
    assert(outputs arrayApproximatelyEquals expected)
  }

  it should "compute values in (0, 1)" in {
    val numFeatures = 3
    val input = Input[Double]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val sigmoid = Sigmoid(inputLayer)
    val batchSize = 2
    val inputs = Map(
      input -> NDArray[Double](Array(-1e6, -1e3, -10, 10, 1e3, 1e6)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val outputs = sigmoid(inputs)
    assert(outputs.flatten().forall(x => (0 <= x) && (x <= 1)))
  }

  it should "compute its gradient" in {
    // Computed these gradients using sigmoid's symbolic derivative:
    // https://math.stackexchange.com/questions/78575/derivative-of-sigmoid-function-sigma-x-frac11e-x
    val numFeatures = 3
    val input = Input[Double]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    val sigmoid = Sigmoid(inputLayer)
    val batchSize = 2
    val inputs = Map(
      input -> NDArray[Double](Array(0, 1, 2, -3, 4, 0)).reshape(
        Array(batchSize, numFeatures)
      )
    )
    val gradient = sigmoid.getComputationGraph.gradient(input)
    val outputs = gradient.compute(inputs)
    val expected = NDArray[Double](
      Array(0.25, 0.19661193324148188, 0.1049935854035065, 0.04517665973091213,
        0.017662706213291114, 0.25)
    ).reshape(Array(batchSize, numFeatures))
    assert(outputs arrayApproximatelyEquals expected)
  }
}
