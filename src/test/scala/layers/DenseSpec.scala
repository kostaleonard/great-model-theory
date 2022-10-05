package layers

import ndarray.NDArray
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class DenseSpec extends AnyFlatSpec with Matchers {
  private val BATCH_SIZE_PLACEHOLDER = 1

  "A Dense layer" should "have randomly initialized weights" in {
    val dense = new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, 4), 2)
    val head = dense.weights(List(0, 0))
    assert(!dense.weights.flatten().forall(_ == head))
  }

  it should "have zero-initialized biases" in {
    val dense = new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, 4), 2)
    assert(dense.biases.flatten().forall(_ == 0))
  }

  it should "have weights and biases of the correct shapes" in {
    val dense = new Dense[Float](Array(BATCH_SIZE_PLACEHOLDER, 4), 2)
    assert(dense.weights.shape sameElements Array(4, 2))
    assert(dense.biases.shape sameElements Array(2))
  }

  it should "compute the dot product of the inputs and weights (rank 2)" in {
    val numFeatures = 4
    val units = 2
    val dense = new Dense[Float](
      Array(BATCH_SIZE_PLACEHOLDER, numFeatures),
      units,
      weightsInitialization =
        Some(NDArray.ones[Float](List(numFeatures, units)))
    )
    val sampleBatchSize = 2
    val inputs = NDArray.arange[Float](List(sampleBatchSize, numFeatures))
    val outputs = dense(inputs)
    assert(outputs.shape sameElements Array(sampleBatchSize, units))
    assert(
      outputs arrayApproximatelyEquals NDArray[Float](List(6f, 6f, 22f, 22f))
    )
  }
}
