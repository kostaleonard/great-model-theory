package layers

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
}
