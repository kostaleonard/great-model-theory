package layers

import autodifferentiation.Input
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InputLayerSpec extends AnyFlatSpec with Matchers {
  private val BATCH_SIZE_PLACEHOLDER = 1

  "An InputLayer" should "create a computation graph with a single input node" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(BATCH_SIZE_PLACEHOLDER, numFeatures))
    val inputLayer = InputLayer(input)
    assert(inputLayer.getComputationGraph == input)
  }
}
