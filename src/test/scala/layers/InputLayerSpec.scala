package layers

import autodifferentiation.Input
import org.scalatest.flatspec.AnyFlatSpec
import org.scalatest.matchers.should.Matchers

class InputLayerSpec extends AnyFlatSpec with Matchers {

  "An InputLayer" should "create a computation graph with a single input node" in {
    val numFeatures = 4
    val input = Input[Float]("X", Array(None, Some(numFeatures)))
    val inputLayer = InputLayer(input)
    assert(inputLayer.getComputationGraph == input)
  }
}
